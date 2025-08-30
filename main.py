import os, sys, socket, threading, subprocess
from json import load
from time import sleep
from datetime import datetime

from djitellopy import Tello

from plot import *
from ExitFinding import *
from PointCloudCleaning import *
from utils import *

# ---------- defaults (overridden by config.json) ----------
EXTRA_ANGLE = 40
MAX_ANGLE_WOUT_EXTRA = 360
MAX_ANGLE = 400


# -------------------- config helpers --------------------
def loadConfig():
    with open('config.json', 'r', encoding='utf-8') as f:
        return load(f)

def slam_root_from_exe(exe_path: str) -> str:
    # exe is ...\x64\Release\slam.exe  -> root is THREE dirs up: ...\ (ORB_SLAM3_Windows)
    return os.path.dirname(os.path.dirname(os.path.dirname(exe_path)))

def csv_path_from_config(cfg):
    if cfg.get("wsl_mode"):
        return cfg.get("wsl_point_csv", os.path.expanduser("~/dev/ORB_SLAM3/log/pointData.csv"))
    root_dir = slam_root_from_exe(cfg["slam_exe"])
    return os.path.join(root_dir, "log", "pointData.csv")


# -------------------- Windows → WSL UDP forwarder --------------------
class UDPForwarder(threading.Thread):
    """
    Binds 0.0.0.0:in_port on Windows and forwards to (WSL_IP, out_port).
    Use in_port=11111, out_port=11112 for conflict-free mirrored-mode.
    """
    def __init__(self, wsl_ip: str, in_port: int, out_port: int = None):
        super().__init__(daemon=True)
        self.wsl_ip = wsl_ip
        self.in_port = in_port
        self.out_port = out_port if out_port is not None else in_port
        self._stop_evt = threading.Event()
        self.insock = None
        self.outsock = None

    def run(self):
        try:
            self.insock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.insock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.insock.bind(("0.0.0.0", self.in_port))

            self.outsock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            out_addr = (self.wsl_ip, self.out_port)
            print(f"[FWD] listening :{self.in_port} -> {out_addr}")

            n = 0
            self.insock.settimeout(0.5)
            while not self._stop_evt.is_set():
                try:
                    data, src = self.insock.recvfrom(65536)
                except socket.timeout:
                    continue
                if not data:
                    continue
                n += 1
                if n % 50 == 0:
                    print(f"[FWD] fwd {n} pkts from {src}")
                self.outsock.sendto(data, out_addr)

        except Exception as e:
            print("[FWD] error:", e)

        finally:
            for s in (self.insock, self.outsock):
                try:
                    if s: s.close()
                except Exception:
                    pass
            print("[FWD] stopped.")

    def stop(self):
        self._stop_evt.set()


def detect_wsl_ip() -> str:
    """First IPv4 from `wsl hostname -I`."""
    try:
        res = subprocess.run(["wsl", "hostname", "-I"], capture_output=True, text=True, check=True)
        for t in res.stdout.strip().split():
            if t.count(".") == 3:
                return t
    except Exception as e:
        print("[WSL] unable to detect IP:", e)
    return "172.30.0.1"


# -------------------- WSL SLAM helpers --------------------
def build_wsl_slam_command(cfg) -> str:
    repo         = cfg["wsl_repo_dir"].rstrip("/")
    vocab_rel    = cfg.get("wsl_vocab_rel", "Vocabulary/ORBvoc.txt")
    settings_rel = cfg.get("wsl_settings_rel", "Examples/Monocular/TUM1.yaml")
    source       = cfg.get("source", "tello").lower()
    port         = int(cfg.get("wsl_listen_port", 11112))  # <<< WSL listens on 11112
    udp_opts     = cfg.get("wsl_udp_query", "overrun_nonfatal=1&fifo_size=1000000")

    if source == "tello":
        slam_bin = cfg.get("wsl_slam_binary", "./Examples/Monocular/mono_tello")
        input_arg = f"\"udp://@:{port}?{udp_opts}\""
    elif source == "webcam":
        slam_bin = cfg.get("wsl_slam_binary_video", "./Examples/Monocular/mono_input")
        input_arg = str(cfg.get("wsl_webcam_index", "0"))
    else:  # video file
        slam_bin = cfg.get("wsl_slam_binary_video", "./Examples/Monocular/mono_input")
        input_arg = f"\"{cfg.get('wsl_video_path','')}\""

    return f"cd {repo} && {slam_bin} {vocab_rel} {settings_rel} {input_arg}"

def launch_wsl_slam(cfg):
    cmd = build_wsl_slam_command(cfg)
    print("[WSL] launching ORB-SLAM3:\n    wsl bash -lc \"{}\"".format(cmd))
    try:
        proc = subprocess.Popen(["wsl", "bash", "-lc", cmd],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        return proc
    except Exception as e:
        print("[WSL] failed to launch ORB-SLAM3:", e)
        return None

def drain_proc_output(proc, tag="[WSL]"):
    if proc is None or proc.stdout is None:
        return
    try:
        for _ in range(8):
            line = proc.stdout.readline()
            if not line: break
            print(f"{tag} {line.rstrip()}")
    except Exception:
        pass


# -------------------- Tello helpers --------------------
def tello_prepare_stream(drone: Tello, delay_s: float):
    """Windows is the owner; keep the stream target on Windows."""
    try:
        try: drone.streamoff()
        except Exception: pass
        drone.streamon()
        sleep(delay_s)
    except Exception as e:
        print(f"[WARN] Windows streamon failed: {e}")

def append_telemetry(row_dict: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cols = ["ts","step_i","rotated_deg_total","yaw","height_cm","battery","cmd","cmd_ok","note"]
    header = ",".join(cols) + "\n"
    values = [str(row_dict.get(k, "")) for k in cols]
    line = ",".join(values) + "\n"
    need_header = not os.path.exists(path)
    with open(path, "a", encoding="utf-8") as f:
        if need_header: f.write(header)
        f.write(line)


# -------------------- main mission --------------------
def drone_scan_with_slam(cfg, input_arg):
    """
    Option 2 flow:
      - Start Windows→WSL forwarder (11111 → 11112)
      - Launch ORB-SLAM3 in WSL listening on 11112
      - Use djitellopy on Windows to send streamon + fly (Windows stays last-sender)
      - SLAM consumes forwarded packets; no retargeting
    """
    vocab     = cfg["vocab"]
    settings  = cfg["settings"]
    slam_exe  = cfg["slam_exe"]
    source    = cfg["source"].lower()
    csv_path  = csv_path_from_config(cfg)

    # config
    streamon_delay     = float(cfg.get("streamon_delay_sec", 5.0))
    slam_start_wait    = float(cfg.get("slam_start_wait_s", 25.0))
    rotation_step      = int(cfg.get("rotation_step_deg", 10))
    rotation_pause     = float(cfg.get("rotation_pause_s", 2.0))
    target_height      = int(cfg.get("height", 120))
    speed              = int(cfg.get("speed", 10))
    slam_exit_wait_s   = float(cfg.get("slam_exit_wait_s", 15.0))
    do_nudge           = bool(cfg.get("do_nudge", True))
    nudge_cm           = int(cfg.get("nudge_cm", 20))
    nudge_every_steps  = int(cfg.get("nudge_every_steps", 2))
    telemetry_csv      = cfg.get("telemetry_csv", os.path.join(slam_root_from_exe(slam_exe), "log", "telemetry.csv"))

    # ports
    in_port_windows    = int(cfg.get("forward_port_in", 11111))  # Windows binds here (Tello → Windows)
    out_port_wsl       = int(cfg.get("wsl_listen_port", 11112))  # WSL listens here
    assert in_port_windows != out_port_wsl, "Use different ports (11111 -> 11112) to avoid conflicts."

    # start forwarder FIRST so early frames are captured
    fwd = None
    wsl_ip = detect_wsl_ip()
    fwd = UDPForwarder(wsl_ip, in_port=in_port_windows, out_port=out_port_wsl)
    fwd.start()

    # launch SLAM in WSL
    slam_proc = launch_wsl_slam(cfg)
    drain_proc_output(slam_proc)

    if source != "tello":
        if slam_proc is not None:
            try: slam_proc.wait()
            except Exception: pass
        if fwd: fwd.stop()
        return csv_path, None

    # 1) Tello control (Windows)
    drone = Tello()
    drone.connect()
    drone.set_speed(speed)

    try:
        print(f"[INFO] Battery: {drone.get_battery()}%")
    except Exception as e:
        print(f"[WARN] Could not read battery: {e}")

    # keep Windows as last-sender (no WSL streamon)
    tello_prepare_stream(drone, streamon_delay)

    print(f"[INFO] Waiting {slam_start_wait:.1f}s for ORB-SLAM3 to start...")
    sleep(slam_start_wait)
    drain_proc_output(slam_proc)

    # 360° scan with optional nudges
    drone.takeoff()
    try:
        try: current_h = drone.get_height()
        except Exception: current_h = 0
        delta = int(target_height - current_h)
        if   delta > 0: drone.move_up(delta)
        elif delta < 0: drone.move_down(-delta)
        sleep(0.5)

        rotated = 0
        step_i = 0
        while rotated < MAX_ANGLE:
            step = min(rotation_step, MAX_ANGLE - rotated)
            ok = True; note = ""
            try:
                drone.rotate_clockwise(step)
            except Exception as e:
                ok = False; note = f"rotate_err:{e}"

            try: yaw = drone.get_yaw()
            except Exception: yaw = ""
            try: h = drone.get_height()
            except Exception: h = ""
            try: bat = drone.get_battery()
            except Exception: bat = ""

            append_telemetry({
                "ts": datetime.now().isoformat(timespec="seconds"),
                "step_i": step_i,
                "rotated_deg_total": rotated + step,
                "yaw": yaw, "height_cm": h, "battery": bat,
                "cmd": f"cw {step}", "cmd_ok": "1" if ok else "0",
                "note": note
            }, telemetry_csv)

            rotated += step; step_i += 1

            if do_nudge and (step_i % max(1, nudge_every_steps) == 0):
                try:
                    drone.move_up(nudge_cm); drone.move_down(nudge_cm)
                    append_telemetry({
                        "ts": datetime.now().isoformat(timespec="seconds"),
                        "step_i": step_i,
                        "rotated_deg_total": rotated,
                        "yaw": yaw, "height_cm": h, "battery": bat,
                        "cmd": f"nudge {nudge_cm}", "cmd_ok": "1",
                        "note": "up/down nudge"
                    }, telemetry_csv)
                except Exception as e:
                    append_telemetry({
                        "ts": datetime.now().isoformat(timespec="seconds"),
                        "step_i": step_i,
                        "rotated_deg_total": rotated,
                        "yaw": yaw, "height_cm": h, "battery": bat,
                        "cmd": f"nudge {nudge_cm}", "cmd_ok": "0",
                        "note": f"nudge_err:{e}"
                    }, telemetry_csv)

            sleep(rotation_pause)
            drain_proc_output(slam_proc)

        # stop stream -> SLAM exits
        close_slam(None, input_arg, drone=drone, wait_s=slam_exit_wait_s)

    except KeyboardInterrupt:
        print("[WARN] KeyboardInterrupt during scan.")
        close_slam(None, input_arg, drone=drone, wait_s=8.0)
        raise
    except Exception as e:
        print("Scan error:", e)
        close_slam(None, input_arg, drone=drone, wait_s=8.0)

    # cleanup
    if fwd:
        fwd.stop()
        fwd.join(timeout=3.0)
    if slam_proc:
        try: slam_proc.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            try: slam_proc.terminate()
            except Exception: pass

    return csv_path, drone


# -------------------- entrypoint --------------------
if __name__ == '__main__':
    cfg = loadConfig()

    # scan angles
    EXTRA_ANGLE = int(cfg.get("extra_angle", EXTRA_ANGLE))
    MAX_ANGLE_WOUT_EXTRA = int(cfg.get("max_angle_wout_extra", MAX_ANGLE_WOUT_EXTRA))
    MAX_ANGLE = MAX_ANGLE_WOUT_EXTRA + EXTRA_ANGLE

    # input arg (legacy use)
    source = cfg["source"].lower()
    if source == "tello":
        tello_mode = cfg.get("tello_input_mode", "udp").lower()
        if tello_mode == "udp":       input_arg = "TELLO"
        elif tello_mode == "webcam":  input_arg = str(cfg.get("tello_webcam_index", "0"))
        else:
            print(f"Unknown tello_input_mode: {tello_mode}"); sys.exit(1)
    elif source == "webcam":
        input_arg = str(cfg.get("webcam_index", "0"))
    else:
        input_arg = cfg["video_path"]

    try:
        csv_path, drone = drone_scan_with_slam(cfg, input_arg)
    except KeyboardInterrupt:
        try:
            if 'drone' in locals() and drone is not None:
                safe_land(drone); drone.end()
        except Exception:
            pass
        sys.exit(1)

    if not os.path.exists(csv_path):
        root_dir = slam_root_from_exe(cfg["slam_exe"])
        exe_rel  = os.path.join("x64", "Release", "slam.exe")
        print("\nERROR: pointData.csv not found.")
        print(f"Expected here: {csv_path}")
        print("Confirm WSL mono_tello listened on 11112 and forwarder was running.")
        if source != "tello":
            print(f"\n(Windows legacy) You can also run exe manually:")
            print(f"     cd {root_dir}")
            print(f"     .\\{exe_rel} \"mono_video\" \"{cfg['vocab']}\" \"{cfg['settings']}\" \"{input_arg}\"\n")
        if 'drone' in locals() and drone is not None:
            safe_land(drone); drone.end()
        raise SystemExit(1)

    # post-process pointcloud
    x, y, z = readCSV(csv_path)
    if len(x) == 0:
        print("---- SLAM produced 0 points. Nothing to process. ----")
        if 'drone' in locals() and drone is not None:
            safe_land(drone); drone.end()
        raise SystemExit(0)

    pcd = makeCloud(x, y, z)
    inlierPCD, outlierPCD = removeStatisticalOutlier(
        pcd,
        voxel_size=float(cfg.get("voxel_size", 0.06)),
        nb_neighbors=int(cfg.get("nb_neighbors", 18)),
        std_ratio=float(cfg.get("std_ratio", 2.0))
    )
    inX, inY, inZ = pcdToArrays(inlierPCD)

    scale = float(cfg.get("avg_rect_scale", 1.2))
    box = getAverageRectangle(inX, inZ, scale=scale)

    plots_dir = cfg.get("plots_dir", "DebugPlots")
    os.makedirs(plots_dir, exist_ok=True)
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")

    plot2DWithBox(inX, inZ, box, save_path=os.path.join(plots_dir, f"scatter_box_{tag}.png"),
                  title="Inlier Points (X,Z) with Average Room Box")
    plot2DWithDensity(inX, inZ, save_path=os.path.join(plots_dir, f"density_hexbin_{tag}.png"),
                      title="Point Density (X,Z) Hexbin")
    plot3DPreview(inX, inY, inZ, save_path=os.path.join(plots_dir, f"cloud_preview_{tag}.png"),
                  title="3D Inlier Cloud (preview)")

    xOut, zOut = pointsOutOfBox(inX, inZ, box)
    clusters = hierarchicalClustering(xOut, zOut, float(cfg.get("thresh", 0.9)))
    centers  = getClustersCenters(clusters)

    if len(centers) == 0:
        print("No exits detected. Done.")
        plot2DWithBoxAndCenters(inX, inZ, box, centers=[],
                                save_path=os.path.join(plots_dir, f"centers_none_{tag}.png"),
                                title="No exits detected")
        if 'drone' in locals() and drone is not None:
            safe_land(drone); drone.end()
        raise SystemExit(0)

    plot2DWithBoxAndCenters(inX, inZ, box, centers,
                            save_path=os.path.join(plots_dir, f"centers_{tag}.png"),
                            title="Detected Exit Cluster Centers (X,Z)")

    if 'drone' in locals() and drone is not None:
        try:
            ensure_airborne(drone)
            moveToExit(drone, centers)
        finally:
            safe_land(drone); drone.end()
    else:
        plot2DWithClustersCenters(inX, inZ, centers,
                                  save_path=os.path.join(plots_dir, f"centers_only_{tag}.png"),
                                  title="Exits (cluster centers)")
        print("Exits (cluster centers):", centers)
