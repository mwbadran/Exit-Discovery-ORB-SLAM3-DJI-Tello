import os
import sys
import socket
import threading
import subprocess
from json import load
from time import sleep
from datetime import datetime

from djitellopy import Tello

from plot import *
from ExitFinding import *
from PointCloudCleaning import *
from utils import *

# real values are in config.json, these are just init
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
        return cfg["wsl_point_csv"]
    root_dir = slam_root_from_exe(cfg["slam_exe"])
    return os.path.join(root_dir, "log", "pointData.csv")


# -------------------- UDP forwarder (Windows -> WSL) --------------------

class UDPForwarder(threading.Thread):
    """
    Forwards UDP H264 from 0.0.0.0:forward_port on Windows to (WSL_IP, forward_port).
    Stop with .stop(); it closes sockets and exits the thread loop.
    """
    def __init__(self, wsl_ip: str, port: int):
        super().__init__(daemon=True)
        self.wsl_ip = wsl_ip
        self.port = port
        self._stop_evt = threading.Event()
        self.insock = None
        self.outsock = None

    def run(self):
        try:
            self.insock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.insock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.insock.bind(("0.0.0.0", self.port))

            self.outsock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            out_addr = (self.wsl_ip, self.port)
            print(f"[FWD] listening :{self.port} -> {out_addr}")

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
            try:
                if self.insock:
                    self.insock.close()
            except Exception:
                pass
            try:
                if self.outsock:
                    self.outsock.close()
            except Exception:
                pass
            print("[FWD] stopped.")

    def stop(self):
        self._stop_evt.set()


def detect_wsl_ip() -> str:
    """
    Query WSL for its current IPv4 (first token of `hostname -I`).
    Works from Windows by invoking `wsl`.
    """
    try:
        res = subprocess.run(
            ["wsl", "hostname", "-I"],
            capture_output=True, text=True, check=True
        )
        # hostname -I may return multiple addresses; take the first IPv4-like token
        tokens = res.stdout.strip().split()
        for t in tokens:
            if t.count(".") == 3:  # naive IPv4 check
                return t
    except Exception as e:
        print("[WSL] unable to detect IP:", e)
    # fallback to a common default; user can still edit config if needed
    return "172.30.0.1"


# -------------------- WSL SLAM launcher --------------------

def build_wsl_slam_command(cfg) -> str:
    repo = cfg["wsl_repo_dir"].rstrip("/")
    vocab_rel    = cfg.get("wsl_vocab_rel", "Vocabulary/ORBvoc.txt")
    settings_rel = cfg.get("wsl_settings_rel", "Examples/Monocular/TUM1.yaml")
    source       = cfg.get("source","tello").lower()
    port         = int(cfg.get("forward_port", 11111))

    if source == "tello":
        slam_bin = cfg.get("wsl_slam_binary", "./Examples/Monocular/mono_tello")
        input_arg = f"\"udp://@:{port}\""
    elif source == "webcam":
        slam_bin = cfg.get("wsl_slam_binary_video", "./Examples/Monocular/mono_input")
        input_arg = str(cfg.get("wsl_webcam_index", "0"))  # or /dev/video0
    else:  # video
        slam_bin = cfg.get("wsl_slam_binary_video", "./Examples/Monocular/mono_input")
        input_arg = f"\"{cfg.get('wsl_video_path','')}\""

    return f"cd {repo} && {slam_bin} {vocab_rel} {settings_rel} {input_arg}"



def launch_wsl_slam(cfg):
    """
    Launch ORB-SLAM3 inside WSL and return the Popen handle.
    We rely on `wsl` command preinstalled on Windows.
    """
    cmd = build_wsl_slam_command(cfg)
    print("[WSL] launching ORB-SLAM3:\n    wsl bash -lc \"{}\"".format(cmd))
    try:
        proc = subprocess.Popen(
            ["wsl", "bash", "-lc", cmd],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        return proc
    except Exception as e:
        print("[WSL] failed to launch ORB-SLAM3:", e)
        return None


def drain_proc_output(proc, tag="[WSL]"):
    """Non-blocking-ish drain: print a few lines to show progress."""
    if proc is None or proc.stdout is None:
        return
    try:
        for _ in range(8):  # just a small burst so we don't block
            line = proc.stdout.readline()
            if not line:
                break
            print(f"{tag} {line.rstrip()}")
    except Exception:
        pass


# -------------------- Tello helpers --------------------

def tello_prepare_stream(drone: Tello, delay_s: float):
    try:
        drone.streamoff()
    except Exception:
        pass
    drone.streamon()
    sleep(delay_s)


def append_telemetry(row_dict: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cols = ["ts", "step_i", "rotated_deg_total", "yaw", "height_cm", "battery",
            "cmd", "cmd_ok", "note"]
    header = ",".join(cols) + "\n"
    values = [str(row_dict.get(k, "")) for k in cols]
    line = ",".join(values) + "\n"
    need_header = not os.path.exists(path)
    with open(path, "a", encoding="utf-8") as f:
        if need_header:
            f.write(header)
        f.write(line)


# -------------------- main mission --------------------

def drone_scan_with_slam(cfg, input_arg):
    """
    Full pipeline:
      - start WSL ORB-SLAM3 (mono_tello) via `wsl`
      - auto-start UDP forwarder to WSL IP
      - connect & fly Tello 360° scan with nudges
      - stop stream -> let SLAM exit -> stop forwarder
      - return CSV path and the live drone handle for post-processing moves
    """
    vocab     = cfg["vocab"]
    settings  = cfg["settings"]
    slam_exe  = cfg["slam_exe"]
    source    = cfg["source"].lower()
    csv_path  = csv_path_from_config(cfg)

    # config params
    streamon_delay     = float(cfg["streamon_delay_sec"])
    slam_start_wait    = float(cfg["slam_start_wait_s"])
    rotation_step      = int(cfg["rotation_step_deg"])
    rotation_pause     = float(cfg["rotation_pause_s"])
    target_height      = int(cfg["height"])
    speed              = int(cfg["speed"])
    slam_exit_wait_s   = float(cfg.get("slam_exit_wait_s", 15.0))
    do_nudge           = bool(cfg.get("do_nudge", True))
    nudge_cm           = int(cfg.get("nudge_cm", 20))
    nudge_every_steps  = int(cfg.get("nudge_every_steps", 2))
    telemetry_csv      = cfg.get("telemetry_csv", os.path.join(slam_root_from_exe(slam_exe), "log", "telemetry.csv"))
    fwd_port           = int(cfg.get("forward_port", 11111))

    # 0) Start WSL ORB-SLAM3 and UDP forwarder (only in wsl_mode)
    slam_proc = None
    fwd = None
    wsl_ip = None
    if cfg.get("wsl_mode"):
        wsl_ip = detect_wsl_ip()
        fwd = UDPForwarder(wsl_ip, fwd_port)
        fwd.start()
        slam_proc = launch_wsl_slam(cfg)
        # give mono_tello a head start and show some logs
        drain_proc_output(slam_proc)
    else:
        # legacy Windows SLAM path
        print("[LEGACY] launching Windows slam.exe")
        slam_proc = runOrbSlam3(slam_exe, vocab, settings, input_arg)

    if source != "tello":
        if slam_proc is not None:
            try:
                slam_proc.wait()
            except Exception:
                pass
        if fwd: fwd.stop()
        return csv_path, None

    # 1) Tello flow
    drone = Tello()
    drone.connect()
    drone.set_speed(speed)

    try:
        battery = drone.get_battery()
        #voltage = drone.get_voltage()
        print(f"[INFO] Battery: {battery}% ")
    except Exception as e:
        print(f"[WARN] Could not read battery/voltage: {e}")

    tello_prepare_stream(drone, streamon_delay)

    print(f"[INFO] Waiting {slam_start_wait:.1f}s for ORB-SLAM3 to start...")
    sleep(slam_start_wait)
    drain_proc_output(slam_proc)

    # 360° scan with nudges
    drone.takeoff()
    try:
        try:
            current_h = drone.get_height()
        except Exception:
            current_h = 0
        delta = int(target_height - current_h)
        if   delta > 0: drone.move_up(delta)
        elif delta < 0: drone.move_down(-delta)
        sleep(0.5)

        rotated = 0
        step_i = 0
        while rotated < MAX_ANGLE:
            step = min(rotation_step, MAX_ANGLE - rotated)
            ok = True
            note = ""
            try:
                drone.rotate_clockwise(step)
            except Exception as e:
                ok = False
                note = f"rotate_err:{e}"

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

            rotated += step
            step_i += 1

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

        # 2) Close SLAM gracefully: stopping stream lets mono_tello exit.
        close_slam(None, input_arg, drone=drone, wait_s=slam_exit_wait_s)

    except KeyboardInterrupt:
        print("[WARN] KeyboardInterrupt during scan.")
        close_slam(None, input_arg, drone=drone, wait_s=8.0)
        raise
    except Exception as e:
        print("Scan error:", e)
        close_slam(None, input_arg, drone=drone, wait_s=8.0)

    # 3) stop forwarder + wait for WSL proc to exit
    if fwd:
        fwd.stop()
        fwd.join(timeout=3.0)
    if slam_proc:
        try:
            slam_proc.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            try:
                slam_proc.terminate()
            except Exception:
                pass

    return csv_path, drone


# -------------------- entrypoint --------------------

if __name__ == '__main__':
    cfg = loadConfig()


    #get angles from config file:
    EXTRA_ANGLE = cfg["extra_angle"]
    MAX_ANGLE_WOUT_EXTRA = cfg["max_angle_wout_extra"]
    MAX_ANGLE = MAX_ANGLE_WOUT_EXTRA + EXTRA_ANGLE


    # decide input arg (mainly relevant for legacy Windows)
    source = cfg["source"].lower()
    if source == "tello":
        tello_mode = cfg.get("tello_input_mode", "udp").lower()
        if tello_mode == "udp":
            input_arg = "TELLO"
        elif tello_mode == "webcam":
            input_arg = str(cfg.get("tello_webcam_index", "0"))
        else:
            print(f"Unknown tello_input_mode: {tello_mode}")
            sys.exit(1)
    elif source == "webcam":
        input_arg = str(cfg.get("webcam_index", "0"))
    else:
        input_arg = cfg["video_path"]

    # 1) run the scan with SLAM (WSL or legacy)
    try:
        csv_path, drone = drone_scan_with_slam(cfg, input_arg)
    except KeyboardInterrupt:
        try:
            if 'drone' in locals() and drone is not None:
                safe_land(drone); drone.end()
        except Exception:
            pass
        sys.exit(1)


    # 2) confirm CSV exists
    if not os.path.exists(csv_path):
        root_dir = slam_root_from_exe(cfg["slam_exe"])
        exe_rel  = os.path.join("x64", "Release", "slam.exe")
        print("\nERROR: pointData.csv not found.")
        print(f"Expected here: {csv_path}")
        print("If running WSL, confirm mono_tello wrote to ~/dev/ORB_SLAM3/log/pointData.csv")
        print("Also check that the UDP forwarder started and WSL IP was detected.")
        if source != "tello":
            print(f"\n(Windows legacy) You can also run exe manually:")
            print(f"     cd {root_dir}")
            print(f"     .\\{exe_rel} \"mono_video\" \"{cfg['vocab']}\" \"{cfg['settings']}\" \"{input_arg}\"\n")
        if drone is not None:
            safe_land(drone); drone.end()
        raise SystemExit(1)

    # 3) process map
    x, y, z = readCSV(csv_path)
    if len(x) == 0:
        print("---- SLAM produced 0 points. Nothing to process. ----")
        if drone is not None:
            safe_land(drone); drone.end()
        raise SystemExit(0)

    pcd = makeCloud(x, y, z)

    inlierPCD, outlierPCD = removeStatisticalOutlier(
        pcd,
        voxel_size=float(cfg["voxel_size"]),
        nb_neighbors=int(cfg["nb_neighbors"]),
        std_ratio=float(cfg["std_ratio"])
    )
    inX, inY, inZ = pcdToArrays(inlierPCD)

    # room rectangle from top-down (X,Z)
    scale = float(cfg.get("avg_rect_scale", 1.2))  # 1.0–1.4
    box = getAverageRectangle(inX, inZ, scale=scale)
    #box = getRobustRoomBox(inX, inZ, q=0.05, margin=0.10)


    # plots
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
    clusters = hierarchicalClustering(xOut, zOut, float(cfg["thresh"]))
    centers  = getClustersCenters(clusters)

    # 4) navigate to exit (keep airborne)
    if len(centers) == 0:
        print("No exits detected. Done.")
        plot2DWithBoxAndCenters(inX, inZ, box, centers=[],
                                save_path=os.path.join(plots_dir, f"centers_none_{tag}.png"),
                                title="No exits detected")
        if drone is not None:
            safe_land(drone); drone.end()
        sys.exit(0)

    plot2DWithBoxAndCenters(inX, inZ, box, centers,
                            save_path=os.path.join(plots_dir, f"centers_{tag}.png"),
                            title="Detected Exit Cluster Centers (X,Z)")

    if drone is not None:
        try:
            ensure_airborne(drone)
            moveToExit(drone, centers)
        finally:
            safe_land(drone)
            drone.end()
    else:
        plot2DWithClustersCenters(inX, inZ, centers,
                                  save_path=os.path.join(plots_dir, f"centers_only_{tag}.png"),
                                  title="Exits (cluster centers)")
        print("Exits (cluster centers):", centers)
