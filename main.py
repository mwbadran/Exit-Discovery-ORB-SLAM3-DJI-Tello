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
from utils import *  # uses close_slam(), readCSV(), makeCloud(), etc.  (see your utils.py)  # :contentReference[oaicite:0]{index=0}

MAX_ANGLE = 360


# -------------------- config helpers --------------------

def loadConfig():
    with open('config.json', 'r', encoding='utf-8') as f:
        return load(f)


def slam_root_from_exe(exe_path: str) -> str:
    # exe is ...\x64\Release\slam.exe  -> root is THREE dirs up: ...\ (ORB_SLAM3_Windows)
    return os.path.dirname(os.path.dirname(os.path.dirname(exe_path)))


def csv_path_from_config(cfg):
    # WSL: read CSV via UNC; Windows legacy: from repo\log
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
        self._stop = threading.Event()
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
            while not self._stop.is_set():
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
        self._stop.set()


def detect_wsl_ip() -> str:
    """
    Query WSL for its current IPv4 (first token of `hostname -I`) using the `wsl` bridge.
    """
    try:
        res = subprocess.run(
            ["wsl", "hostname", "-I"],
            capture_output=True, text=True, check=True
        )
        tokens = res.stdout.strip().split()
        for t in tokens:
            if t.count(".") == 3:
                return t
    except Exception as e:
        print("[WSL] unable to detect IP:", e)
    return "172.30.0.1"


# -------------------- WSL helpers --------------------

def wsl_run(cmd: str, print_cmd=False, capture=False):
    if print_cmd:
        print('[WSL] bash -lc "{}"'.format(cmd))
    if capture:
        return subprocess.run(["wsl", "bash", "-lc", cmd], text=True, capture_output=True)
    return subprocess.run(["wsl", "bash", "-lc", cmd])


def ensure_port_free_in_wsl(port: int):
    # Kill any process in WSL that is bound to this UDP port
    wsl_run(f"fuser -k {port}/udp 2>/dev/null || true")


def build_wsl_slam_command(cfg) -> str:
    repo = cfg["wsl_repo_dir"].rstrip("/")
    slam_bin = cfg.get("wsl_slam_binary", "./Examples/Monocular/mono_tello")
    vocab_rel = cfg.get("wsl_vocab_rel", "Vocabulary/ORBvoc.txt")
    settings_rel = cfg.get("wsl_settings_rel", "Examples/Monocular/TUM1.yaml")
    port = int(cfg.get("forward_port", 11111))
    headless = cfg.get("pangolin_headless", True)

    env = ""
    if headless:
        # make Pangolin run without a window / EGL
        env = "export PANGOLIN_WINDOW_URI=headless://; "

    # FFmpeg-style input for ORB-SLAM3: "udp://@:PORT"
    return f"{env}cd {repo} && {slam_bin} {vocab_rel} {settings_rel} \"udp://@:{port}\""


def launch_wsl_slam(cfg):
    """
    Launch ORB-SLAM3 inside WSL and return the Popen handle.
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


def drain_proc_output(proc, tag="[WSL]", lines=12):
    """Print a small burst of lines to show WSL progress."""
    if proc is None or proc.stdout is None:
        return
    try:
        for _ in range(lines):
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

def try_launch_wsl_slam_with_port_handling(cfg, fwd_port):
    """
    Ensure UDP port is free in WSL, launch SLAM, and if we detect 'bind failed',
    clear the port and retry once.
    """
    ensure_port_free_in_wsl(fwd_port)
    proc = launch_wsl_slam(cfg)
    sleep(0.5)
    drain_proc_output(proc, lines=10)

    # quick scan for bind failure
    failed = False
    try:
        if proc and proc.stdout:
            # non-blocking peek
            proc.stdout.flush()
    except Exception:
        pass

    # crude: fetch one more burst and look for errors
    buf = []
    try:
        for _ in range(6):
            line = proc.stdout.readline()
            if not line:
                break
            buf.append(line)
            if "bind failed" in line or "cannot open input" in line:
                failed = True
    except Exception:
        pass

    if buf:
        for ln in buf:
            print("[WSL]", ln.rstrip())

    if failed:
        print(f"[WSL] Detected UDP bind error. Freeing port {fwd_port} and retrying once...")
        ensure_port_free_in_wsl(fwd_port)
        try:
            proc.terminate()
        except Exception:
            pass
        sleep(0.5)
        proc = launch_wsl_slam(cfg)
        sleep(0.5)
        drain_proc_output(proc, lines=10)

    return proc


def drone_scan_with_slam(cfg, input_arg):
    """
    Full pipeline:
      - start WSL ORB-SLAM3 (mono_tello) headless and free UDP port
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
    if cfg.get("wsl_mode"):
        wsl_ip = detect_wsl_ip()
        fwd = UDPForwarder(wsl_ip, fwd_port)
        fwd.start()

        slam_proc = try_launch_wsl_slam_with_port_handling(cfg, fwd_port)
    else:
        # legacy Windows SLAM path (uses utils.runOrbSlam3)  :contentReference[oaicite:1]{index=1}
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

    tello_prepare_stream(drone, streamon_delay)

    print(f"[INFO] Waiting {slam_start_wait:.1f}s for ORB-SLAM3 to start...")
    sleep(slam_start_wait)
    drain_proc_output(slam_proc, lines=8)

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
            drain_proc_output(slam_proc, lines=4)

        # 2) Close SLAM gracefully: stopping stream lets mono_tello exit (WSL path passes proc=None).  :contentReference[oaicite:2]{index=2}
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
    x, y, z = readCSV(csv_path)  # utils.readCSV  :contentReference[oaicite:3]{index=3}
    if len(x) == 0:
        print("---- SLAM produced 0 points. Nothing to process. ----")
        if drone is not None:
            safe_land(drone); drone.end()
        raise SystemExit(0)

    pcd = makeCloud(x, y, z)     # utils.makeCloud  :contentReference[oaicite:4]{index=4}

    inlierPCD, outlierPCD = removeStatisticalOutlier(
        pcd,
        voxel_size=float(cfg["voxel_size"]),
        nb_neighbors=int(cfg["nb_neighbors"]),
        std_ratio=float(cfg["std_ratio"])
    )
    inX, inY, inZ = pcdToArrays(inlierPCD)  # utils.pcdToArrays  :contentReference[oaicite:5]{index=5}

    # room rectangle from top-down (X,Z)
    box = getAverageRectangle(inX, inZ)     # ExitFinding.getAverageRectangle

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

    xOut, zOut = pointsOutOfBox(inX, inZ, box)                   # utils.pointsOutOfBox  :contentReference[oaicite:7]{index=7}
    clusters = hierarchicalClustering(xOut, zOut, float(cfg["thresh"]))   # utils.hierarchicalClustering  :contentReference[oaicite:8]{index=8}
    centers  = getClustersCenters(clusters)                      # utils.getClustersCenters  :contentReference[oaicite:9]{index=9}

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
            ensure_airborne(drone)      # utils.ensure_airborne
            moveToExit(drone, centers)  # utils.moveToExit
        finally:
            safe_land(drone)            # utils.safe_land
            drone.end()
    else:
        plot2DWithClustersCenters(inX, inZ, centers,
                                  save_path=os.path.join(plots_dir, f"centers_only_{tag}.png"),
                                  title="Exits (cluster centers)")
        print("Exits (cluster centers):", centers)
