# main.py
import os, sys, socket, threading, subprocess, shlex
from json import load
from time import sleep
from datetime import datetime
from typing import Optional

from djitellopy import Tello

# your modules
from plot import *
from ExitFinding import *
from PointCloudCleaning import *
from utils import *  # safe_land, ensure_airborne, moveToExit, readCSV, makeCloud, close_slam, etc.

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
        # UNC path so Windows can read WSL output
        return cfg.get("wsl_point_csv", os.path.expanduser("~/dev/ORB_SLAM3/log/pointData.csv"))
    root_dir = slam_root_from_exe(cfg["slam_exe"])
    return os.path.join(root_dir, "log", "pointData.csv")


# -------------------- non-blocking proc output drain --------------------
import os as _os
if _os.name == "nt":
    import msvcrt, ctypes
    from ctypes import wintypes
    def _pipe_has_data(pipe) -> bool:
        try:
            h = msvcrt.get_osfhandle(pipe.fileno())
            avail = wintypes.DWORD()
            ok = ctypes.windll.kernel32.PeekNamedPipe(h, None, 0, None,
                                                      ctypes.byref(avail), None)
            return bool(ok) and avail.value > 0
        except Exception:
            return False
else:
    import select
    def _pipe_has_data(pipe) -> bool:
        try:
            r, _, _ = select.select([pipe], [], [], 0)
            return bool(r)
        except Exception:
            return False

def drain_proc_output(proc, tag="[WSL]", max_lines=20):
    if proc is None or proc.stdout is None: return
    lines = 0
    while lines < max_lines and _pipe_has_data(proc.stdout):
        try:
            line = proc.stdout.readline()
        except Exception:
            break
        if not line: break
        print(f"{tag} {line.rstrip()}")
        lines += 1


# -------------------- Windows → WSL UDP forwarder --------------------
class UDPForwarder(threading.Thread):
    """
    Binds 0.0.0.0:in_port on Windows and forwards to (WSL_IP, out_port).
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
                if not data: continue
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


# -------------------- WSL helpers (FFmpeg + SLAM) --------------------
def ensure_wsl_has_ffmpeg():
    try:
        subprocess.run(["wsl", "bash", "-lc", "command -v ffmpeg >/dev/null"], check=True)
        return True
    except Exception:
        print("[WSL] ffmpeg not found. Install with: sudo apt-get update && sudo apt-get install -y ffmpeg")
        return False

def stop_proc(proc, name="proc", timeout=5.0):
    if not proc: return
    try:
        proc.terminate()
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try: proc.kill()
        except Exception: pass
    except Exception:
        pass
    print(f"[INFO] {name} stopped.")

def build_wsl_slam_command(cfg, input_for_slam: str):
    repo         = cfg["wsl_repo_dir"].rstrip("/")
    vocab_rel    = cfg.get("wsl_vocab_rel", "Vocabulary/ORBvoc.txt")
    settings_rel = cfg.get("wsl_settings_rel", "Examples/Monocular/telloCal.yaml")
    slam_bin     = cfg.get("wsl_slam_binary_video", "./Examples/Monocular/mono_input")
    return f"cd {repo} && {slam_bin} {vocab_rel} {settings_rel} {input_for_slam}"

def launch_wsl_slam(cfg, input_for_slam: str, is_path=True):
    # Quote paths, not URLs
    arg = f"\"{input_for_slam}\"" if is_path else input_for_slam
    cmd = build_wsl_slam_command(cfg, arg)
    print("[WSL] launching ORB-SLAM3:\n    wsl bash -lc \"{}\"".format(cmd))
    try:
        proc = subprocess.Popen(["wsl", "bash", "-lc", cmd],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        return proc
    except Exception as e:
        print("[WSL] failed to launch ORB-SLAM3:", e)
        return None

def _wsl_wait_nonempty(path, timeout_s=10.0):
    """Wait until a WSL file exists AND has non-zero size."""
    checks = int(max(1, timeout_s * 10))
    q = shlex.quote(path)
    cmd = f"bash -lc 'for i in $(seq 1 {checks}); do [ -s {q} ] && echo READY && exit 0; sleep 0.1; done; echo TIMEOUT; exit 1;'"
    try:
        res = subprocess.run(["wsl", "-e"] + cmd.split(), capture_output=True, text=True, timeout=timeout_s+3)
        return "READY" in (res.stdout + res.stderr)
    except Exception:
        return False

def start_wsl_bridge(cfg):
    """
    Returns (mode, ffmpeg_proc, input_for_slam, is_path)
      mode: "avi" | "fifo" | "http"
      input_for_slam: path or URL for mono_input
      is_path: True if it's a filesystem path (quote it for bash)
    """
    mode = str(cfg.get("bridge_mode", "avi")).lower()
    in_port = int(cfg.get("ffmpeg_in_port", 11113))
    udp_opts = cfg.get("wsl_udp_query", "overrun_nonfatal=1&fifo_size=5000000")

    if mode == "avi":
        avi_path = cfg.get("wsl_avi_path", "/tmp/slam.avi")
        # MJPEG-in-AVI, low-latency flags; easy for OpenCV to read while growing
        cmd = (
            f"set -e;"
            f" rm -f {shlex.quote(avi_path)} 2>/dev/null || true;"
            f" ffmpeg -hide_banner -loglevel warning "
            f" -fflags nobuffer+genpts -use_wallclock_as_timestamps 1 "
            f" -i 'udp://0.0.0.0:{in_port}?{udp_opts}' "
            f" -an -c:v mjpeg -q:v 3 -vf fps=30 "
            f" -flush_packets 1 -f avi {shlex.quote(avi_path)}"
        )
        print("[WSL] starting FFmpeg bridge -> AVI")
        proc = subprocess.Popen(["wsl", "bash", "-lc", cmd],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        # wait until file appears and has data
        _wsl_wait_nonempty(avi_path, timeout_s=5.0)
        return ("avi", proc, avi_path, True)

    if mode == "http":
        port = int(cfg.get("wsl_http_port", 8090))
        path = cfg.get("wsl_http_path", "/stream.mjpg")
        url = f"http://127.0.0.1:{port}{path}"
        # Serve multipart MJPEG over HTTP
        cmd = (
            f"set -e;"
            f" ffmpeg -hide_banner -loglevel warning "
            f" -fflags nobuffer+genpts -use_wallclock_as_timestamps 1 "
            f" -i 'udp://0.0.0.0:{in_port}?{udp_opts}' "
            f" -an -c:v mjpeg -q:v 5 -vf fps=30 "
            f" -f mpjpeg -listen 1 'http://0.0.0.0:{port}{path}'"
        )
        print("[WSL] starting FFmpeg bridge -> HTTP MJPEG")
        proc = subprocess.Popen(["wsl", "bash", "-lc", cmd],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        sleep(0.5)
        return ("http", proc, url, False)

    # default: FIFO (previous approach)
    fifo_path = cfg.get("wsl_fifo_path", "/tmp/slam.ts")
    cmd = (
        f"set -e;"
        f" if [ -p {shlex.quote(fifo_path)} ]; then rm -f {shlex.quote(fifo_path)}; fi;"
        f" mkfifo -m 666 {shlex.quote(fifo_path)};"
        f" ffmpeg -hide_banner -loglevel warning "
        f" -fflags +genpts -use_wallclock_as_timestamps 1 "
        f" -i 'udp://0.0.0.0:{in_port}?{udp_opts}' "
        f" -an -c copy -f mpegts {shlex.quote(fifo_path)}"
    )
    print("[WSL] starting FFmpeg bridge -> FIFO")
    proc = subprocess.Popen(["wsl", "bash", "-lc", cmd],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    sleep(0.3)
    return ("fifo", proc, fifo_path, True)


# -------------------- Tello helpers --------------------
def tello_prepare_stream(drone: Tello, delay_s: float):
    """
    Windows is the last-sender; keep the stream target on Windows:11111.
    """
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
    Modes:
      A) source == "tello" and ffmpeg_bridge == true:
         - Windows→WSL forwarder (11111 → ffmpeg_in_port)
         - WSL ffmpeg writes AVI/FIFO OR serves HTTP
         - WSL mono_input reads that path/URL
      B) source == "video":
         - WSL mono_input reads cfg["wsl_video_path"] directly
    Returns (csv_path, drone_or_none).
    """
    csv_path  = csv_path_from_config(cfg)
    source    = cfg["source"].lower()
    ffmpeg_bridge = bool(cfg.get("ffmpeg_bridge", True))

    # config knobs
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
    telemetry_csv      = cfg.get("telemetry_csv", os.path.join(slam_root_from_exe(cfg["slam_exe"]), "log", "telemetry.csv"))

    # processes we may start
    fwd = None
    bridge_proc = None
    slam_proc = None
    drone: Optional[Tello] = None

    # VIDEO-ONLY PATH
    if source == "video":
        input_path_for_slam = cfg.get("wsl_video_path", "")
        if not input_path_for_slam:
            print("video mode: missing 'wsl_video_path' in config.json")
            return csv_path, None
        slam_proc = launch_wsl_slam(cfg, input_path_for_slam, is_path=True)
        drain_proc_output(slam_proc)
        try:
            slam_proc.wait()
        except Exception:
            pass
        return csv_path, None

    # TELLO + FFmpeg bridge PATH
    if source == "tello" and ffmpeg_bridge:
        if not ensure_wsl_has_ffmpeg():
            print("ffmpeg not found in WSL. Install it and retry.")
            return csv_path, None

        # 1) UDP forwarder: Windows 11111 -> WSL ffmpeg_in_port (e.g., 11113)
        wsl_ip = detect_wsl_ip()
        in_port_windows = int(cfg.get("forward_port_in", 11111))
        ffmpeg_in_port  = int(cfg.get("ffmpeg_in_port", 11113))
        fwd = UDPForwarder(wsl_ip, in_port=in_port_windows, out_port=ffmpeg_in_port)
        fwd.start()

        # 2) Start bridge (AVI/FIFO/HTTP)
        mode, bridge_proc, input_for_slam, is_path = start_wsl_bridge(cfg)
        sleep(0.4)
        drain_proc_output(bridge_proc, tag=f"[FFMPEG:{mode}]")

        # 3) Start SLAM (mono_input) consuming path/URL
        slam_proc = launch_wsl_slam(cfg, input_for_slam, is_path=is_path)
        drain_proc_output(slam_proc)

        # 4) Tello control (Windows) — after SLAM is ready
        drone = Tello()
        drone.connect()
        drone.set_speed(speed)
        try:
            print(f"[INFO] Battery: {drone.get_battery()}%")
        except Exception as e:
            print(f"[WARN] Could not read battery: {e}")

        tello_prepare_stream(drone, streamon_delay)
        print(f"[INFO] Waiting {slam_start_wait:.1f}s for ORB-SLAM3 to warm up...")
        sleep(slam_start_wait)
        drain_proc_output(slam_proc)

        # 5) 360° scan with optional nudges
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

            # 6) stop stream and processes, then land
            try:
                drone.streamoff()
            except Exception:
                pass

            # stop FFmpeg first so SLAM sees EOF / stream end
            stop_proc(bridge_proc, name=f"ffmpeg:{mode}", timeout=3.0)

            if slam_proc:
                try: slam_proc.wait(timeout=int(slam_exit_wait_s))
                except subprocess.TimeoutExpired:
                    try: slam_proc.terminate()
                    except Exception: pass

        except KeyboardInterrupt:
            print("[WARN] KeyboardInterrupt during scan.")
            try: drone.streamoff()
            except Exception: pass
            stop_proc(bridge_proc, name=f"ffmpeg:{mode}", timeout=3.0)
            if slam_proc:
                try: slam_proc.terminate()
                except Exception: pass
        except Exception as e:
            print("Scan error:", e)
            try: drone.streamoff()
            except Exception: pass
            stop_proc(bridge_proc, name=f"ffmpeg:{mode}", timeout=3.0)
            if slam_proc:
                try: slam_proc.terminate()
                except Exception: pass
        finally:
            if fwd:
                fwd.stop(); fwd.join(timeout=3.0)
            if drone is not None:
                safe_land(drone)

        return csv_path, drone

    print("source=tello but ffmpeg_bridge=false. Enable it or switch source to video.")
    return csv_path, None


# -------------------- entrypoint --------------------
def main():
    cfg = loadConfig()

    # scan angles
    global EXTRA_ANGLE, MAX_ANGLE_WOUT_EXTRA, MAX_ANGLE
    EXTRA_ANGLE = int(cfg.get("extra_angle", EXTRA_ANGLE))
    MAX_ANGLE_WOUT_EXTRA = int(cfg.get("max_angle_wout_extra", MAX_ANGLE_WOUT_EXTRA))
    MAX_ANGLE = MAX_ANGLE_WOUT_EXTRA + EXTRA_ANGLE

    # input arg (legacy / for messages & utils.close_slam)
    source = cfg["source"].lower()
    if source == "tello":
        input_arg = "TELLO"
    elif source == "webcam":
        input_arg = str(cfg.get("webcam_index", "0"))
    else:
        input_arg = cfg.get("video_path", "")

    csv_path, drone = drone_scan_with_slam(cfg, input_arg)

    # post-process pointcloud
    if not os.path.exists(csv_path):
        root_dir = slam_root_from_exe(cfg["slam_exe"])
        exe_rel  = os.path.join("x64", "Release", "slam.exe")
        print("\nERROR: pointData.csv not found.")
        print(f"Expected here: {csv_path}")
        print("Confirm SLAM consumed the AVI/FIFO/HTTP bridge and FFmpeg was running (tello mode).")
        if source == "video":
            print(f"\n(Windows legacy) You can also run exe manually:")
            print(f"     cd {root_dir}")
            print(f"     .\\{exe_rel} \"mono_video\" \"{cfg['vocab']}\" \"{cfg['settings']}\" \"{input_arg}\"\n")
        if drone is not None:
            try: drone.end()
            except Exception: pass
        return

    x, y, z = readCSV(csv_path)
    if len(x) == 0:
        print("---- SLAM produced 0 points. Nothing to process. ----")
        if drone is not None:
            try: drone.end()
            except Exception: pass
        return

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
        if drone is not None:
            try: drone.end()
            except Exception: pass
        return

    plot2DWithBoxAndCenters(inX, inZ, box, centers,
                            save_path=os.path.join(plots_dir, f"centers_{tag}.png"),
                            title="Detected Exit Cluster Centers (X,Z)")

    if drone is not None:
        try:
            ensure_airborne(drone)
            moveToExit(drone, centers)
        finally:
            safe_land(drone)
            try: drone.end()
            except Exception: pass
    else:
        plot2DWithClustersCenters(inX, inZ, centers,
                                  save_path=os.path.join(plots_dir, f"centers_only_{tag}.png"),
                                  title="Exits (cluster centers)")
        print("Exits (cluster centers):", centers)


if __name__ == '__main__':
    main()
