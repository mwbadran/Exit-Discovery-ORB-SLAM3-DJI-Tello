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
from utils import *  # readCSV / makeCloud / etc.

MAX_ANGLE = 360


# -------------------- config helpers --------------------

def loadConfig():
    with open('config.json', 'r', encoding='utf-8') as f:
        return load(f)


def slam_root_from_exe(exe_path: str) -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(exe_path)))


def csv_path_from_config(cfg):
    if cfg.get("wsl_mode"):
        return cfg["wsl_point_csv"]
    root_dir = slam_root_from_exe(cfg["slam_exe"])
    return os.path.join(root_dir, "log", "pointData.csv")


# -------------------- UDP forwarder (Windows -> WSL) --------------------

class UDPForwarder(threading.Thread):
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
            self.insock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
            self.insock.bind(("0.0.0.0", self.port))

            self.outsock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.outsock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 * 1024 * 1024)
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
    try:
        with open('config.json', 'r', encoding='utf-8') as f:
            cfg = load(f)
            if cfg.get("wsl_target_ip"):
                print(f"[WSL] using override target IP: {cfg['wsl_target_ip']}")
                return cfg["wsl_target_ip"]
    except Exception:
        pass

    try:
        res = subprocess.run(["wsl", "hostname", "-I"], capture_output=True, text=True, check=True)
        tokens = res.stdout.strip().split()
        for t in tokens:
            if t.count(".") == 3:
                print(f"[WSL] detected IP via hostname -I: {t}")
                return t
    except Exception as e:
        print("[WSL] unable to detect IP:", e)
    return "172.30.0.1"


# -------------------- WSL ORB-SLAM3 launcher + reader --------------------

def build_wsl_slam_command(cfg) -> str:
    repo = cfg["wsl_repo_dir"].rstrip("/")
    vocab_rel    = cfg.get("wsl_vocab_rel", "Vocabulary/ORBvoc.txt")
    settings_rel = cfg.get("wsl_settings_rel", "Examples/Monocular/telloCal.yaml")
    source       = cfg.get("source", "tello").lower()
    port         = int(cfg.get("forward_port", 11111))

    if source == "tello":
        slam_bin = cfg.get("wsl_slam_binary", "./Examples/Monocular/mono_tello")
        # IMPORTANT: use 0.0.0.0 (portable), keep options minimal & safe
        url = f"udp://0.0.0.0:{port}?overrun_nonfatal=1&fifo_size=50000000"
        input_arg = f"'{url}'"   # single-quotes for bash -lc
    elif source == "webcam":
        slam_bin = cfg.get("wsl_slam_binary_video", "./Examples/Monocular/mono_input")
        input_arg = str(cfg.get("wsl_webcam_index", "0"))
    else:  # video
        slam_bin = cfg.get("wsl_slam_binary_video", "./Examples/Monocular/mono_input")
        input_arg = f"'{cfg.get('wsl_video_path','')}'"

    prefix = "export ORB3_VIEWER=0; " if cfg.get("wsl_disable_viewer") else ""
    # mono_tello should print "[READY] capture_open" right after cap.isOpened()
    return f"cd {repo} && {prefix}{slam_bin} {vocab_rel} {settings_rel} {input_arg}"


def launch_wsl_slam(cfg):
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


class ProcReader(threading.Thread):
    def __init__(self, proc, token=None):
        super().__init__(daemon=True)
        self.proc = proc
        self.token = token
        self.ready_event = threading.Event()

    def run(self):
        if self.proc is None or self.proc.stdout is None:
            return
        try:
            for line in iter(self.proc.stdout.readline, ''):
                if not line:
                    break
                line = line.rstrip('\n')
                print(f"[WSL] {line}")
                if self.token and self.token in line:
                    self.ready_event.set()
        except Exception:
            pass


def wait_for_ready(proc, token="[READY] capture_open", timeout=12.0) -> bool:
    if proc is None:
        return False
    reader = ProcReader(proc, token=token)
    reader.start()
    ok = reader.ready_event.wait(timeout=timeout)
    if not ok:
        print("[WSL] WARNING: didn't see READY within timeout")
    return ok


def drain_proc_output(proc, burst_lines=8, tag="[WSL]"):
    if proc is None or proc.stdout is None:
        return
    try:
        for _ in range(burst_lines):
            line = proc.stdout.readline()
            if not line:
                break
            print(f"{tag} {line.rstrip()}")
    except Exception:
        pass


# -------------------- local slam closer --------------------

def safe_close_slam(proc, input_arg, drone=None, wait_s=15.0):
    if str(input_arg).upper() == "TELLO":
        if drone is not None:
            try:
                drone.streamoff()
            except Exception:
                pass
        if proc is not None:
            try:
                proc.wait(timeout=wait_s)
                return
            except subprocess.TimeoutExpired:
                pass
        sleep(min(5.0, wait_s))
        return

    try:
        press_esc_to_slam_window()
    except Exception:
        pass
    if proc is not None:
        try:
            proc.wait(timeout=wait_s); return
        except subprocess.TimeoutExpired:
            pass
        try:
            proc.terminate(); proc.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
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
    vocab     = cfg["vocab"]
    settings  = cfg["settings"]
    slam_exe  = cfg["slam_exe"]
    source    = cfg["source"].lower()
    csv_path  = csv_path_from_config(cfg)

    streamon_delay     = float(cfg.get("streamon_delay_sec", 0.5))
    slam_start_wait    = float(cfg.get("slam_start_wait_s", 2.0))
    rotation_step      = int(cfg.get("rotation_step_deg", 10))
    rotation_pause     = float(cfg.get("rotation_pause_s", 0.4))
    target_height      = int(cfg.get("height", 70))
    speed              = int(cfg.get("speed", 10))
    slam_exit_wait_s   = float(cfg.get("slam_exit_wait_s", 15.0))
    do_nudge           = bool(cfg.get("do_nudge", True))
    nudge_cm           = int(cfg.get("nudge_cm", 20))
    nudge_every_steps  = int(cfg.get("nudge_every_steps", 2))
    telemetry_csv      = cfg.get("telemetry_csv", os.path.join(slam_root_from_exe(slam_exe), "log", "telemetry.csv"))
    fwd_port           = int(cfg.get("forward_port", 11111))

    slam_proc = None
    fwd = None

    if cfg.get("wsl_mode"):
        wsl_ip = detect_wsl_ip()
        fwd = UDPForwarder(wsl_ip, fwd_port)
        fwd.start()
        slam_proc = launch_wsl_slam(cfg)
        wait_for_ready(slam_proc, timeout=12.0)
    else:
        print("[LEGACY] launching Windows slam.exe")
        slam_proc = runOrbSlam3(slam_exe, vocab, settings, input_arg)

    drone = None
    if source == "tello":
        drone = Tello()
        drone.connect()
        drone.set_speed(speed)

        tello_prepare_stream(drone, streamon_delay)

        print(f"[INFO] Waiting {slam_start_wait:.1f}s for ORB-SLAM3 to start...")
        sleep(slam_start_wait)
        drain_proc_output(slam_proc)

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

            safe_close_slam(slam_proc, input_arg, drone=drone, wait_s=slam_exit_wait_s)

        except KeyboardInterrupt:
            print("[WARN] KeyboardInterrupt during scan.")
            safe_close_slam(slam_proc, input_arg, drone=drone, wait_s=8.0)
            raise
        except Exception as e:
            print("Scan error:", e)
            safe_close_slam(slam_proc, input_arg, drone=drone, wait_s=8.0)

    else:
        if slam_proc is not None:
            try:
                slam_proc.wait()
            except Exception:
                pass

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
        print("If running WSL, confirm mono_tello wrote to ~/dev/ORB_SLAM3/log/pointData.csv")
        print("Also check that the UDP forwarder started and WSL IP was detected.")
        if source != "tello":
            print(f"\n(Windows legacy) You can also run exe manually:")
            print(f"     cd {root_dir}")
            print(f"     .\\{exe_rel} \"mono_video\" \"{cfg['vocab']}\" \"{cfg['settings']}\" \"{input_arg}\"\n")
        if drone is not None:
            safe_land(drone); drone.end()
        raise SystemExit(1)

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

    box = getAverageRectangle(inX, inZ)

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

    if 'drone' in locals() and drone is not None:
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
