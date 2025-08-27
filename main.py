import os
import sys
from json import load
from time import sleep, time
from datetime import datetime

from djitellopy import Tello

from plot import *
from ExitFinding import *
from PointCloudCleaning import *
from utils import *

MAX_ANGLE = 360


def loadConfig():
    with open('config.json', 'r', encoding='utf-8') as f:
        return load(f)


def slam_root_from_exe(exe_path: str) -> str:
    # exe is ...\x64\Release\slam.exe  -> root is THREE dirs up: ...\ (ORB_SLAM3_Windows)
    return os.path.dirname(os.path.dirname(os.path.dirname(exe_path)))


def csv_path_from_config(cfg):
    root_dir = slam_root_from_exe(cfg["slam_exe"])
    return os.path.join(root_dir, "log", "pointData.csv")


def tello_prepare_stream(drone: Tello, delay_s: float):
    """Ensure Tello streaming is on and give UDP a moment to settle."""
    try:
        drone.streamoff()
    except Exception:
        pass
    drone.streamon()
    sleep(delay_s)


def append_telemetry(row_dict: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    header = None
    line = None
    # Dict -> header + csv line (stable column order)
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


def drone_scan_with_slam(cfg, input_arg):
    """
    Start ORB-SLAM3, wait EXACTLY slam_start_wait_s, then do a slow 360 scan with Tello.
    After scan, close SLAM CLEANLY (UDP: streamoff; webcam: ESC), then return (csv_path, drone).
    """
    vocab     = cfg["vocab"]
    settings  = cfg["settings"]
    slam_exe  = cfg["slam_exe"]
    source    = cfg["source"].lower()
    csv_path  = csv_path_from_config(cfg)

    # from config
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

    # show exactly what we'll run
    root_dir = slam_root_from_exe(slam_exe)
    exe_rel  = os.path.join("x64", "Release", "slam.exe")
    print(f'cd {root_dir}')
    print(f'.\\{exe_rel} mono_video "{vocab}" "{settings}" "{input_arg}"')

    # Launch SLAM (non-blocking)
    slam_proc = runOrbSlam3(slam_exe, vocab, settings, input_arg)

    if source != "tello":
        # Webcam/video only (no drone control here)
        slam_proc.wait()
        return csv_path, None

    # ---- Tello path ----
    drone = Tello()
    drone.connect()
    drone.set_speed(speed)

    # IMPORTANT: Start Tello stream BEFORE flight; for UDP input this provides frames to SLAM.
    tello_prepare_stream(drone, streamon_delay)

    # Warn if we're using a plain webcam index (easy to mix up with laptop camera)
    if str(input_arg).isdigit():
        print("[WARN] Using webcam index input for SLAM. Make sure this is your OBS VirtualCam that shows the Tello feed.")

    # Wait EXACTLY this long before any flight/rotation (let ORB-SLAM3 fully start).
    print(f"[INFO] Waiting {slam_start_wait:.1f}s for ORB-SLAM3 to start...")
    sleep(slam_start_wait)

    # Perform the 360 scan (slow, step-wise rotation + gentle nudge for parallax)
    drone.takeoff()
    try:
        # Trim altitude
        try:
            current_h = drone.get_height()
        except Exception:
            current_h = 0
        delta = int(target_height - current_h)
        if   delta > 0:
            drone.move_up(delta)
        elif delta < 0:
            drone.move_down(-delta)
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

            # log telemetry after each step
            try:
                yaw = drone.get_yaw()
            except Exception:
                yaw = ""
            try:
                h = drone.get_height()
            except Exception:
                h = ""
            try:
                bat = drone.get_battery()
            except Exception:
                bat = ""

            append_telemetry({
                "ts": datetime.now().isoformat(timespec="seconds"),
                "step_i": step_i,
                "rotated_deg_total": rotated + step,
                "yaw": yaw,
                "height_cm": h,
                "battery": bat,
                "cmd": f"cw {step}",
                "cmd_ok": "1" if ok else "0",
                "note": note
            }, telemetry_csv)

            rotated += step
            step_i += 1

            # Parallax nudge every N steps (helps monocular initialization/tracking)
            if do_nudge and (step_i % max(1, nudge_every_steps) == 0):
                try:
                    drone.move_up(nudge_cm)
                    drone.move_down(nudge_cm)
                    append_telemetry({
                        "ts": datetime.now().isoformat(timespec="seconds"),
                        "step_i": step_i,
                        "rotated_deg_total": rotated,
                        "yaw": yaw, "height_cm": h, "battery": bat,
                        "cmd": f"nudge {nudge_cm}",
                        "cmd_ok": "1",
                        "note": "up/down nudge"
                    }, telemetry_csv)
                except Exception as e:
                    append_telemetry({
                        "ts": datetime.now().isoformat(timespec="seconds"),
                        "step_i": step_i,
                        "rotated_deg_total": rotated,
                        "yaw": yaw, "height_cm": h, "battery": bat,
                        "cmd": f"nudge {nudge_cm}",
                        "cmd_ok": "0",
                        "note": f"nudge_err:{e}"
                    }, telemetry_csv)

            sleep(rotation_pause)

        # Close SLAM gracefully (UDP: stop stream; webcam: send ESC)
        close_slam(slam_proc, input_arg, drone=drone, wait_s=slam_exit_wait_s)

    except KeyboardInterrupt:
        print("[WARN] KeyboardInterrupt during scan.")
        close_slam(slam_proc, input_arg, drone=drone, wait_s=8.0)
        raise
    except Exception as e:
        print("Scan error:", e)
        # Try to close anyway
        close_slam(slam_proc, input_arg, drone=drone, wait_s=8.0)

    # Keep flying; caller will compute exit and move now
    return csv_path, drone


if __name__ == '__main__':
    cfg = loadConfig()

    # Choose input for ORB-SLAM3
    source = cfg["source"].lower()
    if source == "tello":
        tello_mode = cfg.get("tello_input_mode", "udp").lower()
        if tello_mode == "udp":
            input_arg = "TELLO"                                  # <-- direct Tello UDP to mono_video
        elif tello_mode == "webcam":
            input_arg = str(cfg.get("tello_webcam_index", "0"))  # OBS VirtualCam or similar
        else:
            print(f"Unknown tello_input_mode: {tello_mode}")
            sys.exit(1)
    elif source == "webcam":
        input_arg = str(cfg.get("webcam_index", "0"))
    else:
        input_arg = cfg["video_path"]

    # 1) Run SLAM + Tello scan
    try:
        csv_path, drone = drone_scan_with_slam(cfg, input_arg)
    except KeyboardInterrupt:
        # Land safely if user aborts
        try:
            if 'drone' in locals() and drone is not None:
                safe_land(drone)
                drone.end()
        except Exception:
            pass
        sys.exit(1)

    # 2) Verify CSV exists
    if not os.path.exists(csv_path):
        root_dir = slam_root_from_exe(cfg["slam_exe"])
        exe_rel  = os.path.join("x64", "Release", "slam.exe")
        log_dir  = os.path.join(root_dir, "log")
        print("\nERROR: pointData.csv not found.")
        print(f"Expected here: {csv_path}")
        print(f"Check the SLAM output directory: {log_dir}")
        print("\nRun the exe manually to see console errors:")
        print(f"     cd {root_dir}")
        print(f"     .\\{exe_rel} mono_video \"{cfg['vocab']}\" \"{cfg['settings']}\" \"{input_arg}\"\n")
        if drone is not None:
            safe_land(drone); drone.end()
        raise SystemExit(1)

    # 3) Process map
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

    # ceiling POV: use X,Z
    box = getAverageRectangle(inX, inZ)

    # DEBUG PLOTS (also saved to disk)
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

    # 4) Navigate to exit (keep airborne from the scan)
    if len(centers) == 0:
        print("No exits detected. Done.")
        # Save a plot that shows 'no exits'
        plot2DWithBoxAndCenters(inX, inZ, box, centers=[],
                                save_path=os.path.join(plots_dir, f"centers_none_{tag}.png"),
                                title="No exits detected")
        if drone is not None:
            safe_land(drone); drone.end()
        sys.exit(0)

    # Save a centers plot before moving
    plot2DWithBoxAndCenters(inX, inZ, box, centers,
                            save_path=os.path.join(plots_dir, f"centers_{tag}.png"),
                            title="Detected Exit Cluster Centers (X,Z)")

    if drone is not None:
        try:
            ensure_airborne(drone)      # in case motors timed out
            moveToExit(drone, centers)  # rotate & fly forward
        finally:
            safe_land(drone)
            drone.end()
    else:
        plot2DWithClustersCenters(inX, inZ, centers,
                                  save_path=os.path.join(plots_dir, f"centers_only_{tag}.png"),
                                  title="Exits (cluster centers)")
        print("Exits (cluster centers):", centers)
