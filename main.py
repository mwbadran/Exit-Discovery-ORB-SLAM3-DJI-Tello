import os
import sys
from json import load
from time import sleep
from threading import Thread

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

def drone_scan_with_slam(cfg, input_arg):
    """
    Start ORB-SLAM3, wait EXACTLY slam_start_wait_s, then do a smooth 360 scan with Tello.
    After scan, stop SLAM, read outputs, and return (csv_path, drone) while keeping the drone flying.
    """
    vocab     = cfg["vocab"]
    settings  = cfg["settings"]
    slam_exe  = cfg["slam_exe"]
    source    = cfg["source"].lower()
    csv_path  = csv_path_from_config(cfg)

    #get from config
    streamon_delay = float(cfg["streamon_delay_sec"])
    slam_start_wait = float(cfg["slam_start_wait_s"])
    rotation_step = int(cfg["rotation_step_deg"])
    rotation_pause = float(cfg["rotation_pause_s"])
    target_height = int(cfg["height"])
    speed = int(cfg["speed"])

    # prepare orb slam and print
    root_dir = slam_root_from_exe(slam_exe)
    exe_rel  = os.path.join("x64", "Release", "slam.exe")
    print(f'cd {root_dir}')
    print(f'.\\{exe_rel} mono_video "{vocab}" "{settings}" "{input_arg}"')

    # Start SLAM (now returns a process handle)
    slam_proc = runOrbSlam3(slam_exe, vocab, settings, input_arg)

    if source != "tello":
        # Webcam/video: just wait for SLAM to end elsewhere (e.g., ESC)
        slam_proc.wait()
        return csv_path, None

    # ---- Tello path ----
    drone = Tello()
    drone.connect()
    drone.set_speed(speed)

    # Start Tello camera stream (before flying)
    tello_prepare_stream(drone, streamon_delay)

    # *** EXACT 10s wait before any flight ***
    sleep(slam_start_wait)

    # Perform the 360 scan (your current stepwise rotation)
    drone.takeoff()
    try:
        # altitude trim...
        try:
            current_h = drone.get_height()
        except Exception:
            current_h = 0
        delta = int(target_height - current_h)
        if delta > 0:
            drone.move_up(delta)
        elif delta < 0:
            drone.move_down(-delta)
        sleep(0.5)

        rotated = 0
        while rotated < MAX_ANGLE:
            step = min(rotation_step, MAX_ANGLE - rotated)
            drone.rotate_clockwise(step)
            rotated += step
            sleep(rotation_pause)

        # Auto-close SLAM cleanly
        close_slam(slam_proc, input_arg, drone=drone, wait_s=15.0)

    except Exception as e:
        print("Scan error:", e)
        # Try to close anyway
        close_slam(slam_proc, input_arg, drone=drone, wait_s=8.0)

    # Keep flying; caller will compute exit and move now
    return csv_path, drone


if __name__ == '__main__':
    cfg = loadConfig()

    # source = tello || webcam....
    source = cfg["source"].lower()

    if source == "tello":
        tello_mode = cfg.get("tello_input_mode", "udp").lower()   # udp (we dont use it)
        if tello_mode == "webcam":
            input_arg = str(cfg.get("tello_webcam_index", "0"))   # OBS VirtualCam index
    elif source == "webcam":
        input_arg = str(cfg.get("webcam_index", "0"))
    else:
        input_arg = cfg["video_path"]

    # 1) Run SLAM + Tello scan
    csv_path, drone = drone_scan_with_slam(cfg, input_arg)

    # 2) Verify CSV exists
    if not os.path.exists(csv_path):
        root_dir = slam_root_from_exe(cfg["slam_exe"])
        exe_rel  = os.path.join("x64", "Release", "slam.exe")
        log_dir  = os.path.join(root_dir, "log")
        print("\nERROR: pointData.csv not found.")
        print(f"Expected here: {csv_path}")
        print(f"Check the SLAM output directory: {log_dir}")
        print("\nrun the exe manually to see console errors:")
        print(f"     cd {root_dir}")
        print(f"     .\\{exe_rel} mono_video \"{cfg['vocab']}\" \"{cfg['settings']}\" \"{input_arg}\"\n")
        if drone is not None:
            safe_land(drone); drone.end()
        raise SystemExit(1)

    # 3) Process map
    x, y, z = readCSV(csv_path)
    if len(x) == 0:
        print("----SLAM produced 0 points. Nothing to process.----")
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
    plot2DWithBox(inX, inZ, box)

    xOut, zOut = pointsOutOfBox(inX, inZ, box)
    clusters = hierarchicalClustering(xOut, zOut, float(cfg["thresh"]))
    centers  = getClustersCenters(clusters)

    # 4) Navigate to exit (keep airborne from the scan)
    if len(centers) == 0:
        print("No exits detected. Done.")
        if drone is not None:
            safe_land(drone); drone.end()
        sys.exit(0)

    if drone is not None:
        try:
            ensure_airborne(drone)      # in case motors timed out
            moveToExit(drone, centers)  # rotate & fly forward
        finally:
            safe_land(drone)
            drone.end()
    else:
        plot2DWithClustersCenters(inX, inZ, centers)
        print("Exits (cluster centers):", centers)
