import os
import subprocess
import ctypes
from time import sleep
from math import sqrt, atan2, degrees
from numpy import array, asarray, max as npmax, vstack
from pandas import DataFrame
from pyntcloud import PyntCloud
import scipy.cluster.hierarchy as hcluster
import open3d as o3d

# ---------- helpers for paths ----------

def _slam_root_from_exe(slam_exe: str) -> str:
    # exe is ...\x64\Release\slam.exe -> root is THREE dirs up
    return os.path.dirname(os.path.dirname(os.path.dirname(slam_exe)))

def _exe_rel_from_root(slam_exe: str) -> str:
    root = _slam_root_from_exe(slam_exe)
    return os.path.relpath(slam_exe, start=root)  # "x64\\Release\\slam.exe"

def _maybe_dirs(*candidates):
    return [p for p in candidates if p and os.path.isdir(p)]

def _auto_opencv_bins(root: str):
    # Try common OpenCV bin locations under your repo
    candidates = []
    for vc in ("vc17", "vc16", "vc15", "vc14"):
        candidates.append(os.path.join(root, "opencv", "build", "x64", vc, "bin"))
        candidates.append(os.path.join(root, "3rdparty", "opencv", "build", "x64", vc, "bin"))
    return _maybe_dirs(*candidates)

# ---------- launch ORB-SLAM3 (non-blocking) ----------

def runOrbSlam3(slam_exe, vocab, settings, input_arg):
    """Launch ORB-SLAM3 and return a Popen handle (non-blocking)."""
    root    = _slam_root_from_exe(slam_exe)
    exe_abs = os.path.abspath(slam_exe)

    env = os.environ.copy()
    path_parts = [
        os.path.dirname(exe_abs),
        os.path.join(root, "bin"),
        *(_auto_opencv_bins(root))
    ]
    extra = env.get("ORB3_EXTRA_DLL_DIRS", "")
    if extra:
        for p in extra.split(";"):
            p = p.strip()
            if p:
                path_parts.append(p)
    env["PATH"] = os.pathsep.join(path_parts + [env.get("PATH", "")])

    print("Running from root:", root)
    print("Command:", f'{os.path.join("x64","Release","slam.exe")} mono_video "{vocab}" "{settings}" "{input_arg}"')

    CREATE_NEW_PROCESS_GROUP = 0x00000200 if os.name == "nt" else 0
    return subprocess.Popen(
        [exe_abs, "mono_video", vocab, settings, input_arg],
        cwd=root, env=env, creationflags=CREATE_NEW_PROCESS_GROUP
    )

# ---------- graceful close for SLAM ----------

_user32 = ctypes.windll.user32 if os.name == "nt" else None
WM_KEYDOWN, WM_KEYUP = 0x0100, 0x0101
VK_ESCAPE = 0x1B

def _post_key(hwnd, vk):
    _user32.PostMessageW(hwnd, WM_KEYDOWN, vk, 0)
    _user32.PostMessageW(hwnd, WM_KEYUP,   vk, 0)

def press_esc_to_slam_window():
    """
    Best-effort: send ESC to the SLAM window (webcam input mode).
    """
    if _user32 is None:
        return False

    candidates = ["Frame", "ORB-SLAM3: Map", "ORB-SLAM3"]
    for title in candidates:
        hwnd = _user32.FindWindowW(None, title)
        if hwnd:
            try:
                _user32.SetForegroundWindow(hwnd)
                sleep(0.1)
            except Exception:
                pass
            _post_key(hwnd, VK_ESCAPE)
            sleep(0.2)
            return True

    # Fallback: send ESC to active window
    try:
        _user32.keybd_event(VK_ESCAPE, 0, 0, 0)
        _user32.keybd_event(VK_ESCAPE, 0, 2, 0)  # KEYEVENTF_KEYUP = 2
        sleep(0.2)
        return True
    except Exception:
        return False

def close_slam(proc, input_arg, drone=None, wait_s=15.0):
    """
    Close ORB-SLAM3 cleanly:
      - If input_arg == 'TELLO' (UDP): stop Tello stream to end SLAM main loop, then wait.
      - Else (webcam/video): send ESC to the window, then wait.
    Only if it *still* won't exit, terminate/kill as a last resort.
    """
    if proc is None:
        return

    # UDP (Tello) path → stop stream to end SLAM loop
    if str(input_arg).upper() == "TELLO":
        if drone is not None:
            try:
                drone.streamoff()
            except Exception:
                pass
        try:
            proc.wait(timeout=wait_s)
            return
        except subprocess.TimeoutExpired:
            # Try ESC too, just in case a window is open
            press_esc_to_slam_window()

    else:
        # Webcam/video: trigger graceful exit with ESC
        press_esc_to_slam_window()
        try:
            proc.wait(timeout=wait_s)
            return
        except subprocess.TimeoutExpired:
            pass

    # Gentle terminate, then kill if necessary
    try:
        proc.terminate()
        proc.wait(timeout=3.0)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except Exception:
            pass

# ---------- geometry / clustering ----------

def readCSV(path):
    xs, ys, zs = [], [], []
    with open(path, 'r', encoding='utf-8', newline='') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 3:
                continue
            # skip header or any non-numeric first row(s)
            if i == 0 and (parts[0].lower() == 'x' or parts[1].lower() == 'y' or parts[2].lower() == 'z'):
                continue
            try:
                xs.append(float(parts[0]))
                ys.append(float(parts[1]))
                zs.append(float(parts[2]))
            except ValueError:
                # ignore stray malformed lines (e.g., partial writes)
                continue
    return xs, ys, zs


def distanceBetween2Points(point1, point2):
    dx = point1[0] - point2[0]
    dy = point1[1] - point2[1]
    return sqrt(dx*dx + dy*dy)

def pcdToArrays(pcd):
    pts = asarray(pcd.points)
    x, y, z = [], [], []
    for p in pts:
        x.append(float(p[0])); y.append(float(p[1])); z.append(float(p[2]))
    return x, y, z

def makeCloud(x, y, z):
    # produce a PLY then read with Open3D (matches your pipeline)
    points = vstack((x, y, z)).transpose()
    cloud = PyntCloud(DataFrame(data=points, columns=["x", "y", "z"]))
    os.makedirs("PointData", exist_ok=True)
    ply_path = os.path.join("PointData", "output.ply")
    cloud.to_file(ply_path)
    return o3d.io.read_point_cloud(ply_path)

def pointsOutOfBox(x, y, box):
    bottomLeft = box[0]; topRight = box[2]
    outX, outY = [], []
    for i in range(len(x)):
        if bottomLeft[0] <= x[i] <= topRight[0] and bottomLeft[1] <= y[i] <= topRight[1]:
            continue
        outX.append(x[i]); outY.append(y[i])
    return outX, outY

def hierarchicalClustering(x, y, thresh=1.5):
    pts = array([[x[i], y[i]] for i in range(len(x))])
    if len(pts) == 0:
        return []
    clustersIndex = hcluster.fclusterdata(pts, thresh, criterion="distance")
    k = int(npmax(clustersIndex))
    clusters = [[] for _ in range(k)]
    for i, idx in enumerate(clustersIndex):
        clusters[idx - 1].append([pts[i,0], pts[i,1]])
    return clusters

def getClustersCenters(clusters):
    centers = []
    for cl in clusters:
        if not cl:
            continue
        sx = sum(p[0] for p in cl); sy = sum(p[1] for p in cl)
        centers.append((float(sx/len(cl)), float(sy/len(cl))))
    return centers

# ---------- Tello helpers ----------

def ensure_airborne(drone):
    """If motors are off, try to take off."""
    try:
        h = drone.get_height()
    except Exception:
        h = 0
    if h < 10:
        try:
            drone.takeoff()
            sleep(1.0)
        except Exception as e:
            print("ensure_airborne: takeoff failed:", e)

def safe_land(drone):
    """Try to land a few times (Tello sometimes returns transient 'error')."""
    for _ in range(3):
        try:
            drone.land()
            return
        except Exception:
            sleep(0.5)

def moveToExit(drone, exits):
    """
    Choose the farthest exit and fly there.
    Uses atan2 for heading. Includes auto-retry if motors were off.
    """
    origin = (0.0, 0.0)
    maxDist, target = -1.0, None
    for e in exits:
        d = distanceBetween2Points(e, origin)
        if d > maxDist:
            maxDist, target = d, e
    if target is None:
        print("No exit target.")
        return

    x, y = target
    heading_deg = (degrees(atan2(y, x)) + 360.0) % 360.0
    print(f"Target exit {target}, heading {heading_deg:.1f}°, distance {maxDist:.3f} [SLAM units]")

    # Rotate to face the exit
    try:
        drone.rotate_clockwise(int(round(heading_deg)))
    except Exception as e:
        print("Rotate failed:", e)
        ensure_airborne(drone)
        try:
            drone.rotate_clockwise(int(round(heading_deg)))
        except Exception as e2:
            print("Rotate retry failed:", e2)

    # Convert SLAM distance to centimeters (tune this!)
    scale_cm = 160.0  # 1 SLAM unit ≈ 1.6 m
    distance_cm = int(round(maxDist * scale_cm))
    print("Move forward (cm):", distance_cm)

    remaining = distance_cm
    while remaining > 0:
        step = min(500, remaining)
        try:
            drone.move_forward(step)
            remaining -= step
        except Exception as e:
            print("Move failed:", e)
            ensure_airborne(drone)
            try:
                drone.move_forward(step)
                remaining -= step
            except Exception as e2:
                print("Move retry failed:", e2)
                break
