import json
import subprocess
import time
from pathlib import Path
from typing import Union

# numpy / clustering used by exit-finding utils
from numpy import array, asarray, max as npmax, vstack
import scipy.cluster.hierarchy as hcluster

def ensure_dirs(*paths):
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)

# ------------------------------
# Pretty print JSON for logging
# ------------------------------
def pretty_json(obj) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True)

# ------------------------------
# Subprocess helpers
# ------------------------------
def run(cmd, shell=False, passthrough=False, env=None):
    """Run a local command. If passthrough=True, stream stdout/stderr."""
    print(f"[RUN] {cmd}")
    if passthrough:
        return subprocess.call(cmd, shell=shell, env=env)
    res = subprocess.run(cmd, shell=shell, env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(res.stdout)
    return res.returncode

def wsl_cmd(cmd: str, passthrough: bool = True) -> int:
    """Run a command inside WSL."""
    full = f"wsl bash -lc \"{cmd}\""
    return run(full, shell=True, passthrough=passthrough)

def to_wsl_path(path: Union[str, Path]) -> str:
    """Convert a Windows path to a WSL path if needed; otherwise return as-is."""
    s = str(path)
    if s.startswith('/') or s.startswith('udp://') or s.startswith('http://') or s.startswith('https://'):
        return s
    try:
        completed = subprocess.run(['wsl', 'wslpath', '-a', s],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if completed.returncode == 0:
            return completed.stdout.strip()
    except Exception:
        pass
    if len(s) >= 2 and s[1] == ':' and s[0].isalpha():
        drive = s[0].lower()
        p = s[2:].replace('\\', '/').lstrip('/')
        return f"/mnt/{drive}/{p}"
    return s

def wsl_to_unc(wsl_path: Union[str, Path], distro: str = "Ubuntu-20.04") -> str:
    """
    Convert a WSL Linux path like /home/user/dir into a Windows UNC path like:
      \\wsl.localhost\\<distro>\\home\\user\\dir
    Works for /mnt/<drive>/ paths too.
    """
    s = str(wsl_path).strip()
    if not s.startswith('/'):
        # already looks like a Windows path; return as-is
        return s
    parts = s.strip('/').split('/')
    return r"\\wsl.localhost\{distro}\{rest}".format(
        distro=distro,
        rest="\\".join(parts)
    )

# ------------------------------
# Recording helpers
# ------------------------------
def start_ffmpeg_record(input_url: str, output_path: str, ffmpeg_exe: str,
                        use_ffmpeg: bool = True, max_fps: int = 30, copy_stream: bool = True):
    """
    Start ffmpeg to record the Tello UDP stream to a file. Returns a Popen handle or None.
    """
    if not use_ffmpeg:
        print("[REC] use_ffmpeg is False; (OpenCV capture not implemented).")
        return None

    args = [
        ffmpeg_exe,
        '-y',
        '-hide_banner',
        '-loglevel', 'warning',
        '-fflags', 'nobuffer',
        '-flags', 'low_delay',
        '-use_wallclock_as_timestamps', '1',
        '-i', input_url,
    ]
    if copy_stream:
        # stream copy (fast, no re-encode)
        args += ['-c:v', 'copy']
    else:
        # re-encode for compatibility
        args += ['-c:v', 'libx264', '-preset', 'veryfast', '-r', str(max_fps)]
    args += [output_path]

    print("[REC] starting ffmpeg:\n  " + ' '.join(args))
    # give ffmpeg a stdin pipe so we can send 'q' to finalize containers (MP4/MKV)
    proc = subprocess.Popen(
        args,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True
    )
    time.sleep(0.8)  # warmup
    return proc

def stop_ffmpeg_record(proc, grace_s: float = 2.0):
    if proc is None:
        return
    print("[REC] stopping ffmpeg (graceful)...")
    try:
        if proc.stdin:
            try:
                proc.stdin.write('q')   # finalize container cleanly
                proc.stdin.flush()
            except Exception:
                pass
        timeout = max(5.0, float(grace_s) + 5.0)
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        print("[REC] ffmpeg did not exit in time; terminating...")
        try:
            proc.terminate()
            proc.wait(timeout=3.0)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass

# ------------------------------
# Geometry / point utils
# ------------------------------
def distanceBetween2Points(a, b):
    """Euclidean distance between two 2D points (x, y)."""
    ax, ay = a
    bx, by = b
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5

def makeCloud(x, y, z):
    """
    Build an Open3D point cloud from parallel x, y, z arrays/lists.
    """
    import numpy as np
    try:
        import open3d as o3d
    except ImportError as e:
        raise SystemExit(
            "open3d is required for 3D cloud viewing. Install with: pip install open3d"
        ) from e

    pts = vstack([x, y, z]).T.astype(float)  # Open3D expects float64
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts)
    return pc

def read_xyz_csv(file_path: Union[str, Path]):
    """Read a CSV with x,y, z in cols 0,1,2 → return 3 float lists."""
    import csv
    X, Y, Z = [], [], []
    p = Path(file_path)
    if not p.exists():
        print(f"[read_xyz_csv] missing: {p}")
        return X, Y, Z
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        for row in csv.reader(f):
            if not row or row[0].strip().startswith("#"):
                continue
            try:
                X.append(float(row[0])); Y.append(float(row[1])); Z.append(float(row[2]))
            except Exception:
                pass
    return X, Y, Z

def pcdToArrays(pcd):
    pts = asarray(pcd.points)
    x, y, z = [], [], []
    for p in pts:
        x.append(float(p[0])); y.append(float(p[1])); z.append(float(p[2]))
    return x, y, z

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
