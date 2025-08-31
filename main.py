import json
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List

# optional for --slam-only
try:
    from djitellopy import Tello
except Exception:
    Tello = None

# our helpers
from utils import (
    ensure_dirs,
    start_ffmpeg_record,
    stop_ffmpeg_record,
    to_wsl_path,
    wsl_cmd,
    run,
    pretty_json,
    wsl_to_unc,
    read_xyz_csv,
    pointsOutOfBox,
    hierarchicalClustering,
    getClustersCenters,
)

# exit/pcd tools
from ExitFinding import getAverageRectangle
from PointCloudCleaning import removeStatisticalOutlier
from plot import plot2DWithBoxAndCenters

THIS_DIR = Path(__file__).resolve().parent


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def default_config_path() -> Path:
    for cand in [THIS_DIR / "config.json", Path("config.json").resolve()]:
        if cand.exists():
            return cand
    raise SystemExit("config.json not found")


# --------------------------
# Phase A: record a video
# --------------------------
def phase_record(cfg: dict) -> Path:
    if Tello is None:
        raise SystemExit("djitellopy is required for recording. Install: pip install djitellopy")

    tel = Tello(retry_count=3)
    tel.connect()

    # output
    out_dir = Path(cfg["record"]["video_dir"]).expanduser().resolve()
    ensure_dirs(out_dir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ext = cfg["record"].get("container", "mp4")
    out_path = out_dir / f"tello_{ts}.{ext}"

    # reset stream & start
    tel.streamoff()
    time.sleep(0.5)
    tel.streamon()
    time.sleep(float(cfg["record"].get("warmup_s", 2.0)))

    # start recorder
    rec = start_ffmpeg_record(
        input_url=cfg["record"].get("input_url", "udp://0.0.0.0:11111"),
        output_path=str(out_path),
        ffmpeg_exe=cfg["record"]["ffmpeg_exe"],
        use_ffmpeg=cfg["record"].get("use_ffmpeg", True),
        max_fps=int(cfg["record"].get("max_fps", 30)),
        copy_stream=bool(cfg["record"].get("ffmpeg_copy", True)),
    )

    # flight script: takeoff -> up -> 360° in steps -> optional hover -> land
    try:
        try:
            tel.set_speed(int(cfg["drone"].get("speed_cmps", 10)))
        except Exception:
            pass

        tel.takeoff()

        up_cm = int(cfg["drone"].get("up_after_takeoff_cm", 70))
        if up_cm > 0:
            tel.move_up(up_cm)
            time.sleep(0.5)

        yaw_step = int(cfg["drone"].get("yaw_step_deg", 15))
        pause_s = float(cfg["drone"].get("yaw_pause_s", 0.2))
        steps = max(1, int(360 / max(1, yaw_step)))

        for _ in range(steps):
            tel.rotate_clockwise(yaw_step)
            time.sleep(pause_s)

        extra = float(cfg["record"].get("extra_hover_s", 0.0))
        if extra > 0:
            time.sleep(extra)

        tel.land()

    finally:
        stop_ffmpeg_record(rec, grace_s=float(cfg["record"].get("grace_s", 2.0)))
        try:
            tel.streamoff()
        except Exception:
            pass
        try:
            tel.end()
        except Exception:
            pass

    print(f"[REC] saved video -> {out_path}")
    return out_path


# --------------------------
# Phase B: ORB-SLAM3 on file (mono_input)
# --------------------------
def phase_slam(cfg: dict, video_path: Path) -> Tuple[List[str], str, Path]:
    """
    Run ORB-SLAM3 mono_input on a video.
    Returns:
      (plot_paths, log_unc_dir, artifacts_dir)
    """
    orb = cfg['wsl_orbslam']

    # Inputs & paths
    video_wsl = to_wsl_path(str(video_path))
    orb_root  = orb['root'].rstrip('/') # e.g. /home/USER/dev/ORB_SLAM3
    exe       = orb.get('mono_input_exe', 'Examples/Monocular/mono_input')
    voc       = orb.get('vocabulary',   'Vocabulary/ORBvoc.txt')
    yaml      = orb.get('settings',     'Examples/Monocular/TUM1.yaml')
    log_dir   = orb.get('log_dir',      'log').strip('/')
    log_wsl   = f"{orb_root}/{log_dir}".rstrip('/')

    # Where to save plots on Windows
    outputs_base = Path(orb.get('outputs_dir', 'artifacts')).resolve()
    plot_dir = outputs_base / video_path.stem # artifacts/tello_YYYYMMDD_HHMMSS
    ensure_dirs(plot_dir)

    # 1) Run ORB-SLAM3
    print(f"[WSL] launching ORB-SLAM3 mono_input on {video_wsl}")
    slam_cmd = f"cd '{orb_root}' && ./{exe} '{voc}' '{yaml}' '{video_wsl}'"
    rc = wsl_cmd(slam_cmd, passthrough=True)

    # 2) UNC path to WSL logs
    distro = orb.get('wsl_distro', 'Ubuntu-20.04')
    log_unc = wsl_to_unc(log_wsl, distro=distro)
    print(f"[SLAM] logs UNC: {log_unc}")

    # 3) Quick static plots (topdown + 3D) into plot_dir
    plot_paths = save_quick_plots_from_logs(source_dir=log_unc, out_dir=plot_dir)
    for p in plot_paths:
        print(f"[PLOT] saved -> {p}")

    # 4) Error policy
    if rc != 0 and orb.get('ignore_errors', True):
        print(f"[SLAM] non-zero exit ({rc}) ignored; plots saved to: {plot_dir}")
    elif rc != 0:
        raise SystemExit(f"ORB-SLAM3 exited with code {rc}")
    else:
        print(f"[SLAM] plots saved to: {plot_dir}")

    return [str(p) for p in plot_paths], log_unc, plot_dir


# --------------------------
# Exit analysis
# --------------------------
def analyze_exit_from_logs(cfg: dict, log_unc_dir: str, artifacts_dir: Path) -> Tuple[Optional[float], Optional[int]]:
    """
    1) Read pointData.csv (X,Y,Z)
    2) Clean with Statistical Outlier removal
    3) Compute room rectangle on (X, Z)
    4) Cluster out-of-box points -> centers = candidate exits
    5) Return heading (deg CW) and forward distance (cm) for the furthest exit
    Also saves 'exit_detection.png' plot in artifacts_dir.
    """
    # defaults
    ex = cfg.get("exit_algo", {})
    voxel_size  = float(ex.get("voxel_size", 0.02))
    nb_neighbors= int(ex.get("nb_neighbors", 20))
    std_ratio   = float(ex.get("std_ratio", 2.0))
    thresh      = float(ex.get("thresh",    1.5))
    cm_per_unit = float(ex.get("cm_per_unit", 160.0))

    pts_file = Path(log_unc_dir) / "pointData.csv"
    x, y, z = read_xyz_csv(pts_file)
    if not x:
        print(f"[EXIT] no points in {pts_file}; cannot compute exit.")
        return None, None

    # Open3D clean
    import open3d as o3d
    import numpy as np

    P = np.vstack([x, y, z]).T.astype(np.float64)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(P)
    inlier, _ = removeStatisticalOutlier(pcd, voxel_size=voxel_size,
                                         nb_neighbors=nb_neighbors, std_ratio=std_ratio)

    # arrays again
    A = np.asarray(inlier.points)
    if A.shape[0] < 10:
        print("[EXIT] too few inlier points.")
        return None, None
    inX = A[:, 0].tolist()
    inZ = A[:, 2].tolist()

    # room rectangle & out-of-box clustering
    room_box = getAverageRectangle(inX, inZ)
    xOut, zOut = pointsOutOfBox(inX, inZ, room_box)
    clusters = hierarchicalClustering(xOut, zOut, thresh=thresh)
    centers  = getClustersCenters(clusters)

    # save a quick visualization
    out_img = artifacts_dir / "exit_detection.png"
    try:
        plot2DWithBoxAndCenters(inX, inZ, room_box, centers, save_path=str(out_img),
                                title="Exit candidates (top-down X vs Z)")
        print(f"[EXIT] plot saved -> {out_img}")
    except Exception as e:
        print(f"[EXIT] plot failed: {e}")

    if not centers:
        print("[EXIT] no exit clusters found.")
        return None, None

    # choose furthest exit from (0,0) in the (X,Z) plane and compute angle/distance
    angle_deg, fwd_cm = compute_exit_plan_from_centers(centers, cm_per_unit=cm_per_unit)
    print(f"[EXIT] heading={angle_deg:.1f}°  forward={fwd_cm} cm")
    return angle_deg, fwd_cm


def compute_exit_plan_from_centers(centers: List[Tuple[float, float]], cm_per_unit: float = 160.0) -> Tuple[float, int]:
    """
    angle/distance logic: pick furthest center and
    compute clockwise heading (deg) + forward distance (cm).
    """
    from math import degrees, tan

    # pick furthest from origin (0,0)
    max_d = float("-inf")
    fur = (0.0, 0.0)
    for (x, z) in centers:
        d = (x * x + z * z) ** 0.5
        if d > max_d:
            max_d = d
            fur = (x, z)

    x, y = fur[0], fur[1]  # (x,y) even though it's (X,Z)
    # angle base:
    base = 90 - int(degrees(tan(float(abs(y) / max(1e-9, abs(x))))))
    if x > 0 > y:
        base += 90
    elif x < 0 and y < 0:
        base += 180
    elif x < 0 < y:
        base += 270
    angle = float(base % 360)

    # distance scale: 1 SLAM unit =approx. 160 cm
    distance_cm = int(max_d * cm_per_unit)
    return angle, distance_cm


# --------------------------
# Phase C: exit flight
# --------------------------
def phase_exit_flight(cfg: dict,
                      heading_override: Optional[float] = None,
                      forward_override: Optional[int] = None) -> None:
    if Tello is None:
        raise SystemExit("djitellopy is required for flight. Install: pip install djitellopy")

    tel = Tello(retry_count=3)
    tel.connect()
    try:
        tel.takeoff()
        up_cm = int(cfg["exit_mission"].get("up_after_takeoff_cm", 50))
        if up_cm > 0:
            tel.move_up(up_cm)
            time.sleep(0.5)

        head = heading_override if heading_override is not None else float(cfg["exit_mission"].get("exit_heading_deg", 0.0))
        if head >= 0:
            tel.rotate_clockwise(int(head))
        else:
            tel.rotate_counter_clockwise(int(-head))
        time.sleep(0.5)

        dist_cm = forward_override if forward_override is not None else int(cfg["exit_mission"].get("forward_cm", 300))
        step = int(cfg["exit_mission"].get("segment_cm", 50))
        pause = float(cfg["exit_mission"].get("segment_pause_s", 0.3))

        traveled = 0
        while traveled < dist_cm:
            tel.move_forward(min(step, dist_cm - traveled))
            traveled += step
            time.sleep(pause)

        tel.land()
    finally:
        try:
            tel.end()
        except Exception:
            pass


# --- Quick plots from WSL log folder -> saved under artifacts/<video-stem>/ ---
def save_quick_plots_from_logs(source_dir, out_dir) -> list:
    """
    Reads pointData.csv and KeyFrameTrajectory.txt from source_dir (UNC path to WSL log)
    and writes two static plots into out_dir:
      - point_cloud_topdown.png (X vs Z with route)
      - point_cloud_3d.png (3D scatter + route)
    Returns list of saved file paths.
    """
    import csv
    import matplotlib
    matplotlib.use("Agg")  # headless
    from matplotlib import pyplot as plt
    import numpy as np
    from pathlib import Path as _Path

    src = _Path(source_dir)
    dst = _Path(out_dir)
    dst.mkdir(parents=True, exist_ok=True)

    # Read points
    pts_file = src / "pointData.csv"
    if not pts_file.exists():
        print(f"[PLOT] {pts_file} not found. Skipping plots.")
        return []
    pts = []
    with pts_file.open("r", encoding="utf-8", errors="ignore") as f:
        for row in csv.reader(f):
            if not row or row[0].strip().startswith("#"):
                continue
            try:
                x, y, z = float(row[0]), float(row[1]), float(row[2])
                pts.append((x, y, z))
            except Exception:
                continue
    if not pts:
        print(f"[PLOT] No valid points parsed from {pts_file}.")
        return []

    # Read route (keyframes) if present
    route = []
    kf_file = src / "KeyFrameTrajectory.txt"
    if kf_file.exists():
        with kf_file.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if not line.strip() or line.startswith("#"):
                    continue
                cols = line.strip().split()
                if len(cols) >= 8:
                    try:
                        tx, ty, tz = float(cols[1]), float(cols[2]), float(cols[3])
                        route.append((tx, ty, tz))
                    except Exception:
                        pass

    P = np.array(pts)

    # 2D top-down (X vs Z)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.scatter(P[:, 0], P[:, 2], s=1)
    if route:
        R = np.array(route)
        ax.plot(R[:, 0], R[:, 2], linewidth=1)
    ax.set_xlabel("X"); ax.set_ylabel("Z"); ax.set_aspect("equal", adjustable="box")
    ax.set_title("SLAM point cloud (top-down)")
    topdown_path = str(dst / "point_cloud_topdown.png")
    plt.tight_layout(); plt.savefig(topdown_path, dpi=160); plt.close(fig)

    # 3D scatter
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=1)
    if route:
        R = np.array(route)
        ax.plot(R[:, 0], R[:, 1], R[:, 2], linewidth=1)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title("SLAM point cloud (3D)")
    cloud3d_path = str(dst / "point_cloud_3d.png")
    plt.tight_layout(); plt.savefig(cloud3d_path, dpi=160); plt.close(fig)

    return [topdown_path, cloud3d_path]


def main():
    ap = argparse.ArgumentParser(
        description="Record Tello video -> run ORB-SLAM3 on it in WSL -> compute exit -> fly."
    )
    ap.add_argument("--config", default=str(default_config_path()), help="Path to config.json")
    ap.add_argument("--record-only", action="store_true", help="Only record (no SLAM, no mission)")
    ap.add_argument("--slam-only", action="store_true", help="Run SLAM on latest recording or --video; also compute exit")
    ap.add_argument("--mission-only", action="store_true", help="Only run the exit mission")
    ap.add_argument("--video", default="", help="Video file for --slam-only")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    print("[CFG]\n" + pretty_json(cfg))

    if args.slam_only:
        if args.video:
            video_path = Path(args.video).expanduser().resolve()
        else:
            rec_dir = Path(cfg["record"]["video_dir"]).expanduser().resolve()
            vids = (
                sorted(rec_dir.glob("tello_*.mp4"))
                + sorted(rec_dir.glob("tello_*.mkv"))
                + sorted(rec_dir.glob("tello_*.avi"))
            )
            if not vids:
                raise SystemExit("No recordings found. Run --record-only first, or pass --video.")
            video_path = vids[-1]
        _, log_unc, art_dir = phase_slam(cfg, video_path)
        head, dist = analyze_exit_from_logs(cfg, log_unc, art_dir)
        print(f"[RESULT] exit_heading_deg={head}  forward_cm={dist}")
        return

    if args.mission_only:
        # uses config's exit_mission values
        phase_exit_flight(cfg)
        return

    # full pipeline
    video_path = phase_record(cfg)
    _, log_unc, art_dir = phase_slam(cfg, video_path)
    head, dist = analyze_exit_from_logs(cfg, log_unc, art_dir)
    phase_exit_flight(cfg, heading_override=head, forward_override=dist)


if __name__ == "__main__":
    main()
