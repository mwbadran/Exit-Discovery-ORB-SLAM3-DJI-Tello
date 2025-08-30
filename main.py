import json
import time
import argparse
from datetime import datetime
from pathlib import Path

# Third-party (optional for --slam-only)
try:
    from djitellopy import Tello
except Exception:
    Tello = None

from utils import (
    ensure_dirs,
    start_ffmpeg_record,
    stop_ffmpeg_record,
    to_wsl_path,
    wsl_cmd,
    run,
    pretty_json,
)

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

    # flight script: takeoff → up → 360° in steps → optional hover → land
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
def phase_slam(cfg: dict, video_path: Path) -> None:
    """
    Run ORB-SLAM3 mono_input on a video and export logs to artifacts/<video-stem>/.
    - Ignores non-zero ORB-SLAM3 exit if 'ignore_errors' is True.
    - Lists what WSL wrote to its log dir, previews first lines, then copies to Windows.
    - Glob quoting fixed so *.csv/*.txt/*.pcd/*.bin actually match.
    """
    orb = cfg['wsl_orbslam']

    # Inputs & paths
    video_wsl = to_wsl_path(str(video_path))
    orb_root  = orb['root'].rstrip('/')                         # e.g. /home/USER/dev/ORB_SLAM3
    exe       = orb.get('mono_input_exe', 'Examples/Monocular/mono_input')
    voc       = orb.get('vocabulary',   'Vocabulary/ORBvoc.txt')
    yaml      = orb.get('settings',     'Examples/Monocular/TUM1.yaml')
    log_dir   = orb.get('log_dir',      'log').strip('/')
    log_wsl   = f"{orb_root}/{log_dir}".rstrip('/')

    # Windows export
    outputs_base = Path(orb.get('outputs_dir', 'artifacts')).resolve()
    export_dir   = outputs_base / video_path.stem               # .../artifacts/tello_YYYYMMDD_HHMMSS
    ensure_dirs(export_dir)
    dest_wsl     = f"{to_wsl_path(str(outputs_base)).rstrip('/')}/{video_path.stem}"

    # 1) Run ORB-SLAM3
    print(f"[WSL] launching ORB-SLAM3 mono_input on {video_wsl}")
    slam_cmd = f"cd '{orb_root}' && ./{exe} '{voc}' '{yaml}' '{video_wsl}'"
    rc = wsl_cmd(slam_cmd, passthrough=True)

    # 2) List WSL log dir (debug) — NOTE: no quotes around globs
    print(f"[RUN] wsl bash -lc \"echo '[WSL] listing log dir: {log_wsl}'; "
          f"if [ -d '{log_wsl}' ]; then "
          f"  ls -lah --time-style=long-iso '{log_wsl}' || true; "
          f"  echo '[WSL] preview first lines (if any):'; "
          f"  shopt -s nullglob; "
          f"  for f in {log_wsl}/*.csv {log_wsl}/*.txt; do "
          f"    echo '---' \"$f\"; head -n 5 \"$f\"; "
          f"  done; "
          f"else echo '[WSL] log dir not found'; fi\"")
    list_cmd = (
        f"echo '[WSL] listing log dir: {log_wsl}'; "
        f"if [ -d '{log_wsl}' ]; then "
        f"  ls -lah --time-style=long-iso '{log_wsl}' || true; "
        f"  echo '[WSL] preview first lines (if any):'; "
        f"  shopt -s nullglob; "
        f"  for f in {log_wsl}/*.csv {log_wsl}/*.txt; do "
        f"    echo '---' \"$f\"; head -n 5 \"$f\"; "
        f"  done; "
        f"else echo '[WSL] log dir not found'; fi"
    )
    wsl_cmd(list_cmd, passthrough=True)

    # 3) Robust copy from WSL -> Windows (no quoted globs, no unary-operator error)
    print(
        f"[RUN] wsl bash -lc \""
        f"shopt -s nullglob; "
        f"mkdir -p '{dest_wsl}'; "
        f"files=({log_wsl}/*.txt {log_wsl}/*.csv {log_wsl}/*.pcd {log_wsl}/*.bin); "
        f"if [ ${{#files[@]}} -gt 0 ]; then "
        f"  cp -f \"${{files[@]}}\" '{dest_wsl}/'; "
        f"  echo '[WSL] exported files to {dest_wsl}:'; ls -1 '{dest_wsl}' 2>/dev/null || true; "
        f"else "
        f"  echo '[WSL] wildcard copy found nothing; trying explicit known files...'; "
        f"  [ -f '{log_wsl}/KeyFrameTrajectory.txt' ] && cp -f '{log_wsl}/KeyFrameTrajectory.txt' '{dest_wsl}/' || true; "
        f"  [ -f '{log_wsl}/pointData.csv' ] && cp -f '{log_wsl}/pointData.csv' '{dest_wsl}/' || true; "
        f"  echo '[WSL] after explicit copy:'; ls -1 '{dest_wsl}' 2>/dev/null || true; "
        f"  if [ ! -e '{dest_wsl}/KeyFrameTrajectory.txt' ] && [ ! -e '{dest_wsl}/pointData.csv' ]; then "
        f"    echo '[WSL] still nothing; archiving the whole log dir...'; "
        f"    if [ -d '{log_wsl}' ]; then "
        f"      tar -C '{log_wsl}' -cf '{dest_wsl}/log_dump.tar' . 2>/dev/null || true; "
        f"      echo '[WSL] archive contents:'; tar -tf '{dest_wsl}/log_dump.tar' 2>/dev/null | head -n 50 || true; "
        f"    fi; "
        f"  fi; "
        f"fi\""
    )
    copy_cmd = (
        f"shopt -s nullglob; "
        f"mkdir -p '{dest_wsl}'; "
        f"files=({log_wsl}/*.txt {log_wsl}/*.csv {log_wsl}/*.pcd {log_wsl}/*.bin); "
        f"if [ ${{#files[@]}} -gt 0 ]; then "
        f"  cp -f \"${{files[@]}}\" '{dest_wsl}/'; "
        f"  echo '[WSL] exported files to {dest_wsl}:'; ls -1 '{dest_wsl}' 2>/dev/null || true; "
        f"else "
        f"  echo '[WSL] wildcard copy found nothing; trying explicit known files...'; "
        f"  [ -f '{log_wsl}/KeyFrameTrajectory.txt' ] && cp -f '{log_wsl}/KeyFrameTrajectory.txt' '{dest_wsl}/' || true; "
        f"  [ -f '{log_wsl}/pointData.csv' ] && cp -f '{log_wsl}/pointData.csv' '{dest_wsl}/' || true; "
        f"  echo '[WSL] after explicit copy:'; ls -1 '{dest_wsl}' 2>/dev/null || true; "
        f"  if [ ! -e '{dest_wsl}/KeyFrameTrajectory.txt' ] && [ ! -e '{dest_wsl}/pointData.csv' ]; then "
        f"    echo '[WSL] still nothing; archiving the whole log dir...'; "
        f"    if [ -d '{log_wsl}' ]; then "
        f"      tar -C '{log_wsl}' -cf '{dest_wsl}/log_dump.tar' . 2>/dev/null || true; "
        f"      echo '[WSL] archive contents:'; tar -tf '{dest_wsl}/log_dump.tar' 2>/dev/null | head -n 50 || true; "
        f"    fi; "
        f"  fi; "
        f"fi"
    )
    wsl_cmd(copy_cmd, passthrough=True)

    # 4) Don’t hard-fail on ORB-SLAM3 crashes unless you ask for it
    if rc != 0 and orb.get('ignore_errors', True):
        print(f"[SLAM] non-zero exit ({rc}) ignored; artifacts saved to: {export_dir}")
    elif rc != 0:
        raise SystemExit(f"ORB-SLAM3 exited with code {rc}")
    else:
        print(f"[SLAM] artifacts saved to: {export_dir}")



# --------------------------
# Phase C: simple exit flight
# --------------------------
def phase_exit_flight(cfg: dict) -> None:
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

        head = float(cfg["exit_mission"].get("exit_heading_deg", 0.0))
        if head >= 0:
            tel.rotate_clockwise(int(head))
        else:
            tel.rotate_counter_clockwise(int(-head))
        time.sleep(0.5)

        dist_cm = int(cfg["exit_mission"].get("forward_cm", 200))
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


def main():
    ap = argparse.ArgumentParser(
        description="Record Tello video → run ORB-SLAM3 on it in WSL → fly to exit."
    )
    ap.add_argument("--config", default=str(default_config_path()), help="Path to config.json")
    ap.add_argument("--record-only", action="store_true", help="Only record (no SLAM, no mission)")
    ap.add_argument("--slam-only", action="store_true", help="Run SLAM on latest recording or --video")
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
        phase_slam(cfg, video_path)
        return

    if args.mission_only:
        phase_exit_flight(cfg)
        return

    # full pipeline
    video_path = phase_record(cfg)
    phase_slam(cfg, video_path)
    phase_exit_flight(cfg)


if __name__ == "__main__":
    main()
