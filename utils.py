import json
import subprocess
import time
from pathlib import Path
from typing import Union

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
        # re-encode if you prefer increased compatibility
        args += ['-c:v', 'libx264', '-preset', 'veryfast', '-r', str(max_fps)]
    args += [output_path]

    print("[REC] starting ffmpeg:\n  " + ' '.join(args))
    # IMPORTANT: give ffmpeg a stdin pipe so we can send 'q' to finalize containers (MP4/MKV)
    # Send ffmpeg output to DEVNULL so pipes don't fill and block.
    proc = subprocess.Popen(
        args,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True
    )
    time.sleep(0.8)  # small warmup so the process is ready
    return proc

def stop_ffmpeg_record(proc, grace_s: float = 2.0):
    if proc is None:
        return
    print("[REC] stopping ffmpeg (graceful)...")
    try:
        if proc.stdin:
            try:
                proc.stdin.write('q')   # ask ffmpeg to quit cleanly (writes MP4/MKV headers)
                proc.stdin.flush()
            except Exception:
                pass
        # give ffmpeg enough time to finalize
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
        # last-resort safety
        try:
            proc.kill()
        except Exception:
            pass
