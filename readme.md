# Exit Discovery with ORB-SLAM3 on DJI Tello

Real-time room **exit discovery** using **DJI Tello** and **ORB-SLAM3**.

**Authors:** Mohammed Badran, Mahmoud Watted 

**Supervisor:** Prof. Dan Feldman

The workflow is split into three phases:

1. **Record** – fly a 360° scan and save a video.
2. **Map** – run ORB-SLAM3 **`mono_input` on WSL (Ubuntu 20.04)** over the recorded file.
3. **Analyze / Act** – clean the point cloud, detect exit candidates, and (optionally) fly a short exit mission toward the chosen exit.

**Demo video:** [https://youtu.be/1HpYW8Gs0oU](https://youtu.be/1HpYW8Gs0oU)

> This project was completed as part of **Prof. Dan Feldman’s “Project in Real-Time Systems”** (University of Haifa, RBD Lab, Etgar Program).

---


## Version Running on ORB-SLAM3-Windows

[See this commit](https://github.com/mwbadran/Exit-Discovery-ORB-SLAM3-DJI-Tello/tree/96bf31afeae864551388d223273d302ab471167a)

If you want a native Windows demo (webcam/video) instead of WSL:
**[https://github.com/mwbadran/ORB\_SLAM3\_Windows](https://github.com/mwbadran/ORB_SLAM3_Windows)**

Older commits in this project use that path; the **current** flow runs mapping on WSL for stability.

---

## Repo layout

```
.
├─ main.py                 # orchestrator (record → slam → analyze/mission)
├─ ExitFinding.py          # 2D (X,Z) projection, room rectangle, clustering, centers
├─ PointCloudCleaning.py   # Open3D outlier removal & voxel downsample
├─ plot.py                 # quick top-down & 3D plots, exit overlay
├─ utils.py                # FFmpeg helpers, WSL path tools, subprocess wrappers
├─ config.json             # edit this before running!
├─ recordings/             # saved videos (Phase A)
├─ artifacts/<run>/        # PNG plots (topdown/3D/exit) from analysis
└─ (WSL) <ORB_SLAM3>/log   # ORB-SLAM3 logs (KeyFrameTrajectory/CameraTrajectory/pointData.csv)
```

---

## Requirements

### Hardware

* DJI **Tello** (standard model)
* Windows 10/11 PC with **WSL2** + **Ubuntu 20.04** distro
* Stable indoor space for scanning (clear line-of-sight)

### Windows (host) software

* **Python 3.9+** (3.10+ recommended)
* **FFmpeg** (Windows build). Example path used by the code:

  * `C:\ffmpeg\bin\ffmpeg.exe`
* **Allow inbound UDP 11111** in Windows Firewall when recording the Tello stream.

### Python packages (Windows host)

Create a venv and install:

```bash
pip install --upgrade pip
pip install numpy scipy open3d matplotlib djitellopy
```

> If `open3d` wheels are unavailable for your Python version, use the closest supported (e.g., Python 3.10/3.11).

### WSL (Ubuntu 20.04) – ORB-SLAM3

* Build **ORB-SLAM3** (UZ-SLAMLab) with `mono_input` enabled (OpenCV + Pangolin).
* Ensure you have the **vocabulary** and a **camera settings YAML**:

  * `Vocabulary/ORBvoc.txt`
  * `Examples/Monocular/telloCal.yaml` (your Tello calibration)

You may also keep a Windows build for demos via our forked project:

* **Windows fork (for demos)**: [https://github.com/mwbadran/ORB\_SLAM3\_Windows](https://github.com/mwbadran/ORB_SLAM3_Windows)
  (Older commits in this repo use that path; current flow runs WSL for mapping.)

---

## Configure

Open **`config.json`** and set:

```json
{
  "drone": {
    "speed_cmps": 20,
    "up_after_takeoff_cm": 70,
    "yaw_step_deg": 10,
    "yaw_pause_s": 0.5
  },
  "record": {
    "video_dir": "recordings",
    "container": "mp4",
    "use_ffmpeg": true,
    "ffmpeg_exe": "C:/ffmpeg/bin/ffmpeg.exe",
    "ffmpeg_copy": true,
    "max_fps": 30,
    "input_url": "udp://0.0.0.0:11111",   // <- ensure this is exactly 11111
    "warmup_s": 2.0,
    "extra_hover_s": 0.0,
    "grace_s": 2.0
  },
  "wsl_orbslam": {
    "root": "/home/<user>/dev/ORB_SLAM3",
    "mono_input_exe": "Examples/Monocular/mono_input",
    "vocabulary": "Vocabulary/ORBvoc.txt",
    "settings": "Examples/Monocular/telloCal.yaml",
    "log_dir": "log",
    "outputs_dir": "artifacts",
    "wsl_distro": "Ubuntu-20.04",
    "ignore_errors": true
  },
  "exit_algo": {
    "voxel_size": 0.02,
    "nb_neighbors": 20,
    "std_ratio": 2.0,
    "thresh": 1.5,
    "cm_per_unit": 160.0
  },
  "exit_mission": {
    "up_after_takeoff_cm": 50,
    "exit_heading_deg": 0.0,
    "forward_cm": 300,
    "segment_cm": 50,
    "segment_pause_s": 0.3
  },
  "watchdog": {
    "idle_sec": 6.0,
    "max_trackloss_lines": 80,
    "hard_timeout_sec": 0
  }
}
```

> **Important:** If you previously experimented with `input_url`, make sure it’s **exactly** `udp://0.0.0.0:11111`.

`"wsl_orbslam.root"` must point to the **WSL path** of your ORB-SLAM3 repo (where `mono_input` resides).

---

## How to run

### 0) Prep

* Connect your PC to the **Tello Wi-Fi** (SSID `TELLO-xxxxxx`).
* Place the drone in a safe, open indoor area.

### 1) Phase A - **Record only**

```bash
python main.py --record-only
```

* The drone takes off, performs yaw steps (per config), and lands.
* Video is saved to `recordings/tello_YYYYMMDD_HHMMSS.mp4`.

### 2) Phase B - **SLAM only** (WSL `mono_input` on the recorded file)

Use the **latest** recording:

```bash
python main.py --slam-only
```

Or specify a Windows path:

```bash
python main.py --slam-only --video "C:\Users\W10\Desktop\orbVids\saree3.mp4"
```

The script converts the Windows path with `wslpath`, runs `mono_input` inside WSL, and collects logs.
Artifacts (plots) are written to `artifacts/<video-stem>/`.

### 3) Phase C - **Mission only** (optional short exit flight)

```bash
python main.py --mission-only
```

Uses `exit_mission` values from `config.json`, or the values computed in Phase B if you run the full pipeline.

### 4) Full pipeline (A→B→C)

```bash
python main.py
```

---

## Outputs

* **WSL logs**: `<ORB_SLAM3>/log/`

  * `KeyFrameTrajectory.txt`, `CameraTrajectory.txt`, `pointData.csv`
* **Plots (Windows)**: `artifacts/<video-stem>/`

  * `point_cloud_topdown.png` (X–Z top view)
  * `point_cloud_3d.png`
  * `exit_detection.png` (room rectangle + exit candidates)
* **Recordings**: `recordings/tello_*.mp4`

---

## Troubleshooting

* **No video recorded**

  * Verify you’re connected to the Tello Wi-Fi.
  * Ensure Windows Firewall allows **UDP 11111** inbound for `ffmpeg.exe`.
  * Confirm `record.input_url` is `udp://0.0.0.0:11111`.

* **ORB-SLAM3 can’t find vocabulary/settings**

  * Check `wsl_orbslam.root`, `vocabulary`, and `settings` paths (WSL paths).
  * Test directly in WSL:

    ```bash
    cd /home/<user>/dev/ORB_SLAM3
    ./Examples/Monocular/mono_input Vocabulary/ORBvoc.txt \
      Examples/Monocular/telloCal.yaml /mnt/c/path/to/your.mp4
    ```

* **`open3d` install issues**

  * Try Python 3.10 or 3.11 and the corresponding Open3D wheel.


---


## Acknowledgments

* UZ-SLAMLab **ORB-SLAM3**
* OBS / VirtualCam (used previously for Windows demos)
* **Course:** Project in Real-Time Systems, Prof. Dan Feldman, University of Haifa (RBD Lab, Etgar Program)



