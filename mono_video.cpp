/**
 * ORB-SLAM3 mono_video: webcam / video file / TELLO UDP stream (Windows-friendly)
 * Exports tracked MapPoints to log/pointData.csv and drops "ready" flags so Python
 * can safely wait before flying.
 */

#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <sstream>
#include <unordered_set>
#include <string>
#include <iomanip>
#include <vector>
#include <cctype>   // std::isdigit
#include <cmath>    // llround

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/imgcodecs.hpp>

#include <System.h>
#include <MapPoint.h>

#ifdef _WIN32
#include <windows.h>
static inline void usleep(unsigned long usec) { Sleep((DWORD)(usec / 1000)); }
#endif

using namespace std;

static inline bool is_numeric(const std::string& s) {
    if (s.empty()) return false;
    return std::all_of(s.begin(), s.end(), [](unsigned char c) { return std::isdigit(c) != 0; });
}

static inline string qkey(double x, double y, double z, double q = 1000.0) {
    long long ix = (long long)llround(x * q);
    long long iy = (long long)llround(y * q);
    long long iz = (long long)llround(z * q);
    return to_string(ix) + "," + to_string(iy) + "," + to_string(iz);
}

static inline void ensure_log_dir() {
#ifdef _WIN32
    system("if not exist log mkdir log");
#else
    system("mkdir -p log");
#endif
}

static inline void write_flag(const std::string& name, const std::string& payload = "1") {
    ensure_log_dir();
    std::ofstream f(std::string("log/") + name, ios::out | ios::trunc);
    if (f.is_open()) { f << payload; f.close(); }
}

int mono_video(int argc, char** argv)
{
    if (argc != 4)
    {
        cerr << endl << "Usage: ./slam.exe mono_video path_to_vocabulary path_to_settings <0|webcam_index|video_path|TELLO>" << endl;
        cerr << "Examples:" << endl;
        cerr << "  webcam: ./slam.exe mono_video ORBvoc.txt TUM1.yaml 0" << endl;
        cerr << "  video : ./slam.exe mono_video ORBvoc.txt TUM1.yaml C:\\video.mp4" << endl;
        cerr << "  TELLO : ./slam.exe mono_video ORBvoc.txt TUM1.yaml TELLO" << endl;
        return 1;
    }

    const string vocabFile = string(argv[1]);
    const string settingsFile = string(argv[2]);
    const string inputArg = string(argv[3]);

    cout << endl << "-------" << endl;
    cout << "ORB-SLAM3 mono_video " << vocabFile << " " << settingsFile << " " << inputArg << endl;

    cv::VideoCapture cap;
    bool opened = false;

    // 0) Make sure log/ exists (for flags)
    ensure_log_dir();

    // 1) TELLO UDP (use extra ffmpeg opts to lower latency / reduce stutter)
    if (!opened && (inputArg == "TELLO" || inputArg == "tello"))
    {
        // Requires OpenCV built with FFMPEG
        const string url =
            "udp://@0.0.0.0:11111"
            "?overrun_nonfatal=1"
            "&fifo_size=50000000"
            "&buffer_size=10485760"
            "&max_delay=0"
            "&flags=low_delay"
            "&probesize=32000"
            "&analyzeduration=0";
        cout << "Trying TELLO stream via FFMPEG: " << url << endl;
        opened = cap.open(url, cv::CAP_FFMPEG);
        cout << (opened ? "Opened TELLO stream" : "Failed to open TELLO stream") << endl;
    }

    // 2) Webcam (numeric index; try a couple Windows backends)
    if (!opened && is_numeric(inputArg)) {
        const int camIndex = stoi(inputArg);
        struct BackendTry { int id; const char* name; };
        BackendTry backends[] = {
        #ifdef CV_CAP_DSHOW
            {cv::CAP_DSHOW,  "CAP_DSHOW"},
        #endif
        #ifdef CV_CAP_MSMF
            {cv::CAP_MSMF,   "CAP_MSMF"},
        #endif
            {cv::CAP_ANY,    "CAP_ANY"}
        };
        for (const auto& b : backends) {
            cout << "Trying webcam index " << camIndex << " with " << b.name << "..." << endl;
            if (cap.open(camIndex, b.id)) { cout << "Opened webcam index " << camIndex << " with " << b.name << endl; opened = true; break; }
            cout << "Failed to open webcam with " << b.name << endl;
        }
    }

    // 3) Video file (try several backends)
    if (!opened) {
        struct BackendTry { int id; const char* name; };
        BackendTry backends[] = {
        #ifdef CV_CAP_FFMPEG
            {cv::CAP_FFMPEG, "CAP_FFMPEG"},
        #endif
        #ifdef CV_CAP_MSMF
            {cv::CAP_MSMF,   "CAP_MSMF"},
        #endif
        #ifdef CV_CAP_DSHOW
            {cv::CAP_DSHOW,  "CAP_DSHOW"},
        #endif
            {cv::CAP_ANY,    "CAP_ANY"}
        };
        for (const auto& b : backends) {
            try {
                cout << "Trying file with backend: " << b.name << endl;
                if (cap.open(inputArg, b.id)) { cout << "Opened video with backend: " << b.name << endl; opened = true; break; }
                cout << "Failed to open with backend: " << b.name << endl;
            }
            catch (const cv::Exception& e) {
                cout << "Exception while opening with " << b.name << ": " << e.what() << endl;
            }
        }
    }

    if (!opened) {
        cerr << "Failed to open input: " << inputArg << endl;
        cerr << "Hints: for webcam use a numeric index; for video ensure OpenCV ffmpeg DLLs are available; "
            "for TELLO ensure the drone is streaming (streamon)." << endl;
        return 1;
    }

    // Drop a flag immediately after we have a live capture
    write_flag("VIDEO_READY.flag", "capture_opened");

    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0.0 || fps > 120.0) fps = 30.0; // e.g., webcams report 0
    double T = 1.0 / fps;

    cout << "Input: " << inputArg << endl;
    cout << "FPS (reported): " << fps << "   frame interval T = " << T << " s" << endl << endl;

    // Create SLAM system (loads vocabulary)
    ORB_SLAM3::System SLAM(vocabFile, settingsFile, ORB_SLAM3::System::MONOCULAR, true);

    // Now the vocabulary is loaded and tracking thread is up: announce ready
    write_flag("SLAM_READY.flag", "ready");

    // Collect tracked MapPoints across frames (dedup by ~1mm quantization)
    std::unordered_set<std::string> seen;
    std::vector<cv::Vec3d> points_accum; points_accum.reserve(200000);

    cv::Mat im;
    std::vector<float> vTimesTrack;

    while (true)
    {
        if (!cap.read(im)) break;

        double tframe;
        if (inputArg == "TELLO" || inputArg == "tello" || is_numeric(inputArg)) {
            tframe = chrono::duration_cast<chrono::duration<double>>(chrono::steady_clock::now().time_since_epoch()).count();
        }
        else {
            tframe = cap.get(cv::CAP_PROP_POS_MSEC) / 1000.0;
        }

        auto t1 = chrono::steady_clock::now();
        SLAM.TrackMonocular(im, tframe);
        auto t2 = chrono::steady_clock::now();

        double ttrack = chrono::duration_cast<chrono::duration<double>>(t2 - t1).count();
        vTimesTrack.push_back((float)ttrack);

        // Collect currently tracked map points
        const std::vector<ORB_SLAM3::MapPoint*> vmp = SLAM.GetTrackedMapPoints();
        for (auto* mp : vmp) {
            if (!mp) continue;
            cv::Mat X = mp->GetWorldPos();
            if (X.empty()) continue;

            double x, y, z;
            if (X.type() == CV_32F) {
                x = (double)X.at<float>(0);
                y = (double)X.at<float>(1);
                z = (double)X.at<float>(2);
            }
            else {
                x = X.at<double>(0);
                y = X.at<double>(1);
                z = X.at<double>(2);
            }

            string key = qkey(x, y, z);
            if (seen.insert(key).second) points_accum.emplace_back(x, y, z);
        }

        cv::imshow("Frame", im);
        int key = cv::waitKey(1);
        if (key == 27) break; // ESC

        if (ttrack < T) usleep((unsigned long)((T - ttrack) * 1e6));
    }

    SLAM.Shutdown();

#ifdef _WIN32
    system("if not exist log mkdir log");
#else
    system("mkdir -p log");
#endif

    // Save trajectories (KeyFrameTrajectory works in mono; CameraTrajectory prints a warning)
    SLAM.SaveKeyFrameTrajectoryTUM("log/KeyFrameTrajectory.txt");
    SLAM.SaveTrajectoryTUM("log/CameraTrajectory.txt");

    // Save accumulated points (CSV)
    {
        std::ofstream out("log/pointData.csv");
        for (const auto& v : points_accum) {
            out << std::fixed << std::setprecision(6)
                << v[0] << "," << v[1] << "," << v[2] << "\n";
        }
        out.close();
        cout << "Saved " << points_accum.size() << " unique tracked points to log/pointData.csv" << endl;
        if (points_accum.empty()) {
            cout << "WARNING: 0 points saved. Check calibration, scene texture, exposure, and motion." << endl;
        }
    }

    cout << "Finished. Outputs in log/ folder." << endl;
    return 0;
}
