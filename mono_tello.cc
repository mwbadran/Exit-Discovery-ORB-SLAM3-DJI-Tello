// Examples/Monocular/mono_tello.cc
// Live Tello or MJPEG URL, robust exit & saving for monocular.

#include <opencv2/opencv.hpp>
#include <System.h>
#include <MapPoint.h>

#include <atomic>
#include <csignal>
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>

#ifdef _WIN32
  #include <direct.h>
  #define MKDIR(dir) _mkdir(dir)
  #define SEP "\\"
#else
  #include <sys/stat.h>
  #include <sys/types.h>
  #include <unistd.h>
  #include <limits.h>
  #define MKDIR(dir) mkdir(dir, 0755)
  #define SEP "/"
#endif

static std::atomic<bool> g_run{true};
static void on_sigint (int){ g_run = false; }
static void on_sigterm(int){ g_run = false; }

static inline double rel_time() {
  static const auto t0 = std::chrono::steady_clock::now();
  return std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now()-t0).count();
}

static std::string get_cwd() {
#ifdef _WIN32
  char buf[_MAX_PATH]; _getcwd(buf, _MAX_PATH); return std::string(buf);
#else
  char buf[PATH_MAX]; if (getcwd(buf, sizeof(buf))) return std::string(buf); return ".";
#endif
}

int main(int argc, char** argv)
{
  if (argc != 4) {
    std::cerr << "usage:\n"
              << "  ./mono_tello <path_to_vocabulary> <path_to_settings> <input>\n"
              << "  input: TELLO | udp://... | http://...\n";
    return 1;
  }

  std::signal(SIGINT,  on_sigint);
  std::signal(SIGTERM, on_sigterm);

  const std::string vocabFile    = argv[1];
  const std::string settingsFile = argv[2];
  std::string       input        = argv[3];

  if (input == "TELLO" || input == "tello") {
    input = "udp://@:11111?overrun_nonfatal=1&fifo_size=50000000&timeout=3000000";
  }

  bool useViewer = true;
  if (const char* v = std::getenv("ORB3_VIEWER")) useViewer = (std::string(v) != "0");
  std::cout << "[info] Pangolin viewer: " << (useViewer ? "ENABLED" : "DISABLED") << "\n";

  ORB_SLAM3::System SLAM(vocabFile, settingsFile, ORB_SLAM3::System::MONOCULAR, useViewer);

  cv::VideoCapture cap(input, cv::CAP_FFMPEG);
if(!cap.isOpened() && input.rfind("udp://@:", 0) == 0) {
    // fallback: some builds don’t accept "@:" form; rewrite to 0.0.0.0
    std::string alt = input;
    alt.replace(0, std::string("udp://@:").size(), "udp://0.0.0.0:");
    cap.open(alt, cv::CAP_FFMPEG);
}
if (!cap.isOpened()) {
    std::cerr << "[err] cannot open input: " << input << std::endl;
    return 1;
}
std::cout << "[info] opened input: " << input << std::endl;
std::cout << "[READY] capture_open\n" << std::flush;

  cv::Mat frame, gray;
  bool printed_meta = false;

  auto last_ok = std::chrono::steady_clock::now();
  const auto max_no_frame = std::chrono::seconds(3);
  int fail_streak = 0;

  while (g_run) {
    if (!cap.read(frame) || frame.empty()) {
      ++fail_streak;
      cv::waitKey(1);
      auto now = std::chrono::steady_clock::now();
      if (now - last_ok > max_no_frame || fail_streak > 120) {
        std::cerr << "[warn] input stalled → exiting main loop\n";
        break;
      }
      continue;
    }

    fail_streak = 0;
    last_ok = std::chrono::steady_clock::now();

    if (!printed_meta) {
      std::cout << "[info] first frame: " << frame.cols << "x" << frame.rows
                << "  fps(cap)=" << cap.get(cv::CAP_PROP_FPS) << "\n";
      printed_meta = true;
    }

    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    SLAM.TrackMonocular(gray, rel_time());

    cv::imshow("input", frame);
    int k = cv::waitKey(1) & 0xFF;
    if (k == 'q' || k == 27) { g_run = false; break; }
  }

  cap.release();
  cv::destroyAllWindows();

  // prevent a second Ctrl-C interrupting saving:
  std::signal(SIGINT,  SIG_IGN);
  std::signal(SIGTERM, SIG_IGN);

  // orderly stop threads
  SLAM.Shutdown();

  const std::string cwd    = get_cwd();
  const std::string logDir = cwd + SEP + "log";
  MKDIR(logDir.c_str());

  const std::string kfTrajPath  = logDir + SEP + "KeyFrameTrajectory.txt";
  const std::string csvPath     = logDir + SEP + "pointData.csv";

  std::cout << "[info] writing outputs under: " << logDir << "\n";

  // monocular: only keyframe trajectory
  SLAM.SaveKeyFrameTrajectoryTUM(kfTrajPath);

  // dump map points
  std::ofstream csv(csvPath, std::ios::trunc);
  if (!csv) {
    std::cerr << "[err] unable to open " << csvPath << " for writing\n";
  } else if (SLAM.mpAtlas) {
    csv << "x,y,z\n";
    auto mps = SLAM.mpAtlas->GetAllMapPoints();
    csv.setf(std::ios::fixed); csv.precision(6);
    size_t n = 0;
    for (auto* mp : mps) if (mp && !mp->isBad()) {
      Eigen::Vector3f p = mp->GetWorldPos();
      csv << p[0] << "," << p[1] << "," << p[2] << "\n";
      ++n;
    }
    csv.flush();
    std::cout << "[info] wrote " << n << " points to " << csvPath << "\n";
  } else {
    std::cerr << "[warn] mpAtlas not available; " << csvPath << " will be empty\n";
  }

  return 0;
}
