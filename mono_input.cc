// mono_input.cc — run ORB-SLAM3 on a video file and export ALL map points
// Outputs in ./log:
//   - KeyFrameTrajectory.txt
//   - pointData_all.csv          (union of ALL maps in the Atlas)
//   - pointData_map000.csv ...   (per-map)
//   - pointData.csv              (legacy alias = same as union)

#include <opencv2/opencv.hpp>

#include <System.h>
#include <Atlas.h>
#include <Map.h>
#include <MapPoint.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <ctime>

static inline double now_sec()
{
    return static_cast<double>(cv::getTickCount()) / cv::getTickFrequency();
}

int main(int argc, char** argv)
{
    if (argc != 4) {
        std::cerr << "Usage: ./mono_input <vocab> <settings> <video>\n";
        return 1;
    }

    const std::string vocab    = argv[1];
    const std::string settings = argv[2];
    const std::string video    = argv[3];

    // Create SLAM system (viewer ON; if Pangolin is an issue in WSL, set to false)
    ORB_SLAM3::System SLAM(vocab, settings, ORB_SLAM3::System::MONOCULAR, /*useViewer=*/true);

    // Open the video with FFMPEG backend (best for mp4/avi)
    cv::VideoCapture cap(video, cv::CAP_FFMPEG);
    if (!cap.isOpened()) {
        std::cerr << "[err] Cannot open video: " << video << "\n";
        return 1;
    }

    bool printed_meta = false;
    cv::Mat frame;

    while (true) {
        if (!cap.read(frame)) break;
        if (frame.empty()) break;

        if (!printed_meta) {
            double fps = cap.get(cv::CAP_PROP_FPS);
            std::cout << "[info] first frame " << frame.cols << "x" << frame.rows
                      << "  fps=" << fps << "\n";
            printed_meta = true;
        }

        SLAM.TrackMonocular(frame, now_sec());
        // if you want to preview frames, uncomment:
        // cv::imshow("mono_input", frame);
        // if (cv::waitKey(1) == 27) break;
    }

    // Ensure ./log exists
    std::system("mkdir -p log");
    std::cout << "[info] writing outputs under: ./log\n";

    // Save KeyFrame trajectory (TUM format). Note: SaveTrajectoryTUM is not for monocular.
    SLAM.SaveKeyFrameTrajectoryTUM("log/KeyFrameTrajectory.txt");

    // -------- Atlas-wide MapPoint export (union + per-map + legacy) --------
    {
        using namespace ORB_SLAM3;

        // Use public getter for the Atlas; some trees keep mpAtlas private.
        Atlas* pAtlas = SLAM.GetAtlas();
        if (!pAtlas) {
            std::cerr << "[warn] Atlas is null; no points exported.\n";
        } else {
            std::vector<Map*> maps = pAtlas->GetAllMaps();

            std::ofstream all("log/pointData_all.csv", std::ios::trunc);
            std::ofstream legacy("log/pointData.csv",     std::ios::trunc);
            all.setf(std::ios::fixed);    all.precision(6);
            legacy.setf(std::ios::fixed); legacy.precision(6);

            size_t total = 0;
            for (size_t i = 0; i < maps.size(); ++i) {
                Map* mpMap = maps[i];
                if (!mpMap) continue;

                std::ostringstream oss;
                oss << "log/pointData_map" << std::setw(3) << std::setfill('0') << i << ".csv";
                std::ofstream per(oss.str(), std::ios::trunc);
                per.setf(std::ios::fixed); per.precision(6);

                const std::vector<MapPoint*> vMPs = mpMap->GetAllMapPoints();
                for (MapPoint* pMP : vMPs) {
                    if (!pMP || pMP->isBad()) continue;

                    // ORB-SLAM3 MapPoint::GetWorldPos() returns Eigen::Matrix<float,3,1>
                    const auto Xw = pMP->GetWorldPos();
                    const float x = Xw[0];
                    const float y = Xw[1];
                    const float z = Xw[2];

                    per    << x << "," << y << "," << z << "\n";
                    all    << x << "," << y << "," << z << "\n";
                    legacy << x << "," << y << "," << z << "\n";
                    ++total;
                }
                per.close();
            }
            all.close();
            legacy.close();

            std::cout << "[info] wrote " << total
                      << " MapPoints across " << maps.size()
                      << " map(s) to pointData_all.csv and per-map files.\n";
        }
    }

    // Clean shutdown (after writing files)
    SLAM.Shutdown();
    return 0;
}
