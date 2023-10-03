#include "utility.hpp"
#include "lio_sam/msg/cloud_info.hpp"
#include "lio_sam/srv/save_map.hpp"
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>

#include <gtsam/nonlinear/ISAM2.h>

using namespace gtsam;

using symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)
using symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)
using symbol_shorthand::G; // GPS pose

/**
 * MapOptimization의 주요 기능
 * 1. scan-to-map matching: 현재 라이다 프레임의 특징 포인트(코너, 평면) 및 local keyframe map의 특징 포인트를 추출하여 scan-to-map 반복 최적화를 수행하고 현재 프레임의 포즈를 업데이트
 * 2. keyframe factor 그래프 최적화: keyframe을 factor 그래프에 추가하고 라이다 오도메트리 factor, GPS factor, loop closer factor를 추가하여 factor 그래프 최적화를 수행하고 모든 keyframe의 pose를 업데이트
 * 3. Loop-closer-detection: keyframe 중에서 거리가 가깝고 시간이 지난 프레임을 찾아서 일치하는 프레임으로 설정하고 일치 프레임 주변에서 로컬 keyframe 맵을 추출하여 scan-to-map 매칭을 수행하고 포즈 변환을 언어서 루프 클로저 factor 데이터를 구성하고 factor 그래프 최적화에 추가
*/

/**
 * 구독:
 * 1. 현재 라이다 프레임 포인트 클라우드 정보를 구독합니다. 이 정보는 FeatureExtraction에서 제공됩니다.
 * 2. GPS 오도메트리를 구독합니다.
 * 3. 외부 루프 클로저 감지 프로그램에서 제공하는 루프 데이터를 구독합니다. 이 프로그램에서는 사용되지 않습니다.
 * 
 * 발행:
 * 1. 과거 키프레임 오도메트리를 발행합니다.
 * 2. 로컬 키프레임 맵의 특징 포인트 클라우드를 발행합니다.
 * 3. 라이다 오도메트리를 발행하며, RViz에서는 좌표축으로 시각화됩니다.
 * 4. 라이다 오도메트리를 발행합니다.
 * 5. 라이다 오도메트리 경로를 발행하며, RViz에서는 이동 경로로 시각화됩니다.
 * 6. 지도 저장 서비스를 발행합니다.
 * 7. 루프 클로저 매치 키프레임의 로컬 맵을 발행합니다.
 * 8. 현재 키프레임을 루프 클로저 최적화 후의 포즈 변환을 적용한 특징 포인트 클라우드로 발행합니다.
 * 9. 루프 클로저 엣지를 발행하며, RViz에서는 루프 프레임 간의 선으로 시각화됩니다.
 * 10. 로컬 맵의 다운샘플 평면 포인트 집합을 발행합니다.
 * 11. 이전 프레임 (누적) 코너 및 평면 포인트 다운샘플 집합을 발행합니다.
 * 12. 현재 프레임 원시 포인트 클라우드를 등록 후 발행합니다.
*/


/**
* A point cloud type that has 6D pose info ([x,y,z,roll,pitch,yaw] intensity is time stamp)
*/
struct PointXYZIRPYT
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;                  // preferred way of adding a XYZ+padding
    float roll;
    float pitch;
    float yaw;
    double time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW   // make sure our new allocators are aligned
} EIGEN_ALIGN16;                    // enforce SSE padding for correct memory alignment

POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZIRPYT,
                                   (float, x, x) (float, y, y)
                                   (float, z, z) (float, intensity, intensity)
                                   (float, roll, roll) (float, pitch, pitch) (float, yaw, yaw)
                                   (double, time, time))

typedef PointXYZIRPYT  PointTypePose;


class mapOptimization : public ParamServer
{

public:

    // gtsam
    NonlinearFactorGraph gtSAMgraph;
    Values initialEstimate;
    Values optimizedEstimate;
    ISAM2 *isam;
    Values isamCurrentEstimate;
    Eigen::MatrixXd poseCovariance;

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudSurround;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubLaserOdometryGlobal;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubLaserOdometryIncremental;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubKeyPoses;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubPath;

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubHistoryKeyFrames;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubIcpKeyFrames;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubRecentKeyFrames;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubRecentKeyFrame;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubCloudRegisteredRaw;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pubLoopConstraintEdge;

    rclcpp::Service<lio_sam::srv::SaveMap>::SharedPtr srvSaveMap;
    rclcpp::Subscription<lio_sam::msg::CloudInfo>::SharedPtr subCloud;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subGPS;
    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr subLoop;

    // GPS 데이터 큐
    std::deque<nav_msgs::msg::Odometry> gpsQueue;
    // 클라우드 정보
    lio_sam::msg::CloudInfo cloudInfo;

    // 과거 모든 키프레임의 코너 포인트 클라우드(다운샘플링)
    vector<pcl::PointCloud<PointType>::Ptr> cornerCloudKeyFrames;
    // 과저 모든 키프레임의 평면 포인트 클라우드(다운샘플링)
    vector<pcl::PointCloud<PointType>::Ptr> surfCloudKeyFrames;
    
    // 과거 키프레임의 위치 정보 포인트 클라우드
    pcl::PointCloud<PointType>::Ptr cloudKeyPoses3D;
    // 과거 키프레임의 위치 및 자세 정보 포인트 클라우드
    pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D;
    pcl::PointCloud<PointType>::Ptr copy_cloudKeyPoses3D;
    pcl::PointCloud<PointTypePose>::Ptr copy_cloudKeyPoses6D;

    // 현재 라이다 프레임 코너 포인트 클라우드
    pcl::PointCloud<PointType>::Ptr laserCloudCornerLast; // corner feature set from odoOptimization
    // 현재 라이다 프레임 평면 포인트 클라우드
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLast; // surf feature set from odoOptimization
    // 현재 라이다 프레임 코너 포인트 클라우드 (다운샘플링)
    pcl::PointCloud<PointType>::Ptr laserCloudCornerLastDS; // downsampled corner feature set from odoOptimization
    // 현재 라이다 프레임 평면 포인트 클라우드 (다운샘플링)
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLastDS; // downsampled surf feature set from odoOptimization

    // 현재 프레임에서 로컬 맵 매칭에 성공한 코너 및 평면 포인트가 결합된 클라우드
    pcl::PointCloud<PointType>::Ptr laserCloudOri;
    pcl::PointCloud<PointType>::Ptr coeffSel;

    // 현재 프레임에서 로컬 맵 매칭에 성공한 코너 포인트와 그에 해당하는 계수
    std::vector<PointType> laserCloudOriCornerVec; // corner point holder for parallel computation 
    std::vector<PointType> coeffSelCornerVec;
    std::vector<bool> laserCloudOriCornerFlag;
    // 현재 프레임에서 지역 맵 매칭에 성공한 평면 포인트와 그에 해당하는 계수
    std::vector<PointType> laserCloudOriSurfVec; // surf point holder for parallel computation
    std::vector<PointType> coeffSelSurfVec;
    std::vector<bool> laserCloudOriSurfFlag;

    map<int, pair<pcl::PointCloud<PointType>, pcl::PointCloud<PointType>>> laserCloudMapContainer;
    // 지역 맵의 코너 포인트 클라우드
    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMap;
    // 지역 맵의 서피스 포인트 클라우드
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMap;
    // 지역 맵의 코너 포인트 클라우드 (다운샘플링)
    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMapDS;
    // 지역 맵의 서피스 포인트 클라우드 (다운샘플링)
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMapDS;

    // 로컬 맵 키프레임에 의해 구축된 맵 포인트 클라우드, 스캔 투 맵 최적화에서 인접 포인트를 찾기 위한 kdtree와 대응
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerFromMap;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromMap;

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurroundingKeyPoses;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeHistoryKeyPoses;

    // 다운샘플링 필터
    pcl::VoxelGrid<PointType> downSizeFilterCorner;
    pcl::VoxelGrid<PointType> downSizeFilterSurf;
    pcl::VoxelGrid<PointType> downSizeFilterICP;
    pcl::VoxelGrid<PointType> downSizeFilterSurroundingKeyPoses; // for surrounding key poses of scan-to-map optimization - scan-to-map 최적화에 필요한 주변 키프레임을 위한 다운샘플링 필터

    rclcpp::Time timeLaserInfoStamp;
    double timeLaserInfoCur;

    float transformTobeMapped[6];

    std::mutex mtx;
    std::mutex mtxLoopInfo;

    bool isDegenerate = false;
    Eigen::Matrix<float, 6, 6> matP;

    // 로컬 맵의 코너 포인트 수
    int laserCloudCornerFromMapDSNum = 0;
    // 로컬 맵의 서피스 포인트 수
    int laserCloudSurfFromMapDSNum = 0;
    // 현재 라이다 프레임의 코너 포인트 수
    int laserCloudCornerLastDSNum = 0;
    // 현재 라이다 프레임의 서피스 포인트 수
    int laserCloudSurfLastDSNum = 0;

    bool aLoopIsClosed = false;
    map<int, int> loopIndexContainer; // from new to old - 새로운 것에서 이전으로
    vector<pair<int, int>> loopIndexQueue;
    vector<gtsam::Pose3> loopPoseQueue;
    vector<gtsam::noiseModel::Diagonal::shared_ptr> loopNoiseQueue;
    deque<std_msgs::msg::Float64MultiArray> loopInfoVec;

    nav_msgs::msg::Path globalPath;

    // 현재 프레임 포인트를 지도에 매핑하기 위한 변환
    Eigen::Affine3f transPointAssociateToMap;
    // 이전 프레임의 오도메트리 적용 변환
    Eigen::Affine3f incrementalOdometryAffineFront;
    // 현재 프레임의 오도메트리 적용 변환
    Eigen::Affine3f incrementalOdometryAffineBack;

    std::unique_ptr<tf2_ros::TransformBroadcaster> br;

    /**
     * 생성자
    */
    mapOptimization(const rclcpp::NodeOptions & options) : ParamServer("lio_sam_mapOptimization", options)
    {
        // ISM2 매개변수 설정
        ISAM2Params parameters;
        parameters.relinearizeThreshold = 0.1;
        parameters.relinearizeSkip = 1;
        isam = new ISAM2(parameters);

        // 과거 키프레임의 위치 정보를 게시
        pubKeyPoses = create_publisher<sensor_msgs::msg::PointCloud2>("lio_sam/mapping/trajectory", 1);
        // 전역 맵의 특징 포인트 클라우드를 게시
        pubLaserCloudSurround = create_publisher<sensor_msgs::msg::PointCloud2>("lio_sam/mapping/map_global", 1);
        // 라이다 오도메트리를 게시하며, rviz에서 좌표축으로 표시
        pubLaserOdometryGlobal = create_publisher<nav_msgs::msg::Odometry>("lio_sam/mapping/odometry", qos);
         // 라이다 오도메트리를 게시하며, 상단의 라이다 오도메트리와 유사하지만 roll 및 pitch는 IMU 데이터를 가중치 평균하고 z는 제한이 적용
        pubLaserOdometryIncremental = create_publisher<nav_msgs::msg::Odometry>(
            "lio_sam/mapping/odometry_incremental", qos);
        // 라이다 오도메트리 경로를 게시하며, rviz에서 이동 경로로 표시
        pubPath = create_publisher<nav_msgs::msg::Path>("lio_sam/mapping/path", 1);
        br = std::make_unique<tf2_ros::TransformBroadcaster>(this);

        // 현재 라이다 프레임의 포인트 클라우드 정보를 구독합니다. (FeatureExtraction으로부터)
        subCloud = create_subscription<lio_sam::msg::CloudInfo>(
            "lio_sam/feature/cloud_info", qos,
            std::bind(&mapOptimization::laserCloudInfoHandler, this, std::placeholders::_1));
        // GPS 오도메트리를 구독
        subGPS = create_subscription<nav_msgs::msg::Odometry>(
            gpsTopic, 200,
            std::bind(&mapOptimization::gpsHandler, this, std::placeholders::_1));
        // 외부 루프 검출 프로그램에서 제공하는 루프 데이터를 구독 (이 코드에서는 사용되지 않음)
        subLoop = create_subscription<std_msgs::msg::Float64MultiArray>(
            "lio_loop/loop_closure_detection", qos,
            std::bind(&mapOptimization::loopInfoHandler, this, std::placeholders::_1));


        auto saveMapService = [this](const std::shared_ptr<rmw_request_id_t> request_header, const std::shared_ptr<lio_sam::srv::SaveMap::Request> req, std::shared_ptr<lio_sam::srv::SaveMap::Response> res) -> void {
            (void)request_header;
            string saveMapDirectory;
            cout << "****************************************************" << endl;
            cout << "Saving map to pcd files ..." << endl;
            if(req->destination.empty()) saveMapDirectory = std::getenv("HOME") + savePCDDirectory;
            else saveMapDirectory = std::getenv("HOME") + req->destination;
            cout << "Save destination: " << saveMapDirectory << endl;
            // create directory and remove old files;
            int unused = system((std::string("exec rm -r ") + saveMapDirectory).c_str());
            unused = system((std::string("mkdir -p ") + saveMapDirectory).c_str());
            // save key frame transformations
            pcl::io::savePCDFileBinary(saveMapDirectory + "/trajectory.pcd", *cloudKeyPoses3D);
            pcl::io::savePCDFileBinary(saveMapDirectory + "/transformations.pcd", *cloudKeyPoses6D);
            // extract global point cloud map
            pcl::PointCloud<PointType>::Ptr globalCornerCloud(new pcl::PointCloud<PointType>());
            pcl::PointCloud<PointType>::Ptr globalCornerCloudDS(new pcl::PointCloud<PointType>());
            pcl::PointCloud<PointType>::Ptr globalSurfCloud(new pcl::PointCloud<PointType>());
            pcl::PointCloud<PointType>::Ptr globalSurfCloudDS(new pcl::PointCloud<PointType>());
            pcl::PointCloud<PointType>::Ptr globalMapCloud(new pcl::PointCloud<PointType>());
            for (int i = 0; i < (int)cloudKeyPoses3D->size(); i++) 
            {
                *globalCornerCloud += *transformPointCloud(cornerCloudKeyFrames[i],  &cloudKeyPoses6D->points[i]);
                *globalSurfCloud   += *transformPointCloud(surfCloudKeyFrames[i],    &cloudKeyPoses6D->points[i]);
                cout << "\r" << std::flush << "Processing feature cloud " << i << " of " << cloudKeyPoses6D->size() << " ...";
            }
            if(req->resolution != 0)
            {
               cout << "\n\nSave resolution: " << req->resolution << endl;
               // down-sample and save corner cloud
               downSizeFilterCorner.setInputCloud(globalCornerCloud);
               downSizeFilterCorner.setLeafSize(req->resolution, req->resolution, req->resolution);
               downSizeFilterCorner.filter(*globalCornerCloudDS);
               pcl::io::savePCDFileBinary(saveMapDirectory + "/CornerMap.pcd", *globalCornerCloudDS);
               // down-sample and save surf cloud
               downSizeFilterSurf.setInputCloud(globalSurfCloud);
               downSizeFilterSurf.setLeafSize(req->resolution, req->resolution, req->resolution);
               downSizeFilterSurf.filter(*globalSurfCloudDS);
               pcl::io::savePCDFileBinary(saveMapDirectory + "/SurfMap.pcd", *globalSurfCloudDS);
            }
            else
            {
            // save corner cloud
               pcl::io::savePCDFileBinary(saveMapDirectory + "/CornerMap.pcd", *globalCornerCloud);
               // save surf cloud
               pcl::io::savePCDFileBinary(saveMapDirectory + "/SurfMap.pcd", *globalSurfCloud);
            }
            // save global point cloud map
            *globalMapCloud += *globalCornerCloud;
            *globalMapCloud += *globalSurfCloud;
            int ret = pcl::io::savePCDFileBinary(saveMapDirectory + "/GlobalMap.pcd", *globalMapCloud);
            res->success = ret == 0;
            downSizeFilterCorner.setLeafSize(mappingCornerLeafSize, mappingCornerLeafSize, mappingCornerLeafSize);
            downSizeFilterSurf.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
            cout << "****************************************************" << endl;
            cout << "Saving map to pcd files completed\n" << endl;
            return;
        };

        // 지도 저장 서비스를 게시
        srvSaveMap = create_service<lio_sam::srv::SaveMap>("lio_sam/save_map", saveMapService);
        // 클라우드의 포인트를 지도에 할당하기 위한 변환
        pubHistoryKeyFrames = create_publisher<sensor_msgs::msg::PointCloud2>("lio_sam/mapping/icp_loop_closure_history_cloud", 1);
        // 루프 클로저를 적용한 후 현재 키프레임의 포인트 클라우드를 게시
        pubIcpKeyFrames = create_publisher<sensor_msgs::msg::PointCloud2>("lio_sam/mapping/icp_loop_closure_history_cloud", 1);
        // 루프 제약 조건을 게시하며, rviz에서 루프 프레임 간의 연결선으로 표시
        pubLoopConstraintEdge = create_publisher<visualization_msgs::msg::MarkerArray>("/lio_sam/mapping/loop_closure_constraints", 1);

        // 지역 맵의 다운샘플링된 평면 포인트 클라우드를 게시
        pubRecentKeyFrames = create_publisher<sensor_msgs::msg::PointCloud2>("lio_sam/mapping/map_local", 1);
        // 과거 프레임(누적)의 코너 및 서피스 포인트의 다운샘플링된 클라우드를 게시
        pubRecentKeyFrame = create_publisher<sensor_msgs::msg::PointCloud2>("lio_sam/mapping/cloud_registered", 1);
        // 현재 프레임의 원본 포인트 클라우드를 등록한 후 게시
        pubCloudRegisteredRaw = create_publisher<sensor_msgs::msg::PointCloud2>("lio_sam/mapping/cloud_registered_raw", 1);

        downSizeFilterCorner.setLeafSize(mappingCornerLeafSize, mappingCornerLeafSize, mappingCornerLeafSize);
        downSizeFilterSurf.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
        downSizeFilterICP.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);
        // scan-to-map 최적화에 필요한 주변 키프레임을 위한 다운샘플링 필터
        downSizeFilterSurroundingKeyPoses.setLeafSize(surroundingKeyframeDensity, surroundingKeyframeDensity, surroundingKeyframeDensity); // for surrounding key poses of scan-to-map optimization

        allocateMemory();
    }

    void allocateMemory()
    {
        cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
        cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());
        copy_cloudKeyPoses3D.reset(new pcl::PointCloud<PointType>());
        copy_cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());

        kdtreeSurroundingKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());
        kdtreeHistoryKeyPoses.reset(new pcl::KdTreeFLANN<PointType>());

        laserCloudCornerLast.reset(new pcl::PointCloud<PointType>()); // corner feature set from odoOptimization
        laserCloudSurfLast.reset(new pcl::PointCloud<PointType>()); // surf feature set from odoOptimization
        laserCloudCornerLastDS.reset(new pcl::PointCloud<PointType>()); // downsampled corner featuer set from odoOptimization
        laserCloudSurfLastDS.reset(new pcl::PointCloud<PointType>()); // downsampled surf featuer set from odoOptimization

        laserCloudOri.reset(new pcl::PointCloud<PointType>());
        coeffSel.reset(new pcl::PointCloud<PointType>());

        laserCloudOriCornerVec.resize(N_SCAN * Horizon_SCAN);
        coeffSelCornerVec.resize(N_SCAN * Horizon_SCAN);
        laserCloudOriCornerFlag.resize(N_SCAN * Horizon_SCAN);
        laserCloudOriSurfVec.resize(N_SCAN * Horizon_SCAN);
        coeffSelSurfVec.resize(N_SCAN * Horizon_SCAN);
        laserCloudOriSurfFlag.resize(N_SCAN * Horizon_SCAN);

        std::fill(laserCloudOriCornerFlag.begin(), laserCloudOriCornerFlag.end(), false);
        std::fill(laserCloudOriSurfFlag.begin(), laserCloudOriSurfFlag.end(), false);

        laserCloudCornerFromMap.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfFromMap.reset(new pcl::PointCloud<PointType>());
        laserCloudCornerFromMapDS.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfFromMapDS.reset(new pcl::PointCloud<PointType>());

        kdtreeCornerFromMap.reset(new pcl::KdTreeFLANN<PointType>());
        kdtreeSurfFromMap.reset(new pcl::KdTreeFLANN<PointType>());

        for (int i = 0; i < 6; ++i){
            transformTobeMapped[i] = 0;
        }

        matP.setZero();
    }

    /**
     * 현재 라이다 정보를 처리합니다. 이 정보는 featureExtraction에서 제공됩니다.
     * 1. 현재 프레임의 포즈 초기화
     *   1) 첫 번째 프레임인 경우, 원시 IMU 데이터의 RPY를 사용하여 현재 프레임의 포즈(회전 부분)를 초기화합니다.
     *   2) 이후 프레임에서는 IMU 오도메트리를 사용하여 두 프레임 간의 포즈 변환을 계산하고, 이를 이전 프레임의 라이다 포즈에 적용하여 현재 프레임의 라이다 포즈를 얻습니다.
     * 2. 로컬 각도 및 평면 포인트 클라우드를 추출하고 로컬 맵에 추가합니다.
     *   1) 가장 최근의 키프레임에서 공간 및 시간 차원에서 인접한 키프레임 세트를 검색하고 다운샘플링합니다.
     *   2) 키프레임 세트의 각 프레임에 대해 해당 각도 및 평면 포인트를 추출하고 로컬 맵에 추가합니다.
     * 3. 현재 라이다 프레임의 각도 및 평면 포인트 클라우드를 다운샘플링합니다.
     * 4. 스캔 대 맵을 사용하여 현재 프레임의 포즈를 최적화합니다.
     *   (1) 현재 프레임의 특징 포인트 수가 충분하고 매칭 포인트 수가 충분한 경우에만 최적화를 수행합니다.
     *   (2) 최대 30회 반복하여 최적화합니다.
     *       1) 현재 라이다 프레임 각도 포인트는 로컬 맵에 매칭 포인트를 찾습니다.
     *          a. 현재 프레임 포즈를 업데이트하고 현재 프레임의 각도 포인트 좌표를 맵 좌표로 변환하고 로컬 맵에서 가장 가까운 1m 이내의 5개 포인트를 검색하며, 이 5개 포인트가 1m 이내에 있고 5개 포인트가 직선을 형성하는 경우 (거리 중심점 및 공분산 행렬, 고유값을 사용하여 확인) 매치된 것으로 간주합니다.
     *          b. 현재 프레임의 각도 포인트에서 직선까지의 거리 및 수직 단위 벡터를 계산하여 각도 포인트 파라미터로 저장합니다.
     *       2) 현재 라이다 프레임의 평면 포인트는 로컬 맵에 매칭 포인트를 찾습니다.
     *          a. 현재 프레임 포즈를 업데이트하고 현재 프레임의 평면 포인트 좌표를 맵 좌표로 변환하고 로컬 맵에서 가장 가까운 1m 이내의 5개 포인트를 검색하며, 이 5개 포인트가 1m 이내에 있고 5개 포인트가 평면을 형성하는 경우 (최소 제곱 평면 적합을 사용하여 확인) 매치된 것으로 간주합니다.
     *          b. 현재 프레임의 평면 포인트에서 평면까지의 거리 및 수직 단위 벡터를 계산하여 평면 포인트 파라미터로 저장합니다.
     *       3) 로컬 맵과 매치된 현재 프레임의 각도 및 평면 포인트를 추출하고 동일한 집합에 추가합니다.
     *       4) 매치된 특징 포인트에 대한 Jacobian 행렬을 계산하고 관측 값은 포인트에서 직선 또는 평면까지의 거리이며, 가우스 뉴턴 방정식을 구성하여 현재 포즈를 반복적으로 최적화합니다. 최적화된 결과는 transformTobeMapped에 저장됩니다.
     *   (3) IMU 원시 RPY 데이터와 스캔 대 맵 최적화 후의 포즈를 가중 평균하여 현재 프레임의 롤 및 피치를 업데이트하고 Z 좌표를 제한합니다.
     * 5. 현재 프레임을 키프레임으로 설정하고 팩터 그래프 최적화를 실행합니다.
     *   1) 현재 프레임과 이전 프레임 사이의 포즈 변환을 계산하고 변화가 작으면 키프레임으로 설정하지 않고, 클 경우 키프레임으로 설정합니다.
     *   2) 라이다 오도메트리 팩터, GPS 팩터, 루프 클로저 팩터를 추가합니다.
     *   3) 팩터 그래프 최적화를 수행합니다.
     *   4) 최적화된 현재 프레임 포즈와 포즈 공분산을 얻습니다.
     *   5) cloudKeyPoses3D 및 cloudKeyPoses6D를 추가하고 transformTobeMapped을 업데이트하고 현재 키프레임의 각도 및 평면 포인트 클라우드를 추가합니다.
     * 6. 모든 변수 노드의 포즈를 업데이트하여 모든 이전 키프레임의 포즈를 업데이트하고 오도메트리 경로를 업데이트합니다.
     * 7. 라이다 오도메트리를 게시합니다.
     * 8. 오도메트리, 포인트 클라우드, 경로를 게시합니다.
 */
    void laserCloudInfoHandler(const lio_sam::msg::CloudInfo::SharedPtr msgIn)
    {
        // extract time stamp
        // 현재 라이다 프레임의 타임스탬프
        timeLaserInfoStamp = msgIn->header.stamp;
        timeLaserInfoCur = stamp2Sec(msgIn->header.stamp);

        // extract info and feature cloud
        // 현재 라이다 포인트 클라우드 정보 추출
        cloudInfo = *msgIn;
        pcl::fromROSMsg(msgIn->cloud_corner,  *laserCloudCornerLast);
        pcl::fromROSMsg(msgIn->cloud_surface, *laserCloudSurfLast);

        std::lock_guard<std::mutex> lock(mtx);

        // 매핑 실행 빈도 제어
        static double timeLastProcessing = -1;
        if (timeLaserInfoCur - timeLastProcessing >= mappingProcessInterval)
        {
            timeLastProcessing = timeLaserInfoCur;
            // 현재 프레임의 초기 추정값 업데이트
            // 1. 첫 번째 프레임이면 원시 imu 데이터의 RPY를 사용하여 현재 프레임의 자세를 초기화합니다 (회전 부분).
            // 2. 이후의 프레임에서는 imu 오도미터를 사용하여 두 프레임 간의 증가 자세 변환을 계산하고 이를 이전 프레임의 레이저 자세에 적용하여 현재 프레임 레이저 자세를 얻습니다.
            updateInitialGuess();

            // 주변 키프레임에서 로컬 코너 및 서피스 포인트 클라우드 추출 및 로컬 맵에 추가
            // 1. 가장 최근의 키프레임에서 공간 및 시간 차원에서 이웃하는 키프레임 집합을 검색하고 샘플링합니다.
            // 2. 키프레임 집합의 각 프레임에서 해당하는 코너 및 서피스 포인트를 추출하고 로컬 맵에 추가합니다.
            extractSurroundingKeyFrames();

            // 현재 레이저 프레임의 코너 및 평면 포인트 클라우드 다운샘플링
            downsampleCurrentScan();

            // scan-to-map 최적화를 통해 현재 프레임 자세 최적화
            // 1. 현재 프레임의 특징 포인트 수가 충분하고 매치 포인트가 충분한 경우에만 최적화를 수행합니다.
            // 2. 최대 30번 반복 (상한) 최적화
            //    1) 현재 레이저 프레임의 코너를 로컬 맵에 매칭
            //       a. 현재 프레임 자세를 업데이트하고 현재 프레임 코너 좌표를 맵 좌표로 변환하고 로컬 맵에서 5개의 가장 가까운 포인트를 찾습니다. 거리가 1m 미만이고 5개의 포인트가 직선을 형성하면 (중심점으로부터의 거리 및 공분산 행렬, 고유값 사용) 일치했다고 간주합니다.
            //       b. 현재 프레임 코너에서 직선까지의 거리 및 수직선의 단위 벡터를 계산하고 코너 매개 변수로 저장합니다.
            //    2) 현재 레이저 프레임의 평면을 로컬 맵에 매칭
            //       a. 현재 프레임 자세를 업데이트하고 현재 프레임 평면 좌표를 맵 좌표로 변환하고 로컬 맵에서 5개의 가장 가까운 포인트를 찾습니다. 거리가 1m 미만이고 5개의 포인트가 평면을 형성하면 (최소 제곱 평면을 적합) 일치했다고 간주합니다.
            //       b. 현재 프레임 평면에서 평면까지의 거리 및 수직선의 단위 벡터를 계산하고 평면 매개 변수로 저장합니다.
            //    3) 현재 프레임에서 로컬 맵에 일치하는 코너 및 평면을 추출하고 동일한 집합에 추가합니다.
            //    4) 일치하는 특징 포인트에 대한 Jacobian 행렬을 계산하고 관측 값은 특징 포인트에서 직선 또는 평면까지의 거리입니다. 가우스 뉴턴 방정식을 구성하고 현재 자세에 대한 반복 최적화를 수행하여 transformTobeMapped을 저장합니다.
            // 3. imu 원시 RPY 데이터와 scan-to-map 최적화된 자세를 가중치로 결합하여 현재 프레임의 roll, pitch를 업데이트하고 z 좌표를 제한합니다.
            scan2MapOptimization();

            // 현재 프레임을 키프레임으로 설정하고 그래프 최적화 수행
            // 1. 현재 프레임과 이전 프레임 자세 변환을 계산하고 변화가 너무 작으면 키프레임으로 설정하지 않습니다. 그렇지 않으면 키프레임으로 설정합니다.
            // 2. 레이저 오도미터 팩터, GPS 팩터, 루프 클로저 팩터를 추가합니다.
            // 3. 그래프 최적화를 수행합니다.
            // 4. 최적화된 현재 프레임 자세, 자세 공분산을 얻습니다. cloudKeyPoses3D, cloudKeyPoses6D를 추가하고 transformTobeMapped을 업데이트하고 현재 키프레임의 코너 및 평면 포인트를 추가합니다.            
            saveKeyFramesAndFactor();

            // 그래프에서 모든 변수 노드의 자세를 업데이트합니다. 즉, 모든 이전 키프레임의 자세를 업데이트하고 오도메트리 경로를 업데이트합니다.
            correctPoses();

            // 라이다 오도메트리 게시
            publishOdometry();

            // 오도미터, 포인트 클라우드, 트라젝토리를 게시합니다.
            // 1. 과거 키프레임 자세를 게시합니다.
            // 2. 로컬 맵의 다운샘플링된 평면 포인트를 게시합니다.
            // 3. 히스토리 프레임 (누적된)의 코너 및 평면 포인트 다운샘플링 세트를 게시합니다.
            // 4. 오도미터 트라젝토리를 게시합니다.
            publishFrames();
        }
    }

    void gpsHandler(const nav_msgs::msg::Odometry::SharedPtr gpsMsg)
    {
        gpsQueue.push_back(*gpsMsg);
    }

    void pointAssociateToMap(PointType const * const pi, PointType * const po)
    {
        po->x = transPointAssociateToMap(0,0) * pi->x + transPointAssociateToMap(0,1) * pi->y + transPointAssociateToMap(0,2) * pi->z + transPointAssociateToMap(0,3);
        po->y = transPointAssociateToMap(1,0) * pi->x + transPointAssociateToMap(1,1) * pi->y + transPointAssociateToMap(1,2) * pi->z + transPointAssociateToMap(1,3);
        po->z = transPointAssociateToMap(2,0) * pi->x + transPointAssociateToMap(2,1) * pi->y + transPointAssociateToMap(2,2) * pi->z + transPointAssociateToMap(2,3);
        po->intensity = pi->intensity;
    }

    pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn, PointTypePose* transformIn)
    {
        pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

        int cloudSize = cloudIn->size();
        cloudOut->resize(cloudSize);

        Eigen::Affine3f transCur = pcl::getTransformation(transformIn->x, transformIn->y, transformIn->z, transformIn->roll, transformIn->pitch, transformIn->yaw);
        
        #pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < cloudSize; ++i)
        {
            const auto &pointFrom = cloudIn->points[i];
            cloudOut->points[i].x = transCur(0,0) * pointFrom.x + transCur(0,1) * pointFrom.y + transCur(0,2) * pointFrom.z + transCur(0,3);
            cloudOut->points[i].y = transCur(1,0) * pointFrom.x + transCur(1,1) * pointFrom.y + transCur(1,2) * pointFrom.z + transCur(1,3);
            cloudOut->points[i].z = transCur(2,0) * pointFrom.x + transCur(2,1) * pointFrom.y + transCur(2,2) * pointFrom.z + transCur(2,3);
            cloudOut->points[i].intensity = pointFrom.intensity;
        }
        return cloudOut;
    }

    gtsam::Pose3 pclPointTogtsamPose3(PointTypePose thisPoint)
    {
        return gtsam::Pose3(gtsam::Rot3::RzRyRx(double(thisPoint.roll), double(thisPoint.pitch), double(thisPoint.yaw)),
                                  gtsam::Point3(double(thisPoint.x),    double(thisPoint.y),     double(thisPoint.z)));
    }

    gtsam::Pose3 trans2gtsamPose(float transformIn[])
    {
        return gtsam::Pose3(gtsam::Rot3::RzRyRx(transformIn[0], transformIn[1], transformIn[2]), 
                                  gtsam::Point3(transformIn[3], transformIn[4], transformIn[5]));
    }

    Eigen::Affine3f pclPointToAffine3f(PointTypePose thisPoint)
    {
        return pcl::getTransformation(thisPoint.x, thisPoint.y, thisPoint.z, thisPoint.roll, thisPoint.pitch, thisPoint.yaw);
    }

    Eigen::Affine3f trans2Affine3f(float transformIn[])
    {
        return pcl::getTransformation(transformIn[3], transformIn[4], transformIn[5], transformIn[0], transformIn[1], transformIn[2]);
    }

    PointTypePose trans2PointTypePose(float transformIn[])
    {
        PointTypePose thisPose6D;
        thisPose6D.x = transformIn[3];
        thisPose6D.y = transformIn[4];
        thisPose6D.z = transformIn[5];
        thisPose6D.roll  = transformIn[0];
        thisPose6D.pitch = transformIn[1];
        thisPose6D.yaw   = transformIn[2];
        return thisPose6D;
    }

    void visualizeGlobalMapThread()
    {
        rclcpp::Rate rate(0.2);
        while (rclcpp::ok()){
            rate.sleep();
            publishGlobalMap();
        }
        if (savePCD == false)
            return;
        cout << "****************************************************" << endl;
        cout << "Saving map to pcd files ..." << endl;
        savePCDDirectory = std::getenv("HOME") + savePCDDirectory;
        int unused = system((std::string("exec rm -r ") + savePCDDirectory).c_str());
        unused = system((std::string("mkdir ") + savePCDDirectory).c_str());
        pcl::io::savePCDFileASCII(savePCDDirectory + "trajectory.pcd", *cloudKeyPoses3D);
        pcl::io::savePCDFileASCII(savePCDDirectory + "transformations.pcd", *cloudKeyPoses6D);
        pcl::PointCloud<PointType>::Ptr globalCornerCloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalCornerCloudDS(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalSurfCloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalSurfCloudDS(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapCloud(new pcl::PointCloud<PointType>());
        for (int i = 0; i < (int)cloudKeyPoses3D->size(); i++) {
            *globalCornerCloud += *transformPointCloud(cornerCloudKeyFrames[i],  &cloudKeyPoses6D->points[i]);
            *globalSurfCloud   += *transformPointCloud(surfCloudKeyFrames[i],    &cloudKeyPoses6D->points[i]);
            cout << "\r" << std::flush << "Processing feature cloud " << i << " of " << cloudKeyPoses6D->size() << " ...";
        }
        downSizeFilterCorner.setInputCloud(globalCornerCloud);
        downSizeFilterCorner.filter(*globalCornerCloudDS);
        pcl::io::savePCDFileASCII(savePCDDirectory + "cloudCorner.pcd", *globalCornerCloudDS);
        downSizeFilterSurf.setInputCloud(globalSurfCloud);
        downSizeFilterSurf.filter(*globalSurfCloudDS);
        pcl::io::savePCDFileASCII(savePCDDirectory + "cloudSurf.pcd", *globalSurfCloudDS);
        *globalMapCloud += *globalCornerCloud;
        *globalMapCloud += *globalSurfCloud;
        pcl::io::savePCDFileASCII(savePCDDirectory + "cloudGlobal.pcd", *globalMapCloud);
        cout << "****************************************************" << endl;
        cout << "Saving map to pcd files completed" << endl;
    }

    void publishGlobalMap()
    {
        if (pubLaserCloudSurround->get_subscription_count() == 0)
            return;

        if (cloudKeyPoses3D->points.empty() == true)
            return;

        pcl::KdTreeFLANN<PointType>::Ptr kdtreeGlobalMap(new pcl::KdTreeFLANN<PointType>());;
        pcl::PointCloud<PointType>::Ptr globalMapKeyPoses(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyPosesDS(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyFrames(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr globalMapKeyFramesDS(new pcl::PointCloud<PointType>());

        // kd-tree to find near key frames to visualize
        std::vector<int> pointSearchIndGlobalMap;
        std::vector<float> pointSearchSqDisGlobalMap;
        // search near key frames to visualize
        mtx.lock();
        kdtreeGlobalMap->setInputCloud(cloudKeyPoses3D);
        kdtreeGlobalMap->radiusSearch(cloudKeyPoses3D->back(), globalMapVisualizationSearchRadius, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap, 0);
        mtx.unlock();

        for (int i = 0; i < (int)pointSearchIndGlobalMap.size(); ++i)
            globalMapKeyPoses->push_back(cloudKeyPoses3D->points[pointSearchIndGlobalMap[i]]);
        // downsample near selected key frames
        pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyPoses; // for global map visualization
        downSizeFilterGlobalMapKeyPoses.setLeafSize(globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity); // for global map visualization
        downSizeFilterGlobalMapKeyPoses.setInputCloud(globalMapKeyPoses);
        downSizeFilterGlobalMapKeyPoses.filter(*globalMapKeyPosesDS);
        for(auto& pt : globalMapKeyPosesDS->points)
        {
            kdtreeGlobalMap->nearestKSearch(pt, 1, pointSearchIndGlobalMap, pointSearchSqDisGlobalMap);
            pt.intensity = cloudKeyPoses3D->points[pointSearchIndGlobalMap[0]].intensity;
        }

        // extract visualized and downsampled key frames
        for (int i = 0; i < (int)globalMapKeyPosesDS->size(); ++i){
            if (pointDistance(globalMapKeyPosesDS->points[i], cloudKeyPoses3D->back()) > globalMapVisualizationSearchRadius)
                continue;
            int thisKeyInd = (int)globalMapKeyPosesDS->points[i].intensity;
            *globalMapKeyFrames += *transformPointCloud(cornerCloudKeyFrames[thisKeyInd],  &cloudKeyPoses6D->points[thisKeyInd]);
            *globalMapKeyFrames += *transformPointCloud(surfCloudKeyFrames[thisKeyInd],    &cloudKeyPoses6D->points[thisKeyInd]);
        }
        // downsample visualized points
        pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyFrames; // for global map visualization
        downSizeFilterGlobalMapKeyFrames.setLeafSize(globalMapVisualizationLeafSize, globalMapVisualizationLeafSize, globalMapVisualizationLeafSize); // for global map visualization
        downSizeFilterGlobalMapKeyFrames.setInputCloud(globalMapKeyFrames);
        downSizeFilterGlobalMapKeyFrames.filter(*globalMapKeyFramesDS);
        publishCloud(pubLaserCloudSurround, globalMapKeyFramesDS, timeLaserInfoStamp, odometryFrame);
    }










    /**
     * 루프 클로저 검출 스레드
     * 1. 루프 클로저 검출 scan-to-map, ICP를 사용하여 자세 최적화
     *   1) 과거 키프레임 중 현재 키프레임과 가장 가까운 키프레임 집합을 찾고, 일정 시간이 지난 키프레임 중 하나를 선택하여 후보로 선정
     *   2) 현재 키프레임 특징점 집합을 추출하고 다운샘플링; 루프 클로저 매칭 키프레임 앞뒤의 일부 키프레임 특징점 집합을 추출하고 다운샘플링
     *   3) scan-to-map 최적화 실행, ICP 메소드 호출, 최적화된 자세 획득, 闭环 인자에 필요한 데이터를 생성하여 그래프 최적화에서 자세를 업데이트
     * 2. rviz에 루프 클로저 경계 표시
    */
    void loopClosureThread()
    {
        // 루피 클로저 감지 플래그 확인
        if (loopClosureEnableFlag == false)
            return;

        rclcpp::Rate rate(loopClosureFrequency);
        while (rclcpp::ok())
        {
            rate.sleep();

            // 루프 클로저 검출, scan-to-map 자세 최적화
            // 1. 현재 키프레임과 가장 가까운 키프레임 집합을 찾아 일정 시간이 지난 키프레임 중 하나를 선택하여 후보로 선정
            // 2. 현재 키프레임 특징점 집합을 추출하고 다운샘플링; 闭环 매칭 키프레임 앞뒤의 일부 키프레임 특징점 집합을 추출하고 다운샘플링
            // 3. scan-to-map 최적화 실행, ICP 메소드 호출, 최적화된 자세 획득, 闭环 인자에 필요한 데이터를 생성하여 그래프 최적화에서 자세를 업데이트
            // 주의: 루프 클로저 시 현재 프레임 자세를 즉시 업데이트하지 않고, 루프 클로저 인자를 추가하여 그래프 최적화가 자세를 업데이트하도록 함
            performLoopClosure();
            
            // rviz에 루프 클로저 경계 표시 
            visualizeLoopClosure();
        }
    }

    void loopInfoHandler(const std_msgs::msg::Float64MultiArray::SharedPtr loopMsg)
    {
        std::lock_guard<std::mutex> lock(mtxLoopInfo);
        if (loopMsg->data.size() != 2)
            return;

        loopInfoVec.push_back(*loopMsg);

        while (loopInfoVec.size() > 5)
            loopInfoVec.pop_front();
    }


    /**
     * 루프 클로저 스캔-투-맵, ICP로 최적화된 자세
     * 1. 현재 키프레임과 가장 가까운 역사적 키프레임 모음을 찾아 시간 차이가 큰 프레임을 후보로 선택합니다.
     * 2. 현재 키프레임 특징점 집합을 추출하고 다운 샘플링하며, 닫힌 루프 매칭 키프레임 주변의 몇 프레임의 특징점을 추출하고 다운 샘플링합니다.
     * 3. 스캔-투-맵 최적화를 수행하고 ICP 메소드를 호출하여 최적화된 자세를 얻으며, 닫힌 루프 요소에 필요한 데이터를 생성하고, 그래프 최적화에서 업데이트된 자세를 동시에 추가합니다.
     * 참고: 닫힌 루프시 현재 프레임 자세를 즉시 업데이트하지 않고, 대신 닫힌 루프 요소를 추가하여 그래프 최적화에서 자세를 업데이트하도록 합니다.
    */
    void performLoopClosure()
    {
        if (cloudKeyPoses3D->points.empty() == true)
            return;

        mtx.lock();
        *copy_cloudKeyPoses3D = *cloudKeyPoses3D;
        *copy_cloudKeyPoses6D = *cloudKeyPoses6D;
        mtx.unlock();

        // find keys
        // 현재 키프레임 인덱스, 닫힌 루프 매칭 프레임 인덱스
        int loopKeyCur;
        int loopKeyPre;
        if (detectLoopClosureExternal(&loopKeyCur, &loopKeyPre) == false)
            // 현재 키프레임과 가장 가까운 역사적 키프레임 모음을 찾아 시간 차이가 큰 프레임을 후보로 선택합니다.
            if (detectLoopClosureDistance(&loopKeyCur, &loopKeyPre) == false)
                return;

        // extract cloud
        // 추출
        pcl::PointCloud<PointType>::Ptr cureKeyframeCloud(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr prevKeyframeCloud(new pcl::PointCloud<PointType>());
        {
            // 현재 키프레임 특징점 집합을 추출하고 다운 샘플링합니다.
            loopFindNearKeyframes(cureKeyframeCloud, loopKeyCur, 0);
            // 닫힌 루프 매칭 키프레임 주변의 몇 프레임의 특징점을 추출하고 다운 샘플링합니다.
            loopFindNearKeyframes(prevKeyframeCloud, loopKeyPre, historyKeyframeSearchNum);
            // 특징점이 충분히 적으면 반환합니다.
            if (cureKeyframeCloud->size() < 300 || prevKeyframeCloud->size() < 1000)
                return;
            // 닫힌 루프 매칭 키프레임의 지역 맵을 게시합니다.
            if (pubHistoryKeyFrames->get_subscription_count() != 0)
                publishCloud(pubHistoryKeyFrames, prevKeyframeCloud, timeLaserInfoStamp, odometryFrame);
        }

        // ICP Settings
        // ICP 매개변수 설정
        static pcl::IterativeClosestPoint<PointType, PointType> icp;
        icp.setMaxCorrespondenceDistance(historyKeyframeSearchRadius*2);
        icp.setMaximumIterations(100);
        icp.setTransformationEpsilon(1e-6);
        icp.setEuclideanFitnessEpsilon(1e-6);
        icp.setRANSACIterations(0);

        // Align clouds
        // 스캔-투-맵, ICP 매칭 수행
        icp.setInputSource(cureKeyframeCloud);
        icp.setInputTarget(prevKeyframeCloud);
        pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
        icp.align(*unused_result);

        // 수렴하지 않았거나 매칭이 충분히 좋지 않으면 반환합니다.
        if (icp.hasConverged() == false || icp.getFitnessScore() > historyKeyframeFitnessScore)
            return;

        // publish corrected cloud
        // 닫힌 루프 최적화 후 현재 키프레임의 자세를 변환한 특징점 클라우드를 게시합니다.
        if (pubIcpKeyFrames->get_subscription_count() != 0)
        {
            pcl::PointCloud<PointType>::Ptr closed_cloud(new pcl::PointCloud<PointType>());
            pcl::transformPointCloud(*cureKeyframeCloud, *closed_cloud, icp.getFinalTransformation());
            publishCloud(pubIcpKeyFrames, closed_cloud, timeLaserInfoStamp, odometryFrame);
        }

        // Get pose transformation
        // 닫힌 루프 최적화 후 현재 키프레임과 닫힌 루프 키프레임 간의 자세 변환
        float x, y, z, roll, pitch, yaw;
        Eigen::Affine3f correctionLidarFrame;
        correctionLidarFrame = icp.getFinalTransformation();
        // transform from world origin to wrong pose
        // 닫힌 루프 최적화 이전 현재 프레임 자세
        Eigen::Affine3f tWrong = pclPointToAffine3f(copy_cloudKeyPoses6D->points[loopKeyCur]);
        // transform from world origin to corrected pose
        // 닫힌 루프 최적화 후 현재 프레임 자세
        Eigen::Affine3f tCorrect = correctionLidarFrame * tWrong;// pre-multiplying -> successive rotation about a fixed frame
        pcl::getTranslationAndEulerAngles (tCorrect, x, y, z, roll, pitch, yaw);
        gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(x, y, z));
        // 닫힌 루프 매칭 프레임의 자세
        gtsam::Pose3 poseTo = pclPointTogtsamPose3(copy_cloudKeyPoses6D->points[loopKeyPre]);
        gtsam::Vector Vector6(6);
        float noiseScore = icp.getFitnessScore();
        Vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore, noiseScore;
        noiseModel::Diagonal::shared_ptr constraintNoise = noiseModel::Diagonal::Variances(Vector6);

        // Add pose constraint
        // 닫힌 루프 요소에 필요한 데이터 추가
        mtx.lock();
        loopIndexQueue.push_back(make_pair(loopKeyCur, loopKeyPre));
        loopPoseQueue.push_back(poseFrom.between(poseTo));
        loopNoiseQueue.push_back(constraintNoise);
        mtx.unlock();

        // add loop constriant
        loopIndexContainer[loopKeyCur] = loopKeyPre;
    }

    /**
     * 현재 키프레임과 거리가 가장 가까운 키프레임 집합을 기반으로 하여, 시간 차이가 큰 키프레임을 선택하여 닫힌 루프 후보 프레임으로 결정합니다.
    */
    bool detectLoopClosureDistance(int *latestID, int *closestID)
    {
        // 현재 키프레임 인덱스
        int loopKeyCur = copy_cloudKeyPoses3D->size() - 1;
        int loopKeyPre = -1;

        // check loop constraint added before
        // 현재 프레임에 대한 닫힌 루프 매칭이 이미 추가되었습니다. 추가하지 않습니다.
        auto it = loopIndexContainer.find(loopKeyCur);
        if (it != loopIndexContainer.end())
            return false;

        // find the closest history key frame
        // 현재 키프레임과 거리가 가장 가까운 키프레임을 찾습니다.
        std::vector<int> pointSearchIndLoop;
        std::vector<float> pointSearchSqDisLoop;
        kdtreeHistoryKeyPoses->setInputCloud(copy_cloudKeyPoses3D);
        kdtreeHistoryKeyPoses->radiusSearch(copy_cloudKeyPoses3D->back(), historyKeyframeSearchRadius, pointSearchIndLoop, pointSearchSqDisLoop, 0);
        
        // 후보 키프레임 세트에서 현재 프레임과의 시간 차이가 큰 프레임을 찾아 닫힌 루프 매칭 프레임으로 선택합니다.
        for (int i = 0; i < (int)pointSearchIndLoop.size(); ++i)
        {
            int id = pointSearchIndLoop[i];
            if (abs(copy_cloudKeyPoses6D->points[id].time - timeLaserInfoCur) > historyKeyframeSearchTimeDiff)
            {
                loopKeyPre = id;
                break;
            }
        }

        if (loopKeyPre == -1 || loopKeyCur == loopKeyPre)
            return false;

        *latestID = loopKeyCur;
        *closestID = loopKeyPre;

        return true;
    }

    bool detectLoopClosureExternal(int *latestID, int *closestID)
    {
        // this function is not used yet, please ignore it
        int loopKeyCur = -1;
        int loopKeyPre = -1;

        std::lock_guard<std::mutex> lock(mtxLoopInfo);
        if (loopInfoVec.empty())
            return false;

        double loopTimeCur = loopInfoVec.front().data[0];
        double loopTimePre = loopInfoVec.front().data[1];
        loopInfoVec.pop_front();

        if (abs(loopTimeCur - loopTimePre) < historyKeyframeSearchTimeDiff)
            return false;

        int cloudSize = copy_cloudKeyPoses6D->size();
        if (cloudSize < 2)
            return false;

        // latest key
        loopKeyCur = cloudSize - 1;
        for (int i = cloudSize - 1; i >= 0; --i)
        {
            if (copy_cloudKeyPoses6D->points[i].time >= loopTimeCur)
                loopKeyCur = round(copy_cloudKeyPoses6D->points[i].intensity);
            else
                break;
        }

        // previous key
        loopKeyPre = 0;
        for (int i = 0; i < cloudSize; ++i)
        {
            if (copy_cloudKeyPoses6D->points[i].time <= loopTimePre)
                loopKeyPre = round(copy_cloudKeyPoses6D->points[i].intensity);
            else
                break;
        }

        if (loopKeyCur == loopKeyPre)
            return false;

        auto it = loopIndexContainer.find(loopKeyCur);
        if (it != loopIndexContainer.end())
            return false;

        *latestID = loopKeyCur;
        *closestID = loopKeyPre;

        return true;
    }


    /**
     * 주어진 키프레임 인덱스의 앞뒤로 몇 프레임의 키프레임 특징점을 추출하고 다운샘플링합니다.
    */
    void loopFindNearKeyframes(pcl::PointCloud<PointType>::Ptr& nearKeyframes, const int& key, const int& searchNum)
    {
        // extract near keyframes
        // 주어진 키프레임 인덱스의 앞뒤로 몇 프레임의 키프레임 특징점을 추출합니다.
        nearKeyframes->clear();
        int cloudSize = copy_cloudKeyPoses6D->size();
        for (int i = -searchNum; i <= searchNum; ++i)
        {
            int keyNear = key + i;
            if (keyNear < 0 || keyNear >= cloudSize )
                continue;
            *nearKeyframes += *transformPointCloud(cornerCloudKeyFrames[keyNear], &copy_cloudKeyPoses6D->points[keyNear]);
            *nearKeyframes += *transformPointCloud(surfCloudKeyFrames[keyNear],   &copy_cloudKeyPoses6D->points[keyNear]);
        }

        if (nearKeyframes->empty())
            return;

        // downsample near keyframes
        // 다운샘플링
        pcl::PointCloud<PointType>::Ptr cloud_temp(new pcl::PointCloud<PointType>());
        downSizeFilterICP.setInputCloud(nearKeyframes);
        downSizeFilterICP.filter(*cloud_temp);
        *nearKeyframes = *cloud_temp;
    }

    void visualizeLoopClosure()
    {
        if (loopIndexContainer.empty())
            return;

        visualization_msgs::msg::MarkerArray markerArray;
        // loop nodes
        visualization_msgs::msg::Marker markerNode;
        markerNode.header.frame_id = odometryFrame;
        markerNode.header.stamp = timeLaserInfoStamp;
        markerNode.action = visualization_msgs::msg::Marker::ADD;
        markerNode.type = visualization_msgs::msg::Marker::SPHERE_LIST;
        markerNode.ns = "loop_nodes";
        markerNode.id = 0;
        markerNode.pose.orientation.w = 1;
        markerNode.scale.x = 0.3; markerNode.scale.y = 0.3; markerNode.scale.z = 0.3; 
        markerNode.color.r = 0; markerNode.color.g = 0.8; markerNode.color.b = 1;
        markerNode.color.a = 1;
        // loop edges
        visualization_msgs::msg::Marker markerEdge;
        markerEdge.header.frame_id = odometryFrame;
        markerEdge.header.stamp = timeLaserInfoStamp;
        markerEdge.action = visualization_msgs::msg::Marker::ADD;
        markerEdge.type = visualization_msgs::msg::Marker::LINE_LIST;
        markerEdge.ns = "loop_edges";
        markerEdge.id = 1;
        markerEdge.pose.orientation.w = 1;
        markerEdge.scale.x = 0.1;
        markerEdge.color.r = 0.9; markerEdge.color.g = 0.9; markerEdge.color.b = 0;
        markerEdge.color.a = 1;

        for (auto it = loopIndexContainer.begin(); it != loopIndexContainer.end(); ++it)
        {
            int key_cur = it->first;
            int key_pre = it->second;
            geometry_msgs::msg::Point p;
            p.x = copy_cloudKeyPoses6D->points[key_cur].x;
            p.y = copy_cloudKeyPoses6D->points[key_cur].y;
            p.z = copy_cloudKeyPoses6D->points[key_cur].z;
            markerNode.points.push_back(p);
            markerEdge.points.push_back(p);
            p.x = copy_cloudKeyPoses6D->points[key_pre].x;
            p.y = copy_cloudKeyPoses6D->points[key_pre].y;
            p.z = copy_cloudKeyPoses6D->points[key_pre].z;
            markerNode.points.push_back(p);
            markerEdge.points.push_back(p);
        }

        markerArray.markers.push_back(markerNode);
        markerArray.markers.push_back(markerEdge);
        pubLoopConstraintEdge->publish(markerArray);
    }







    


    /**
     * 현재 프레임 자세 초기화
     * 1. 첫 번째 프레임인 경우, 원시 IMU 데이터의 RPY를 사용하여 현재 프레임 자세를 초기화합니다 (회전 부분).
     * 2. 이후의 프레임에서는 IMU 오도미터를 사용하여 두 프레임 간의 증분 자세 변환을 계산하고 이를 이전 프레임의 레이저 자세에 적용하여 현재 프레임 레이저 자세를 얻습니다.
    */
    void updateInitialGuess()
    {
        // save current transformation before any processing
        // 이전 프레임의 자세, 여기서는 LIDAR의 자세를 의미하며 이후에는 자세로 간략히 표시합니다.
        incrementalOdometryAffineFront = trans2Affine3f(transformTobeMapped);

        // 이전 프레임의 초기 IMU 변환 (RPY는 IMU의 원시 데이터에서 가져옴) - 첫 번째 프레임이 아닌 경우에만 사용됨
        static Eigen::Affine3f lastImuTransformation;
        // initialization
        // 키프레임 세트가 비어 있는 경우 초기화 계속 진행
        if (cloudKeyPoses3D->points.empty())
        {   
            // 현재 프레임의 자세의 회전 부분을 초기화합니다. 이 값은 IMU의 원시 데이터에서 얻은 RPY입니다.
            transformTobeMapped[0] = cloudInfo.imu_roll_init;
            transformTobeMapped[1] = cloudInfo.imu_pitch_init;
            transformTobeMapped[2] = cloudInfo.imu_yaw_init;

            // IMU 헤딩 초기화를 사용하지 않는 경우, Yaw 값을 0으로 설정합니다.
            if (!useImuHeadingInitialization)
                transformTobeMapped[2] = 0;

            // 초기 IMU 변환을 저장하고 반환합니다. 
            lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imu_roll_init, cloudInfo.imu_pitch_init, cloudInfo.imu_yaw_init); // save imu before return;
            return;
        }

        // use imu pre-integration estimation for pose guess
        // 현재 프레임과 해당하는 이전 프레임의 IMU 오도미터를 사용하여 상대적인 자세 변환을 계산하고, 이전 프레임의 자세에 이를 적용하여 현재 프레임의 자세를 얻습니다.
        static bool lastImuPreTransAvailable = false;
        static Eigen::Affine3f lastImuPreTransformation;
        // IMU 오도미터 정보가 있는 경우
        if (cloudInfo.odom_available == true)
        {
            // 현재 프레임의 초기 추정 자세 (IMU 오도미터에서 얻음)
            Eigen::Affine3f transBack = pcl::getTransformation(
                cloudInfo.initial_guess_x, cloudInfo.initial_guess_y, cloudInfo.initial_guess_z,
                cloudInfo.initial_guess_roll, cloudInfo.initial_guess_pitch, cloudInfo.initial_guess_yaw);
            // 이전 IMU 변환 정보가 사용 가능하지 않은 경우, 현재 IMU 변환 정보를 저장하고 반환합니다.
            if (lastImuPreTransAvailable == false)
            {
                lastImuPreTransformation = transBack;
                lastImuPreTransAvailable = true;
            } else {
                // 현재 프레임과 이전 프레임 간의 상대적인 자세 변환 (IMU 오도미터에 의해 계산됨)
                Eigen::Affine3f transIncre = lastImuPreTransformation.inverse() * transBack;
                // 이전 프레임의 자세
                Eigen::Affine3f transTobe = trans2Affine3f(transformTobeMapped);
                // 현재 프레임의 자세
                Eigen::Affine3f transFinal = transTobe * transIncre;
                // 현재 프레임 자세를 업데이트합니다.
                pcl::getTranslationAndEulerAngles(transFinal, transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5], 
                                                              transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);

                // 이전 IMU 변환 정보를 저장합니다.
                lastImuPreTransformation = transBack;

                // IMU 변환 정보를 저장하고 반환합니다.
                lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imu_roll_init, cloudInfo.imu_pitch_init, cloudInfo.imu_yaw_init); // save imu before return;
                return;
            }
        }

        // use imu incremental estimation for pose guess (only rotation)
        // 첫 번째 프레임에서만 호출되며, IMU 데이터를 사용하여 현재 프레임 자세를 초기화합니다. 여기서는 회전 부분만 초기화합니다.
        if (cloudInfo.imu_available == true)
        {
            // 현재 프레임의 자세 각도 (원시 IMU 데이터에서 가져옴)
            Eigen::Affine3f transBack = pcl::getTransformation(0, 0, 0, cloudInfo.imu_roll_init, cloudInfo.imu_pitch_init, cloudInfo.imu_yaw_init);
            // 현재 프레임과 이전 프레임 간의 상대적인 자세 변환
            Eigen::Affine3f transIncre = lastImuTransformation.inverse() * transBack;

            // 이전 프레임의 자세
            Eigen::Affine3f transTobe = trans2Affine3f(transformTobeMapped);
            // 현재 프레임의 자세
            Eigen::Affine3f transFinal = transTobe * transIncre;
            // 현재 프레임 자세를 업데이트합니다.
            pcl::getTranslationAndEulerAngles(transFinal, transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5], 
                                                          transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);

            // IMU 변환 정보를 저장하고 반환합니다.
            lastImuTransformation = pcl::getTransformation(0, 0, 0, cloudInfo.imu_roll_init, cloudInfo.imu_pitch_init, cloudInfo.imu_yaw_init); // save imu before return;
            return;
        }
    }

    void extractForLoopClosure()
    {
        pcl::PointCloud<PointType>::Ptr cloudToExtract(new pcl::PointCloud<PointType>());
        int numPoses = cloudKeyPoses3D->size();
        for (int i = numPoses-1; i >= 0; --i)
        {
            if ((int)cloudToExtract->size() <= surroundingKeyframeSize)
                cloudToExtract->push_back(cloudKeyPoses3D->points[i]);
            else
                break;
        }

        extractCloud(cloudToExtract);
    }

    /**
     * 주변 지점에서의 키포즈 및 지점을 추출하고 로컬 맵에 추가합니다.
     * 1. 가장 최근의 키포즈 키프레임에 대해 시공간 상에서 인접한 키포즈 프레임 세트를 검색하고 다운샘플링합니다.
     * 2. 검색된 각 키포즈 프레임에서 해당하는 코너 및 평면 포인트를 추출하여 로컬 맵에 추가합니다.
    */
    void extractNearby()
    {
        pcl::PointCloud<PointType>::Ptr surroundingKeyPoses(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr surroundingKeyPosesDS(new pcl::PointCloud<PointType>());
        std::vector<int> pointSearchInd;
        std::vector<float> pointSearchSqDis;

        // extract all the nearby key poses and downsample them
        // kdtree의 입력으로 전역 키포즈 포인트 클라우드를 설정합니다.
        kdtreeSurroundingKeyPoses->setInputCloud(cloudKeyPoses3D); // create kd-tree
        // 최근의 키포즈 프레임을 기준으로 반경 내에서 시공간에 대한 이웃 키포즈 프레임 세트를 검색합니다.
        kdtreeSurroundingKeyPoses->radiusSearch(cloudKeyPoses3D->back(), (double)surroundingKeyframeSearchRadius, pointSearchInd, pointSearchSqDis);
        for (int i = 0; i < (int)pointSearchInd.size(); ++i)
        {
            int id = pointSearchInd[i];
            // 주변 키포즈 집합에 추가합니다.
            surroundingKeyPoses->push_back(cloudKeyPoses3D->points[id]);
        }

        // 다운샘플링합니다.
        downSizeFilterSurroundingKeyPoses.setInputCloud(surroundingKeyPoses);
        downSizeFilterSurroundingKeyPoses.filter(*surroundingKeyPosesDS);
        for(auto& pt : surroundingKeyPosesDS->points)
        {
            kdtreeSurroundingKeyPoses->nearestKSearch(pt, 1, pointSearchInd, pointSearchSqDis);
            pt.intensity = cloudKeyPoses3D->points[pointSearchInd[0]].intensity;
        }

        // also extract some latest key frames in case the robot rotates in one position
        // 시간상으로 가까운 몇몇 키포즈도 추가합니다. 예를 들어, 차량이 원형 회전하는 경우 이러한 프레임을 추가하는 것이 합리적입니다.
        int numPoses = cloudKeyPoses3D->size();
        for (int i = numPoses-1; i >= 0; --i)
        {
            if (timeLaserInfoCur - cloudKeyPoses6D->points[i].time < 10.0)
                surroundingKeyPosesDS->push_back(cloudKeyPoses3D->points[i]);
            else
                break;
        }

        // 추출된 주변 키포즈 세트에 해당하는 코너 및 평면 포인트를 로컬 맵에 추가하여 scan-to-map 매칭에 사용될 로컬 포인트 클라우드 맵을 생성합니다.
        extractCloud(surroundingKeyPosesDS);
    }
    

    /**
     * 이웃한 키포즈 집합에 해당하는 코너 및 평면 포인트를 로컬 맵에 추가하고, scan-to-map 매칭에 사용될 로컬 포인트 클라우드 맵을 생성합니다.
    */
    void extractCloud(pcl::PointCloud<PointType>::Ptr cloudToExtract)
    {
        // fuse the map
        // 이웃한 키포즈 집합에 해당하는 코너 및 평면 포인트를 로컬 맵에 추가합니다.
        laserCloudCornerFromMap->clear();
        laserCloudSurfFromMap->clear(); 
        // 현재 프레임 (실제로는 가장 가까운 키포즈를 사용하여 해당 키포즈의 이웃을 찾음) 에 대한 시간 및 공간 차원상의 이웃 키포즈 집합을 반복합니다.
        for (int i = 0; i < (int)cloudToExtract->size(); ++i)
        {
            // 설정한 거리보다 크면 제외합니다.
            if (pointDistance(cloudToExtract->points[i], cloudKeyPoses3D->back()) > surroundingKeyframeSearchRadius)
                continue;

            // 이웃 키포즈 인덱스
            int thisKeyInd = (int)cloudToExtract->points[i].intensity;
            if (laserCloudMapContainer.find(thisKeyInd) != laserCloudMapContainer.end()) 
            {
                // transformed cloud available
                *laserCloudCornerFromMap += laserCloudMapContainer[thisKeyInd].first;
                *laserCloudSurfFromMap   += laserCloudMapContainer[thisKeyInd].second;
            } else {
                // transformed cloud not available
                // 이웃 키포즈에 해당하는 코너 및 평면 포인트 클라우드를 6D 포즈로 변환하여 월드 좌표계로 가져옵니다.
                pcl::PointCloud<PointType> laserCloudCornerTemp = *transformPointCloud(cornerCloudKeyFrames[thisKeyInd],  &cloudKeyPoses6D->points[thisKeyInd]);
                pcl::PointCloud<PointType> laserCloudSurfTemp = *transformPointCloud(surfCloudKeyFrames[thisKeyInd],    &cloudKeyPoses6D->points[thisKeyInd]);
                // 로컬 맵에 추가합니다.
                *laserCloudCornerFromMap += laserCloudCornerTemp;
                *laserCloudSurfFromMap   += laserCloudSurfTemp;
                laserCloudMapContainer[thisKeyInd] = make_pair(laserCloudCornerTemp, laserCloudSurfTemp);
            }
            
        }

        // Downsample the surrounding corner key frames (or map)
        // 로컬 코너 맵 다운샘플링
        downSizeFilterCorner.setInputCloud(laserCloudCornerFromMap);
        downSizeFilterCorner.filter(*laserCloudCornerFromMapDS);
        laserCloudCornerFromMapDSNum = laserCloudCornerFromMapDS->size();
        // Downsample the surrounding surf key frames (or map)
        // 로컬 서피스 맵 다운샘플링
        downSizeFilterSurf.setInputCloud(laserCloudSurfFromMap);
        downSizeFilterSurf.filter(*laserCloudSurfFromMapDS);
        laserCloudSurfFromMapDSNum = laserCloudSurfFromMapDS->size();

        // clear map cache if too large
        // 메모리가 너무 크면 지웁니다.
        if (laserCloudMapContainer.size() > 1000)
            laserCloudMapContainer.clear();
    }


    /**
     * 이웃한 키포즈의 코너 및 평면 포인트 클라우드를 추출하여 로컬 맵에 추가합니다.
     * 1. 최근 키포즈 프레임에 대해 시공간상에서 인접한 키포즈 프레임 세트를 검색하고 다운샘플링합니다.
     * 2. 검색된 각 키포즈 프레임에서 해당하는 코너 및 평면 포인트를 추출하여 로컬 맵에 추가합니다.
    */
    void extractSurroundingKeyFrames()
    {
        if (cloudKeyPoses3D->points.empty() == true)
            return; 
        
        // if (loopClosureEnableFlag == true)
        // {
        //     extractForLoopClosure();    
        // } else {
        //     extractNearby();
        // }

        // 로컬 코너 및 평면 포인트 클라우드를 추출하고 로컬 맵에 추가합니다.
        // 1. 최근의 키포즈 프레임에 대해 시공간상에서 인접한 키포즈 프레임 세트를 검색하고 다운샘플링합니다.
        // 2. 검색된 각 키포즈 프레임에서 해당하는 코너 및 평면 포인트를 추출하여 로컬 맵에 추가합니다.
        extractNearby();
    }

    void downsampleCurrentScan()
    {
        // Downsample cloud from current scan
        laserCloudCornerLastDS->clear();
        downSizeFilterCorner.setInputCloud(laserCloudCornerLast);
        downSizeFilterCorner.filter(*laserCloudCornerLastDS);
        laserCloudCornerLastDSNum = laserCloudCornerLastDS->size();

        laserCloudSurfLastDS->clear();
        downSizeFilterSurf.setInputCloud(laserCloudSurfLast);
        downSizeFilterSurf.filter(*laserCloudSurfLastDS);
        laserCloudSurfLastDSNum = laserCloudSurfLastDS->size();
    }

    void updatePointAssociateToMap()
    {
        transPointAssociateToMap = trans2Affine3f(transformTobeMapped);
    }
    

    /**
     * 현재 레이저 프레임 코너 포인트를 로컬 맵 매칭 포인트로 찾습니다.
     * 1. 현재 프레임 위치를 업데이트하고, 현재 프레임 코너 포인트 좌표를 맵 좌표계로 변환한 후, 로컬 맵에서 5개의 가장 가까운 포인트를 찾습니다.
     *    이 포인트들은 1m 이내에 있어야 하고, 5개의 포인트가 선을 이루어야 합니다. (중심점으로부터의 거리의 공분산 행렬과 특이값을 사용하여 판단)
     * 2. 현재 프레임 코너 포인트에서 찾은 선까지의 거리와 수직 단위 벡터를 계산하여 코너 매개변수로 저장합니다.
    */
    void cornerOptimization()
    {
        // 현재 프레임 위치를 업데이트합니다.
        updatePointAssociateToMap();

        // 현재 프레임 코너 포인트를 순회합니다.
        #pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < laserCloudCornerLastDSNum; i++)
        {
            PointType pointOri, pointSel, coeff;
            std::vector<int> pointSearchInd;
            std::vector<float> pointSearchSqDis;
            // 코너 포인트 (여전히 lidar 좌표계)
            pointOri = laserCloudCornerLastDS->points[i];
            // 현재 프레임 위치에 따라 맵 좌표계로 변환합니다.
            pointAssociateToMap(&pointOri, &pointSel);
            // 로컬 코너 맵에서 현재 코너 포인트에 가장 가까운 5개의 코너 포인트를 찾습니다.
            kdtreeCornerFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

            cv::Mat matA1(3, 3, CV_32F, cv::Scalar::all(0));
            cv::Mat matD1(1, 3, CV_32F, cv::Scalar::all(0));
            cv::Mat matV1(3, 3, CV_32F, cv::Scalar::all(0));
                    
            // 거리가 1m보다 작아야 합니다.
            if (pointSearchSqDis[4] < 1.0) {
                // 5개 포인트의 평균 좌표를 계산하여 중심점으로 사용합니다.
                float cx = 0, cy = 0, cz = 0;
                for (int j = 0; j < 5; j++) {
                    cx += laserCloudCornerFromMapDS->points[pointSearchInd[j]].x;
                    cy += laserCloudCornerFromMapDS->points[pointSearchInd[j]].y;
                    cz += laserCloudCornerFromMapDS->points[pointSearchInd[j]].z;
                }
                cx /= 5; cy /= 5;  cz /= 5;

                // 공분산을 계산합니다.
                float a11 = 0, a12 = 0, a13 = 0, a22 = 0, a23 = 0, a33 = 0;
                for (int j = 0; j < 5; j++) {
                    float ax = laserCloudCornerFromMapDS->points[pointSearchInd[j]].x - cx;
                    float ay = laserCloudCornerFromMapDS->points[pointSearchInd[j]].y - cy;
                    float az = laserCloudCornerFromMapDS->points[pointSearchInd[j]].z - cz;

                    a11 += ax * ax; a12 += ax * ay; a13 += ax * az;
                    a22 += ay * ay; a23 += ay * az;
                    a33 += az * az;
                }
                a11 /= 5; a12 /= 5; a13 /= 5; a22 /= 5; a23 /= 5; a33 /= 5;

                // 공분산 행렬을 구성합니다.
                matA1.at<float>(0, 0) = a11; matA1.at<float>(0, 1) = a12; matA1.at<float>(0, 2) = a13;
                matA1.at<float>(1, 0) = a12; matA1.at<float>(1, 1) = a22; matA1.at<float>(1, 2) = a23;
                matA1.at<float>(2, 0) = a13; matA1.at<float>(2, 1) = a23; matA1.at<float>(2, 2) = a33;

                // 특이값 분해를 수행합니다.
                cv::eigen(matA1, matD1, matV1);
                // 만약 최대 특징값이 다음으로 큰 특징값보다 훨씬 크다면, 이는 선을 형성한다고 간주합니다.
                if (matD1.at<float>(0, 0) > 3 * matD1.at<float>(0, 1)) {
                    // 현재 프레임의 코너 좌표 (맵 좌표계)
                    float x0 = pointSel.x;
                    float y0 = pointSel.y;
                    float z0 = pointSel.z;
                    // 로컬 맵에 해당하는 중심 코너, 특징 벡터 (선의 방향)을 따라 전방 및 후방으로 각각 가져옵니다.
                    float x1 = cx + 0.1 * matV1.at<float>(0, 0);
                    float y1 = cy + 0.1 * matV1.at<float>(0, 1);
                    float z1 = cz + 0.1 * matV1.at<float>(0, 2);
                    float x2 = cx - 0.1 * matV1.at<float>(0, 0);
                    float y2 = cy - 0.1 * matV1.at<float>(0, 1);
                    float z2 = cz - 0.1 * matV1.at<float>(0, 2);

                    // area_012, 세 점으로 이루어진 삼각형의 면적 * 2, 외적의 크기 |axb|=a*b*sin(theta)
                    float a012 = sqrt(((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) * ((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                                    + ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) * ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) 
                                    + ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1)) * ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1)));

                    // line_12, 기본 변 길이
                    float l12 = sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));

                    // 두 번의 외적을 통해 점에서 선까지의 수직 단위 벡터, x 성분, 이하 동일
                    float la = ((y1 - y2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                              + (z1 - z2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))) / a012 / l12;

                    float lb = -((x1 - x2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                               - (z1 - z2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;

                    float lc = -((x1 - x2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) 
                               + (y1 - y2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;

                    // 삼각형의 높이, 즉 점에서 선까지의 거리
                    float ld2 = a012 / l12;

                    // 거리가 멀어지면 s가 작아지는 거리 페널티 요소 (가중치)
                    float s = 1 - 0.9 * fabs(ld2);

                    // 점에서 선까지의 수직 단위 벡터
                    coeff.x = s * la;
                    coeff.y = s * lb;
                    coeff.z = s * lc;
                    // 점에서 선까지의 거리
                    coeff.intensity = s * ld2;

                    if (s > 0.1) {
                        // 현재 레이저 프레임 코너를 매칭 세트에 추가합니다.
                        laserCloudOriCornerVec[i] = pointOri;
                        // 코너 매개 변수
                        coeffSelCornerVec[i] = coeff;
                        laserCloudOriCornerFlag[i] = true;
                    }
                }
            }
        }
    }



    /**
     * 현재 레이저 프레임의 서피스 포인트를 로컬 맵 매칭 포인트로 찾습니다.
     * 1. 현재 프레임 위치를 업데이트하고, 현재 프레임 서피스 포인트 좌표를 맵 좌표계로 변환한 후, 로컬 맵에서 5개의 가장 가까운 포인트를 찾습니다.
     *    이 포인트들은 1m 이내에 있어야 하고, 5개의 포인트가 평면을 이루어야 합니다. (최소 자승 평면을 이루는데 사용) 그렇지 않으면 매칭 실패로 간주합니다.
     * 2. 현재 프레임 서피스 포인트에서 찾은 평면까지의 거리와 수직 단위 벡터를 계산하여 서피스 포인트의 매개변수로 저장합니다.
    */
    void surfOptimization()
    {
        // 현재 프레임 위치를 업데이트합니다.
        updatePointAssociateToMap();

        // 현재 프레임 서피스 포인트를 순회합니다.
        #pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < laserCloudSurfLastDSNum; i++)
        {
            PointType pointOri, pointSel, coeff;
            std::vector<int> pointSearchInd;
            std::vector<float> pointSearchSqDis;

            // 서피스 포인트 (여전히 lidar 좌표계)
            pointOri = laserCloudSurfLastDS->points[i];
            // 현재 프레임 위치에 따라 맵 좌표계로 변환합니다.
            pointAssociateToMap(&pointOri, &pointSel); 
            // 로컬 서피스 포인트 맵에서 현재 서피스 포인트에 가장 가까운 5개의 서피스 포인트를 찾습니다.
            kdtreeSurfFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);

            Eigen::Matrix<float, 5, 3> matA0;
            Eigen::Matrix<float, 5, 1> matB0;
            Eigen::Vector3f matX0;

            matA0.setZero();
            matB0.fill(-1);
            matX0.setZero();

            // 거리가 1m보다 작아야 합니다.
            if (pointSearchSqDis[4] < 1.0) {
                for (int j = 0; j < 5; j++) {
                    matA0(j, 0) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].x;
                    matA0(j, 1) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].y;
                    matA0(j, 2) = laserCloudSurfFromMapDS->points[pointSearchInd[j]].z;
                }

                // 평면을 형성하는데 사용되는 최소 자승 평면을 계산합니다.
                matX0 = matA0.colPivHouseholderQr().solve(matB0);

                // 평면 방정식의 계수, 동시에 평면의 법선 벡터의 구성 요소입니다.
                float pa = matX0(0, 0);
                float pb = matX0(1, 0);
                float pc = matX0(2, 0);
                float pd = 1;
                
                // 단위 법선 벡터
                float ps = sqrt(pa * pa + pb * pb + pc * pc);
                pa /= ps; pb /= ps; pc /= ps; pd /= ps;

                // 평면이 적절한지 확인합니다. 5개 포인트 중 하나라도 평면에서의 거리가 0.2m를 초과하면 너무 흩어져 있어서 평면을 이루지 않는 것으로 간주합니다.
                bool planeValid = true;
                for (int j = 0; j < 5; j++) {
                    if (fabs(pa * laserCloudSurfFromMapDS->points[pointSearchInd[j]].x +
                             pb * laserCloudSurfFromMapDS->points[pointSearchInd[j]].y +
                             pc * laserCloudSurfFromMapDS->points[pointSearchInd[j]].z + pd) > 0.2) {
                        planeValid = false;
                        break;
                    }
                }

                // 평면이 적절하면
                if (planeValid) {
                    // 현재 레이저 프레임 포인트에서 평면까지의 거리를 계산합니다.
                    float pd2 = pa * pointSel.x + pb * pointSel.y + pc * pointSel.z + pd;

                    float s = 1 - 0.9 * fabs(pd2) / sqrt(sqrt(pointOri.x * pointOri.x
                            + pointOri.y * pointOri.y + pointOri.z * pointOri.z));
                    
                    // 포인트에서 평면으로 수직인 단위 벡터 (사실상 평면의 법선 벡터와 동일)
                    coeff.x = s * pa;
                    coeff.y = s * pb;
                    coeff.z = s * pc;

                    // 포인트에서 평면까지의 거리
                    coeff.intensity = s * pd2;

                    if (s > 0.1) {
                        // 현재 레이저 프레임 서피스 포인트를 매칭 세트에 추가합니다.
                        laserCloudOriSurfVec[i] = pointOri;
                        // 평면 매개변수
                        coeffSelSurfVec[i] = coeff;
                        laserCloudOriSurfFlag[i] = true;
                    }
                }
            }
        }
    }

    // 현재 프레임에서 로컬 맵과 일치하는 코너 및 평면 점을 추출하여 동일한 세트에 추가합니다.
    void combineOptimizationCoeffs()
    {
        // combine corner coeffs
        // 현재 프레임 코너 집합을 반복하면서 로컬 맵과 일치하는 코너를 추출합니다.
        for (int i = 0; i < laserCloudCornerLastDSNum; ++i){
            if (laserCloudOriCornerFlag[i] == true){
                laserCloudOri->push_back(laserCloudOriCornerVec[i]);
                coeffSel->push_back(coeffSelCornerVec[i]);
            }
        }
        // combine surf coeffs
        // 현재 프레임의 서피스 포인트 세트를 반복하면서 로컬 맵과 일치하는 서피스 포인트를 추출합니다.
        for (int i = 0; i < laserCloudSurfLastDSNum; ++i){
            if (laserCloudOriSurfFlag[i] == true){
                laserCloudOri->push_back(laserCloudOriSurfVec[i]);
                coeffSel->push_back(coeffSelSurfVec[i]);
            }
        }
        // reset flag for next iteration
        // 플래그를 지우기
        std::fill(laserCloudOriCornerFlag.begin(), laserCloudOriCornerFlag.end(), false);
        std::fill(laserCloudOriSurfFlag.begin(), laserCloudOriSurfFlag.end(), false);
    }


    /**
     * 스캔 투 맵 최적화
     * 일치하는 특징 점에 대한 자코비안 행렬을 계산하고, 관측 값은 특징 점에서 선 또는 평면까지의 거리이며, 가우스 뉴턴 방정식을 구성하여 현재 포즈를 반복적으로 최적화합니다.
     * 수식 유도: todo
    */
    bool LMOptimization(int iterCount)
    {
        // This optimization is from the original loam_velodyne by Ji Zhang, need to cope with coordinate transformation
        // 현재 코드는 Ji Zhang의 loam_velodyne에서 가져온 최적화 코드로, 좌표 변환을 다루어야 합니다.
        // lidar <- camera      ---     camera <- lidar
        // x = z                ---     x = y
        // y = x                ---     y = z
        // z = y                ---     z = x
        // roll = yaw           ---     roll = pitch
        // pitch = roll         ---     pitch = yaw
        // yaw = pitch          ---     yaw = roll

        // lidar -> camera
        float srx = sin(transformTobeMapped[1]);
        float crx = cos(transformTobeMapped[1]);
        float sry = sin(transformTobeMapped[2]);
        float cry = cos(transformTobeMapped[2]);
        float srz = sin(transformTobeMapped[0]);
        float crz = cos(transformTobeMapped[0]);

        // 현재 프레임 매칭 특징점 수가 너무 적습니다.
        int laserCloudSelNum = laserCloudOri->size();
        if (laserCloudSelNum < 50) {
            return false;
        }

        cv::Mat matA(laserCloudSelNum, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matAt(6, laserCloudSelNum, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtA(6, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matB(laserCloudSelNum, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtB(6, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matX(6, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matP(6, 6, CV_32F, cv::Scalar::all(0));

        PointType pointOri, coeff;

        // 매칭 특징점을 반복하며 Jacobian 행렬을 구성합니다.
        for (int i = 0; i < laserCloudSelNum; i++) {
            // lidar -> camera
            pointOri.x = laserCloudOri->points[i].y;
            pointOri.y = laserCloudOri->points[i].z;
            pointOri.z = laserCloudOri->points[i].x;
            // lidar -> camera
            coeff.x = coeffSel->points[i].y;
            coeff.y = coeffSel->points[i].z;
            coeff.z = coeffSel->points[i].x;
            coeff.intensity = coeffSel->points[i].intensity;
            // in camera
            float arx = (crx*sry*srz*pointOri.x + crx*crz*sry*pointOri.y - srx*sry*pointOri.z) * coeff.x
                      + (-srx*srz*pointOri.x - crz*srx*pointOri.y - crx*pointOri.z) * coeff.y
                      + (crx*cry*srz*pointOri.x + crx*cry*crz*pointOri.y - cry*srx*pointOri.z) * coeff.z;

            float ary = ((cry*srx*srz - crz*sry)*pointOri.x 
                      + (sry*srz + cry*crz*srx)*pointOri.y + crx*cry*pointOri.z) * coeff.x
                      + ((-cry*crz - srx*sry*srz)*pointOri.x 
                      + (cry*srz - crz*srx*sry)*pointOri.y - crx*sry*pointOri.z) * coeff.z;

            float arz = ((crz*srx*sry - cry*srz)*pointOri.x + (-cry*crz-srx*sry*srz)*pointOri.y)*coeff.x
                      + (crx*crz*pointOri.x - crx*srz*pointOri.y) * coeff.y
                      + ((sry*srz + cry*crz*srx)*pointOri.x + (crz*sry-cry*srx*srz)*pointOri.y)*coeff.z;
            // lidar -> camera
            matA.at<float>(i, 0) = arz;
            matA.at<float>(i, 1) = arx;
            matA.at<float>(i, 2) = ary;
            matA.at<float>(i, 3) = coeff.z;
            matA.at<float>(i, 4) = coeff.x;
            matA.at<float>(i, 5) = coeff.y;
            // 점에서 직선까지의 거리, 평면까지의 거리를 관찰 값으로 사용
            matB.at<float>(i, 0) = -coeff.intensity;
        }

        cv::transpose(matA, matAt);
        matAtA = matAt * matA;
        matAtB = matAt * matB;
        // J^T·J·delta_x = -J^T·f를 해결하여 가우스-뉴턴을 수행합니다.
        cv::solve(matAtA, matAtB, matX, cv::DECOMP_QR);

        // 첫 번째 반복에서는 근사 헤시안 행렬(J^T·J)이 특이한지 확인합니다. 행렬식 값=0 등
        if (iterCount == 0) {

            cv::Mat matE(1, 6, CV_32F, cv::Scalar::all(0));
            cv::Mat matV(6, 6, CV_32F, cv::Scalar::all(0));
            cv::Mat matV2(6, 6, CV_32F, cv::Scalar::all(0));

            cv::eigen(matAtA, matE, matV);
            matV.copyTo(matV2);

            isDegenerate = false;
            float eignThre[6] = {100, 100, 100, 100, 100, 100};
            for (int i = 5; i >= 0; i--) {
                if (matE.at<float>(0, i) < eignThre[i]) {
                    for (int j = 0; j < 6; j++) {
                        matV2.at<float>(i, j) = 0;
                    }
                    isDegenerate = true;
                } else {
                    break;
                }
            }
            matP = matV.inv() * matV2;
        }

        if (isDegenerate)
        {
            cv::Mat matX2(6, 1, CV_32F, cv::Scalar::all(0));
            matX.copyTo(matX2);
            matX = matP * matX2;
        }

        // 현재 포즈를 업데이트합니다. x = x + delta_x
        transformTobeMapped[0] += matX.at<float>(0, 0);
        transformTobeMapped[1] += matX.at<float>(1, 0);
        transformTobeMapped[2] += matX.at<float>(2, 0);
        transformTobeMapped[3] += matX.at<float>(3, 0);
        transformTobeMapped[4] += matX.at<float>(4, 0);
        transformTobeMapped[5] += matX.at<float>(5, 0);

        float deltaR = sqrt(
                            pow(pcl::rad2deg(matX.at<float>(0, 0)), 2) +
                            pow(pcl::rad2deg(matX.at<float>(1, 0)), 2) +
                            pow(pcl::rad2deg(matX.at<float>(2, 0)), 2));
        float deltaT = sqrt(
                            pow(matX.at<float>(3, 0) * 100, 2) +
                            pow(matX.at<float>(4, 0) * 100, 2) +
                            pow(matX.at<float>(5, 0) * 100, 2));

        // delta_x가 충분히 작으면 수렴했다고 가정합니다.
        if (deltaR < 0.05 && deltaT < 0.05) {
            return true; // converged
        }
        return false; // keep optimizing
    }


    /**
     * "스캔 투 맵"을 사용하여 현재 프레임의 자세 최적화
     * 1. 현재 프레임의 특징점 수가 충분하고 매칭된 포인트 수가 충분한 경우에만 최적화를 수행합니다.
     * 2. 30번 반복하여 최적화를 수행합니다.
     *    1) 현재 라이다 프레임의 코너 포인트가 로컬 맵 매칭 포인트를 찾습니다.
     *       a. 현재 프레임 자세를 업데이트하고 현재 프레임 코너 포인트 좌표를 맵 좌표로 변환하여 로컬 맵에서 1m 이하의 거리에 있는 5개의 가장 가까운 포인트를 찾습니다. 또한, 이 5개의 포인트가 직선을 형성해야 하며, 중심점으로부터의 거리의 공분산 행렬 및 특징값을 사용하여 판단합니다. 이러한 경우를 일치했다고 간주합니다.
     *       b. 현재 프레임 코너 포인트에서 직선까지의 거리 및 수직선의 단위 벡터를 계산하여 코너 포인트 매개변수로 저장합니다.
     *    2) 현재 라이다 프레임의 평면 포인트가 로컬 맵 매칭 포인트를 찾습니다.
     *       a. 현재 프레임 자세를 업데이트하고 현재 프레임 평면 포인트 좌표를 맵 좌표로 변환하여 로컬 맵에서 1m 이하의 거리에 있는 5개의 가장 가까운 포인트를 찾습니다. 또한, 이 5개의 포인트가 평면을 형성해야 하며, 최소자승법으로 평면을 적합시켜야 합니다. 이러한 경우를 일치했다고 간주합니다.
     *       b. 현재 프레임 평면 포인트에서 평면까지의 거리 및 수직선의 단위 벡터를 계산하여 평면 포인트 매개변수로 저장합니다.
     *    3) 로컬 맵과 일치하는 현재 프레임의 코너 및 평면 포인트를 추출하여 동일한 세트에 추가합니다.
     *    4) 매칭된 특징 포인트에 대해 Jacobian 행렬을 계산하고, 관측값은 포인트에서 직선 또는 평면까지의 거리이며, 가우스 뉴턴 방정식을 구성하여 현재 자세를 반복적으로 최적화합니다. 최종 결과는 transformTobeMapped에 저장됩니다.
     * 3. IMU의 원시 RPY 데이터와 "스캔 투 맵"으로 최적화된 후속 자세를 가중 평균하여 현재 프레임의 자세 롤 및 피치를 업데이트하고, Z 좌표를 제한합니다.
    */
    void scan2MapOptimization()
    {
        // 키프레임이 있을 경우에만 실행
        if (cloudKeyPoses3D->points.empty())
            return;

        // 현재 레이저 프레임의 코너 및 평면 포인트 수가 충분한 경우
        if (laserCloudCornerLastDSNum > edgeFeatureMinValidNum && laserCloudSurfLastDSNum > surfFeatureMinValidNum)
        {
            // kdtree 입력으로 로컬 맵 포인트 클라우드 설정
            kdtreeCornerFromMap->setInputCloud(laserCloudCornerFromMapDS);
            kdtreeSurfFromMap->setInputCloud(laserCloudSurfFromMapDS);

            // 최대 30번 반복
            for (int iterCount = 0; iterCount < 30; iterCount++)
            {
                // 각 반복마다 특징점 집합 지우기
                laserCloudOri->clear();
                coeffSel->clear();

                // 현재 레이저 프레임 코너 포인트를 로컬 맵 매칭 포인트로 찾기
                // 1. 현재 프레임 포즈 업데이트, 현재 프레임 코너 포인트 좌표를 맵 좌표로 변환하고, 로컬 맵에서 1m 이내의 거리에 있고 5개 포인트가 직선을 형성하는 경우 (포인트 중심으로부터의 거리 및 공분산 행렬 및 고유값 사용), 매칭된 것으로 판단
                // 2. 현재 프레임 코너 포인트에서 직선까지의 거리 및 수직 선의 단위 벡터를 계산하고, 코너 포인트 매개 변수로 저장
                cornerOptimization();

                // 현재 레이저 프레임 평면 포인트를 로컬 맵 매칭 포인트로 찾기
                // 1. 현재 프레임 포즈 업데이트, 현재 프레임 평면 포인트 좌표를 맵 좌표로 변환하고, 로컬 맵에서 1m 이내의 거리에 있고 5개 포인트가 평면을 형성하는 경우 (최소 자승 평면을 맞추는 최소자승법), 매칭된 것으로 판단
                // 2. 현재 프레임 평면 포인트에서 평면까지의 거리 및 수직 선의 단위 벡터를 계산하고, 평면 포인트 매개 변수로 저장
                surfOptimization();

                // 현재 프레임에서 로컬 맵에 매칭된 코너 포인트 및 평면 포인트를 추출하여 동일한 집합에 추가
                combineOptimizationCoeffs();

                // 스캔 투 맵 최적화
                // 매칭 특징점에 대한 Jacobian 행렬을 계산하고, 관측 값은 특징점에서 직선 및 평면까지의 거리로, 가우스 뉴턴 방정식을 구성하여 현재 포즈를 반복적으로 최적화하고 transformTobeMapped에 저장          
                if (LMOptimization(iterCount) == true)
                    break;              
            }

            // IMU의 원시 RPY 데이터와 스캔 투 맵으로 최적화된 후속 자세를 가중 평균하여 현재 프레임의 자세 롤 및 피치를 업데이트하고, Z 좌표를 제한
            transformUpdate();
        } else {
            RCLCPP_WARN(get_logger(), "Not enough features! Only %d edge and %d planar features available.", laserCloudCornerLastDSNum, laserCloudSurfLastDSNum);
        }
    }


    /**
     * IMU의 원시 RPY 데이터와 "scan-to-map"으로 최적화된 후속 자세를 가중 평균하여 현재 프레임의 자세 롤 및 피치를 업데이트하고, Z 좌표를 제한합니다.
    */
    void transformUpdate()
    {
        if (cloudInfo.imu_available == true)
        {
             // 피치 각도가 1.4보다 작을 때
            if (std::abs(cloudInfo.imu_pitch_init) < 1.4)
            {
                double imuWeight = imuRPYWeight;
                tf2::Quaternion imuQuaternion;
                tf2::Quaternion transformQuaternion;
                double rollMid, pitchMid, yawMid;

                // slerp roll
                // 롤 각도의 가중 평균을 계산하고, "scan-to-map"으로 최적화된 자세 및 IMU 원시 RPY 데이터로 가중 평균을 수행합니다.
                transformQuaternion.setRPY(transformTobeMapped[0], 0, 0);
                imuQuaternion.setRPY(cloudInfo.imu_roll_init, 0, 0);
                tf2::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                transformTobeMapped[0] = rollMid;

                // slerp pitch
                // 피치 각도의 가중 평균을 계산하고, "scan-to-map"으로 최적화된 자세 및 IMU 원시 RPY 데이터로 가중 평균을 수행합니다.
                transformQuaternion.setRPY(0, transformTobeMapped[1], 0);
                imuQuaternion.setRPY(0, cloudInfo.imu_pitch_init, 0);
                tf2::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                transformTobeMapped[1] = pitchMid;
            }
        }

        // 현재 프레임 자세의 롤, 피치, Z 좌표를 업데이트합니다. 자동차이므로 롤 및 피치는 상대적으로 안정적이며 크게 변하지 않습니다. 일부 경우에는 imu의 데이터를 신뢰할 수 있습니다. Z는 높이 제약이 있습니다.
        transformTobeMapped[0] = constraintTransformation(transformTobeMapped[0], rotation_tollerance);
        transformTobeMapped[1] = constraintTransformation(transformTobeMapped[1], rotation_tollerance);
        transformTobeMapped[5] = constraintTransformation(transformTobeMapped[5], z_tollerance);

        // 현재 프레임 자세
        incrementalOdometryAffineBack = trans2Affine3f(transformTobeMapped);
    }

    float constraintTransformation(float value, float limit)
    {
        if (value < -limit)
            value = -limit;
        if (value > limit)
            value = limit;

        return value;
    }


    /**
     * 현재 프레임과 이전 프레임의 자세 변환을 계산하고, 변화가 너무 작으면 키프레임으로 설정하지 않고, 그렇지 않으면 키프레임으로 설정합니다.
    */
    bool saveFrame()
    {
        if (cloudKeyPoses3D->points.empty())
            return true;

        if (sensor == SensorType::LIVOX)
        {
            if (timeLaserInfoCur - cloudKeyPoses6D->back().time > 1.0)
                return true;
        }

        // 이전 프레임 자세
        Eigen::Affine3f transStart = pclPointToAffine3f(cloudKeyPoses6D->back());
        // 현재 프레임 자세
        Eigen::Affine3f transFinal = pcl::getTransformation(transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5], 
                                                            transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
        // 자세 변환 증가
        Eigen::Affine3f transBetween = transStart.inverse() * transFinal;
        float x, y, z, roll, pitch, yaw;
        pcl::getTranslationAndEulerAngles(transBetween, x, y, z, roll, pitch, yaw);

        // 회전 및 이동이 모두 작으면 현재 프레임을 키프레임으로 설정하지 않음
        if (abs(roll)  < surroundingkeyframeAddingAngleThreshold &&
            abs(pitch) < surroundingkeyframeAddingAngleThreshold &&
            abs(yaw)   < surroundingkeyframeAddingAngleThreshold &&
            sqrt(x*x + y*y + z*z) < surroundingkeyframeAddingDistThreshold)
            return false;

        return true;
    }

    void addOdomFactor()
    {
        if (cloudKeyPoses3D->points.empty())
        {
            noiseModel::Diagonal::shared_ptr priorNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-2, 1e-2, M_PI*M_PI, 1e8, 1e8, 1e8).finished()); // rad*rad, meter*meter
            gtSAMgraph.add(PriorFactor<Pose3>(0, trans2gtsamPose(transformTobeMapped), priorNoise));
            initialEstimate.insert(0, trans2gtsamPose(transformTobeMapped));
        }else{
            noiseModel::Diagonal::shared_ptr odometryNoise = noiseModel::Diagonal::Variances((Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
            gtsam::Pose3 poseFrom = pclPointTogtsamPose3(cloudKeyPoses6D->points.back());
            gtsam::Pose3 poseTo   = trans2gtsamPose(transformTobeMapped);
            gtSAMgraph.add(BetweenFactor<Pose3>(cloudKeyPoses3D->size()-1, cloudKeyPoses3D->size(), poseFrom.between(poseTo), odometryNoise));
            initialEstimate.insert(cloudKeyPoses3D->size(), poseTo);
        }
    }

    void addGPSFactor()
    {
        if (gpsQueue.empty())
            return;

        // wait for system initialized and settles down
        if (cloudKeyPoses3D->points.empty())
            return;
        else
        {
            if (pointDistance(cloudKeyPoses3D->front(), cloudKeyPoses3D->back()) < 5.0)
                return;
        }

        // pose covariance small, no need to correct
        if (poseCovariance(3,3) < poseCovThreshold && poseCovariance(4,4) < poseCovThreshold)
            return;

        // last gps position
        static PointType lastGPSPoint;

        while (!gpsQueue.empty())
        {
            if (stamp2Sec(gpsQueue.front().header.stamp) < timeLaserInfoCur - 0.2)
            {
                // message too old
                gpsQueue.pop_front();
            }
            else if (stamp2Sec(gpsQueue.front().header.stamp) > timeLaserInfoCur + 0.2)
            {
                // message too new
                break;
            }
            else
            {
                nav_msgs::msg::Odometry thisGPS = gpsQueue.front();
                gpsQueue.pop_front();

                // GPS too noisy, skip
                float noise_x = thisGPS.pose.covariance[0];
                float noise_y = thisGPS.pose.covariance[7];
                float noise_z = thisGPS.pose.covariance[14];
                if (noise_x > gpsCovThreshold || noise_y > gpsCovThreshold)
                    continue;
                float gps_x = thisGPS.pose.pose.position.x;
                float gps_y = thisGPS.pose.pose.position.y;
                float gps_z = thisGPS.pose.pose.position.z;
                if (!useGpsElevation)
                {
                    gps_z = transformTobeMapped[5];
                    noise_z = 0.01;
                }

                // GPS not properly initialized (0,0,0)
                if (abs(gps_x) < 1e-6 && abs(gps_y) < 1e-6)
                    continue;

                // Add GPS every a few meters
                PointType curGPSPoint;
                curGPSPoint.x = gps_x;
                curGPSPoint.y = gps_y;
                curGPSPoint.z = gps_z;
                if (pointDistance(curGPSPoint, lastGPSPoint) < 5.0)
                    continue;
                else
                    lastGPSPoint = curGPSPoint;

                gtsam::Vector Vector3(3);
                Vector3 << max(noise_x, 1.0f), max(noise_y, 1.0f), max(noise_z, 1.0f);
                noiseModel::Diagonal::shared_ptr gps_noise = noiseModel::Diagonal::Variances(Vector3);
                gtsam::GPSFactor gps_factor(cloudKeyPoses3D->size(), gtsam::Point3(gps_x, gps_y, gps_z), gps_noise);
                gtSAMgraph.add(gps_factor);

                aLoopIsClosed = true;
                break;
            }
        }
    }

    void addLoopFactor()
    {
        if (loopIndexQueue.empty())
            return;

        for (int i = 0; i < (int)loopIndexQueue.size(); ++i)
        {
            int indexFrom = loopIndexQueue[i].first;
            int indexTo = loopIndexQueue[i].second;
            gtsam::Pose3 poseBetween = loopPoseQueue[i];
            gtsam::noiseModel::Diagonal::shared_ptr noiseBetween = loopNoiseQueue[i];
            gtSAMgraph.add(BetweenFactor<Pose3>(indexFrom, indexTo, poseBetween, noiseBetween));
        }

        loopIndexQueue.clear();
        loopPoseQueue.clear();
        loopNoiseQueue.clear();
        aLoopIsClosed = true;
    }


    /**
     * 현재 프레임을 키프레임으로 설정하고, 레이스 팩터 그래프 최적화를 수행합니다.
     * 1. 현재 프레임과 이전 프레임의 자세 변환을 계산하고, 변화가 너무 작으면 키프레임으로 설정하지 않고, 그렇지 않으면 키프레임으로 설정합니다.
     * 2. 레이스 팩터 그래프에 레이더 오도메트리 팩터, GPS 팩터, 루프 팩터를 추가합니다.
     * 3. 레이스 팩터 그래프 최적화를 실행합니다.
     * 4. 현재 프레임의 최적화된 자세 및 자세 공분산을 얻습니다.
     * 5. cloudKeyPoses3D 및 cloudKeyPoses6D를 추가하고, transformTobeMapped을 업데이트하며, 현재 키프레임의 코너 및 서피스 포인트 세트를 추가합니다.
    */
    void saveKeyFramesAndFactor()
    {
        // 1. 현재 프레임과 이전 프레임의 자세 변환을 계산하고, 변화가 너무 작으면 키프레임으로 설정하지 않고, 그렇지 않으면 키프레임으로 설정합니다.
        if (saveFrame() == false)
            return;

        // odom factor
        // 라이다 오도미터(factor) 추가
        addOdomFactor();

        // gps factor
        // GPS(factor) 추가
        addGPSFactor();

        // loop factor
        // 루프 클로저(factor) 추가
        addLoopFactor();

        // cout << "****************************************************" << endl;
        // gtSAMgraph.print("GTSAM Graph:\n");

        // update iSAM
        // 그래프 최적화 수행
        isam->update(gtSAMgraph, initialEstimate);
        isam->update();

        // 루프가 닫혔을 경우 추가 업데이트를 수행하여 최적화
        if (aLoopIsClosed == true)
        {
            isam->update();
            isam->update();
            isam->update();
            isam->update();
            isam->update();
        }

        // 업데이트 후에는 저장된 그래프를 초기화, 주의: 과거 데이터는 지워지지 않으며 ISAM에 저장됨
        gtSAMgraph.resize(0);
        initialEstimate.clear();

        //save key poses
        // 4. 현재 프레임의 최적화된 자세 및 자세 공분산을 얻습니다.
        PointType thisPose3D;
        PointTypePose thisPose6D;
        Pose3 latestEstimate;

        // ISAM을 사용하여 현재 프레임의 최적화 결과 계산
        isamCurrentEstimate = isam->calculateEstimate();
        // 현재 프레임의 최적화된 자세 결과 얻기
        latestEstimate = isamCurrentEstimate.at<Pose3>(isamCurrentEstimate.size()-1);
        // cout << "****************************************************" << endl;
        // isamCurrentEstimate.print("Current estimate: ");

        // cloudKeyPoses3D에 현재 프레임의 자세 추가
        thisPose3D.x = latestEstimate.translation().x();
        thisPose3D.y = latestEstimate.translation().y();
        thisPose3D.z = latestEstimate.translation().z();
        // 인덱스 추가
        thisPose3D.intensity = cloudKeyPoses3D->size(); // this can be used as index
        cloudKeyPoses3D->push_back(thisPose3D);

        // cloudKeyPoses6D에 현재 프레임의 자세 추가
        thisPose6D.x = thisPose3D.x;
        thisPose6D.y = thisPose3D.y;
        thisPose6D.z = thisPose3D.z;
        thisPose6D.intensity = thisPose3D.intensity ; // this can be used as index
        thisPose6D.roll  = latestEstimate.rotation().roll();
        thisPose6D.pitch = latestEstimate.rotation().pitch();
        thisPose6D.yaw   = latestEstimate.rotation().yaw();
        thisPose6D.time = timeLaserInfoCur;
        cloudKeyPoses6D->push_back(thisPose6D);

        // cout << "****************************************************" << endl;
        // cout << "Pose covariance:" << endl;
        // cout << isam->marginalCovariance(isamCurrentEstimate.size()-1) << endl << endl;
        
        // 프레임의 자세에 대한 공분산 계산 및 저장
        poseCovariance = isam->marginalCovariance(isamCurrentEstimate.size()-1);

        // save updated transform
        // transformTobeMapped을 현재 프레임의 자세로 업데이트
        transformTobeMapped[0] = latestEstimate.rotation().roll();
        transformTobeMapped[1] = latestEstimate.rotation().pitch();
        transformTobeMapped[2] = latestEstimate.rotation().yaw();
        transformTobeMapped[3] = latestEstimate.translation().x();
        transformTobeMapped[4] = latestEstimate.translation().y();
        transformTobeMapped[5] = latestEstimate.translation().z();

        // save all the received edge and surf points
        // 현재 프레임의 코너 및 서피스 키포인트를 복사 및 다운샘플링하여 저장
        pcl::PointCloud<PointType>::Ptr thisCornerKeyFrame(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr thisSurfKeyFrame(new pcl::PointCloud<PointType>());
        pcl::copyPointCloud(*laserCloudCornerLastDS,  *thisCornerKeyFrame);
        pcl::copyPointCloud(*laserCloudSurfLastDS,    *thisSurfKeyFrame);

        // save key frame cloud
        // 특징점 다운샘플링된 데이터 저장
        cornerCloudKeyFrames.push_back(thisCornerKeyFrame);
        surfCloudKeyFrames.push_back(thisSurfKeyFrame);

        // save path for visualization
        // ISAM 업데이트 후의 프레임 자세로 경로 업데이트
        updatePath(thisPose6D);
    }




    /**
     * 모든 변수 노드의 자세를 업데이트하여, 모든 과거 키프레임의 자세를 업데이트하고, 오도미터 경로를 업데이트합니다.
     */
    void correctPoses()
    {
        // 키프레임이 없으면 함수 종료
        if (cloudKeyPoses3D->points.empty())
            return;

        // 루프가 닫혔을 경우
        if (aLoopIsClosed == true)
        {
            // clear map cache
            // 로컬 맵을 지우고
            laserCloudMapContainer.clear();
            // clear path
            // 오도미터 경로를 지우고
            globalPath.poses.clear();
            // update key poses
            // 모든 변수 노드의 자세를 업데이트하여, 모든 과거 키프레임의 자세를 업데이트하고, 오도미터 경로를 업데이트합니다.
            int numPoses = isamCurrentEstimate.size();
            for (int i = 0; i < numPoses; ++i)
            {
                // 과거 키프레임의 자세를 업데이트
                cloudKeyPoses3D->points[i].x = isamCurrentEstimate.at<Pose3>(i).translation().x();
                cloudKeyPoses3D->points[i].y = isamCurrentEstimate.at<Pose3>(i).translation().y();
                cloudKeyPoses3D->points[i].z = isamCurrentEstimate.at<Pose3>(i).translation().z();

                cloudKeyPoses6D->points[i].x = cloudKeyPoses3D->points[i].x;
                cloudKeyPoses6D->points[i].y = cloudKeyPoses3D->points[i].y;
                cloudKeyPoses6D->points[i].z = cloudKeyPoses3D->points[i].z;
                cloudKeyPoses6D->points[i].roll  = isamCurrentEstimate.at<Pose3>(i).rotation().roll();
                cloudKeyPoses6D->points[i].pitch = isamCurrentEstimate.at<Pose3>(i).rotation().pitch();
                cloudKeyPoses6D->points[i].yaw   = isamCurrentEstimate.at<Pose3>(i).rotation().yaw();

                // 오도미터 경로 업데이트
                updatePath(cloudKeyPoses6D->points[i]);
            }

            // 루프 클로저 플래그 초기화
            aLoopIsClosed = false;
        }
    }

    void updatePath(const PointTypePose& pose_in)
    {
        geometry_msgs::msg::PoseStamped pose_stamped;
        pose_stamped.header.stamp = rclcpp::Time(pose_in.time * 1e9);
        pose_stamped.header.frame_id = odometryFrame;
        pose_stamped.pose.position.x = pose_in.x;
        pose_stamped.pose.position.y = pose_in.y;
        pose_stamped.pose.position.z = pose_in.z;
        tf2::Quaternion q;
        q.setRPY(pose_in.roll, pose_in.pitch, pose_in.yaw);
        pose_stamped.pose.orientation.x = q.x();
        pose_stamped.pose.orientation.y = q.y();
        pose_stamped.pose.orientation.z = q.z();
        pose_stamped.pose.orientation.w = q.w();

        globalPath.poses.push_back(pose_stamped);
    }

    void publishOdometry()
    {
        // Publish odometry for ROS (global)
        nav_msgs::msg::Odometry laserOdometryROS;
        laserOdometryROS.header.stamp = timeLaserInfoStamp;
        laserOdometryROS.header.frame_id = odometryFrame;
        laserOdometryROS.child_frame_id = "odom_mapping";
        laserOdometryROS.pose.pose.position.x = transformTobeMapped[3];
        laserOdometryROS.pose.pose.position.y = transformTobeMapped[4];
        laserOdometryROS.pose.pose.position.z = transformTobeMapped[5];
        tf2::Quaternion quat_tf;
        quat_tf.setRPY(transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
        geometry_msgs::msg::Quaternion quat_msg;
        tf2::convert(quat_tf, quat_msg);
        laserOdometryROS.pose.pose.orientation = quat_msg;
        pubLaserOdometryGlobal->publish(laserOdometryROS);

        // Publish TF
        quat_tf.setRPY(transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
        tf2::Transform t_odom_to_lidar = tf2::Transform(quat_tf, tf2::Vector3(transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5]));
        tf2::TimePoint time_point = tf2_ros::fromRclcpp(timeLaserInfoStamp);
        tf2::Stamped<tf2::Transform> temp_odom_to_lidar(t_odom_to_lidar, time_point, odometryFrame);
        geometry_msgs::msg::TransformStamped trans_odom_to_lidar;
        tf2::convert(temp_odom_to_lidar, trans_odom_to_lidar);
        trans_odom_to_lidar.child_frame_id = "lidar_link";
        br->sendTransform(trans_odom_to_lidar);

        // Publish odometry for ROS (incremental)
        static bool lastIncreOdomPubFlag = false;
        static nav_msgs::msg::Odometry laserOdomIncremental; // incremental odometry msg
        static Eigen::Affine3f increOdomAffine; // incremental odometry in affine
        if (lastIncreOdomPubFlag == false)
        {
            lastIncreOdomPubFlag = true;
            laserOdomIncremental = laserOdometryROS;
            increOdomAffine = trans2Affine3f(transformTobeMapped);
        } else {
            Eigen::Affine3f affineIncre = incrementalOdometryAffineFront.inverse() * incrementalOdometryAffineBack;
            increOdomAffine = increOdomAffine * affineIncre;
            float x, y, z, roll, pitch, yaw;
            pcl::getTranslationAndEulerAngles (increOdomAffine, x, y, z, roll, pitch, yaw);
            if (cloudInfo.imu_available == true)
            {
                if (std::abs(cloudInfo.imu_pitch_init) < 1.4)
                {
                    double imuWeight = 0.1;
                    tf2::Quaternion imuQuaternion;
                    tf2::Quaternion transformQuaternion;
                    double rollMid, pitchMid, yawMid;

                    // slerp roll
                    transformQuaternion.setRPY(roll, 0, 0);
                    imuQuaternion.setRPY(cloudInfo.imu_roll_init, 0, 0);
                    tf2::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                    roll = rollMid;

                    // slerp pitch
                    transformQuaternion.setRPY(0, pitch, 0);
                    imuQuaternion.setRPY(0, cloudInfo.imu_pitch_init, 0);
                    tf2::Matrix3x3(transformQuaternion.slerp(imuQuaternion, imuWeight)).getRPY(rollMid, pitchMid, yawMid);
                    pitch = pitchMid;
                }
            }
            laserOdomIncremental.header.stamp = timeLaserInfoStamp;
            laserOdomIncremental.header.frame_id = odometryFrame;
            laserOdomIncremental.child_frame_id = "odom_mapping";
            laserOdomIncremental.pose.pose.position.x = x;
            laserOdomIncremental.pose.pose.position.y = y;
            laserOdomIncremental.pose.pose.position.z = z;
            tf2::Quaternion quat_tf;
            quat_tf.setRPY(roll, pitch, yaw);
            geometry_msgs::msg::Quaternion quat_msg;
            tf2::convert(quat_tf, quat_msg);
            laserOdomIncremental.pose.pose.orientation = quat_msg;
            if (isDegenerate)
                laserOdomIncremental.pose.covariance[0] = 1;
            else
                laserOdomIncremental.pose.covariance[0] = 0;
        }
        pubLaserOdometryIncremental->publish(laserOdomIncremental);
    }

    void publishFrames()
    {
        if (cloudKeyPoses3D->points.empty())
            return;
        // publish key poses
        publishCloud(pubKeyPoses, cloudKeyPoses3D, timeLaserInfoStamp, odometryFrame);
        // Publish surrounding key frames
        publishCloud(pubRecentKeyFrames, laserCloudSurfFromMapDS, timeLaserInfoStamp, odometryFrame);
        // publish registered key frame
        if (pubRecentKeyFrame->get_subscription_count() != 0)
        {
            pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
            PointTypePose thisPose6D = trans2PointTypePose(transformTobeMapped);
            *cloudOut += *transformPointCloud(laserCloudCornerLastDS,  &thisPose6D);
            *cloudOut += *transformPointCloud(laserCloudSurfLastDS,    &thisPose6D);
            publishCloud(pubRecentKeyFrame, cloudOut, timeLaserInfoStamp, odometryFrame);
        }
        // publish registered high-res raw cloud
        if (pubCloudRegisteredRaw->get_subscription_count() != 0)
        {
            pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
            pcl::fromROSMsg(cloudInfo.cloud_deskewed, *cloudOut);
            PointTypePose thisPose6D = trans2PointTypePose(transformTobeMapped);
            *cloudOut = *transformPointCloud(cloudOut,  &thisPose6D);
            publishCloud(pubCloudRegisteredRaw, cloudOut, timeLaserInfoStamp, odometryFrame);
        }
        // publish path
        if (pubPath->get_subscription_count() != 0)
        {
            globalPath.header.stamp = timeLaserInfoStamp;
            globalPath.header.frame_id = odometryFrame;
            pubPath->publish(globalPath);
        }
    }
};


int main(int argc, char** argv)
{   
    rclcpp::init(argc, argv);

    rclcpp::NodeOptions options;
    options.use_intra_process_comms(true);
    rclcpp::executors::SingleThreadedExecutor exec;

    auto MO = std::make_shared<mapOptimization>(options);
    exec.add_node(MO);

    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "\033[1;32m----> Map Optimization Started.\033[0m");

    std::thread loopthread(&mapOptimization::loopClosureThread, MO);
    std::thread visualizeMapThread(&mapOptimization::visualizeGlobalMapThread, MO);

    exec.spin();

    rclcpp::shutdown();

    loopthread.join();
    visualizeMapThread.join();

    return 0;
}
