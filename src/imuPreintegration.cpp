#include "utility.hpp"

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
#include <gtsam_unstable/nonlinear/IncrementalFixedLagSmoother.h>

using gtsam::symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)
using gtsam::symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using gtsam::symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)

/**
* TransformFusion 클래스
* MapOptimization의 라이다 오도메트리 및 IMU 오도메트리를 구독하고 이전 순간의 라이다 오도메트리와 
* 해당 순간부터 현재 순간까지의 IMU 오도메트리 변화량을 기반으로 현재 순간의 IMU 오도메트리를 계산합니다. 
* rviz는 IMU 주행 거리계를 표시합니다. 궤적(부분).
*/

class TransformFusion : public ParamServer
{
public:
    std::mutex mtx;

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subImuOdometry;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subLaserOdometry;

    rclcpp::CallbackGroup::SharedPtr callbackGroupImuOdometry;
    rclcpp::CallbackGroup::SharedPtr callbackGroupLaserOdometry;

    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubImuOdometry;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubImuPath;

    Eigen::Isometry3d lidarOdomAffine;
    Eigen::Isometry3d imuOdomAffineFront;
    Eigen::Isometry3d imuOdomAffineBack;

    std::shared_ptr<tf2_ros::Buffer> tfBuffer;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tfBroadcaster;
    std::shared_ptr<tf2_ros::TransformListener> tfListener;
    tf2::Stamped<tf2::Transform> lidar2Baselink;

    double lidarOdomTime = -1;
    deque<nav_msgs::msg::Odometry> imuOdomQueue;

    /**
     * 생성자
    */
    TransformFusion(const rclcpp::NodeOptions & options) : ParamServer("lio_sam_transformFusion", options)
    {
        tfBuffer = std::make_shared<tf2_ros::Buffer>(get_clock());
        tfListener = std::make_shared<tf2_ros::TransformListener>(*tfBuffer);

        callbackGroupImuOdometry = create_callback_group(
            rclcpp::CallbackGroupType::MutuallyExclusive);
        callbackGroupLaserOdometry = create_callback_group(
            rclcpp::CallbackGroupType::MutuallyExclusive);

        auto imuOdomOpt = rclcpp::SubscriptionOptions();
        imuOdomOpt.callback_group = callbackGroupImuOdometry;
        auto laserOdomOpt = rclcpp::SubscriptionOptions();
        laserOdomOpt.callback_group = callbackGroupLaserOdometry;

        // mapOptimization에서 수신한 라이다 오도메트리를 구독
        subLaserOdometry = create_subscription<nav_msgs::msg::Odometry>(
            "lio_sam/mapping/odometry", qos,
            std::bind(&TransformFusion::lidarOdometryHandler, this, std::placeholders::_1),
            laserOdomOpt);
        // IMUPreintegration에서 수신한 IMU 오도메트리를 구독
        subImuOdometry = create_subscription<nav_msgs::msg::Odometry>(
            odomTopic+"_incremental", qos_imu,
            std::bind(&TransformFusion::imuOdometryHandler, this, std::placeholders::_1),
            imuOdomOpt);

        // Rviz에서 표시하기 위해 IMU 오도메트리를 게시
        pubImuOdometry = create_publisher<nav_msgs::msg::Odometry>(odomTopic, qos_imu);
        // IMU 오도메트리 경로를 게시
        pubImuPath = create_publisher<nav_msgs::msg::Path>("lio_sam/imu/path", qos);

        tfBroadcaster = std::make_unique<tf2_ros::TransformBroadcaster>(this);
    }

    /**
     * 오도메트리에 해당하는 변환 행렬
    */
    Eigen::Isometry3d odom2affine(nav_msgs::msg::Odometry odom)
    {
        tf2::Transform t;
        tf2::fromMsg(odom.pose.pose, t);
        return tf2::transformToEigen(tf2::toMsg(t));
    }

    /**
     * 라이다 오도메트리를 구독하여 처리 
     * mapOptimization에서 받은 라이다 오도메트리 처리
    */
    void lidarOdometryHandler(const nav_msgs::msg::Odometry::SharedPtr odomMsg)
    {
        std::lock_guard<std::mutex> lock(mtx);

        // 라이다 오도메트리에 해당하는 변환 행렬 추출
        lidarOdomAffine = odom2affine(*odomMsg);

        // 해당 라이다 오도메트리의 타임스탬프 저장
        lidarOdomTime = stamp2Sec(odomMsg->header.stamp);
    }


    /**
     * IMUPreintegration에서 받은 IMU 오도메트리를 구독하여 처리
     * 1. 가장 최근의 라이다 오도메트리를 기반으로, 해당 시점과 현재 시점 사이의 IMU 오도메트리의 변환의 변화량을 계산하고,
     * 이를 곱하여 현재 시점의 IMU 오도메트리를 얻습니다. 
     * 2. Rviz에 표시할 현재 시점의 오도메트리를 게시합니다. IMU 오도메트리 경로를 게시합니다. (참고: 라이다 오도메트리와 현재 시점 사이의 경로)
    */
    void imuOdometryHandler(const nav_msgs::msg::Odometry::SharedPtr odomMsg)
    {
        std::lock_guard<std::mutex> lock(mtx);

        // IMU 오도메트리를 대기열에 추가
        imuOdomQueue.push_back(*odomMsg);

        // get latest odometry (at current IMU stamp)
        // IMU 오도메트리 큐에서 현재 라이다 오도메트리의 시간 이전의 데이터를 삭제
        if (lidarOdomTime == -1)
            return;
        while (!imuOdomQueue.empty())
        {
            if (stamp2Sec(imuOdomQueue.front().header.stamp) <= lidarOdomTime)
                imuOdomQueue.pop_front();
            else
                break;
        }
        // 최근 라이다 오도메트리 시간에 해당하는 IMU 오도메트리 포즈
        Eigen::Isometry3d imuOdomAffineFront = odom2affine(imuOdomQueue.front());
        // 현재 시간의 IMU 오도메트리 포즈
        Eigen::Isometry3d imuOdomAffineBack = odom2affine(imuOdomQueue.back());
        // IMU 오도메트리 변화량 포즈 변환
        Eigen::Isometry3d imuOdomAffineIncre = imuOdomAffineFront.inverse() * imuOdomAffineBack;
        // 최근 라이다 오도메트리 포즈 *  IMU 오도메트리 변화량 포즈 변환 =  현재 시간의 IMU 오도메트리 포즈 
        Eigen::Isometry3d imuOdomAffineLast = lidarOdomAffine * imuOdomAffineIncre; //todo 정확히 값이 어떤 형태인지 모르겠음
        auto t = tf2::eigenToTransform(imuOdomAffineLast);
        tf2::Stamped<tf2::Transform> tCur;
        tf2::convert(t, tCur);

        // publish latest odometry
        // 현재 시간의 오도메트리 포즈를 게시
        nav_msgs::msg::Odometry laserOdometry = imuOdomQueue.back();
        laserOdometry.pose.pose.position.x = t.transform.translation.x;
        laserOdometry.pose.pose.position.y = t.transform.translation.y;
        laserOdometry.pose.pose.position.z = t.transform.translation.z;
        laserOdometry.pose.pose.orientation = t.transform.rotation;
        pubImuOdometry->publish(laserOdometry);

        // publish tf
        // lidarFrame과 baselinkFrame이 다르다면 
        if(lidarFrame != baselinkFrame)
        {
            try
            {   
                // lidar frame에서 base_link 프레임으로 변환
                tf2::fromMsg(tfBuffer->lookupTransform(
                    lidarFrame, baselinkFrame, rclcpp::Time(0)), lidar2Baselink);
            }
            catch (tf2::TransformException ex)
            {
                RCLCPP_ERROR(get_logger(), "%s", ex.what());
            }
            tf2::Stamped<tf2::Transform> tb(
                tCur * lidar2Baselink, tf2_ros::fromMsg(odomMsg->header.stamp), odometryFrame);
            tCur = tb;
        }

        // 현재 시간의 오도메트리와 baselink 프레임 사이의 변환 TF를 게시
        geometry_msgs::msg::TransformStamped ts;
        tf2::convert(tCur, ts);
        ts.child_frame_id = baselinkFrame;
        tfBroadcaster->sendTransform(ts);

        // publish IMU path
        // IMU 오도메트리 경로 게시
        static nav_msgs::msg::Path imuPath;
        static double last_path_time = -1;
        double imuTime = stamp2Sec(imuOdomQueue.back().header.stamp);
        // 0.1초 간격으로 추가
        if (imuTime - last_path_time > 0.1)
        {
            last_path_time = imuTime;
            geometry_msgs::msg::PoseStamped pose_stamped;
            pose_stamped.header.stamp = imuOdomQueue.back().header.stamp;
            pose_stamped.header.frame_id = odometryFrame;
            pose_stamped.pose = laserOdometry.pose.pose;
            imuPath.poses.push_back(pose_stamped);
            // 가장 최근 라이다 오도메트리 시간 이전의 IMU 오도메트리를 삭제
            while(!imuPath.poses.empty() && stamp2Sec(imuPath.poses.front().header.stamp) < lidarOdomTime - 1.0)
                imuPath.poses.erase(imuPath.poses.begin());
            if (pubImuPath->get_subscription_count() != 0)
            {
                imuPath.header.stamp = imuOdomQueue.back().header.stamp;
                imuPath.header.frame_id = odometryFrame;
                pubImuPath->publish(imuPath);
            }
        }
    }
};

/**
 * ImuPreintegration 클래스
 * 1. 라이다 오도메트리 데이터를 사용하여 두개의 라이다 오도메트리 프레임 간의 IMU preintegration 구성을 위한 factor 그래프를 작성하고 현재 프레임 상태 (위치, 속도, bias)를 최적화합니다.
 * 2. 최적화된 상태를 기반으로 IMU preintegration를 적용하여 각 timestamp의 IMU 오도메트리를 얻습니다.
 * (factor graph: 어떤 함수의 fatorisation을 표현하는 이분 그래프이다. -> variable node(robot position, landmark position) 와  factor node(센서로부터 측정된 값)로 이루어진다.)
*/
class IMUPreintegration : public ParamServer
{
public:

    std::mutex mtx;

    // Subscriber 와 Pulblisher 설정
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr subImu;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subOdometry;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubImuOdometry;

    rclcpp::CallbackGroup::SharedPtr callbackGroupImu;
    rclcpp::CallbackGroup::SharedPtr callbackGroupOdom;

    bool systemInitialized = false;


    // 노이즈 공분산
    gtsam::noiseModel::Diagonal::shared_ptr priorPoseNoise;
    gtsam::noiseModel::Diagonal::shared_ptr priorVelNoise;
    gtsam::noiseModel::Diagonal::shared_ptr priorBiasNoise;
    gtsam::noiseModel::Diagonal::shared_ptr correctionNoise;
    gtsam::noiseModel::Diagonal::shared_ptr correctionNoise2;
    gtsam::Vector noiseModelBetweenBias;


    // iMU Preintegration
    gtsam::PreintegratedImuMeasurements *imuIntegratorOpt_;
    gtsam::PreintegratedImuMeasurements *imuIntegratorImu_;

    // IMU 데이터 큐
    std::deque<sensor_msgs::msg::Imu> imuQueOpt;
    std::deque<sensor_msgs::msg::Imu> imuQueImu;

    // IMU factor 그래프 최적화 중의 상태 변수
    gtsam::Pose3 prevPose_;
    gtsam::Vector3 prevVel_;
    gtsam::NavState prevState_;
    gtsam::imuBias::ConstantBias prevBias_;

    // IMU 상태
    gtsam::NavState prevStateOdom;
    gtsam::imuBias::ConstantBias prevBiasOdom;

    bool doneFirstOpt = false;
    double lastImuT_imu = -1;
    double lastImuT_opt = -1;

    // ISAM2 최적화 (ISAM2: 그래프 기반 최적화 알고리즘, 불열속적인 데이터 스트림을 처리하는데 사용)
    gtsam::ISAM2 optimizer;
    gtsam::NonlinearFactorGraph graphFactors;
    gtsam::Values graphValues;

    const double delta_t = 0;

    int key = 1;

    // IMU-LiDAR pose 변환
    gtsam::Pose3 imu2Lidar = gtsam::Pose3(gtsam::Rot3(1, 0, 0, 0), gtsam::Point3(-extTrans.x(), -extTrans.y(), -extTrans.z()));
    gtsam::Pose3 lidar2Imu = gtsam::Pose3(gtsam::Rot3(1, 0, 0, 0), gtsam::Point3(extTrans.x(), extTrans.y(), extTrans.z()));
    
    /**
     * 생성자
    */
    IMUPreintegration(const rclcpp::NodeOptions & options) :
            ParamServer("lio_sam_imu_preintegration", options)
    {
        callbackGroupImu = create_callback_group(
            rclcpp::CallbackGroupType::MutuallyExclusive);
        callbackGroupOdom = create_callback_group(
            rclcpp::CallbackGroupType::MutuallyExclusive);

        auto imuOpt = rclcpp::SubscriptionOptions();
        imuOpt.callback_group = callbackGroupImu;
        auto odomOpt = rclcpp::SubscriptionOptions();
        odomOpt.callback_group = callbackGroupOdom;

        // IMU 원시 데이터를 구독하고 아래의 그래프 최적화 결과를 사용하여 두 프레임 간의 IMU preintegration을 적용하여 매 시간 (IMU 주파수)의 IMU 오도메트리를 추정
        subImu = create_subscription<sensor_msgs::msg::Imu>(
            imuTopic, qos_imu,
            std::bind(&IMUPreintegration::imuHandler, this, std::placeholders::_1),
            imuOpt);

        // Map 최저화에서 dhs 라이다 오도메트리를 구독하고 두 프레임 간의 IMU Preintegration을 사용하여 그래프를 구축하고 현재 프레임 pose를 최적화 (이 Pose는 IMU 오도메트리를 업데이트하고 다음 번 그래프 최적화에만 사용)
        subOdometry = create_subscription<nav_msgs::msg::Odometry>(
            "lio_sam/mapping/odometry_incremental", qos,
            std::bind(&IMUPreintegration::odometryHandler, this, std::placeholders::_1),
            odomOpt);

        // IMU 오도메트리를 publish
        pubImuOdometry = create_publisher<nav_msgs::msg::Odometry>(odomTopic+"_incremental", qos_imu);

        // IMU Preintegration의 noise 모델
        boost::shared_ptr<gtsam::PreintegrationParams> p = gtsam::PreintegrationParams::MakeSharedU(imuGravity);
        p->accelerometerCovariance  = gtsam::Matrix33::Identity(3,3) * pow(imuAccNoise, 2); // acc white noise in continuous - 연속적인 가속도의 noise
        p->gyroscopeCovariance      = gtsam::Matrix33::Identity(3,3) * pow(imuGyrNoise, 2); // gyro white noise in continuous - 연속적인 자이로스코프의 noise
        p->integrationCovariance    = gtsam::Matrix33::Identity(3,3) * pow(1e-4, 2); // error committed in integrating position from velocities - 속도에서 위치로의 적분에서 발생하는 오차
        gtsam::imuBias::ConstantBias prior_imu_bias((gtsam::Vector(6) << 0, 0, 0, 0, 0, 0).finished());; // assume zero initial bias - 초기 biase 0으로 설정

        // noise 모델 지정
        priorPoseNoise  = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2).finished()); // rad,rad,rad,m, m, m
        priorVelNoise   = gtsam::noiseModel::Isotropic::Sigma(3, 1e4); // m/s
        priorBiasNoise  = gtsam::noiseModel::Isotropic::Sigma(6, 1e-3); // 1e-2 ~ 1e-3 seems to be good

        // 라이다 오도메트리 scan-to-map 최적화 중에 축소가 발생하면 더 큰 공분산을 선택
        correctionNoise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 0.05, 0.05, 0.05, 0.1, 0.1, 0.1).finished()); // rad,rad,rad,m, m, m
        correctionNoise2 = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1, 1, 1, 1, 1, 1).finished()); // rad,rad,rad,m, m, m
        noiseModelBetweenBias = (gtsam::Vector(6) << imuAccBiasN, imuAccBiasN, imuAccBiasN, imuGyrBiasN, imuGyrBiasN, imuGyrBiasN).finished();
        
        // IMU Preintegration을 설정하여 매 시간 (IMU 주파수)의 IMU 오도메트리를 예측 (라이다 프레임으로 변환되며, 라이다 오도메트리와 동일한 프레임으로 사용)
        imuIntegratorImu_ = new gtsam::PreintegratedImuMeasurements(p, prior_imu_bias); // setting up the IMU integration for IMU message thread - IMU 메시지 스레드에 대한 IMU integration 설정
        // 그래프 최적화를 위해 IMU Preintegration을 설정
        imuIntegratorOpt_ = new gtsam::PreintegratedImuMeasurements(p, prior_imu_bias); // setting up the IMU integration for optimization - 최적화를 위한 IMU Integraion 설정
    }

    /**
     * ISAM2 최적화를 재설정
    */
    void resetOptimization()
    {
        gtsam::ISAM2Params optParameters;
        optParameters.relinearizeThreshold = 0.1;
        optParameters.relinearizeSkip = 1;
        optimizer = gtsam::ISAM2(optParameters);

        gtsam::NonlinearFactorGraph newGraphFactors;
        graphFactors = newGraphFactors;

        gtsam::Values NewGraphValues;
        graphValues = NewGraphValues;
    }

    /**
     * 매개변수 재설정
    */
    void resetParams()
    {
        lastImuT_imu = -1;
        doneFirstOpt = false;
        systemInitialized = false;
    }

    /**
     * 라이다 오도메트리 핸들러. mapOptimization으로부터 데이터를 수집
     * 1. 매 100 프레임 라이다 오도메트리마다 ISAM2 최적화를 재설정하고 오도메트리, 속도, bias pre factor를 추가하고 최적화를 수행
     * 2. 이전 라이다 오도메트리와 현재 라이다 오도메트리 간의 IMU preintegration을 계산하고 이전 상태에 integraion 값을 적용하여 현재 프레임의 초기 상태를 추정
     *      mapOptimization에서 가져온 현재 프레임 포즈를 추가하고 그래프 최적화를 수행하여 현재 프레임의 상태를 업데이트
     * 3. 최적화 후 re-Propagate 수행. 최적화는 IMU bias를 업데이트하며, 최신 bias로 현재 라이다 오도메트리 타임 스탬프 이후의 IMU preintegraion을 다시 계산하며 이 값은 각 타임 스탬프의 pose 계산에 적용
    */
    void odometryHandler(const nav_msgs::msg::Odometry::SharedPtr odomMsg)
    {
        std::lock_guard<std::mutex> lock(mtx);
        // 현재 라이다 오도메트리의 타임 스탬프
        double currentCorrectionTime = stamp2Sec(odomMsg->header.stamp);

        // make sure we have imu data to integrate
        // IMU 최적화 큐에 IMU 데이터가 있는지 확인
        if (imuQueOpt.empty())
            return;

        // 현재 라이다 포즈, 스캔 매칭 및 그래프 최적화 후의 포즈에서 가져옴
        float p_x = odomMsg->pose.pose.position.x;
        float p_y = odomMsg->pose.pose.position.y;
        float p_z = odomMsg->pose.pose.position.z;
        float r_x = odomMsg->pose.pose.orientation.x;
        float r_y = odomMsg->pose.pose.orientation.y;
        float r_z = odomMsg->pose.pose.orientation.z;
        float r_w = odomMsg->pose.pose.orientation.w;
        RCLCPP_INFO(this->get_logger(), "ncurrent lidar timestamp: %f\n",currentCorrectionTime);
        //printf("current lidar pose x: %f\n", p_x);
        //printf("current lidar pose y: %f\n", p_y);
        //printf("current lidar pose z: %f\n", p_z);
        
        bool degenerate = (int)odomMsg->pose.covariance[0] == 1 ? true : false; 
        //printf("lidar pose true?: %d\n", degenerate);
        gtsam::Pose3 lidarPose = gtsam::Pose3(gtsam::Rot3::Quaternion(r_w, r_x, r_y, r_z), gtsam::Point3(p_x, p_y, p_z));


        // 0. initialize system - 시스템 초기화, 첫 번째 프레임
        if (systemInitialized == false)
        {
            // ISAM2 최적화 재설정
            resetOptimization();

            // pop old IMU message
            // IMU 최적화 큐에서 현재 라이다 오도메트리 타임 스탬프 이전의 IMU 데이터를 제거
            while (!imuQueOpt.empty())
            {
                if (stamp2Sec(imuQueOpt.front().header.stamp) < currentCorrectionTime - delta_t)
                {
                    lastImuT_opt = stamp2Sec(imuQueOpt.front().header.stamp);
                    imuQueOpt.pop_front();
                }
                else
                    break;
            }
            // initial pose
            prevPose_ = lidarPose.compose(lidar2Imu);
            gtsam::PriorFactor<gtsam::Pose3> priorPose(X(0), prevPose_, priorPoseNoise);
            graphFactors.add(priorPose);
            // initial velocity
            prevVel_ = gtsam::Vector3(0, 0, 0);
            gtsam::PriorFactor<gtsam::Vector3> priorVel(V(0), prevVel_, priorVelNoise);
            graphFactors.add(priorVel);
            // initial bias
            prevBias_ = gtsam::imuBias::ConstantBias();

            // code commnet 
            //printf("prevBias_ Value: %f\n", prevBias_.accelerometer());
            //printf("priorPose_ Value: %f  %f  %f\n", prevPose_.x(), prevPose_.y(), prevPose_.z() );

            gtsam::PriorFactor<gtsam::imuBias::ConstantBias> priorBias(B(0), prevBias_, priorBiasNoise);
            graphFactors.add(priorBias);
            // add values 
            graphValues.insert(X(0), prevPose_);
            graphValues.insert(V(0), prevVel_);
            graphValues.insert(B(0), prevBias_);
            // optimize once - 최적화 한번 수행
            optimizer.update(graphFactors, graphValues);
            graphFactors.resize(0);
            graphValues.clear();

            //최적화 후 bias를 재설정
            imuIntegratorImu_->resetIntegrationAndSetBias(prevBias_);
            imuIntegratorOpt_->resetIntegrationAndSetBias(prevBias_);
            
            key = 1;
            systemInitialized = true;
            return;
        }


        // reset graph for speed
        // 100 프레임 마다 ISAM2 최적화를 재설정하여 최적화의 효율성 유지
        if (key == 100)
        {
            // get updated noise before reset
            // 이전 프레임의 포즈, 속도, bias의 노이즈 모델을 가져옴
            gtsam::noiseModel::Gaussian::shared_ptr updatedPoseNoise = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(X(key-1)));
            gtsam::noiseModel::Gaussian::shared_ptr updatedVelNoise  = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(V(key-1)));
            gtsam::noiseModel::Gaussian::shared_ptr updatedBiasNoise = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(B(key-1)));
            // reset graph
            // ISAM2 최적화를 재설정
            resetOptimization();
            // add pose
            // priorPose factor를 이전 프레임 값으로 초기화
            gtsam::PriorFactor<gtsam::Pose3> priorPose(X(0), prevPose_, updatedPoseNoise);
            graphFactors.add(priorPose);
            // add velocity
            // priorVel factore를 이전 프레임 값으로 초기화
            gtsam::PriorFactor<gtsam::Vector3> priorVel(V(0), prevVel_, updatedVelNoise);
            graphFactors.add(priorVel);
            // add bias
            // IMU bias priorBias를 이전 프레임 값으로 초기화
            gtsam::PriorFactor<gtsam::imuBias::ConstantBias> priorBias(B(0), prevBias_, updatedBiasNoise);
            graphFactors.add(priorBias);
            // add values
            // 변수 노드에 초기 값을 할당
            graphValues.insert(X(0), prevPose_);
            graphValues.insert(V(0), prevVel_);
            graphValues.insert(B(0), prevBias_);
            // optimize once
            // 최적화를 수행
            optimizer.update(graphFactors, graphValues);
            graphFactors.resize(0);
            graphValues.clear();

            key = 1;
        }

        // 1. integrate imu data and optimize
        // 이전 프레임과 현재 프레임 간의 IMU preintegration을 계산하고 이전 상태에 integration 값을 적용하여 현재 프레임의 초기 상태를 추정

        RCLCPP_INFO(this->get_logger(),"imuQue Size: %d\n", (int)imuQueOpt.size());

        while (!imuQueOpt.empty())
        {
            // pop and integrate imu data that is between two optimizations
            // 현재 IMU 데이터와 타임 스탬프를 추출하고 preintegration을 수행
            sensor_msgs::msg::Imu *thisImu = &imuQueOpt.front();
            double imuTime = stamp2Sec(thisImu->header.stamp);

            
            if (imuTime < currentCorrectionTime - delta_t)
            {   
                // 두 프레임 간의 IMU 데이터 시간 간격을 계산
                double dt = (lastImuT_opt < 0) ? (1.0 / 500.0) : (imuTime - lastImuT_opt);
                
                // IMU preintegration 데이터를 입력 (가속도, 각속도, 시간 간격) 
                imuIntegratorOpt_->integrateMeasurement(
                        gtsam::Vector3(thisImu->linear_acceleration.x, thisImu->linear_acceleration.y, thisImu->linear_acceleration.z),
                        gtsam::Vector3(thisImu->angular_velocity.x,    thisImu->angular_velocity.y,    thisImu->angular_velocity.z), dt);

                //imuIntegratorOpt_->integrateMeasurement(
                //        gtsam::Vector3(0, 0, 0),
                //        gtsam::Vector3(0, 0, 0), 0.001);

                //printf("[%d] current imu: %f %f %f %f %f %f %f\n", cnt, thisImu->linear_acceleration.x, thisImu->linear_acceleration.y, thisImu->linear_acceleration.z, 
                //thisImu->angular_velocity.x, thisImu->angular_velocity.y, thisImu->angular_velocity.z, dt);
                lastImuT_opt = imuTime;
                
                // 처리한 IMU 데이터를 큐에서 제거
                imuQueOpt.pop_front();
            }
            else
                break;
        }



        // add imu factor to graph
        // IMU preintegration factor를 추가
        const gtsam::PreintegratedImuMeasurements& preint_imu = dynamic_cast<const gtsam::PreintegratedImuMeasurements&>(*imuIntegratorOpt_);
        //printf("key2: %d\n", key);

        // IMU factor를 추가 (파라미터: 이전 포즈, 이전 속도, 현재 포즈, 현재 속도, 이전 bias, preintegration 값)
        gtsam::ImuFactor imu_factor(X(key - 1), V(key - 1), X(key), V(key), B(key - 1), preint_imu);
        //gtsam::ImuFactor imu_factor(0, 0, X(key), V(key), B(key - 1), preint_imu);
        // graphValues.at(X(key-1)).print();

        //preint_imu.print();

        graphFactors.add(imu_factor);


        // add imu bias between factor
        // IMU bias factor를 추가(파라미터: 이전 bias, 현재 bias, measurement, 노이즈, detaTij()는 적분 세그먼트 시간 )
        graphFactors.add(gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>(B(key - 1), B(key), gtsam::imuBias::ConstantBias(),
                         gtsam::noiseModel::Diagonal::Sigmas(sqrt(imuIntegratorOpt_->deltaTij()) * noiseModelBetweenBias)));
        // add pose factor
        // pose factor 추가
        gtsam::Pose3 curPose = lidarPose.compose(lidar2Imu);
        gtsam::PriorFactor<gtsam::Pose3> pose_factor(X(key), curPose, degenerate ? correctionNoise2 : correctionNoise);
        graphFactors.add(pose_factor);
        // insert predicted values
        // 이전 상태와 bias를 사용하여 IMU prediction을 수행하고, 현재 프레임의 초기 상태를 얻음
        gtsam::NavState propState_ = imuIntegratorOpt_->predict(prevState_, prevBias_);
        // 변수 노드에 초기 값 할당
        graphValues.insert(X(key), propState_.pose());
        graphValues.insert(V(key), propState_.v());
        graphValues.insert(B(key), prevBias_);
        // optimize
        // 최적화 수행
        optimizer.update(graphFactors, graphValues);
        optimizer.update();
        graphFactors.resize(0);
        graphValues.clear();
        // Overwrite the beginning of the preintegration for the next step.
        // 최적화 결과를 얻고 현재 프레임의 포즈와 속도를 업데이트
        gtsam::Values result = optimizer.calculateEstimate();
        prevPose_  = result.at<gtsam::Pose3>(X(key));
        prevVel_   = result.at<gtsam::Vector3>(V(key));
      

        // 현재 프레임의 상태와 bias 업데이트
        prevState_ = gtsam::NavState(prevPose_, prevVel_);
        prevBias_  = result.at<gtsam::imuBias::ConstantBias>(B(key));

        //printf("2. prevBias_ Value: %f\n", prevBias_.accelerometer());
        //printf("2. priorPose_ Value: %f  %f  %f\n", prevPose_.x(), prevPose_.y(), prevPose_.z() );
        //printf("2. prevVel_ Value: %f\n", prevState_.v());
        // Reset the optimization preintegration object.
        // 다음 프레임에서 preIntegration을 두 프레임 사이의 변화량으로 설정하기 위해 Preintegration을 재설정
        imuIntegratorOpt_->resetIntegrationAndSetBias(prevBias_);
        // check optimization
        // IMU factor 그래프 최적화 결과, 속도나 bias가 지나치게 크면 실패로 간주
        if (failureDetection(prevVel_, prevBias_))
        {
            // 매개 변수 재설정
            resetParams();
            return;
        }


        // 2. after optiization, re-propagate imu odometry preintegration
        // 2. 최적화 이후,  re-propagate 수행, 최적화는 IMU bias를 업데이트하며, 최신 bias를 사용하여 현재 라이다 오도메트리 시점 이후 IMU preintegration을 다시 계산하고, 이 값은 각 시점의 포즈 계산에 사용
        prevStateOdom = prevState_;
        prevBiasOdom  = prevBias_;
        // first pop imu message older than current correction data
        // 현재 라이다 오도메트리 시점 이전의 IMU 데이터를 IMU 큐에서 제거
        double lastImuQT = -1;
        while (!imuQueImu.empty() && stamp2Sec(imuQueImu.front().header.stamp) < currentCorrectionTime - delta_t)
        {
            lastImuQT = stamp2Sec(imuQueImu.front().header.stamp);
            imuQueImu.pop_front();
        }
        // repropogate
        // 나머지 IMU 데이터에 대한 preintegration을 계산
        if (!imuQueImu.empty())
        {
            // reset bias use the newly optimized bias
            // preintegraion을 reset하고 최신 bais를 설정
            imuIntegratorImu_->resetIntegrationAndSetBias(prevBiasOdom);
            // integrate imu message from the beginning of this optimization
            //  preintegration을 계산
            for (int i = 0; i < (int)imuQueImu.size(); ++i)
            {
                sensor_msgs::msg::Imu *thisImu = &imuQueImu[i];
                double imuTime = stamp2Sec(thisImu->header.stamp);
                double dt = (lastImuQT < 0) ? (1.0 / 500.0) :(imuTime - lastImuQT);

                imuIntegratorImu_->integrateMeasurement(gtsam::Vector3(thisImu->linear_acceleration.x, thisImu->linear_acceleration.y, thisImu->linear_acceleration.z),
                                                        gtsam::Vector3(thisImu->angular_velocity.x,    thisImu->angular_velocity.y,    thisImu->angular_velocity.z), dt);
                
                lastImuQT = imuTime;
            }
        }

        // key를 증가 시키고 최적화가 처음 수행되었음을 표시
        ++key;
        doneFirstOpt = true;
    }


    /**
    * IMU factor 그래프 최적화 결과, 속도나 bias가 지나치게 크면 실패로 간주
    */
    bool failureDetection(const gtsam::Vector3& velCur, const gtsam::imuBias::ConstantBias& biasCur)
    {
        Eigen::Vector3f vel(velCur.x(), velCur.y(), velCur.z());
        if (vel.norm() > 30)
        {
            RCLCPP_WARN(get_logger(), "Large velocity, reset IMU-preintegration!");
            return true;
        }

        Eigen::Vector3f ba(biasCur.accelerometer().x(), biasCur.accelerometer().y(), biasCur.accelerometer().z());
        Eigen::Vector3f bg(biasCur.gyroscope().x(), biasCur.gyroscope().y(), biasCur.gyroscope().z());
        if (ba.norm() > 1.0 || bg.norm() > 1.0)
        {
            RCLCPP_WARN(get_logger(), "Large bias, reset IMU-preintegration!");
            return true;
        }

        return false;
    }

    /**
     * IMU 원본 데이터를 구독
     * 1. 이전 프레임 라이다 오도메트리 타임스탬프에 해당하는 상태와 bias를 사용하여 이 시점부터 현재 시점까지의 IMU preintegraion을 적용하여 현재 시점의 상태를 얻음. (IMU 오도메트리)
     * 2. IMU 오도메트리의 포즈를 라이다 좌표계로 변환하고 오도메트리를 게시
    */
    void imuHandler(const sensor_msgs::msg::Imu::SharedPtr imu_raw)
    {
        std::lock_guard<std::mutex> lock(mtx);

        // IMU 원시 츨정 데이터를 라이다 좌표계로 변환. 가속도, 각속도, 롤-피치-요 방향으로 변환
        sensor_msgs::msg::Imu thisImu = imuConverter(*imu_raw);

        // 현재 프레임의 IMU 데이터를 큐에 추가
        imuQueOpt.push_back(thisImu);
        imuQueImu.push_back(thisImu);

        // 이전 IMU factor 그래프 최적화가 성공적으로 수행되었는지 확인하여, 이전 프레임(라이다 오도메트리 프레임)의 상태와 bias가 업데이트되었고, preintegraion이 다시 계산되었음을 보장
        if (doneFirstOpt == false)
            return;

        double imuTime = stamp2Sec(thisImu.header.stamp);
        double dt = (lastImuT_imu < 0) ? (1.0 / 500.0) : (imuTime - lastImuT_imu);
        lastImuT_imu = imuTime;

        // integrate this single imu message
        // IMU preintegration에 현재 IMU 데이터를 추가 (주의: 이 IMU preintegraion의 시작 시간은 이전 라이다 오도메트리 프레임의 타임스탬프 입니다.)
        imuIntegratorImu_->integrateMeasurement(gtsam::Vector3(thisImu.linear_acceleration.x, thisImu.linear_acceleration.y, thisImu.linear_acceleration.z),
                                                gtsam::Vector3(thisImu.angular_velocity.x,    thisImu.angular_velocity.y,    thisImu.angular_velocity.z), dt);

        // printf("this linear_acceleration.x: %f\n", thisImu.linear_acceleration.x);
        // printf("this linear_acceleration.y: %f\n", thisImu.linear_acceleration.y);
        // printf("this linear_acceleration.z: %f\n", thisImu.linear_acceleration.z);
        // printf("this angular_velocity.x: %f\n", thisImu.angular_velocity.x);
        // printf("this angular_velocity.y: %f\n", thisImu.angular_velocity.y);
        // printf("this angular_velocity.z: %f\n\n", thisImu.angular_velocity.z);


        // predict odometry
        // 이전 라이다 오도메트리 프레임의 상태와 바이어스를 사용하여 이 시점부터 현재 시점까지의 IMU preintegraion을 적용하여 현재 시점의 상태를 얻음
        gtsam::NavState currentState = imuIntegratorImu_->predict(prevStateOdom, prevBiasOdom);

        // publish odometry
        // IMU 오도메트리를 게시하기 위해 라이다 좌표계로 변환
        auto odometry = nav_msgs::msg::Odometry();
        odometry.header.stamp = thisImu.header.stamp;
        odometry.header.frame_id = odometryFrame;
        odometry.child_frame_id = "odom_imu";

        // transform imu pose to ldiar
        // 라이다 좌표계로 변환
        gtsam::Pose3 imuPose = gtsam::Pose3(currentState.quaternion(), currentState.position());
        gtsam::Pose3 lidarPose = imuPose.compose(imu2Lidar);

        odometry.pose.pose.position.x = lidarPose.translation().x();
        odometry.pose.pose.position.y = lidarPose.translation().y();
        odometry.pose.pose.position.z = lidarPose.translation().z();
        odometry.pose.pose.orientation.x = lidarPose.rotation().toQuaternion().x();
        odometry.pose.pose.orientation.y = lidarPose.rotation().toQuaternion().y();
        odometry.pose.pose.orientation.z = lidarPose.rotation().toQuaternion().z();
        odometry.pose.pose.orientation.w = lidarPose.rotation().toQuaternion().w();
        
        odometry.twist.twist.linear.x = currentState.velocity().x();
        odometry.twist.twist.linear.y = currentState.velocity().y();
        odometry.twist.twist.linear.z = currentState.velocity().z();
        odometry.twist.twist.angular.x = thisImu.angular_velocity.x + prevBiasOdom.gyroscope().x();
        odometry.twist.twist.angular.y = thisImu.angular_velocity.y + prevBiasOdom.gyroscope().y();
        odometry.twist.twist.angular.z = thisImu.angular_velocity.z + prevBiasOdom.gyroscope().z();
        pubImuOdometry->publish(odometry);
    }
};


int main(int argc, char** argv)
{   
    rclcpp::init(argc, argv);

    rclcpp::NodeOptions options;
    options.use_intra_process_comms(true);
    rclcpp::executors::MultiThreadedExecutor e;

    auto ImuP = std::make_shared<IMUPreintegration>(options);
    auto TF = std::make_shared<TransformFusion>(options);
    e.add_node(ImuP);
    e.add_node(TF);

    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "\033[1;32m----> IMU Preintegration Started.\033[0m");

    e.spin();

    rclcpp::shutdown();
    return 0;
}
