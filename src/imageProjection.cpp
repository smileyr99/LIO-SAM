#include "utility.hpp"
#include "lio_sam/msg/cloud_info.hpp"

//// point cloud 구조 정의
/**
 * Velodyne point cloud 구조체, 변수, 이름 XYZIRT는 각 변수의 첫글자 입니다.
*/
struct VelodynePointXYZIRT
{
    PCL_ADD_POINT4D         // 위치
    PCL_ADD_INTENSITY;      // 라이다 포인트 반사 강도 또는 포인트 인덱스로 사용 가능
    uint16_t ring;          // 스캔 라인
    float time;             // time stamp, 현재 프레임의 첫번째 라이다 포인트와 시간 차이를 기록합니다. 첫 번째 포인트의 경우 time=0 입니다.
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;            // 16바이트 메로리 정렬, EIGEN SSE(벡터 및 행렬 연산을 최적화하는 기술) 최적화 요구사항
//PCL 포인트 클라우드 형식으로 등록
POINT_CLOUD_REGISTER_POINT_STRUCT (VelodynePointXYZIRT,
    (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
    (uint16_t, ring, ring) (float, time, time)
)

/**
 * ouster point cloud 구조체, 변수, 이름 XYZIRT는 각 변수의 첫글자 입니다.
*/
struct OusterPointXYZIRT {
    PCL_ADD_POINT4D;        // 위치
    float intensity;        // 라이다 포인트 반사 강도 또는 포인트 인덱스로 사용 가능
    uint32_t t;             // time stamp, 현재 프레임의 첫번째 라이다 포인트와 시간 차이를 기록합니다. 첫 번째 포인트의 경우 time=0 입니다.
    uint16_t reflectivity;  // 반사 정도 
    uint8_t ring;           // 스캔 라인
    uint16_t noise;         // 노이즈에 대한 정보
    uint32_t range;         //거리 정보
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW     // 16바이트 메로리 정렬
} EIGEN_ALIGN16;
// PCL 포인트 클라우드 형식으로 등록        
POINT_CLOUD_REGISTER_POINT_STRUCT(OusterPointXYZIRT,
    (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
    (uint32_t, t, t) (uint16_t, reflectivity, reflectivity)
    (uint8_t, ring, ring) (uint16_t, noise, noise) (uint32_t, range, range)
)

// Use the Velodyne point format as a common representation
// 이 프로그램은 Velodyne 포인트 클라우드 구조체를 사용합니다.
using PointXYZIRT = VelodynePointXYZIRT;

//IMU 데이터 큐 길이
const int queueLength = 2000;


//// ImageProjection class의 맴버 변수 정의
class ImageProjection : public ParamServer
{
private:

    //IMU 및 Odometry 데이터의 상호배제 mutex
    std::mutex imuLock;
    std::mutex odoLock;

    // 원시 라이다 클라우드 데이터 subscriber
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subLaserCloud;
    rclcpp::CallbackGroup::SharedPtr callbackGroupLidar;

    // 수정된 라이다 클라우드 데이터 publisher
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloud;

    // 현재 프레임의 보정된 포인트 클라우드 및 유효한 포인트 publisher
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubExtractedCloud;
    rclcpp::Publisher<lio_sam::msg::CloudInfo>::SharedPtr pubLaserCloudInfo;

    // IMU 데이터 큐 (원시 데이터, LiDAR  좌표계 변환)
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr subImu;
    rclcpp::CallbackGroup::SharedPtr callbackGroupImu;
    std::deque<sensor_msgs::msg::Imu> imuQueue;

    // IMU odometry 큐
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subOdom;
    rclcpp::CallbackGroup::SharedPtr callbackGroupOdom;
    std::deque<nav_msgs::msg::Odometry> odomQueue;

    // 라이다 포인트 클라우드 데이터 큐
    std::deque<sensor_msgs::msg::PointCloud2> cloudQueue;

    // 현재 처리 중인 프레임의 포인트 클라우드
    sensor_msgs::msg::PointCloud2 currentCloudMsg;

    // 현재 라이다 프레임의 시작 및 종료 시간에 해당하는 IMU 데이터
    // 시작 시간부터의 회전 증가 및 타임스탬프를 계산하고, 현재 라이다 프레임의 시작 및 종료 시간 범위 내의 각 시점의 회전 자세를 보간하는데 사용됩니다.
    double *imuTime = new double[queueLength];
    double *imuRotX = new double[queueLength];
    double *imuRotY = new double[queueLength];
    double *imuRotZ = new double[queueLength];

    int imuPointerCur;      // 현재 처리 중인 IMU 데이터의 인덱스
    bool firstPointFlag;    // 현재 처리 중인 데이터가 첫번째 데이터인지 여부
    Eigen::Affine3f transStartInverse;

    // 현재 프레임의 원시 라이다 포인트 클라우드
    pcl::PointCloud<PointXYZIRT>::Ptr laserCloudIn;
    pcl::PointCloud<OusterPointXYZIRT>::Ptr tmpOusterCloudIn;
    // 현재 프레임의 motion distortion 보정 후 라이다 포인트 클라우드
    pcl::PointCloud<PointType>::Ptr   fullCloud;
    // fullCloud에서 유효한 포인트 추출
    pcl::PointCloud<PointType>::Ptr   extractedCloud;


    /**
     * deskew - 데이터나 이미지에서 왜곡을 보정하는 과정을 의미
     * cv::Mat - OpenCV 라이브러리에서 이미지나 행렬과 같은 데이터를 다루기 위한 자료구조
    */
    int deskewFlag;     // deskew 제어 flag
    cv::Mat rangeMat;   // 범위 데이터를 저장하기 위한 행렬?

    bool odomDeskewFlag;    

    //현재 라이다 프레임의 시작 및 종료 시간에 해당하는 IMU Odometry Pose 변환 및 해당 변환의 이동 증가량. 시작 시간부터의 위치를 각 시점에서 보간하는 데 사용
    float odomIncreX;
    float odomIncreY;
    float odomIncreZ;

    // 현재 프레임의 라이다 포인트 클라우드 motion distrotion 보정 후 데이터, 포인트 클라우드 데이터, 초기 pose, 자세 각도 등의 featureExtraction에 publish
    lio_sam::msg::CloudInfo cloudInfo;
    // 현재 프레임의 시작 시간
    double timeScanCur;
    // 현재 프레임의 종료 시간
    double timeScanEnd;
    // 현재 프레임의 헤더 (타임스탬프 정보 포함)
    std_msgs::msg::Header cloudHeader;

    vector<int> columnIdnCountVec;


//// ImageProjection 생성자
public:
    ImageProjection(const rclcpp::NodeOptions & options) :
            ParamServer("lio_sam_imageProjection", options), deskewFlag(0)
    {
        callbackGroupLidar = create_callback_group(
            rclcpp::CallbackGroupType::MutuallyExclusive);
        callbackGroupImu = create_callback_group(
            rclcpp::CallbackGroupType::MutuallyExclusive);
        callbackGroupOdom = create_callback_group(
            rclcpp::CallbackGroupType::MutuallyExclusive);

        auto lidarOpt = rclcpp::SubscriptionOptions();
        lidarOpt.callback_group = callbackGroupLidar;
        auto imuOpt = rclcpp::SubscriptionOptions();
        imuOpt.callback_group = callbackGroupImu;
        auto odomOpt = rclcpp::SubscriptionOptions();
        odomOpt.callback_group = callbackGroupOdom;

        // 원시 IMU 데이터를 subscribe
        subImu = create_subscription<sensor_msgs::msg::Imu>(
            imuTopic, qos_imu,
            std::bind(&ImageProjection::imuHandler, this, std::placeholders::_1),
            imuOpt);
        // IMU Odometry를 scbscribe (IMU Preintegration을 통해 얻어진 시간당 IMU pose 의미)
        subOdom = create_subscription<nav_msgs::msg::Odometry>(
            odomTopic + "_incremental", qos_imu,
            std::bind(&ImageProjection::odometryHandler, this, std::placeholders::_1),
            odomOpt);
        // 원시 LiDAR 데이터를 subscribe
        subLaserCloud = create_subscription<sensor_msgs::msg::PointCloud2>(
            pointCloudTopic, qos_lidar,
            std::bind(&ImageProjection::cloudHandler, this, std::placeholders::_1),
            lidarOpt);

        // 현재 LiDAR 프레임의 motion distortion 보정 후 포인트 클라우드 및 유효한 포인트 publish
        pubExtractedCloud = create_publisher<sensor_msgs::msg::PointCloud2>(
            "lio_sam/deskew/cloud_deskewed", 1);
        // 현재 LiDAR 프레임의 motion distrotion 보정 후 포인트 클라우드 정보 publish
        pubLaserCloudInfo = create_publisher<lio_sam::msg::CloudInfo>(
            "lio_sam/deskew/cloud_info", qos);

        // 메모리 할당
        allocateMemory();
        // 매개 변수 초기화
        resetParameters();

        pcl::console::setVerbosityLevel(pcl::console::L_ERROR);
    }

    /**
     * 메모리 할당
    */
    void allocateMemory()
    {
        laserCloudIn.reset(new pcl::PointCloud<PointXYZIRT>());
        tmpOusterCloudIn.reset(new pcl::PointCloud<OusterPointXYZIRT>());
        fullCloud.reset(new pcl::PointCloud<PointType>());
        extractedCloud.reset(new pcl::PointCloud<PointType>());

        fullCloud->points.resize(N_SCAN*Horizon_SCAN);

        cloudInfo.start_ring_index.assign(N_SCAN, 0);
        cloudInfo.end_ring_index.assign(N_SCAN, 0);

        cloudInfo.point_col_ind.assign(N_SCAN*Horizon_SCAN, 0);
        cloudInfo.point_range.assign(N_SCAN*Horizon_SCAN, 0);

        resetParameters();
    }

    /**
     * 파라미터 초기화 - 매 LiDAR 프레임마다 (LiDAR 데이터를 수신할 때마다) 파라미터 재설정을 해야합니다.
    */

    void resetParameters()
    {
        laserCloudIn->clear();
        extractedCloud->clear();
        // reset range matrix for range image projection
        rangeMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32F, cv::Scalar::all(FLT_MAX));

        imuPointerCur = 0;
        firstPointFlag = true;
        odomDeskewFlag = false;

        for (int i = 0; i < queueLength; ++i)
        {
            imuTime[i] = 0;
            imuRotX[i] = 0;
            imuRotY[i] = 0;
            imuRotZ[i] = 0;
        }
    }

    // ImageProjection destructor?
    ~ImageProjection(){}

    /**
     * 원시 IMU 데이터를 구독
     * 이 데이터는 IMU 원시 측정값을 LiDAR 좌료계로 변환한 acc, gyro, RPY 정보를 포함합니다.
    */
    void imuHandler(const sensor_msgs::msg::Imu::SharedPtr imuMsg)
    {
        // IMU 원시 데이터를 LiDAR 좌표계로 변환하고 acc, gyro, RPY 요소를 추출
        sensor_msgs::msg::Imu thisImu = imuConverter(*imuMsg);

        // 데이터를 추가할 때마다 큐를 잠그고 사용 중지 합니다.
        std::lock_guard<std::mutex> lock1(imuLock);
        imuQueue.push_back(thisImu);

        // debug IMU data
        // cout << std::setprecision(6);
        // cout << "IMU acc: " << endl;
        // cout << "x: " << thisImu.linear_acceleration.x << 
        //       ", y: " << thisImu.linear_acceleration.y << 
        //       ", z: " << thisImu.linear_acceleration.z << endl;
        // cout << "IMU gyro: " << endl;
        // cout << "x: " << thisImu.angular_velocity.x << 
        //       ", y: " << thisImu.angular_velocity.y << 
        //       ", z: " << thisImu.angular_velocity.z << endl;
        // double imuRoll, imuPitch, imuYaw;
        // tf::Quaternion orientation;
        // tf::quaternionMsgToTF(thisImu.orientation, orientation);
        // tf::Matrix3x3(orientation).getRPY(imuRoll, imuPitch, imuYaw);
        // cout << "IMU roll pitch yaw: " << endl;
        // cout << "roll: " << imuRoll << ", pitch: " << imuPitch << ", yaw: " << imuYaw << endl << endl;
    }

    /*
     * imuPreintegration을 통해 얻은 IMU Odometry 정보를 처리하기 위해 imuPreintegration 메시지를 구독합니다.
    */
    void odometryHandler(const nav_msgs::msg::Odometry::SharedPtr odometryMsg)
    {
        std::lock_guard<std::mutex> lock2(odoLock);
        odomQueue.push_back(*odometryMsg);
    }

    /**
     * 원시 LiDAR 데이터를 구독
     * 1. 라이다 포인트를 큐에 추가하고, 가장 이전 프레임을 현재 프레임으로 선택하여 시작 및 종료 타임 스탬프를 계산하고 데이터 유효성을 확인합니다.
     * 2. 현재 프레임의 시작 및 종료에 해당하는 IMU 데이터 및 IMU Odometry 데이터를 처리합니다.
     *   IMU 데이터:
     *     2.1 현재 프레임의 시작 시간부터 종료시간까지의 IMU 데이터를 반복하여 시작 시간에 해당하는 IMU pose, RPY를 현재 프레임의 초기 pose, RPY로 설정합니다.
     *     2.2 각 소도 및 시간 적분을 사용하여 각 시간대에서 초기 시점 대비 상대 회전을 계산하고, 초기 시점에서의 회전을 0으로 설정합니다.
     *   IMU Odometry 데이터:
     *     2.3 현재 프레임의 시작 및 종료 시간에 해당하는 IMU Odometry 데이터를 반복하여 시작 시간에 해당하는 현재 프레임의 초기 위치로 설정합니다.
     *     2.4 시작 및 종료 시점에 해당하는 IMU Odometry를 사용하여 상대적인 자세 변화을 계산하고 이동 변위를 저장합니다. 
     * 3. 현재 프레임의 라이다 포인트 클라우드를 motion distortion 합니다.
     *   3.1 라이다 포인트의 거리 및 스캔라인 유효성을 확인합니다.
     *   3.2 라이다 motion distortion으로 보정하고 포인트를 저장합니다.
     * 4. 유효한 라이다 포인트를 추출하여 extractedCloud에 저장합니다.
     * 5. 현재 프레임의 보정된 포인트 클라우드와 유효한 포인트를 게시합니다.
     * 6. 매 프레임의 LiDAR 데이터를 처리할 때마다 이러한 매개 변수를 재설정합니다.
    */
    void cloudHandler(const sensor_msgs::msg::PointCloud2::SharedPtr laserCloudMsg)
    {
        // 라이다 클라우드 데이터를 큐에 추가하고 , 가장 이전의 프레임을 현재 프레임으로 선택하여, 타임스탬프를 계산하고 데이터 유효성을 확인
        if (!cachePointCloud(laserCloudMsg))
            return;

        // 현재 프레임의 시작 및 종료 시간에 해당하는 IMU 데이터와  IMU Odometry 데이터를 처리
        if (!deskewInfo())
            return;
        
        // 현재 프레임의 라이다 포인트 클라우드에 motion distortion을 보정
        // 1) 라이다 포인트의 거리 및 스캔 라인 유효성을 확인합니다.
        // 2) 라이다 포인트의 motion distortion을 보정하고 포인트를 저장
        projectPointCloud();

        // 유효한 라이다 포인트를 추출하여 extractedCloud에 저장
        cloudExtraction();

        // 현재 프레임의 보정된 포인트 클라우드와 유효한 포인트를 게시
        publishClouds();

        // 매 프레임의 LiDAR 데이터를 처리할 때마다 이러한 매개 변수를 재설정
        resetParameters();
    }

    /**
     * 한 프레임을 LiDAR 포인트 클라우드 대기열에 추가하고, 가장 오래된 프레임을 현재 프레임으로 선택하여 타임스탬프 범위를 계산하고 데이터 유효성을 확인
     * 
     * @param laserCloudMsg 현재 프레임의 LiDAR 포인트 클라우드 메시지
     * @return 성공 true, 실패 false로 반환
    */
    bool cachePointCloud(const sensor_msgs::msg::PointCloud2::SharedPtr& laserCloudMsg)
    {
        // cache point cloud 
        // 1. 현재 LiDAR 클라우드 메시지를 큐에 추가
        cloudQueue.push_back(*laserCloudMsg);
        // 2. 현재 큐의 크기가 2 이하인 경우 처리하지 않고 반환
        if (cloudQueue.size() <= 2)
            return false;

        // convert cloud - 3. 가장 오래된 LiDAR 포인트 클라우드를 현재 클라우드로 선택하고 큐에서 제거
        currentCloudMsg = std::move(cloudQueue.front());
        cloudQueue.pop_front();

        // 4. 현재 센서 타입이 Velodyne인 경우 , LiDAR 클라우드를 PCL 포인트 클라우드로 변환
        if (sensor == SensorType::VELODYNE || sensor == SensorType::LIVOX)
        {
            pcl::moveFromROSMsg(currentCloudMsg, *laserCloudIn);
        }
        // 5. 현재 센서 타입이 Ouster인 경우, LiDAR 클라우드를 Velodyne 형식으로 변환
        else if (sensor == SensorType::OUSTER)
        {
            // Convert to Velodyne format
            pcl::moveFromROSMsg(currentCloudMsg, *tmpOusterCloudIn);
            laserCloudIn->points.resize(tmpOusterCloudIn->size());
            laserCloudIn->is_dense = tmpOusterCloudIn->is_dense;
            for (size_t i = 0; i < tmpOusterCloudIn->size(); i++)
            {
                auto &src = tmpOusterCloudIn->points[i];
                auto &dst = laserCloudIn->points[i];
                dst.x = src.x;
                dst.y = src.y;
                dst.z = src.z;
                dst.intensity = src.intensity;
                dst.ring = src.ring;
                dst.time = src.t * 1e-9f;
            }
        }
        else    // 6. 현재 센서 타입이 VELODYNE 또는 OUSTER가 아닌 경우 오류 메시지를 출력하고 ROS를 종료
        {
            RCLCPP_ERROR_STREAM(get_logger(), "Unknown sensor type: " << int(sensor));
            rclcpp::shutdown();
        }

        // get timestamp
        // 7. 현재 프레임의 헤더 정보를 저장
        cloudHeader = currentCloudMsg.header;
        // 8. 현재 프레임의 시작 타임스탬프를 계산
        timeScanCur = stamp2Sec(cloudHeader.stamp);
        // 9. 현재 프레임의 끝 타임스탬프를 계산 (주의: 시간 기록은 첫번째 포인트와의 상대적인 시간 차이를 나타냅니다. 첫 번째 포인트의 시간은 0입니다.)
        timeScanEnd = timeScanCur + laserCloudIn->points.back().time;

        // check dense flag
        // 10. LiDAR 클라우드가 밀도 형식이 아닌 경우 오류 메시지를 출력하고 ROS종료
        if (laserCloudIn->is_dense == false)
        {
            RCLCPP_ERROR(get_logger(), "Point cloud is not in dense format, please remove NaN points first!");
            rclcpp::shutdown();
        }

        // check ring channel
        // 11. 'ring' 채널이 있는지 확인 (한번만 체크하도록 static 변수로 선언)
        static int ringFlag = 0;
        if (ringFlag == 0)
        {
            ringFlag = -1;
            for (int i = 0; i < (int)currentCloudMsg.fields.size(); ++i)
            {
                if (currentCloudMsg.fields[i].name == "ring")
                {
                    ringFlag = 1;
                    break;
                }
            }
            if (ringFlag == -1)
            {
                RCLCPP_ERROR(get_logger(), "Point cloud ring channel not available, please configure your point cloud data!");
                rclcpp::shutdown();
            }
        }

        // check point time
        // 12. time 또는 t 채널이 있는지 확인 (한번만 체크하도록 deskewFlag 변수 사용)
        if (deskewFlag == 0)
        {
            deskewFlag = -1;
            for (auto &field : currentCloudMsg.fields)
            {
                if (field.name == "time" || field.name == "t")
                {
                    deskewFlag = 1;
                    break;
                }
            }
            if (deskewFlag == -1)
                RCLCPP_WARN(get_logger(), "Point cloud timestamp not available, deskew function disabled, system will drift significantly!");
        }

        return true;
    }

    /**
     * 현재 프레임의 시작 및 종료 시간에 해당하는 IMU 데이터 및 IMU Odometry 데이터 처리
     * 
     * @return 처리가 성공적으로 완료되면 true, 그렇지 않으면 false 반환
    */
    bool deskewInfo()
    {
        std::lock_guard<std::mutex> lock1(imuLock);
        std::lock_guard<std::mutex> lock2(odoLock);

        // make sure IMU data available for the scan
        // IMU 데이터에 라이다 데이터가 포함되어 있지 않으면 처리하지 않음
        if (imuQueue.empty() ||
            stamp2Sec(imuQueue.front().header.stamp) > timeScanCur ||
            stamp2Sec(imuQueue.back().header.stamp) < timeScanEnd)
        {
            RCLCPP_INFO(get_logger(), "Waiting for IMU data ...");
            return false;
        }

        // 현재 프레임과 관련된 IMU 데이터 처리
        imuDeskewInfo();    

        // 
        odomDeskewInfo();

        return true;
    }

    /**
     * 현재 프레임과 관련된 IMU 데이터 처리
     * 1. 현재 라이다 프레임의 시작 및 종료 시간 내의 IMU 데이터를 반복하여 처리
     * 2. IMU의 roll-pitch-yaw 오일러 각도 중 현재 프레임의 초기 각도로 설정
     * 참고: IMU 데이터는 이미 LiADAR 좌표계로 변환되었음
    */
    void imuDeskewInfo()
    {
        cloudInfo.imu_available = false;

        // 현재 프레임 시작 시간 이전의 IMU 데이터를 큐에서 제거
        while (!imuQueue.empty())
        {
            if (stamp2Sec(imuQueue.front().header.stamp) < timeScanCur - 0.01)
                imuQueue.pop_front();
            else
                break;
        }

        if (imuQueue.empty())
            return;

        imuPointerCur = 0;

        // 현재 프레임 시작 및 종료 시간 내의 IMU 데이터 처리
        for (int i = 0; i < (int)imuQueue.size(); ++i)
        {
            sensor_msgs::msg::Imu thisImuMsg = imuQueue[i];
            double currentImuTime = stamp2Sec(thisImuMsg.header.stamp);

            // get roll, pitch, and yaw estimation for this scan
            // 현재 IMU 시간이 현재 라이다 프레임의 시작 시간보다 이전인 경우, 현재 라이다 프레임의 초기 각도로 설정할 IMU의 roll-pitch-yaw 오일러 각도 추출
            if (currentImuTime <= timeScanCur)
                imuRPY2rosRPY(&thisImuMsg, &cloudInfo.imu_roll_init, &cloudInfo.imu_pitch_init, &cloudInfo.imu_yaw_init);
            // 현재 라이다 프레임 종료 시간 이후의 IMU 데이터는 처리하지 않음
            if (currentImuTime > timeScanEnd + 0.01)
                break;
            // 첫 번째 IMU 데이터 프레임의 초기 회전 각도를 설정
            if (imuPointerCur == 0){
                imuRotX[0] = 0;
                imuRotY[0] = 0;
                imuRotZ[0] = 0;
                imuTime[0] = currentImuTime;
                ++imuPointerCur;
                continue;
            }

            // get angular velocity
            // 현재 IMU의 각속도 추출
            double angular_x, angular_y, angular_z;
            imuAngular2rosAngular(&thisImuMsg, &angular_x, &angular_y, &angular_z);

            // integrate rotation
            // 현재 IMU 데이터와 이전 IMU 데이터 간의 시간 차이 계산
            double timeDiff = currentImuTime - imuTime[imuPointerCur-1];
            // 현재 시점의 회전 각도 = 이전 시점의 회전 각도 + 각속도 * 시간 차이
            imuRotX[imuPointerCur] = imuRotX[imuPointerCur-1] + angular_x * timeDiff; 
            imuRotY[imuPointerCur] = imuRotY[imuPointerCur-1] + angular_y * timeDiff;
            imuRotZ[imuPointerCur] = imuRotZ[imuPointerCur-1] + angular_z * timeDiff;
            imuTime[imuPointerCur] = currentImuTime;
            ++imuPointerCur;
        }

        --imuPointerCur;
        // 적합한 IMU 데이터가 없으면 반환
        if (imuPointerCur <= 0)
            return;

        cloudInfo.imu_available = true;
    }

    /**
     * 현재 프레임에 해당하는 IMU Odometry 처리
     * 1. 현재 라이다 프레임의 시작과 끝 사이에 있는 IMU Odometry 데이터를 반복하여 처리
     * 2. 시작 및 끝 시간에 해당하는 IMU Odometry를 사용하여 상대적인 pose 변환 계산 및 이동 변화량 저장
     * 참고: IMU 데이터는 이미 LiADAR 좌표계로 변환되었음
    */
    void odomDeskewInfo()
    {
        cloudInfo.odom_available = false;

        // 현재 라이다 프레임 시작 시간 0.01초 이전의 IMU Odometry 데이터를 제거
        while (!odomQueue.empty())
        {
            if (stamp2Sec(odomQueue.front().header.stamp) < timeScanCur - 0.01)
                odomQueue.pop_front();
            else
                break;
        }

        if (odomQueue.empty())
            return;

        // 현재 라이다 프레임 시작 시간 이후의 IMU Odometry 데이터일 경우 함수 종료
        if (stamp2Sec(odomQueue.front().header.stamp) > timeScanCur)
            return;

        // get start odometry at the beinning of the scan
        // 현재 라이다 프레임의 시작 시간에 해당하는 IMU Odometry 추출
        nav_msgs::msg::Odometry startOdomMsg;

        for (int i = 0; i < (int)odomQueue.size(); ++i)
        {
            startOdomMsg = odomQueue[i];

            if (stamp2Sec(startOdomMsg.header.stamp) < timeScanCur)
                continue;
            else
                break;
        }

        // IMU Odometry의 pose 각도 추출
        tf2::Quaternion orientation;
        tf2::fromMsg(startOdomMsg.pose.pose.orientation, orientation);

        double roll, pitch, yaw;
        tf2::Matrix3x3(orientation).getRPY(roll, pitch, yaw);

        // Initial guess used in mapOptimization
        // 현재 라이다 프레임의 초기 위치 및 자세를 LiDAR 좌표계로 설정하고 나중에 mapOptimization에 사용
        cloudInfo.initial_guess_x = startOdomMsg.pose.pose.position.x;
        cloudInfo.initial_guess_y = startOdomMsg.pose.pose.position.y;
        cloudInfo.initial_guess_z = startOdomMsg.pose.pose.position.z;
        cloudInfo.initial_guess_roll = roll;
        cloudInfo.initial_guess_pitch = pitch;
        cloudInfo.initial_guess_yaw = yaw;

        cloudInfo.odom_available = true;

        // get end odometry at the end of the scan
        odomDeskewFlag = false;

        // 만약 현재 라이다 프레임 종료시간 이전의 IMU Odometry 데이터가 있다면 함수 종료
        if (stamp2Sec(odomQueue.back().header.stamp) < timeScanEnd)
            return;

        // 현재 라이다 프레임 종료 시간에 해당하는 IMU Odometry 추출
        nav_msgs::msg::Odometry endOdomMsg;
        for (int i = 0; i < (int)odomQueue.size(); ++i)
        {
            endOdomMsg = odomQueue[i];

            if (stamp2Sec(endOdomMsg.header.stamp) < timeScanEnd)
                continue;
            else
                break;
        }

        // 시작 및 종료 시간에 해당하는 IMU Odometry의 공분산이 같지 않다면 함수 종료
        if (int(round(startOdomMsg.pose.covariance[0])) != int(round(endOdomMsg.pose.covariance[0])))
            return;

        Eigen::Affine3f transBegin = pcl::getTransformation(startOdomMsg.pose.pose.position.x, startOdomMsg.pose.pose.position.y, startOdomMsg.pose.pose.position.z, roll, pitch, yaw);

        tf2::fromMsg(endOdomMsg.pose.pose.orientation, orientation);
        tf2::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
        Eigen::Affine3f transEnd = pcl::getTransformation(endOdomMsg.pose.pose.position.x, endOdomMsg.pose.pose.position.y, endOdomMsg.pose.pose.position.z, roll, pitch, yaw);

        // 시작 및 종료 시간에 해당하는 IMU Odometry의 상대적인 변환을 계산???
        Eigen::Affine3f transBt = transBegin.inverse() * transEnd;

        // 상대적인 변환에서 이동 및 회전 변화량을 추출 (오일러 각도로)
        float rollIncre, pitchIncre, yawIncre;
        pcl::getTranslationAndEulerAngles(transBt, odomIncreX, odomIncreY, odomIncreZ, rollIncre, pitchIncre, yawIncre);

        odomDeskewFlag = true;
    }

    /**
     * 현재 프레임이 시작 및 종료 시간 범위 내에서 특정 시간에 대한 회전 (시작시간 대비 회전각도변화량)을 찾음
    */
    void findRotation(double pointTime, float *rotXCur, float *rotYCur, float *rotZCur)
    {
        *rotXCur = 0; *rotYCur = 0; *rotZCur = 0;

        // pointTime이 imuTime 내에서 어디에 위치하는지 찾음
        int imuPointerFront = 0;
        while (imuPointerFront < imuPointerCur)
        {
            if (pointTime < imuTime[imuPointerFront])
                break;
            ++imuPointerFront;
        }

        // pointTime이 imuTime내에 없는 경우 또는 첫 번째 IMU 데이터인 경우, 가장 가까운 회전 변화량을 사용
        if (pointTime > imuTime[imuPointerFront] || imuPointerFront == 0)
        {
            *rotXCur = imuRotX[imuPointerFront];
            *rotYCur = imuRotY[imuPointerFront];
            *rotZCur = imuRotZ[imuPointerFront];
        } 
        // pointTime을 둘러싼 두개(앞,뒤)의 IMU 데이터 사이를 보간하여 현재 시간의 회전 변화량을 계산
        else {
            int imuPointerBack = imuPointerFront - 1;
            double ratioFront = (pointTime - imuTime[imuPointerBack]) / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
            double ratioBack = (imuTime[imuPointerFront] - pointTime) / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
            *rotXCur = imuRotX[imuPointerFront] * ratioFront + imuRotX[imuPointerBack] * ratioBack;
            *rotYCur = imuRotY[imuPointerFront] * ratioFront + imuRotY[imuPointerBack] * ratioBack;
            *rotZCur = imuRotZ[imuPointerFront] * ratioFront + imuRotZ[imuPointerBack] * ratioBack;
        }
    }

    /**
     * 현재 프레임의 시작 및 종료 시간 범위 내에서 특정 시간에 대한 이동 (시작시간 대비 이동변화량)을 찾음
     */
    void findPosition(double relTime, float *posXCur, float *posYCur, float *posZCur)
    {
        // 센서의 이동 속도가 상대적으로 느린 경우 (예: 걷는 속도 또는 정지 상태) 위치 보정의 효과가 적을 수 있음
        *posXCur = 0; *posYCur = 0; *posZCur = 0;
        *posXCur = 0; *posYCur = 0; *posZCur = 0;

        // If the sensor moves relatively slow, like walking speed, positional deskew seems to have little benefits. Thus code below is commented.

        // if (cloudInfo.odomAvailable == false || odomDeskewFlag == false)
        //     return;

        // float ratio = relTime / (timeScanEnd - timeScanCur);

        // *posXCur = ratio * odomIncreX;
        // *posYCur = ratio * odomIncreY;
        // *posZCur = ratio * odomIncreZ;
    }

    /**
     * 라이다 motion distortion 보정
     * 현재 프레임의 시작 및 종료 시간 내의 IMU 데이터를 사용하여 회전변화량 및 IMU Odometry 데이터를 사용하여 
     * 이동변화량을 계산하여 모든 시간에 대한 라이다 포인트 위치를 첫 번째 라이다 포인트 좌표계로 변환하여 motion distortion 수행
    */
    PointType deskewPoint(PointType *point, double relTime)
    {
        if (deskewFlag == -1 || cloudInfo.imu_available == false)
            return *point;

        // relTime은 현재 라이다 포인트가 현재 라이다 프레임의 시작 시간으로부터 얼마나 떨어져 있는지 나타내며, pointTime은 현재 라이다 포인트의 timestamp
        // pointTime = timeScanCur + relTime
        double pointTime = timeScanCur + relTime; 

        // 현재 라이다 포인트의 시간에 따른 회전 변화량 계산
        float rotXCur, rotYCur, rotZCur;
        findRotation(pointTime, &rotXCur, &rotYCur, &rotZCur);

        // 현재 라이다 포인트의 시간에 따른 이동 변화량 계산
        float posXCur, posYCur, posZCur;
        findPosition(relTime, &posXCur, &posYCur, &posZCur);

        // 첫 번째 포인트의 위치 변환(변화량 없음) 및 역행렬 계산
        if (firstPointFlag == true)
        {
            transStartInverse = (pcl::getTransformation(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur)).inverse();
            firstPointFlag = false;
        }

        // transform points to start
        // 현재 시간에 대한 라이다 포인트와 첫 번째 라이다 포인트 간의 위치 변환 계산
        Eigen::Affine3f transFinal = pcl::getTransformation(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur);
        Eigen::Affine3f transBt = transStartInverse * transFinal;

        // 현재 라이다 포인트를 첫 번쨰 라이다 포인트 좌표계로 변환
        PointType newPoint;
        newPoint.x = transBt(0,0) * point->x + transBt(0,1) * point->y + transBt(0,2) * point->z + transBt(0,3);
        newPoint.y = transBt(1,0) * point->x + transBt(1,1) * point->y + transBt(1,2) * point->z + transBt(1,3);
        newPoint.z = transBt(2,0) * point->x + transBt(2,1) * point->y + transBt(2,2) * point->z + transBt(2,3);
        newPoint.intensity = point->intensity;

        return newPoint;
    }

    //// motion distortion 보정 후에 포인트 클라우드 저장 및 유효한 특징점 추출 및 게시 
    /**
    * 현재 프레임의 라이다 포인트 클라우드 motion distortion 보정
    * 1. 라이다 포인트의 거리와 스캔라인이 유효한지 확인
    * 2. 라이다 motion distortion 보정 및 라이다 포인트 저장
    */
    void projectPointCloud()
    {
        int cloudSize = laserCloudIn->points.size();
        // range image projection
        // 현재 프레임의 라이다 포인트 클라우드를 반복 처리
        for (int i = 0; i < cloudSize; ++i)
        {
            // PCL 형식으로 현재 포인트 클라우드 복사
            PointType thisPoint;
            thisPoint.x = laserCloudIn->points[i].x;
            thisPoint.y = laserCloudIn->points[i].y;
            thisPoint.z = laserCloudIn->points[i].z;
            thisPoint.intensity = laserCloudIn->points[i].intensity;

            // 거리 확인
            float range = pointDistance(thisPoint);
            if (range < lidarMinRange || range > lidarMaxRange)
                continue;

            // 스캔 라인 확인
            int rowIdn = laserCloudIn->points[i].ring;  
            if (rowIdn < 0 || rowIdn >= N_SCAN)
                continue;

            // 다운 샘플링된 스캔 라인은 건너뜀
            if (rowIdn % downsampleRate != 0)
                continue;

            int columnIdn = -1;

            // 수평 스캔 각도 단계 (한바뀌 스캔: 1024번, 스캔 간격: ?)
            if (sensor == SensorType::VELODYNE || sensor == SensorType::OUSTER)
            {
                float horizonAngle = atan2(thisPoint.x, thisPoint.y) * 180 / M_PI;
                static float ang_res_x = 360.0/float(Horizon_SCAN);
                columnIdn = -round((horizonAngle-90.0)/ang_res_x) + Horizon_SCAN/2;
                if (columnIdn >= Horizon_SCAN)
                    columnIdn -= Horizon_SCAN;
            }
            else if (sensor == SensorType::LIVOX)
            {
                columnIdn = columnIdnCountVec[rowIdn];
                columnIdnCountVec[rowIdn] += 1;
            }

            if (columnIdn < 0 || columnIdn >= Horizon_SCAN)
                continue;

            // 이미 처리한 포인트는 건너뜀
            if (rangeMat.at<float>(rowIdn, columnIdn) != FLT_MAX)
                continue;
            // 라이다 motion distortion 보정
            thisPoint = deskewPoint(&thisPoint, laserCloudIn->points[i].time);

            // 포인트 거리를 행렬에 저장
            rangeMat.at<float>(rowIdn, columnIdn) = range;

            // 1차원 인덱스로 변환하고 보정된 포인트 저장
            int index = columnIdn + rowIdn * Horizon_SCAN;
            fullCloud->points[index] = thisPoint;
        }
    }


    /**
     * 유효한 라이다 포인트 추출 및 extractedCloud에 저장
    */
    void cloudExtraction()
    {
        // 유효한 라이다 포인트 수
        int count = 0;
        
        // extract segmented cloud for lidar odometry
        // 모든 스캔 라인 반복
        for (int i = 0; i < N_SCAN; ++i)
        {
            // 각 스캔 라인의 시작점에서 5번쨰 포인트가 배열에서의 인덱스 기록
            cloudInfo.start_ring_index[i] = count - 1 + 5;
            for (int j = 0; j < Horizon_SCAN; ++j)
            {
                // 유효한 라이다 포인트
                if (rangeMat.at<float>(i,j) != FLT_MAX)
                {
                    // mark the points' column index for marking occlusion later
                    //해당 스캔 라인에서의 포인트 인덱스를 기록
                    cloudInfo.point_col_ind[count] = j;
                    // save range info
                    // 포인트의 거리를 기록
                    cloudInfo.point_range[count] = rangeMat.at<float>(i,j);
                    // save extracted cloud
                    // 유효한 라이다 포인트를 추출된 클라우드에 추가
                    extractedCloud->push_back(fullCloud->points[j + i*Horizon_SCAN]);
                    // size of extracted cloud
                    ++count;
                }
            }
            // 각 스캔 라인의 마지막에서 5번째 포인트가 배열에서의 인덱스를 기록
            cloudInfo.end_ring_index[i] = count -1 - 5;
        }
    }
    
    /**
     * 현재 프레임의 보정된 포인트 클라우드 및 유효한 포인트를 게시
    */
    void publishClouds()
    {
        cloudInfo.header = cloudHeader;
        cloudInfo.cloud_deskewed  = publishCloud(pubExtractedCloud, extractedCloud, cloudHeader.stamp, lidarFrame);
        pubLaserCloudInfo->publish(cloudInfo);
    }
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);

    rclcpp::NodeOptions options;
    options.use_intra_process_comms(true);
    rclcpp::executors::MultiThreadedExecutor exec;

    auto IP = std::make_shared<ImageProjection>(options);
    exec.add_node(IP);

    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "\033[1;32m----> Image Projection Started.\033[0m");

    exec.spin();

    rclcpp::shutdown();
    return 0;
}
