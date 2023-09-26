#include "utility.hpp"
#include "lio_sam/msg/cloud_info.hpp"

// 레이저 포인트 곡률 정의
struct smoothness_t{ 
    float value;    // 곡륙 값
    size_t ind;     // 레이저 포인트 1차원 인덱스
};

/**
 * 곡률 비교 함수, 작은 값부터 정렬
*/
struct by_value{ 
    bool operator()(smoothness_t const &left, smoothness_t const &right) { 
        return left.value < right.value;
    }
};


//// FeatureExtraction class
class FeatureExtraction : public ParamServer
{

// 맴버 변수
public:

    // 레이저 클라우드 정보를 구독하는 변수
    rclcpp::Subscription<lio_sam::msg::CloudInfo>::SharedPtr subLaserCloudInfo;

    // 현재 레이저 프레임에서 특징을 추출한 후의 포인트 클라우드 정보를 발행하는 변수
    rclcpp::Publisher<lio_sam::msg::CloudInfo>::SharedPtr pubLaserCloudInfo;
    // 현재 레이저 프레임에서 추출한 코너 포인트 클라우드를 발행하는 변수
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubCornerPoints;
    // 현재 레이저 프레임에서 추출한 평면 포인트 클라우드를 발행하는 변수
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubSurfacePoints;

    // 현재 레이저 프레임에서 motion distortion 보정 후의 유효한 포인트 클라우드
    pcl::PointCloud<PointType>::Ptr extractedCloud;
    // 현재 레이저 프레임의 코너 포인트 클라우드
    pcl::PointCloud<PointType>::Ptr cornerCloud;
    // 현재 레이저 프레임의 평면 포인트 클라우드
    pcl::PointCloud<PointType>::Ptr surfaceCloud;
    // 포인트 클라우드 다운샘플링 필터
    pcl::VoxelGrid<PointType> downSizeFilter;

    // 현재 레이저 프레임의 포인트 클라우드 정보, 
    // (motion distortion 보정 포인트 데이터, 초기 위치, 자세 각도, 유효한 포인트 클라우드 데이터, 코너 포인트 클라우드, 평면 포인트 클라우드)
    lio_sam::msg::CloudInfo cloudInfo;
    std_msgs::msg::Header cloudHeader;

    // 현재 레이저 프레임의 포인트 클라우드 곡률 정보
    std::vector<smoothness_t> cloudSmoothness;
    float *cloudCurvature;
    // 특징 추출 플래그, (1은 가려진 포인트, 평행한 포인트 또는 이미 특징 추출이 수행된 포인트를 의미. 0은 아직 추출되지 않은 포인트를 의미)
    int *cloudNeighborPicked;
    // 1은 코너 포인트, -1은 평명 포인트를 의미
    int *cloudLabel;

    /**
     * 생성자
    */
    FeatureExtraction(const rclcpp::NodeOptions & options) :
        ParamServer("lio_sam_featureExtraction", options)
    {
        // 현재 레이저 프레임의 motion distortion이 보정된 포인트 클라우드 정보를 구독
        subLaserCloudInfo = create_subscription<lio_sam::msg::CloudInfo>(
            "lio_sam/deskew/cloud_info", qos,
            std::bind(&FeatureExtraction::laserCloudInfoHandler, this, std::placeholders::_1));

        // 특징을 추출한 현재 레이저 프레임의 포인트 클라우드 정보를 발행
        pubLaserCloudInfo = create_publisher<lio_sam::msg::CloudInfo>(
            "lio_sam/feature/cloud_info", qos);

        // 현재 레이저 프레임의 코너 포인트 클라우드를 발행
        pubCornerPoints = create_publisher<sensor_msgs::msg::PointCloud2>(
            "lio_sam/feature/cloud_corner", 1);

        // 현재 레이저 프레임의 서피스 포인트 클라우드를 발행
        pubSurfacePoints = create_publisher<sensor_msgs::msg::PointCloud2>(
            "lio_sam/feature/cloud_surface", 1);

        // 초기화 수행 작업
        initializationValue();
    }

    /**
     * 초기화 
    */
    void initializationValue()
    {
        // cloudSmmothness 백터를 크기 (N_SCAN*Horizon_SCAN)로 설정
        cloudSmoothness.resize(N_SCAN*Horizon_SCAN);

        // 다운샘플링 필터를 설정하고 초기화
        downSizeFilter.setLeafSize(odometrySurfLeafSize, odometrySurfLeafSize, odometrySurfLeafSize);

        // 포인트 클라우드를 저장할 포인터들을 초기화함
        extractedCloud.reset(new pcl::PointCloud<PointType>());
        cornerCloud.reset(new pcl::PointCloud<PointType>());
        surfaceCloud.reset(new pcl::PointCloud<PointType>());

        // 포인트 클라우드의 곡률을 저장할 배열을 초기화
        cloudCurvature = new float[N_SCAN*Horizon_SCAN];
        // 포인트 클라우드의 이웃 포인트 정보를 저장할 배열을 초기화
        cloudNeighborPicked = new int[N_SCAN*Horizon_SCAN];
        // 포인트 클라우드의 레이블 정보를 저장할 배열을 초기화 (코너 포인트인지 평면 포인트인지)
        cloudLabel = new int[N_SCAN*Horizon_SCAN];
    }

    /**
     * 현재 레이저 프레임의 motion distortion이 보정된 포인트 클라우드 정보를 처리
     * 1. 각 포인트의 곡률을 계산합니다.
     * 2. 가려진 점 또는 평행한 점으로 표시된 점은 특징 추출을 하지 않도록 표시합니다.
     * 3. 포인트 클라우드의 코너 포인트 및 평면 포인트 특징을 추출합니다.
     *      1) 각 스캔 라인을 반복하면서 한 라인을 6개의 구간으로 나누고 ,
     *         각 구간에서 20개의 코너 포인트와 제한 없이 많은 평면 포인트를 추출하여 코너 포인트 및 평면 포인트 세트에 추가합니다.
     *      2) 코너 포인트가 아닌 점은 모두 평면 포인트로 간주하고, 평면 포인트 클라우드 세트에 추가하여, 최종적으로 다운 샘플링합니다.
     * 4. 코너 포인트 및 평면 포인트 클라우드를 게시하고, 특징이 추출된 현재 라이다 프레임의 포인트 클라우드 정보를 게시합니다.
    */
    void laserCloudInfoHandler(const lio_sam::msg::CloudInfo::SharedPtr msgIn)
    {
        cloudInfo = *msgIn; // new cloud info - 라이다 클라우드 정보를 복사
        cloudHeader = msgIn->header; // new cloud header - 라이다 클라우드 정보의 헤더를 가져옴
        pcl::fromROSMsg(msgIn->cloud_deskewed, *extractedCloud); // new cloud for extraction - 현재 라이다 프레임의 motion distortion된 클라우드 정보를 PCL 포인트 클라우드로 변환합니다.

        // 각 포인트의 곡률을 계산
        calculateSmoothness();

        // 가려진 점 또는 평행한 점으로 표시된 점을 특징 추출하지 않도록 표시
        markOccludedPoints();

        // 포인트 클라우드에서 코너 포인트 및 평면 포인트 특징을 추출
        extractFeatures();

        // 코너 포인트 및 평면 포인트 클라우드를 게시하고, 특징이 추출된 현재 라이다 프레임의 포인트 클라우드 정보를 게시
        publishFeatureCloud();
    }


    /**
     * 현재 라이다 프레임의 포인트 클라우드에서 각 포인트의 곡률 계산 
    */
    void calculateSmoothness()
    {   
        // 현재 포인트의 앞뒤 5개 포인트를 사용하여 현재 포인트의 곡률을 계산 
        int cloudSize = extractedCloud->points.size();
        for (int i = 5; i < cloudSize - 5; i++)
        {
            float diffRange = cloudInfo.point_range[i-5] + cloudInfo.point_range[i-4]
                            + cloudInfo.point_range[i-3] + cloudInfo.point_range[i-2]
                            + cloudInfo.point_range[i-1] - cloudInfo.point_range[i] * 10
                            + cloudInfo.point_range[i+1] + cloudInfo.point_range[i+2]
                            + cloudInfo.point_range[i+3] + cloudInfo.point_range[i+4]
                            + cloudInfo.point_range[i+5];
            
            // 거리 차이 값의 제곱을 곡률 값으로 설정
            cloudCurvature[i] = diffRange*diffRange;//diffX * diffX + diffY * diffY + diffZ * diffZ;

            cloudNeighborPicked[i] = 0;
            cloudLabel[i] = 0;
            
            // cloudSmoothness for sorting
            // 현재 포인트의 곡률값과 라이다 포인트의 1차원 인덱스를 저장
            cloudSmoothness[i].value = cloudCurvature[i];
            cloudSmoothness[i].ind = i;
        }
    }

    /**
     * 가려진 포인트 및 평행 빔 포인트를 표시하고, 특징 추출을 수행하지 않습니다. 
    */
    void markOccludedPoints()
    {
        int cloudSize = extractedCloud->points.size();
        // mark occluded points and parallel beam points
        // 가려진 포인트 및 평행 빔 포인트를 표시
        for (int i = 5; i < cloudSize - 6; ++i)
        {
            // occluded points
            // 현재 포인트와 다음 포인트의 거리 값
            float depth1 = cloudInfo.point_range[i];
            float depth2 = cloudInfo.point_range[i+1];
            
            // 두 라이다 포인트 간의 일차원 인덱스 차이, 같은 스캔 라인에 있으면 1 입니다. (두 포인트 간에 무효한 포인트가 몇개 제거되면 1보다 크게 될 수 있지만 크게 중요하지 않습니다.)
            // 이전 포인트가 스캔 한 주기 끝에 있고 다음 포인트가 다른 스캔 라인의 시작에 위치하는 경우 값이 크게 될 수 있습니다. -> 서로 다른 스캔라인의 차이일때
            int columnDiff = std::abs(int(cloudInfo.point_col_ind[i+1] - cloudInfo.point_col_ind[i]));

            // 두 포인트가 같은 스캔라인에 있고 거리가 0.3보다 큰 경우, 가려진 관계가 있음을 의미합니다.( 두 포인트가 같은 평면에 있지 않으면 거리가 크게 차이나지 않습니다.)
            // 먼 거리의 포인트가 가려지고 해당 포인트 및 인접한 5개 포인트를 표시하고 이후에는 특징 추출을 수행하지 않습니다.
            if (columnDiff < 10){
                // 10 pixel diff in range image
                if (depth1 - depth2 > 0.3){
                    cloudNeighborPicked[i - 5] = 1;
                    cloudNeighborPicked[i - 4] = 1;
                    cloudNeighborPicked[i - 3] = 1;
                    cloudNeighborPicked[i - 2] = 1;
                    cloudNeighborPicked[i - 1] = 1;
                    cloudNeighborPicked[i] = 1;
                }else if (depth2 - depth1 > 0.3){
                    cloudNeighborPicked[i + 1] = 1;
                    cloudNeighborPicked[i + 2] = 1;
                    cloudNeighborPicked[i + 3] = 1;
                    cloudNeighborPicked[i + 4] = 1;
                    cloudNeighborPicked[i + 5] = 1;
                    cloudNeighborPicked[i + 6] = 1;
                }
            }
            // parallel beam
            // 현재 포인트가 라이다 빔 방향과 평행한지 여부를 판단하기 위해 이전 포인트와 다음 포인트를 사용합니다.
            float diff1 = std::abs(float(cloudInfo.point_range[i-1] - cloudInfo.point_range[i]));
            float diff2 = std::abs(float(cloudInfo.point_range[i+1] - cloudInfo.point_range[i]));

            // 평행하면 표시합니다.--> 
            if (diff1 > 0.02 * cloudInfo.point_range[i] && diff2 > 0.02 * cloudInfo.point_range[i])
                cloudNeighborPicked[i] = 1;
        }
    }


    /**
     * 포인트 클라우드에서 각 스캔 라인에 대해 모서리 및 평면 특징을 추출합니다.
     * 1. 각 스캔 라인에 대해 스캔 라인 내에서 균등하게 20개의 모서리 및 임의의 수의 평면 포인트를 추출합니다.
     * 2. 모서리가 아닌 포인트로 간주되는 경우 이를 평면 포인트로 간주하고 평면 포인트 클라우드에 추가합니다.
     *      추출된 포인트는 최종적으로 다운 샘플링됩니다.
    */
    void extractFeatures()
    {
        cornerCloud->clear();
        surfaceCloud->clear();

        pcl::PointCloud<PointType>::Ptr surfaceCloudScan(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr surfaceCloudScanDS(new pcl::PointCloud<PointType>());

        // 각 스캔 라인에 대한 루프
        for (int i = 0; i < N_SCAN; i++)
        {
            surfaceCloudScan->clear();

            // 스캔 라인 내에서 포인트 클라우드 데이터를 6개의 세그먼트로 분할하고, 각 세그먼트에서 일정 수의 특징을 추출합니다.
            for (int j = 0; j < 6; j++)
            {
                // 각 세그먼트의 포인트 시작 및 끝 인덱스
                int sp = (cloudInfo.start_ring_index[i] * (6 - j) + cloudInfo.end_ring_index[i] * j) / 6;
                int ep = (cloudInfo.start_ring_index[i] * (5 - j) + cloudInfo.end_ring_index[i] * (j + 1)) / 6 - 1;

                if (sp >= ep)
                    continue;

                // 포인트를 곡률 기준으로 오름차순으로 정렬
                std::sort(cloudSmoothness.begin()+sp, cloudSmoothness.begin()+ep, by_value());

                // 곡률을 기준으로 내림차순으로 포인트를 반복합니다.
                int largestPickedNum = 0;
                for (int k = ep; k >= sp; k--)
                {
                    // 포인트 인덱스
                    int ind = cloudSmoothness[k].ind;
                    // 현재 포인트가 처리되지 않았고, 곡률이 edgeThreshold 보다 크면 코너로 간주합니다.
                    if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] > edgeThreshold)
                    {
                        // 각 세그먼트에서 최대 20개의 코너를 추출합니다.
                        largestPickedNum++;
                        if (largestPickedNum <= 20){
                            // 코너로 표시합니다.
                            cloudLabel[ind] = 1;
                            // 코너 포인트 클라우드에 추가합니다.
                            cornerCloud->push_back(extractedCloud->points[ind]);
                        } else {
                            break;
                        }
                        // 처리 완료를 표시합니다.
                        cloudNeighborPicked[ind] = 1;
                        // 동일한 스캔 라인에서 뒤에 있는 5개의 포인트를 처리하지 않도록 표시하여 특징이 모이는 것을 방지합니다.
                        for (int l = 1; l <= 5; l++)
                        {   
                            int columnDiff = std::abs(int(cloudInfo.point_col_ind[ind + l] - cloudInfo.point_col_ind[ind + l - 1]));
                            if (columnDiff > 10)
                                break;
                            cloudNeighborPicked[ind + l] = 1;
                        }
                        // 동일한 스캔 라인에서 앞에 있는 5개의 포인트를 처리하지 않도록 표시하여 특징이 모이는 것을 방지합니다.
                        for (int l = -1; l >= -5; l--)
                        {
                            int columnDiff = std::abs(int(cloudInfo.point_col_ind[ind + l] - cloudInfo.point_col_ind[ind + l + 1]));
                            if (columnDiff > 10)
                                break;
                            cloudNeighborPicked[ind + l] = 1;
                        }
                    }
                }

                // 곡률 기준으로 오름차순으로 포인트를 반복합니다.
                for (int k = sp; k <= ep; k++)
                {
                    // 포인트 인덱스
                    int ind = cloudSmoothness[k].ind;
                    // 현재 포인트가 처리되지 않았고, 곡률이 surfTheshold보다 작으면 평면 포인트로 간주합니다.
                    if (cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] < surfThreshold)
                    {
                        // 평면 포인트로 표시합니다.
                        cloudLabel[ind] = -1;
                        // 처리 완료를 표시합니다.
                        cloudNeighborPicked[ind] = 1;

                        // 동일한 스캔 라인에서 뒤에 있는 5개의 포인트를 처리하지 않도록 표시하여 특징이 모이는 것을 방지합니다.
                        for (int l = 1; l <= 5; l++) {
                            int columnDiff = std::abs(int(cloudInfo.point_col_ind[ind + l] - cloudInfo.point_col_ind[ind + l - 1]));
                            if (columnDiff > 10)
                                break;

                            cloudNeighborPicked[ind + l] = 1;
                        }
                        // 동일한 스캔 라인에서 앞에 있는 5개의 포인트를 처리하지 않도록 표시하여 특징이 모이는 것을 방지합니다.
                        for (int l = -1; l >= -5; l--) {
                            int columnDiff = std::abs(int(cloudInfo.point_col_ind[ind + l] - cloudInfo.point_col_ind[ind + l + 1]));
                            if (columnDiff > 10)
                                break;

                            cloudNeighborPicked[ind + l] = 1;
                        }
                    }
                }

                // 평면 포인트와 처리되지 않은 포인트는 모두 평면 포인트로 간주하고 평면 포인트 클라우드 스캔에 추가합니다.
                for (int k = sp; k <= ep; k++)
                {
                    if (cloudLabel[k] <= 0){
                        surfaceCloudScan->push_back(extractedCloud->points[k]);
                    }
                }
            }

            // 평면 포인트 클라우드 다운 샘플링
            surfaceCloudScanDS->clear();
            downSizeFilter.setInputCloud(surfaceCloudScan);
            downSizeFilter.filter(*surfaceCloudScanDS);

            // 평면 포인트 클라우드에 추가
            *surfaceCloud += *surfaceCloudScanDS;
        }
    }

    /**
     * 메모리 정리 
    */
    void freeCloudInfoMemory()
    {
        cloudInfo.start_ring_index.clear();
        cloudInfo.end_ring_index.clear();
        cloudInfo.point_col_ind.clear();
        cloudInfo.point_range.clear();
    }

    /**
    * 코너 및 평면 포인트 클라우드를 게시하고, 특징 포인트가 있는 현재 라이다 프레임 포인트 클라우드 정보를 게시합니다.
    */
    void publishFeatureCloud()
    {
        // free cloud info memory
        // 메모리 정리
        freeCloudInfoMemory();
        // save newly extracted features
        // 코너 및 평면 포인트 클라우드를 publish하여 rviz에 표시합니다.
        cloudInfo.cloud_corner = publishCloud(pubCornerPoints,  cornerCloud,  cloudHeader.stamp, lidarFrame);
        cloudInfo.cloud_surface = publishCloud(pubSurfacePoints, surfaceCloud, cloudHeader.stamp, lidarFrame);
        // publish to mapOptimization
        // 코너와 평면 포인트 클라우드 데이터가 추가된 현재 라이다 프레임 포인트 클라우드 정보를 publish하여 mapOptimization에 전달합니다.
        pubLaserCloudInfo->publish(cloudInfo);
    }
};


int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::NodeOptions options;
    options.use_intra_process_comms(true);
    rclcpp::executors::SingleThreadedExecutor exec;

    auto FE = std::make_shared<FeatureExtraction>(options);

    exec.add_node(FE);

    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "\033[1;32m----> Feature Extraction Started.\033[0m");

    exec.spin();

    rclcpp::shutdown();
    return 0;
}
