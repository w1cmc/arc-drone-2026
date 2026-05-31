#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>

#include <iostream>
#include <string>
#include <vector>
#include <cmath>

#include <thread>
#include <mutex>
#include <atomic>
#include <cstring>
#include <netinet/in.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cstdint>
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>

// This program requires ROS2/tf2 for rotation/quaternion conversions.
#ifndef USE_ROS2
#error "Build with -DENABLE_ROS2=ON to enable ROS2/tf2 conversions (no manual fallbacks)."
#endif

namespace {

constexpr int kMaxCorners = 2000;
constexpr double kQualityLevel = 0.01;
constexpr double kMinDistance = 8.0;
constexpr int kBlockSize = 7;
constexpr double kFeatureRefreshRatio = 0.6;
constexpr int kKeyDelayMs = 1;

cv::Mat makeCameraMatrix(const cv::Size& imageSize) {
	const double focal = 0.9 * static_cast<double>(imageSize.width);
	const cv::Point2d principalPoint(imageSize.width * 0.5, imageSize.height * 0.5);
	return (cv::Mat_<double>(3, 3) << focal, 0.0, principalPoint.x,
									  0.0, focal, principalPoint.y,
									  0.0, 0.0, 1.0);
}

struct IMUState {
	std::mutex m;
	double roll = 0.0;
	double pitch = 0.0;
	double yaw = 0.0;
	bool has = false;
	// accelerometer integration state (world frame)
	double lastAccTime = 0.0; // seconds
	double velX = 0.0;
	double velY = 0.0;
	double velZ = 0.0;
	double posDx = 0.0; // integrated displacement since last read (meters)
	double posDy = 0.0;
	double posDz = 0.0;
	bool hasAccel = false;
};

// Note: conversions now use tf2 directly; helper removed.

// Minimal MAVLink v1 ATTITUDE(30) parser from a UDP stream. This is lenient
// (no CRC verification) and only extracts the roll/pitch/yaw floats.
static void mavlinkUdpListener(int port, double accelScale, IMUState &imu, std::atomic<bool> &running) {
	int sock = socket(AF_INET, SOCK_DGRAM, 0);
	if (sock < 0) return;

	struct sockaddr_in addr;
	std::memset(&addr, 0, sizeof(addr));
	addr.sin_family = AF_INET;
	addr.sin_addr.s_addr = INADDR_ANY;
	addr.sin_port = htons(static_cast<uint16_t>(port));

	if (bind(sock, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
		close(sock);
		return;
	}

	constexpr size_t kBuf = 2048;
	uint8_t buffer[kBuf];
	while (running.load()) {
		ssize_t r = recv(sock, buffer, kBuf, 0);
		if (r <= 0) continue;

		for (ssize_t i = 0; i < r; ++i) {
			if (buffer[i] == 0xFE) { // MAVLink v1 header
				if (i + 6 >= r) break;
				uint8_t len = buffer[i + 1];
				// full packet size: 6(header bytes up to msgid) + len + 2(checksum)
				ssize_t full = 6 + len + 2;
				if (i + full > r) continue;
				uint8_t msgid = buffer[i + 5];
				if (msgid == 30 && len >= 28) { // ATTITUDE
					const uint8_t* payload = buffer + i + 6;
					// payload layout: time_boot_ms (4), roll (4), pitch (4), yaw (4), rollspeed, pitchspeed, yawspeed
					float roll_f = 0.f, pitch_f = 0.f, yaw_f = 0.f;
					std::memcpy(&roll_f, payload + 4, sizeof(float));
					std::memcpy(&pitch_f, payload + 8, sizeof(float));
					std::memcpy(&yaw_f, payload + 12, sizeof(float));
					{
						std::lock_guard<std::mutex> lk(imu.m);
						imu.roll = static_cast<double>(roll_f);
						imu.pitch = static_cast<double>(pitch_f);
						imu.yaw = static_cast<double>(yaw_f);
						imu.has = true;
					}
				}
				if (msgid == 27 && len >= 26) { // RAW_IMU
					const uint8_t* payload = buffer + i + 6;
					// payload: time_usec (uint64), xacc(int16), yacc(int16), zacc(int16), xgyro, ygyro, zgyro, xmag,y, z
					uint64_t time_usec = 0;
					std::memcpy(&time_usec, payload + 0, sizeof(uint64_t));
					int16_t xacc = 0, yacc = 0, zacc = 0;
					std::memcpy(&xacc, payload + 8, sizeof(int16_t));
					std::memcpy(&yacc, payload + 10, sizeof(int16_t));
					std::memcpy(&zacc, payload + 12, sizeof(int16_t));

					// convert raw -> m/s^2 using provided scale
					double ax = static_cast<double>(xacc) * accelScale;
					double ay = static_cast<double>(yacc) * accelScale;
					double az = static_cast<double>(zacc) * accelScale;

					double t = static_cast<double>(time_usec) * 1e-6;
					{
						std::lock_guard<std::mutex> lk(imu.m);
						if (imu.lastAccTime > 0.0) {
							double dt = t - imu.lastAccTime;
							if (dt > 0 && dt < 1.0) {
								// rotate body accel to world using latest IMU attitude (tf2)
								double wx, wy, wz;
								if (imu.has) {
									tf2::Quaternion qrot;
									qrot.setRPY(imu.roll, imu.pitch, imu.yaw);
									tf2::Matrix3x3 mat(qrot);
									tf2::Vector3 b(ax, ay, az);
									tf2::Vector3 w = mat * b;
									wx = w.x(); wy = w.y(); wz = w.z();
								} else {
									wx = ax; wy = ay; wz = az;
								}
								// subtract gravity (m/s^2)
								wz -= 9.80665;

								// integrate velocity and position (simple forward Euler)
								imu.velX += wx * dt;
								imu.velY += wy * dt;
								imu.velZ += wz * dt;

								imu.posDx += imu.velX * dt + 0.5 * wx * dt * dt;
								imu.posDy += imu.velY * dt + 0.5 * wy * dt * dt;
								imu.posDz += imu.velZ * dt + 0.5 * wz * dt * dt;
								imu.hasAccel = true;
							}
						}
						imu.lastAccTime = t;
					}
				}
				i += full - 1;
			}
		}
	}

	close(sock);
}

void detectFeatures(const cv::Mat& grayFrame, std::vector<cv::Point2f>& points) {
	cv::goodFeaturesToTrack(
		grayFrame,
		points,
		kMaxCorners,
		kQualityLevel,
		kMinDistance,
		cv::Mat(),
		kBlockSize,
		false,
		0.04);
}

void drawTrajectory(cv::Mat& canvas, const cv::Point2d& position) {
	const int centerX = canvas.cols / 2;
	const int centerY = canvas.rows / 2;
	const int scale = 8;

	cv::circle(canvas,
			   cv::Point(centerX + static_cast<int>(position.x * scale),
						 centerY + static_cast<int>(position.y * scale)),
			   2,
			   cv::Scalar(0, 255, 0),
			   cv::FILLED,
			   cv::LINE_AA);

	cv::circle(canvas, cv::Point(centerX, centerY), 3, cv::Scalar(0, 0, 255), cv::FILLED, cv::LINE_AA);
}

}  // namespace

int main(int argc, char** argv) {
	#ifdef USE_ROS2
		// initialize ROS2
		rclcpp::init(argc, argv);
		auto ros_node = rclcpp::Node::make_shared("optical_flow_bridge");
		auto pose_pub = ros_node->create_publisher<geometry_msgs::msg::PoseStamped>("vision_pose", 10);
		auto odom_pub = ros_node->create_publisher<nav_msgs::msg::Odometry>("vision_odometry", 10);
	#endif
	cv::VideoCapture capture;
	if (argc > 1) {
		capture.open(argv[1]);
	} else {
		capture.open(0);
	}

	if (!capture.isOpened()) {
		std::cerr << "Failed to open video source.\n";
		return 1;
	}

	cv::Mat frame;
	if (!capture.read(frame) || frame.empty()) {
		std::cerr << "Failed to read the first frame.\n";
		return 1;
	}

	cv::Mat previousGray;
	cv::cvtColor(frame, previousGray, cv::COLOR_BGR2GRAY);

	std::vector<cv::Point2f> previousPoints;
	detectFeatures(previousGray, previousPoints);

	cv::Mat pose = cv::Mat::eye(4, 4, CV_64F);
	cv::Mat trajectory = cv::Mat::zeros(600, 600, CV_8UC3);

	const cv::Mat cameraMatrix = makeCameraMatrix(frame.size());

	// Parse optional MAVLink UDP port from command line: --mavport <port>
	int mavPort = 0;
    double accelScale = 0.00980665; // default: raw in milli-g -> m/s^2
	for (int i = 1; i < argc; ++i) {
		if (std::string(argv[i]) == "--mavport" && i + 1 < argc) {
			mavPort = std::atoi(argv[i + 1]);
			break;
		}
	}

	// optional accel scale
	for (int i = 1; i < argc; ++i) {
		if (std::string(argv[i]) == "--accelscale" && i + 1 < argc) {
			accelScale = std::atof(argv[i + 1]);
			break;
		}
	}

	IMUState imu;
	std::atomic<bool> imuRunning{false};
	std::thread imuThread;
	if (mavPort > 0) {
		imuRunning = true;
		imuThread = std::thread(mavlinkUdpListener, mavPort, accelScale, std::ref(imu), std::ref(imuRunning));
		std::cout << "MAVLink listener started on UDP port " << mavPort << " (accelScale=" << accelScale << ")\n";
	}

	while (true) {
		if (!capture.read(frame) || frame.empty()) {
			break;
		}

		cv::Mat currentGray;
		cv::cvtColor(frame, currentGray, cv::COLOR_BGR2GRAY);

		if (previousPoints.size() < 8) {
			detectFeatures(previousGray, previousPoints);
		}

		if (previousPoints.size() < 8) {
			previousGray = currentGray;
			continue;
		}

		std::vector<cv::Point2f> currentPoints;
		std::vector<uchar> status;
		std::vector<float> errors;

		cv::calcOpticalFlowPyrLK(
			previousGray,
			currentGray,
			previousPoints,
			currentPoints,
			status,
			errors,
			cv::Size(21, 21),
			3,
			cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01),
			0,
			1e-4);

		std::vector<cv::Point2f> previousInliers;
		std::vector<cv::Point2f> currentInliers;
		previousInliers.reserve(previousPoints.size());
		currentInliers.reserve(currentPoints.size());

		for (size_t i = 0; i < status.size(); ++i) {
			if (status[i]) {
				previousInliers.push_back(previousPoints[i]);
				currentInliers.push_back(currentPoints[i]);
			}
		}

		if (previousInliers.size() >= 8) {
			cv::Mat inlierMask;
			const cv::Mat essentialMatrix = cv::findEssentialMat(
				previousInliers,
				currentInliers,
				cameraMatrix,
				cv::RANSAC,
				0.999,
				1.0,
				inlierMask);

			if (!essentialMatrix.empty()) {
				cv::Mat rotation;
				cv::Mat translation;
				const int inliers = cv::recoverPose(
					essentialMatrix,
					previousInliers,
					currentInliers,
					cameraMatrix,
					rotation,
					translation,
					inlierMask);

				if (inliers > 0) {
					cv::Mat transform = cv::Mat::eye(4, 4, CV_64F);
					rotation.copyTo(transform(cv::Rect(0, 0, 3, 3)));
					translation.copyTo(transform(cv::Rect(3, 0, 1, 3)));

					// If IMU orientation is available, use it to replace the rotation
					{
						std::lock_guard<std::mutex> lk(imu.m);
						if (imu.has) {
							// build rotation matrix from IMU RPY using tf2
							tf2::Quaternion qrot;
							qrot.setRPY(imu.roll, imu.pitch, imu.yaw);
							tf2::Matrix3x3 mat(qrot);
							cv::Mat Rimu = cv::Mat::zeros(3,3,CV_64F);
							{
								tf2::Vector3 r;
								mat.getRow(0, r); Rimu.at<double>(0,0)=r.x(); Rimu.at<double>(0,1)=r.y(); Rimu.at<double>(0,2)=r.z();
								mat.getRow(1, r); Rimu.at<double>(1,0)=r.x(); Rimu.at<double>(1,1)=r.y(); Rimu.at<double>(1,2)=r.z();
								mat.getRow(2, r); Rimu.at<double>(2,0)=r.x(); Rimu.at<double>(2,1)=r.y(); Rimu.at<double>(2,2)=r.z();
							}
							Rimu.copyTo(transform(cv::Rect(0, 0, 3, 3)));
						}
					}

					// Scale recovery using integrated accelerometer displacement (RAW_IMU integration)
					double imu_dx = 0.0, imu_dy = 0.0, imu_dz = 0.0;
					{
						std::lock_guard<std::mutex> lk(imu.m);
						if (imu.hasAccel) {
							imu_dx = imu.posDx;
							imu_dy = imu.posDy;
							imu_dz = imu.posDz;
							// reset accumulated displacement after consuming
							imu.posDx = imu.posDy = imu.posDz = 0.0;
						}
					}

					if (imu_dx != 0.0 || imu_dy != 0.0 || imu_dz != 0.0) {
						double imu_norm = std::sqrt(imu_dx*imu_dx + imu_dy*imu_dy + imu_dz*imu_dz);
						double vo_norm = cv::norm(translation);
						if (imu_norm > 1e-6 && vo_norm > 1e-6) {
							double scale = imu_norm / vo_norm;
							translation *= scale;
							translation.copyTo(transform(cv::Rect(3, 0, 1, 3)));
							std::cout << " [scale=" << scale << "]";
						}
					}

					pose = pose * transform.inv();

					const cv::Point2d position(pose.at<double>(0, 3), pose.at<double>(2, 3));
					drawTrajectory(trajectory, position);

					std::cout << "Pose [x y z]: "
						  << pose.at<double>(0, 3) << ' '
						  << pose.at<double>(1, 3) << ' '
						  << pose.at<double>(2, 3);
					{
						std::lock_guard<std::mutex> lk(imu.m);
						if (imu.has) {
							std::cout << "  IMU [r p y]: " << imu.roll << ' ' << imu.pitch << ' ' << imu.yaw;
						}
					}
					std::cout << '\n';
					#ifdef USE_ROS2
						// publish PoseStamped
						geometry_msgs::msg::PoseStamped ps;
						ps.header.stamp = ros_node->now();
						ps.header.frame_id = "map";
						ps.pose.position.x = pose.at<double>(0,3);
						ps.pose.position.y = pose.at<double>(1,3);
						ps.pose.position.z = pose.at<double>(2,3);
						// fill orientation from transform rotation using tf2 when available
						{
							cv::Mat R = pose(cv::Rect(0,0,3,3)).clone();
							double m00 = R.at<double>(0,0), m01 = R.at<double>(0,1), m02 = R.at<double>(0,2);
							double m10 = R.at<double>(1,0), m11 = R.at<double>(1,1), m12 = R.at<double>(1,2);
							double m20 = R.at<double>(2,0), m21 = R.at<double>(2,1), m22 = R.at<double>(2,2);
							tf2::Matrix3x3 mat(m00, m01, m02, m10, m11, m12, m20, m21, m22);
							tf2::Quaternion q;
							mat.getRotation(q);
							ps.pose.orientation.w = q.getW();
							ps.pose.orientation.x = q.getX();
							ps.pose.orientation.y = q.getY();
							ps.pose.orientation.z = q.getZ();
						}
						pose_pub->publish(ps);
						// publish Odometry similarly
						nav_msgs::msg::Odometry od;
						od.header = ps.header;
						od.pose.pose = ps.pose;
						od.child_frame_id = "base_link";
						odom_pub->publish(od);
						// allow ROS2 background work
						rclcpp::spin_some(ros_node);
					#endif

					previousPoints.clear();
					previousPoints.reserve(currentInliers.size());
					for (size_t i = 0; i < currentInliers.size(); ++i) {
						if (!inlierMask.empty() && inlierMask.at<uchar>(static_cast<int>(i))) {
							previousPoints.push_back(currentInliers[i]);
						}
					}
				}
			}
		}

		if (previousPoints.size() < static_cast<size_t>(kMaxCorners * kFeatureRefreshRatio)) {
			std::vector<cv::Point2f> freshFeatures;
			detectFeatures(currentGray, freshFeatures);
			if (!freshFeatures.empty()) {
				previousPoints.insert(previousPoints.end(), freshFeatures.begin(), freshFeatures.end());
			}
		}

		for (const auto& point : previousPoints) {
			cv::circle(frame, point, 2, cv::Scalar(0, 255, 0), cv::FILLED, cv::LINE_AA);
		}

		cv::imshow("Optical Flow", frame);
		cv::imshow("Trajectory", trajectory);

		const int key = cv::waitKey(kKeyDelayMs);
		if (key == 27 || key == 'q') {
			break;
		}

		previousGray = currentGray;
	}

	// Stop MAVLink listener if running
	if (mavPort > 0) {
		imuRunning = false;
		if (imuThread.joinable()) imuThread.join();
	}
	#ifdef USE_ROS2
		rclcpp::shutdown();
	#endif

	return 0;
}
