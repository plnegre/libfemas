// Copyright (c) 2017, Pep Lluís Negre
// All rights reserved.

#include "libfemas/libfemas.h"

namespace femas {

void Femas::estimatePose(const std::vector<cv::Point3d>& points_a,
                         const std::vector<cv::Point2d>& points_b,
                         tf::Transform* pose,
                         std::vector<int>* inliers,
                         const float& reproj_err) {
  // Sanity checks
  if (!is_camera_model_set_) {
    ROS_ERROR("[Femas::estimatePose]: set camera model before match stereo features.");
  }

  // Init
  pose->setIdentity();

  // Get camera matrix
  cv::Matx34d camera_matrix = camera_model_.left().fullProjectionMatrix();

  cv::Mat rvec, tvec;
  cv::solvePnPRansac(points_a, points_b, camera_matrix, cv::Mat(), rvec, tvec,
            false, 100, reproj_err, 0.99, *inliers, cv::SOLVEPNP_ITERATIVE);

  // Sanity check
  if (rvec.empty() || tvec.empty()) {
    return;
  }

  // Convert output transform
  tf::Vector3 axis(rvec.at<double>(0, 0),
                   rvec.at<double>(1, 0),
                   rvec.at<double>(2, 0));
  double angle = norm(rvec);
  tf::Quaternion quaternion(axis, angle);

  tf::Vector3 translation(tvec.at<double>(0, 0), tvec.at<double>(1, 0),
      tvec.at<double>(2, 0));

  pose->setRotation(quaternion);
  pose->setOrigin(translation);
}

}  // namespace femas

