// Copyright (c) 2017, Pep Lluís Negre
// All rights reserved.

#ifndef INCLUDE_LIBFEMAS_LIBFEMAS_H_
#define INCLUDE_LIBFEMAS_LIBFEMAS_H_

#include <libfemas/types.h>

#include <ros/ros.h>
#include <tf/transform_datatypes.h>
#include <image_geometry/stereo_camera_model.h>

#include <string>

#include <opencv2/xfeatures2d.hpp>

namespace femas {

class Femas {
 public:
  /**
   * @brief      Empty class constructor
   */
  Femas();

  /**
   * @brief      Sets the configuration.
   *
   * @param[in]  config  The configuration
   */
  inline void setConfig(const Config& config) {config_ = config;}

  /**
   * @brief      Gets the configuration.
   *
   * @return     The configuration.
   */
  inline Config getConfig() const { return config_; }

  /**
   * @brief      Sets the camera model.
   *
   * @param[in]  camera_model  The camera model
   */
  inline void setCameraModel(
    const image_geometry::StereoCameraModel& camera_model) {
    is_camera_model_set_ = true;
    camera_model_ = camera_model;
  }

  /**
   * @brief      Gets the camera model.
   *
   * @return     The camera model.
   */
  inline image_geometry::StereoCameraModel getCameraModel() const {
    return camera_model_;
  }

  /**
   * @brief      Perform a stereo matching
   *
   * @param[in]  left          The left image
   * @param[in]  right         The right image
   *
   * @return     Output struct with the matched keypoints, descriptors and 3D
   * points relative to camera frame
   */
  StereoFeatures matchStereo(const cv::Mat& left,
                             const cv::Mat& right);

  /**
   * @brief      Performs a matching between 2 frames without applying the
   * epipolar constrain
   *
   * @param[in]  img_a         The image a
   * @param[in]  img_b         The image b
   *
   * @return     Output struct with the matched keypoints, descriptors and 3D
   * points relative to camera frame
   */
  MonoMatches matchPair(const cv::Mat& img_a,
                        const cv::Mat& img_b);

  /**
   * @brief      Performs a matching between 2 frames (mono) without applying the
   * epipolar constrain
   *
   * @param[in]  feat_img_a    The extracted features of image a
   * @param[in]  img_b         The extracted features image b
   *
   * @return     Output struct with the matched keypoints, descriptors and 3D
   * points relative to camera frame
   */
  MonoMatches matchPair(const MonoFeatures& feat_a,
                        const MonoFeatures& feat_b);

  /**
   * @brief      Performs a matching between 2 frames (stereo) without applying the
   * epipolar constrain
   *
   * @param[in]  feat_img_a    The extracted features of stereo frame a
   * @param[in]  img_b         The extracted features stereo frame b
   *
   * @return     Output struct with the matched keypoints, descriptors and 3D
   * points relative to camera frame
   */
  StereoMatches matchPair(const StereoFeatures& feat_a,
                          const StereoFeatures& feat_b);

  /**
   * @brief      Estimates the position of camera b with respect to camera a
   *
   * @param[in]  points_a    The 3D points of camera a
   * @param[in]  points_b    The 2D points of camera b
   * @param      pose        The output pose
   * @param      inliers     The inliers vector
   */
  void estimatePose(const std::vector<cv::Point3d>& points_a,
                    const std::vector<cv::Point2d>& points_b,
                    tf::Transform* pose,
                    std::vector<int>* inliers);

 private:
  /**
   * @brief      Extract features (kp and desc) for a single image
   *
   * @param[in]  img        The image
   *
   * @return     Output struct with keypoints and descriptors
   */
  MonoFeatures extractFeatures(const cv::Mat& img);

  /**
   * @brief      Generic feature matching
   *
   * @param[in]  a             The vector of features a
   * @param[in]  b             The vector of features b
   *
   * @return     Output vector of matches
   */
  std::vector<cv::DMatch> match(const MonoFeatures& a,
                                const MonoFeatures& b);

  /**
   * @brief      Ratio feature matching
   *
   * @param[in]  feat_a    The vector of features a
   * @param[in]  feat_b    The vector of features b
   *
   * @tparam     FeatureT  FeatureType
   *
   * @return     Output vector of matches
   */
  std::vector<cv::DMatch> ratioMatching(const MonoFeatures& feat_a,
                                        const MonoFeatures& feat_b,
                                        const double& ratio_th);

  /**
   * @brief      Chrosscheck feature matching
   *
   * @param[in]  feat_a    The vector of features a
   * @param[in]  feat_b    The vector of features b
   *
   * @tparam     FeatureT  FeatureType
   *
   * @return     Output vector of matches
   */
  std::vector<cv::DMatch> crossCheckMatching(const MonoFeatures& feat_a,
                                             const MonoFeatures& feat_b,
                                             const double& ratio_th);

  Config config_;  //!< Stores config
  bool is_camera_model_set_;  //!< Determines if camera model has been set or not
  image_geometry::StereoCameraModel camera_model_;  //!< Stereo camera model
};

}  // namespace femas

#endif  // INCLUDE_LIBFEMAS_LIBFEMAS_H_
