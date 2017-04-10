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
   * @param[in]  match_type    The match type (can be RATIO or CROSSCHECK)
   * @param[in]  match_thresh  The match thresh (typically between 0.7-0.9)
   * @param[in]  epi_thresh    The epipolar threshold (for rectified images
   * this should be less than 1.5)
   *
   * @return     Output struct with the matched keypoints, descriptors and 3D
   * points relative to camera frame
   */
  StereoFeatures matchStereo(const cv::Mat& left,
                             const cv::Mat& right,
                             const KeyPointType& kp_type = ORB_KP,
                             const DescriptorType& desc_type = ORB_DESC,
                             const MatchType& match_type = RATIO,
                             const float& match_thresh = 0.8,
                             const float& epi_thresh = 1.2);

  /**
   * @brief      Performs a matching between 2 frames without applying the
   * epipolar constrain
   *
   * @param[in]  img_a         The image a
   * @param[in]  img_b         The image b
   * @param[in]  match_type    The match type (can be RATIO or CROSSCHECK)
   * @param[in]  match_thresh  The match thresh (typically between 0.7-0.9)
   *
   * @return     Output struct with the matched keypoints, descriptors and 3D
   * points relative to camera frame
   */
  MonoMatches matchNoStereo(const cv::Mat& img_a,
                            const cv::Mat& img_b,
                            const KeyPointType& kp_type = ORB_KP,
                            const DescriptorType& desc_type = ORB_DESC,
                            const MatchType& match_type = RATIO,
                            const float& match_thresh = 0.8);

  /**
   * @brief      Performs a matching between 2 frames (mono) without applying the
   * epipolar constrain
   *
   * @param[in]  feat_img_a    The extracted features of image a
   * @param[in]  img_b         The extracted features image b
   * @param[in]  match_type    The match type (can be RATIO or CROSSCHECK)
   * @param[in]  match_thresh  The match thresh (typically between 0.7-0.9)
   *
   * @return     Output struct with the matched keypoints, descriptors and 3D
   * points relative to camera frame
   */
  MonoMatches matchNoStereo(const MonoFeatures& feat_img_a,
                            const MonoFeatures& feat_img_b,
                            const KeyPointType& kp_type = ORB_KP,
                            const DescriptorType& desc_type = ORB_DESC,
                            const MatchType& match_type = RATIO,
                            const float& match_thresh = 0.8);

  /**
   * @brief      Performs a matching between 2 frames (stereo) without applying the
   * epipolar constrain
   *
   * @param[in]  feat_img_a    The extracted features of stereo frame a
   * @param[in]  img_b         The extracted features stereo frame b
   * @param[in]  match_type    The match type (can be RATIO or CROSSCHECK)
   * @param[in]  match_thresh  The match thresh (typically between 0.7-0.9)
   *
   * @return     Output struct with the matched keypoints, descriptors and 3D
   * points relative to camera frame
   */
  StereoMatches matchNoStereo(const StereoFeatures& feat_img_a,
                              const StereoFeatures& feat_img_b,
                              const KeyPointType& kp_type = ORB_KP,
                              const DescriptorType& desc_type = ORB_DESC,
                              const MatchType& match_type = RATIO,
                              const float& match_thresh = 0.8);

  /**
   * @brief      Estimates the pose of image B with respect to image A
   *
   * @param[in]  img_a         The image a
   * @param[in]  img_b         The image b
   * @param[in]  match_type    The match type (can be RATIO or CROSSCHECK)
   * @param[in]  match_thresh  The match thresh (typically between 0.7-0.9)
   * @param[in]  reproj_err    The reprojection error (typically between 4.0 and 12.0)
   * @param      pose          The output pose of camera
   *
   * @return     number of inliers
   */
  int estimatePose(const cv::Mat& img_a,
                   const cv::Mat img_b,
                   tf::Transform* pose,
                   const MatchType& match_type = RATIO,
                   const float& match_thresh = 0.8,
                   const float& reproj_err = 6.0);

 private:
  /**
   * @brief      Extract features (kp and desc) for a single image
   *
   * @param[in]  img        The image
   * @param[in]  kp_type    The kp type
   * @param[in]  desc_type  The description type
   *
   * @return     Output struct with keypoints and descriptors
   */
  MonoFeatures extractFeatures(const cv::Mat& img,
                               const KeyPointType& kp_type = ORB_KP,
                               const DescriptorType& desc_type = ORB_DESC);

  /**
   * @brief      Ratio feature matching
   *
   * @param[in]  feat_a    The vector of features a
   * @param[in]  feat_b    The vector of features b
   * @param[in]  ratio_th  The ratio threshold
   *
   * @tparam     FeatureT  FeatureType
   *
   * @return     Output vector of matches
   */
  std::vector<cv::DMatch> ratioMatching(const MonoFeatures& feat_a,
                                        const MonoFeatures& feat_b,
                                        const float& ratio_th = 0.7);

  /**
   * @brief      Chrosscheck feature matching
   *
   * @param[in]  feat_a    The vector of features a
   * @param[in]  feat_b    The vector of features b
   * @param[in]  ratio_th  The crosscheck threshold
   *
   * @tparam     FeatureT  FeatureType
   *
   * @return     Output vector of matches
   */
  std::vector<cv::DMatch> crossCheckMatching(const MonoFeatures& feat_a,
                                             const MonoFeatures& feat_b,
                                             const float& ratio_th = 0.7);

  Config config_;  //!< Stores config
  bool is_camera_model_set_;  //!< Determines if camera model has been set or not
  image_geometry::StereoCameraModel camera_model_;  //!< Stereo camera model
};

}  // namespace femas

#endif  // INCLUDE_LIBFEMAS_LIBFEMAS_H_