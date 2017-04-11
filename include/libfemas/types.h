// Copyright (c) 2017, Pep Lluís Negre
// All rights reserved.

#ifndef INCLUDE_LIBFEMAS_TYPES_H_
#define INCLUDE_LIBFEMAS_TYPES_H_

#include <opencv2/opencv.hpp>

#include <string>
#include <vector>

namespace femas {
enum KeyPointType {
  SIFT_KP, SURF_KP, ORB_KP
};

enum DescriptorType {
  SIFT_DESC, SURF_DESC, ORB_DESC
};

enum MatchType {
  RATIO, CROSSCHECK
};

/**
 * @brief      Base class configuration parameters
 */
struct Config {
  Config() : kp_type(ORB_KP), desc_type(ORB_DESC), desc_matching_type(RATIO),
  desc_thresh_ratio(0.8), epipolar_thresh(1.2), reproj_thresh(6.0) {}

  KeyPointType kp_type;           //!< Type of the keypoints
  DescriptorType desc_type;       //!< Type of the descriptors
  MatchType desc_matching_type;   //!< CROSSCHECK or RATIO matching
  double desc_thresh_ratio;       //!< Descriptor threshold for crosscheck
                                  //   matching (typically between 0.7-0.9) or
                                  //   ratio for ratio matching (typically
                                  //   between 0.7-0.9)
  double epipolar_thresh;         //!< Epipolar threshold. For rectified stereo
                                  //   pairs should be < 1.5
  double reproj_thresh;           //!< The reprojection threshold for the pose
                                  //   estimation
};

/**
 * @brief      Stores keypoints and descriptors for a single images
 */
struct MonoFeatures {
  std::vector<cv::KeyPoint> kp;  //!< Image keypoints
  cv::Mat desc;                  //!< image descriptors
};

/**
 * @brief      Stores keypoints, descriptors and 3D points (relative to camera
 * frame) for a stereo pair.
 */
struct StereoFeatures {
  MonoFeatures left;                //!< Left image features
  MonoFeatures right;               //!< Right image features
  std::vector<cv::Point3d> points;  //!< 3D camera points
};

/**
 * @brief      Stores the matched features between 2 images (mono)
 */
struct MonoMatches {
  MonoFeatures a;               //!< A image mono features
  MonoFeatures b;               //!< B image mono features
};

/**
 * @brief      Stores the matched features between 2 images (stereo)
 */
struct StereoMatches {
  StereoFeatures a;             //!< A image stereo features
  StereoFeatures b;             //!< B image stereo features
};

}  // namespace femas

#endif  // INCLUDE_LIBFEMAS_TYPES_H_
