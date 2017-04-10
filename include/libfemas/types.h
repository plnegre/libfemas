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
  Config() : desc_type("ORB"), desc_matching_type("RATIO"),
  desc_thresh_ratio(0.8), epipolar_thresh(1.2) {}

  std::string desc_type;           //!< Type of the descriptors
  std::string desc_matching_type;  //!< Can be "CROSSCHECK" or "RATIO"
  double desc_thresh_ratio;        //!< Descriptor threshold for crosscheck
                                   //   matching (typically between 0.7-0.9) or
                                   //   ratio for ratio matching (typically
                                   //   between 0.7-0.9)
  int epipolar_thresh;             //!< Epipolar threshold. For rectified stereo
                                   //   pairs should be < 1.5
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
  std::vector<cv::Point3f> points;  //!< 3D camera points
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
