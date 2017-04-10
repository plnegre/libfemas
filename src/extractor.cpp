// Copyright (c) 2017, Pep Lluís Negre
// All rights reserved.

#include "libfemas/libfemas.h"

namespace femas {

MonoFeatures Femas::extractFeatures(const cv::Mat& img,
                                    const KeyPointType& kp_type,
                                    const DescriptorType& desc_type) {
  // Proceed depending on the keypoint type
  cv::Ptr<cv::Feature2D> detector(new cv::Feature2D());
  switch (kp_type) {
    case SIFT_KP:
    {
      detector = cv::xfeatures2d::SIFT::create();
      break;
    }
    case SURF_KP:
    {
      detector = cv::xfeatures2d::SURF::create();
      break;
    }
    case ORB_KP:
    {
      detector = cv::ORB::create();
      break;
    }
    default:
    {
      ROS_ERROR("[Femas::extractFeatures]: Keypoint type not permitted.");
    }
  }

  // Extract keypoints
  std::vector<cv::KeyPoint> kp;
  detector->detect(img, kp);

  // Proceed depending on the descriptor type
  cv::Ptr<cv::Feature2D> extractor(new cv::Feature2D());
  switch (desc_type) {
    case SIFT_DESC:
    {
      extractor = cv::xfeatures2d::SIFT::create();
      break;
    }
    case SURF_DESC:
    {
      extractor = cv::xfeatures2d::SURF::create();
      break;
    }
    case ORB_DESC:
    {
      extractor = cv::ORB::create();
      break;
    }
    default:
    {
      ROS_ERROR("[Femas::extractFeatures]: Descriptor type not permitted.");
    }
  }

  // Compute descriptors
  cv::Mat desc;
  extractor->compute(img, kp, desc);

  MonoFeatures mf;
  mf.kp = kp;
  mf.desc = desc;
  return mf;
}

}  // namespace femas

