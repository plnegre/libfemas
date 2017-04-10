// Copyright (c) 2017, Pep Lluís Negre
// All rights reserved.

#include "libfemas/libfemas.h"

namespace femas {

StereoFeatures Femas::matchStereo(const cv::Mat& left,
                                  const cv::Mat& right,
                                  const KeyPointType& kp_type,
                                  const DescriptorType& desc_type,
                                  const MatchType& match_type,
                                  const float& match_thresh,
                                  const float& epi_thresh) {
  // Sanity checks
  if (!is_camera_model_set_) {
    ROS_ERROR("[Femas::matchStereo]: set camera model before match stereo features.");
  }

  // Extract features
  MonoFeatures l_feat = extractFeatures(left,  kp_type, desc_type);
  MonoFeatures r_feat = extractFeatures(right, kp_type, desc_type);

  // Match
  std::vector<cv::DMatch> matches;
  switch (match_type) {
    case RATIO:
    {
      matches = ratioMatching(l_feat, r_feat, match_thresh);
      break;
    }
    case CROSSCHECK:
    {
      matches = crossCheckMatching(l_feat, r_feat, match_thresh);
      break;
    }
    default:
    {
      ROS_ERROR("[Femas::matchStereo]: match type not permitted.");
    }
  }

  // Epipolar filter and 3D
  MonoFeatures l_feat_matched;
  MonoFeatures r_feat_matched;
  std::vector<cv::Point3f> points;

  for (std::size_t i=0; i < matches.size(); ++i) {
    cv::KeyPoint l_kp = l_feat.kp[matches[i].queryIdx];
    cv::KeyPoint r_kp = r_feat.kp[matches[i].trainIdx];

    // Epipolar constrain
    if (abs(l_kp.pt.y - r_kp.pt.y) <= epi_thresh) {

      // Compute 3D point
      cv::Point3d p;
      double disparity = l_kp.pt.x - r_kp.pt.x;
      camera_model_.projectDisparityTo3d(l_kp.pt, disparity, p);

      if (std::isfinite(p.x) && std::isfinite(p.y) &&
          std::isfinite(p.z) && p.z > 0) {
        l_feat_matched.kp.push_back(l_kp);
        r_feat_matched.kp.push_back(r_kp);
        l_feat_matched.desc.push_back(l_feat.desc.row(matches[i].queryIdx));
        r_feat_matched.desc.push_back(r_feat.desc.row(matches[i].trainIdx));
        points.push_back(p);
      }
    }
  }

  StereoFeatures sf;
  sf.left   = l_feat_matched;
  sf.right  = r_feat_matched;
  sf.points = points;
  return sf;
}

std::vector<cv::DMatch> Femas::ratioMatching(const MonoFeatures& feat_a,
                                             const MonoFeatures& feat_b,
                                             const float& ratio_th) {
  cv::Mat a_desc = feat_a.desc;
  cv::Mat b_desc = feat_a.desc;

  if (a_desc.type() != b_desc.type()) {
    ROS_ERROR_STREAM("[Femas::RatioMatching]: left and right descriptors" <<
    " are not of the same type.");
  }
  if (a_desc.cols   == b_desc.cols) {
    ROS_ERROR_STREAM("[Femas::RatioMatching]: left and right descriptors" <<
    " have different dimensions.");
  }

  cv::Ptr<cv::DescriptorMatcher> descriptor_matcher;
  if (a_desc.type() == CV_8U)
    descriptor_matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
  else
    descriptor_matcher = cv::DescriptorMatcher::create("BruteForce");

  const int knn = 2;
  const cv::Mat match_mask;
  std::vector<cv::DMatch> matches;
  std::vector<std::vector<cv::DMatch> > knn_matches;
  descriptor_matcher->knnMatch(a_desc, b_desc, knn_matches, knn, match_mask);
  for (std::size_t m=0; m < knn_matches.size(); m++) {
    if (knn_matches[m].size() < 2) continue;
    if (knn_matches[m][0].distance <= knn_matches[m][1].distance * ratio_th)
      matches.push_back(knn_matches[m][0]);
  }
  return matches;
}

std::vector<cv::DMatch> Femas::crossCheckMatching(const MonoFeatures& feat_a,
                                                  const MonoFeatures& feat_b,
                                                  const float& ratio_th) {
  std::vector<cv::DMatch> query_to_train_matches = ratioMatching(feat_a,
                                                                 feat_b,
                                                                 ratio_th);
  std::vector<cv::DMatch> train_to_query_matches = ratioMatching(feat_b,
                                                                 feat_a,
                                                                 ratio_th);

  std::vector<cv::DMatch> matches;
  for (std::size_t i=0; i < query_to_train_matches.size(); ++i) {
    bool match_found = false;
    const cv::DMatch& forward_match = query_to_train_matches[i];
    for (std::size_t j=0;
      j < train_to_query_matches.size() && match_found == false; ++j) {
      const cv::DMatch& backward_match = train_to_query_matches[j];
      if (forward_match.trainIdx == backward_match.queryIdx &&
          forward_match.queryIdx == backward_match.trainIdx) {
        matches.push_back(forward_match);
        match_found = true;
      }
    }
  }
}


}  // namespace femas
