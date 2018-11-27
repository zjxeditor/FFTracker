/*M///////////////////////////////////////////////////////////////////////////////////////
 //
 //  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
 //
 //  By downloading, copying, installing or using the software you agree to this license.
 //  If you do not agree to this license, do not download, install,
 //  copy or use the software.
 //
 //
 //                           License Agreement
 //                For Open Source Computer Vision Library
 //
 // Copyright (C) 2013, OpenCV Foundation, all rights reserved.
 // Third party copyrights are property of their respective owners.
 //
 // Redistribution and use in source and binary forms, with or without modification,
 // are permitted provided that the following conditions are met:
 //
 //   * Redistribution's of source code must retain the above copyright notice,
 //     this list of conditions and the following disclaimer.
 //
 //   * Redistribution's in binary form must reproduce the above copyright notice,
 //     this list of conditions and the following disclaimer in the documentation
 //     and/or other materials provided with the distribution.
 //
 //   * The name of the copyright holders may not be used to endorse or promote products
 //     derived from this software without specific prior written permission.
 //
 // This software is provided by the copyright holders and contributors "as is" and
 // any express or implied warranties, including, but not limited to, the implied
 // warranties of merchantability and fitness for a particular purpose are disclaimed.
 // In no event shall the Intel Corporation or contributors be liable for any direct,
 // indirect, incidental, special, exemplary, or consequential damages
 // (including, but not limited to, procurement of substitute goods or services;
 // loss of use, data, or profits; or business interruption) however caused
 // and on any theory of liability, whether in contract, strict liability,
 // or tort (including negligence or otherwise) arising in any way out of
 // the use of this software, even if advised of the possibility of such damage.
 //
 //M*/

#ifndef __CVCSRT_OPENCV_TRACKER_HPP__
#define __CVCSRT_OPENCV_TRACKER_HPP__

#include "opencv2/core.hpp"
#include "opencv2/imgproc/types_c.h"

namespace cvcsrt
{

using namespace cv;

class CV_EXPORTS_W TrackerCSRT
{
public:
  struct CV_EXPORTS Params
  {
    /**
    * \brief Constructor
    */
    Params();

    /**
    * \brief Read parameters from file
    */
    void read(const FileNode& /*fn*/);

    /**
    * \brief Write parameters from file
    */
    void write(cv::FileStorage& fs) const;

    bool use_hog;
    bool use_color_names;
    bool use_gray;
    bool use_rgb;
    bool use_channel_weights;
    bool use_segmentation;

    std::string window_function; //!<  Window function: "hann", "cheb", "kaiser"
    float kaiser_alpha;
    float cheb_attenuation;

    float template_size;
    float gsl_sigma;
    float hog_orientations;
    float hog_clip;
    float padding;
    float filter_lr;
    float weights_lr;
    int num_hog_channels_used;
    int admm_iterations;
    int histogram_bins;
    float histogram_lr;
    int background_ratio;
    int number_of_scales;
    float scale_sigma_factor;
    float scale_model_max_area;
    float scale_lr;
    float scale_step;

    float psr_threshold; //!< we lost the target, if the psr is lower than this.
  };

  /** @brief Constructor
  @param parameters CSRT parameters TrackerCSRT::Params
  */
  static Ptr<TrackerCSRT> create(const TrackerCSRT::Params &parameters);

  CV_WRAP static Ptr<TrackerCSRT> create();

  CV_WRAP virtual void setInitialMask(const Mat mask) = 0;

  CV_WRAP virtual bool init( const Mat image, const Rect2d& boundingBox) = 0;

  CV_WRAP virtual bool update( const Mat image, CV_OUT Rect2d& boundingBox) = 0;
};

} /* namespace cv */

#endif