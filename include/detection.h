#ifndef DETECTION_H
#define DETECTION_H

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <vector>

/*
 * Definition of output sample
 * @ id : Id of the sample
 * @ x,y: position in world coordinates.
 */
typedef struct
{
	unsigned int id;
	double x;
	double y;
	double projected_width;
}DETECTED_SAMPLE;

void find_objects(const cv::Mat *imgPtr, cv::Mat *out_image,std::vector<DETECTED_SAMPLE> &detected_samples);
void register_sample(unsigned int Id, const std::vector<int>&hsv_min, const std::vector<int>&hsv_max,
					 double min_width, double max_width, double min_height, double max_height);
void register_camera(unsigned int camera_id, double camera_height, double camera_pitch,
						double camera_HFov, double camera_VFov, unsigned int Hpixels, unsigned int Vpixels);
int  get_registered_sample_size();
void set_sample_filter(const std::vector<unsigned int> &filter);
void Set_debug(bool enable);

#endif
