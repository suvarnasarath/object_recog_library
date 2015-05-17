#ifndef DETECTION_H
#define DETECTION_H

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <vector>

typedef enum
{
	OFF,
	ERROR,
	DEBUG,
	VERBOSE
}LOGLEVEL;

/*
 * Definition of output sample
 * @ id : Id of the sample
 * @ x,y: position in world coordinates.
 */
typedef struct
{
	unsigned int id;	// Sample Id
	double x;			// Sample world x
	double y;			// Sample world y
	double projected_width; // Estimated size of the sample
}DETECTED_SAMPLE;

/*
 * Definition of camera platform specs
 * @ height: Height of the camera from ground plane
 * @ pitch : Pitch of the camera
 * @ HFov  : Horizontal field of view of the camera lense
 * @ VFov  : Vertical field of view of the camera lense
 */
typedef struct
{
	double height; 			// height of the camera from ground plane
	double pitch;  			// Pitch angle of the camera (up from down)
	double HFov;   			// Horizontal field of view
	double VFov;   			// Vertical field of view
	unsigned int Hpixels;	// Imager height
	unsigned int Vpixels;   // Imager width
	double max_detection_dist; // Maximum distance to detect samples
	double x_offset;		// Camera x offset w.r.t robot frame
	double y_offset;		// Camera y offset w.r.t robot frame
	double yaw;				// Camera yaw offset w.r.t robot frame
	unsigned int min_bb_area_in_pixels;
}platform_camera_parameters;

void find_objects(unsigned int camera_index,const cv::Mat *imgPtr, cv::Mat *out_image,std::vector<DETECTED_SAMPLE> &detected_samples);
void register_sample(unsigned int Id, const std::vector<double>&hue_param,
									  const std::vector<double>&sat_param,
									  const std::vector<double>&val_param,
									  const std::vector<double>width,
									  const std::vector<double>height);
void register_camera(unsigned int camera_id, const platform_camera_parameters * param);
int  get_registered_sample_size();
void set_sample_filter(const std::vector<unsigned int> &filter);
void Set_debug(LOGLEVEL level );

#endif
