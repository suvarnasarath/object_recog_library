#include "detection.h"

#define MAX_SAMPLES (256)

using namespace cv;

// Globals
RNG rng(12345);
int kernel_size = 3;

/*
 * Definition of input sample
 * @ id : Id of the sample
 * @ HSV_MIN: Min values for HCV color space
 * @ HSV_MAX: Max values for HCV color space
 * @ min and max width of the sample to search
 * @ min and max height of the sample to search
 */
typedef struct
{
	unsigned int Id;
	std::vector<int> HSV_MIN;
	std::vector<int> HSV_MAX;
	double min_width;
	double max_width;
	double min_height;
	double max_height;
}REGISTERED_SAMPLE;

/*
 * Definition of camera platform specs
 * @ height: Height of the camera from ground plane
 * @ pitch : Pitch of the camera
 * @ HFov  : Horizontal field of view of the camera lense
 * @ VFov  : Vertical field of view of the camera lense
 */
typedef struct
{
	unsigned int camera_Id; // camera number
	double height; // height of the camera from ground plane
	double pitch;  // Pitch angle of the camera (up from down)
	double HFov;   // Horizontal field of view
	double VFov;   // Vertical field of view
}platform_camera_parameters;

std::vector<REGISTERED_SAMPLE> samples;
std::vector<DETECTED_SAMPLE> detected_samples;
std::vector<platform_camera_parameters>camera_parameters;

cv::Mat Input_image;

void register_sample(unsigned int Id, const std::vector<int> &hsv_min, const std::vector<int>&hsv_max, double min_width, double max_width, double min_height, double max_height)
{
	REGISTERED_SAMPLE new_sample;
	new_sample.Id = Id;
	new_sample.HSV_MIN = hsv_min;
	new_sample.HSV_MAX = hsv_max;
	new_sample.min_width = min_width;
	new_sample.max_width = max_width;
	new_sample.min_height = min_height;
	new_sample.max_height = max_height;

	samples.push_back(new_sample);
	std::cout<<"added new sample Id = " << Id << std::endl;
}

void register_camera(unsigned int camera_id, double camera_height, double camera_pitch,
						double camera_HFov, double camera_VFov)
{
	platform_camera_parameters camera_spec;

	camera_spec.camera_Id = camera_id;
	camera_spec.height = camera_height;
	camera_spec.pitch = camera_pitch;
	camera_spec.HFov = camera_HFov;
	camera_spec.VFov = camera_VFov;

	camera_parameters.push_back(camera_spec);
}

int get_registered_sample_size()
{
	return samples.size();
}

void set_sample_filter(const std::vector<unsigned int> &filter)
{
	// Todo:

}

bool process_image(cv::Mat image_hsv, int index)
{    
	DETECTED_SAMPLE sample;
	// sample index is same for all samples this call
	sample.id = index;

    cv::Mat temp_image1, temp_image2 ;
    std::vector<vector<Point> > contours;
    std::vector<Vec4i> hierarchy;

    // Mark all pixels in the required color range high and other pixels low.
    inRange(image_hsv,samples[index].HSV_MIN,samples[index].HSV_MAX,temp_image1);

    // Gives the kernel shape for erosion.
    // To do: Experiment with different kernels
    Mat element = getStructuringElement( MORPH_RECT, Size(2*kernel_size+1,2*kernel_size+1), Point(0,0));

    // Erode the image to get rid of trace elements with similar color to the required sample
    erode(temp_image1,temp_image2,element);

    // Find contours in the thresholded image to determine shapes
    findContours(temp_image2,contours,hierarchy,CV_RETR_TREE,CV_CHAIN_APPROX_SIMPLE,Point(0,0));

    std::vector<vector<Point> > contours_poly( contours.size() );
    std::vector<Rect> boundRect( contours.size() );
    for( int i = 0; i < contours.size(); i++ )
     {
        approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
        boundRect[i] = boundingRect( Mat(contours_poly[i]) );
     }
   
    // Print the number of samples found
    std::cout << "Number of samples found: "<< contours.size()<< std::endl;

    // Draw all the contours found in the previous step
    Mat drawing = Mat::zeros( temp_image2.size(), CV_8UC3 );
    for( int i = 0; i< contours.size(); i++ )
     {
       Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
       drawContours( drawing, contours_poly, i, color, 2, 8, hierarchy, 0, Point() );

       // Draw a bounding box
       rectangle( Input_image, boundRect[i].tl(), boundRect[i].br(), (0,0,255), 2, 8, 0 );

	   //detected_samples.push_back(sample);
     }
    return true;
}
void find_objects(const cv::Mat *imgPtr, cv::Mat *out_image,std::vector<DETECTED_SAMPLE> &detected_samples)
{
	cv::Mat hsv_image;
	DETECTED_SAMPLE test_sample;
	// Test sample values for integration
	test_sample.id = 0;
	test_sample.x = 0;
	test_sample.y = 0;
	test_sample.projected_width = 0;

	Input_image = *imgPtr;

	if(! Input_image.data) {
		std::cout << "could not read image"<< std::endl;
		return;
	}

	// Convert the color space to HSV
	cv::cvtColor(Input_image,hsv_image,CV_BGR2HSV);

	// Clear detected_sample struct before filling in with new image data
	detected_samples.clear();

	// Get the iterator for the vector color space and loop through all sample color's
	for(int index = 0; index < samples.size(); ++index)
	{
		if(!process_image(hsv_image, index))
		{
			std::cout << "Processing images failed" << std::endl;
		}
	}
	//return test_sample;
}

