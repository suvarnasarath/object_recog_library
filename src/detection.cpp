#include "detection.h"

#define MAX_SAMPLES (256)

using namespace cv;

// Globals
RNG rng(12345);  // Don't panic, used only for color display
int kernel_size = 3;

bool bPrintDebugMsg = false;

cv::Mat Input_image, Rotation_matrix;

typedef struct
{
	unsigned int x;
	unsigned int y;
}PIXEL;

void get_world_pos(unsigned int cameraId, PIXEL &pos);
void calculate_pixel_attitude_and_azimuth(PIXEL pixel_pos, double &psi, double &theta);
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
	bool isValid;  // Should we check for this sample in out detector
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
	unsigned int Hpixels;
	unsigned int Vpixels;
}platform_camera_parameters;

std::vector<REGISTERED_SAMPLE> registered_sample;
std::vector<DETECTED_SAMPLE> detected_samples;
std::vector<platform_camera_parameters>camera_parameters;

void Set_debug(bool enable)
{
	bPrintDebugMsg = enable;
	if(bPrintDebugMsg)
	{
		std::cout << "Debug messages enabled:" << std::endl;
	}
}

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
	new_sample.isValid = true; // true by default for all samples

	registered_sample.push_back(new_sample);
	std::cout<<"added new sample Id = " << Id << std::endl;
}

void register_camera(unsigned int camera_id, double camera_height, double camera_pitch,
						double camera_HFov, double camera_VFov, unsigned int Hpixels, unsigned int Vpixels)
{
	platform_camera_parameters camera_spec;

	// Supports only one camera at this time.
	if(camera_parameters.size() < 1)
	{
		camera_spec.camera_Id = camera_id;
		camera_spec.height = camera_height;
		camera_spec.pitch = camera_pitch;
		camera_spec.HFov = camera_HFov;
		camera_spec.VFov = camera_VFov;
		camera_spec.Hpixels = Hpixels;
		camera_spec.Vpixels = Vpixels;

		camera_parameters.push_back(camera_spec);
	} else {
		std::cout <<"WARNING: Library supports only one camera for now" << std::endl;
	}
}

int get_registered_sample_size()
{
	return registered_sample.size();
}

void set_sample_filter(const std::vector<unsigned int> &filter)
{
	// Todo: implement filter valid samples
}

bool process_image(cv::Mat image_hsv,cv::Mat *out_image, int index,std::vector<DETECTED_SAMPLE> &detected_samples)
{    
	DETECTED_SAMPLE sample;
	// sample index is same for all samples this call
	sample.id = index;

    cv::Mat temp_image1, temp_image2 ;
    std::vector<vector<Point> > contours;
    std::vector<Vec4i> hierarchy;

    // Mark all pixels in the required color range high and other pixels low.
    inRange(image_hsv,registered_sample[index].HSV_MIN,registered_sample[index].HSV_MAX,temp_image1);

    // Gives the kernel shape for erosion.
    // To do: Experiment with different kernels
    Mat element = getStructuringElement( MORPH_RECT, Size(2*kernel_size+1,2*kernel_size+1), Point(0,0));

    // Erode the image to get rid of trace elements with similar color to the required sample
    erode(temp_image1,temp_image2,element);

    // Find contours in the thresholded image to determine shapes
    findContours(temp_image2,contours,hierarchy,CV_RETR_TREE,CV_CHAIN_APPROX_SIMPLE,Point(0,0));

    std::vector<vector<Point> > contours_poly( contours.size() );
    std::vector<Rect> boundRect( contours.size() );
    for( int i = 0; i < contours.size(); ++i)
     {
        approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
        boundRect[i] = boundingRect( Mat(contours_poly[i]) );
        // Get world pos
        //get_world_pos();
        //detected_samples.push_back(sample);
     }
   
    // Print the number of samples found
    if(bPrintDebugMsg)
    	std::cout << "Number of samples found: "<< contours.size()<< std::endl;

    if(out_image != NULL)
    {
		// Draw all the contours found in the previous step
		Mat drawing = Mat::zeros( temp_image2.size(), CV_8UC3 );
		for( int i = 0; i< contours.size(); i++ )
		 {
		   Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
		   drawContours( drawing, contours_poly, i, color, 2, 8, hierarchy, 0, Point() );

		   // Draw a bounding box
		   rectangle(*out_image, boundRect[i].tl(), boundRect[i].br(), (0,0,255), 2, 8, 0 );
		 }
    }
    return true;
}

/*
 * Gives the world position
 * @ pos	  : Location of the desired pixel
 * @ camera ID: Assuming that there is only 1 camera for now
 */
void get_world_pos(unsigned int cameraId, PIXEL &pos)
{
	double psi;
	double theta;

	for(int index =0; index < camera_parameters.size();++index)
	{
		if(camera_parameters[index].camera_Id != cameraId)
		{
			if(bPrintDebugMsg)
				std::cout << "cameraId does not match" << std::endl;
			continue;
		} else {
			break;
		}
	}

	if(pos.x < camera_parameters[cameraId].Hpixels && pos.y > camera_parameters[cameraId].Vpixels)
	{
		calculate_pixel_attitude_and_azimuth(pos, psi, theta);
		// get rotation matrix and compute the vector for ray plane intersection

	}
}

void calculate_pixel_attitude_and_azimuth(PIXEL pixel_pos, double &psi, double &theta)
{
	psi =   (1/camera_parameters[0].Hpixels)*(pixel_pos.x * camera_parameters[0].HFov);
	theta = (1/camera_parameters[0].Vpixels)*(pixel_pos.y * camera_parameters[0].VFov);
}

void find_objects(const cv::Mat *imgPtr, cv::Mat *out_image,std::vector<DETECTED_SAMPLE> &detected_samples)
{
	cv::Mat hsv_image;
	if(! imgPtr->data) {
		std::cout << "ERROR: could not read image"<< std::endl;
		return;
	}
	Input_image = *imgPtr;

	// Convert the color space to HSV
	cv::cvtColor(Input_image,hsv_image,CV_BGR2HSV);

	// Clear detected_sample structure before filling in with new image data
	detected_samples.clear();

	// Get the iterator for the vector color space and loop through all sample color's
	for(int index = 0; index < registered_sample.size(); ++index)
	{
		if(registered_sample[index].isValid)
		{
		  process_image(hsv_image, out_image,index,detected_samples);
		}
	}
}

