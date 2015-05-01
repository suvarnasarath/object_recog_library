#include <math.h>
#include "detection.h"

using namespace cv;

#define MAX_SAMPLES (256)
#define PI          (3.14159265)
#define INVALID     (32767)  // some large number
#define DEFAULT_CAMERAID (0)
#define DEFAULT_MAX_DIST (5.0)


// Globals
RNG rng(12345);  // Don't panic, used only for color display
int kernel_size = 3;

// Init flag
bool bInit = false;
// Set debug messages OFF by default
bool bPrintDebugMsg = true; // Turn OFF later

cv::Mat Input_image;

/*
 * pixel coordinates
 */
typedef struct
{
	double u;
	double v;
}PIXEL;

/*
 * world coordinates
 */
typedef struct
{
	double x;
	double y;
}WORLD;

// Lookup to store world positions
std::vector<WORLD> WORLD_LOOKUP;

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



std::vector<REGISTERED_SAMPLE> registered_sample;
std::vector<DETECTED_SAMPLE> detected_samples;
std::vector<platform_camera_parameters>camera_parameters;

/*
 * When enabled, prints debugging messages
 */
void Set_debug(bool enable)
{
	bPrintDebugMsg = enable;
	if(bPrintDebugMsg)
	{
		std::cout << "Debug messages enabled:" << std::endl;
	}
}

void calculate_pixel_attitude_and_azimuth(PIXEL pixel_pos, double &elevation, double &azimuth)
{
	azimuth   = (pixel_pos.u * camera_parameters[DEFAULT_CAMERAID].HFov)/static_cast<double>(camera_parameters[DEFAULT_CAMERAID].Hpixels);
	elevation = (pixel_pos.v * camera_parameters[DEFAULT_CAMERAID].VFov)/static_cast<double>(camera_parameters[DEFAULT_CAMERAID].Vpixels);
}

#define Epsilon 0.001
/*
 * This function pre-computes the distances for each pixel
 */
void precompute_world_pos_lookup(unsigned int cameraId)
{
	PIXEL pixel_pos;
	WORLD world_pos, world_pos_copy;
	double elevation, azimuth, R;
	double c_x,c_y,c_z;
	double alpha = camera_parameters[cameraId].pitch;

	double c_theta = cos(camera_parameters[cameraId].yaw);
	double s_theta = sin(camera_parameters[cameraId].yaw);

	bool print = false;

	for(int i =0; i < camera_parameters[cameraId].Hpixels; ++i)
	{
		for(int j =0; j < camera_parameters[cameraId].Vpixels; ++j)
		{
			if(((i == 0) && (j == 0)) ||
			   ((i == camera_parameters[cameraId].Hpixels-1) && (j == 0)) ||
			   ((i == 0) && (j == camera_parameters[cameraId].Vpixels-1)) ||
			   ((i == camera_parameters[cameraId].Hpixels-1) && (j == camera_parameters[cameraId].Vpixels-1)) )
			{
				std::cout << "i: " << i << "j: " << j<< std::endl;
				print = true;
			}
			pixel_pos.u = static_cast<double>(camera_parameters[cameraId].Hpixels/2) - i;
			pixel_pos.v = static_cast<double>(camera_parameters[cameraId].Vpixels/2) - j;

			calculate_pixel_attitude_and_azimuth(pixel_pos, elevation, azimuth);

			double c_x = cos(alpha)*cos(elevation)*cos(azimuth) + sin(alpha)*sin(elevation);
			double c_y = cos(elevation)*sin(azimuth);
			double c_z = -sin(alpha)*cos(elevation)*cos(azimuth) + cos(alpha)*sin(elevation);

			if(c_z < Epsilon)
			{
				R = -(camera_parameters[cameraId].height)/c_z;
				double next_x = R * c_x;
				double next_y = R * c_y;
				if(next_x*next_x + next_y*next_y > camera_parameters[cameraId].max_detection_dist*camera_parameters[cameraId].max_detection_dist)
				{
					world_pos.x = camera_parameters[cameraId].max_detection_dist * c_x;
					world_pos.y = camera_parameters[cameraId].max_detection_dist * c_y;
				} else {
					world_pos.x = next_x;
					world_pos.y = next_y;
				}
			} else {

				world_pos.x = camera_parameters[cameraId].max_detection_dist * c_x;
				world_pos.y = camera_parameters[cameraId].max_detection_dist * c_y;
			}
			world_pos_copy = world_pos;
			// Offset for camera pos wrt robot pos
			world_pos.x = camera_parameters[cameraId].x_offset +
							(c_theta*world_pos_copy.x - s_theta*world_pos_copy.y) ;
			world_pos.y = camera_parameters[cameraId].y_offset +
							(s_theta*world_pos_copy.x + c_theta*world_pos_copy.y) ;


			if(print) {
				std::cout << "pixel_pos.u : "<< pixel_pos.u << std::endl;
				std::cout << "pixel_pos.v : "<< pixel_pos.v << std::endl;
				std::cout << "C_X: "<< c_x << std::endl;
				std::cout << "C_Y: "<< c_y << std::endl;
				std::cout << "C_Z: "<< c_z << std::endl;
				std::cout << "R: "<< R << std::endl;
				std::cout << "X:  "<< world_pos.x << " " << "Y:  "<< world_pos.y << std::endl;
				print = false;
			}

			WORLD_LOOKUP.push_back(world_pos);
		}
	}
}

/*
 * registers a sample to the database
 */
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
	if(bPrintDebugMsg) std::cout<<"added new sample Id = " << Id << std::endl;
}

void register_camera(unsigned int camera_id, const platform_camera_parameters * param)
{
	// Supports only one camera at this time.
	if(camera_parameters.size() < 1)
	{
		camera_parameters.push_back(*param);
		precompute_world_pos_lookup(camera_id);
	} else {
		if(bPrintDebugMsg) std::cout <<"WARNING: Library supports only one camera for now" << std::endl;
	}
}

int get_registered_sample_size()
{
	return registered_sample.size();
}

void set_sample_filter(const std::vector<unsigned int> &filter)
{
	// Todo: implement filter to valid samples
}


/*
 * Gives the world position of a pixel
 * @ pos	  : Location of the desired pixel
 * @ camera ID: Assuming that there is only 1 camera for now
 */
WORLD get_world_pos(unsigned int cameraId, PIXEL pos)
{
	WORLD world_pos = {INVALID,INVALID};
	// Check if the pixel is within the range
	if(pos.u < camera_parameters[cameraId].Hpixels && pos.v > camera_parameters[cameraId].Vpixels)
	{
		world_pos = WORLD_LOOKUP[pos.u*camera_parameters[cameraId].Hpixels+pos.v];
	} else {
		if(bPrintDebugMsg) 	std::cout << "image pixel out of range "<< std::endl;
	}
	return world_pos;
}

bool process_image(cv::Mat image_hsv,cv::Mat *out_image, int index,std::vector<DETECTED_SAMPLE> &detected_samples)
{    
	DETECTED_SAMPLE sample;
	PIXEL Center; //center of the bounding box;
	WORLD World;
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
        // Get the pixel coordinates and return (x,y) from the lookup
        Center.u = boundRect[i].tl().x + boundRect[i].br().x;
        Center.v = boundRect[i].tl().y + boundRect[i].br().y;
        World = get_world_pos(DEFAULT_CAMERAID,Center);
        sample.x = World.x;
        sample.y = World.y;
        sample.projected_width = boundRect[i].area();
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
