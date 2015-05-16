#include <math.h>
#include "detection.h"

using namespace cv;

#define MAX_SAMPLES (256)
#define PI          (3.14159265)
#define DEFAULT_CAMERAID (0)
#define DEFAULT_MAX_DIST (5.0)

// Globals
cv::RNG rng(12345);  // Don't panic, used only for color display
int kernel_size = 2;

// Init flag
bool bInit = false;
// Set debug messages OFF by default
LOGLEVEL bPrintDebugMsg = OFF; // Turn OFF later

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
	std::vector<int> H_Range;  // Origin, min and max values to filter out for
	std::vector<int> S_Range;
	std::vector<int> V_Range;
	std::vector<double> HSV_Weights;
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
void Set_debug(LOGLEVEL level)
{
	bPrintDebugMsg = level;
	if(bPrintDebugMsg > DEBUG)
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
void precompute_world_lookup(unsigned int cameraId)
{
	PIXEL pixel_pos;
	WORLD world_pos, world_pos_copy;
	double elevation, azimuth, R;
	double c_x,c_y,c_z;
	double alpha = camera_parameters[cameraId].pitch;

	double c_theta = cos(camera_parameters[cameraId].yaw);
	double s_theta = sin(camera_parameters[cameraId].yaw);

	for(int i =0; i < camera_parameters[cameraId].Hpixels; ++i)
	{
		for(int j =0; j < camera_parameters[cameraId].Vpixels; ++j)
		{
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
				if((next_x*next_x + next_y*next_y) >
						camera_parameters[cameraId].max_detection_dist*camera_parameters[cameraId].max_detection_dist)
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
			// Offset for camera position w.r.t robot position
			world_pos.x = camera_parameters[cameraId].x_offset +
							(c_theta*world_pos_copy.x - s_theta*world_pos_copy.y) ;
			world_pos.y = camera_parameters[cameraId].y_offset +
							(s_theta*world_pos_copy.x + c_theta*world_pos_copy.y) ;

			/*** Debug Print ****/
			if(bPrintDebugMsg == VERBOSE)
			{
				if(((i == 0) && (j == 0)) ||
				   ((i == camera_parameters[cameraId].Hpixels-1) && (j == 0)) ||
				   ((i == 0) && (j == camera_parameters[cameraId].Vpixels-1)) ||
				   ((i == camera_parameters[cameraId].Hpixels-1) && (j == camera_parameters[cameraId].Vpixels-1)) )
				{
					std::cout << "i: " << i << "j: " << j<< std::endl;
					std::cout << "pixel_pos.u : "<< pixel_pos.u << std::endl;
					std::cout << "pixel_pos.v : "<< pixel_pos.v << std::endl;
					std::cout << "C_X: "<< c_x << std::endl;
					std::cout << "C_Y: "<< c_y << std::endl;
					std::cout << "C_Z: "<< c_z << std::endl;
					std::cout << "R: "<< R << std::endl;
					std::cout << "X:  "<< world_pos.x << " " << "Y:  "<< world_pos.y << std::endl;
				}
			}

			WORLD_LOOKUP.push_back(world_pos);
		}
	}
}

/*
 * registers a sample to the database
 */
void register_sample(unsigned int Id, const std::vector<int>&hue_detection_range,
						  const std::vector<int>&sat_detection_range,
						  const std::vector<int>&val_detection_range,
						  const std::vector<double>&hsv_weights,double min_width,
						  double max_width, double min_height, double max_height) {
		REGISTERED_SAMPLE new_sample;
		new_sample.Id = Id;
		new_sample.H_Range = hue_detection_range;
		new_sample.S_Range = sat_detection_range;
		new_sample.V_Range = val_detection_range;
		new_sample.HSV_Weights = hsv_weights;
		new_sample.min_width = min_width;
		new_sample.max_width = max_width;
		new_sample.min_height = min_height;
		new_sample.max_height = max_height;
		new_sample.isValid = true; // true by default for all samples

		registered_sample.push_back(new_sample);
		if(bPrintDebugMsg > ERROR) std::cout<<"added new sample Id = " << Id << std::endl;
}


void register_camera(unsigned int camera_id, const platform_camera_parameters * param)
{
	// Supports only one camera at this time.
	if(camera_parameters.size() < 1)
	{
		camera_parameters.push_back(*param);
		precompute_world_lookup(camera_id);
	} else {
		if(bPrintDebugMsg > DEBUG) std::cout <<"WARNING: Library supports only one camera for now" << std::endl;
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
WORLD get_world_pos(unsigned int cameraId, PIXEL &pos)
{
	WORLD world_pos = {DEFAULT_MAX_DIST,DEFAULT_MAX_DIST};
	int index = 0;
	// Check if the pixel is within the range
	if(pos.u < camera_parameters[cameraId].Hpixels && pos.v < camera_parameters[cameraId].Vpixels)
	{
		index = pos.u*camera_parameters[cameraId].Vpixels + pos.v;
		world_pos = WORLD_LOOKUP[index];
	} else {
		if(bPrintDebugMsg > DEBUG) 	std::cout << "image pixel out of range "<< std::endl;
	}
	return world_pos;
}

cv::Mat response;
void generate_heat_map(cv::Mat &in_hsv,
					   std::vector<int> &HRange,
					   std::vector<int> &SRange,
					   std::vector<int> &VRange,
		    		   std::vector<double> &HSVWeights,
		    		   cv::Mat &out) {

	cv::Mat input = in_hsv;

	// Vector of Mat elements to store H,S,V planes
	std::vector<cv::Mat> hsv_planes(3);

	cv::split( input, hsv_planes );

	cv::Mat H = hsv_planes[0];
	cv::Mat S = hsv_planes[1];
	cv::Mat V = hsv_planes[2];

	response = cv::Mat::zeros(H.rows,H.cols,CV_32FC1);

	int hue_ref = 0; //HRange[1]*255/179;
	int32_t sat_ref = 10 ; //SRange[1];
	int32_t val_ref = 175; //VRange[1];

	// Assuming uniform deviation for now.
	const int32_t MAX_HUE_DEV = 10;//std::abs(HRange[0] - HRange[2]);
	const int32_t MAX_SAT_DEV = 10;//std::abs(SRange[0] - SRange[2]);
	const int32_t MAX_VAL_DEV = 10;//std::abs(VRange[0] - VRange[2]);

	const float HUE_WEIGHTING_FACTOR = 0.65; //HSVWeights[0];//0.25;
	const float SAT_WEIGHTING_FACTOR = 0.35; //HSVWeights[1];//0.25;
	const float VAL_WEIGHTING_FACTOR = 0.0 ; //HSVWeights[2];//0.5;

	const float HUE_MULT_FACTOR = 1.0/static_cast<double>(MAX_HUE_DEV);
	const float SAT_MULT_FACTOR = 1.0/static_cast<double>(MAX_SAT_DEV);
	const float VAL_MULT_FACTOR = 1.0/static_cast<double>(MAX_VAL_DEV);


	for(int rows = 0 ;rows < input.rows; rows++)
	{
		for(int cols =0; cols < input.cols; cols++)
		{
			uint8_t hue_image = static_cast<uint32_t>(H.at<uint8_t>(rows,cols))*255/179;
			uint8_t hue_diff = hue_image - hue_ref;

			int32_t saturation = static_cast<int32_t>(S.at<uint8_t>(rows,cols));
			int32_t value = static_cast<int32_t>(V.at<uint8_t>(rows,cols));

			int32_t saturation_deviation = saturation - sat_ref;
			int32_t value_deviation = value - val_ref;
			int32_t hue_deviation = static_cast<int8_t>(hue_diff);

			saturation_deviation = std::max(-MAX_SAT_DEV,std::min(MAX_SAT_DEV,saturation_deviation));
			saturation_deviation = MAX_SAT_DEV - std::abs(saturation_deviation);
			float sat_factor = SAT_MULT_FACTOR * saturation_deviation;


			value_deviation = std::max(-MAX_VAL_DEV,std::min(MAX_VAL_DEV,value_deviation));
			value_deviation = MAX_VAL_DEV - std::abs(value_deviation);
			float val_factor = VAL_MULT_FACTOR * value_deviation;

			hue_deviation = std::max(-MAX_HUE_DEV,std::min(MAX_HUE_DEV,hue_deviation));
			hue_deviation = MAX_HUE_DEV - std::abs(hue_deviation);
			float hue_factor = HUE_MULT_FACTOR * hue_deviation;

			float response_value = hue_factor * HUE_WEIGHTING_FACTOR +
					               sat_factor * SAT_WEIGHTING_FACTOR +
					               val_factor * VAL_WEIGHTING_FACTOR;

			uint8_t image_value = response_value * 255;
			image_value = image_value > 160 ? 255 : 0;
			response.at<float>(rows,cols) = image_value;
		}
	}
	out = response;
}

int count = 0;
char file_name[100];

bool process_image(cv::Mat image_hsv,cv::Mat *out_image, int index,std::vector<DETECTED_SAMPLE> &detected_samples)
{    
	bool draw_sample = false;
	DETECTED_SAMPLE sample;

	std::cerr << "*******************************" << std::endl;

	PIXEL pxl_cntr_btm, pxl_left_btm , pxl_right_btm;
	WORLD world_cntr_btm, world_left_btm, world_right_btm;

	// sample index is same for all samples this call
	sample.id = index;

    cv::Mat temp_image1, temp_image2;
    std::vector<vector<Point> > contours;
    std::vector<Vec4i> hierarchy;

    generate_heat_map(image_hsv,
    		registered_sample[index].H_Range,registered_sample[index].S_Range,
    		registered_sample[index].V_Range,registered_sample[index].HSV_Weights,temp_image1);

    // Convert CV_32FC1 to CV_8UC1
    cv::imwrite("/home/sarath/out_before.png",response);

    response.convertTo(response, CV_8UC1);
    cv::imwrite("/home/sarath/out_after.png",response);



    // Mark all pixels in the required color range high and other pixels low.
    //inRange(image_hsv,registered_sample[index].HSV_MIN,registered_sample[index].HSV_MAX,temp_image1);

    // Gives the kernel shape for erosion.
    // To do: Experiment with different kernels
    Mat element = getStructuringElement( MORPH_RECT, Size(2*kernel_size+1,2*kernel_size+1), Point(0,0));

    // Erode the image to get rid of trace elements with similar color to the required sample
    //erode(response,temp_image2,element);

    // Find contours in the thresholded image to determine shapes
    findContours(response,contours,hierarchy,CV_RETR_TREE,CV_CHAIN_APPROX_SIMPLE,Point(0,0));

    // Draw all the contours found in the previous step
    Mat drawing = Mat::zeros( response.size(), CV_8UC3 );

    std::vector<vector<Point> > contours_poly( contours.size() );
    std::vector<Rect> boundRect(contours.size() );
    for( int i = 0; i < contours.size(); ++i)
     {
    	std::cout << "here1  " << std::endl;
    	const std::vector<Point> & countour = contours_poly[i];

        approxPolyDP( Mat(contours[i]), contours_poly[i], 5, false );
        boundRect[i] = boundingRect( Mat(contours_poly[i]) );
        //boundRect[i] = boundingRect( Mat(contours[i]) );


        // Get the pixel coordinates of the rectangular bounding box
        Point tl = boundRect[i].tl();
        Point br = boundRect[i].br();
        if(bPrintDebugMsg > DEBUG)
        {
			std::cout << "TL: "<<tl.x << " "<< tl.y << std::endl;
			std::cout << "BR: "<<br.x << " "<< br.y << std::endl;
        }

        // Mid point of the bounding box bottom side
        pxl_cntr_btm.u = (tl.x + br.x)/2;
        pxl_cntr_btm.v = br.y;

        if(bPrintDebugMsg > DEBUG)
        {
			std::cout << "pxl_cntr_btm X:  "<< pxl_cntr_btm.u << std::endl;
			std::cout << "pxl_cntr_btm Y:  "<< pxl_cntr_btm.v << std::endl;
        }

        // Left point of the bounding box bottom side
        pxl_left_btm.u = tl.x;
        pxl_left_btm.v = br.y;

        if(bPrintDebugMsg > DEBUG)
        {
			std::cout << "pxl_left_btm X:  "<< pxl_left_btm.u << std::endl;
			std::cout << "pxl_left_btm Y:  "<< pxl_left_btm.v << std::endl;
        }

        // Left point of the bounding box bottom side
        pxl_right_btm.u = br.x;
        pxl_right_btm.v = br.y;

        if(bPrintDebugMsg > DEBUG)
        {
			std::cout << "pxl_right_btm X:  "<< pxl_right_btm.u << std::endl;
			std::cout << "pxl_right_btm Y:  "<< pxl_right_btm.v << std::endl;
        }

        // Get world position of the above 3 pixels in world
        world_cntr_btm  = get_world_pos(DEFAULT_CAMERAID,pxl_cntr_btm);
        world_left_btm  = get_world_pos(DEFAULT_CAMERAID,pxl_left_btm);
        world_right_btm = get_world_pos(DEFAULT_CAMERAID,pxl_right_btm);

        // These will be used to get the center and the width of the detected sample in world frame.
        sample.x = world_cntr_btm.x;
        sample.y = world_cntr_btm.y;

        sample.projected_width = std::abs(world_right_btm.y - world_left_btm.y);
        std::cout << sample.projected_width << std::endl;

        if(bPrintDebugMsg > DEBUG)
        {
			std::cout << "world_right_btm Y:  "<< world_right_btm.y << std::endl;
			std::cout << "world_left_btm  Y:  "<< world_left_btm.y << std::endl;
			std::cout << "diff  Y:  "<< world_right_btm.y - world_left_btm.y << std::endl;
        }

        if(sample.projected_width > registered_sample[index].min_width &&
        		sample.projected_width < registered_sample[index].max_width)
        {
           	// Push the sample
			detected_samples.push_back(sample);

			if(bPrintDebugMsg > ERROR)
			{
				std::cout << "sample X:  "<< sample.x << std::endl;
				std::cout << "sample Y:  "<< sample.y << std::endl;
				std::cout << "sample width:  "<< sample.projected_width << std::endl;
			}

			if(out_image != NULL)
			{
			   Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
			   drawContours( drawing, contours_poly, i, color, 2, 8, hierarchy, 0, Point() );

			   // Draw a bounding box
			   rectangle(Input_image, boundRect[i].tl(), boundRect[i].br(), (0,0,255), 2, 8, 0 );
			} else {
		    	if(bPrintDebugMsg > OFF)std::cout << "img ptr null" << std::endl;
		    }
        } else {
        	if(bPrintDebugMsg > DEBUG)  std::cout << "detected very small sample" <<std::endl;
        }
     }

    // Print the number of samples found
    if(bPrintDebugMsg > DEBUG) std::cout << "Number of samples found: "<< detected_samples.size() << std::endl;
    cv::imwrite("/home/sarath/bounding_box.png",Input_image);
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
