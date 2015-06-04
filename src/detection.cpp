#include <math.h>
#include "detection.h"
#include <time.h>

#define DEBUG_DUMP
#define USE_GLOBAL_THRESHOLD   (1)
#define USE_ADAPTIVE_THRESHOLD (!USE_GLOBAL_THRESHOLD)
#define USE_HSV_SPACE		   (0)
#define USE_LAB_SPACE		   (!USE_HSV_SPACE)
#define USE_MORPHOLOGICAL_OPS  (1)
#define ENABLE_DEPTH_TEST	   (0)
#define ENABLE_SHAPE_TEST	   (0)
#define ENABLE_TEXTURE_TEST    (1)
#define ENABLE_TIMING		   (1)

// Number of row pixels to remove from the bottom of the image to create ROI.
// At the current pitch, tray is in the way causing reflections and thus false detections.
#define ROWS_TO_ELIMINATE (65)

#ifdef DEBUG_DUMP
#define DUMP_IMAGE(IMG,FILE) cv::imwrite(FILE,IMG);
#else
#define DUMP_IMAGE(IMG,FILE)
#endif

#define LOG_TRANSF(X) (((X) < (0)) ? (-1*std::log10(std::abs(X))) : (1*std::log10(std::abs(X))))

#define DEFAULT_MAX_DIST (1000.0)
#define MIN_INTENSITY_THRESHOLD_VALUE (160)
#define Epsilon (0.001)
#define THRESHOLD (0.5)

#define CLOCKS_PER_SEC  1000000l
#define CLOCKS_PER_MS   (CLOCKS_PER_SEC/1000)
void Display_time(clock_t time_elapsed)
{
	std::cout << "-------------------------" << std::endl;
	std::cout<<"Total time: "<< time_elapsed <<std::endl;
	std::cout << "-------------------------" << std::endl;
}

// Set debug messages OFF by default
LOGLEVEL bPrintDebugMsg = OFF;

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

typedef struct
{
	int origin;
	int deviation;
	double weight;
}channel_info;

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
	channel_info channel1;
	channel_info channel2;
	channel_info channel3;
	double min_width;
	double max_width;
	double min_depth;
	double max_depth;
	bool isValid;  // Should we check for this sample in out detector
	double pixel_dist_factor;
	std::vector<double>moments;
}REGISTERED_SAMPLE;

std::vector<REGISTERED_SAMPLE> registered_sample;
std::vector<DETECTED_SAMPLE> detected_samples;
std::vector<platform_camera_parameters>camera_parameters;

/*
 * When enabled, prints debugging messages
 */
void set_debug(LOGLEVEL level)
{
	bPrintDebugMsg = level;
	if(bPrintDebugMsg > DEBUG)
	{
		std::cout << "Debug messages enabled:" << std::endl;
	}
}

void calculate_pixel_attitude_and_azimuth(unsigned int camera_index, PIXEL pixel_pos, double &elevation, double &azimuth)
{
	azimuth   = (pixel_pos.u * camera_parameters[camera_index].HFov)/static_cast<double>(camera_parameters[camera_index].Hpixels);
	elevation = (pixel_pos.v * camera_parameters[camera_index].VFov)/static_cast<double>(camera_parameters[camera_index].Vpixels);
}

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

			calculate_pixel_attitude_and_azimuth(cameraId,pixel_pos, elevation, azimuth);

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

std::vector<double> log_transform(std::vector<double> &in)
{
	std::vector<double>out(in.size(),0);
	std::vector<double>::iterator it;
	for(it=in.begin();it!=in.end();++it)
	{
		*it = LOG_TRANSF(*it);
		out.push_back(*it);
	}
	return out;
}

/*
 * registers a sample to the database
 */
void register_sample(unsigned int Id, const std::vector<double>&hue_param,
									  const std::vector<double>&sat_param,
									  const std::vector<double>&val_param,
									  const std::vector<double>width,
									  const std::vector<double>depth,
									  std::vector<double>&moments,
									  double pixel_dist_factor)
{
		REGISTERED_SAMPLE new_sample;
		new_sample.Id = Id;
		new_sample.channel1.origin = hue_param[0]+0.5;
		new_sample.channel1.deviation = hue_param[1]+0.5;
		new_sample.channel1.weight = hue_param[2];

		new_sample.channel2.origin = sat_param[0]+0.5;
		new_sample.channel2.deviation = sat_param[1]+0.5;
		new_sample.channel2.weight = sat_param[2];

		new_sample.channel3.origin = val_param[0]+0.5;
		new_sample.channel3.deviation = val_param[1]+0.5;
		new_sample.channel3.weight = val_param[2];

		new_sample.min_width = width[0];
		new_sample.max_width = width[1];

		new_sample.min_depth = depth[0];
		new_sample.max_depth = depth[1];
		new_sample.isValid = true; // true by default for all samples
		new_sample.pixel_dist_factor = pixel_dist_factor;

		new_sample.moments = log_transform(moments);

		registered_sample.push_back(new_sample);
		if(bPrintDebugMsg > ERROR) std::cout<<"added new sample Id = " << Id << std::endl;
}


void register_camera(unsigned int camera_id, const platform_camera_parameters *param)
{
	if(param)
	{
		// Check for already added cameras
		camera_parameters.push_back(*param);
		precompute_world_lookup(camera_id);
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
		if(bPrintDebugMsg > DEBUG)
			std::cout << "pixel out of range: check camera parameters "<< std::endl;
	}
	return world_pos;
}

void GetTextureImage(cv::Mat &src, cv::Mat &dst)
{
	/// Generate grad_x and grad_y
	cv::Mat grad_x, grad_y,smoothed_image, normalized_image;
	cv::Mat abs_grad_x, abs_grad_y, erosion_dst, dilation_dst;
	int scale = 1;
	int delta = 0;
	int ddepth = -1;

	/// Gradient X
	cv::Sobel( src, grad_x, ddepth, 1, 0, 7, scale, delta, cv::BORDER_DEFAULT );
	cv::convertScaleAbs( grad_x, abs_grad_x );

	/// Gradient Y
	cv::Sobel( src, grad_y, ddepth, 0, 1, 7, scale, delta, cv::BORDER_DEFAULT );
	cv::convertScaleAbs( grad_y, abs_grad_y );

	/// Total Gradient (approximate)
	cv::addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, dst );
	DUMP_IMAGE(dst,"/home/sarath/texture_derivative_out.png");

	cv::medianBlur(dst,smoothed_image,39);
	DUMP_IMAGE(smoothed_image,"/home/sarath/texture_median_out.png");

	cv::normalize(smoothed_image,normalized_image,1.0,255.0,cv::NORM_MINMAX);
	DUMP_IMAGE(normalized_image,"/home/sarath/texture_normalised_out.png");
	cv::subtract(255.0,normalized_image,dst);
#if (USE_MORPHOLOGICAL_OPS)
    int erosion_size = 4;
	int dilation_size = 4;

	// erode and dilate
	cv::Mat erosion_element = cv::getStructuringElement( cv::MORPH_RECT,
										 cv::Size( 2*erosion_size + 1, 2*erosion_size+1 ),
										 cv::Point( erosion_size, erosion_size ) );
	cv::erode(dst,erosion_dst,erosion_element);

	DUMP_IMAGE(erosion_dst,"/home/sarath/texture_erosion_out.png");
	cv::Mat dilation_element = cv::getStructuringElement( cv::MORPH_RECT,
											 cv::Size( 2*dilation_size + 1, 2*dilation_size+1 ),
											 cv::Point( dilation_size, dilation_size ) );

	cv::dilate(erosion_dst,dst,dilation_element);
	DUMP_IMAGE(dst,"/home/sarath/texture_dilation_out.png");
#endif
}


void generate_heat_map_in_HSV(cv::Mat &in_hsv,const channel_info & hue,
									   const channel_info & sat,
									   const channel_info & val,cv::Mat &out,
									   std::vector<cv::Mat>&image_planes)
{
	cv::Mat input = in_hsv,response;
	cv::Mat H = image_planes[0];
	cv::Mat S = image_planes[1];
	cv::Mat V = image_planes[2];

	response = cv::Mat::zeros(H.rows,H.cols,CV_32FC1);

	int hue_ref = hue.origin*255/179;       //0; //HRange[1]*255/179;hi
	int32_t sat_ref = sat.origin; 			//0 ; //SRange[1];
	int32_t val_ref = val.origin; 			//115; //VRange[1];

	// Assuming uniform deviation for now.
	const int32_t max_hue_dev = hue.deviation;//10;
	const int32_t max_sat_dev = sat.deviation;//5;
	const int32_t max_val_dev = val.deviation;//20;

	const float hue_weighting_factor = hue.weight; // 0.05;
	const float sat_weighting_factor = sat.weight; // 0.5;
	const float val_weighting_factor = val.weight; // 0.45;

	const float hue_mult_factor = 1.0/static_cast<double>(max_hue_dev);
	const float sat_mult_factor = 1.0/static_cast<double>(max_sat_dev);
	const float val_mult_factor = 1.0/static_cast<double>(max_val_dev);

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

			saturation_deviation = std::max(-max_sat_dev,std::min(max_sat_dev,saturation_deviation));
			saturation_deviation = max_sat_dev - std::abs(saturation_deviation);
			float sat_factor = sat_mult_factor * saturation_deviation;


			value_deviation = std::max(-max_val_dev,std::min(max_val_dev,value_deviation));
			value_deviation = max_val_dev - std::abs(value_deviation);
			float val_factor = val_mult_factor * value_deviation;

			hue_deviation = std::max(-max_hue_dev,std::min(max_hue_dev,hue_deviation));
			hue_deviation = max_hue_dev - std::abs(hue_deviation);
			float hue_factor = hue_mult_factor * hue_deviation;

			float response_value = hue_factor * hue_weighting_factor +
					               sat_factor * sat_weighting_factor +
					               val_factor * val_weighting_factor;

			//uint8_t image_value = response_value * 255;
			//image_value = image_value > MIN_INTENSITY_THRESHOLD_VALUE ? 255 : 0;
			response.at<float>(rows,cols) = response_value;
		}
	}
	out = response;
}


void generate_heat_map_LAB(cv::Mat &in_lab,const channel_info & L_info,
									   const channel_info & a_info,
									   const channel_info & b_info,cv::Mat &out,
									   std::vector<cv::Mat>&image_planes)
{

	cv::Mat input = in_lab,response,L_median,L_inter;
	cv::Mat L_channel = image_planes[0];
	cv::Mat a_channel = image_planes[1];
	cv::Mat b_channel = image_planes[2];

	std::cout << L_channel.type() << std::endl;

	cv::medianBlur(L_channel,L_median,11);
	DUMP_IMAGE(L_median,"/home/sarath/L_median.png");
	L_channel = L_median;
	L_channel.convertTo(L_inter,CV_32FC1);

    cv::Mat blurred_heatmap;//
    cv::Mat dst;// = cv::Mat::zeros(heat_map.rows,heat_map.rows,CV_8U);
    cv::Mat heat_map_copy;

    cv::GaussianBlur(L_inter,blurred_heatmap,cv::Size(159,159),50,0,cv::BORDER_DEFAULT);
    //cv::boxFilter(heat_map,blurred_heatmap,5,cv::Size(159,159));
    cv::add(1,blurred_heatmap,dst);
    DUMP_IMAGE(dst,"/home/sarath/heat_map_blur.png");

    cv::divide(L_inter,dst,L_inter);
    cv::multiply(128,L_inter,L_inter);
    DUMP_IMAGE(L_inter,"/home/sarath/heat_map_division.png");

	response = cv::Mat::zeros(L_channel.rows,L_channel.cols,CV_32FC1);

	int L_ref = L_info.origin;
	int32_t a_ref = a_info.origin;
	int32_t b_ref = b_info.origin;

	// Assuming uniform deviation for now.
	const int32_t max_L_dev = L_info.deviation;
	const int32_t max_a_dev = a_info.deviation;
	const int32_t max_b_dev = b_info.deviation;

	const float L_weighting_factor = L_info.weight;
	const float a_weighting_factor = a_info.weight;
	const float b_weighting_factor = b_info.weight;

	const float L_mult_factor = 1.0/static_cast<double>(max_L_dev);
	const float a_mult_factor = 1.0/static_cast<double>(max_a_dev);
	const float b_mult_factor = 1.0/static_cast<double>(max_b_dev);

	for(int rows = 0 ;rows < input.rows; rows++)
	{
		for(int cols =0; cols < input.cols; cols++)
		{
			uint8_t L = static_cast<int32_t>(L_channel.at<uint8_t>(rows,cols));
			int32_t a = static_cast<int32_t>(a_channel.at<uint8_t>(rows,cols));
			int32_t b = static_cast<int32_t>(b_channel.at<uint8_t>(rows,cols));

			int32_t L_deviation = L - L_ref;
			int32_t a_deviation = a - a_ref;
			int32_t b_deviation = b - b_ref;

			a_deviation = std::max(-max_a_dev,std::min(max_a_dev,a_deviation));
			a_deviation = max_a_dev - std::abs(a_deviation);
			float a_factor = a_mult_factor * a_deviation;
			a_factor = std::min(1.0,2.0*a_factor);


			b_deviation = std::max(-max_b_dev,std::min(max_b_dev,b_deviation));
			b_deviation = max_b_dev - std::abs(b_deviation);
			float b_factor = b_mult_factor * b_deviation;
			b_factor = std::min(1.0,2.0*b_factor);

			L_deviation = std::max(-max_L_dev,std::min(max_L_dev,L_deviation));
			L_deviation = max_L_dev - std::abs(L_deviation);
			float L_factor = L_mult_factor * L_deviation;
			L_factor = std::min(1.0,2.0*L_factor);

			float response_value = L_factor * L_weighting_factor +
					               a_factor * a_weighting_factor +
					               b_factor * b_weighting_factor;

			//const float THRESHOLD = static_cast<double>(MIN_INTENSITY_THRESHOLD_VALUE) / 255.0;
			//uint8_t image_value = response_value > THRESHOLD ? 255 : 0;
			response.at<float>(rows,cols) = response_value * 255.0;
		}
	}
	out = response;
}

bool compare_HuMoments(const std::vector<double> &GroundtruthHuMoments, const double *ComputedHuMoments)
{
	double Hu_similarity;
	for(int i =0; i < 7 ; i++)
	{
		Hu_similarity = std::abs(GroundtruthHuMoments[i] - LOG_TRANSF(ComputedHuMoments[i]));

		if(Hu_similarity < THRESHOLD) {
			continue;
		} else {
			return false;
		}
	}
	return true;
}

bool process_image(unsigned int camera_index,cv::Mat image_hsv,cv::Mat *out_image,
				   int index,std::vector<DETECTED_SAMPLE> &detected_samples,cv::Mat texture_image,
				   std::vector<cv::Mat> &image_planes)
{    
	bool draw_sample = false;
	DETECTED_SAMPLE sample;

	if(bPrintDebugMsg > DEBUG)
		std::cerr << "*******************************" << std::endl;

	PIXEL pxl_cntr_btm, pxl_left_btm , pxl_right_btm, pxl_left_tp,pxl_right_tp, pxl_cntr_tp;
	WORLD world_cntr_btm, world_left_btm, world_right_btm,world_left_tp,world_right_tp, world_cntr_tp;

	// sample index is same for all samples this call
	sample.id = index;

    cv::Mat heat_map, sobel_out, erosion_dst, dilation_dst;

    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;
    double contour_area;

#if(USE_HSV_SPACE)
    generate_heat_map_in_HSV(image_hsv,registered_sample[index].channel1,registered_sample[index].channel2,
    							registered_sample[index].channel3,heat_map,image_planes);
#elif(USE_LAB_SPACE)

    generate_heat_map_LAB(image_hsv,registered_sample[index].channel1,registered_sample[index].channel2,
        							registered_sample[index].channel3,heat_map,image_planes);
#endif

    DUMP_IMAGE(heat_map,"/home/sarath/heat_map.png");
#if 0
    cv::Mat blurred_heatmap;//
    cv::Mat dst;// = cv::Mat::zeros(heat_map.rows,heat_map.rows,CV_8U);
    cv::Mat heat_map_copy;

    cv::GaussianBlur(heat_map,blurred_heatmap,cv::Size(159,159),50,0,cv::BORDER_DEFAULT);
    //cv::boxFilter(heat_map,blurred_heatmap,5,cv::Size(159,159));
    cv::add(1,blurred_heatmap,dst);
    DUMP_IMAGE(dst,"/home/sarath/heat_map_blur.png");
    cv::divide(heat_map,dst,heat_map);
    cv::multiply(128,heat_map,heat_map);
    DUMP_IMAGE(heat_map,"/home/sarath/heat_map_division.png");
#endif

    if(texture_image.data)
    {
    	cv::multiply(texture_image,heat_map,heat_map,1/255.0,CV_32FC1);
    }
    DUMP_IMAGE(heat_map,"/home/sarath/heat_map_mul.png");

#if(USE_GLOBAL_THRESHOLD)
    cv::threshold(heat_map,heat_map,140,255,CV_THRESH_BINARY);
    heat_map.convertTo(heat_map, CV_8UC1);
#elif(USE_ADAPTIVE_THRESHOLD)
    heat_map.convertTo(heat_map, CV_8UC1);
    cv::adaptiveThreshold(heat_map,heat_map,255,cv::ADAPTIVE_THRESH_MEAN_C,CV_THRESH_BINARY,151,-35);
#endif
    DUMP_IMAGE(heat_map,"/home/sarath/thresh_heat_map.png");


#if USE_MORPHOLOGICAL_OPS
    int erosion_size = 4;
	int dilation_size = 4;

	// erode and dilate
	cv::Mat erosion_element = cv::getStructuringElement( cv::MORPH_RECT,
										 cv::Size( 2*erosion_size + 1, 2*erosion_size+1 ),
										 cv::Point( erosion_size, erosion_size ) );
	cv::erode(heat_map,erosion_dst,erosion_element);

	DUMP_IMAGE(erosion_dst,"/home/sarath/erosion_out.png");
	cv::Mat dilation_element = cv::getStructuringElement( cv::MORPH_RECT,
											 cv::Size( 2*dilation_size + 1, 2*dilation_size+1 ),
											 cv::Point( dilation_size, dilation_size ) );

	cv::dilate(erosion_dst,heat_map,dilation_element);
	DUMP_IMAGE(heat_map,"/home/sarath/dilation_out.png");
#endif
    // Find contours in the thresholded image to determine shapes
    cv::findContours(heat_map,contours,hierarchy,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE,cv::Point(0,0));

    // Draw all the contours found in the previous step
    cv::Mat drawing = cv::Mat::zeros( heat_map.size(), CV_8UC3 );

    std::vector<std::vector<cv::Point> > contours_poly( contours.size() );
    std::vector<cv::Rect> boundRect(contours.size() );

    for( int i = 0; i < contours.size(); ++i)
     {
    	const std::vector<cv::Point> & countour = contours_poly[i];

        cv::approxPolyDP( cv::Mat(contours[i]), contours_poly[i], 10.0, false );
        boundRect[i] = cv::boundingRect( cv::Mat(contours_poly[i]) );

        contour_area = cv::contourArea(contours[i]);

   	    if(contour_area < camera_parameters[camera_index].min_bb_area_in_pixels)
		{
   	    	//if(bPrintDebugMsg > DEBUG)std::cout << "failed area test: " << contour_area << std::endl;
			continue;
		} else {
			if(bPrintDebugMsg > DEBUG)std::cout << "passed area test: " << contour_area << std::endl;
		}

#if ENABLE_SHAPE_TEST
   	    double HuMoments[7];
   	    // Compute Moments
        cv::HuMoments(cv::moments(contours[i]),HuMoments);

        // Compare Hu moments of this contour with our stored Hu moments
        if(compare_HuMoments(registered_sample[index].moments,HuMoments))
        {
        	if(bPrintDebugMsg > DEBUG) std::cout << "Passed shape test: " << std::endl;
		} else {
			if(bPrintDebugMsg > DEBUG) std::cout << "Failed shape test: " << std::endl;
			continue;
		}
#endif

         // Get the pixel coordinates of the rectangular bounding box
        cv::Point tl = boundRect[i].tl();
        cv::Point br = boundRect[i].br();

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
			std::cout << "pxl_cntr_btm:("<< pxl_cntr_btm.u <<","<< pxl_cntr_btm.v << ")"<<std::endl;
        }

        // Left point of the bounding box bottom side
        pxl_left_btm.u = tl.x;
        pxl_left_btm.v = br.y;

        if(bPrintDebugMsg > DEBUG)
        {
			std::cout << "pxl_left_btm:("<< pxl_left_btm.u <<","<< pxl_left_btm.v <<")" <<std::endl;
        }

        // Left point of the bounding box bottom side
        pxl_right_btm.u = br.x;
        pxl_right_btm.v = br.y;

        if(bPrintDebugMsg > DEBUG)
        {
			std::cout << "pxl_right_btm:("<< pxl_right_btm.u <<","<< pxl_right_btm.v <<")" <<std::endl;
        }


        pxl_left_tp.u = tl.x;
        pxl_left_tp.v = tl.y;

        pxl_right_tp.u = br.x;
        pxl_right_tp.v = tl.y;

        if(bPrintDebugMsg > DEBUG)
        {
        	std::cout << "pxl_left_tp:("<< pxl_left_tp.u <<","<< pxl_left_tp.v <<")"<<std::endl;
        	std::cout << "pxl_right_tp:("<< pxl_right_tp.u <<" , "<< pxl_right_tp.v <<")" <<std::endl;
        }
        // Get world position of the above 3 pixels in world
        world_cntr_btm  = get_world_pos(camera_index,pxl_cntr_btm);
        world_left_btm  = get_world_pos(camera_index,pxl_left_btm);
        world_right_btm = get_world_pos(camera_index,pxl_right_btm);
        world_left_tp  = get_world_pos(camera_index,pxl_left_tp);
        world_right_tp  = get_world_pos(camera_index,pxl_right_tp);

        // These will be used to get the center and the width of the detected sample in world frame.
        sample.x = world_cntr_btm.x;
        sample.y = world_cntr_btm.y;

        world_cntr_tp.x = 0.5*(world_right_tp.x+ world_left_tp.x);
        world_cntr_tp.y = 0.5*(world_right_tp.y+ world_left_tp.y);


        if(bPrintDebugMsg > DEBUG)
		{
        	std::cout << "world_cntr_tp:("<<  world_cntr_tp.x <<","<< world_cntr_tp.y <<")" << std::endl;
        	std::cout << "world_left_tp:("<<  world_left_tp.x <<","<< world_left_tp.y <<")" << std::endl;
        	std::cout << "world_right_tp:("<<  world_right_tp.x <<"," << world_right_tp.y <<")" << std::endl;
        	std::cout << "world_cntr_btm:("<<  world_cntr_btm.x <<","<< world_cntr_btm.y <<")" << std::endl;
			std::cout << "world_left_btm:("<<  world_left_btm.x <<"," << world_left_btm.y <<")" << std::endl;
			std::cout << "world_right_btm:( "<<  world_right_btm.x <<"," << world_right_btm.y <<")" << std::endl;
		}


        sample.projected_width = std::abs(world_right_btm.y - world_left_btm.y);
        sample.projected_depth = std::abs(world_cntr_tp.x - world_cntr_btm.x);

        if (bPrintDebugMsg > DEBUG) {
			std::cout << "sample:(" << sample.x  <<","<< sample.y  <<")" <<std::endl;

			std::cout << "sample width:  " << sample.projected_width<< std::endl;
			std::cout << "sample depth:  " << sample.projected_depth<< std::endl;
		}

        if((sample.projected_width > registered_sample[index].min_width &&
        	sample.projected_width < registered_sample[index].max_width)
#if ENABLE_DEPTH_TEST
        		&&
           (sample.projected_depth > registered_sample[index].min_depth &&
        	sample.projected_depth < registered_sample[index].max_depth)
#endif
        	)
        {

          double height = camera_parameters[camera_index].height * camera_parameters[camera_index].height;
          double dist = std::max(std::sqrt(sample.x*sample.x +  sample.y*sample.y + height),1.0);
          double expected_area = registered_sample[index].pixel_dist_factor/dist;

          if(1/*contour_area > expected_area*/)
          {
        	  if(bPrintDebugMsg > DEBUG)
        		  std::cout << "accepted sample area: " << contour_area << " "
						  << "expected_area:  " <<expected_area <<std::endl;
				// Push the sample
				detected_samples.push_back(sample);

				if (out_image != NULL) {
					//cv::drawContours(Input_image, contours_poly, i, (0, 0, 255), 2, 8,hierarchy, 0, cv::Point());
					// Draw a bounding box
					rectangle(Input_image, boundRect[i].tl(), boundRect[i].br(),(0, 0, 255), 2, 8, 0);
				} else {
					if (bPrintDebugMsg > OFF)std::cout << "img ptr null" << std::endl;
				}
			} else {
				if (bPrintDebugMsg > DEBUG) {
					std::cout << "detected small contour" << std::endl;
					std::cout << "accepted sample area: " << contour_area << " "
										  << "expected_area:  " <<expected_area <<std::endl;
				}
			}
		} else {
			if (bPrintDebugMsg > DEBUG)
				std::cout << "detected very small sample" << std::endl;
		}
	}
    // Print the number of samples found
	if(bPrintDebugMsg > DEBUG)
		std::cout << "Number of samples found: "<< detected_samples.size() << std::endl;
    return true;
}

#if 0
void Compute_Luminance(cv::Mat &in, cv::Mat &out)
{
	double Luminance;
	std::vector<cv::Mat> image_planes(3);
	// Split the image in to 3 planes
	cv::split( in, image_planes );
	cv::Mat R = image_planes[0];
	cv::Mat G = image_planes[1];
	cv::Mat B = image_planes[2];

	for(int i=0;i<in.rows;++i)
	{
		for(int j=0;j<in.cols;++j)
		{
			out.at<int>(rows,cols) = 0.299*R.at<int>(rows,cols) + 0.587*G.at<int>(rows,cols) + 0.114*B.at<int>(rows,cols);
		}
	}
}
#endif

void find_objects(unsigned int camera_index,const cv::Mat *imgPtr, cv::Mat *out_image,std::vector<DETECTED_SAMPLE> &detected_samples)
{
#ifdef ENABLE_TIMING
	// Get clock
	clock_t start_s=clock();
#endif
	cv::Mat lab_image,src_gray,texture_out;
	if(! imgPtr->data) {
		std::cout << "ERROR: could not read image"<< std::endl;
		return;
	}

	Input_image = *imgPtr;

	// Add ROI to the image
	if(camera_index >= 0 && camera_index < MAX_CAMERAS_SUPPORTED)
	{
		Input_image = Input_image(cv::Rect( 0,0,
											camera_parameters[camera_index].Hpixels,
											camera_parameters[camera_index].Vpixels-ROWS_TO_ELIMINATE));
	} else {
		std::cout << "ERROR: Unknown number of camera's registered "<< std::endl;
		return;
	}

	// Convert the color space to Lab
	cv::cvtColor(Input_image,lab_image,CV_RGB2Lab);
	DUMP_IMAGE(Input_image,"/home/sarath/input.png");


#ifdef ENABLE_TEXTURE_TEST
	cv::cvtColor(Input_image,src_gray,CV_RGB2GRAY);
	GetTextureImage(src_gray,texture_out);
	DUMP_IMAGE(texture_out,"/home/sarath/texture_out.png");
#endif
	// Clear detected_samples structure before filling in with new data
	detected_samples.clear();

	cv::Mat channel1 = cv::Mat::zeros(Input_image.rows,Input_image.rows,1);
	cv::Mat channel2 = cv::Mat::zeros(Input_image.rows,Input_image.rows,1);
	cv::Mat channel3 = cv::Mat::zeros(Input_image.rows,Input_image.rows,1);

	// Vector of Mat elements to store individual image planes
	std::vector<cv::Mat> image_planes(3);

	// Split the image in to 3 planes
	cv::split( lab_image, image_planes );

	channel1 = image_planes[0];
	channel2 = image_planes[1];
	channel3 = image_planes[2];

	DUMP_IMAGE(channel1,"/home/sarath/channel1.png");
	DUMP_IMAGE(channel2,"/home/sarath/channel2.png");
	DUMP_IMAGE(channel3,"/home/sarath/channel3.png");

	// Get the iterator for the vector color space and loop through all sample color's
	for(int index = 0; index < registered_sample.size(); ++index)
	{
		if(registered_sample[index].isValid)
		{
		  process_image(camera_index,lab_image, out_image,index,detected_samples,texture_out,image_planes);
		}
	}
#ifdef ENABLE_TIMING
	clock_t stop_s=clock();  // end
	Display_time((stop_s - start_s)/CLOCKS_PER_MS);
#endif
}
