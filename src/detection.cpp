#include <math.h>
#include "detection.h"
#include <pthread.h>
#include <stdio.h>

//#define DEBUG_DUMP
#define USE_GLOBAL_THRESHOLD  		(1)
#define USE_ADAPTIVE_THRESHOLD		(!USE_GLOBAL_THRESHOLD)
#define USE_MORPHOLOGICAL_OPS 		(1)
#define ENABLE_TEXTURE_TEST			(1)
#define ENABLE_TIMING				(1)
#define USE_THREADS  				(1)

// Maximum/Minimum distance the detector will detect samples - any detections outside this range are not considered
#define MAX_DISTANCE_DETECTION			4.5
#define MIN_DISTANCE_DETECTION			0.5

// Number of row pixels to remove from the bottom of the image to create ROI.
// At the current pitch, tray is in the way causing reflections and thus false detections.
#define ROWS_TO_ELIMINATE_AT_BOTTOM (95)
#define ROWS_TO_ELIMINATE_AT_TOP    		(35)

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

#define CLOCKS_PER_MS   (CLOCKS_PER_SEC/1000)

bool texture_ready = false;

struct texture_struct{
	cv::Mat in;
	cv::Mat out;
};

pthread_t texture_thread;
texture_struct texture_data;


void Display_time(double  time_elapsed)
{
	std::cout << "-------------------------" << std::endl;
	std::cout<<"Time in secs: "<< time_elapsed <<std::endl;
	std::cout << "-------------------------" << std::endl;
}

// Set debug messages OFF by default
LOGLEVEL bPrintDebugMsg = OFF;

cv::Mat Input_image, erosion_element, dilation_element;

// Gaussian kernels in X and Y
cv::Mat GKernelX = cv::getGaussianKernel(159,50,CV_32FC1);
cv::Mat GKernelY = cv::getGaussianKernel(159,0,CV_32FC1);

// Init flag
bool bInit = false;

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

typedef struct
{


}sample_info;

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
	double width;
	double min_width;
	double max_width;
	double depth;
	double min_depth;
	double max_depth;
	bool isValid;  // Should we check for this sample in out detector
}REGISTERED_SAMPLE;

std::vector<REGISTERED_SAMPLE> registered_sample;
std::vector<DETECTED_SAMPLE> detected_samples;
std::vector<platform_camera_parameters>camera_parameters;

bool Initialize_lib()
{
	if(!bInit)
	{
		// Compute the structuring elements for erosion and dilation before hand
		int erosion_size = 4;
		int dilation_size = 4;
		// erode and dilate
		erosion_element = cv::getStructuringElement( cv::MORPH_RECT,
											 cv::Size( 2*erosion_size + 1, 2*erosion_size+1 ),
											 cv::Point( erosion_size, erosion_size ) );

		dilation_element = cv::getStructuringElement( cv::MORPH_RECT,
											 cv::Size( 2*dilation_size + 1, 2*dilation_size+1 ),
											 cv::Point( dilation_size, dilation_size ) );
	}
	return true;
}

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

/*
 * registers a sample to the database
 */
void register_sample(unsigned int Id, const std::vector<double>&hue_param,
									  const std::vector<double>&sat_param,
									  const std::vector<double>&val_param,
									  const std::vector<double>width,
									  const std::vector<double>depth)
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

		new_sample.width = width[0];
		new_sample.min_width = width[1];
		new_sample.max_width = width[2];

		new_sample.depth = depth[0];
		new_sample.min_depth = depth[1];
		new_sample.max_depth = depth[2];

		new_sample.isValid = true; // true by default for all samples

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

	if(!bInit)
	{
		Initialize_lib();
		bInit = true;
	}
}

int get_registered_sample_size()
{
	return registered_sample.size();
}

void set_sample_filter(const std::vector<bool> &filter)
{
	for(int i=0;i< registered_sample.size(); ++i)
	{
		registered_sample[i].isValid = filter[i];
	}
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
		if(bPrintDebugMsg > DEBUG)		std::cout << "pixel out of range: check camera parameters "<< std::endl;
	}
	return world_pos;
}

void GetTextureImage(void* arguments)
{
#if (CUDA_GPU)
	cv::gpu::GpuMat gpu_in, gpu_in2, gpu_out;
#endif
	/// Generate grad_x and grad_y
	cv::Mat grad_x, grad_y,smoothed_image, normalized_image;
	cv::Mat abs_grad_x, abs_grad_y, erosion_dst, dilation_dst;

	struct texture_struct *args =(struct texture_struct *) arguments;
	cv::Mat src = args->in;
	cv::Mat dst = args->out;

	int scale = 1;
	int delta = 0;
	int ddepth = -1;
	int kern = 7;

	/// Gradient X
#if (CUDA_GPU)
	gpu_in.upload(src);
	cv::gpu::Sobel(gpu_in, gpu_out, ddepth, 1, 0, kern, scale, cv::BORDER_DEFAULT);
	gpu_out.download(grad_x);
#else
	cv::Sobel( src, grad_x, ddepth, 1, 0, kern, scale, delta, cv::BORDER_DEFAULT );
#endif
	cv::convertScaleAbs( grad_x, abs_grad_x );

	/// Gradient Y
#if (CUDA_GPU)
	gpu_in.upload(src);
	cv::gpu::Sobel(gpu_in, gpu_out, ddepth, 0, 1, kern, scale, cv::BORDER_DEFAULT);
	gpu_out.download(grad_y);
#else
	cv::Sobel( src, grad_y, ddepth, 0, 1, kern, scale, delta, cv::BORDER_DEFAULT );
#endif
	cv::convertScaleAbs( grad_y, abs_grad_y );

	/// Total Gradient (approximate)
#if (CUDA_GPU)
	gpu_in.upload(abs_grad_x);
	gpu_in2.upload(abs_grad_y);
	cv::gpu::addWeighted( gpu_in, 0.5, gpu_in2, 0.5, 0, gpu_out );
	gpu_out.download(dst);
#else
	cv::addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, dst );
#endif
	DUMP_IMAGE(dst,"/tmp/texture_derivative_out.png");

#if (CUDA_GPU & 0)
	cv::gpu::boxFilter(gpu_in, gpu_out,-1, cv::Size(39,39));
	gpu_out.download(smoothed_image);
#else
	cv::medianBlur(dst,smoothed_image,39);
#endif
	DUMP_IMAGE(smoothed_image,"/tmp/texture_median_out.png");

	cv::normalize(smoothed_image,normalized_image,1.0,255.0,cv::NORM_MINMAX);
	DUMP_IMAGE(normalized_image,"/tmp/texture_normalised_out.png");

	cv::subtract(255.0,normalized_image,dst);

#if (USE_MORPHOLOGICAL_OPS)
	#if (CUDA_GPU)
		gpu_in.upload(dst);
		cv::gpu::erode(gpu_in, gpu_in2, erosion_element);
		cv::gpu::dilate(gpu_in2, gpu_out, dilation_element);
		gpu_in2.download(erosion_dst);
		gpu_out.download(dst);
	#else
		cv::erode(dst,erosion_dst,erosion_element);
		cv::dilate(erosion_dst,dst,dilation_element);
	#endif // CUDA
		DUMP_IMAGE(dst,"/tmp/texture_dilation_out.png");
		DUMP_IMAGE(erosion_dst,"/tmp/texture_erosion_out.png");
#endif //USE_MORPHOLOGICAL_OPS
}


#if( USE_THREADS)
void *GetTextureImageThread(void * arguments)
{
	cv::Mat texture_out;

#if (CUDA_GPU)
	cv::gpu::GpuMat gpu_in, gpu_in2, gpu_out;
#endif
	/// Generate grad_x and grad_y
	cv::Mat grad_x, grad_y,smoothed_image, normalized_image;
	cv::Mat abs_grad_x, abs_grad_y, erosion_dst, dilation_dst;

	struct texture_struct *args =(struct texture_struct *) arguments;
	cv::Mat src_gray = args->in;

	int scale = 1;
	int delta = 0;
	int ddepth = -1;
	int kern = 7;

	/// Gradient X
#if (CUDA_GPU)
	gpu_in.upload( src_gray);
	cv::gpu::Sobel(gpu_in, gpu_out, ddepth, 1, 0, kern, scale, cv::BORDER_DEFAULT);
	gpu_out.download(grad_x);
#else
	cv::Sobel( src_gray, grad_x, ddepth, 1, 0, kern, scale, delta, cv::BORDER_DEFAULT );
#endif
	cv::convertScaleAbs( grad_x, abs_grad_x );

	/// Gradient Y
#if (CUDA_GPU)
	gpu_in.upload( src_gray);
	cv::gpu::Sobel(gpu_in, gpu_out, ddepth, 0, 1, kern, scale, cv::BORDER_DEFAULT);
	gpu_out.download(grad_y);
#else
	cv::Sobel( src_gray, grad_y, ddepth, 0, 1, kern, scale, delta, cv::BORDER_DEFAULT );
#endif
	cv::convertScaleAbs( grad_y, abs_grad_y );

	/// Total Gradient (approximate)
#if (CUDA_GPU)
	gpu_in.upload(abs_grad_x);
	gpu_in2.upload(abs_grad_y);
	cv::gpu::addWeighted( gpu_in, 0.5, gpu_in2, 0.5, 0, gpu_out );
	gpu_out.download(texture_out);
#else
	cv::addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, texture_out );
#endif
	//DUMP_IMAGE(dst,"/tmp/texture_derivative_out.png");


#if (CUDA_GPU&0)
	cv::gpu::boxFilter(gpu_in, gpu_out,-1, cv::Size(39,39));
	gpu_out.download(smoothed_image);
#else
	cv::medianBlur(texture_out,smoothed_image,39);
#endif
	DUMP_IMAGE(smoothed_image,"/tmp/texture_median_out.png");

	cv::normalize(smoothed_image,normalized_image,1.0,255.0,cv::NORM_MINMAX);
	DUMP_IMAGE(normalized_image,"/tmp/texture_normalised_out.png");

	cv::subtract(255.0,normalized_image,texture_out);

#if (USE_MORPHOLOGICAL_OPS)
	#if (CUDA_GPU)
		gpu_in.upload(texture_out);
		cv::gpu::erode(gpu_in, gpu_in2, erosion_element);
		cv::gpu::dilate(gpu_in2, gpu_out, dilation_element);
		gpu_in2.download(erosion_dst);
		gpu_out.download(texture_out);
	#else
		cv::erode(texture_out,erosion_dst,erosion_element);
		cv::dilate(erosion_dst,texture_out,dilation_element);
	#endif // CUDA
#endif //USE_MORPHOLOGICAL_OPS
	DUMP_IMAGE(texture_out,"/tmp/texture_out_thread.png");

	args->out = texture_out;
}
#endif

void generate_heat_map_LAB(cv::Mat &in_lab,const channel_info & L_info,
									   const channel_info & a_info,
									   const channel_info & b_info,cv::Mat &out,
									   std::vector<cv::Mat>&image_planes)
{
	cv::Mat input = in_lab,response,L_median,L_inter;
	cv::Mat L_channel = image_planes[0];
	cv::Mat a_channel = image_planes[1];
	cv::Mat b_channel = image_planes[2];

#if (CUDA_GPU)
    cv::gpu::GpuMat gpu_in, gpu_gblur, gpu_div, gpu_out;
	gpu_in.upload(L_channel);
	cv::gpu::boxFilter(gpu_in, gpu_out, -1, cv::Size(11,11));
	gpu_out.download(L_median);
	DUMP_IMAGE(L_median,"/tmp/L_median.png");
	gpu_out.convertTo(gpu_out, CV_32FC1);
	gpu_out.download(L_inter);
#else
	cv::medianBlur(L_channel,L_median,11);
	DUMP_IMAGE(L_median,"/tmp/L_median.png");
	L_channel = L_median;
	L_channel.convertTo(L_inter,CV_32FC1);
#endif

	response = cv::Mat::zeros(L_channel.rows,L_channel.cols,CV_32FC1);

	int32_t L_ref = L_info.origin;
	int32_t a_ref = a_info.origin;
	int32_t b_ref = b_info.origin;

	// Assuming uniform deviation for now.
	const int32_t max_L_dev = L_info.deviation;
	const int32_t max_a_dev = a_info.deviation;
	const int32_t max_b_dev = b_info.deviation;

	const float L_weighting_factor = L_info.weight;
	const float a_weighting_factor = a_info.weight;
	const float b_weighting_factor = b_info.weight;

	const float L_mult_factor = 1.0/static_cast<float>(max_L_dev);
	const float a_mult_factor = 1.0/static_cast<float>(max_a_dev);
	const float b_mult_factor = 1.0/static_cast<float>(max_b_dev);

	for(int rows = 0 ;rows < input.rows; rows++)
	{
		for(int cols =0; cols < input.cols; cols++)
		{
			int32_t L = static_cast<int32_t>(L_channel.at<uint8_t>(rows,cols));
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

			response.at<float>(rows,cols) = response_value * 255.0;
		}
	}
	out = response;
}

void getPixelCount(unsigned int camera_index, unsigned int sample_index, double Dist2Sample, double &min_size, double &max_size)
{
	double K1 =  camera_parameters[camera_index].Hpixels/1.4;
	double K2 = camera_parameters[camera_index].Vpixels/0.7;

	double width = registered_sample[sample_index].width;
	double height = registered_sample[sample_index].width;

	double min_width = width - registered_sample[sample_index].min_width;
	double max_width = width + registered_sample[sample_index].max_width;
	double min_height = height - registered_sample[sample_index].min_depth;
	double max_height = height + registered_sample[sample_index].max_depth;

	double min_width_angle = std::atan(min_width/(2*Dist2Sample));
	double max_width_angle = std::atan(max_width/(2*Dist2Sample));

	double min_height_angle = std::atan(min_height/(2*Dist2Sample));
	double max_height_angle = std::atan(max_height/(2*Dist2Sample));

	min_size = 4*K1*K2 * min_width_angle * min_height_angle ;
	max_size = 4*K1*K2 * max_width_angle * max_height_angle ;

	if (bPrintDebugMsg > VERBOSE)
	{
		std::cout << "K1: " << K1 << std::endl;
		std::cout << "K2: " << K2 << std::endl;
		std::cout << "Distance to sample: " << Dist2Sample << std::endl;

		std::cout << "width: "<< width << std::endl;
		std::cout << "height: "<< height << std::endl;

		std::cout << "expected min_size: " << min_size << std::endl;
		std::cout << "expected min_size: " << min_size << std::endl;
	}
}

bool process_image(unsigned int camera_index,cv::Mat image_hsv,cv::Mat *out_image,
				   int index,std::vector<DETECTED_SAMPLE> &detected_samples,
				   std::vector<cv::Mat> &image_planes)
{

	std::vector<cv::Rect> BB_Points;

	cv::Mat heat_map;
    // Define cv::Mat structures to work with
#if (CUDA_GPU)
    cv::gpu::GpuMat gpu_in, gpu_in2, gpu_erode, gpu_dilate, gpu_out;
#else
    cv::Mat erosion_dst, dilation_dst;
#endif

    // define vectors of contour points
    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;

    // define variables
    double computed_area_in_pixels, height, dist, expected_area_in_pixels, min_expected_size, max_expected_size;

    // Struct to hold any detected samples
	DETECTED_SAMPLE sample;

	if(bPrintDebugMsg > DEBUG)
		std::cerr << "*******************************" << std::endl;

	PIXEL pxl_cntr_btm,  pxl_left_btm , pxl_right_btm, pxl_left_tp,pxl_right_tp, pxl_cntr_tp;
	WORLD world_cntr_btm, world_left_btm, world_right_btm,world_left_tp,world_right_tp, world_cntr_tp;

	// sample index is same for all samples this call
	sample.id = index;

	// Generate heat map in Lab color space
    generate_heat_map_LAB(image_hsv,
    														 registered_sample[index].channel1,
    														 registered_sample[index].channel2,
    														 registered_sample[index].channel3,
    														 heat_map,
    														 image_planes);

    DUMP_IMAGE(heat_map,"/tmp/heat_map.png");


#if (USE_THREADS)
	pthread_join(texture_thread,NULL);
#endif

	// Make sure that texture map is already computed and ready to use here
	if(texture_data.out.data && sample.id == WHITE)
	{
#if (CUDA_GPU)
		gpu_in.upload(texture_data.out);
		gpu_in2.upload(heat_map);
		gpu_in.convertTo(gpu_in, CV_32FC1);
		gpu_in2.convertTo(gpu_in2, CV_32FC1);
		cv::gpu::multiply(gpu_in, gpu_in2, gpu_out, 1/255.0,CV_32FC1);
		gpu_out.download(heat_map);
#else
			cv::multiply(texture_data.out,heat_map,heat_map,1/255.0,CV_32FC1);
#endif
	}

	DUMP_IMAGE(heat_map,"/tmp/heat_map_mul.png");

#if(USE_GLOBAL_THRESHOLD)

	#if (CUDA_GPU)
		cv::gpu::threshold(gpu_out, gpu_out,140,255,CV_THRESH_BINARY);
		gpu_out.download(heat_map);
	#else
		cv::threshold(heat_map,heat_map,140,255,CV_THRESH_BINARY);
	#endif
		heat_map.convertTo(heat_map, CV_8UC1);
#elif(USE_ADAPTIVE_THRESHOLD)
    heat_map.convertTo(heat_map, CV_8UC1);
    cv::adaptiveThreshold(heat_map,heat_map,255,cv::ADAPTIVE_THRESH_MEAN_C,CV_THRESH_BINARY,151,-35);
#endif // USE_GLOBAL_THRESHOLD

    DUMP_IMAGE(heat_map,"/tmp/thresh_heat_map.png");


#if USE_MORPHOLOGICAL_OPS

	#if (CUDA_GPU)
		gpu_in.upload(heat_map);
		cv::gpu::erode(gpu_in, gpu_erode, erosion_element);
		cv::gpu::dilate(gpu_erode, gpu_dilate, dilation_element);
		gpu_dilate.download(heat_map);
	#else
		cv::erode(heat_map,erosion_dst,erosion_element);
		cv::dilate(erosion_dst,heat_map,dilation_element);
	#endif  // CUDA
		DUMP_IMAGE(heat_map,"/tmp/dilation_out.png");
#endif //USE_MORPHOLOGICAL_OPS

    // Find contours in the thresholded image to determine shapes
    cv::findContours(heat_map,contours,hierarchy,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE,cv::Point(0,0));

    if(bPrintDebugMsg > DEBUG)
    	std::cout << "Number of contours found: " << contours.size() <<std::endl;

    // Draw all the contours found in the previous step
    cv::Mat drawing = cv::Mat::zeros( heat_map.size(), CV_8UC3 );

    std::vector<std::vector<cv::Point> > contours_poly( contours.size() );
    std::vector<cv::Rect> boundRect(contours.size() );

    /*
     * Loop through all the contours and eliminate those that do not match various constraints
     * 1. Reject contours that do not pass area(size) check - we know that the sample is of a certain area
     * 2. Reject contours that do not pass distance check -  we know that the sample cannot be at unreal distances from the robot
     */
    for( int contourindex = 0;  contourindex < contours.size();  ++contourindex)
     {
    	const std::vector<cv::Point> & countour = contours_poly[contourindex];

        cv::approxPolyDP( cv::Mat(contours[contourindex]), contours_poly[contourindex], 10.0, false );
        boundRect[contourindex] = cv::boundingRect( cv::Mat(contours_poly[contourindex]) );

         // Get the pixel coordinates of the rectangular bounding box
        cv::Point tl = boundRect[contourindex].tl();
        cv::Point br = boundRect[contourindex].br();

        // Mid point of the bounding box bottom side
        pxl_cntr_btm.u = (tl.x + br.x)/2;
        pxl_cntr_btm.v = br.y;

        // Left point of the bounding box bottom side
        pxl_left_btm.u = tl.x;
        pxl_left_btm.v = br.y;

        // Right point of the bounding box bottom side
        pxl_right_btm.u = br.x;
        pxl_right_btm.v = br.y;

        // Left point of the bounding box top
        pxl_left_tp.u = tl.x;
        pxl_left_tp.v = tl.y;

        // Right point of the bounding box top
        pxl_right_tp.u = br.x;
        pxl_right_tp.v = tl.y;

        // Get world position of the above 3 pixels in world
        world_cntr_btm  = get_world_pos(camera_index,pxl_cntr_btm);
        world_left_btm  = get_world_pos(camera_index,pxl_left_btm);
        world_right_btm = get_world_pos(camera_index,pxl_right_btm);
        world_left_tp  = get_world_pos(camera_index,pxl_left_tp);
        world_right_tp  = get_world_pos(camera_index,pxl_right_tp);

        // These will be used to get the center and the width of the detected sample in world frame.
        sample.x = world_cntr_btm.x;
        sample.y = world_cntr_btm.y;

        // World position of the top center pixel in the bounding box
        world_cntr_tp.x = 0.5*(world_right_tp.x+ world_left_tp.x);
        world_cntr_tp.y = 0.5*(world_right_tp.y+ world_left_tp.y);

        // Sample world position
        sample.projected_width = std::abs(world_right_btm.y - world_left_btm.y);
        sample.projected_depth = std::abs(world_cntr_tp.x - world_cntr_btm.x);

        // Compute sample distance
        dist = std::max(std::sqrt(sample.x*sample.x + sample.y*sample.y +
        		 	 	 	 	  camera_parameters[camera_index].height*camera_parameters[camera_index].height),0.5);

        if(dist > MAX_DISTANCE_DETECTION || dist < MIN_DISTANCE_DETECTION)
        {
        	if (bPrintDebugMsg > DEBUG)  std::cout << "Too far or too close"<< std::endl;
        	continue;
        }

        // Compute sample size
       	getPixelCount(camera_index, index,dist,min_expected_size,max_expected_size);
       	computed_area_in_pixels = cv::contourArea(contours[contourindex]);

        if((computed_area_in_pixels > min_expected_size && computed_area_in_pixels < max_expected_size)
        		|| sample.id == PURPLE)  // forgive purple rock for size constraints for now
        {
			// Push the sample
			detected_samples.push_back(sample);

			if (out_image != NULL)
			{
				// store bounding boxes to draw later
				BB_Points.push_back(boundRect[contourindex]);

		        // Log all the positions
		        if(bPrintDebugMsg > DEBUG)
		        {
					std::cout << "TL: "<<tl.x << " "<< tl.y << std::endl;
					std::cout << "BR: "<<br.x << " "<< br.y << std::endl;

					// Pixel coordinates in Image frame
					std::cout << "pxl_cntr_btm:("<< pxl_cntr_btm.u <<","<< pxl_cntr_btm.v << ")"<<std::endl;
					std::cout << "pxl_left_btm:("<< pxl_left_btm.u <<","<< pxl_left_btm.v <<")" <<std::endl;
					std::cout << "pxl_right_btm:("<< pxl_right_btm.u <<","<< pxl_right_btm.v <<")" <<std::endl;
					std::cout << "pxl_left_tp:("<< pxl_left_tp.u <<","<< pxl_left_tp.v <<")"<<std::endl;
					std::cout << "pxl_right_tp:("<< pxl_right_tp.u <<" , "<< pxl_right_tp.v <<")" <<std::endl;

					// Pixel coordinates in World frame
		        	std::cout << "world_cntr_tp:("<<  world_cntr_tp.x <<","<< world_cntr_tp.y <<")" << std::endl;
		        	std::cout << "world_left_tp:("<<  world_left_tp.x <<","<< world_left_tp.y <<")" << std::endl;
		        	std::cout << "world_right_tp:("<<  world_right_tp.x <<"," << world_right_tp.y <<")" << std::endl;
		        	std::cout << "world_cntr_btm:("<<  world_cntr_btm.x <<","<< world_cntr_btm.y <<")" << std::endl;
					std::cout << "world_left_btm:("<<  world_left_btm.x <<"," << world_left_btm.y <<")" << std::endl;
					std::cout << "world_right_btm:( "<<  world_right_btm.x <<"," << world_right_btm.y <<")" << std::endl;

					// Sample world position
					std::cout << "sample:(" << sample.x  <<","<< sample.y  <<")" <<std::endl;

					// Sample width and depth in world
					std::cout << "sample width:  " << sample.projected_width<< std::endl;
					std::cout << "sample depth:  " << sample.projected_depth<< std::endl;

					// Area
					std::cout << "min_expected_area_in_pixels: "<<min_expected_size << std::endl;
					std::cout << "max_expected_area_in_pixels: "<<max_expected_size <<std::endl;
					std::cout << "computed_area_in_pixels: "<<computed_area_in_pixels << std::endl;
				}
			} else {
				if (bPrintDebugMsg > OFF)  std::cout << "Invalid output image pointer" << std::endl;
			}
		} else {
			if (bPrintDebugMsg > DEBUG)
			{
				std::cout << "measured area: " << computed_area_in_pixels		<< " "
									 << "min  area: " 				<< min_expected_size					<< " "
									 << "max  area: " 			<<max_expected_size					<< " "
									 << "Sample: " 					<<sample.id					<< " "				<< std::endl;
			}
		}
	}

    cv::Scalar color;
	// draw bounding boxes on the input image
	for (int index = 0; index < BB_Points.size(); ++index)
	{
		if(sample.id == WHITE) {
			 color = (0,0,255);
		} else if(sample.id == PURPLE) {
			color = (255,0,0);
		}

		rectangle(Input_image, BB_Points[index].tl(), BB_Points[index].br(),color, 2, 8, 0);
		//cv::drawContours(Input_image,contours,index,(255,0,0),1,8,hierarchy);
	}



	// Clear the samples for next iteration
	BB_Points.clear();

    // Print the number of samples found
	if(bPrintDebugMsg > DEBUG)
		std::cout << "Number of samples found: "<< detected_samples.size() << std::endl;
    return true;
}

void find_objects(unsigned int camera_index,const cv::Mat *imgPtr, cv::Mat *out_image,std::vector<DETECTED_SAMPLE> &detected_samples)
{
#ifdef ENABLE_TIMING
	// Get clock
	clock_t start_s=clock();
	double t = (double)cv::getTickCount();
#endif
	texture_ready = false;
	// Initialise the library if not already done
	if(!bInit)
	{
		Initialize_lib();
		bInit = true;
	}

	cv::Mat lab_image,src_rescaled,src_gray;
	if(! imgPtr->data) {
		std::cout << "ERROR: could not read image"<< std::endl;
		return;
	}

	// Global mat structure to work with
	Input_image = *imgPtr;

	// Add ROI to the image
	if(camera_index >= 0 && camera_index < MAX_CAMERAS_SUPPORTED)
	{
		Input_image = Input_image(cv::Rect(0									   ,    0,
																						  Input_image.cols, 	 Input_image.rows - ROWS_TO_ELIMINATE_AT_BOTTOM));
	} else {
		std::cout << "ERROR: Unknown number of camera's registered "<< std::endl;
		return;
	}


#if (CUDA_GPU)
	cv::gpu::GpuMat gpu_in, gpu_out;
	gpu_in.upload(Input_image);
	cv::gpu::cvtColor(gpu_in, gpu_out,CV_RGB2Lab);
	gpu_out.download(lab_image);
#else
	// Convert the color space to Lab
	cv::cvtColor(Input_image,lab_image,CV_RGB2Lab);
	DUMP_IMAGE(Input_image,"/tmp/input.png");
#endif

#ifdef ENABLE_TEXTURE_TEST
#if (CUDA_GPU)
        gpu_in.upload(Input_image);
        cv::gpu::cvtColor(gpu_in, gpu_out,CV_RGB2GRAY);
        gpu_out.download(src_gray);
#else 
	cv::cvtColor(Input_image,src_gray,CV_RGB2GRAY);
#endif // CUDA_GPU

	texture_data.in = src_gray;

#if (USE_THREADS)
	 // Create thread to handle texture image
	pthread_create(&texture_thread,NULL,&GetTextureImageThread,&texture_data);
#else
	GetTextureImage(&texture_data);
#endif
#endif // ENABLE_TEXTURE_TEST

	// Clear detected_samples structure before filling in with new data
	detected_samples.clear();

	// Create 3 Mat structures to hold each plane of Lab color space
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

	DUMP_IMAGE(channel1,"/tmp/channel1.png");
	DUMP_IMAGE(channel2,"/tmp/channel2.png");
	DUMP_IMAGE(channel3,"/tmp/channel3.png");

	// Get the iterator for the vector color space and loop through all sample color's
	for(int index = 0; index < registered_sample.size(); ++index)
	{
		if(registered_sample[index].isValid)
		{
			process_image(camera_index,lab_image, out_image,index,detected_samples,image_planes);
		}
	}

	// Log image
	DUMP_IMAGE(Input_image,"/tmp/BB.png");

#ifdef ENABLE_TIMING
	Display_time(((double)cv::getTickCount() - t)/cv::getTickFrequency());
#endif
}
