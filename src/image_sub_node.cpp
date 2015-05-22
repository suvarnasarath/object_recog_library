#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include "detection.h"

#define SIMULATOR

cv_bridge::CvImagePtr cv_ptr;
image_transport::Publisher pub;

void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{

    std::vector<DETECTED_SAMPLE> detected_samples;
	try {
		  std::cout << "." << std::endl;
		  cv_ptr = cv_bridge::toCvCopy(msg,"rgb8");
		  cv::Mat * imagePtr = &(cv_ptr->image);
		  cv::Mat *out_image = &(cv_ptr->image);

		  find_objects(0,imagePtr,out_image,detected_samples);
		  //std::cout << detected_samples.size() << std::endl;
		  sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "rgb8", *out_image).toImageMsg();
		  pub.publish(msg);
	  } catch (cv_bridge::Exception& e) {
	    ROS_ERROR("Could not convert from '%s' to 'rgb8'.", msg->encoding.c_str());
	  }
}

#if 0
void AddSampleforDetection(int id, int H_min,int S_min,int V_min,int H_max,int S_max,int V_max,
		int H_origin,int S_origin,int V_origin,double min_width, double max_width,double min_height,double max_height,
		double H_weight,double S_weight,double V_weight)
{
	std::vector<int> hue_detection_range{H_max,H_origin,H_min};
    std::vector<int> sat_detection_range{S_max,S_origin,S_min};
    std::vector<int> val_detection_range{V_max,V_origin,V_min};
    std::vector<double> hsv_weights{H_weight,S_weight,V_weight};
    register_sample(id,hue_detection_range,sat_detection_range,val_detection_range,
    				hsv_weights,min_width,max_width,min_height,max_height);
}
#endif

int main(int argc, char **argv)
{
	// Turn off debug messages.
	Set_debug(VERBOSE);
	/*******************************/
	/******** Register camera ******/
	/*******************************/
	platform_camera_parameters param;
	param.height = 0.7492; //0.65; // height of the camera from ground plane
	param.pitch = 0.366;//0.65;  // Pitch angle of the camera (up from down)
	param.HFov = 1.3962634;   // Horizontal field of view
	param.VFov = 0.7853981625;   // Vertical field of view
	param.Hpixels = 1920;
	param.Vpixels = 1080;
	param.max_detection_dist = 100.0;
	param.x_offset = 0.0;
	param.y_offset = 0.0;
	param.yaw = 0;
	param.min_bb_area_in_pixels =400;
	register_camera(0,&param);

	/********************************/
	/******** Register samples ******/
	/********************************/
#ifdef USE_HSV
	// White sample
	std::vector<double>Hue{100,10,0.30};
	std::vector<double>Sat{20,10,0.0};
	std::vector<double>Val{230,25,0.70};
	std::vector<double>width{0.05,0.3};
	std::vector<double>depth{0.05,0.3};
	register_sample(1,Hue,Sat,Val,width,depth);
#else

#ifndef SIMULATOR
	/*
	 * White
	 */
	std::vector<double>L{235,40,0.60};
	std::vector<double>a{128,20,0.20};
	std::vector<double>b{128,20,0.20};
	std::vector<double>width{0.03,0.3};
	std::vector<double>depth{0.05,4.6};
	double pixel_dist_factor_white = 6000;
	register_sample(1,L,a,b,width,depth,pixel_dist_factor_white);

#else
	/*
	 * White
	 */
	std::vector<double>L{235,20,0.40};
	std::vector<double>a{128,10,0.30};
	std::vector<double>b{128,10,0.30};
	std::vector<double>width{0.03,0.3};
	std::vector<double>depth{0.05,4.6};
	std::vector<double>moments{ 0.234046,
								0.015061,
								0.00422825,
								0.00475167,
								2.12851e-05,
								0.00058314,
								-7.57831e-07};
	double pixel_dist_factor_white = 6000;
	register_sample(1,L,a,b,width,depth,moments,pixel_dist_factor_white);
#endif
	/*
	 * Red
	 *
	std::vector<double>L_red{123,40,0.0};
	std::vector<double>a_red{200,60,0.65};
	std::vector<double>b_red{128,10,0.35};
	std::vector<double>width_red{0.02,0.2};
	std::vector<double>depth_red{0.02,0.9};
	double pixel_dist_factor_red = 600;
	register_sample(2,L_red,a_red,b_red,width_red,depth_red,pixel_dist_factor_red);
	*/

#endif

	/********************************/
	/********* Ros node handle ******/
	/********************************/
	ros::init(argc, argv, "image_listener");
	ros::NodeHandle nh;
	ros::NodeHandle np("~");
	std::string topic;

	if(!np.getParam("topic",topic))
	{
		topic = "camera/image";
	}
	image_transport::ImageTransport it(topic);
	image_transport::Subscriber sub = it.subscribe(topic, 1, imageCallback);
	pub = it.advertise("chatter",1);
	ros::spin();
}
