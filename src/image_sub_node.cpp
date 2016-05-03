#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include "detection.h"

#define MAX_DEPTH  100.0
#define MIN_DEPTH    0.0

//#define SIMULATOR

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
		  sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "rgb8", *out_image).toImageMsg();
		  pub.publish(msg);
	  } catch (cv_bridge::Exception& e) {
	    ROS_ERROR("Could not convert from '%s' to 'rgb8'.", msg->encoding.c_str());
	  }
}

int main(int argc, char **argv)
{
	// Turn off debug messages.
	set_debug(VERBOSE);

	/*******************************/
	/******** Register camera ******/
	/*******************************/
	platform_camera_parameters param;
	param.height = 0.90305;   						 // height of the camera from ground plane
	param.pitch = 0.6283185307179586; // 0.593411945678072; 	// Pitch angle of the camera (up from down)
	param.HFov = 1.3962634;   						 // Horizontal field of view
	param.VFov = 0.7853981625;					 // Vertical field of view
	param.Hpixels = 960;
	param.Vpixels = 720;
	param.max_detection_dist = 5.0;
	param.x_offset = 0.0;
	param.y_offset = 0.0;
	param.yaw = 0;
	param.min_bb_area_in_pixels =1000;
	register_camera(0,&param);

	/********************************/
	/******** Register samples ******/
	/********************************/

	/*
	 * White
	 */
	std::vector<double>L{200,55,0.6};			 				//{Origin,Deviation,Weight}
	std::vector<double>a{128,20,0.1};			 				//{Origin,Deviation,Weight}
	std::vector<double>b{128,30,0.3};			 				//{Origin,Deviation,Weight}
	std::vector<double>width{0.0685,0.01, 0.06}; 	//{width,min,max}
	std::vector<double>depth{0.0635,0.01, 0.03}; 	//{height,min,max}
	register_sample(WHITE,L,a,b,width,depth);

	// Todo: Add purple rock


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
