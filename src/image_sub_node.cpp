#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include "detection.h"

cv_bridge::CvImagePtr cv_ptr;
image_transport::Publisher pub;

void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
	try {
		  cv_ptr = cv_bridge::toCvCopy(msg,"bgr8");
		  const cv::Mat * imagePtr = &(cv_ptr->image);

		  cv::Mat out_image;
		  out_image = find_objects(imagePtr);
		  sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", out_image).toImageMsg();
		  pub.publish(msg);
	  } catch (cv_bridge::Exception& e) {
	    ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
	  }

	  std::cout << "." << std::endl;
}

int main(int argc, char **argv)
{
  register_sample();
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
