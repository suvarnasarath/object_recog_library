#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>

void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
/*
  try
  {
    cv::imshow("view", cv_bridge::toCvShare(msg, "bgr8")->image);
    cv::WaitKey(30);
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
  }
*/
  std::cout << "." << std::endl;
}

int main(int argc, char **argv)
{
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
  ros::spin();
  //cv::destroyWindow("view");
}
