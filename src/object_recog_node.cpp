#include <iostream>
#include <string>
#include "ros/ros.h"
#include "object_recog/Object.h"
#include <cv_bridge/cv_bridge.h>
#include <std_msgs/Int8.h>
#include <image_transport/image_transport.h>
#include "detection.h"

//publish #of objects
//ros::Publisher pub;
image_transport::Publisher pub; 


// List of names of the objects as a reference
const std::vector<std::string> ObjectTypes = {
		"base",
		"fence",
		"main_sample",
		"pipe",
		"other"
};

// Where a pointer to the latest image is stored
cv_bridge::CvImagePtr cv_ptr;


void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
  std::cout << "." << std::endl;
  try {
	  cv_ptr = cv_bridge::toCvCopy(msg,"bgr8");
	  const cv::Mat * imagePtr = &(cv_ptr->image);

	  /*
	  std_msgs::Int8 objects;
	  objects.data = find_objects(imagePtr);
	  num_Objects.publish(objects);
	  */

	  cv::Mat out_image;
	  out_image = find_objects(imagePtr);
	  sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", out_image).toImageMsg();
	  pub.publish(msg);

  } catch (cv_bridge::Exception& e) {
    ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
  }
}

int main(int argc, char ** argv)
{
	std::cout << "Object recognition node" << std::endl;

	ros::init(argc, argv, "object_recognition_node");
	ros::NodeHandle n;
	ros::NodeHandle np("~");

	std::string imageTopic;

	if(!np.getParam("image",imageTopic)) {
		ROS_ERROR("No image topic specified");
		return -1;
	}

	image_transport::ImageTransport it(n);
	image_transport::Subscriber sub = it.subscribe(imageTopic, 1, imageCallback);

	// topics
  	//num_Objects = n.advertise<std_msgs::Int8>("chatter", 1000);
  	pub = it.advertise("chatter",1);
	
	ros::spin();
	return 0;
}




#if 0
void displayRequested(const object_recog::Object::Request & req)
{
	std::string info_string = "Requested: ";

	for(const auto & o : req.searchID) {
		try {
			info_string = info_string + ObjectTypes.at(o) + ";";
		} catch(const std::out_of_range & r) {
			info_string += "unknown;";
		}

	}
	ROS_INFO("%s",info_string.c_str());

}


// Find objects in the image using color based blob detection
int findObjects(const cv::Mat * imgPtr, std::vector<int> & objectType, std::vector<cv::Rect> boundingBoxes)
{
    if(imgPtr == NULL) {
        std::cout << "Got bad image in findObjects"<< std::endl;
        return false;
    } else {
    	// Call detection code with image ptr
    	return find_objects(imgPtr);       
    }

}

bool filterDetections(const std::vector<int> & valid, std::vector<int> & objectType, std::vector<cv::Rect> boundingBoxes)
{
	//TODO: filter only valid detections
	return true;

}

void execute_recognition(object_recog::Object::Request & req,
						 object_recog::Object::Response & res)
{
	displayRequested(req);

	const cv::Mat * imagePtr = &(cv_ptr->image);

	std::vector<int> objects;
	std::vector<cv::Rect> boundingBoxes;

	std::msgs int objects;
	objects = find_objects(imgPtr);
#if 0
	if(!findObjects(imagePtr,objects,boundingBoxes)) {
        return false;
	}

	if(!filterDetections(req.searchID, objects, boundingBoxes)) {
        std::cout<< "No valid samples"<< std::endl;
	}
#endif 
	num_Objects.publish(objects);

	// TODO: convert the detections into xyz points in the map reference frame
	// will need tf and camera calibration information
}
#endif