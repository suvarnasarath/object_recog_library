#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <vector>

using namespace cv;

/*
 * Definition of output sample
 * @ id : Id of the sample
 * @ x,y: position in world coordinates.
 */
typedef struct
{
	unsigned int id;
	double x;
	double y;
}DETECTED_SAMPLE;

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
}SAMPLE;

cv::Mat find_objects(const Mat * imgPtr);
void register_sample(unsigned int Id,
					 std::vector<int>hsv_min, std::vector<int>hsv_max,
					 double min_width, double max_width, double min_height, double max_height);
int getSampleSize();
