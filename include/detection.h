#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <vector>

using namespace cv;

cv::Mat find_objects(const Mat * imgPtr);
void register_sample(unsigned int Id,
					 std::vector<int>hsv_min, std::vector<int>hsv_max,
					 double min_width, double max_width, double min_height, double max_height);
int getSampleSize();
