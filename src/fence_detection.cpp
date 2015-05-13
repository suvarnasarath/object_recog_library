#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <vector>

cv::RNG rng(12345);


int main(int argc, char **argv)
{
	cv::namedWindow("video",CV_WINDOW_AUTOSIZE); 
	cv::Mat frame,src_hsv;
	cv::Vec3d CurrentPixel;
	cv::Vec3d Origin(10,127,200); // HSV mid values
	std::vector<cv::Mat> hsv_planes(3);


	frame = cv::imread(argv[1],CV_LOAD_IMAGE_COLOR);
	if(! frame.data) {
		std::cout << "could not read image"<< std::endl;
		return -1;
	}

	cv::cvtColor(frame,src_hsv,CV_BGR2HSV);
	cv::split( src_hsv, hsv_planes );
	
	cv::Mat H = hsv_planes[0]; 
	cv::Mat S = hsv_planes[1]; 
	cv::Mat V = hsv_planes[2]; 

    cv::Mat response = cv::Mat::zeros(H.rows,H.cols,CV_32FC1);
    int hue_ref = Origin[0]*255/179;
    int32_t sat_ref = Origin[1];
    int32_t val_ref = Origin[2];

    const int32_t MAX_SAT_DEV = 120;
    const int32_t MAX_VAL_DEV = 120;
    const int32_t MAX_HUE_DEV = 30;

    const float HUE_MULT_FACTOR = 1.0/static_cast<double>(MAX_HUE_DEV);
    const float SAT_MULT_FACTOR = 1.0/static_cast<double>(MAX_SAT_DEV);
    const float VAL_MULT_FACTOR = 1.0/static_cast<double>(MAX_VAL_DEV);



	for(int rows = 0 ;rows < frame.rows; rows++)
	{
		for(int cols =0; cols < frame.cols; cols++)
		{
			uint8_t hue_image = static_cast<uint32_t>(H.at<uint8_t>(rows,cols))*255/179;
			uint8_t hue_diff = hue_image - hue_ref;

			int32_t saturation = static_cast<int32_t>(S.at<uint8_t>(rows,cols));
			int32_t value = static_cast<int32_t>(V.at<uint8_t>(rows,cols));

			int32_t saturation_deviation = saturation - sat_ref;

			int32_t value_deviation = value - val_ref;

			saturation_deviation = std::max(-MAX_SAT_DEV,std::min(MAX_SAT_DEV,saturation_deviation));
			saturation_deviation = MAX_SAT_DEV - std::abs(saturation_deviation);
			float sat_factor = SAT_MULT_FACTOR * saturation_deviation;


			value_deviation = std::max(-MAX_VAL_DEV,std::min(MAX_VAL_DEV,value_deviation));
			value_deviation = MAX_VAL_DEV - std::abs(value_deviation);
			float val_factor = VAL_MULT_FACTOR * value_deviation;

			int32_t hue_deviation = static_cast<int8_t>(hue_diff);
			hue_deviation = std::max(-MAX_HUE_DEV,std::min(MAX_HUE_DEV,hue_deviation));
			hue_deviation = MAX_HUE_DEV - std::abs(hue_deviation);
			float hue_factor = HUE_MULT_FACTOR * hue_deviation;


			//float hue_factor = static_cast<float>(1.0/128.0)*(128 - std::abs(static_cast<int8_t>(hue_diff)));

			const float A1 = 0.65;
			const float A2 = 0.35;
			const float A3 = 0.0;

			float response_value = hue_factor * A1 + sat_factor * A2 + val_factor * A3;
			//uint8_t image_value = response * 255;
			//image_value = image_value > 200 ? 255 : 0;
			response.at<float>(rows,cols) = response_value;
		}
	}		

	cv::imwrite("hue_out.png",response);
	cv::imshow("video",response);
	cv::waitKey(0);   
	return 0;
}
