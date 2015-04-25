#include "detection.h"

#define MAX_SAMPLES (256)

// Globals
RNG rng(12345);
int kernel_size = 3;

std::vector<SAMPLE> samples;

cv::Mat Input_image;

void register_sample(unsigned int Id, std::vector<int>hsv_min, std::vector<int>hsv_max, double min_width, double max_width, double min_height, double max_height)
{
	SAMPLE new_sample;
	new_sample.Id = Id;
	new_sample.HSV_MIN = hsv_min;
	new_sample.HSV_MAX = hsv_max;
	new_sample.min_width = min_width;
	new_sample.max_width = max_width;
	new_sample.min_height = min_height;
	new_sample.max_height = max_height;

	samples.push_back(new_sample);
	std::cout<<"added new sample Id = " << Id << std::endl;
}

int getSampleSize()
{
	return samples.size();
}

bool process_image(cv::Mat image_hsv, int index)
{    
    cv::Mat temp_image1, temp_image2 ;
    std::vector<vector<Point> > contours;
    std::vector<Vec4i> hierarchy;

    // Mark all pixels in the required color range high and other pixels low.
    inRange(image_hsv,samples[index].HSV_MIN,samples[index].HSV_MAX,temp_image1);

    // Gives the kernel shape for erosion.
    // To do: Experiment with different kernels
    Mat element = getStructuringElement( MORPH_RECT, Size(2*kernel_size+1,2*kernel_size+1), Point(0,0));


    // Erode the image to get rid of trace elements with similar color to the required sample
    erode(temp_image1,temp_image2,element);

    // Find contours in the thresholded image to determine shapes
    findContours(temp_image2,contours,hierarchy,CV_RETR_TREE,CV_CHAIN_APPROX_SIMPLE,Point(0,0));

    std::vector<vector<Point> > contours_poly( contours.size() );
    std::vector<Rect> boundRect( contours.size() );
    for( int i = 0; i < contours.size(); i++ )
     {
        approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
        boundRect[i] = boundingRect( Mat(contours_poly[i]) );
     }
   
    // Print the number of samples found
     std::cout << "Number of samples found: "<< contours.size()<< std::endl;

    // Draw all the contours found in the previous step
    Mat drawing = Mat::zeros( temp_image2.size(), CV_8UC3 );
    for( int i = 0; i< contours.size(); i++ )
     {
       Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
       drawContours( drawing, contours_poly, i, color, 2, 8, hierarchy, 0, Point() );

       // Draw a bounding box
       rectangle( Input_image, boundRect[i].tl(), boundRect[i].br(), (0,0,255), 2, 8, 0 );
     }

    return true;
}

void display_image(cv::Mat orig)
{
   // cv::namedWindow("Input Image: ", WINDOW_AUTOSIZE);
   // cv::imshow("Input Image: ",orig);
}

DETECTED_SAMPLE find_objects(const Mat *image)
{
	cv::Mat hsv_image;
	DETECTED_SAMPLE test_sample;
	// Test sample values for integration
	test_sample.id = 0;
	test_sample.x = 0;
	test_sample.y = 0;

	Input_image = *image;

	if(! Input_image.data) {
		std::cout << "could not read image"<< std::endl;
	}

	// Convert the color space to HSV
	cv::cvtColor(Input_image,hsv_image,CV_BGR2HSV);

	// Get the iterator for the vector color space and loop through all sample color's
	//for(it = ObjectTypes.begin(); it != ObjectTypes.end(); it++)
	for(int index = 0; index < samples.size(); ++index)
	{
		if(!process_image(hsv_image, index))
		{
			std::cout << "Processing images failed" << std::endl;
		}
	}
	return test_sample;

}

/*
cv::Mat find_objects(const Mat * image)
{
    cv::Mat hsv_image;
    Input_image = *image;
    
    if(! Input_image.data) {
        std::cout << "could not read image"<< std::endl;
    }
    
    // Convert the color space to HSV
    cv::cvtColor(Input_image,hsv_image,CV_BGR2HSV);

    // Get the iterator for the vector color space and loop through all sample color's
    //for(it = ObjectTypes.begin(); it != ObjectTypes.end(); it++)
    for(int index = 0; index < samples.size(); ++index)
    {
        if(!process_image(hsv_image, index))
        {
            std::cout << "Processing images failed" << std::endl;
        }  
    }

    // Display images
#ifdef DRAW
    display_image(Input_image);
    //cv::waitKey(0);
#endif
    return Input_image;
}
*/
