#include "detection.h"

#define MAX_SAMPLES (256)

// Globals
RNG rng(12345);
int kernel_size = 3;

typedef struct
{
	unsigned int id;
	double x;
	double y;
}DETECTED_SAMPLE;

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

std::vector<SAMPLE> sample;

cv::Mat Input_image;

// Number of sample color's
#define NUM_SAMPLES (5)  // number of samples to detect
#define COLOR_SPACE (6)  // Min and Max HSV valuse

static const int samplearray[COLOR_SPACE * NUM_SAMPLES]={165,50,50,175,255,255   // Red hockey puck
													  ,20,50,50,30,255,255     // Yellow PVC pipe
													  ,5,50,50,15,255,255      // Orange PVC pipe
													  ,90,60,60,110,255,255    // White hooked sample
													  ,0,50,50,5,255,255       // Pink Tennis Ball
                                    /******** Add more color spaces here ***********/
                                        };

std::vector<int> ObjectTypes ( samplearray, samplearray + sizeof(samplearray) / sizeof(samplearray[0]) );


bool register_sample(unsigned int Id, std::vector<int>hsv_min, std::vector<int>hsv_max, double min_width, double max_width, double min_height, double max_height)
{
	SAMPLE new_sample;
	new_sample.Id = Id;
	new_sample.HSV_MIN = hsv_min;
	new_sample.HSV_MAX = hsv_max;
	new_sample.min_width = min_width;
	new_sample.max_width = max_width;
	new_sample.min_height = min_height;
	new_sample.max_height = max_height;

	sample.push_back(new_sample);
	std::cout<<"added new sample Id = " << Id << std::endl;
}

bool process_image(cv::Mat image_hsv, int index)
{    
    cv::Mat temp_image1, temp_image2 ;
    std::vector<vector<Point> > contours;
    std::vector<Vec4i> hierarchy;
    cv::Scalar min,max;

    min.val[0] = ObjectTypes[index++];
    min.val[1] = ObjectTypes[index++];
    min.val[2] = ObjectTypes[index++];

    max.val[0] = ObjectTypes[index++];
    max.val[1] = ObjectTypes[index++];
    max.val[2] = ObjectTypes[index++];

    // Mark all pixels in the required color range high and other pixels low.
    inRange(image_hsv,min,max,temp_image1);

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
    for(int index = 0; index < ObjectTypes.size(); index += COLOR_SPACE)
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
