#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main()
{
    //OpenCV°æ±¾ºÅ
    cout << "OpenCV_Version: " << CV_VERSION << endl;

    //¶ÁÈ¡Í¼Æ¬
    Mat img = imread("C:/Users/HCK/Pictures/Saved Pictures/Water.jpeg");

    imshow("picture", img);
    waitKey(0);
    return 0;
}
