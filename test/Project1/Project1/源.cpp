#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main()
{
    //OpenCV�汾��
    cout << "OpenCV_Version: " << CV_VERSION << endl;

    //��ȡͼƬ
    Mat img = imread("C:/Users/HCK/Pictures/Saved Pictures/Water.jpeg");

    imshow("picture", img);
    waitKey(0);
    return 0;
}
