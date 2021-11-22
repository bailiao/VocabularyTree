#include<opencv2/opencv.hpp>
#include<iostream>
#include<math.h>
#include<opencv2/xfeatures2d.hpp>

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;

Mat src;
int main(int argc, char** argv)
{
	src = imread("C:\\Users\\HCK\\Pictures\\Saved Pictures\\104300.jpg", IMREAD_GRAYSCALE);
	if (!src.data)
	{
		cout << "图片未找到" << endl;
		return -1;
	}
	imshow("input title", src);
	int numfeature = 400;
	Ptr<SIFT>detector = SIFT::create(numfeature);//与SURF一样，剩余的取默认值
	vector<KeyPoint>keypoints;
	detector->detect(src, keypoints, Mat());
	Mat resultImg;
	drawKeypoints(src, keypoints, resultImg, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	imshow("SIFT keypoint", resultImg);
	waitKey(0);
	return 0;



}
