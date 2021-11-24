// 使用Flann进行特征点匹配.cpp : 定义控制台应用程序的入口点。

//

#include <opencv2/opencv.hpp>

#include <highgui/highgui.hpp>

#include <features2d.hpp>

#include<opencv2/xfeatures2d.hpp>

#include <xfeatures2d/nonfree.hpp>

#include <vector>

using namespace cv;

using namespace std;

int main(int argc, char** argv[])

{

    Mat input1 = imread("C:\\Users\\HCK\\Pictures\\Saved Pictures\\104300.jpg", 1);

    Mat input2 = imread("C:\\Users\\HCK\\Pictures\\Saved Pictures\\104301.jpg", 1);

    if (input1.empty() || input2.empty())

    {

        cout << "不能正常加载图片" << endl;

        system("pause");

        return -1;

    }

    /************************************************************************/

    /*下面进行提取特征点*/

    /************************************************************************/

    //SiftFeatureDetector *feature = SIFT::create(400);
    Ptr<SIFT>feature = SIFT::create();

    vector<KeyPoint> kerpoints1;

    feature->detect(input1, kerpoints1, Mat());

    Mat output1;

    drawKeypoints(input1, kerpoints1, output1);

    vector<KeyPoint> kerpoints2;

    feature->detect(input2, kerpoints2, Mat());

    Mat output2;

    drawKeypoints(input2, kerpoints2, output2);

    imshow("提取特征点后的box.png", output1);

    imshow("提取特征点后的box_in_scene.png", output2);

    imwrite("提取特征点后的box.png", output1);

    imwrite("提取特征点后的box_in_scene.png", output2);

    cout << "box提取的特征点数为:" << kerpoints1.size() << endl;

    cout << "box_in_scene的特征点数为:" << kerpoints2.size() << endl;

    /************************************************************************/

    /* 下面进行特征向量提取 */

    /************************************************************************/

    Ptr<SiftDescriptorExtractor> descript = SiftDescriptorExtractor::create();

    Mat description1 = Mat();

    descript->compute(input1, kerpoints1, description1);

    Mat description2 = Mat();

    descript->compute(input2, kerpoints2, description2);
    cout << description1.rows << endl;
    cout << description2.cols << endl;

    /************************************************************************/

    /* 下面进行特征向量临近匹配 */

    /************************************************************************/

    vector<DMatch> matches;

    FlannBasedMatcher matcher;

    Mat image_match;

    matcher.match(description1, description2, matches);

    /************************************************************************/

    /* 下面计算向量距离的最大值与最小值 */

    /************************************************************************/

    double max_dist = 0, min_dist = 100;

    for (int i = 0; i < description1.rows; i++)

    {

        if (matches.at(i).distance > max_dist)

        {

            max_dist = matches[i].distance;

        }

        if (matches[i].distance < min_dist)

        {

            min_dist = matches[i].distance;

        }

    }

    cout << "最小距离为" << min_dist << endl;

    cout << "最大距离为" << max_dist << endl;

    /************************************************************************/

    /* 得到距离小于而V诶最小距离的匹配 */

    /************************************************************************/

    vector<DMatch> good_matches;

    for (int i = 0; i < matches.size(); i++)

    {

        if (matches[i].distance < 2 * min_dist)

        {

            good_matches.push_back(matches[i]);

            cout << "第一个图中的" << matches[i].queryIdx << "匹配了第二个图中的" << matches[i].trainIdx << endl;

        }

    }

    drawMatches(input1, kerpoints1, input2, kerpoints2, good_matches, image_match);

    imshow("匹配后的图片", image_match);

    imwrite("匹配后的图片.png", image_match);

    cout << "匹配的特征点数为:" << good_matches.size() << endl;
    
    waitKey(0);

    return 0;

}