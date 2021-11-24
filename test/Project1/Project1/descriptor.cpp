// ʹ��Flann����������ƥ��.cpp : �������̨Ӧ�ó������ڵ㡣

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

        cout << "������������ͼƬ" << endl;

        system("pause");

        return -1;

    }

    /************************************************************************/

    /*���������ȡ������*/

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

    imshow("��ȡ��������box.png", output1);

    imshow("��ȡ��������box_in_scene.png", output2);

    imwrite("��ȡ��������box.png", output1);

    imwrite("��ȡ��������box_in_scene.png", output2);

    cout << "box��ȡ����������Ϊ:" << kerpoints1.size() << endl;

    cout << "box_in_scene����������Ϊ:" << kerpoints2.size() << endl;

    /************************************************************************/

    /* �����������������ȡ */

    /************************************************************************/

    Ptr<SiftDescriptorExtractor> descript = SiftDescriptorExtractor::create();

    Mat description1 = Mat();

    descript->compute(input1, kerpoints1, description1);

    Mat description2 = Mat();

    descript->compute(input2, kerpoints2, description2);
    cout << description1.rows << endl;
    cout << description2.cols << endl;

    /************************************************************************/

    /* ����������������ٽ�ƥ�� */

    /************************************************************************/

    vector<DMatch> matches;

    FlannBasedMatcher matcher;

    Mat image_match;

    matcher.match(description1, description2, matches);

    /************************************************************************/

    /* �������������������ֵ����Сֵ */

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

    cout << "��С����Ϊ" << min_dist << endl;

    cout << "������Ϊ" << max_dist << endl;

    /************************************************************************/

    /* �õ�����С�ڶ�V����С�����ƥ�� */

    /************************************************************************/

    vector<DMatch> good_matches;

    for (int i = 0; i < matches.size(); i++)

    {

        if (matches[i].distance < 2 * min_dist)

        {

            good_matches.push_back(matches[i]);

            cout << "��һ��ͼ�е�" << matches[i].queryIdx << "ƥ���˵ڶ���ͼ�е�" << matches[i].trainIdx << endl;

        }

    }

    drawMatches(input1, kerpoints1, input2, kerpoints2, good_matches, image_match);

    imshow("ƥ����ͼƬ", image_match);

    imwrite("ƥ����ͼƬ.png", image_match);

    cout << "ƥ�����������Ϊ:" << good_matches.size() << endl;
    
    waitKey(0);

    return 0;

}