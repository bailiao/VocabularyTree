#include "VocabularyTree.h"
#include <opencv2/opencv.hpp>

#include <highgui/highgui.hpp>

#include <features2d.hpp>

#include<opencv2/xfeatures2d.hpp>

#include <xfeatures2d/nonfree.hpp>

#include <string>

#include <vector>
using namespace std;
using namespace cv;

//==========================functions in class imageRetriver========================


void imageRetriver::addFeature2DataBase(vector<vector<double>> tfidfVector) {
	for (int i = 0; i < nImages; i++) {
		imageDatabase.insert(make_pair(tfidfVector[i], imagePath[i]));		
	}
}


void Print(vocabularyTreeNode* Root) {
	printf("Root = %p\n", Root);
	for (int i = 0; i < 10; i++) {
		printf("Child %d\t = %p\n", i, Root->children[i]);
		for (int j = 0; j < DEFAULT_BRANCH; j++) {
			printf("\tChild %d\tof Child%d\t = %p\n", i, j, Root->children[i]->children[j]);
		}
	}
}

void imageRetriver::buildDataBase(char* directoryPath) {
	printf("%s\n", directoryPath);
	vector<string> databaseImagePath;
	string postfix = ".jpg";
<<<<<<< Updated upstream
	
	DirectoryList("E:\\Donald Duck\\MyClass5\\SRTP\\DemoBase", databaseImagePath,postfix);
	this->nImages = databaseImagePath.size();
	Features_PerImage = new int[nImages];




=======

	DirectoryList("D:\\opencv\\VocabularyTree\\Demo3\\test", databaseImagePath,postfix);
>>>>>>> Stashed changes
	double** trainFeatures = NULL;
	int nFeatures = getTrainFeatures(trainFeatures, databaseImagePath,this->featureLength);							// 只有这里需要替换掉SIFT，同时还要注意图片查询时也需要特征提取			
	cout << "-------------------------------------------------------------" << endl;
	for (int i = 0; i < nImages; i++) {
		cout << Features_PerImage[i] << endl;
	}
	cout << "-------------------------------------------------------------" << endl;
	//for (int i = 0; i < 10; i++) {
	//	for (int j = 0; j < 128; j++) {
	//		cout << trainFeatures[i][j] << "   ";
	//	}
	//	cout << endl;
	//}
	//cout << 1111111;
	
	
	this->tree = new vocabularyTree;
	tree->buildTree(trainFeatures, nFeatures, tree->nBranch, tree->depth, featureLength);


	cout << "build finished" << endl;
	Print(this->tree->root);
	vector<vector<double>> tfidfVector = getTFIDFVector(trainFeatures, nFeatures);
	addFeature2DataBase(tfidfVector);
}

vector<string> imageRetriver::queryImage(const char* imagePath) {
	vector<string> ans;

	string image_str = imagePath;
	Mat input = imread(image_str,1);
	if (input.empty()) {
		cout << "wrong image path" << endl;
		system("pause");
		exit(-1);
	}
	Ptr<SiftDescriptorExtractor> descript = SiftDescriptorExtractor::create();

	Mat description = Mat();
	vector<KeyPoint> kerpoints;
	descript->compute(input, kerpoints, description);
	int nFeatures = description.rows;				// the number of features of this picture 
	double** queryFeat = new double* [nFeatures];

	for (int j = 0; j < nFeatures; j++) {
		queryFeat[j] = new double[description.cols];
		uchar* data = description.ptr<uchar>(j);		// ptr to jth row
		for (int k = 0; k < description.cols; k++) {	// each row has 128 cols
			queryFeat[j][k] = data[k];
		}
	}





	vector<double> tfidfVector = getOneTFIDFVector(queryFeat, nFeatures, 0);

	multimap<double, string> candidates;
	map<vector<double>, string>::iterator iter; 
	double maxDistance = 1e20;
	for(iter = imageDatabase.begin(); iter != imageDatabase.end(); iter++) {
		vector<double> cur = iter->first;
		double distance = vector_sqr_distance(cur, tfidfVector);
		candidates.insert(make_pair(distance, iter->second));
	}

	int count = 0;
	multimap<double, string>::iterator iter1;
	for(iter1 = candidates.begin(); iter1 != candidates.end(); iter1++) {
		ans.push_back(iter1->second);
		count++;
		if(count == ANSNUM)
			break;
	}

	return ans;
}

int imageRetriver::getTrainFeatures(double** &trainFeatures, vector<string> imagePaths, int& featureLength) {		//这个函数内部需要修改，替换掉原有的特征提取函数，仍然返回特征的总数
																								// 根据每一个path 创建MAT，提取特征向量
																								// 把对应的MAT中的结果放到double**内部
	int nImages = imagePaths.size();
	cout << "Total images number: " << nImages << endl;
	trainFeatures = new double*[nImages * MAXFEATNUM];
	Features_PerImage = new int[nImages];
	int featCount = 0;

	for(int i = 0; i < nImages; i++) {
		cout << imagePaths[i] << endl;
		Mat input = imread(imagePaths[i],1);
		if (input.empty()) {
			cout << "wrong image path" << endl;
			system("pause");
			return -1;
		}

		Ptr<SiftDescriptorExtractor> descript = SiftDescriptorExtractor::create(400);
		Ptr<SIFT>feature = SIFT::create();
		Mat description = Mat();

		vector<KeyPoint> kerpoints;

		feature->detect(input, kerpoints, Mat());

		descript->compute(input, kerpoints, description);
		int n = description.rows;				// the number of features of this picture 
		for(int j = 0; j < n; j++) {
			trainFeatures[featCount] = new double[description.cols];
			uchar* data = description.ptr<uchar>(j);		// ptr to jth row
			for (int k = 0; k < description.cols; k++) {	// each row has 128 cols
				trainFeatures[featCount][k] = data[k];
			}
			featureLength = description.cols;
			featCount++;
		}
		Features_PerImage[i] = n;
	}

	//for (int i = 0; i < 10; i++) {
	//	for (int j = 0; j < 128; j++) {
	//		cout << trainFeatures[i][j] << "   ";
	//	}
	//	cout << endl;
	//}
	//cout << 1111111;

	cout << "Total features number: " << featCount << endl;
	return featCount;				
}

void imageRetriver::HKAdd(double* feature, int depth, vocabularyTreeNode* cur) {  
	if(depth == tree->depth)	return;
	if (cur == NULL)			return;
	if(cur->add)		
		cur->tf++;
	int minIndex = -1;
	double minDis = 10000000;
	if (cur->nFeatures <= cur->nBranch);
	else {
		for (int i = 0; i < cur->nBranch; i++) {
			double curDis = sqr_distance(feature, (cur->children[i])->feature, featureLength);
			if (curDis < minDis) {
				curDis = minDis;
				minIndex = i;
			}
		}
		HKAdd(feature, depth + 1, cur->children[minIndex]);
	}
	
}

void imageRetriver::HKDiv(vocabularyTreeNode* curNode, int curDepth) {		
	if(curDepth == tree->depth)	return;
	if (curNode == NULL)		return;
	// curNode->idf = 1.0 * nImages / curNode->tf;				
	curNode->idf = log((1.0 * nImages / curNode->tf));				
	for(int i = 0; i < curNode->nBranch; i++) {
		HKDiv(curNode->children[i], curDepth + 1); 
	}
}

void imageRetriver::calIDF(double** features) {			// read N pictures
	int featureCount = 0;
	tree->clearTF(tree->root, 0);
	for(int i = 0; i < nImages; i++) {
		
		for(int j = 0; j < Features_PerImage[i]; j++) {
			HKAdd(features[featureCount], 0, tree->root);   //add the number of images at least one descriptor path through for each node
			featureCount++;
		}
	}
	HKDiv(tree->root, 0);    //cal N / Ni, where N is the number of total images and Ni is tf
}


vector<double> imageRetriver::getOneTFIDFVector(double** features, int featNums, int nStart) {
	tree->clearTF(tree->root, 0);
	for (int i = 0; i < featNums; i++)
		HKAdd(features[nStart + featNums], 0, tree->root);		
	vector<double> oneImgTFIDF;
	tree->getTFIDF(oneImgTFIDF, tree->root, 0);
	return oneImgTFIDF;
}

vector<vector<double>> imageRetriver::getTFIDFVector(double** features, int nImages) { 	
	//calIDF(features);        //calculate idf for each node in the tree 
	vector<vector<double>> tfidfVector;
	int featureCount = 0;
	//cout << "------------------" << endl;
	//for (int i = 0; i < nImages; i++) {
	//	cout << Features_PerImage[i] << endl;
	//}
	for(int i = 0; i < nImages; i++) {
		vector<double> oneImgTFIDF = getOneTFIDFVector(features, Features_PerImage[i], featureCount);
		tfidfVector.push_back(oneImgTFIDF);
		featureCount += Features_PerImage[i];
	}

	int valid_feature_count = tfidfVector[0].size();
	double* weight = new double[valid_feature_count];
	for (int i = 0; i < valid_feature_count; i++) {
		weight[i] = 0;
	}
	for (int i = 0; i < nImages; i++) {
		for (int j = 0; j < valid_feature_count; j++) {
			weight[j] += (tfidfVector[i][j] > 0);
		}
	}
	for (int i = 0; i < valid_feature_count; i++) {
		weight[i] = log(nImages / weight[i]);
	}
	for (int i = 0; i < nImages; i++) {
		for (int j = 0; j < valid_feature_count; j++) {
			tfidfVector[i][j] *= weight[j];
		}
	}



	return tfidfVector;
}



