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
	//for (int i = 0; i < nImages; i++) {
	//	imageDatabase.insert(make_pair(tfidfVector[i], imagePath[i]));		
	//}
	cout << "not use" << endl;
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

void Print(vocabularyTreeNode* Root, int depth) {
	printf("----------------------------------\n");
	cout << "depth = " << depth << endl;
	cout << "count = " << Root->nFeatures << endl;
	//cout << "feature = ";
	//for (int i = 0; i < 10; i++) {
	//	printf("%.3f\t", Root->feature[i]);
	//}
	//cout << endl;
	for (int i = 0; i < Root->nBranch; i++) {
		if (Root->children[i] == NULL);// printf("child %d\t = NULL\n", i);
		else {
			printf("child %d\n", i);
			Print(Root->children[i], depth + 1);
		}
	}
}
void DFS(vocabularyTreeNode* cur, vector<int> &tmp) {
	if (cur == NULL) return;
	tmp.push_back(cur->nFeatures);
	for (int i = 0; i < cur->nBranch; i++) {
		DFS(cur->children[i],tmp);
	}
}

void imageRetriver::buildDataBase(char* directoryPath) {
	printf("%s\n", directoryPath);
	string postfix = ".jpg";
	
	DirectoryList("E:\\Donald Duck\\MyClass5\\SRTP\\DemoBase", databaseImagePath,postfix);
	this->nImages = databaseImagePath.size();
	Features_PerImage = new int[nImages];




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
	//vector<int> tmp;
	//DFS(this->tree->root,tmp);
	//for (int i = 0; i < tmp.size(); i++) {
	//	printf("i = %d\t\tnFeature = %d\t\t\n", i, tmp[i]);
	//}
	//Print(this->tree->root,0);
	//Print(this->tree->root, 0);

	image_vector_cluster = getTFIDFVector(trainFeatures, this->nImages);
	//addFeature2DataBase(tfidfVector);
}

struct tmpcmp_set{
	int seq;
	double dis;
};

int tmpcmp(const void* a, const void* b) {
	double dis_a = ((struct tmpcmp_set*)a)->dis;
	double dis_b = ((struct tmpcmp_set*)b)->dis;
	return dis_a > dis_b ? 1 : -1;
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
	struct tmpcmp_set* cluster = new struct tmpcmp_set[nImages];
	for (int i = 0; i < nImages; i++) {
		cluster[i].seq = i;
		cluster[i].dis = vector_sqr_distance(image_vector_cluster[i], tfidfVector);
	}
	qsort(cluster, nImages, sizeof(struct tmpcmp_set), tmpcmp);
	for (int i = 0; i < nImages; i++) {
		cout << i << "  " << cluster[i].seq << "  " << cluster[i].dis << endl;
	}
	////multimap<double, string> candidates;
	////map<vector<double>, string>::iterator iter; 
	//double maxDistance = 100000000;
	//for(iter = imageDatabase.begin(); iter != imageDatabase.end(); iter++) {
	//	vector<double> cur = iter->first;
	//	double distance = vector_sqr_distance(cur, tfidfVector);
	//	candidates.insert(make_pair(distance, iter->second));
	//}

	//int count = 0;
	//multimap<double, string>::iterator iter1;
	//for(iter1 = candidates.begin(); iter1 != candidates.end(); iter1++) {
	//	ans.push_back(iter1->second);
	//	count++;
	//	if(count == ANSNUM)
	//		break;
	//}
	for (int i = 0; i < 5; i++) {
		ans.push_back(databaseImagePath[cluster[i].seq]);
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
		Ptr<SIFT>feature = SIFT::create(400);
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
	//if(cur->add)		
	cur->tf++;
	int minIndex = -1;
	double minDis = 10000000;
	if (cur->nFeatures <= cur->nBranch);
	else {
		for (int i = 0; i < cur->nBranch; i++) {
			if (cur->children[i] == NULL) continue;
			double curDis = sqr_distance(feature, (cur->children[i])->feature, featureLength);
			if (curDis < minDis) {
				//curDis = minDis;
				minDis = curDis;
				minIndex = i;
			}
		}
		HKAdd(feature, depth + 1, cur->children[minIndex]);
	}
	
}

void imageRetriver::HKDiv(vocabularyTreeNode* curNode, int curDepth) {			// not used
	if(curDepth == tree->depth)	return;
	if (curNode == NULL)		return;
	// curNode->idf = 1.0 * nImages / curNode->tf;				
	curNode->idf = log((1.0 * nImages / curNode->tf));				
	for(int i = 0; i < curNode->nBranch; i++) {
		HKDiv(curNode->children[i], curDepth + 1); 
	}
}

void imageRetriver::calIDF(double** features) {			// read N pictures		// not used
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
	for (int i = 0; i < featNums; i++) {
		HKAdd(features[nStart + i], 0, tree->root);
		//cout << "feature" << 
	}
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
		if (i == 9) {
			cout << "stop" << endl;
		}
		vector<double> oneImgTFIDF = getOneTFIDFVector(features, Features_PerImage[i], featureCount);
		tfidfVector.push_back(oneImgTFIDF);
		featureCount += Features_PerImage[i];	// accumulative count before
	}

	int valid_feature_count = tfidfVector[0].size();


	//for (int i = 0; i < nImages; i++) {
	//	cout << "-----------------------------\n";
	//	cout << "Image: " << i << endl;
	//	for (int j = 0; j < valid_feature_count; j++) {
	//		printf("%d\t%.3f\n", j, tfidfVector[i][j]);
	//	}
	//}





	double* weight = new double[valid_feature_count];
	for (int i = 0; i < valid_feature_count; i++) {
		weight[i] = 0;
	}
	for (int i = 0; i < nImages; i++) {
		for (int j = 0; j < valid_feature_count; j++) {
			weight[j] += (tfidfVector[i][j] > 0);
		}
	}
	//for (int i = 0; i < valid_feature_count; i++) {
	//	printf("%d\t%.3f\n", i, weight[i]);
	//}
	for (int i = 0; i < valid_feature_count; i++) {
		if(weight[i] != 0.0) weight[i] = log(nImages / weight[i]);
	}

	//for (int i = 0; i < valid_feature_count; i++) {
	//	printf("%d\t%.3f\n", i, weight[i]);
	//}
	for (int i = 0; i < nImages; i++) {
		for (int j = 0; j < valid_feature_count; j++) {
			tfidfVector[i][j] *= weight[j];
		}
	}

	//cout << "------------------------------------------------------------\n\n";
	//for (int i = 0; i < nImages; i++) {
	//	cout << "-----------------------------\n";
	//	cout << "Image: " << i << endl;
	//	for (int j = 0; j < valid_feature_count; j++) {
	//		printf("%d\t%.3f\n", j, tfidfVector[i][j]);
	//	}
	//}

	return tfidfVector;
}



