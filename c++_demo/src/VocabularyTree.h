#ifndef _VOCABULARYTREE_H_
#define _VOCABULARYTREE_H_

#include <opencv.hpp>
#include <stdio.h>
#include <iostream>
#include "imgfeatures.h"
#include "utils.h"
#include <stdlib.h>
#include <string>
#include <string.h>
#include <windows.h>
#include <map>
#include <vector>

using namespace std;

#define INTMAX 2147483647
#define INTMIN -2147483648
#define NRESULT 20
#define ANSNUM 20     //the most similiar 20 images
#define MAXFEATNUM 15000

#define DEFAULT_DEPTH 10
#define DEFAULT_BRANCH 200

class vocabularyTreeNode {
public:
	int nBranch;
	int nFeatures;
	double* feature;
	double weight;
	vocabularyTreeNode** children;

	vocabularyTreeNode() { 
		nBranch = DEFAULT_BRANCH;
		children = new vocabularyTreeNode * [nBranch];
		for (int i = 0; i < nBranch; i++) {
			children[i] = NULL;
		}
	}
	vocabularyTreeNode(int branchNum, int featureLength, double* features) {
		nBranch = branchNum;
		nFeatures = featureLength;
		feature = features;
	}

	double tf;
	double idf;
	bool add;                     //the tf varible can be added per image once
};

class featureClustering {
public:
	double* feature;
	int label;					// 属于第几个类，有一个标号
	featureClustering() { feature = NULL; label = 0; }
};

class vocabularyTree {
public:
	vocabularyTreeNode* root;
	int nNodes;
	int nBranch;
	int depth;

	vocabularyTree() { root = NULL; depth = DEFAULT_DEPTH; nBranch = DEFAULT_BRANCH; }
	void buildTree(double** features, int nFeatures, int nBranch, int depth, int featureLength);
	void buildRecursion(int curDepth, vocabularyTreeNode* &curNode, featureClustering* features, int nFeatures, int branchNum, int featureLength);
	void clearTF(vocabularyTreeNode* root, int curDepth);
	void getTFIDF(vector<double>& tfidf, vocabularyTreeNode* curNode, int curDepth);
};

class imageRetriver {
public:
	vocabularyTree* tree;
	//map<vector<double>, string> imageDatabase;
	vector<vector<double>> image_vector_cluster;
	vector<string> databaseImagePath;
	int featureLength;  
	int nImages;
	int *Features_PerImage;                        //features per image
	int totalFeatures;

	imageRetriver() { tree = NULL; nImages = 0; featureLength = 0; Features_PerImage = NULL;}
	void buildDataBase( char* directoryPath );
	vector<string> queryImage( const char* imagePath ); 

	int getTrainFeatures(double** &features, vector<string> imagePaths, int & featureLength);
	void calIDF(double** features);               //cal IDF for each node in the tree
	vector<vector<double>> getTFIDFVector(double** features, int nImages);
	vector<double> getOneTFIDFVector(double** features, int featNums, int nStart); 
	void addFeature2DataBase(vector<vector<double>> tfidfVector);

	void HKAdd(double* feature, int depth, vocabularyTreeNode* node);
	void HKDiv(vocabularyTreeNode* curNode, int curDepth);
};

extern double sqr_distance(double* vector1, double* vector2, int featureLength);
extern double vector_sqr_distance(vector<double> vector1, vector<double> vector2);
extern void node_add(double* &vector1, double* &vector2, int featureLength);
extern void node_divide_cnt(double* &vector1, int cnt, int featureLength);
extern void kmeans(featureClustering*& features, int nFeatures, int branchNum, int*& nums, int featureLength);
extern int cmp(const void* a, const void* b);
//extern bool DirectoryList(LPCSTR Path, vector<string>& path, char* ext);
extern bool DirectoryList(string Path, vector<string>& pathvector, string postfix);
#endif