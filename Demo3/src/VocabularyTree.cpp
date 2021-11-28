#include "VocabularyTree.h"

//==========================functions in class vocabularyTree========================
void vocabularyTree::buildTree(double** features, int nFeatures, int nBranch, int depth, int featureLength) {
	featureClustering* feature2Cluster;
	feature2Cluster = new featureClustering[nFeatures];		// point to nFeatures structure{double array + int label}
	//for (int i = 0; i < 10; i++) {
	//	for (int j = 0; j < 128; j++) {
	//		cout << features[i][j] << "   ";
	//	}
	//	cout << endl;
	//}
	//cout << 1111111;
	for(int i = 0; i < nFeatures; i++) {
		feature2Cluster[i].label = 0;
		feature2Cluster[i].feature = features[i];
	}

	buildRecursion(0, root, feature2Cluster, nFeatures, nBranch, featureLength);	//?where is root?
}

void vocabularyTree::buildRecursion(int curDepth, vocabularyTreeNode* &curNode, featureClustering* features, int nFeatures, int branchNum, int featureLength) {
	if (curDepth == depth) 		return;
	if (curNode == NULL) {
		cout << "curNode == NULL" << endl;
		curNode = new vocabularyTreeNode;
	}
	//curNode->children = new vocabularyTreeNode * [branchNum];
	//for(int i=0;i<branchNum)
	int* nums = new int[branchNum];
	double** clusterCenter = NULL;													// 在kmean中赋值并分配空间

	kmeans(features, nFeatures, branchNum, nums, featureLength, clusterCenter);
	qsort(features, nFeatures, sizeof(featureClustering*), cmp);		//??

	
	int count = 0;
	for(int i = 0; i < nBranch; i++) {
		//if (i == 0) {
		//	printf("%p\n", features);
		//	printf("%p\n", &features[count]);
		//}
		buildRecursion(curDepth + 1, curNode->children[i], (featureClustering*)(&features[count]), nums[i], branchNum, featureLength);	// 在子函数中会分配空间
		count += nums[i];
	}
}

void vocabularyTree::clearTF(vocabularyTreeNode* curNode, int curDepth) {
	if(curDepth == depth)
		return;
	curNode->tf = 0;
	for(int i = 0; i < curNode->nBranch; i++) {
		clearTF(curNode->children[i], curDepth + 1);
	}
}

void vocabularyTree::getTFIDF(vector<double>& tfidf, vocabularyTreeNode* curNode, int curDepth) {
	if(curDepth == depth)
		return;

	tfidf.push_back(curNode->tf * curNode->idf);
	for(int i = 0; i < curNode->nBranch; i++) {
		getTFIDF(tfidf, curNode->children[i], curDepth + 1);
	}
}




