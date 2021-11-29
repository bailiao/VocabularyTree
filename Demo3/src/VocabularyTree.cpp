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
	//for (int i = 0; i < nFeatures; i++) {
	//	feature2Cluster[i].label = -1;
	//	feature2Cluster[i].feature = features[i];

	//}


	for(int i = 0; i < nFeatures; i++) {
		feature2Cluster[i].label = -1;
		feature2Cluster[i].feature = new double[featureLength];
		for (int j = 0; j < featureLength; j++) {
			feature2Cluster[i].feature[j] = features[i][j];
		}
		
	}
	//for (int i = 0; i < nFeatures; i++) {
	//	delete []features[i];
	//}
	//delete[]features;
	buildRecursion(0, root, feature2Cluster, nFeatures, nBranch, featureLength);	//?where is root?
	//delete []feature2Cluster;
}

void vocabularyTree::buildRecursion(int curDepth, vocabularyTreeNode* &curNode, featureClustering* features, int nFeatures, int branchNum, int featureLength) {
	if (curDepth == depth) 		return;
	if (nFeatures == 0)			return;
	if (curNode == NULL) {
		cout << "curNode == NULL" << endl;
		curNode = new vocabularyTreeNode;
	}
	//curNode->children = new vocabularyTreeNode * [branchNum];
	//for(int i=0;i<branchNum)
	int* nums = new int[branchNum];
	double** clusterCenter = NULL;													// 在kmean中赋值并分配空间


	double* sumvec = new double[featureLength];
	for (int i = 0; i < featureLength; i++) {
		sumvec[i] = 0.0;
	}
	for (int i = 0; i < nFeatures; i++) {
		for (int j = 0; j < featureLength; j++) {
			sumvec[j] += features[i].feature[j];
		}
	}
	curNode->feature = new double[featureLength];
	for (int i = 0; i < featureLength; i++) {
		curNode->feature[i] = sumvec[i] / nFeatures;
	}



	kmeans(features, nFeatures, branchNum, nums, featureLength, clusterCenter);
<<<<<<< Updated upstream
	if (nFeatures <= branchNum) return;
=======
	qsort(features, nFeatures, sizeof(featureClustering), cmp);		//??
		for (int i = 0; i < nFeatures; i++) {
		cout << "Label:   " << (*features++).label << endl;
	}
>>>>>>> Stashed changes

	qsort(features, nFeatures, sizeof(featureClustering), cmp);		//??

	//int cnttmp = 0;
	//for (int i = 0; i < nBranch; i++) {
	//	cout << "number of this cluster = " << nums[i] << endl;
	//	for (int j = 0; j < nums[i]; j++) {
	//		cout << features[j + cnttmp].label << endl;
	//	}
	//	cout << "-------------------------" << endl;
	//	cnttmp += nums[i];
	//}
	//system("pause");
	
	int count = 0;
	for(int i = 0; i < nBranch; i++) {
		//if (i == 0) {
		//	printf("%p\n", features);
		//	printf("%p\n", &features[count]);
		//}
		//for (int j = 0; j < nums[i]; j++) {
		//	cout << features[j+count].label << endl;
		//}
		//cout << "-------------------------" << endl;
		buildRecursion(curDepth + 1, curNode->children[i], (featureClustering*)(&features[count]), nums[i], branchNum, featureLength);	// 在子函数中会分配空间
		count += nums[i];
	}
}

void vocabularyTree::clearTF(vocabularyTreeNode* curNode, int curDepth) {
	if(curDepth == depth)	return;
	if (curNode == NULL)	return;
	curNode->tf = 0;
	for(int i = 0; i < curNode->nBranch; i++) {
		//cout << "clear   " << i << endl;
		clearTF(curNode->children[i], curDepth + 1);
	}
}

void vocabularyTree::getTFIDF(vector<double>& tfidf, vocabularyTreeNode* curNode, int curDepth) {
	if(curDepth == depth)	return;
	if (curNode == NULL)	return;

	tfidf.push_back(curNode->tf * curNode->idf);
	for(int i = 0; i < curNode->nBranch; i++) {
		getTFIDF(tfidf, curNode->children[i], curDepth + 1);
	}
}




