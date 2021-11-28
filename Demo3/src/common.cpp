#include "VocabularyTree.h"
#include <string>
#include <chrono>
#include <iomanip>
#include <direct.h>
#include <cstdlib>
#include <algorithm>
#include <iterator>
#include <io.h>
//==========================other functions===================================
#define MAX_ITER 200
#define THRESHOLD 0.01   //not sure

double sqr_distance(double* vector1, double* vector2, int featureLength) {
	double sum = 0;
	for(int i = 0; i < featureLength; i++)
		sum += (vector1[i] - vector2[i]) * (vector1[i] - vector2[i]);
		//sum += (vector1[i] - vector2[i]) * (vector1[1] - vector2[i]);


	//if (sum < 0) {
	//	for (int i = 0; i < featureLength; i++) {
	//		printf("vector1[%d] = %lf\t", i, vector1[i]);
	//		printf("vector2[%d] = %lf\t", i, vector2[i]);
	//		printf("diff = %lf\t", vector1[1] - vector2[i]);
	//		//cout << (vector1[i] - vector2[i]) * (vector1[1] - vector2[i]) << endl;
	//		cout << (vector1[i] - vector2[i]) * (vector1[1] - vector2[i]) << endl;
	//	}
	//	system("pause");
	//}

	return sum;
}

double vector_sqr_distance(vector<double> vector1, vector<double> vector2) {
	int size = vector1.size();
	double sum = 0;
	for(int i = 0; i < size; i++)
		sum += (vector1[i] - vector2[i]) * (vector1[i] - vector2[i]);

	return sum;
}

void node_add(double* &vector1, double* &vector2, int featureLength) {
	for(int i = 0; i < featureLength; i++) {
		vector1[i] += vector2[i];
	}
}

void node_divide_cnt(double* &vector1, int cnt, int featureLength) {
	//if (cnt == 0) {
	//	cnt = 1e-3;
	//	cout << "cnt = " << cnt << endl;
	//}

	// 上面的不管用，因为cnt是整型变量，还是cnt = 0
	
	if (cnt == 0) {
		for (int i = 0; i < featureLength; i++) {
			vector1[i]  = 0;
		}
		return;
	}
	for (int i = 0; i < featureLength; i++) {
		vector1[i] /= cnt;
		//cout << vector1[i] << endl;
	}
		
}

void kmeans(featureClustering* &features, int nFeatures, int branchNum, int* &nums, int featureLength, double** &clusterCenter) {
	//for (int i = 0; i < nFeatures; i++) {
	//	cout << features[i].label << endl;
	//}
	//cout << "-------------------------" << endl;
	if (nFeatures == 0) return;
											// features 	结构体数组	(label + 长度为featurelength的double*)
											// nFeatures	目前有多少个feature, 如果少于branchNum个,每一个单独成类
											// 目的是构建出 branchNum 个 cluster
											// nums			每个cluster中的点的个数
	for(int i = 0; i < branchNum; i++)
		nums[i] = 0;						// 已经分配了内存

	if(nFeatures <= branchNum) {
		clusterCenter = new double*[nFeatures];
		for(int i = 0; i < nFeatures; i++) {
			clusterCenter[i] = features[i].feature;
			nums[i] = 1;
			features[i].label = i;
		}

		for (int i = nFeatures; i < branchNum; i++) {
			//clusterCenter[i] = NULL;
			nums[i] = 0;
			features[i].label = -1;
		}
		return;
	}

	int* idx = new int[nFeatures];
	int* cnt = new int[branchNum];

	clusterCenter = new double*[branchNum];
	for (int i = 0; i < branchNum; i++) {
		clusterCenter[i] = new double[featureLength];
		for (int j = 0; j < featureLength; j++) {
			//cout << "i = " << i << "        j = " << j << endl;
			//cout << "%p\n" << &(features[i]) << endl;
			//cout << "%p\n" << &(features[i].feature[j]) << endl;
			clusterCenter[i][j] = features[i].feature[j];		// 初始化分配内存 + 深拷贝
			//clusterCenter[i][j] = rand() % 10 - 5;		// 初始化分配内存 + 深拷贝
			
		}
			
	}

	double** tempCenters;							// 构建每次的临时N个center, 记录center的坐标, 但是最初全部初始化为0
													// tmpCenter,先对相应的属于该中心的点进行累加,之后除以基数得到中心,最后重新赋给clusterCenter
	tempCenters = new double*[branchNum];
	for(int i = 0; i < branchNum; i++) {
		tempCenters[i] = new double[featureLength];
		for(int j = 0; j < featureLength; j++)
			tempCenters[i][j] = 0;
	}
	double sum;
	for(int iter = 0; iter < MAX_ITER; iter++) {
		cout << "enter iteration   " ;
		//sum = 0;
		//if (iter == 21) {
		//	;
		//	cout << "------------------------------------------------------" << endl;
		//	;
		//	;
		//}
		memset(cnt, 0, sizeof(int) * branchNum);
		//for(int i = 0; i < branchNum; i++)
			// memset(tempCenters, 0, sizeof(double) * featureLength);	XXX

			
		for (int i = 0; i < branchNum; i++) {
			for (int j = 0; j < featureLength; j++) tempCenters[i][j] = 0;
		}

		// for(int i = 0; i < featureLength; i++) {	  XXX
		for(int i = 0; i < nFeatures; i++) {				

			// double mindis = 1e20;
			// int minIndex = 0;
			double mindis = 10000000;
			int minIndex = -1;

			for(int j = 0; j < branchNum; j++) {
				// double dis = sqr_distance(clusterCenter[i], features[i].feature, featureLength);
				double dis = sqr_distance(clusterCenter[j], features[i].feature, featureLength);
				if (_isnan(dis)) continue;
				if(dis < mindis) {
					mindis = dis;
					minIndex = j;
				}
			}
			cnt[minIndex]++;
			features[i].label = minIndex;
			node_add(tempCenters[idx[i] = minIndex], features[i].feature, featureLength);	
		}

		for (int i = 0; i < branchNum; i++) {
			//if (i == 7) {
			//	cout << cnt[i] << endl;
			//	for (int tmp = 0; tmp < 128; tmp++) {
			//		cout << tmp << "      " << tempCenters[i][tmp] <<    endl;
			//	}
			//}
			//cout << "i = " << cnt[i] << endl;
			node_divide_cnt(tempCenters[i], cnt[i], featureLength);
			//if (i == 7) {
			//	cout << cnt[i] << endl;
			//	for (int tmp = 0; tmp < 128; tmp++) {
			//		cout << tmp << "      " << tempCenters[i][tmp] << endl;
			//	}
			//}
		}
			
		//if (iter == 22) {
		//	cout << "-----------------------------------" << endl;
		//}
		sum = 0;
		for (int i = 0; i < branchNum; i++) {
			/*if (i == 7) {
				for (int tmp = 0; tmp < 128; tmp++) {
					cout << tmp << "      " << tempCenters[i][tmp] << endl;
				}
			}*/
			sum += sqr_distance(tempCenters[i], clusterCenter[i], featureLength);
			//cout << sum << endl;
			if (_isnan(sum))  continue;
		}
			

		// clusterCenter = tempCenters;												
		for(int i = 0; i < branchNum; i++) {		
			for(int j = 0; j < featureLength; j++)
				clusterCenter[i][j] = tempCenters[i][j];
		}
		
		if (_isnan(sum)) {
			cout << "iteration:  " << iter << "sum = NaN" << endl;
		}
		else {
			if (sum < THRESHOLD || iter == MAX_ITER) {
				for (int i = 0; i < branchNum; i++)
					nums[i] = cnt[i];
				cout << "iteration:  " << iter << "      sum = " << sum << endl;
				break;
			}
			cout << "iteration:  " << iter << "      sum = " << sum << endl;
		}
	}

	for (int i = 0; i < branchNum; i++) {
		delete []tempCenters[i];
	}
	delete []tempCenters;

	delete[] idx;
	delete[] cnt;
	cout << "kmeans end" << endl;
	cout << "sum = " << sum << endl;
	cout << "Threshold = " << THRESHOLD << endl;
	return;
	
}

int cmp(const void* a, const void* b) {
	int label_a = ((featureClustering*)a)->label;
	int label_b = ((featureClustering*)b)->label;
	return label_a > label_b ? 1 : -1;
}

#define LEN 1024

bool DirectoryList(string Path, vector<string>& pathvector, string postfix) {

	_finddata_t FileInfo;
	string FilePathName = Path + "\\*" + postfix;
	decltype(_findfirst(FilePathName.c_str(), &FileInfo)) k;
	decltype(_findfirst(FilePathName.c_str(), &FileInfo)) HANDLE;
	k = HANDLE = _findfirst(FilePathName.c_str(), &FileInfo);

	while (k != -1)
	{
		// 如果是普通文件夹则输出
		if (!(FileInfo.attrib & _A_SUBDIR) && strcmp(FileInfo.name, ".") != 0 && strcmp(FileInfo.name, "..") != 0)
		{
			string tmp_file_name = FileInfo.name;
			tmp_file_name = Path + "\\" + tmp_file_name;
			pathvector.push_back(tmp_file_name);
			//std::cout << FileInfo.name << std::endl;
		}

		k = _findnext(HANDLE, &FileInfo);
	}
	_findclose(HANDLE);

	return pathvector.size() > 0;

}
//bool DirectoryList(LPCSTR Path, vector<string>& path, char* ext) {			
//	WIN32_FIND_DATA FindData;
//	HANDLE hError;
//	int FileCount = 0;
//	char FilePathName[LEN];
//	WCHAR FullPathName[LEN];		// IN WINDOWS FORMAT
//	strcpy(FilePathName, Path);
//	strcat(FilePathName, "\\*.*");
//
//	WCHAR LongClassName[256];							// convert char* to LPCWSTR
//	memset(LongClassName, 0, sizeof(LongClassName));
//	MultiByteToWideChar(CP_ACP, 0, FilePathName, strlen(FilePathName) + 1, LongClassName,
//	sizeof(LongClassName) / sizeof(LongClassName[0]));
//
//	hError = FindFirstFile(LongClassName, &FindData);
//	if (hError == INVALID_HANDLE_VALUE) {
//		printf("error");
//		return 0;
//	}
//	char filename_char[256];
//	
//	while(::FindNextFile(hError, &FindData)) {
//		sprintf(filename_char, "%ws", FindData.cFileName);
//		if (strcmp(filename_char, ".") == 0 
//		 || strcmp(filename_char, "..") == 0 ) {
//			continue;
//		}
//		
//		char format_char[256] = "%s\\%s";
//		WCHAR format_Wchar[256];							// convert char* to LPCWSTR
//		memset(format_Wchar, 0, sizeof(format_Wchar));
//		MultiByteToWideChar(CP_ACP, 0, format_char, strlen(format_char) + 1, format_Wchar,
//			sizeof(format_Wchar) / sizeof(format_Wchar[0]));
//		//WCHAR format[256];
//		//wscanf(format, "%x", "%s\\%s");
//
//		wsprintf(FullPathName, format_Wchar, Path, FindData.cFileName);
//		FileCount++;
//	
//		
//		int nlen = WideCharToMultiByte(CP_ACP, 0, FullPathName, -1, NULL, 0, NULL, NULL);
//		char* full_name_char = new char(nlen);
//
//		WideCharToMultiByte(CP_ACP, 0, FullPathName, -1, full_name_char, nlen, NULL, NULL);
//		full_name_char[nlen - 1] = 0;
//		
//		string temp = full_name_char;
//		delete[]full_name_char;
//		if(temp.find(ext) != temp.npos) 
//			path.push_back(string(temp));
//
//
//		
//		int nChar = WideCharToMultiByte(CP_ACP, 0, FullPathName, -1, NULL, 0, NULL, NULL);
//		nChar = nChar * sizeof(char);
//		char* outPara = new char[nChar];//输出的参数
//		ZeroMemory(outPara, nChar);
//		WideCharToMultiByte(CP_ACP, 0, FullPathName, -1, outPara, nChar, NULL, NULL);
//
//
//		if (FindData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
//			printf("<Dir>");
//			DirectoryList(outPara, path, ext);			//input must be LPCSTR
//		}
//	}
//}