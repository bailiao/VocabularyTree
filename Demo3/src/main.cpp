#include "VocabularyTree.h"

int main() {
	imageRetriver retriver;
	char imagepath[100] = "E:\\Donald Duck\\MyClass5\\SRTP\\DemoBase";
	retriver.buildDataBase(imagepath);
	string queryPath = "E:\\Donald Duck\\MyClass5\\SRTP\\DemoBase\\103203.jpg";
	retriver.queryImage(queryPath.c_str());
	system("pause");
}