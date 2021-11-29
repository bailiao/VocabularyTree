#include "VocabularyTree.h"

int main() {
	imageRetriver retriver;
	char imagepath[100] = "D:\\opencv\\VocabularyTree\\Demo3\\test";
	retriver.buildDataBase(imagepath);
	string queryPath = "D:\\opencv\\VocabularyTree\\Demo3\\query\\103203.jpg";
	retriver.queryImage(queryPath.c_str());
	system("pause");
}