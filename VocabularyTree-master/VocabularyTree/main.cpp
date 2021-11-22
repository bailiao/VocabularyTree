#include "VocabularyTree.h"

int main() {
	imageRetriver retriver;
	retriver.buildDataBase("C:/Users/HCK/Pictures/Saved Pictures/Water.jpeg");
	string queryPath;
	cin >> queryPath;
	retriver.queryImage(queryPath.c_str());
}