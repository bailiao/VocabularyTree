#include "VocabularyTree.h"

int main() {
	imageRetriver retriver;
	char imagepath[100] = "E:\\Donald Duck\\MyClass5\\SRTP\\DemoBase";
	retriver.buildDataBase(imagepath);
	string prefix = "E:\\Donald Duck\\MyClass5\\SRTP\\DemoBase\\";
	string postfix = ".jpg";
	string imagename;
	vector<string> queryResult;
	while (1) {
		cout << "Please input image name: ";
		cin >> imagename;
		if (imagename == "exit") break;
		queryResult = retriver.queryImage((prefix + imagename + postfix).c_str());
		cout << "----------------------------------\nHere are the query result:\n";
		for (int i = 0; i < queryResult.size(); i++) {
			cout << queryResult[i] << endl;
		}
	}
	
	
	system("pause");
}