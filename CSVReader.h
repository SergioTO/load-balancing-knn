#pragma once
#include "common_include.h"
using namespace std;
class CSVReader {
public:
	CSVReader(const string& fileName, const string& delimeter = ",") :
		fileName(fileName),
		delimeter(delimeter)
	{}
	/*
	* Function to fetch the data from a CSV file
	*/
	vector<vector<string>> getData() {

		ifstream file(this->fileName);
		
		string line = "";
		vector<string> dataRaw;
		vector<vector<string>> data;
		
		int tam = 0;
		getline(file, line);
		auto tmp = split(line, ",", tam);
		tam = tmp.size();
		data.push_back(tmp);
	
		while (getline(file, line)) {
			//dataRaw.push_back(line);
			//vector<string> dataRaw= );
			data.push_back(split(line, ",",tam));
			//dataRaw.clear();
		}

		//int tam = dataRaw.size();

		//#pragma omp parallel for
		//for (int i = 0; i < tam; i++) {
	
		//}

		file.close();
		return data;
	}

private:
	string fileName;
	string delimeter;

	/*
	* Function used to split each line by the delim
	*/
	vector<string> split(string target, string delim, int tam)
	{
		vector<string> v;
		if (tam > 0) { v.reserve(tam); }
		
			if (!target.empty()) {
				size_t start = 0;
				do {
					size_t x = target.find(delim, start);
					// a check whether the target is found
					if (x == -1)
					{
						break;
					}
					string tmp = target.substr(start, x - start);
					v.push_back(tmp);
					start += delim.size() + tmp.size();
				} while (true);

				v.push_back(target.substr(start));
			}
		
	
		return v;
	}
};