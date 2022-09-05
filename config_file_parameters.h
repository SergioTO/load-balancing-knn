#pragma once
#include "matrix_confusion.h"
#include <omp.h>
class config_file_parameters {
public:

	//Value 0
	int desired_features_start = 0;
	//Value 1
	int desired_features_end = 0;
	//Value 2
	int desired_k_start = 0;
	//Value 3
	int desired_k_end = 0;
	//Value 4
	string path_training_data = "";
	//Value 5
	string path_training_label = "";
	//Value 6
	string path_test_data = "";
	//Value 7
	string path_test_label = "";
	//Value 8
	string path_features = "";
	//Value 9
	string path_best_result_by_feature = "";
	//Value 10
	string value_threads = "";




	const int total_values = 11; //Value start in 0 so value+1

	int n_threads_start_included;
	int n_threads_end_included;

	int desired_features_total;
	int desired_k_total;


	config_file_parameters(vector<vector<string>>& values)
	{
		if (values.size() != total_values)
		{
			string text = "The config file must to has " + to_string(total_values) + " values and it has: " + to_string(values.size());
			throw std::invalid_argument(text);
		}

		int i = 0;
		desired_features_start = stoi(values[i++][0]);//Value 0 
		desired_features_end = stoi(values[i++][0]);//Value 1
		desired_k_start = stoi(values[i++][0]);//Value 2
		desired_k_end = stoi(values[i++][0]);//Value 3
		path_training_data = values[i++][0];//Value 4
		path_training_label = values[i++][0];//Value 5
		path_test_data = values[i++][0];//Value 6
		path_test_label = values[i++][0];//Value 7
		path_features = values[i++][0];//Value 8
		path_best_result_by_feature = values[i++][0];//Value 9
		value_threads = values[i++][0];//Value 10


		desired_features_total = desired_features_end - desired_features_start + 1;
		desired_k_total = desired_k_end - desired_k_start + 1;

		if (value_threads == "All")
		{
			n_threads_start_included = omp_get_max_threads();
			n_threads_end_included = omp_get_max_threads();
		}
		else if (value_threads == "All range")
		{
			n_threads_start_included = 1;
			n_threads_end_included = omp_get_max_threads();
		}
		else if (value_threads == "None")
		{
			n_threads_start_included = 1;
			n_threads_end_included = 1;
		}
		else
		{
			n_threads_start_included = stoi(value_threads);
			n_threads_end_included = stoi(value_threads);
			if (omp_get_max_threads() < n_threads_end_included)
			{
				n_threads_start_included = omp_get_max_threads();
				n_threads_end_included = omp_get_max_threads();
			}
		}
	}


};


