#pragma once
#include "pch.h"
#include "mrmr.h"
#include "matrix_confusion.h"

#include "common_include.h"
#include <chrono>


#include "config_file_parameters.h"
#include "container_matrix_confusion.h"
#include "knn_classify_horizontal.h"
#include "knn_classify_vertical.h"
#include "point.h"
#include "point_result.h"
#include "CSVReader.h"
#include "omp.h"
#include "load-balancing-knn.h"

class mpi_flow
{
public:
	int type_columm_row = 0; 
	int type_simd = 0;
	int type_run_parallel = 0;
	string path_config_file;
	string path_results;
	string hostname;


	int n_threads_start_included;
	int	n_threads_end_included;
	int n_threads = 1;
	int size_columns;
	int size_training;
	int size_test;
	int desired_features_start;
	int desired_features_end;
	int desired_features_total;
	int desired_k_start;
	int desired_k_end;
	int desired_k_total;
	int execute_parallel_int;

	int total_classes;
	container_matrix_confusion_mpi c_m_c_mpi;
	int rank_mpi;
	int size_mpi;

	//Vectors and paths for the construct
	vector<int> index_features;

	vector<point> training_point; //Training leidos 
	vector<point> test_point; //Test leidos 
	vector<point> reduced_point_training_float; //Training leidos reducidos
	vector<point> reduced_point_test_float; //Test leidos reducidos


	vector< point_result>training_point_result_reduced_column_float; //Training reducidos en formato para Knn Column (float)
	vector <vector<float>> datos_reducidos_row_float; // 1/2 Training reducidos en formato para Knn Row, solo los datos de los puntos (float)
	vector<vertical_distance_class> tipo_nuevo_enviar_reduced_row_float; // 2/2 Training reducidos en formato para Knn Row, solo la distancia y clase de los datos (float)


	vector<point_double>reduced_point_test_reduced_column_double; //Training reducidos en formato para Knn Column (double)
	vector<vertical_distance_class_double> tipo_nuevo_enviar_reduced_row_double; // 1/2 Training reducidos en formato para Knn Row, solo los datos de los puntos (double)
	vector<vector<double>> datos_nuevos_reduced_row_double; // 2/2 Training reducidos en formato para Knn Row, solo la distancia y clase de los datos (double)


	 string path_training_data;
	 string path_training_label;
	 string path_test_data;
	 string path_test_label;
	 string path_features_mrmr;
	 string path_best_result_by_feature;
	//End
	 long execute_knn(int init_index_value, int final_index_value, container_matrix_confusion& c_matrix_confusion);
	 void sync_mpi();

	void init(string path_config_file,  string path_results,int rank_mpi, int size_mpi, int type_run_parallel, int type_columm_row, int type_simd, string hostname);


	int total_differents_classes(std::vector<point>& data);

	void reducir_feature(int desired_number_features, std::vector<point>& data, std::vector<int>& index_features, std::vector<point>& data_reduced_and_order);

	string join_string(vector<string>& values, string delimiter);

	string put_values_result(float accuracy, long seconds, string delimiter);

	void write_results_csv(const string result_path, float accuracy, long seconds);
	void write_csv(vector<int>& data, const string result_path);
	void write_csv(vector<point>& data, const string path_data, const string path_label);
	void write_csv(vector<matrix_confusion>& data, const string result_path);

	void increment_data(int total_values_quiero, int tam_data_training_original, vector<point>& data);

	vector<vector<float>> readCSVdata(const string path);

	vector<int> readCSVlabel(const string path);

	config_file_parameters create_config_file_parameters(const string config_file_path);

	vector<point> createVectorPoint(vector<point>& result, const string path_data, const string path_label);

	void create_data_tipo_vertical(vector<vertical_distance_class>& tipo_nuevo_enviar, vector<vector<float>>& datos_nuevos, vector<point>& training_point);

	void create_points(vector<point>& training_point, vector<point>& test_point, const string path_training_data, const string path_training_label, const string path_test_data, const string path_test_label);

	void run_knn_vertical_individual(int value_index_init, int value_index_final, container_matrix_confusion& c_matrix_confusion, vector<vertical_distance_class>& data_vertical_sort, vector<vector<float>>& data_vertical, vector<point>& test_point, int features, int k);
	void run_knn_vertical(int value_index_init, int value_index_final, container_matrix_confusion& c_matrix_confusion, vector<vertical_distance_class>& data_vertical_sort, vector<vector<float>>& data_vertical, vector<point>& test_point);

	void run_knn_horizontal_individual(int value_index_init, int value_index_final, container_matrix_confusion& c_matrix_confusion, vector<point>& training_point, vector<point>& test_point, vector<point_result>& training_point_result, int features, int k);
	void run_knn_horizontal(int value_index_init, int value_index_final, container_matrix_confusion& c_matrix_confusion, vector<point>& training_point, vector<point>& test_point, vector<point_result>& training_point_result);

	void run_knn_vertical_individual_double(int value_index_init, int value_index_final, container_matrix_confusion& c_matrix_confusion, vector<vertical_distance_class_double>& data_vertical_sort, vector<vector<double>>& data_vertical, vector<point_double>& test_point, int features, int k);
	void run_knn_vertical_double(int value_index_init, int value_index_final, container_matrix_confusion& c_matrix_confusion, vector<vertical_distance_class_double>& data_vertical_sort, vector < vector<double>>& data_vertical, vector<point_double>& test_point);
	

	void mrmn_and_save_index_features(const string config_file_path, int total_classes, int size_columns, int desired_features_end, int execute_parallel_int, vector<point>& data_training, vector<point>& data_test, vector<point>& reduced_point_training, vector<point>& reduced_point_test);

	void reducir_features_training_test(int desired_features_end, vector<point>& data_training, vector<point>& data_test, vector<point>& reduced_point_training, vector<point>& reduced_point_test, vector<int>& index_features);

	void create_vector_point_result(vector<point_result>& training_point_result, vector<point>& training_point);

	void create_data(std::vector<point>& training_point, const std::string& path_training_data, const std::string& path_training_label, std::vector<point>& test_point, const std::string& path_test_data, const std::string& path_test_label);

	void virtual run_knn() = 0;

};

