#include "pch.h"
#include "mpi_flow.h"




void mpi_flow::init(string path_config_file, string path_results, int rank_mpi, int size_mpi, int type_run_parallel, int type_columm_row, int type_simd, string hostname)
{
	this->hostname = hostname;
	this->c_m_c_mpi.rank_mpi = rank_mpi;
	this->type_run_parallel = type_run_parallel;
	this->type_columm_row = type_columm_row;
	this->type_simd = type_simd;
	this->rank_mpi = rank_mpi;
	this->size_mpi = size_mpi;
	this->path_results = path_results;
	config_file_parameters config_file = create_config_file_parameters(path_config_file);
	auto clock1 = std::chrono::high_resolution_clock::now();
	auto clock2 = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::seconds>(clock2 - clock1);

	this->n_threads_start_included = config_file.n_threads_start_included;
	this->n_threads_end_included = config_file.n_threads_end_included;


	this->desired_features_start = config_file.desired_features_start;
	this->desired_features_end = config_file.desired_features_end;
	this->desired_features_total = config_file.desired_features_total;

	this->desired_k_start = config_file.desired_k_start;
	this->desired_k_end = config_file.desired_k_end;
	this->desired_k_total = config_file.desired_k_total;

	this->path_training_data = config_file.path_training_data;
	this->path_training_label = config_file.path_training_label;
	this->path_test_data = config_file.path_test_data;
	this->path_test_label = config_file.path_test_label;
	this->path_features_mrmr = config_file.path_features;
	this->path_best_result_by_feature = config_file.path_best_result_by_feature;



	//const string config_file_path_mrmr = "C:\\Users\\Usuario\\Desktop\\Master Granada\\TFM\\essex104_csv\\mrmr_index_features.csv";

	//vector<point> training_point;
	//vector<point> test_point;

	create_data(training_point, path_training_data, path_training_label, test_point, path_test_data, path_test_label);

	//vector<point> reduced_point_training;
	reduced_point_training_float.reserve(training_point.size());
	//vector<point> reduced_point_test;
	reduced_point_test_float.reserve(test_point.size());





	//mrmn_and_save_index_features(config_file_path_mrmr,  total_classes,  size_columns,  desired_features_end,  execute_parallel_int, training_point, test_point,  reduced_point_training,  reduced_point_test);
	//vector<int> index_features = readCSVlabel(path_features_mrmr);
	index_features = readCSVlabel(path_features_mrmr);


	/*
	int total_values_quiero = 10000;
	increment_data(total_values_quiero, size_training, training_point);
	increment_data(total_values_quiero, size_training, test_point);

	const string guardar1 = "/home/sergio/TFM/10000_data_training.csv";
	const string guardar2 = "/home/sergio/TFM/10000_labels_training.csv";
	const string guardar3 = "/home/sergio/TFM/10000_data_test.csv";
	const string guardar4 = "/home/sergio/TFM/10000_labels_test.csv";


	write_csv(training_point, guardar1, guardar2);
	write_csv(test_point, guardar3, guardar4);
	return;
	*/


	clock1 = std::chrono::high_resolution_clock::now();
	reducir_features_training_test(desired_features_end, training_point, test_point, reduced_point_training_float, reduced_point_test_float, index_features);
	size_columns = reduced_point_training_float[0].characteristics.size();
	//vector<vertical_distance_class> tipo_nuevo_enviar;
	//vector <vector<float>> datos_nuevos;
	create_data_tipo_vertical(tipo_nuevo_enviar_reduced_row_float, datos_reducidos_row_float, reduced_point_training_float);



	//vector< point_result>training_point_result_reduced;
	create_vector_point_result(training_point_result_reduced_column_float, reduced_point_training_float);

	
//int total_values_quiero = 1000000;
//increment_data(total_values_quiero, size_training, reduced_point_training_float);
//increment_data(total_values_quiero, size_training, reduced_point_test_float);
//
//const string guardar1 = "/home/sergiomc/data/1000000_data_training.csv";
//const string guardar2 = "/home/sergiomc/data/1000000_labels_training.csv";
//const string guardar3 = "/home/sergiomc/data/1000000_data_test.csv";
//const string guardar4 = "/home/sergiomc/data/1000000_labels_test.csv";
//
//
//write_csv(reduced_point_training_float, guardar1, guardar2);
//write_csv(reduced_point_test_float, guardar3, guardar4);
//return;





	for (auto& a : tipo_nuevo_enviar_reduced_row_float)
	{
		auto n = vertical_distance_class_double();
		n.distance = a.distance;
		n.classes = a.classes;
		tipo_nuevo_enviar_reduced_row_double.push_back(n);
	}

	for (auto& a : datos_reducidos_row_float)
	{
		datos_nuevos_reduced_row_double.push_back(vector<double>(a.begin(), a.end()));
	}
	for (auto& a : reduced_point_test_float)
	{
		reduced_point_test_reduced_column_double.push_back(point_double(vector<double>(a.characteristics.begin(), a.characteristics.end()), a.class_of_the_point));
	}

	clock2 = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::seconds>(clock2 - clock1);
	cout << "Tiempo en reducir los datos : "
		<< duration.count() << " seconds" << endl;

	//run_knn(path_best_result_by_feature, reduced_point_training_float, reduced_point_test_float, training_point_result_reduced_column_float, tipo_nuevo_enviar_row_float, datos_nuevos_float);


	/*
	int total_values_quiero = 10000;
	increment_data(total_values_quiero, size_training, training_point);
	increment_data(total_values_quiero, size_training, test_point);

	const string guardar1 = "/home/sergio/TFM/10000data_training.csv";
	const string guardar2 = "/home/sergio/TFM/10000labels_training.csv";
	const string guardar3 = "/home/sergio/TFM/10000data_test.csv";
	const string guardar4 = "/home/sergio/TFM/10000labels_test.csv";


	const string guardar1 = "C:\\Users\\Usuario\\Desktop\\Master Granada\\TFM\\essex104_csv\\50000_data_training.csv";
	const string guardar2 = "C:\\Users\\Usuario\\Desktop\\Master Granada\\TFM\\essex104_csv\\50000_labels_training.csv";
	const string guardar3 = "C:\\Users\\Usuario\\Desktop\\Master Granada\\TFM\\essex104_csv\\50000_data_test.csv";
	const string guardar4 = "C:\\Users\\Usuario\\Desktop\\Master Granada\\TFM\\essex104_csv\\50000_labels_test.csv";
	write_csv(training_point, guardar1, guardar2);
	write_csv(test_point, guardar3, guardar4);
	return;
	*/

	//switch (rank_mpi) 
	//{
	//case 0:
	//	n_threads = 7;
	//case 1:
	//	n_threads = 11;
	//	break;
	//case 2:
	//	n_threads =15;
	//		break;
	//case 3:
	//	n_threads = 20;
	//	break;
	//case 4:
	//	n_threads =35;
	//		break;
	//case 5:
	//	n_threads = 48;
	//		break;
	//case 6:
	//	n_threads = 7;
	//		break;
	//default:
	//	break;
	//}


	/*
	switch (rank_mpi)
	{
	case 0:
		n_threads = 8;
		break;
	case 1:
		n_threads = 11;
		break;
	case 2:
		n_threads = 15;
		break;
	case 3:
		n_threads = 20;
		break;
	case 4:
		n_threads = 35;
		break;
	case 5:
		n_threads = 48;
		break;
	case 6:
		n_threads =8;
		break;
	default:
		break;
	}
	*/

	cout << " Mi rank " << rank_mpi << " number_threads " << n_threads << endl;

}



int  mpi_flow::total_differents_classes(std::vector<point>& data)
{
	int total_classes;
	total_classes = 0;
	for (auto tp : data)
	{
		if (tp.class_of_the_point > total_classes) { total_classes = tp.class_of_the_point; }
	}
	total_classes++; //Need to sum 1 because the first class is zero
	return total_classes;
}
void mpi_flow::reducir_feature(int desired_number_features, std::vector<point>& data, std::vector<int>& index_features, std::vector<point>& data_reduced_and_order)
{
	int size_training = data.size();
	for (int i = 0; i < size_training; i++)
	{
		vector<float> v;
		v.reserve(desired_number_features);
		int class_of_the_point = data[i].class_of_the_point;
		for (int j = 0; j < desired_number_features; j++)
		{
			int index = index_features[j];
			v.push_back(data[i].characteristics[index]);
		}
		data_reduced_and_order.push_back(point(v, class_of_the_point));
	}
}

string mpi_flow::join_string(vector<string>& values, string delimiter = ",") {

	string result = "";
	for (int i = 0; i < values.size(); i++)
	{
		auto v = values[i];
		if (i == 0)
		{
			result = v;
		}
		else
		{
			result += delimiter + v;
		}
	}

	result += "\n";
	return result;
}

string mpi_flow::put_values_result(float accuracy, long seconds, string delimiter = ",")
{
	vector<string> values;
	string text;

	if (type_columm_row == 0) {
		text = "Column";
	}
	else if (type_columm_row == 1)
	{
		text = "Row";
	}
	else
	{
		text = "Error";
	}

	values.push_back(text);
	if (type_simd == 0) {
		text = "Yes";
	}
	else if (type_simd == 1)
	{
		text = "No";
	}
	else
	{
		text = "Error";
	}
	values.push_back(text);
	switch (type_run_parallel) //donde opción es la variable a comparar
	{
	case 0:
		text = "test instance";
		break;
	case 1:
		text = "fillDistances (inside knn)";
		break;
	case 2:
		text = "euclideanDistance (inside knn)";
		break;
	case 3:
		text = "None";
		break;
	default:
		text = "Error";
	}
	values.push_back(text);


	text = to_string(size_training);
	values.push_back(text);
	text = to_string(size_test);
	values.push_back(text);
	text = to_string(desired_features_start);
	values.push_back(text);
	text = to_string(desired_features_end);
	values.push_back(text);
	text = to_string(desired_features_total);
	values.push_back(text);
	text = to_string(desired_k_end);
	values.push_back(text);
	text = to_string(desired_k_total);
	values.push_back(text);
	text = to_string(desired_k_total);
	values.push_back(text);
	text = to_string(accuracy);
	values.push_back(text);
	text = to_string(n_threads);
	values.push_back(text);
	text = to_string(seconds);
	values.push_back(text);
	text = hostname;
	values.push_back(text);

	return join_string(values);
}




void mpi_flow::write_results_csv(const string result_path, float accuracy, long seconds)
{

	string delimiter = ";";
	//path_results
	// file pointer
	fstream fout_data;

	vector<string>title;
	vector<string>values;
	title.push_back("Horizontal_Vertical");
	title.push_back("SIMD");
	title.push_back("Type Paralle");
	title.push_back("Size training");
	title.push_back("Size test");
	title.push_back("Features start");
	title.push_back("Features end");
	title.push_back("Features total");
	title.push_back("K start");
	title.push_back("K end");
	title.push_back("K total");
	title.push_back("accuracy");
	title.push_back("Number Threads");
	title.push_back("seconds");
	title.push_back("Hostname");



	fout_data.open(result_path);
	if (fout_data.fail()) {
		fout_data.close();
		fout_data.open(result_path, ios::out | ios::app);
		fout_data << join_string(title, delimiter);
	}
	else
	{
		fout_data.close();
		fout_data.open(result_path, ios::out | ios::app);
	}

	fout_data << put_values_result(accuracy, seconds, delimiter);

	fout_data.close();
}

void mpi_flow::write_csv(vector<int>& data, const string result_path)
{

	string delimiter = ";";
	// file pointer
	fstream fout_data;

	// overwrite an existing csv file or creates a new file.
	fout_data.open(result_path, ios::out | ios::trunc);

	string text;

	for (auto matrix : data)
	{
		text = to_string(matrix);
		fout_data << text << "\n";
	}
	fout_data.close();
}

void mpi_flow::write_csv(vector<matrix_confusion>& data, const string result_path)
{

	string delimiter = ";";
	// file pointer
	fstream fout_data;

	// overwrite an existing csv file or creates a new file.
	fout_data.open(result_path, ios::out | ios::app);

	string text;

	text = "Features" + delimiter + "K" + delimiter + "accuracy" + "\n";
	fout_data << text;
	for (auto matrix : data)
	{
		text = to_string(matrix.number_characteristics_use) + delimiter + to_string(matrix.k) + delimiter + to_string(matrix.accuracy()) + "\n";
		fout_data << text;
	}
	fout_data.close();
}
void mpi_flow::write_csv(vector<point>& data, const string path_data, const string path_label)
{

	string delimiter = ";";
	// file pointer
	fstream fout_data;
	fstream fout_label;

	// overwrite an existing csv file or creates a new file.
	fout_data.open(path_data, ios::out | ios::trunc);
	fout_label.open(path_label, ios::out | ios::trunc);

	string text;
	fout_data << text;
	for (auto p : data)
	{
		for (int i = 0; i < p.characteristics.size(); i++)
		{
			if (i == 0)
			{
				text = to_string(p.characteristics[i]);
			}
			else
			{
				text += delimiter + to_string(p.characteristics[i]);
			}
		}
		text += "\n";
		fout_data << text;
		fout_label << (to_string(p.class_of_the_point) + "\n");
	}
	fout_data.close();
}

void mpi_flow::increment_data(int total_values_quiero, int tam_data_training_original, vector<point>& data)
{
	//void(*foo)(int);
	///* the ampersand is actually optional */
	//foo = &my_int_func;


	int current_size = data.size();
	auto copi_vector = data;
	data.reserve(total_values_quiero);


	while (true) {
		for (point p : copi_vector) {
			data.push_back(p);
			current_size++;
			if (current_size == total_values_quiero) { return; }
		}
	}
}

vector<vector<float>> mpi_flow::readCSVdata(const string path)
{
	CSVReader reader(path);
	vector<vector<string>> rawData = reader.getData();
	vector<vector<float>> rawDataTransformer(rawData.size());;



	int tam_columns = rawData[0].size();
	int tam = rawData.size();
#pragma omp parallel for schedule(static) num_threads(n_threads)   if(n_threads>1)
	for (int i = 0; i < tam; i++) {
		vector<float> value;
		value.reserve(tam_columns);
		for (string l : rawData[i]) {
			value.push_back(stof(l));
		}
		rawDataTransformer[i] = value;
	}
	return rawDataTransformer;
}
vector<int> mpi_flow::readCSVlabel(const string path)
{
	CSVReader reader(path);
	vector<vector<string>> rawData = reader.getData();
	vector<int> rawDataTransformer(rawData.size());

#pragma omp parallel for  schedule(static) num_threads(n_threads)   if(n_threads>1)
	for (int i = 0; i < rawData.size(); i++) {
		string line = (rawData[i])[0];
		rawDataTransformer[i] = (stoi(line));
	}
	return rawDataTransformer;
}

config_file_parameters mpi_flow::create_config_file_parameters(const string config_file_path)
{
	CSVReader reader(config_file_path);
	vector<vector<string>> rawData = reader.getData();
	return config_file_parameters(rawData);
}

vector<point> mpi_flow::createVectorPoint(vector<point>& result, const string path_data, const string path_label)
{
	vector<int> label_training = readCSVlabel(path_label);
	vector<vector<float>> data_training = readCSVdata(path_data);

	//vector<point> result;

	cout << "tam data " << to_string(data_training.size()) << endl;
	cout << "tam label " << to_string(label_training.size()) << endl;

	int sizeTraining = data_training.size();
	result = vector<point>(sizeTraining);

#pragma omp parallel for  schedule(static) num_threads(n_threads)   if(n_threads>1)
	for (int i = 0; i < sizeTraining; i++)
	{
		result[i] = point(data_training[i], label_training[i]);
	}
	return result;
}

void mpi_flow::create_data_tipo_vertical(vector<vertical_distance_class>& tipo_nuevo_enviar, vector < vector<float>>& datos_nuevos, vector<point>& training_point)
{
	tipo_nuevo_enviar = vector<vertical_distance_class>(size_training);
	for (int i = 0; i < size_training; i++)
	{
		tipo_nuevo_enviar[i].distance = 0;
		tipo_nuevo_enviar[i].classes = training_point[i].class_of_the_point;
	}


	for (int i = 0; i < size_columns; i++) {
		vector<float> v;
		for (int j = 0; j < size_training; j++)
		{
			v.push_back(training_point[j].characteristics[i]);
		}
		datos_nuevos.push_back(v);
	}
}

void mpi_flow::create_points(vector<point>& training_point, vector<point>& test_point, const string path_training_data, const string path_training_label, const string path_test_data, const string path_test_label)
{
	createVectorPoint(training_point, path_training_data, path_training_label);
	createVectorPoint(test_point, path_test_data, path_test_label);
}
void mpi_flow::run_knn_vertical_individual(int value_index_init, int value_index_final, container_matrix_confusion& c_matrix_confusion, vector<vertical_distance_class>& data_vertical_sort, vector<vector<float>>& data_vertical, vector<point>& test_point, int features, int k)
{
	knn_classify_vertical knn(n_threads, type_simd, type_run_parallel);
	vector<container_matrix_confusion> c_array_matrix_confusion(n_threads);
	for (int i = 0; i < n_threads; i++)
	{
		c_array_matrix_confusion[i].init(desired_features_total, desired_k_total, total_classes, desired_features_start, desired_k_start);
	}
	container_matrix_confusion* array_cmc = c_array_matrix_confusion.data();

	if (n_threads > 1 && type_run_parallel == 0) {
#pragma omp parallel for schedule(runtime) num_threads(n_threads) 
		for (int i = value_index_init; i < value_index_final; i++)//Test points
		{
			int id = omp_get_thread_num();
			container_matrix_confusion* container_matrix = &array_cmc[id];
			vector<vertical_distance_class> copy_data_vertical_sort = data_vertical_sort;
			point current_point_test_copy_value = test_point[i];
			int result_assigned = knn.classify(data_vertical, current_point_test_copy_value, k, features, total_classes, copy_data_vertical_sort);
			container_matrix->m_c[0][0].fill_matrix(current_point_test_copy_value.class_of_the_point, result_assigned);
		}
	}
	else
	{
		for (int i = value_index_init; i < value_index_final; i++)//Test points
		{
			int id = omp_get_thread_num();
			container_matrix_confusion* container_matrix = &array_cmc[id];
			vector<vertical_distance_class> copy_data_vertical_sort = data_vertical_sort;
			point current_point_test_copy_value = test_point[i];
			int result_assigned = knn.classify(data_vertical, current_point_test_copy_value, k, features, total_classes, copy_data_vertical_sort);
			container_matrix->m_c[0][0].fill_matrix(current_point_test_copy_value.class_of_the_point, result_assigned);
		}
	}

	c_matrix_confusion.fusion_container_matrix(array_cmc, n_threads);

}

void mpi_flow::run_knn_vertical_individual_double(int value_index_init, int value_index_final, container_matrix_confusion& c_matrix_confusion, vector<vertical_distance_class_double>& data_vertical_sort, vector<vector<double>>& data_vertical, vector<point_double>& test_point, int features, int k)
{
	knn_classify_vertical_double knn(n_threads, type_simd, type_run_parallel);
	vector<container_matrix_confusion> c_array_matrix_confusion(n_threads);
	for (int i = 0; i < n_threads; i++)
	{
		c_array_matrix_confusion[i].init(desired_features_total, desired_k_total, total_classes, desired_features_start, desired_k_start);
	}
	container_matrix_confusion* array_cmc = c_array_matrix_confusion.data();


	if (n_threads > 1 && type_run_parallel == 0) {
#pragma omp parallel for schedule(runtime)  num_threads(n_threads) 
		for (int i = value_index_init; i < value_index_final; i++)//Test points
		{
			int id = omp_get_thread_num();
			//cout << " " << id << endl;
			container_matrix_confusion* container_matrix = &array_cmc[id];
			vector<vertical_distance_class_double> copy_data_vertical_sort = data_vertical_sort;
			point_double current_point_test_copy_value = test_point[i];
			int result_assigned = knn.classify(data_vertical, current_point_test_copy_value, k, features, total_classes, copy_data_vertical_sort);
			container_matrix->m_c[0][0].fill_matrix(current_point_test_copy_value.class_of_the_point, result_assigned);
		}
	}
	else {
		for (int i = value_index_init; i < value_index_final; i++)//Test points
		{
			int id = omp_get_thread_num();
			//cout << " " << id << endl;
			container_matrix_confusion* container_matrix = &array_cmc[id];
			vector<vertical_distance_class_double> copy_data_vertical_sort = data_vertical_sort;
			point_double current_point_test_copy_value = test_point[i];
			int result_assigned = knn.classify(data_vertical, current_point_test_copy_value, k, features, total_classes, copy_data_vertical_sort);
			container_matrix->m_c[0][0].fill_matrix(current_point_test_copy_value.class_of_the_point, result_assigned);
		}
	}
	c_matrix_confusion.fusion_container_matrix(array_cmc, n_threads);
}

void mpi_flow::run_knn_vertical_double(int value_index_init, int value_index_final, container_matrix_confusion& c_matrix_confusion, vector<vertical_distance_class_double>& data_vertical_sort, vector < vector<double>>& data_vertical, vector<point_double>& test_point)
{


	knn_classify_vertical_double knn(n_threads, type_simd, type_run_parallel);
	vector<container_matrix_confusion> c_array_matrix_confusion(n_threads);
	for (int i = 0; i < n_threads; i++)
	{
		c_array_matrix_confusion[i].init(desired_features_total, desired_k_total, total_classes, desired_features_start, desired_k_start);
	}
	container_matrix_confusion* array_cmc = c_array_matrix_confusion.data();


	if (n_threads > 1 && type_run_parallel == 0) {
#pragma omp parallel for schedule(runtime)  num_threads(n_threads) 
		for (int i = value_index_init; i < value_index_final; i++)//Test points
		{
			int id = omp_get_thread_num();
			//cout << " " << id << endl;
			container_matrix_confusion* container_matrix = &array_cmc[id];
			for (int z = 0; z < desired_features_total; z++) //features
			{
				int current_z = z + desired_features_start;
				for (int k = 0; k < desired_k_total; k++)  // K
				{
					int current_k = k + desired_k_start;
					vector<vertical_distance_class_double> copy_data_vertical_sort = data_vertical_sort;
					point_double current_point_test_copy_value = test_point[i];
					int result_assigned = knn.classify(data_vertical, current_point_test_copy_value, current_k, current_z, total_classes, copy_data_vertical_sort);
					container_matrix->m_c[z][k].fill_matrix(current_point_test_copy_value.class_of_the_point, result_assigned);
				}
			}
		}
	}
	else {
		for (int i = value_index_init; i < value_index_final; i++)//Test points
		{
			int id = omp_get_thread_num();
			//cout << " " << id << endl;
			container_matrix_confusion* container_matrix = &array_cmc[id];
			for (int z = 0; z < desired_features_total; z++) //features
			{
				int current_z = z + desired_features_start;
				for (int k = 0; k < desired_k_total; k++)  // K
				{
					int current_k = k + desired_k_start;
					vector<vertical_distance_class_double> copy_data_vertical_sort = data_vertical_sort;
					point_double current_point_test_copy_value = test_point[i];
					int result_assigned = knn.classify(data_vertical, current_point_test_copy_value, current_k, current_z, total_classes, copy_data_vertical_sort);
					container_matrix->m_c[z][k].fill_matrix(current_point_test_copy_value.class_of_the_point, result_assigned);
				}
			}
		}
	}


	c_matrix_confusion.fusion_container_matrix(array_cmc, n_threads);
}
void mpi_flow::run_knn_vertical(int value_index_init, int value_index_final, container_matrix_confusion& c_matrix_confusion, vector<vertical_distance_class>& data_vertical_sort, vector < vector<float>>& data_vertical, vector<point>& test_point)
{
	knn_classify_vertical knn(n_threads, type_simd, type_run_parallel);
	vector<container_matrix_confusion> c_array_matrix_confusion(n_threads);
	for (int i = 0; i < n_threads; i++)
	{
		c_array_matrix_confusion[i].init(desired_features_total, desired_k_total, total_classes, desired_features_start, desired_k_start);
	}
	container_matrix_confusion* array_cmc = c_array_matrix_confusion.data();

	if (n_threads > 1 && type_run_parallel == 0) {
#pragma omp parallel for schedule(runtime)  num_threads(n_threads)
		for (int i = value_index_init; i < value_index_final; i++)//Test points
		{
			int id = omp_get_thread_num();
			//cout << " " << id << endl;
			container_matrix_confusion* container_matrix = &array_cmc[id];
			for (int z = 0; z < desired_features_total; z++) //features
			{
				int current_z = z + desired_features_start;
				for (int k = 0; k < desired_k_total; k++)  // K
				{
					int current_k = k + desired_k_start;
					vector<vertical_distance_class> copy_data_vertical_sort = data_vertical_sort;
					point current_point_test_copy_value = test_point[i];
					int result_assigned = knn.classify(data_vertical, current_point_test_copy_value, current_k, current_z, total_classes, copy_data_vertical_sort);
					container_matrix->m_c[z][k].fill_matrix(current_point_test_copy_value.class_of_the_point, result_assigned);
				}
			}
		}
	}
	else {
		for (int i = value_index_init; i < value_index_final; i++)//Test points
		{
			int id = omp_get_thread_num();
			//cout << " " << id << endl;
			container_matrix_confusion* container_matrix = &array_cmc[id];
			for (int z = 0; z < desired_features_total; z++) //features
			{
				int current_z = z + desired_features_start;
				for (int k = 0; k < desired_k_total; k++)  // K
				{
					int current_k = k + desired_k_start;
					vector<vertical_distance_class> copy_data_vertical_sort = data_vertical_sort;
					point current_point_test_copy_value = test_point[i];
					int result_assigned = knn.classify(data_vertical, current_point_test_copy_value, current_k, current_z, total_classes, copy_data_vertical_sort);
					container_matrix->m_c[z][k].fill_matrix(current_point_test_copy_value.class_of_the_point, result_assigned);
				}
			}
		}
	}
	c_matrix_confusion.fusion_container_matrix(array_cmc, n_threads);
}

void mpi_flow::run_knn_horizontal_individual(int value_index_init, int value_index_final, container_matrix_confusion& c_matrix_confusion, vector<point>& training_point, vector<point>& test_point, vector< point_result>& training_point_result, int features, int k)
{
	//cout << "Entrado en horizontal individual " << endl;
	knn_classify_horizontal knn(n_threads, type_simd, type_run_parallel);
	vector<container_matrix_confusion> c_array_matrix_confusion(n_threads);
	for (int i = 0; i < n_threads; i++)
	{
		c_array_matrix_confusion[i].init(desired_features_total, desired_k_total, total_classes, desired_features_start, desired_k_start);
	}
	container_matrix_confusion* array_cmc = c_array_matrix_confusion.data();

	if (n_threads > 1 && type_run_parallel == 0) {

#pragma omp parallel for schedule(runtime) num_threads(n_threads)   
		for (int i = value_index_init; i < value_index_final; i++)//Test points
		{
			int id = omp_get_thread_num();

			//if (rank_mpi == 1){
			//cout << "Rank 1 ID:" << id << endl;
			//}
			container_matrix_confusion* container_matrix = &array_cmc[id];
			vector<point_result> copy_value_training = training_point_result;
			point current_point_test_copy_value = test_point[i];
			int result_assigned = knn.classify(copy_value_training, current_point_test_copy_value, desired_k_start, desired_features_start, total_classes);
			container_matrix->m_c[0][0].fill_matrix(current_point_test_copy_value.class_of_the_point, result_assigned);
		}
	}
	else {
		for (int i = value_index_init; i < value_index_final; i++)//Test points
		{
			int id = omp_get_thread_num();

			//if (rank_mpi == 1){
			//cout << "Rank 1 ID:" << id << endl;
			//}
			container_matrix_confusion* container_matrix = &array_cmc[id];
			vector<point_result> copy_value_training = training_point_result;
			point current_point_test_copy_value = test_point[i];
			int result_assigned = knn.classify(copy_value_training, current_point_test_copy_value, desired_k_start, desired_features_start, total_classes);
			container_matrix->m_c[0][0].fill_matrix(current_point_test_copy_value.class_of_the_point, result_assigned);
		}
	}


	c_matrix_confusion.fusion_container_matrix(array_cmc, n_threads);
}


//void mpi_flow::run_knn_horizontal_individual(int value_index_init, int value_index_final, container_matrix_confusion& c_matrix_confusion, vector<point>& training_point, vector<point>& test_point, vector< point_result>& training_point_result, int features, int k)
//{
//	cout << "Entrado en horizontal individual " << endl;
//	knn_classify_horizontal knn(n_threads, type_simd, type_run_parallel);
//	vector<container_matrix_confusion> c_array_matrix_confusion(n_threads);
//	for (int i = 0; i < n_threads; i++)
//	{
//		c_array_matrix_confusion[i].init(desired_features_total, desired_k_total, total_classes, desired_features_start, desired_k_start);
//	}
//	container_matrix_confusion* array_cmc = c_array_matrix_confusion.data();
//
//
//#pragma omp parallel for  num_threads(n_threads) if(n_threads>1 &&  type_run_parallel==0)
//	for (int i = value_index_init; i < value_index_final; i++)//Test points
//	{
//		int id = omp_get_thread_num();
//		//cout << " " << id << endl;
//		container_matrix_confusion* container_matrix = &array_cmc[id];
//		vector<point_result> copy_value_training = training_point_result;
//		point current_point_test_copy_value = test_point[i];
//		int result_assigned = knn.classify(copy_value_training, current_point_test_copy_value, desired_k_start, desired_features_start, total_classes);
//		container_matrix->m_c[0][0].fill_matrix(current_point_test_copy_value.class_of_the_point, result_assigned);
//	}
//
//
//	c_matrix_confusion.fusion_container_matrix(array_cmc, n_threads);
//}
void mpi_flow::run_knn_horizontal(int value_index_init, int value_index_final, container_matrix_confusion& c_matrix_confusion, vector<point>& training_point, vector<point>& test_point, vector< point_result>& training_point_result)
{
	knn_classify_horizontal knn(n_threads, type_simd, type_run_parallel);
	vector<container_matrix_confusion> c_array_matrix_confusion(n_threads);
	for (int i = 0; i < n_threads; i++)
	{
		c_array_matrix_confusion[i].init(desired_features_total, desired_k_total, total_classes, desired_features_start, desired_k_start);
	}
	container_matrix_confusion* array_cmc = c_array_matrix_confusion.data();

	if (n_threads > 1 && type_run_parallel == 0) {
#pragma omp parallel for schedule(runtime)  num_threads(n_threads) 
		for (int i = value_index_init; i < value_index_final; i++)//Test points
		{
			int id = omp_get_thread_num();
			//cout << " " << id << endl;
			container_matrix_confusion* container_matrix = &array_cmc[id];
			for (int z = 0; z < desired_features_total; z++) //features
			{
				int current_z = z + desired_features_start;
				for (int k = 0; k < desired_k_total; k++)  // K
				{
					int current_k = k + desired_k_start;
					vector<point_result> copy_value_training = training_point_result;
					point current_point_test_copy_value = test_point[i];
					int result_assigned = knn.classify(copy_value_training, current_point_test_copy_value, current_k, current_z, total_classes);
					container_matrix->m_c[z][k].fill_matrix(current_point_test_copy_value.class_of_the_point, result_assigned);
				}
			}
		}
	}
	else {
		for (int i = value_index_init; i < value_index_final; i++)//Test points
		{
			int id = omp_get_thread_num();
			//cout << " " << id << endl;
			container_matrix_confusion* container_matrix = &array_cmc[id];
			for (int z = 0; z < desired_features_total; z++) //features
			{
				int current_z = z + desired_features_start;
				for (int k = 0; k < desired_k_total; k++)  // K
				{
					int current_k = k + desired_k_start;
					vector<point_result> copy_value_training = training_point_result;
					point current_point_test_copy_value = test_point[i];
					int result_assigned = knn.classify(copy_value_training, current_point_test_copy_value, current_k, current_z, total_classes);
					container_matrix->m_c[z][k].fill_matrix(current_point_test_copy_value.class_of_the_point, result_assigned);
				}
			}
		}
	}
	c_matrix_confusion.fusion_container_matrix(array_cmc, n_threads);
}

void mpi_flow::mrmn_and_save_index_features(const string config_file_path, int total_classes, int size_columns, int desired_features_end, int execute_parallel_int, vector<point>& data_training, vector<point>& data_test, vector<point>& reduced_point_training, vector<point>& reduced_point_test)
{
	auto clock1 = std::chrono::high_resolution_clock::now();

	mrmr mrmr1(total_classes, size_columns, desired_features_end, data_training, execute_parallel_int);

	auto clock2 = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::seconds>(clock2 - clock1);
	cout << "Tiempo en hacer el MRMR " << duration.count() << " seconds" << endl;
	clock1 = std::chrono::high_resolution_clock::now();

	write_csv(mrmr1.features_finall, config_file_path);
}
void mpi_flow::reducir_features_training_test(int desired_features_end, vector<point>& data_training, vector<point>& data_test, vector<point>& reduced_point_training, vector<point>& reduced_point_test, vector<int>& index_features)
{
	reducir_feature(desired_features_end, data_training, index_features, reduced_point_training);
	reducir_feature(desired_features_end, data_test, index_features, reduced_point_test);
}


void mpi_flow::create_vector_point_result(vector<point_result>& training_point_result, vector<point>& training_point)
{
	training_point_result.reserve(size_training);
	//Fill the training_point_result
	for (int i = 0; i < training_point.size(); i++)
	{
		point* p1 = &training_point[i];
		point_result pr(p1);
		training_point_result.push_back(pr);
	}
}

void mpi_flow::create_data(std::vector<point>& training_point, const std::string& path_training_data, const std::string& path_training_label, std::vector<point>& test_point, const std::string& path_test_data, const std::string& path_test_label)
{
	auto clock1 = std::chrono::high_resolution_clock::now();
	createVectorPoint(training_point, path_training_data, path_training_label);
	createVectorPoint(test_point, path_test_data, path_test_label);
	auto clock2 = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::seconds>(clock2 - clock1);

	cout << "Tiempo en crear los datos : "
		<< duration.count() << " seconds" << endl;

	size_columns = training_point[0].characteristics.size();
	size_training = training_point.size();
	size_test = test_point.size();
	total_classes = total_differents_classes(training_point);

}


long mpi_flow::execute_knn(int init_index_value, int final_index_value, container_matrix_confusion& c_matrix_confusion)
{
	auto clock1 = std::chrono::high_resolution_clock::now();
	if (desired_k_total == 1 && desired_features_total == 1)
	{
		if (type_columm_row == 0)
		{
			//@@@@@@@@@@@@@@
			//Column
			run_knn_horizontal_individual(init_index_value, final_index_value, c_matrix_confusion, reduced_point_training_float, reduced_point_test_float, training_point_result_reduced_column_float, desired_features_start, desired_k_start);
			//run_knn_vertical_double(init_index_value, final_index_value, c_matrix_confusion, tipo_nuevo_enviar_reduced_row_double, datos_nuevos_reduced_row_double, reduced_point_test_reduced_column_double);

		}
		else if (type_columm_row == 1)
		{
		
			//Row
			run_knn_vertical_individual(init_index_value, final_index_value, c_matrix_confusion, tipo_nuevo_enviar_reduced_row_float, datos_reducidos_row_float, reduced_point_test_float, desired_features_start, desired_k_start);
			//Row double
			//run_knn_vertical_individual_double(init_index_value, final_index_value, c_matrix_confusion, tipo_nuevo_enviar_reduced_row_double, datos_nuevos_reduced_row_double, reduced_point_test_reduced_column_double, desired_features_start, desired_k_start);
		}
	}
	else
	{
		if (type_columm_row == 0)
		{
			//Column
			run_knn_horizontal(init_index_value, final_index_value, c_matrix_confusion, reduced_point_training_float, reduced_point_test_float, training_point_result_reduced_column_float);
		}
		else if (type_columm_row == 1)
		{
			//Row
			run_knn_vertical(init_index_value, final_index_value, c_matrix_confusion, tipo_nuevo_enviar_reduced_row_float, datos_reducidos_row_float, reduced_point_test_float);
			//Row double
			//run_knn_vertical_double(init_index_value, final_index_value, c_matrix_confusion, tipo_nuevo_enviar_reduced_row_double, datos_nuevos_reduced_row_double, reduced_point_test_reduced_column_double);
		}
	}
	auto clock2 = std::chrono::high_resolution_clock::now();
	auto duration_milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(clock2 - clock1);
	long milli = duration_milliseconds.count();
	return milli;
}

void mpi_flow::sync_mpi()
{
	if (size_mpi > 1)
	{
		if (rank_mpi == 0)
		{
			for (int i = 1; i < size_mpi; i++) {
				c_m_c_mpi.send_tag_sync(i);
			}
		}
		else
		{
			c_m_c_mpi.recv_tag_sync();
		}
	}
}