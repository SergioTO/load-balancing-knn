#include "pch.h"
#include "knn_classify_vertical.h"
struct {
	bool operator()(vertical_distance_class& p1, vertical_distance_class& p2) const { return p1.distance < p2.distance; }
} comparison2;

knn_classify_vertical::knn_classify_vertical(int threads, int s, int t_r) :n_threads(threads), simd(s), tipo_run(t_r) {}

//void __attribute__ ((noinline)) knn_classify_vertical::euclideanDistance(vector<float>& data, vector<float>& distances, float test) {
void  knn_classify_vertical::euclideanDistance(vector<float>& data, vector<float>& distances, float test) {



	int tam = data.size();
	float* p_data = data.data();
	float* p_distances = distances.data();

	//cout << "TIPO_RUN "<< horizontal_vertical<< endl; 
	//cout << "SIMD "<< simd<< endl; 
	//cout << "TIPO_RUN "<< tipo_run<< endl; 


//#if TIPO_RUN==2 && SIMD==0
//#pragma omp parallel for simd schedule(runtime) num_threads(n_threads) if(n_threads>1)
//#elif TIPO_RUN==2 && SIMD==1
//#pragma omp parallel for schedule(runtime) num_threads(n_threads)   if(n_threads>1)
//#elif SIMD==0
//#pragma omp simd
//#elif SIMD==1 
//#endif 
//	for (int i = 0; i < tam; i++) {
//		p_distances[i] += (p_data[i] - test) * (p_data[i] - test);
//	}


	if (tipo_run == 2 && simd == 0) {
#pragma omp parallel for simd schedule(runtime) num_threads(n_threads) if(n_threads>1)
		for (int i = 0; i < tam; i++) {
			p_distances[i] += (p_data[i] - test) * (p_data[i] - test);
		}
	}
	else if (tipo_run == 2 && simd == 1) {
#pragma omp parallel for schedule(runtime) num_threads(n_threads)   if(n_threads>1)
		for (int i = 0; i < tam; i++) {
			p_distances[i] += (p_data[i] - test) * (p_data[i] - test);
		}
	}
	else if (simd == 0) {
#pragma omp simd
		for (int i = 0; i < tam; i++) {
			p_distances[i] += (p_data[i] - test) * (p_data[i] - test);
		}
	}
	else {
		for (int i = 0; i < tam; i++) {
			p_distances[i] += (p_data[i] - test) * (p_data[i] - test);
		}
	}
}
void knn_classify_vertical::fillDistances(vector<vector<float>>& data, vector<float>& distances, point& test, int number_characteristics_use) {



	if (tipo_run == 1) {
		int tam_training = distances.size();
		vector<vector<float>> distances_reduction(n_threads);
		for (int i = 0; i < n_threads; i++)
		{
			distances_reduction[i] = distances;
		}
	
	

#pragma omp parallel for schedule(runtime) num_threads(n_threads)   if(n_threads>1)
		for (int i = 0; i < number_characteristics_use; i++) {
			
			int id = omp_get_thread_num();
			euclideanDistance(data[i], distances_reduction[id], test.characteristics[i]);
		}

		//Unificamos

		for (int i = 0; i < n_threads; i++)
		{
#pragma omp parallel for simd schedule(runtime)  num_threads(n_threads) if(n_threads>1)
			for (int j = 0; j < tam_training; j++) {
				distances[j] += distances_reduction[i][j];
			}
		}

		int tam = distances.size();
#pragma omp parallel for simd schedule(runtime) num_threads(n_threads) if(n_threads>1)
		for (int i = 0; i < tam; i++)
		{
			distances[i] = sqrt(distances[i]);
		}
	}
	else {
		for (int i = 0; i < number_characteristics_use; i++) {
			euclideanDistance(data[i], distances, test.characteristics[i]);
		}

		int tam = distances.size();
		if (simd == 0) {
#pragma omp simd
			for (int i = 0; i < tam; i++)
			{
				distances[i] = sqrt(distances[i]);
			}
		}
		else {
			for (int i = 0; i < tam; i++)
			{
				distances[i] = sqrt(distances[i]);
			}
		}
	}


}

int knn_classify_vertical::classify(vector<vector<float>>& data, point& test, const int k, const int number_features_use, const int total_classes, vector<vertical_distance_class>& datos_para_ordenar) {
	//cout << "Vertical, n_htreads " << this->n_threads << " simd "<<this->simd << " tipo paralel "<<this->tipo_run << endl;
	
	//vector<tipo_nuevo> datos_para_ordenar;
	int tam_data = data[0].size();

	vector<float> distances(tam_data,0);

	fillDistances(data, distances, test, number_features_use);

#pragma omp simd
		for (int i = 0; i < tam_data; i++)
		{
			datos_para_ordenar[i].distance = distances[i];
		}


	//sorting so that we can get the k nearest
	//sort(data.begin(), data.end(), comparison); // 	
	
	//ordenar_datos(datos_para_ordenar); //SORT
	ordenar_datos(datos_para_ordenar, k); //PARTIAL SORT

	//int assigned_neighbor=distance_weighting_inverse(k, data, total_classes);
	int assigned_neighbor = majority_voting(k, datos_para_ordenar, total_classes);
	return assigned_neighbor;
}
//void __attribute__ ((noinline)) knn_classify_vertical::ordenar_datos(vector<vertical_distance_class>& data, int k)
void  knn_classify_vertical::ordenar_datos(vector<vertical_distance_class>& data, int k)

{
	partial_sort(data.begin(), data.begin() + k, data.end(), comparison2);
}
void  knn_classify_vertical::ordenar_datos(vector<vertical_distance_class>& data)
//void  __attribute__ ((noinline))  knn_classify_vertical::ordenar_datos(vector<vertical_distance_class>& data)
{
	sort(data.begin(), data.end(), comparison2);
}
int knn_classify_vertical::majority_voting(const int& k, std::vector<vertical_distance_class>& data, const int& total_classes)
{
	vector<int> k_neighbors_for_class(total_classes, 0);

	for (int i = 0; i < k; i++) {
		k_neighbors_for_class[data[i].classes]++;
	}


	int assigned_neighbor = -1;
	int max_value = -1;
	for (int i = 0; i < total_classes; i++)
	{
		if (k_neighbors_for_class[i] > max_value)
		{
			assigned_neighbor = i;
			max_value = k_neighbors_for_class[i];
		}
	}

	return assigned_neighbor;

};








//DOUBLE
struct {
	bool operator()(vertical_distance_class_double& p1, vertical_distance_class_double& p2) const { return p1.distance < p2.distance; }
} comparison3;

knn_classify_vertical_double::knn_classify_vertical_double(int threads, int s, int t_r) :n_threads(threads), simd(s), tipo_run(t_r) {}

void knn_classify_vertical_double::euclideanDistance(vector<double>& data, vector<double>& distances, double test) {


	int tam = data.size();
	double* p_data = data.data();
	double* p_distances = distances.data();

	//#if TIPO_RUN==2 && SIMD==0
	//#pragma omp parallel for simd schedule(runtime) num_threads(n_threads) if(n_threads>1)
	//#elif TIPO_RUN==2 && SIMD==1
	//#pragma omp parallel for schedule(runtime) num_threads(n_threads)   if(n_threads>1)
	//#elif SIMD==0 
	//#pragma omp simd
	//#elif SIMD==1 
	//#endif 
	//	for (int i = 0; i < tam; i++) {
	//		p_distances[i] += (p_data[i] - test) * (p_data[i] - test);
	//	}


	if (tipo_run == 2 && simd == 0) {
#pragma omp parallel for simd schedule(runtime) num_threads(n_threads) if(n_threads>1)
		for (int i = 0; i < tam; i++) {
			p_distances[i] += (p_data[i] - test) * (p_data[i] - test);
		}
	}
	else if (tipo_run == 2 && simd == 1) {
#pragma omp parallel for schedule(runtime) num_threads(n_threads)   if(n_threads>1)
		for (int i = 0; i < tam; i++) {
			p_distances[i] += (p_data[i] - test) * (p_data[i] - test);
		}
	}
	else if (simd == 0) {
#pragma omp simd
		for (int i = 0; i < tam; i++) {
			p_distances[i] += (p_data[i] - test) * (p_data[i] - test);
		}
	}
	else if (simd == 1)
	{
		for (int i = 0; i < tam; i++) {
			p_distances[i] += (p_data[i] - test) * (p_data[i] - test);
		}
	}

}
void knn_classify_vertical_double::fillDistances(vector<vector<double>>& data, vector<double>& distances, point_double& test, int number_characteristics_use) {



	int number_hilos = n_threads;
	int tam_training = distances.size();
	vector<vector<double>> distances_reduction(tam_training);


	if (tipo_run == 1) {
#pragma omp parallel for schedule(runtime) num_threads(n_threads)   if(n_threads>1)
		for (int i = 0; i < number_hilos; i++)
		{
			distances_reduction[i] = distances;
		}

#pragma omp parallel for schedule(runtime) num_threads(n_threads)   if(n_threads>1)
		for (int i = 0; i < number_characteristics_use; i++) {
			euclideanDistance(data[i], distances, test.characteristics[i]);
		}

		//Unificamos

		for (int i = 0; i < number_hilos; i++)
		{
#pragma omp parallel for schedule(runtime) num_threads(n_threads)   if(n_threads>1)
			for (int j = 0; j < tam_training; j++) {
				distances[j] += distances_reduction[i][j];
			}
		}

		int tam = distances.size();
#pragma omp parallel for schedule(runtime) num_threads(n_threads)   if(n_threads>1)
		for (int i = 0; i < tam; i++)
		{
			distances[i] = sqrt(distances[i]);
		}
	}
	else {
		for (int i = 0; i < number_characteristics_use; i++) {
			euclideanDistance(data[i], distances, test.characteristics[i]);
		}

		int tam = distances.size();
		for (int i = 0; i < tam; i++)
		{
			distances[i] = sqrt(distances[i]);
		}
	}

}

int knn_classify_vertical_double::classify(vector<vector<double>>& data, point_double& test, const int k, const int number_features_use, const int total_classes, vector<vertical_distance_class_double>& datos_para_ordenar) {
	//vector<tipo_nuevo> datos_para_ordenar;
	int tam_data = data[0].size();

	vector<double> distances(tam_data);

	fillDistances(data, distances, test, number_features_use);


#pragma omp simd 
	for (int i = 0; i < tam_data; i++)
	{
		datos_para_ordenar[i].distance = distances[i];
	}

	ordenar_datos(datos_para_ordenar, k);
	//ordenar_datos(datos_para_ordenar);
	//int assigned_neighbor=distance_weighting_inverse(k, data, total_classes);
	int assigned_neighbor = majority_voting(k, datos_para_ordenar, total_classes);
	return assigned_neighbor;
}
void  knn_classify_vertical_double::ordenar_datos(vector<vertical_distance_class_double>& data, int k)
{
	partial_sort(data.begin(), data.begin() + k, data.end(), comparison3);
}
void knn_classify_vertical_double::ordenar_datos(vector<vertical_distance_class_double>& data)
{
	sort(data.begin(), data.end(), comparison3);
}
int knn_classify_vertical_double::majority_voting(const int& k, std::vector<vertical_distance_class_double>& data, const int& total_classes)
{
	vector<int> k_neighbors_for_class(total_classes, 0);

#pragma omp simd 
	for (int i = 0; i < k; i++) {
		k_neighbors_for_class[data[i].classes]++;
	}


	int assigned_neighbor = -1;
	int max_value = -1;
	for (int i = 0; i < total_classes; i++)
	{
		if (k_neighbors_for_class[i] > max_value)
		{
			assigned_neighbor = i;
			max_value = k_neighbors_for_class[i];
		}
	}

	return assigned_neighbor;

};





