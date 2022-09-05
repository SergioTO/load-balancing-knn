#include "pch.h"
#include "knn_classify_horizontal.h"


struct {
	bool operator()(point_result& p1, point_result& p2) const { return p1.distance < p2.distance; }
} comparison;

knn_classify_horizontal::knn_classify_horizontal(int threads, int s, int t_r) :n_threads(threads), simd(s), tipo_run(t_r) {}

float knn_classify_horizontal::euclideanDistance(point& data, point& test, int number_characteristics_use) {
	//float __attribute__ ((noinline)) knn_classify_horizontal::euclideanDistance(point& data, point& test, int number_characteristics_use) {
	float distance = 0;
	float* data_pointer = data.data;
	float* test_pointer = test.data;

	//El código con compilación condicional era más compacto
	//#if TIPO_RUN==2 && SIMD==0
	//#pragma omp parallel for simd schedule(runtime) reduction(+:distance) num_threads(n_threads) if(n_threads>1)
	//#elif TIPO_RUN==2 && SIMD==1
	//#pragma omp parallel for reduction(+:distance) num_threads(n_threads) if(n_threads>1)
	//#elif SIMD==0 
	//#pragma omp simd reduction(+:distance)
	//#elif SIMD==1 
	//#endif 
	//	for (int i = 0; i < number_characteristics_use; i++) {
	//		distance += (data_pointer[i] - test_pointer[i]) * (data_pointer[i] - test_pointer[i]);
	//	}
	//
	//	return sqrt(distance);


	if (tipo_run == 2 && simd == 0)
	{
#pragma omp parallel for simd schedule(runtime) reduction(+:distance) num_threads(n_threads) if(n_threads>1)
		for (int i = 0; i < number_characteristics_use; i++) {
			distance += (data_pointer[i] - test_pointer[i]) * (data_pointer[i] - test_pointer[i]);
		}

		return sqrt(distance);
	}
	else if (tipo_run == 2 && simd == 1)
	{
#pragma omp parallel for reduction(+:distance) num_threads(n_threads) if(n_threads>1)
		for (int i = 0; i < number_characteristics_use; i++) {
			distance += (data_pointer[i] - test_pointer[i]) * (data_pointer[i] - test_pointer[i]);
		}

		return sqrt(distance);
	}
	else if (simd == 0)
	{
#pragma omp simd reduction(+:distance)
		for (int i = 0; i < number_characteristics_use; i++) {
			distance += (data_pointer[i] - test_pointer[i]) * (data_pointer[i] - test_pointer[i]);
		}

		return sqrt(distance);
	}
	else
	{
		for (int i = 0; i < number_characteristics_use; i++) {
			distance += (data_pointer[i] - test_pointer[i]) * (data_pointer[i] - test_pointer[i]);
		}

		return sqrt(distance);
	}
}

float knn_classify_horizontal::manhattanDistance(point& data, point& test, int number_characteristics_use) {
	float resultadoFinal = 0;
	for (int i = 0; i < number_characteristics_use; i++) {
		resultadoFinal += abs((data.characteristics[i] - test.characteristics[i]));
	}
	return resultadoFinal;
}

void knn_classify_horizontal::fillDistances(vector<point_result>& data, point& test, int number_characteristics_use) {
	int tam = data.size();

	if (tipo_run == 1) {
#pragma omp parallel for schedule(runtime) num_threads(n_threads)   if(n_threads>1)
		for (int i = 0; i < tam; i++) {
			data[i].distance = euclideanDistance(*data[i].p, test, number_characteristics_use);
		}
	}
	else
	{
		for (int i = 0; i < tam; i++) {
			data[i].distance = euclideanDistance(*data[i].p, test, number_characteristics_use);
		}
	}
}

int knn_classify_horizontal::classify(vector<point_result>& data, point& test, const int k, const int number_characteristics_use, const int total_classes) {
	fillDistances(data, test, number_characteristics_use);
	//sorting so that we can get the k nearest
	//ordenar_datos(data); //SORT
	ordenar_datos(data, k); //PARTIAL SORT
	int assigned_neighbor = majority_voting(k, data, total_classes);
	return assigned_neighbor;
}
void knn_classify_horizontal::ordenar_datos(vector<point_result>& data, int k)
//void __attribute__ ((noinline)) knn_classify_horizontal::ordenar_datos(vector<point_result>& data, int k)
{
	partial_sort(data.begin(), data.begin() + k, data.end(), comparison);
}
void knn_classify_horizontal::ordenar_datos(vector<point_result>& data)
{
	sort(data.begin(), data.end(), comparison);
}

int knn_classify_horizontal::distance_weighting_inverse(const int& k, std::vector<point_result>& data, const int& total_classes)
{
	vector<float> k_neighbors_for_class(total_classes, 0);
	for (int i = 0; i < k; i++) {
		float distance = data[i].distance;
		if (distance == 0) { distance = 0.000001; }
		distance = 1 / distance;

		k_neighbors_for_class[data[i].p->class_of_the_point] += distance;
	}


	int assigned_neighbor = -1;
	float max_value = -1;
	for (int i = 0; i < total_classes; i++)
	{
		if (k_neighbors_for_class[i] > max_value)
		{
			assigned_neighbor = i;
			max_value = k_neighbors_for_class[i];
		}
	}
	return assigned_neighbor;
}
int   knn_classify_horizontal::majority_voting(const int& k, std::vector<point_result>& data, const int& total_classes)
//int  __attribute__ ((noinline)) knn_classify_horizontal::majority_voting(const int& k, std::vector<point_result>& data, const int& total_classes)
{
	vector<int> k_neighbors_for_class(total_classes, 0);


	for (int i = 0; i < k; i++) {
		k_neighbors_for_class[data[i].p->class_of_the_point]++;
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
}


//Diferencia entre una clase y la siguiente

knn_classify_horizontal_euclidian_diferente::knn_classify_horizontal_euclidian_diferente(int threads, int s, int t_r) :n_threads(threads), simd(s), tipo_run(t_r) {}

double knn_classify_horizontal_euclidian_diferente::euclideanDistance(point& data, point& test, int number_characteristics_use) {
	float distance = 0;
	float* data_pointer = data.data;
	float* test_pointer = test.data;

	//#pragma omp simd
	//for (int i = 0; i < number_characteristics_use; i++) {
	//	distance += (data_pointer[i] * test_pointer[i]);
	//}
	distance = std::inner_product(begin(data.characteristics), end(data.characteristics), begin(test.characteristics), 0.0);

	distance *= 2;
	distance = data.valor_escalar - distance + test.valor_escalar;

	return sqrt(distance);
}

double knn_classify_horizontal_euclidian_diferente::manhattanDistance(point& data, point& test, int number_characteristics_use) {
	double resultadoFinal = 0;
	for (int i = 0; i < number_characteristics_use; i++) {
		resultadoFinal += abs((data.characteristics[i] - test.characteristics[i]));
	}
	return resultadoFinal;
}

void knn_classify_horizontal_euclidian_diferente::fillDistances(vector<point_result>& data, point& test, int number_characteristics_use) {
	int tam = data.size();

	for (int i = 0; i < tam; i++) {
		data[i].distance = euclideanDistance(*data[i].p, test, number_characteristics_use);
	}
}

int knn_classify_horizontal_euclidian_diferente::classify(vector<point_result>& data, vector<point>& tests, const int k, const int number_characteristics_use, const int total_classes, container_matrix_confusion& c_matrix_confusion, vector<container_matrix_confusion>& c_array_matrix_confusion) {
	int size_training = data.size();
	int size_test = tests.size();

	//#pragma omp parallel for
	for (int i = 0; i < size_training; i++)
	{
		float* p = data[i].p->data;

		float sum = std::inner_product(begin(data[i].p->characteristics), end(data[i].p->characteristics), begin(data[i].p->characteristics), 0.0);

		/*
			float sum = 0;
			#pragma omp simd
			for (int j = 0; j < number_characteristics_use; j++)
			{
				sum+=p[j] * p[j];
			}
				}*/
		data[i].p->valor_escalar = sum;
	}

	//#pragma omp parallel for
	for (int i = 0; i < size_test; i++)
	{
		float* p = tests[i].data;
		//	float sum = 0;

		float sum = std::inner_product(begin(tests[i].characteristics), end(tests[i].characteristics), begin(tests[i].characteristics), 0.0);
		/*
		#pragma omp simd
					for (int j = 0; j < number_characteristics_use; j++)
					{
						sum += p[j] * p[j];
					}
					*/
		tests[i].valor_escalar = sum;
	}

#pragma omp parallel for schedule(runtime) num_threads(n_threads)   if(n_threads>1)
	for (int i = 0; i < size_test; i++)
	{
		auto d2 = data;
		fillDistances(d2, tests[i], number_characteristics_use);
		//sorting so that we can get the k nearest
		//sort(data.begin(), data.end(), comparison);
		ordenar_datos(d2, k);
		//int assigned_neighbor=distance_weighting_inverse(k, data, total_classes);
		int assigned_neighbor = knn_classify_horizontal_euclidian_diferente::majority_voting(k, d2, total_classes);

		int id = omp_get_thread_num();
		c_array_matrix_confusion[id].m_c[0][0].fill_matrix(tests[i].class_of_the_point, assigned_neighbor);
	}


	return 0;
}
void  knn_classify_horizontal_euclidian_diferente::ordenar_datos(vector<point_result>& data, int k)
{
	partial_sort(data.begin(), data.begin() + k, data.end(), comparison);
}
void knn_classify_horizontal_euclidian_diferente::ordenar_datos(vector<point_result>& data)
{
	sort(data.begin(), data.end(), comparison);
}

int knn_classify_horizontal_euclidian_diferente::distance_weighting_inverse(const int& k, std::vector<point_result>& data, const int& total_classes)
{
	vector<double> k_neighbors_for_class(total_classes, 0);
	for (int i = 0; i < k; i++) {
		double distance = data[i].distance;
		if (distance == 0) { distance = 0.000001; }
		distance = 1 / distance;

		k_neighbors_for_class[data[i].p->class_of_the_point] += distance;
	}


	int assigned_neighbor = -1;
	double max_value = -1;
	for (int i = 0; i < total_classes; i++)
	{
		if (k_neighbors_for_class[i] > max_value)
		{
			assigned_neighbor = i;
			max_value = k_neighbors_for_class[i];
		}
	}
	return assigned_neighbor;
}
int knn_classify_horizontal_euclidian_diferente::majority_voting(const int& k, std::vector<point_result>& data, const int& total_classes)
{
	cout << "Entrado en diferente majority_voting " << endl;
	vector<int> k_neighbors_for_class(total_classes, 0);

#pragma omp simd 
	for (int i = 0; i < k; i++) {
		k_neighbors_for_class[data[i].p->class_of_the_point]++;
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
}
