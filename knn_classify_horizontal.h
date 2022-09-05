#pragma once

#ifndef KK_HORIZONTAL
#define KK_HORIZONTAL
#include "common_include.h"
#include "point.h"
#include "point_result.h"
#include "container_matrix_confusion.h"
#include <omp.h>
#include <numeric>
#include <math.h>




class knn_classify_horizontal {
private:
	const int n_threads = 0;;
	const int simd = 0;
	const int tipo_run = 0;
public:
	knn_classify_horizontal(int threads, int s, int t_r);
	float euclideanDistance(point& data, point& test, int number_characteristics_use);
	float manhattanDistance(point& data, point& test, int number_characteristics_use);
	void fillDistances(vector<point_result>& data, point& test, int number_characteristics_use);
	int classify(vector<point_result>& data, point& test, const int k, const int number_characteristics_use, const int total_classes);
	void ordenar_datos(vector<point_result>& data, int k);
	void ordenar_datos(vector<point_result>& data);
	int majority_voting(const int& k, std::vector<point_result>& data, const int& total_classes);
	int distance_weighting_inverse(const int& k, std::vector<point_result>& data, const int& total_classes);
};

class knn_classify_horizontal_euclidian_diferente {
private:
	const int n_threads = 0;;
	const int simd = 0;
	const int tipo_run = 0;
public:
	knn_classify_horizontal_euclidian_diferente(int threads, int s, int t_r);
	double euclideanDistance(point& data, point& test, int number_characteristics_use);
	double manhattanDistance(point& data, point& test, int number_characteristics_use);
	void fillDistances(vector<point_result>& data, point& test, int number_characteristics_use);
	int classify(vector<point_result>& data, vector<point>& tests, const int k, const int number_characteristics_use, const int total_classes, container_matrix_confusion& c_matrix_confusion, vector<container_matrix_confusion>& c_array_matrix_confusion);
	void ordenar_datos(vector<point_result>& data, int k);
	void ordenar_datos(vector<point_result>& data);
	int distance_weighting_inverse(const int& k, std::vector<point_result>& data, const int& total_classes);
	int majority_voting(const int& k, std::vector<point_result>& data, const int& total_classes);
};
#endif //KK_HORIZONTAL