#pragma once
#ifndef knn_classify_vertical1
#define knn_classify_vertical1
#include "common_include.h"
#include "point.h"
#include "point_result.h"
#include <omp.h>
#include <numeric>
#include <math.h>
class vertical_distance_class
{
public:
	float distance;
	int classes;
};
class knn_classify_vertical {

private:
	const int n_threads = 0;;
	const int simd = 0;
	const int tipo_run = 0;



public:

	knn_classify_vertical(int threads, int s, int t_r);
	void euclideanDistance(vector<float>& data, vector<float>& distances, float test);
	void fillDistances(vector<vector<float>>& data, vector<float>& distances, point& test, int number_characteristics_use);
	int classify(vector<vector<float>>& data, point& test, const int k, const int number_features_use, const int total_classes, vector<vertical_distance_class>& datos_para_ordenar);
	void ordenar_datos(vector<vertical_distance_class>& data, int k);
	void ordenar_datos(vector<vertical_distance_class>& data);
	int majority_voting(const int& k, std::vector<vertical_distance_class>& data, const int& total_classes);
};


class vertical_distance_class_double
{
public:
	double distance;
	int classes;
};
class knn_classify_vertical_double {

private:
	const int n_threads = 0;;
	const int simd = 0;
	const int tipo_run = 0;



public:

	knn_classify_vertical_double(int threads, int s, int t_r);
	void euclideanDistance(vector<double>& data, vector<double>& distances, double test);
	void fillDistances(vector<vector<double>>& data, vector<double>& distances, point_double& test, int number_characteristics_use);
	int classify(vector<vector<double>>& data, point_double& test, const int k, const int number_features_use, const int total_classes, vector<vertical_distance_class_double>& datos_para_ordenar);
	void ordenar_datos(vector<vertical_distance_class_double>& data, int k);
	void ordenar_datos(vector<vertical_distance_class_double>& data);
	int majority_voting(const int& k, std::vector<vertical_distance_class_double>& data, const int& total_classes);
};
#endif //PCH_H