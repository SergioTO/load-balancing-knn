#pragma once

#ifndef C_MATRIX_C
#define C_MATRIX_C

#include "pch.h"
#include "matrix_confusion.h"

class container_matrix_confusion {
public:
	//shared_ptr<matrix_confusion> ** m_c; // matrix_confusion m_c[number_differents_features][number_differents_K]
	matrix_confusion **m_c; // matrix_confusion m_c[number_differents_features][number_differents_K]

	int number_differents_features;
	int number_differents_K;
	int different_class;
	int desired_features_start;
	int desired_k_start;

	~container_matrix_confusion();
	container_matrix_confusion();

	void init(int number_differents_features, int number_differents_K, int different_class, int desired_features_start, int desired_k_start);
	
	void print_best_matrix();
	vector<matrix_confusion> get_better_matrix_for_every_number_of_features();
	void get_better_matrix(matrix_confusion*& better_matrix);
	matrix_confusion* get_better_matrix();
	void fusion_container_matrix(container_matrix_confusion* s, int tam);
};
#endif //C_MATRIX_C
