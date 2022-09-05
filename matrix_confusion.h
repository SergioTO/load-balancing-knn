#pragma once
#include "pch.h"
#include "common_include.h"
#include "point.h"
#include "point_result.h"
#include <iomanip>  
class matrix_confusion {
public:
	int different_class;
	int k;
	int number_characteristics_use;
	vector<vector<int>> matrix;



	
	int correctly_classified = 0;
	int incorrectly_classified = 0;

	matrix_confusion();

	void init_matrix_confusion(int k, int different_class, int number_characteristics_use);
	void recalculate_classified();
	double accuracy();
	void fill_matrix(int real, int assigned);
	void fusion_matrix(matrix_confusion* m);
	void printMatrix();
	void printMessures();


};