#pragma once
#include "common_include.h"

class point {
public:
	point() {}
	point(std::vector<float> c, int cla) : characteristics(c), class_of_the_point(cla) 
	{
		data = characteristics.data();
	}
	float* data;
	// distance from test point
	std::vector<float> characteristics;
	int class_of_the_point;
	float valor_escalar;

	point(const point& old_obj) 
	{
		this->characteristics = old_obj.characteristics;
		this->class_of_the_point = old_obj.class_of_the_point;
		this->data = this->characteristics.data();
	}
};


class point_double {
public:
	point_double() {}
	point_double(std::vector<double> c, int cla) : characteristics(c), class_of_the_point(cla)
	{
		data = characteristics.data();
	}
	double* data;
	// distance from test point
	std::vector<double> characteristics;
	int class_of_the_point;
	double valor_escalar;

	point_double(const point_double& old_obj)
	{
		this->characteristics = old_obj.characteristics;
		this->class_of_the_point = old_obj.class_of_the_point;
		this->data = this->characteristics.data();
	}
};
