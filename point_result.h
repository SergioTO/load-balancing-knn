#pragma once
#include "common_include.h"
#include "point.h"

class point_result {
public:
	point_result(point* p) : p(p), distance(-1)
	{

	}
	// distance from test point
	float distance;
	point* p;


};