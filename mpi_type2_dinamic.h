#pragma once
#include <chrono>

class mpi_type2_dinamic
{
public:
	static const int tag_no_working = 0;
	static const int tag_working = 1;

	int state = tag_no_working;
	std::chrono::steady_clock::time_point start;
	std::chrono::steady_clock::time_point end;
	long long duration;
	int total_instances;
	int number_executions = 0;
	int instance_init;
	int instance_final;
};

