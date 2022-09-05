#pragma once
#include "mpi_flow.h"

class mpi_type_1 : public mpi_flow
{
	//int chunk_first = 1024;
	int chunk_first = 4096;

	void run_knn();
	void workload_balancing(container_matrix_confusion& c_matrix_confusion);
	void second_step_master(int& inicial, int& last_instance, std::vector<int>& inicial_v, std::vector<long>& time, long sum_time, std::vector<int>& final_v, container_matrix_confusion& c_matrix_confusion);
	void first_step_master(std::vector<int>& inicial_v, int& inicial, std::vector<int>& final_v, int& final, container_matrix_confusion& c_matrix_confusion, std::vector<long>& time, long& sum_time);
};

