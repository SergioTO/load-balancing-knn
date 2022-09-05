#pragma once
#include "mpi_flow.h"
#include "mpi_type2_dinamic.h"

class mpi_type_2 : public mpi_flow
{

	int distrubir_por_nodo_primera_vez = 0;
	
	//int chunk_min = 1024;
	int chunk_min = 2048;

	int total_repartir = 0;
	int workers_total;
	vector<int> total_instances_worker;
	vector<int> envios_totales_worker;
	vector<mpi_type2_dinamic> workers;
	container_matrix_confusion c_matrix_confusion;


	void  run_knn();
	void workload_balancing();


	//Methods master
	void listen_request_from_workers(int final);
	void send_state_work_and_data_to_worker(int rank);
	void send_state_finish_to_worker(int rank);

	int get_index_from_rank(int rank);
	int get_minimun_number_executions_all_workers();
	bool is_finish_all_workers();




	//Methods workers
	void listen_to_master();
	

};

