// load-balancing-knn.cpp : Este archivo contiene la función "main". La ejecución del programa comienza y termina ahí.
//

#include "pch.h"
#include "mrmr.h"
#include "matrix_confusion.h"

#include "common_include.h"
#include <chrono>


#include "config_file_parameters.h"
#include "container_matrix_confusion.h"
#include "knn_classify_horizontal.h"
#include "knn_classify_vertical.h"
#include "point.h"
#include "point_result.h"
#include "CSVReader.h"
#include "omp.h"
#include "load-balancing-knn.h"
#include "mpi_flow.h";
#include "mpi_type_0.h";
#include "mpi_type_1.h";
#include "mpi_type_2.h";
#include <unistd.h>
#include <limits.h>
#include <iostream>
#include <omp.h>

int type_columm_row = 0;
int type_simd = 0;
int type_run_parallel = 0;
string path_config_file;
string path_results;




int n_threads = 1;
int size_columns;
int size_training;
int size_test;
int desired_features_start;
int desired_features_end;
int desired_features_total;
int desired_k_start;
int desired_k_end;
int desired_k_total;
int execute_parallel_int;

int total_classes;
int type_mpi;

int rank_mpi = 0;
int size_mpi = 1;

int main(int argc, char* argv[])
{
	char hostname[HOST_NAME_MAX];
	char username[LOGIN_NAME_MAX];
	gethostname(hostname, HOST_NAME_MAX);
	getlogin_r(username, LOGIN_NAME_MAX);

	string hostname_s = hostname;

	type_columm_row = 0;
	type_simd = 0;
	type_run_parallel = 0;
	type_mpi = 0;
	path_config_file = "C:\\Users\\Usuario\\Desktop\\Master Granada\\TFM\\essex104_csv\\config_file.csv";
	path_results = "C:\\Users\\Usuario\\Desktop\\Master Granada\\TFM\\essex104_csv\\results_time.csv";

	if (argc > 1)
	{
		vector<string>  arguments; //= std::vector<std::string>(argv, argv + argc);
		arguments.assign(argv + 1, argv + argc);

		type_columm_row = stoi(arguments[0]);
		type_simd = stoi(arguments[1]);
		type_run_parallel = stoi(arguments[2]);
		type_mpi = stoi(arguments[3]);
		path_config_file = arguments[4];
		path_results = arguments[5];

	}

    MPI_Init(&argc, &argv); // Inicializamos la comunicacion de los procesos
    MPI_Comm_size(MPI_COMM_WORLD, &size_mpi); // Obtenemos el numero total de hebras
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_mpi); // Obtenemos el valor de nuestro identificador
	cout<<"size "<<size_mpi<<endl;
	cout<<"rank "<<rank_mpi<<endl;

	omp_sched_t type_schedule_v;
	int chunk_v;
	omp_get_schedule(&type_schedule_v, &chunk_v);
	std::cout << endl << "Init program KNN with MRMR with "<< endl << "    type_mpi" << type_mpi << endl << "    rank: " << rank_mpi << endl << "    hostname: " << hostname << endl << "    username: " << username << endl << "    type_columm_row " << type_columm_row  << endl << "    type_run_parallel " << type_run_parallel << endl << "    type_simd "<< type_simd << endl << "    proc_bind "<< omp_get_proc_bind() << endl << "    schedule " << type_schedule_v << endl << "    chunk_v " <<chunk_v << endl;



	mpi_flow* mpi_f = NULL;
	if (type_mpi == 0) 
	{
		mpi_f = new mpi_type_0();
	}
	else if (type_mpi == 1) {
		mpi_f = new mpi_type_1();
	}
	else if (type_mpi == 2) {
		mpi_f = new mpi_type_2();
	}
	else if (type_mpi == -1) 
	{
		mpi_f = new mpi_type_0(); // Lo inicializamos de un tipo pero da igual de cual
		//Ya que al poner estos valores a 0 es como si no usara MPI y funcionará de forma independiente
		size_mpi = 1;
		rank_mpi = 0;

	}
	else {
		cout << "Error mpi_type is: " << to_string(type_mpi) << endl;
		return -1;
	}

	if (rank_mpi == 0 && size_mpi == 1) { 
		cout << "MPI Finalize al inicio" << endl;
		MPI_Finalize(); }

	mpi_f->n_threads = omp_get_max_threads();
	mpi_f -> init(path_config_file, path_results, rank_mpi, size_mpi, type_run_parallel, type_columm_row, type_simd, hostname_s);

	for (int i = mpi_f ->n_threads_start_included; i <= mpi_f ->n_threads_end_included; i++) {
		cout << "Empieza con hilos: " << to_string(i) << endl;




		mpi_f->n_threads = i;
		mpi_f->run_knn();
	}



	std::cout << endl << "Finish program KNN with MRMR with type_mpi: " <<type_mpi<<" rank: " <<rank_mpi<<   " hostname: " << hostname << " username: " << username<<  endl;

	if (!(rank_mpi == 0 && size_mpi == 1)) 
	{ 
		cout << "MPI Finalize al final" << endl;
		MPI_Finalize(); }
	return 0;



}

