#include "pch.h"
#include "mpi_type_2.h"


void mpi_type_2::run_knn()
{
	workers_total = size_mpi - 1;
	total_repartir = size_test;
	//Init container matrix

	c_matrix_confusion.init(desired_features_total, desired_k_total, total_classes, desired_features_start, desired_k_start);



	//Sync mpi
	sync_mpi();
	auto clock1 = std::chrono::high_resolution_clock::now();
	auto clock2 = std::chrono::high_resolution_clock::now();

	//Enviar a cada uno su parte
	//Hacer mi parte
	//Unificar resultados
	cout << "mpi_type_2"<<endl;

	if (rank_mpi == 0 && size_mpi == 1)
	{
		execute_knn(0, size_test, c_matrix_confusion);
	}
	else {
		workload_balancing();
	}


	//Terminar_mpi
	//---

	clock2 = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::seconds>(clock2 - clock1);
	auto duration_milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(clock2 - clock1);
	long seconds = duration.count();
	long milli = duration_milliseconds.count();



	std::cout << endl << "Finish WORKER program KNN with MRMR with type_mpi: 2  rank: " << rank_mpi << " hostname: " << hostname << "Timpo en hacer knn: " << seconds << " seconds" << endl;
	if (rank_mpi != 0) { 
	
		return; }



	//Solo guardamos en el master
	cout << "Timpo en hacer knn: " << seconds << " seconds" << endl;
	cout << "Timpo en hacer knn: " << milli << " milliseconds" << endl;

	//Guardar
	c_matrix_confusion.print_best_matrix();
	vector<matrix_confusion> better_matrix_for_feature = c_matrix_confusion.get_better_matrix_for_every_number_of_features();
	write_csv(better_matrix_for_feature, path_best_result_by_feature);
	write_results_csv(path_results, ((c_matrix_confusion.get_better_matrix())->accuracy()), seconds);
}

void mpi_type_2::workload_balancing()
{
	int repartir = 2;
	int number_executions_all_workers = 0;
	int repartir_esta_vez = total_repartir / repartir;

	vector<long> time;

	int inicial = 0;
	int final = 0;
	if (rank_mpi == 0) {
		for (int i = 0; i < workers_total; i++)
		{
			workers.push_back(mpi_type2_dinamic());
			total_instances_worker.push_back(0);
			envios_totales_worker.push_back(0);
		}

		 distrubir_por_nodo_primera_vez = repartir_esta_vez / workers_total;
	

		for (int i = 0; i < workers_total; i++)
		{
			workers[i].instance_init = final;
			final += distrubir_por_nodo_primera_vez;
			workers[i].instance_final = final;
		}

		//ENVIAR
		for (int i = 0; i < workers_total; i++)
		{
			workers[i].state = workers[i].tag_working;
			envios_totales_worker[i] +=1;
			total_instances_worker[i] += workers[i].instance_final - workers[i].instance_init;
			c_m_c_mpi.send_values_instances(workers[i].instance_init, workers[i].instance_final, (i + 1));
		}

		//FIN DEL PRIMER PASO

		//Segundo paso
		listen_request_from_workers(final);

	//	ELIMINATAR PARA COUT
		for (int i = 0; i < total_instances_worker.size(); i++) 
		{
			cout << "worker rank " << (i + 1) << " se le han hecho " << envios_totales_worker[i] << " envios y ha clasificado el siguiente total de instancias: " << total_instances_worker[i] << endl;
		}
		

	}
	else
	{
		listen_to_master();
	}


}

int mpi_type_2::get_minimun_number_executions_all_workers()
{
	int minimun_number_executions = workers[0].number_executions;
	for (int i = 1; i < workers_total; i++)
	{
		if (minimun_number_executions > workers[i].number_executions)
		{
			minimun_number_executions = workers[i].number_executions;
		}
	}
	return minimun_number_executions;
}

void mpi_type_2::listen_request_from_workers(int final)
{
	bool finish_all_workers = false;
	bool finish_instances = false;

	int last_instance = final;
	bool already_calculate = false;
	int executions_has = 1;
	while (finish_all_workers == false)
	{
		for (int i = 0; i < workers_total; i++)
		{

			int rank = i + 1;
			bool is_sending = c_m_c_mpi.is_ready_recv_time(rank);
			if (is_sending)
			{
				//Recibir time
				auto t = c_m_c_mpi.recv_time(rank);

				if (finish_instances == true && get_minimun_number_executions_all_workers() >= executions_has)
				{
					if (workers[i].number_executions > executions_has) {
						c_m_c_mpi.send_tag_finish(rank);
						c_m_c_mpi.recv_from_and_fusion_container_matrix_confusion(c_matrix_confusion, rank);
						workers[i].state = workers[i].tag_no_working;
						finish_all_workers = is_finish_all_workers();

					}
					else {
						workers[i].number_executions++;
						send_state_work_and_data_to_worker(rank);
					}
					continue; //Ya no hace nada de abajo
				}
				// Modificar worker
				if (workers[i].number_executions == 0)
				{
					workers[i].number_executions++;
					workers[i].duration = t;
				}
				//Enviar state "Más datos" 	//Enviar state "Fin"
				//Enviar state "inicio/fin"

				if (finish_instances == true)
				{
					c_m_c_mpi.send_tag_finish(rank);
					c_m_c_mpi.recv_from_and_fusion_container_matrix_confusion(c_matrix_confusion, rank);
					workers[i].state = workers[i].tag_no_working;
					finish_all_workers = is_finish_all_workers();
				}
				else if (get_minimun_number_executions_all_workers() == executions_has)
				{
					if (already_calculate == false) {
						vector<double> time_inverse;
						already_calculate = true;
						finish_instances = true;

						int total_instances_remain = size_test - last_instance;
						double sum_times = 0;

						double duration = workers[i].duration;
						for (int j = 0; j < workers_total; j++)
						{
							double instance_per_miliseconds = duration / total_instances_worker[j];
							auto inv = 1 / instance_per_miliseconds;
							time_inverse.push_back(inv);
							sum_times += inv;
						}

						//for (int j = 0; j < workers_total; j++)
						//{
						//	auto inv = 1 / (double)workers[j].duration;
						//	time_inverse.push_back(inv);
						//	sum_times += inv;
						//}


						/*					for (int j = 0; j < workers_total; j++)
											{
												auto percent = (100.0 * workers[j].duration / sum_times);
												int number_instances = 12;
											}*/

						for (int j = 0; j < workers_total - 1; j++)
						{

							int v = floor((time_inverse[j] / sum_times) * total_instances_remain );
							workers[j].instance_init = last_instance;
							last_instance += v;
							workers[j].instance_final = last_instance;
						}
						//El último se le lleva el resto si la división no es exacta
						workers[workers_total - 1].instance_init = last_instance;
						workers[workers_total - 1].instance_final = size_test;

						cout << "Primera clasificación de cada worker de tamaño: " << distrubir_por_nodo_primera_vez<<endl;
						for (int j = 0; j < workers_total; j++)
						{
							cout << "worker " << (j+1) << " tiempo ha durado: "<< workers[j].duration << " inversa" << time_inverse[j] << " porcentaje " << (time_inverse[j] / sum_times) << " inicial " << workers[j].instance_init << " final " << workers[j].instance_final << endl;
						}
						//for (int j = 0; j < workers_total;j++) 
						//{
							//cout << "";
						//}
					}

					workers[i].number_executions++;
					send_state_work_and_data_to_worker(rank);


				}
				else {
					if ((last_instance + chunk_min) >= size_test)
					{
						workers[i].instance_init = last_instance;
						workers[i].instance_final = size_test;
						finish_instances = true;
					}
					else
					{
						workers[i].instance_init = last_instance;
						last_instance += chunk_min;
						workers[i].instance_final = last_instance;
					}
					send_state_work_and_data_to_worker(rank);
				}
			}
		}

	}
}



//void mpi_type_2::listen_request_from_workers()
//{
//	bool finish_all_workers = false;
//	bool finish_instances = false;
//
//	int last_instance = 0;
//	while (finish_all_workers == false)
//	{
//		for (int i = 0; i < workers_total; i++)
//		{
//			if (finish_instances == true && workers[i].state == workers[i].tag_no_working)
//			{
//				//Aquí ya no hay que hacer nada, se han acabado las instancias y solo falta que acaben de enviar los workers pendientes
//			}
//			else {
//				int rank = i + 1;
//				bool is_sending = c_m_c_mpi.is_ready_recv_time(rank);
//				if (is_sending)
//				{
//					//Recibir time
//					auto t = c_m_c_mpi.recv_time(rank);
//					// Modificar worker
//					workers[i].duration = t;
//					//Enviar state "Más datos" 	//Enviar state "Fin"
//					//Enviar state "inicio/fin"
//
//					if (finish_instances == true)
//					{
//						c_m_c_mpi.send_tag_finish(rank);
//						c_m_c_mpi.recv_from_and_fusion_container_matrix_confusion(c_matrix_confusion, rank);
//						workers[i].state = workers[i].tag_no_working;
//						finish_all_workers = is_finish_all_workers();
//					}
//					else {
//						if ((last_instance + chunk_min) >= size_test)
//						{
//							workers[i].instance_init = last_instance;
//							workers[i].instance_final = size_test;
//							finish_instances = true;
//						}
//						else
//						{
//							workers[i].instance_init = last_instance;
//							last_instance += chunk_min;
//							workers[i].instance_final = last_instance;
//						}
//						send_state_work_and_data_to_worker(rank);
//					}
//				}
//			}
//		}
//	}
//}

bool mpi_type_2::is_finish_all_workers()
{
	for (int i = 0; i < workers_total; i++)
	{
		if (workers[i].state != workers[i].tag_no_working) { return false; }
	}
	return true;
}

void mpi_type_2::send_state_work_and_data_to_worker(int rank)
{
	int index = get_index_from_rank(rank);
	c_m_c_mpi.send_tag_more_values(rank);
	envios_totales_worker[index]+=1;
	total_instances_worker[index] += workers[index].instance_final - workers[index].instance_init;
	c_m_c_mpi.send_values_instances(workers[index].instance_init, workers[index].instance_final, rank);

}
void mpi_type_2::send_state_finish_to_worker(int rank)
{
	c_m_c_mpi.send_tag_finish(rank);
}


int mpi_type_2::get_index_from_rank(int rank)
{
	return rank - 1;
}



//Method worker
void mpi_type_2::listen_to_master()
{
	int initial = 0;
	int final = 0;

	//First step
	c_m_c_mpi.recv_values_instances(initial, final); //Recv instances from master
	auto t = execute_knn(initial, final, c_matrix_confusion); //Execute instances
	//if (rank_mpi==1) 
	//{
	//	cout << "Soy rank 1 mpi y voy a imprimir la matrix" << endl;
	//	c_matrix_confusion.print_best_matrix();
	//}
	c_m_c_mpi.send_time(t); //Send time execution to master

	//Second step
	bool finish = false;
	while (finish == false) {
		int tag = c_m_c_mpi.recv_tag_finish_or_more_values();
		if (tag == c_m_c_mpi.tag_send_more_values)
		{
			c_m_c_mpi.recv_values_instances(initial, final); //Recv instances from master
			auto t = execute_knn(initial, final, c_matrix_confusion); //Execute instances
			//if (rank_mpi == 1)
			//{
			//	cout << "Soy rank 1 mpi y voy a imprimir la matrix" << endl;
			//	c_matrix_confusion.print_best_matrix();
			//}
			c_m_c_mpi.send_time(t); //Send time execution to master
		}
		else if (tag == c_m_c_mpi.tag_finish)
		{
			c_m_c_mpi.send_container_matrix_confusion(c_matrix_confusion, 10);
			finish = true;//return; equivalente
		}
	}
}

