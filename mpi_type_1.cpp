#include "pch.h"
#include "mpi_type_1.h"


void mpi_type_1::run_knn()
{
	//Init container matrix
	container_matrix_confusion c_matrix_confusion;
	c_matrix_confusion.init(desired_features_total, desired_k_total, total_classes, desired_features_start, desired_k_start);



	//Sync mpi
	sync_mpi();
	auto clock1 = std::chrono::high_resolution_clock::now();
	auto clock2 = std::chrono::high_resolution_clock::now();

//Enviar a cada uno su parte
//Hacer mi parte
//Unificar resultados
	cout << "mpi_type_1" << endl;
	int min_size=size_mpi* chunk_first;

	if (rank_mpi == 0 && size_mpi == 1)
	{
		execute_knn(0, size_test, c_matrix_confusion);
	}
	else {
		workload_balancing(c_matrix_confusion);
	}


	//Terminar_mpi
	//---

	clock2 = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::seconds>(clock2 - clock1);
	auto duration_milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(clock2 - clock1);
	long seconds = duration.count();
	long milli = duration_milliseconds.count();




	if (rank_mpi != 0) {
		std::cout << endl << "Finish WORKER program KNN with MRMR with type_mpi: 1  rank: " << rank_mpi << " hostname: " << hostname << "Timpo en hacer knn: " << seconds << " seconds" << endl;
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

void mpi_type_1::workload_balancing(container_matrix_confusion& c_matrix_confusion)
{
	vector<long> time;
	vector<int> inicial_v;
	vector<int> final_v;
	if (rank_mpi == 0)
	{
		long sum_time = 0;
		int inicial = 0;
		int last_instance = chunk_first; //No inclusive

		first_step_master(inicial_v, inicial, final_v, last_instance, c_matrix_confusion, time, sum_time);
		second_step_master(inicial, last_instance, inicial_v, time, sum_time, final_v, c_matrix_confusion);
	}

	if (rank_mpi != 0)
	{
		int init = 0;
		int final = 0;
		//FIRST STEP
		//Recibir
		c_m_c_mpi.recv_values_instances(init, final);
		//Ejecutar
		auto t=execute_knn(init, final, c_matrix_confusion);
		c_m_c_mpi.send_time(t);


		//SECOND STEP
		//Recibir
		c_m_c_mpi.recv_values_instances(init, final);
		t = execute_knn(init, final, c_matrix_confusion);
		//Enviar
		c_m_c_mpi.send_container_matrix_confusion(c_matrix_confusion, 10);
	}
}

void mpi_type_1::second_step_master(int& inicial, int& last_instance, std::vector<int>& inicial_v, std::vector<long>& time, long sum_time, std::vector<int>& final_v, container_matrix_confusion& c_matrix_confusion)
{

	vector<double> inverse_time;
	double inverse_time_sum = 0;
	for (int i = 0; i < time.size(); i++) {
		auto inv = 1 / (double)time[i];
		inverse_time.push_back(inv);
		inverse_time_sum += inv;

	}
	
	int instances_remain = size_test - last_instance;

	//cout << "instances remain " << instances_remain << endl;
	for (int i = 0; i < inicial_v.size() - 1; i++)
	{
	
		int v = floor((inverse_time[i] / inverse_time_sum) * instances_remain );
		//cout << "time[i] " << time[i] << endl;
		//cout << "(double)sum_time " << ((double)sum_time) << endl;

		//cout << "v " << time[i] << endl;
		inicial_v[i] = last_instance;
		last_instance += v;
		final_v[i] = last_instance;

		cout << "Proceso " << i+1 << " time: " << time[i] << " Percent time: " << (inverse_time[i] / (double)inverse_time_sum * 100) << " total: " << (final_v[i] - inicial_v[i]) << " inicial: " << inicial_v[i] << " final: " << final_v[i] << endl;
	}
	//El último se le lleva el resto si la división no es exacta
	inicial_v[inicial_v.size() - 1] = last_instance;
	final_v[inicial_v.size() - 1] = size_test;


	cout << "Proceso 0" << " time: " << time[inicial_v.size() - 1] << " Percent time: " << (inverse_time[inicial_v.size() - 1] / (double)inverse_time_sum * 100) << " total: " << (final_v[inicial_v.size() - 1] - inicial_v[inicial_v.size() - 1]) << " inicial: " << inicial_v[inicial_v.size() - 1] << " final: " << final_v[inicial_v.size() - 1] << endl;




	//for (int i = 0; i < size_mpi; i++)
	//{
	//	cout << "Proceso: " << i 
	//}
	//Enviar
	if (inicial_v.size() > 1)
	{

		for (int i = 1; i < inicial_v.size(); i++)
		{
			c_m_c_mpi.send_values_instances(inicial_v[i], final_v[i], i);
		}
	}
	//Ejecutar el hilo propio
	auto t = execute_knn(inicial_v[0], final_v[0], c_matrix_confusion);

	//Recibir
	if (inicial_v.size() > 1)
	{

		for (int i = 1; i < inicial_v.size(); i++)
		{
			//cout << "Recibir de rank " <<i<<endl;
			c_m_c_mpi.recv_from_and_fusion_container_matrix_confusion(c_matrix_confusion, i);
			//cout << "Recibido de rank " << i << endl;
		}
	}
	cout << "Terminar fase 2" << endl;
}

void mpi_type_1::first_step_master(std::vector<int>& inicial_v, int& inicial, std::vector<int>& final_v, int& last_instance, container_matrix_confusion& c_matrix_confusion, std::vector<long>& time, long& sum_time)
{
	inicial_v.push_back(inicial);
	final_v.push_back(last_instance);

	for (int i = 1; i < size_mpi; i++)
	{
		inicial = last_instance;
		last_instance += chunk_first;
		inicial_v.push_back(inicial);
		final_v.push_back(last_instance);
	}

	for (int i = 0; i < size_mpi; i++)
	{
		cout << "Proceso: " << i << " total: " << (final_v[i] - inicial_v[i]) << " inicial: " << inicial_v[i] << " final: " << final_v[i]  << endl;
	}

	//Enviar
	if (inicial_v.size() > 1)
	{

		for (int i = 1; i < inicial_v.size(); i++)
		{
			c_m_c_mpi.send_values_instances(inicial_v[i], final_v[i], i);
		}
	}

	//Ejecutar el hilo propio
	auto t = execute_knn(inicial_v[0], final_v[0], c_matrix_confusion);

	
	time.push_back(t);
	//Recibir time
	if (inicial_v.size() > 1)
	{
		for (int i = 1; i < inicial_v.size(); i++)
		{
			t = c_m_c_mpi.recv_time(i);
			time.push_back(t);
		}
	}
	sum_time = std::accumulate(time.begin(), time.end(), 0L);
}
