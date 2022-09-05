#include "pch.h"
#include "mpi_type_0.h"

void mpi_type_0::run_knn()
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
	cout << "mpi_type_0" << endl;
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
		std::cout << endl << "Finish WORKER program KNN with MRMR with type_mpi: 0  rank: " << rank_mpi << " hostname: " << hostname << "Timpo en hacer knn: " << seconds << " seconds" << endl;

		return; }



	//Solo guardamos en el master
	cout << "Timpo en hacer knn: " << seconds << " seconds" << endl;
	cout << "Timpo en hacer knn: " << milli << " milliseconds" << endl;

	//Guardar
	c_matrix_confusion.print_best_matrix();
	vector<matrix_confusion> better_matrix_for_feature = c_matrix_confusion.get_better_matrix_for_every_number_of_features();
	cout << "Timpo en hacer knn: " << path_best_result_by_feature << " seconds" << endl;
	write_csv(better_matrix_for_feature, path_best_result_by_feature);
	cout << "Timpo en hacer knn: " << path_results << " seconds" << endl;
	write_results_csv(path_results, ((c_matrix_confusion.get_better_matrix())->accuracy()), seconds);
	cout << "Timpo en hacer knn: " << milli << " milliseconds" << endl;
}




void mpi_type_0::workload_balancing(container_matrix_confusion& c_matrix_confusion)
{
	vector<long> time_v;
	vector<int> inicial_v;
	vector<int> final_v;
	if (rank_mpi == 0)
	{
		int total_instances_every_node = size_test / size_mpi;
		int rest = size_test % size_mpi;

		int inicial = 0;
		int final = total_instances_every_node + rest; //No inclusive

		inicial_v.push_back(inicial);
		final_v.push_back(final);

		for (int i = 1; i < size_mpi; i++)
		{
			inicial = final;
			final += total_instances_every_node;
			inicial_v.push_back(inicial);
			final_v.push_back(final);
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
		auto t=execute_knn(inicial_v[0], final_v[0], c_matrix_confusion);
		time_v.push_back(t);

		//Recibir
		if (inicial_v.size() > 1)
		{

			for (int i = 1; i < inicial_v.size(); i++)
			{
				c_m_c_mpi.recv_from_and_fusion_container_matrix_confusion(c_matrix_confusion, i);
				t = c_m_c_mpi.recv_time(i);
				time_v.push_back(t);
			}
		}

		for (int i = 0; i < size_mpi; i++)
		{
			cout << "Proceso: " << i << " total: " << (final_v[i] - inicial_v[i]) << " inicial: " << inicial_v[i] << " final: " << final_v[i] << " segundos tardados: "<<time_v[i]<<endl;
		}

	}

	if (rank_mpi != 0)
	{
		int init = 0;
		int final = 0;
		//Recibir
		c_m_c_mpi.recv_values_instances(init, final);
		//Ejecutar
		auto t= execute_knn(init, final, c_matrix_confusion);
		//Enviar
		c_m_c_mpi.send_container_matrix_confusion(c_matrix_confusion, 10);
		c_m_c_mpi.send_time(t);
	}
}