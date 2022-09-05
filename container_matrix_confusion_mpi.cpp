#include "container_matrix_confusion_mpi.h"
#include "pch.h"




void container_matrix_confusion_mpi::send_time(long time)
{
	long arraySend[1];
	arraySend[0] = time;
	MPI_Send(arraySend //referencia al vector de elementos a enviar
		, 1 // tamaño del vector a enviar
		, MPI_LONG // Tipo de dato que envias
		, 0 // pid del proceso destino
		, 0 //etiqueta
		, MPI_COMM_WORLD); //Comunicador por el que se manda
}
long container_matrix_confusion_mpi::recv_time(int recv_from)
{
	MPI_Status estado;
	long arrayRecv[1];

	MPI_Recv(arrayRecv // Referencia al vector donde se almacenara lo recibido
		, 1 // tamaño del vector a recibir
		, MPI_LONG // Tipo de dato que recibe
		, recv_from // pid del proceso origen de la que se recibe
		, 0 // etiqueta
		, MPI_COMM_WORLD // Comunicador por el que se recibe
		, &estado); // estructura informativa del estado

	return arrayRecv[0];
}


bool container_matrix_confusion_mpi::is_ready_recv_time(int recv_from)
{
	MPI_Status estado;
	int flag;
	MPI_Iprobe(recv_from // pid del proceso origen de la que se recibe
		, 0 // etiqueta	
		, MPI_COMM_WORLD // Comunicador por el que se recibe
		, &flag //Resultado de si hay algo esperando a ser recibido o no
		, &estado); // estructura informativa del estado

	return flag;
}


void container_matrix_confusion_mpi::send_tag_more_values(int to_send)
{
	int arraySend[1];
	arraySend[0] = tag_send_more_values;
	MPI_Send(arraySend //referencia al vector de elementos a enviar
		, 1 // tamaño del vector a enviar
		, MPI_INT // Tipo de dato que envias
		, to_send // pid del proceso destino
		, 0 //etiqueta
		, MPI_COMM_WORLD); //Comunicador por el que se manda
}
void container_matrix_confusion_mpi::send_tag_finish(int to_send)
{
	int arraySend[1];
	arraySend[0] = tag_finish;
	MPI_Send(arraySend //referencia al vector de elementos a enviar
		, 1 // tamaño del vector a enviar
		, MPI_INT // Tipo de dato que envias
		, to_send // pid del proceso destino
		, 0 //etiqueta
		, MPI_COMM_WORLD); //Comunicador por el que se manda
}

int container_matrix_confusion_mpi::recv_tag_finish_or_more_values()
{
	MPI_Status estado;
	int arrayRecv[1];

	MPI_Recv(arrayRecv // Referencia al vector donde se almacenara lo recibido
		, 1 // tamaño del vector a recibir
		, MPI_INT // Tipo de dato que recibe
		, 0 // pid del proceso origen de la que se recibe
		, 0 // etiqueta
		, MPI_COMM_WORLD // Comunicador por el que se recibe
		, &estado); // estructura informativa del estado

	if (arrayRecv[0] != tag_send_more_values && arrayRecv[0] != tag_finish) {
		cout << "Error in recv_tag_finish_or_more_values" << arrayRecv[0] << endl;
		exit;
	}
	return arrayRecv[0];
}



void container_matrix_confusion_mpi::send_tag_sync(int to_send) 
{
	int arraySend[1];
	arraySend[0] = tag_sync;
	MPI_Send(arraySend //referencia al vector de elementos a enviar
		, 1 // tamaño del vector a enviar
		, MPI_INT // Tipo de dato que envias
		, to_send // pid del proceso destino
		, 0 //etiqueta
		, MPI_COMM_WORLD); //Comunicador por el que se manda
}
void container_matrix_confusion_mpi::recv_tag_sync() 
{
	MPI_Status estado;
	int arrayRecv[1];

	MPI_Recv(arrayRecv // Referencia al vector donde se almacenara lo recibido
		, 1 // tamaño del vector a recibir
		, MPI_INT // Tipo de dato que recibe
		, 0 // pid del proceso origen de la que se recibe
		, 0 // etiqueta
		, MPI_COMM_WORLD // Comunicador por el que se recibe
		, &estado); // estructura informativa del estado

	if (arrayRecv[0] != tag_sync) {
		cout << "Error in recv_sync" << arrayRecv[0] << endl;
		exit;
	}
}

    void container_matrix_confusion_mpi::send_values_instances(int init, int final, int to_send)
	{
		   int arraySend[2];
		   arraySend[0]=init;
		   arraySend[1]=final;


		    MPI_Send(arraySend //referencia al vector de elementos a enviar
                ,2 // tamaño del vector a enviar
                ,MPI_INT // Tipo de dato que envias
                ,to_send // pid del proceso destino
                ,0 //etiqueta
                ,MPI_COMM_WORLD); //Comunicador por el que se manda

		//cout << "Soy el proceso " << rank_mpi << " y envio a "<<to_send<< " valor: " << init << " y " << final << endl;

	}
    void container_matrix_confusion_mpi::recv_values_instances(int& init, int& final)
	{
		    MPI_Status estado;
			int arrayRecv[2];

		 MPI_Recv(arrayRecv // Referencia al vector donde se almacenara lo recibido
                ,2 // tamaño del vector a recibir
                ,MPI_INT // Tipo de dato que recibe
                ,0 // pid del proceso origen de la que se recibe
                ,0 // etiqueta
                ,MPI_COMM_WORLD // Comunicador por el que se recibe
                ,&estado); // estructura informativa del estado
		
		init=arrayRecv[0];
		final=arrayRecv[1];
	//	cout<<"Soy el proceso " <<rank_mpi <<" y he recibido de 0 el valor : "<<init<<" y "<<final<<endl;
	}

void container_matrix_confusion_mpi::send_container_matrix_confusion(container_matrix_confusion& c_m_c, long miliseconds) {
    
	


    	int number_differents_features = c_m_c.number_differents_features;
		int number_differents_K = c_m_c.number_differents_K;
		for (int i = 0; i < number_differents_features; i++)
		{
			for (int k = 0; k < number_differents_K; k++)
			{
				auto m_c = &c_m_c.m_c[i][k];
				this->m.send_matrix_confusion(m_c);
			}
		}

        //ENVIAR TIEMPO
}

long container_matrix_confusion_mpi::recv_from_and_fusion_container_matrix_confusion(container_matrix_confusion& c_m_c, int recv_from)
{
        long miliseconds=0;
        
        int number_differents_features = c_m_c.number_differents_features;
		int number_differents_K = c_m_c.number_differents_K;
		int different_class = c_m_c.different_class;
		int desired_features_start = c_m_c.desired_features_start;
		int desired_k_start = c_m_c.desired_k_start;


       container_matrix_confusion c_m_c_temp;
       c_m_c_temp.init(number_differents_features,number_differents_K,different_class,desired_features_start,desired_k_start);
        
        
		for (int i = 0; i < number_differents_features; i++)
		{
			for (int k = 0; k < number_differents_K; k++)
			{

				auto m_c = &c_m_c_temp.m_c[i][k];
				this->m.recv_matrix_confusion(m_c,recv_from);
			}
		}


		//cout<<"Vamos a inmprimir la de 0"<<endl;
		//c_m_c.print_best_matrix();
		//		cout<<"Vamos a inmprimir la de 1"<<endl;
		//c_m_c_temp.print_best_matrix();

		//Le doy un puntero y le digo que el tamaño es 1, por lo tanto la lista tiene
		//solo el elemento que le paso
		auto list_container=&c_m_c_temp;
         c_m_c.fusion_container_matrix(list_container,1); 

         //recibir tiempo
		 //milisecondos=11;

         return miliseconds;

}