#include "matrix_confusion_mpi.h"
#include "pch.h"

void matrix_confusion_mpi::send_matrix_confusion(matrix_confusion* m){


    int total_values_matrix =  m->different_class * m->different_class;


        int *arraySend;
        int send_to=0;
        int contador=0;

        arraySend= new int[total_values_matrix];
 	  
         for(int i=0;i<m->different_class;i++)
         {
             for(int j=0;j<m->different_class;j++)
             {
                 arraySend[contador]=m->matrix[i][j];
                 contador++;
             }
         }
                  
                   	    //rellenar_array(arraySend,total_enviar_recibir,1);
        MPI_Send(arraySend //referencia al vector de elementos a enviar
                ,total_values_matrix // tamaño del vector a enviar
                ,MPI_INT // Tipo de dato que envias
                ,send_to // pid del proceso destino
                ,0 //etiqueta
                ,MPI_COMM_WORLD); //Comunicador por el que se manda
        delete[] arraySend; 
}

void matrix_confusion_mpi::recv_matrix_confusion(matrix_confusion* m, int recv_from)
{
    MPI_Status estado;
    int *arrayRecv;
    int contador=0;

 
    int total_values_matrix =  m->different_class * m->different_class;

     arrayRecv= new int[total_values_matrix];

            MPI_Recv(arrayRecv //referencia al vector de elementos a recibir
                ,total_values_matrix // tamaño del vector a recbir
                ,MPI_INT // Tipo de dato que recives
                ,recv_from // pid del proceso origen de la que se recibe
                ,0 //etiqueta
                ,MPI_COMM_WORLD //Comunicador por el que se manda
                ,&estado); // estructura informativa del estado

         for(int i=0;i<m->different_class;i++)
         {
             for(int j=0;j<m->different_class;j++)
             {
                 m->matrix[i][j]=arrayRecv[contador];
                 contador++;
             }
         }
         m->recalculate_classified();
    
    delete[] arrayRecv;
}