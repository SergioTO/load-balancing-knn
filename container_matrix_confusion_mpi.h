#pragma once
#include "container_matrix_confusion.h"
#include "matrix_confusion_mpi.h"
#include <mpi.h>
class container_matrix_confusion_mpi {
public:
    int rank_mpi = -1;
    matrix_confusion_mpi m;
    const int tag_sync = 99999;
    const int tag_send_more_values = 88888;
    const int tag_finish = 11111;
    void send_container_matrix_confusion(container_matrix_confusion& c_m_c, long miliseconds);
    long recv_from_and_fusion_container_matrix_confusion(container_matrix_confusion& c_m_c, int recv_from);
    
    
    void send_values_instances(int init, int final, int to_send);
    void recv_values_instances(int& init, int& final);
    int recv_tag_finish_or_more_values();
    bool is_ready_recv_time(int recv_from);



    void send_time(long time);
    long recv_time(int recv_from);


    void send_tag_finish(int to_send);
    void send_tag_more_values(int to_send);

    void send_tag_sync(int to_send);
    void recv_tag_sync();
};