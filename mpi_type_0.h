#pragma once
#include "mpi_flow.h"
class mpi_type_0 : public mpi_flow
{
public:

    void run_knn();

    void workload_balancing(container_matrix_confusion& c_matrix_confusion);

};

