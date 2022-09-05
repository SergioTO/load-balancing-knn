#pragma once
#include "matrix_confusion.h"
#include "mpi.h"
#include "pch.h"
class matrix_confusion_mpi {
public:
    void send_matrix_confusion(matrix_confusion* m);
    void recv_matrix_confusion(matrix_confusion* m, int recv_from);
};