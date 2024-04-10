#include <iostream>
#include <cstdlib>
#include <ctime>
#include <mpi.h>

using namespace std;

const int MATRIX_SIZE = 10;

void generateRandomMatrix(int matrix[MATRIX_SIZE][MATRIX_SIZE]) {
    for (int i = 0; i < MATRIX_SIZE; ++i) {
        for (int j = 0; j < MATRIX_SIZE; ++j) {
            matrix[i][j] = rand() % 10; // генерация случайных чисел от 0 до 9
        }
    }
}

void printMatrix(int matrix[MATRIX_SIZE][MATRIX_SIZE]) {
    for (int i = 0; i < MATRIX_SIZE; ++i) {
        for (int j = 0; j < MATRIX_SIZE; ++j) {
            cout << matrix[i][j] << " ";
        }
        cout << endl;
    }
    cout << endl;
}

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int A[MATRIX_SIZE][MATRIX_SIZE], B[MATRIX_SIZE][MATRIX_SIZE], C[MATRIX_SIZE][MATRIX_SIZE];

    srand(time(NULL) + rank); // каждый процесс имеет свое семя для генерации случайных чисел

    if (rank == 0) {
        generateRandomMatrix(A);
        generateRandomMatrix(B);
        cout << "Matrix A:" << endl;
        printMatrix(A);
        cout << "Matrix B:" << endl;
        printMatrix(B);
    }

    MPI_Bcast(A, MATRIX_SIZE * MATRIX_SIZE, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(B, MATRIX_SIZE * MATRIX_SIZE, MPI_INT, 0, MPI_COMM_WORLD);

    int blockSize = MATRIX_SIZE / size;
    int localC[blockSize][MATRIX_SIZE];

    for (int i = 0; i < blockSize; ++i) {
        for (int j = 0; j < MATRIX_SIZE; ++j) {
            localC[i][j] = 0;
            for (int k = 0; k < MATRIX_SIZE; ++k) {
                localC[i][j] += A[i + rank * blockSize][k] * B[k][j];
            }
        }
    }

    MPI_Gather(localC, blockSize * MATRIX_SIZE, MPI_INT, C, blockSize * MATRIX_SIZE, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        cout << "Result Matrix C:" << endl;
        printMatrix(C);
    }

    MPI_Finalize();
    return 0;
}
