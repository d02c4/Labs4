#include <iostream>
#include <chrono>
#include <vector>
#include <mpi.h>



// Размер матрицы и вектора
const int N = 6;

// Инициализация матрицы случайными значениями
void init_matrix(std::vector<double> &ab)
{
    srand(time(0));
    for (int i = 0; i < N * (N + 1); ++i) {
        ab[i] = static_cast<double>(rand() / (1000000));
    }
}

// Вывод матрицы
void print_matrix(std::vector<double> ab)
{
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N + 1; j++){
            std::cout << ab[i * (N + 1) + j] << " \t \t ";
        }
        std::cout << std::endl;
    }
}

// Вывод вектора x
void print_x(std::vector<double> x, std::vector<double> right_x)
{
    std::cout << std::endl;
    for(int i = 0; i < N; i++){
        std::cout << "x[" << i << "] = ";
        std::cout << x[i] << std::endl;
    }
}


int main(int argc, char *argv[])
{
    int size;
    int rank;
    double time;
    std::vector<double> ab(N * (N + 1));
    std::vector<double> x(N);
    std::vector<double> x_solution(N);

    // Инициализация MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0)
    {
        // Инициализация матрицы на корневом процессе
        init_matrix(ab);

        // Начало отсчёта времени
        time = -MPI_Wtime();
    }

    const int count_rows_rank = N / size;
    const int start_row = rank * count_rows_rank;
    const int end_row = start_row + count_rows_rank;

    std::vector<double> ab_local(count_rows_rank * (N + 1));
    std::vector<double> selected_row(N + 1);

    // Рассылка частей матрицы каждому процессу
    MPI_Scatter(ab.data(), count_rows_rank * (N + 1), MPI_DOUBLE, ab_local.data(), count_rows_rank * (N + 1), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Прямой ход метода Гаусса
    for (int row = 0; row < end_row; row++)
    {
        int selected_rank = row / count_rows_rank;

        if (rank == selected_rank)
        {
            int local_row = row % count_rows_rank;
            double scale = ab_local[local_row * (N + 1) + row];

            // Нормализация строки
            for (int col = row; col < N + 1; col++)
            {
                ab_local[local_row * (N + 1) + col] /= scale;
            }

            // Рассылка нормализованной строки другим процессам
            for (int i = selected_rank + 1; i < size; i++)
            {
                MPI_Send(ab_local.data() + (N + 1) * local_row, N + 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
            }

            // Приведение к нулю нижележащих строк
            for (int i = local_row + 1; i < count_rows_rank; i++)
            {
                double op = ab_local[i * (N + 1) + row];

                for (int j = row; j < N + 1; j++)
                {
                    ab_local[i * (N + 1) + j] -= ab_local[local_row * (N + 1) + j] * op;
                }
            }
        }
        else
        {
            // Получение нормализованной строки от активного процесса
            MPI_Recv(selected_row.data(), N + 1, MPI_DOUBLE, selected_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Приведение к нулю соответствующих строк
            for (int i = 0; i < count_rows_rank; i++)
            {
                double op = ab_local[i * (N + 1) + row];

                for (int j = row; j < N + 1; j++)
                {
                    ab_local[i * (N + 1) + j] -= selected_row[j] * op;
                }
            }
        }
    }

    // Обратный ход метода Гаусса
    for (int row = N - 1; row >= 0; row--)
    {
        int selected_rank = row / count_rows_rank;
        int local_row = row % count_rows_rank;

        if (rank == selected_rank)
        {
            double sum = 0;

            // Вычисление элементов решения
            for (int j = row + 1; j < N; j++)
            {
                sum += ab_local[local_row * (N + 1) + j] * x[j];
            }

            x[row] = ab_local[local_row * (N + 1) + N] - sum;
        }
        // Рассылка найденных элементов решения
        MPI_Bcast(x.data(), N, MPI_DOUBLE, selected_rank, MPI_COMM_WORLD);
    }

    if (rank == 0)
    {
        // Вывод исходной матрицы
        std::cout << "\n\nMatrix:\n";
        print_matrix(ab);

        // Вывод решения
        std::cout << "\n\nSolution:\n";
        print_x(x, x_solution);

        // Окончание отсчёта времени и вывод времени выполнения
        time += MPI_Wtime();
        std::cout << "\n\nTime: " << time << " seconds\n";
    }

    // Завершение MPI
    MPI_Finalize();
    return 0;
}
