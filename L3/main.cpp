#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

int n = 4;
double *a = (double*)malloc(sizeof(*a) * n * n);
double *b = (double*)malloc(sizeof(*b) * n);
double *x = (double*)malloc(sizeof(*x) * n);


void MatrixInit() {
    srand(time(NULL)); // seed the random number generator with the current time

    for (int i = 0; i < n; i++) {
        b[i] = rand() % 100 + 1;

        for (int j = 0; j < n; j++)
            a[i * n + j] = rand() % 500 + 1;
    }
}

void PrintMatrix() {
    printf("Исходная матрица из коэф:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("| %f ", a[i * n + j]);
        }
        printf("   \t| %f", b[i]);
        printf("|\n");
    }
}

void PrintResult() {
    printf("\nРешение:\n");
    for (int i = 0; i < n; i++) {
        printf("x[%d] = %f\n", i, x[i]);
    }
}


int main() {
    MatrixInit();
    PrintMatrix();
    double sum;
    double t1 = omp_get_wtime();


    #pragma omp parallel
    {
        int num_threads = omp_get_num_threads();
        #pragma omp single
        {
            printf("Количество используемых потоков: %d\n", num_threads);
        }

        // Прямой ход
        // Приведение системы к треугольному виду
        for (int k = 0; k < n - 1; k++) {
            // Исключение xk из последующих уравнений
            // Выбирается опорный элемент
            double pivot = a[k * n + k];
            #pragma omp for
            for (int i = k + 1; i < n; i++) {
                // Из уравнения i вычитается уравнение k
                double tmp  = a[i * n + k] / pivot;
                for (int j = k; j < n; j++)
                    a[i * n + j] -= tmp * a[k * n + j];
                b[i] -= tmp * b[k];
            }
        }

        // Обратных ход для системы треугольного вида
        for (int k = n - 1; k >= 0; k--) {
            sum = 0;
            #pragma omp barrier
            #pragma omp for reduction(+:sum)

            for (int i = k + 1; i < n; i++)
                sum += a[k * n + i] * x[i];
            #pragma omp single
            x[k] = (b[k] - sum) / a[k * n + k];
        }
    }
    double t2 = omp_get_wtime();
    PrintResult();
    // Вывод времени расчета решения
    printf("\nВремя расчета решения: %f секунд\n", t2 - t1);

    return 0;
}
