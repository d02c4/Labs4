#include <iostream>
#include <vector>
#include <random>
#include <oneapi/tbb.h>
// #include <tbb/parallel_for.h>
// #include <tbb/blocked_range.h>
#include <chrono>


using namespace std;
using namespace tbb;

// Функция для генерации СЛАУ
void generate_system(vector<vector<double>>& A, vector<double>& b, int n) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(-10, 10);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A[i][j] = dis(gen);
        }
        b[i] = dis(gen);
    }
}

// Функция для вывода СЛАУ
void print_system(const vector<vector<double>>& A, const vector<double>& b) {
    int n = A.size();
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            cout << A[i][j] << "*x" << j;
            if (j < n - 1) cout << " + ";
        }
        cout << " = " << b[i] << "\n";
    }
}

// Функция для решения СЛАУ методом Гаусса
void solve_system(vector<vector<double>>& A, vector<double>& b) {
    int n = A.size();

    for (int i = 0; i < n; ++i) {

        // Нормализовать текущую строку
        for (int j = i + 1; j < n; ++j) {
            A[i][j] /= A[i][i];
        }
        b[i] /= A[i][i];
        A[i][i] = 1;

        // Вычесть текущую строку из остальных
        parallel_for(blocked_range<int>(i + 1, n), [&](const blocked_range<int>& r) {
            for (int k = r.begin(); k != r.end(); ++k) {
                double factor = A[k][i];
                for (int j = i; j < n; ++j) {
                    A[k][j] -= factor * A[i][j];
                }
                b[k] -= factor * b[i];
                A[k][i] = 0;
            }
        });
    }

    // Обратный ход
    for (int i = n - 1; i >= 0; --i) {
        for (int j = i - 1; j >= 0; --j) {
            b[j] -= A[j][i] * b[i];
            A[j][i] = 0;
        }
    }
}

int main() {
    int n = 1000; // Размер СЛАУ
    vector<vector<double>> A(n, vector<double>(n));
    vector<double> b(n);

    // Генерация СЛАУ
    generate_system(A, b, n);

    // Вывод изначальных данных
    //cout << "Исходная СЛАУ:\n";
    //print_system(A, b);
    auto t1 = std::chrono::high_resolution_clock::now();
    // Решение СЛАУ
    solve_system(A, b);
    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);



    // Вывод результата
    //cout << "\nРешение СЛАУ:\n";
    //for (int i = 0; i < n; ++i) {
    //    cout << "x" << i << " = " << b[i] << "\n";
    //}
    cout << "Время выполнения: " << duration.count() << " мс" << endl;
    return 0;
}