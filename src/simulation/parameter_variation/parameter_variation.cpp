#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>

// Define constants
const int MAX_BITS = 30;

// Function to compute direction numbers
void compute_direction_numbers(std::vector<std::vector<unsigned int>>& direction_numbers) {
    int dimension = direction_numbers.size();
    for (int i = 0; i < dimension; ++i) {
        for (int j = 0; j < MAX_BITS; ++j) {
            if (j < i) {
                direction_numbers[i][j] = 1 << (MAX_BITS - 1 - j);
            } else {
                direction_numbers[i][j] = direction_numbers[i][j - i] ^ (direction_numbers[i][j - i] >> i);
                for (int k = 1; k < i; ++k) {
                    direction_numbers[i][j] ^= ((direction_numbers[i][j - k] & (1 << (MAX_BITS - k - 1))) >> (k + 1));
                }
            }
        }
    }
}

// Function to generate Sobol sequence
std::vector<std::vector<double>> generate_sobol_sequence(int n, int dimension) {
    std::vector<std::vector<unsigned int>> direction_numbers(dimension, std::vector<unsigned int>(MAX_BITS, 0));
    compute_direction_numbers(direction_numbers);

    std::vector<std::vector<double>> sobol_sequence(n, std::vector<double>(dimension, 0.0));
    std::vector<unsigned int> x(dimension, 0);

    for (int i = 0; i < n; ++i) {
        unsigned int gray_code = i ^ (i >> 1);
        for (int j = 0; j < MAX_BITS; ++j) {
            if (gray_code & (1 << j)) {
                for (int k = 0; k < dimension; ++k) {
                    x[k] ^= direction_numbers[k][j];
                }
            }
        }

        for (int k = 0; k < dimension; ++k) {
            sobol_sequence[i][k] = x[k] / static_cast<double>(1 << MAX_BITS);
        }
    }

    return sobol_sequence;
}

// Example model function with multiple outputs
std::vector<double> model(const std::vector<double>& x, int num_outputs) {
    // Replace with your actual model
    std::vector<double> outputs(num_outputs);
    outputs[0] = std::sin(x[0]) + std::cos(x[1]);
    outputs[1] = std::cos(x[0]) * std::sin(x[1]);
    return outputs;
}

// Function to calculate Sobol indices
std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> calculate_sobol_indices(const std::vector<std::vector<double>>& sobol_sequence, int n, int dimension, int num_outputs) {
    int half_n = n / 2;
    std::vector<std::vector<double>> f_A(half_n, std::vector<double>(num_outputs));
    std::vector<std::vector<double>> f_B(half_n, std::vector<double>(num_outputs));
    std::vector<std::vector<double>> A(half_n, std::vector<double>(dimension));
    std::vector<std::vector<double>> B(half_n, std::vector<double>(dimension));

    for (int i = 0; i < half_n; ++i) {
        A[i] = sobol_sequence[i];
        B[i] = sobol_sequence[half_n + i];
        f_A[i] = model(A[i], num_outputs);
        f_B[i] = model(B[i], num_outputs);
    }

    std::vector<std::vector<double>> S(dimension, std::vector<double>(num_outputs, 0.0));
    std::vector<std::vector<double>> ST(dimension, std::vector<double>(num_outputs, 0.0));
    std::vector<double> f_A_mean(num_outputs, 0.0), f_B_mean(num_outputs, 0.0);

    for (int i = 0; i < half_n; ++i) {
        for (int j = 0; j < num_outputs; ++j) {
            f_A_mean[j] += f_A[i][j];
            f_B_mean[j] += f_B[i][j];
        }
    }

    for (int j = 0; j < num_outputs; ++j) {
        f_A_mean[j] /= half_n;
        f_B_mean[j] /= half_n;
    }

    std::vector<double> f_A_var(num_outputs, 0.0);
    for (int i = 0; i < half_n; ++i) {
        for (int j = 0; j < num_outputs; ++j) {
            f_A_var[j] += (f_A[i][j] - f_A_mean[j]) * (f_A[i][j] - f_A_mean[j]);
        }
    }

    for (int j = 0; j < num_outputs; ++j) {
        f_A_var[j] /= half_n;
    }

    for (int j = 0; j < dimension; ++j) {
        std::vector<std::vector<double>> f_Bi(half_n, std::vector<double>(num_outputs));
        for (int i = 0; i < half_n; ++i) {
            std::vector<double> A_Bi = A[i];
            A_Bi[j] = B[i][j];
            f_Bi[i] = model(A_Bi, num_outputs);
        }

        std::vector<double> f_Bi_mean(num_outputs, 0.0);
        for (int i = 0; i < half_n; ++i) {
            for (int k = 0; k < num_outputs; ++k) {
                f_Bi_mean[k] += f_Bi[i][k];
            }
        }

        for (int k = 0; k < num_outputs; ++k) {
            f_Bi_mean[k] /= half_n;
        }

        std::vector<double> V_Ej(num_outputs, 0.0), V_Cj(num_outputs, 0.0);
        for (int i = 0; i < half_n; ++i) {
            for (int k = 0; k < num_outputs; ++k) {
                V_Ej[k] += (f_A[i][k] - f_A_mean[k]) * (f_Bi[i][k] - f_B_mean[k]);
                V_Cj[k] += (f_B[i][k] - f_B_mean[k]) * (f_Bi[i][k] - f_Bi_mean[k]);
            }
        }

        for (int k = 0; k < num_outputs; ++k) {
            V_Ej[k] /= half_n;
            V_Cj[k] /= half_n;

            S[j][k] = V_Ej[k] / f_A_var[k];
            ST[j][k] = 1.0 - V_Cj[k] / f_A_var[k];
        }
    }

    return {S, ST};
}
