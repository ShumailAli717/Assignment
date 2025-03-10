#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Function to perform Gaussian Elimination (Sequential)
void gaussianElimination(double** A, double* B, double* X, int N) {
    // Forward Elimination
    for (int k = 0; k < N; k++) {
        for (int i = k + 1; i < N; i++) {
            double factor = A[i][k] / A[k][k];
            for (int j = k; j < N; j++) {
                A[i][j] -= factor * A[k][j];
            }
            B[i] -= factor * B[k];
        }
    }

    // Back Substitution
    for (int i = N - 1; i >= 0; i--) {
        X[i] = B[i];
        for (int j = i + 1; j < N; j++) {
            X[i] -= A[i][j] * X[j];
        }
        X[i] /= A[i][i];
    }
}

// Function to allocate a 2D matrix dynamically
double** allocateMatrix(int N) {
    double** matrix = (double**)malloc(N * sizeof(double*));
    for (int i = 0; i < N; i++) {
        matrix[i] = (double*)malloc(N * sizeof(double));
    }
    return matrix;
}

// Function to free allocated memory for a 2D matrix
void freeMatrix(double** matrix, int N) {
    for (int i = 0; i < N; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

int main() {
    int N = 500; // Matrix size (Adjust for testing)
    double** A = allocateMatrix(N);
    double* B = (double*)malloc(N * sizeof(double));
    double* X = (double*)malloc(N * sizeof(double));

    // Initialize matrix A and vector B with random values
    srand(time(NULL)); // Change the seed for different runs

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = rand() % 10 + 1;
        }
        B[i] = rand() % 10 + 1;
    }

    // Start execution timer
    clock_t start = clock();
    gaussianElimination(A, B, X, N);
    clock_t end = clock();

    // Compute execution time
    double elapsedTime = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Execution Time: %.6f seconds\n", elapsedTime);

    // Free allocated memory
    freeMatrix(A, N);
    free(B);
    free(X);

    return 0;
}
