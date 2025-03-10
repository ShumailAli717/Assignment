#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

#define N 500  // Matrix size (Adjustable)

// Function to perform Gaussian Elimination with OpenMP
void gaussianElimination(double** A, double* B, double* X) {
    int i, j, k;

    // Forward Elimination
    for (k = 0; k < N; k++) {
        #pragma omp parallel for private(i, j) shared(A, B, k) schedule(static)
        for (i = k + 1; i < N; i++) {
            double factor = A[i][k] / A[k][k];  // Safe since each thread works on a different row

            for (j = k; j < N; j++) {
                A[i][j] -= factor * A[k][j];
            }
            B[i] -= factor * B[k];
        }
    }

    // Back Substitution  
    for (i = N - 1; i >= 0; i--) {
        double sum = B[i];

        #pragma omp parallel for reduction(-:sum) schedule(dynamic)
        for (j = i + 1; j < N; j++) {
            sum -= A[i][j] * X[j];
        }

        X[i] = sum / A[i][i];  // No critical section needed
    }
}

// Function to allocate a 2D matrix dynamically
double** allocateMatrix(int size) {
    double** matrix = (double**)malloc(size * sizeof(double*));
    for (int i = 0; i < size; i++) {
        matrix[i] = (double*)malloc(size * sizeof(double));
    }
    return matrix;
}

// Function to free allocated memory for a 2D matrix
void freeMatrix(double** matrix, int size) {
    for (int i = 0; i < size; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

int main() {
    double** A = allocateMatrix(N);
    double* B = (double*)malloc(N * sizeof(double));
    double* X = (double*)malloc(N * sizeof(double));

    // Initialize matrix A and vector B with random values
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = rand() % 10 + 1;
        }
        B[i] = rand() % 10 + 1;
    }

    // Start execution timer
    double start = omp_get_wtime();
    gaussianElimination(A, B, X);
    double end = omp_get_wtime();

    // Compute execution time
    printf("Parallel Execution Time: %.6f seconds\n", end - start);

    // Free allocated memory
    freeMatrix(A, N);
    free(B);
    free(X);

    return 0;
}