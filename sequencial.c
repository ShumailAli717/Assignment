#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MATRIX_SIZE 1024  // Fixed macro name to avoid conflicts

// Function Prototypes (Fix implicit declaration issue)
double** allocateMatrix(int size);
void freeMatrix(double** matrix, int size);
void gaussianElimination(double** A, double* B, double* X, int N);

// Function to allocate a 2D matrix dynamically
double** allocateMatrix(int size) {
    double** matrix = (double**)malloc(size * sizeof(double*));
    if (matrix == NULL) {
        printf("Memory allocation failed for matrix!\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < size; i++) {
        matrix[i] = (double*)malloc(size * sizeof(double));
        if (matrix[i] == NULL) {
            printf("Memory allocation failed for matrix row!\n");
            exit(EXIT_FAILURE);
        }
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

// Function to perform Gaussian Elimination (Sequential)
void gaussianElimination(double** A, double* B, double* X, int N) {
    for (int k = 0; k < N; k++) {
        if (A[k][k] == 0) {  // Prevent division by zero
            printf("Zero pivot detected at row %d! Aborting.\n", k);
            exit(EXIT_FAILURE);
        }

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

int main() {
    int N = MATRIX_SIZE;  // Use renamed macro
    double** A = allocateMatrix(N);
    double* B = (double*)malloc(N * sizeof(double));
    double* X = (double*)malloc(N * sizeof(double));

    if (B == NULL || X == NULL) {
        printf("Memory allocation failed for vectors!\n");
        exit(EXIT_FAILURE);
    }

    srand(time(NULL));  // Seed for randomness

    // Initialize matrix A and vector B with random values
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = (rand() % 10) + 1;
        }
        B[i] = (rand() % 10) + 1;
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
