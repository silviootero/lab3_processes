#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <sys/shm.h>
#include <sys/wait.h>
#include <unistd.h>

// Función para leer una matriz desde un archivo
int** read_matrix_from_file(const char* filename, int* rows, int* cols) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        perror("Error al abrir el archivo");
        exit(EXIT_FAILURE);
    }

    fscanf(file, "%d %d", rows, cols);
    
    int** matrix = (int**)malloc(*rows * sizeof(int*));
    for (int i = 0; i < *rows; i++) {
        matrix[i] = (int*)malloc(*cols * sizeof(int));
        for (int j = 0; j < *cols; j++) {
            fscanf(file, "%d", &matrix[i][j]);
        }
    }
    
    fclose(file);
    return matrix;
}

// Función para escribir una matriz en un archivo
void write_matrix_to_file(const char* filename, int** matrix, int rows, int cols) {
    FILE* file = fopen(filename, "w");
    if (!file) {
        perror("Error al abrir el archivo");
        exit(EXIT_FAILURE);
    }

    fprintf(file, "%d %d\n", rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            fprintf(file, "%d ", matrix[i][j]);
        }
        fprintf(file, "\n");
    }
    
    fclose(file);
}

// Función para multiplicación secuencial
int** sequential_matrix_multiply(int** A, int** B, int N, int M, int P) {
    int** C = (int**)malloc(N * sizeof(int*));
    for (int i = 0; i < N; i++) {
        C[i] = (int*)malloc(P * sizeof(int));
        for (int j = 0; j < P; j++) {
            C[i][j] = 0;
            for (int k = 0; k < M; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return C;
}

// Función para multiplicación paralela usando memoria compartida
int** parallel_matrix_multiply(int** A, int** B, int N, int M, int P, int num_processes) {
    // Crear memoria compartida para la matriz C
    int shmid = shmget(IPC_PRIVATE, sizeof(int)*N*P, IPC_CREAT | 0666);
    if (shmid == -1) {
        perror("shmget failed");
        exit(EXIT_FAILURE);
    }

    int* shared_C = (int*)shmat(shmid, NULL, 0);
    if (shared_C == (int*)-1) {
        perror("shmat failed");
        exit(EXIT_FAILURE);
    }

    // Convertir matrices 2D a 1D para facilitar el manejo en memoria compartida
    int* flat_A = (int*)malloc(N * M * sizeof(int));
    int* flat_B = (int*)malloc(M * P * sizeof(int));
    
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            flat_A[i*M + j] = A[i][j];
        }
    }
    
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < P; j++) {
            flat_B[i*P + j] = B[i][j];
        }
    }

    // Crear procesos hijos
    for (int pid = 0; pid < num_processes; pid++) {
        if (fork() == 0) { // Proceso hijo
            int start_row = pid * (N / num_processes);
            int end_row = (pid + 1) * (N / num_processes);
            
            // El último proceso se queda con las filas restantes
            if (pid == num_processes - 1) {
                end_row = N;
            }
            
            // Realizar la multiplicación para las filas asignadas
            for (int i = start_row; i < end_row; i++) {
                for (int j = 0; j < P; j++) {
                    shared_C[i*P + j] = 0;
                    for (int k = 0; k < M; k++) {
                        shared_C[i*P + j] += flat_A[i*M + k] * flat_B[k*P + j];
                    }
                }
            }
            
            free(flat_A);
            free(flat_B);
            exit(EXIT_SUCCESS);
        }
    }

    // Esperar a que todos los procesos hijos terminen
    for (int i = 0; i < num_processes; i++) {
        wait(NULL);
    }

    // Convertir la matriz C de 1D a 2D
    int** C = (int**)malloc(N * sizeof(int*));
    for (int i = 0; i < N; i++) {
        C[i] = (int*)malloc(P * sizeof(int));
        for (int j = 0; j < P; j++) {
            C[i][j] = shared_C[i*P + j];
        }
    }

    // Liberar recursos
    free(flat_A);
    free(flat_B);
    shmdt(shared_C);
    shmctl(shmid, IPC_RMID, NULL);

    return C;
}

// Función para medir el tiempo
double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        printf("Uso: %s <archivo_matriz_A> <archivo_matriz_B> <num_procesos>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char* file_A = argv[1];
    const char* file_B = argv[2];
    int num_processes = atoi(argv[3]);

    // Leer matrices de entrada
    int N, M, M2, P;
    int** A = read_matrix_from_file(file_A, &N, &M);
    int** B = read_matrix_from_file(file_B, &M2, &P);

    if (M != M2) {
        printf("Error: Las dimensiones de las matrices no son compatibles para multiplicación\n");
        return EXIT_FAILURE;
    }

    // Multiplicación secuencial
    double start_time = get_time();
    int** C_seq = sequential_matrix_multiply(A, B, N, M, P);
    double seq_time = get_time() - start_time;

    // Multiplicación paralela
    start_time = get_time();
    int** C_par = parallel_matrix_multiply(A, B, N, M, P, num_processes);
    double par_time = get_time() - start_time;

    // Verificar que los resultados sean iguales
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < P; j++) {
            if (C_seq[i][j] != C_par[i][j]) {
                printf("Error: Los resultados secuencial y paralelo no coinciden\n");
                return EXIT_FAILURE;
            }
        }
    }

    // Escribir resultado en archivo
    write_matrix_to_file("C.txt", C_seq, N, P);

    // Mostrar tiempos y speedup
    printf("Tiempo secuencial: %.6f segundos\n", seq_time);
    printf("Tiempo paralelo (%d procesos): %.6f segundos\n", num_processes, par_time);
    printf("Speedup: %.2fx\n", seq_time / par_time);

    // Liberar memoria
    for (int i = 0; i < N; i++) {
        free(A[i]);
        free(C_seq[i]);
        free(C_par[i]);
    }
    free(A);
    free(C_seq);
    free(C_par);
    
    for (int i = 0; i < M; i++) {
        free(B[i]);
    }
    free(B);

    return EXIT_SUCCESS;
}