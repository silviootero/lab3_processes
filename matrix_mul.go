package main

import (
	"fmt"
	"os"
	"time"
	"strconv"
	"io/ioutil"
	"strings"
	"syscall"
	"unsafe"
)

// Estructura para representar una matriz
type Matrix struct {
	Rows int
	Cols int
	Data [][]int
}

// Función para leer una matriz desde un archivo
func readMatrixFromFile(filename string) Matrix {
	content, err := ioutil.ReadFile(filename)
	if err != nil {
		panic(err)
	}

	lines := strings.Split(string(content), "\n")
	var rows, cols int
	fmt.Sscanf(lines[0], "%d %d", &rows, &cols)

	matrix := make([][]int, rows)
	for i := 0; i < rows; i++ {
		matrix[i] = make([]int, cols)
		values := strings.Fields(lines[i+1])
		for j := 0; j < cols; j++ {
			matrix[i][j], _ = strconv.Atoi(values[j])
		}
	}

	return Matrix{Rows: rows, Cols: cols, Data: matrix}
}

// Función para escribir una matriz en un archivo
func writeMatrixToFile(filename string, matrix Matrix) {
	file, err := os.Create(filename)
	if err != nil {
		panic(err)
	}
	defer file.Close()

	fmt.Fprintf(file, "%d %d\n", matrix.Rows, matrix.Cols)
	for i := 0; i < matrix.Rows; i++ {
		for j := 0; j < matrix.Cols; j++ {
			fmt.Fprintf(file, "%d ", matrix.Data[i][j])
		}
		fmt.Fprintln(file)
	}
}

// Multiplicación secuencial de matrices
func sequentialMatrixMultiply(A, B Matrix) Matrix {
	if A.Cols != B.Rows {
		panic("Las dimensiones de las matrices no son compatibles para multiplicación")
	}

	C := make([][]int, A.Rows)
	for i := 0; i < A.Rows; i++ {
		C[i] = make([]int, B.Cols)
		for j := 0; j < B.Cols; j++ {
			sum := 0
			for k := 0; k < A.Cols; k++ {
				sum += A.Data[i][k] * B.Data[k][j]
			}
			C[i][j] = sum
		}
	}

	return Matrix{Rows: A.Rows, Cols: B.Cols, Data: C}
}

// Multiplicación paralela usando memoria compartida
func parallelMatrixMultiply(A, B Matrix, numProcesses int) Matrix {
	if A.Cols != B.Rows {
		panic("Las dimensiones de las matrices no son compatibles para multiplicación")
	}

	// Crear memoria compartida para la matriz C
	size := A.Rows * B.Cols * int(unsafe.Sizeof(int(0)))
	shmid, _, err := syscall.Syscall6(
		syscall.SYS_SHMGET,
		uintptr(0),
		uintptr(size),
		uintptr(0666|syscall.IPC_CREAT),
		0, 0, 0,
	)
	if err != 0 {
		panic("shmget failed")
	}

	sharedC, _, err := syscall.Syscall(
		syscall.SYS_SHMAT,
		shmid,
		0,
		0,
	)
	if err != 0 {
		panic("shmat failed")
	}

	// Convertir matrices a slices planos para facilitar el manejo
	flatA := make([]int, A.Rows*A.Cols)
	flatB := make([]int, B.Rows*B.Cols)

	for i := 0; i < A.Rows; i++ {
		for j := 0; j < A.Cols; j++ {
			flatA[i*A.Cols+j] = A.Data[i][j]
		}
	}

	for i := 0; i < B.Rows; i++ {
		for j := 0; j < B.Cols; j++ {
			flatB[i*B.Cols+j] = B.Data[i][j]
		}
	}

	// Crear procesos hijos
	for pid := 0; pid < numProcesses; pid++ {
		childPid, err := syscall.ForkExec(os.Args[0], os.Args, &syscall.ProcAttr{
			Sys: &syscall.SysProcAttr{
				Setsid: true,
			},
		})
		if err != nil {
			panic(err)
		}

		if childPid == 0 { // Proceso hijo
			startRow := pid * (A.Rows / numProcesses)
			endRow := (pid + 1) * (A.Rows / numProcesses)
			if pid == numProcesses-1 {
				endRow = A.Rows
			}

			// Convertir el puntero de memoria compartida a slice
			cSlice := (*[1 << 30]int)(unsafe.Pointer(sharedC))[:A.Rows*B.Cols]

			// Realizar la multiplicación para las filas asignadas
			for i := startRow; i < endRow; i++ {
				for j := 0; j < B.Cols; j++ {
					sum := 0
					for k := 0; k < A.Cols; k++ {
						sum += flatA[i*A.Cols+k] * flatB[k*B.Cols+j]
					}
					cSlice[i*B.Cols+j] = sum
				}
			}

			os.Exit(0)
		}
	}

	// Esperar a que todos los procesos hijos terminen
	for i := 0; i < numProcesses; i++ {
		var status syscall.WaitStatus
		_, err := syscall.Wait4(-1, &status, 0, nil)
		if err != nil {
			panic(err)
		}
	}

	// Convertir el resultado de memoria compartida a matriz
	result := make([][]int, A.Rows)
	for i := 0; i < A.Rows; i++ {
		result[i] = make([]int, B.Cols)
		for j := 0; j < B.Cols; j++ {
			result[i][j] = *(*int)(unsafe.Pointer(sharedC + uintptr(i*B.Cols+j)*unsafe.Sizeof(int(0))))
		}
	}

	// Liberar memoria compartida
	syscall.Syscall(syscall.SYS_SHMDT, sharedC, 0, 0)
	syscall.Syscall(syscall.SYS_SHMCTL, shmid, syscall.IPC_RMID, 0)

	return Matrix{Rows: A.Rows, Cols: B.Cols, Data: result}
}

func main() {
	if len(os.Args) != 4 {
		fmt.Println("Uso: ./matrix_mul <archivo_matriz_A> <archivo_matriz_B> <num_procesos>")
		os.Exit(1)
	}

	fileA := os.Args[1]
	fileB := os.Args[2]
	numProcesses, err := strconv.Atoi(os.Args[3])
	if err != nil {
		panic(err)
	}

	// Leer matrices de entrada
	A := readMatrixFromFile(fileA)
	B := readMatrixFromFile(fileB)

	// Multiplicación secuencial
	start := time.Now()
	C_seq := sequentialMatrixMultiply(A, B)
	seqTime := time.Since(start).Seconds()

	// Multiplicación paralela
	start = time.Now()
	C_par := parallelMatrixMultiply(A, B, numProcesses)
	parTime := time.Since(start).Seconds()

	// Verificar que los resultados sean iguales
	for i := 0; i < A.Rows; i++ {
		for j := 0; j < B.Cols; j++ {
			if C_seq.Data[i][j] != C_par.Data[i][j] {
				fmt.Println("Error: Los resultados secuencial y paralelo no coinciden")
				os.Exit(1)
			}
		}
	}

	// Escribir resultado en archivo
	writeMatrixToFile("C.txt", C_seq)

	// Mostrar tiempos y speedup
	fmt.Printf("Tiempo secuencial: %.6f segundos\n", seqTime)
	fmt.Printf("Tiempo paralelo (%d procesos): %.6f segundos\n", numProcesses, parTime)
	fmt.Printf("Speedup: %.2fx\n", seqTime/parTime)
}