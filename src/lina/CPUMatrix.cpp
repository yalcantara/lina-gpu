/*
 * CPUMatrix.cpp
 *
 *  Created on: May 16, 2016
 *      Author: yaison
 */

#include "CPUMatrix.h"

#include <algorithm>

#include "Exception.h"
#include "utils.h"
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstring>

using namespace std;
using namespace lina;

CPUMatrix::CPUMatrix(size_t m, size_t n) :
		m(m), n(n), grid(nullptr) {
	grid = (float*) calloc(m*n, sizeof(float));
}

CPUMatrix::CPUMatrix(size_t m, size_t n, float* ptr) :
		m(m), n(n) {
	
	grid = (float*) malloc(sizeof(float) * m * n);
	memcpy(grid, ptr, sizeof(float) * m * n);
}

float* CPUMatrix::getGridPtr()const{
	return grid;
}

const CPUMatrix CPUMatrix::operator*(const CPUMatrix& b) const {
	
	size_t m = this->m;
	size_t n = this->n;
	size_t p = b.n;
	
	const float* A = this->grid;
	const float* B = b.grid;
	
	float* C = (float*) calloc(m * p, sizeof(float));
	
	for (size_t i = 0; i < m; i++) {
		for (size_t j = 0; j < n; j++) {
			
			const float pivot = A[i * n + j];
			
			for (size_t k = 0; k < p; k++) {
				C[i * p + k] += pivot * B[j * p + k];
			}
		}
	}
	
	CPUMatrix ans = CPUMatrix(m, p);
	ans.grid = C;
	
	return ans;
}

float CPUMatrix::operator()(size_t i, size_t j) const {
	if (i < m && j < n) {
		return grid[i * n + j];
	}
	
	throw Exception(
			"Index (0 based) out of bounds  (" + to_string(i) + ", " + to_string(j)
					+ ") for matrix   " + to_string(m) + "x" + to_string(n) + ".");
	
}

void CPUMatrix::colStdev(size_t col, double* mean, double* stdev)const{
	
	double n = 0;
	double _mean = 0.0;
	double m2 = 0.0;
	
	for (int i = 0; i < m; i++) {
		n++;
		double x = (*this)(i, col);
		double delta = x - _mean;
		_mean += delta / n;
		m2 += delta * (x - _mean);
		
	}
	*mean =  _mean;
	*stdev =  (m2 / (n - 1));
}

const CPUMatrix CPUMatrix::stdScale()const{
	
	vector<double> stdVec(n);
	vector<double> meanVec(n);
	
	for(size_t j=0; j < n; j++){
		double stdev;
		double mean;
		
		colStdev(j, &mean, &stdev);
		
		stdVec[j] = stdev;
		meanVec[j] = mean;
	}
	const CPUMatrix c(m, n);
	for(size_t i =0; i < m; i++){
		for(size_t j=0; j < n; j++){
			double x = (*this)(i, j);
			double stdev = stdVec[j];
			double mean = meanVec[j];
			
			c.grid[i*n+j] = (double)((x - mean) / stdev);
		}
	}
	
	
	return c;
}

void CPUMatrix::print() {
	
	size_t rows = min((size_t) MATRIX_PRINT_MAX, m);
	size_t cols = min((size_t) MATRIX_PRINT_MAX, n);
	
	if (m > MATRIX_PRINT_MAX || cols > MATRIX_PRINT_MAX) {
		printf("Matrix  %dx%d   (truncated)\n", (int) m, (int) n);
	} else {
		printf("Matrix  %dx%d\n", (int) m, (int) n);
	}
	for (size_t i = 0; i < rows; i++) {
		for (size_t j = 0; j < cols; j++) {
			printf("%12.4f", grid[i * n + j]);
			if (j + 1 < n) {
				printf("  ");
			}
		}
		printf("\n");
	}
	
	fflush(stdout);
}

CPUMatrix::~CPUMatrix() {
	free(grid);
	//printf("CPU deallocated %dx%d\n", (int)m, (int)n);
}

