/*
 * GPUMatrix.cpp
 *
 *  Created on: May 16, 2016
 *      Author: yaison
 */

#include <stdlib.h>
#include <string>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "Matrix.h"
#include "CPUMatrix.h"
#include "utils.h"
#include "Exception.h"
#include "cudaUtils.cuh"

using namespace lina;
using namespace std;

const Matrix operator*(float scalar, const Matrix& A) {
	return A * scalar;
}

Matrix::Matrix() :
		m(0), n(0), devPtr(nullptr) {
	
}

Matrix::Matrix(const Matrix& other) :
		m(other.m), n(other.n) {
	if (m < 1) {
		throw Exception("Invalid row size " + to_string(m) + ".");
	}
	
	if (n < 1) {
		throw Exception("Invalid column size " + to_string(n) + ".");
	}
	checkCuda(cudaMalloc(&devPtr, sizeof(float) * m * n));
	copyMatrix(m, n, other.devPtr, devPtr);
}

Matrix::Matrix(size_t m, size_t n) :
		m(m), n(n) {
	if (m < 1) {
		throw Exception("Invalid row size " + to_string(m) + ".");
	}
	
	if (n < 1) {
		throw Exception("Invalid column size " + to_string(n) + ".");
	}
	checkCuda(cudaMalloc(&devPtr, sizeof(float) * m * n));
	checkCuda(cudaMemset(devPtr, 0, sizeof(float) * m * n));
}

Matrix::Matrix(const CPUMatrix& other) :
		Matrix(other.m, other.n, other.getGridPtr()) {
	
}

Matrix::Matrix(size_t m, size_t n, float* grid) :
		m(m), n(n) {
	if (m < 1) {
		throw Exception("Invalid row size " + to_string(m) + ".");
	}
	
	if (n < 1) {
		throw Exception("Invalid column size " + to_string(n) + ".");
	}
	checkCuda(cudaMalloc(&devPtr, sizeof(float) * m * n));
	checkCuda(cudaMemcpy(devPtr, grid, sizeof(float) * m * n, cudaMemcpyHostToDevice));
}

size_t Matrix::rows() const {
	return m;
}

size_t Matrix::cols() const {
	return n;
}

size_t Matrix::size() const {
	return m * n;
}

const Vector Matrix::col(size_t idx) const {
	
	const Vector v(m);
	
	subSelect(0, idx, m, idx + 1, v.getDevPtr());
	
	return v;
}

const Matrix Matrix::cols(size_t from, size_t to) const {
	
	const size_t sn = to - from;
	const Matrix mtr(m, sn);
	
	subSelect(0, from, m, to, mtr.devPtr);
	
	return mtr;
}

const Matrix Matrix::rows(size_t from, size_t to) const {
	
	if (from >= to) {
		throw Exception(
				"The from argument must be lower than the to param. Expected < " + to_string(to)
						+ " but got " + to_string(from) + " instead.");
	}
	
	const size_t sm = to - from;
	const Matrix mtr(sm, n);
	
	checkCuda(cudaMemcpy(mtr.devPtr, (devPtr + from*n), sizeof(float) * sm * n, cudaMemcpyDeviceToDevice));
	
	return mtr;
}

const Matrix Matrix::select(size_t m1, size_t n1, size_t m2, size_t n2) const {
	
	const size_t sm = m2 - m1;
	const size_t sn = n2 - n1;
	const Matrix mtr(sm, sn);
	
	subSelect(m1, n1, m2, n2, mtr.devPtr);
	
	return mtr;
}

void Matrix::subSelect(size_t m1, size_t n1, size_t m2, size_t n2, float* dest) const {
	if (m1 >= m2) {
		throw Exception(
				"Invalid argument m1 must be lower than m2. Got m1: " + to_string(m1) + ", m2: "
						+ to_string(m2) + ".");
	}
	
	if (n1 >= n2) {
		throw Exception(
				"Invalid argument n1 must be lower than n2. Got n1: " + to_string(n1) + ", n2: "
						+ to_string(n2) + ".");
	}
	
	if (m2 > m) {
		throw Exception(
				"Selection end is out of bounds. Expected <= " + to_string(m) + ", but got "
						+ to_string(m2) + " instead.");
	}
	
	if (n2 > n) {
		throw Exception(
				"Selection end for columns is out of bounds. Expected <= " + to_string(n)
						+ ", but got " + to_string(n2) + " instead.");
	}
	
	size_t sm = m2 - m1;
	size_t sn = n2 - n1;
	
	matrixSelect(m, n, m1, n1, m2, n2, devPtr, dest);
}

Matrix& Matrix::operator=(const Matrix& other) {
	
	//There are 3 case:
	//a. the current object is uninitialized
	//b. the matrixes have the same dimension mxn (no need to allocate memory).
	//c. the dimension are different (we must free and then allocate again).
	
	if (devPtr == nullptr) {
		this->m = other.m;
		this->n = other.n;
		checkCuda(cudaMalloc(&devPtr, sizeof(float) * m * n));
		copyMatrix(m, n, other.devPtr, devPtr);
	} else if (m == other.m && n == other.n) {
		copyMatrix(m, n, other.devPtr, devPtr);
	} else {
		//Idiom to handle self assignment operator: allocate first, deallocate later.
		this->m = other.m;
		this->n = other.n;
		float* tmpPtr;
		checkCuda(cudaMalloc(&tmpPtr, sizeof(float) * m * n));
		copyMatrix(m, n, other.devPtr, tmpPtr);
		
		checkCuda(cudaFree(devPtr));
		devPtr = tmpPtr;
	}
	
	return *this;
}

const Matrix Matrix::operator+(const Matrix& b) const {
	
	if (m != b.m || n != b.n) {
		throw Exception(
				"Invalid matrix dimension for addition. This matrix is " + to_string(m) + "x"
						+ to_string(n) + " and the other one is " + to_string(b.m) + "x"
						+ to_string(b.n) + ".");
	}
	
	Matrix c = Matrix(m, n);
	
	const float alpha = 1;
	
	checkCublas(cublasScopy(cublas_handle, m * n, devPtr, 1, c.devPtr, 1));
	checkCublas(cublasSaxpy(cublas_handle, m * n, &alpha, b.devPtr, 1, c.devPtr, 1));
	
	return c;
}

const Matrix Matrix::operator-(const Matrix& b) const {
	
	if (m != b.m || n != b.n) {
		throw Exception(
				"Invalid matrix dimension for subtraction. This matrix is " + to_string(m) + "x"
						+ to_string(n) + " and the other one is " + to_string(b.m) + "x"
						+ to_string(b.n) + ".");
	}
	
	Matrix c = Matrix(m, n);
	
	const float alpha = -1;
	
	checkCublas(cublasScopy(cublas_handle, m * n, devPtr, 1, c.devPtr, 1));
	checkCublas(cublasSaxpy(cublas_handle, m * n, &alpha, b.devPtr, 1, c.devPtr, 1));
	
	return c;
}

const Matrix Matrix::operator*(const Matrix& b) const {
	
	if (n != b.m) {
		throw Exception(
				"Invalid matrix dimension for multiplication. This matrix is " + to_string(m) + "x"
						+ to_string(n) + " and the other one is " + to_string(b.m) + "x"
						+ to_string(b.n) + ".");
	}
	
	const int m = this->m;
	const int n = this->n;
	const int p = b.n;
	
	const float alpha = 1.0;
	const float beta = 0.0;
	
	Matrix c = Matrix(m, p);
	
	float* devA = this->devPtr;
	float* devB = b.devPtr;
	float* devC = c.devPtr;
	
	//Since cublas assumes column major, and our structure are row major, we need to change the order.
	checkCublas(
			cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, p, m, n, &alpha, devB, p, devA, n,
					&beta, devC, p));
	
	return c;
}

const Matrix Matrix::operator*(const float scalar) const {
	
	Matrix c = Matrix(m, n);
	checkCublas(cublasScopy(cublas_handle, m * n, devPtr, 1, c.devPtr, 1));
	
	checkCublas(cublasSscal(cublas_handle, m * n, &scalar, c.devPtr, 1));
	
	return c;
}

const Matrix Matrix::operator^(const float exp) const {
	
	Matrix c = Matrix(m, n);
	
	matrixPow(m, n, devPtr, exp, c.devPtr);
	
	return c;
}

const Matrix Matrix::eleMult(const Matrix& b) const {
	Matrix c = Matrix(m, n);
	
	matrixEleMult(m, n, devPtr, b.devPtr, c.devPtr);
	
	return c;
}

const Matrix Matrix::insertCol(size_t col, float val) const {
	Matrix c = Matrix(m, n + 1);
	
	matrixInsertCol(m, n, col, val, devPtr, c.devPtr);
	
	return c;
}

const Matrix Matrix::insertCol(size_t col, const Vector& vec) const {
	Matrix c = Matrix(m, n + 1);
	
	matrixInsertColVector(m, n, col, vec.getDevPtr(), devPtr, c.devPtr);
	
	return c;
}

float* Matrix::getDevPtr() const {
	return devPtr;
}

void Matrix::copy(const Matrix& src) {
	if (m != src.m && n != src.n) {
		throw Exception(
				"The dimensions does not match. This matrix is " + to_string(m) + "x" + to_string(n)
						+ " and the other one is " + to_string(src.m) + "x" + to_string(src.n)
						+ ".");
	}
	
	checkCublas(cublasScopy(cublas_handle, m * n, src.devPtr, 1, devPtr, 1));
}

const Matrix Matrix::stdScale() const {
	
	const CPUMatrix c = cpu();
	const CPUMatrix std = c.stdScale();
	
	const Matrix ret(std);
	
	return ret;
	
}

float Matrix::sum() const {
	
	float val;
	checkCublas(cublasSasum(cublas_handle, n * m, devPtr, 1, &val));
	
	return val;
}

const CPUMatrix Matrix::cpu() const {
	
	vector<float> v(m * n);
	
	float* data = v.data();
	checkCuda(cudaMemcpy(data, devPtr, sizeof(float) * m * n, cudaMemcpyDeviceToHost));
	
	CPUMatrix mtr = CPUMatrix(m, n, data);
	return mtr;
}

void Matrix::print() const {
	CPUMatrix c = cpu();
	c.print();
}

Matrix::~Matrix() {
	if (devPtr) {
		checkCuda(cudaFree(devPtr));
	}
}

