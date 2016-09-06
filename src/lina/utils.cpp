/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <algorithm>

#include "utils.h"
#include "cudaUtils.cuh"
#include "Matrix.h"
#include "Exception.h"

using namespace std;
using namespace lina;

cublasHandle_t cublas_handle;
Func SIGMOID = Func(sigmoid_fx, sigmoid_dx);
size_t MATRIX_PRINT_MAX = 100;
size_t VECTOR_PRINT_MAX = 300;

static const char *_cudaGetErrorEnum(cublasStatus_t error) {
	switch (error) {
	case CUBLAS_STATUS_SUCCESS:
		return "CUBLAS_STATUS_SUCCESS";
		
	case CUBLAS_STATUS_NOT_INITIALIZED:
		return "CUBLAS_STATUS_NOT_INITIALIZED";
		
	case CUBLAS_STATUS_ALLOC_FAILED:
		return "CUBLAS_STATUS_ALLOC_FAILED";
		
	case CUBLAS_STATUS_INVALID_VALUE:
		return "CUBLAS_STATUS_INVALID_VALUE";
		
	case CUBLAS_STATUS_ARCH_MISMATCH:
		return "CUBLAS_STATUS_ARCH_MISMATCH";
		
	case CUBLAS_STATUS_MAPPING_ERROR:
		return "CUBLAS_STATUS_MAPPING_ERROR";
		
	case CUBLAS_STATUS_EXECUTION_FAILED:
		return "CUBLAS_STATUS_EXECUTION_FAILED";
		
	case CUBLAS_STATUS_INTERNAL_ERROR:
		return "CUBLAS_STATUS_INTERNAL_ERROR";
	}
	
	return "<unknown>";
}

void checkCublas(cublasStatus_t status) {
	if (status == CUBLAS_STATUS_SUCCESS) {
		return;
	}
	
	fprintf(stderr, "CUBLAS error %d\nMessage: %s.\n", status, _cudaGetErrorEnum(status));
	fflush(stderr);
	throw Exception("Cuda error");
}

void checkCuda(cudaError_t error) {
	if (error == cudaSuccess) {
		return;
	}
	
	fprintf(stderr, "CUDA error: %d %s\nMessage: %s.\n", error, cudaGetErrorName(error),
			cudaGetErrorString(error));
	fflush(stderr);
	throw Exception("Cuda error");
}

/**
 * Convenient methods
 */
const Matrix transp(const Matrix& a) {
	
	size_t m = a.rows();
	size_t n = a.cols();
	
	Matrix t = Matrix(n, m);
	float* src = a.getDevPtr();
	float* dest = t.getDevPtr();
	
	matrixTranspose(m, n, src, dest);
	
	return t;
}

float sum(const Matrix& a) {
	return a.sum();
}

/*
 * Returns the value of a if a > b, and b if b > a. Note that for this method
 * any value is higher than NAN.
 */
float higher(float a, float b) {
	if (isnan(a)) {
		if (isnan(b)) {
			return NAN;
		}
		
		return b;
	}
	
	if (isnan(b)) {
		return a;
	}
	
	if (a > b) {
		return a;
	}
	
	return b;
}

/*
 * Returns the value of a if a < b, and b if b < a. Note that for this method
 * any value is lower than NAN.
 */
float lower(float a, float b) {
	if (isnan(a)) {
		if (isnan(b)) {
			return NAN;
		}
		
		return b;
	}
	
	if (isnan(b)) {
		return a;
	}
	
	if (a < b) {
		return a;
	}
	
	return b;
}

const Matrix sigmoid_fx(const Matrix& x, const Vector& bias) {
	
	unsigned int m = x.rows();
	unsigned int n = x.cols();
	float* src = x.getDevPtr();
	
	Matrix ans = Matrix(m, n);
	
	float* dest = ans.getDevPtr();
	
	float* biasPtr = bias.getDevPtr();
	
	applySigmoidFX(m, n, src, dest, biasPtr);
	
	return ans;
}

const Matrix sigmoid_dx(const Matrix& x, const Vector& bias) {
	
	unsigned int m = x.rows();
	unsigned int n = x.cols();
	float* src = x.getDevPtr();
	
	Matrix ans = Matrix(m, n);
	
	float* dest = ans.getDevPtr();
	
	float* biasPtr = bias.getDevPtr();
	
	applySigmoidDX(m, n, src, dest, biasPtr);
	
	return ans;
}

const Matrix randMatrix(size_t m, size_t n) {
	return randMatrix(m, n, -1.0, 1.0);
}

const Matrix randMatrix(size_t m, size_t n, float min, float max) {
	float _min = lower(min, max);
	float _max = higher(min, max);
	
	size_t l = m * n;
	vector<float> vals(l);
	
	for (size_t i = 0; i < l; i++) {
		float r = (rand() / (float) RAND_MAX) * (_max - _min) + _min;
		vals[i] = r;
	}
	
	float* data = vals.data();
	
	Matrix mtr(m, n, data);
	
	return mtr;
}

const Vector randVector(size_t l) {
	return randVector(l, -1.0, 1.0);
}

const Vector randVector(size_t l, float min, float max) {
	float _min = lower(min, max);
	float _max = higher(min, max);
	
	vector<float> vals(l);
	
	for (size_t i = 0; i < l; i++) {
		float r = (rand() / (float) RAND_MAX) * (_max - _min) + _min;
		vals[i] = r;
	}
	
	Vector v(l, vals.data());
	
	return v;
}

//IO
const string ffull(const char* path) {
	
	FILE* f = fopen(path, "r");
	
	if (f == NULL) {
		fprintf(stderr, "file not found at: %s\n", path);
		fflush(stderr);
		return NULL;
	}
	
	char c;
	
	//A trick to determine how many characters a file has
	size_t count = 0;
	while ((c = fgetc(f))) {
		if (c == EOF) {
			break;
		}
		count++;
	}
	
	char* content = (char*) malloc(sizeof(char) * (count + 1));
	content[count] = 0;
	
	rewind(f);
	size_t r = fread(content, sizeof(char), count, f);
	
	fclose(f);
	
	const string s(content);
	
	return s;
	
}

//API
void rand_fill(size_t n, float* src, float min, float max) {
	
	for (size_t i = 0; i < n; i++) {
		src[i] = (rand() / (float) RAND_MAX) * (max - min) + min;
	}
}

long int indexOf(const vector<string>& src, string& val) {
	auto r = find(src.begin(), src.end(), val);
	if (r == src.end()) {
		return -1;
	}
	
	long int pos = distance(src.begin(), r);
	return pos;
}

bool contains(const vector<string>& src, string& val) {
	return indexOf(src, val) >= 0;
}

long int stringToInt(string str) {
	return atol(str.c_str());
}
float stringToFloat(string str) {
	return atof(str.c_str());
}

void print(vector<string>& vec) {
	
	cout << "[";
	for (size_t i = 0; i < vec.size(); i++) {
		cout << vec[i];
		if (i < vec.size()) {
			cout << ", ";
		}
	}
	cout << "]";
}

string trim(string const& str) {
	if (str.empty())
		return str;
	
	size_t firstScan = str.find_first_not_of(' ');
	size_t first = firstScan == string::npos ? str.length() : firstScan;
	size_t last = str.find_last_not_of(' ');
	return str.substr(first, last - first + 1);
}

namespace lina {
void init() {
	checkCublas(cublasCreate(&cublas_handle));
}
}
