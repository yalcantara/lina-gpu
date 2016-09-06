/*
 * utils.h
 *
 *  Created on: May 16, 2016
 *      Author: yaison
 */

#ifndef UTILS_H_
#define UTILS_H_


#include <stdlib.h>
#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <string>
#include <vector>

#include "Func.h"
#include "Matrix.h"
#include "Vector.h"

using namespace lina;
using namespace std;

extern cublasHandle_t cublas_handle;
extern Func SIGMOID;
extern size_t MATRIX_PRINT_MAX;
extern size_t VECTOR_PRINT_MAX;

//CUDA
static const char *_cudaGetErrorEnum(cublasStatus_t error);
void checkCublas(cublasStatus_t status);
void checkCuda(cudaError_t error);



//Math
const Matrix transp(const Matrix& a);
float sum(const Matrix& a);
float lower(float a, float b);
float higher(float a, float b);
const Matrix sigmoid_fx(const Matrix& x, const Vector& bias);
const Matrix sigmoid_dx(const Matrix& x, const Vector& bias);
const Matrix randMatrix(size_t m, size_t n);
const Matrix randMatrix(size_t m, size_t n, float min, float max);
const Vector randVector(size_t l);
const Vector randVector(size_t l, float min, float max);

//IO
const string ffull(const char* path);

//API
void rand_fill(size_t n, float* src, float min, float max);
long int indexOf(const vector<string>& src, string& val);
bool contains(const vector<string>& src, string& val);
long int stringToInt(string str);
float stringToFloat(string str);
void print(vector<string>& vec);
string trim(string const& str);


namespace lina{
	void init();
}


#endif /* UTILS_H_ */
