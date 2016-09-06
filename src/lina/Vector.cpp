/*
 * Vector.cpp
 *
 *  Created on: May 20, 2016
 *      Author: yaison
 */

#include <stdlib.h>

#include <string>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "Vector.h"
#include "utils.h"
#include "Exception.h"
#include "cudaUtils.cuh"

using namespace lina;
using namespace std;

Vector::Vector(const Vector& other) :
		length(other.length) {
	if (length < 1) {
		throw Exception("Invalid vector length " + to_string(length) + ".");
	}
	checkCuda(cudaMalloc(&devPtr, sizeof(float) * length));
	copyVector(length, other.devPtr, devPtr);
}

Vector::Vector(size_t length) :
		length(length) {
	if (length < 1) {
		throw Exception("Invalid vector length " + to_string(length) + ".");
	}
	checkCuda(cudaMalloc(&devPtr, sizeof(float) * length));
	checkCuda(cudaMemset(devPtr, 0, sizeof(float) * length));
}

Vector::Vector(size_t length, float* vals) :
		length(length) {
	if (length < 1) {
		throw Exception("Invalid vector length " + to_string(length) + ".");
	}
	checkCuda(cudaMalloc(&devPtr, sizeof(float) * length));
	checkCuda(cudaMemcpy(devPtr, vals, sizeof(float) * length, cudaMemcpyHostToDevice));
}

size_t Vector::size() const {
	return length;
}

float* Vector::getDevPtr() const {
	return devPtr;
}

const CPUVector Vector::cpu() const {
	CPUVector c = CPUVector(length);
	
	float* dst = c.getHostPtr();
	float* src = devPtr;
	
	checkCuda(cudaMemcpy(dst, src, sizeof(float) * length, cudaMemcpyDeviceToHost));
	
	return c;
}

void Vector::print()const{
	
	const CPUVector c = cpu();
	c.print();
}

Vector& Vector::operator=(const Vector& other) {
	//There are 2 case:
	//a  the current object is uninitialized
	//b. the vectors have the same dimension length (no need to allocate memory).
	//c. the dimension are different (we must free and then allocate again).
	
	if (devPtr == nullptr) {
		this->length = other.length;
		checkCuda(cudaMalloc(&devPtr, sizeof(float) * length));
		copyVector(length, other.devPtr, devPtr);
	} else if (length == other.length) {
		copyVector(length, other.devPtr, devPtr);
	} else {
		//Idiom to handle self assignment operator: allocate first, deallocate later.
		this->length = other.length;
		float* tmpPtr;
		checkCuda(cudaMalloc(&tmpPtr, sizeof(float) * length));
		copyVector(length, other.devPtr, tmpPtr);
		
		checkCuda(cudaFree(devPtr));
		devPtr = tmpPtr;
	}
	
	return *this;
}

const Vector Vector::operator-(const Vector& b) const {
	
	if (length != b.length) {
		throw Exception(
				"Invalid vector dimension for subtraction. This vector is " + to_string(length) + " and the other one is " + to_string(b.length) + ".");
	}
	
	Vector c = Vector(length);
	
	const float alpha = -1;
	
	checkCublas(cublasScopy(cublas_handle, length, devPtr, 1, c.devPtr, 1));
	checkCublas(cublasSaxpy(cublas_handle, length, &alpha, b.devPtr, 1, c.devPtr, 1));
	
	return c;
}

void Vector::copy(const Vector& src) {
	if (length != src.length) {
		throw Exception(
				"The dimensions does not match. This vector's length is " + to_string(length)
						+ " and the other " + to_string(src.length) + ".");
	}
	
	copyVector(length, src.devPtr, devPtr);
}

Vector::~Vector() {
	if (devPtr) {
		checkCuda(cudaFree(devPtr));
	}
}

