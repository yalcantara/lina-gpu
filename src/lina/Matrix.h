/*
 * GPUMatrix.h
 *
 *  Created on: May 16, 2016
 *      Author: yaison
 */

#ifndef GPUMATRIX_H_
#define GPUMATRIX_H_

#include <stdlib.h>


#include <vector>

#include "CPUMatrix.h"
#include "Vector.h"


using namespace std;
namespace lina{




class Matrix {
private:
	float* devPtr;
	size_t m;
	size_t n;

public:
	Matrix();
	Matrix(const Matrix& other);
	Matrix(const CPUMatrix& other);
	Matrix(size_t m, size_t n);
	Matrix(size_t m, size_t n, float* grid);
	
	size_t rows()const;
	size_t cols()const;
	size_t size()const;
	
	const Vector col(size_t idx)const;
	const Matrix cols(size_t from, size_t to) const;
	
	const Matrix rows(size_t from, size_t to)const;
	
	const Matrix select(size_t m1, size_t n1, size_t m2, size_t n2)const;
	void subSelect(size_t m1, size_t n1, size_t m2, size_t n2, float* dest)const;
	
	Matrix& operator=(const Matrix& other);
	const Matrix operator+(const Matrix& b)const;
	const Matrix operator-(const Matrix& b)const;
	
	const Matrix operator*(const Matrix& b)const;
	const Matrix operator*(const float scalar)const;
	
	const Matrix operator^(const float exp) const;
	const Matrix eleMult(const Matrix& b)const;
	const Matrix insertCol(size_t col, float val) const;
	const Matrix insertCol(size_t col, const Vector& val) const;
	
	const Matrix stdScale()const;
	
	void copy(const Matrix& src);
	float sum()const;
	float* getDevPtr()const;
	
	const CPUMatrix cpu()const;
	void print()const;
	virtual ~Matrix();
};



}

const lina::Matrix operator*(float scalar, const lina::Matrix& A);
#endif /* GPUMATRIX_H_ */
