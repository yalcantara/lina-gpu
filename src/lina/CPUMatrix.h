/*
 * CPUMatrix.h
 *
 *  Created on: May 16, 2016
 *      Author: yaison
 */

#ifndef CPUMATRIX_H_
#define CPUMATRIX_H_

#include <stdlib.h>

namespace lina{
class CPUMatrix {
private:
	float* grid;
	CPUMatrix(size_t m, size_t n);
	
public:
	const size_t m;
	const size_t n;

	CPUMatrix(size_t m, size_t n, float* grid);
	float* getGridPtr()const;
	const CPUMatrix operator+(const CPUMatrix& b)const;
	const CPUMatrix operator*(const CPUMatrix& b)const;
	float operator()(size_t i, size_t j)const;
	void colStdev(size_t col, double* mean, double* stdev)const;
	const CPUMatrix stdScale()const;
	void print();
	virtual ~CPUMatrix();
};
}
#endif /* CPUMATRIX_H_ */
