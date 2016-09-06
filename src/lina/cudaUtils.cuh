/*
 * cudaUtils.cuh
 *
 *  Created on: May 19, 2016
 *      Author: yaison
 */

#ifndef CUDAUTILS_CUH_
#define CUDAUTILS_CUH_

void applySigmoidFX(unsigned int m, unsigned int n, float* src, float* dest, float* bias);
void applySigmoidDX(unsigned int m, unsigned int n, float* src, float* dest, float* bias);
void copyVector(unsigned int length, float* src, float* dest);
void matrixAdd(unsigned int m, unsigned int n, float* a, float* b, float* dest);
void matrixEleMult(unsigned int m, unsigned int n, float* a, float* b, float* dest);
void matrixPow(unsigned int m, unsigned int n, float* a, float exp, float* dest);
void matrixSelect(unsigned int m, unsigned int n, unsigned int ms, unsigned int ns, unsigned int me,
		unsigned int ne, float* src, float* dest);
void copyMatrix(unsigned int m, unsigned int n, float* src, float* dest);
void matrixInsertColVector(unsigned int m, unsigned int n, unsigned int col, float* vec, float* src,
		float* dest);
void matrixInsertCol(unsigned int m, unsigned int n, unsigned int col, float val, float* src,
		float* dest);
void matrixTranspose(unsigned int m, unsigned int n, float* src, float* dest);

#endif /* CUDAUTILS_CUH_ */
