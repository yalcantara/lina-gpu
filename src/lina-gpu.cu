/*
 ============================================================================
 Name        : lina-gpu.cu
 Author      : Yaison Alcantara
 Version     :
 Copyright   : 
 Description : CUDA compute reciprocals
 ============================================================================
 */

#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <locale.h>
#include <sstream>
#include <vector>

#include "lina/utils.h"
#include "lina/CPUMatrix.h"
#include "lina/Matrix.h"
#include "lina/Layer.h"
#include "lina/Network.h"
#include "lina/cudaUtils.cuh"
#include "lina/Grid.h"
#include "lina/Exception.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

using namespace std;
using namespace lina;

void test() {
	size_t n = 1024 * 2;
	
	auto hostA = (float*) malloc(sizeof(float) * n * n);
	auto hostB = (float*) malloc(sizeof(float) * n * n);
	
	for (int i = 0; i < (n * n); i++) {
		hostA[i] = rand() / (float) RAND_MAX;
		hostB[i] = rand() / (float) RAND_MAX;
	}
	
	auto A = Matrix(n, n, hostA);
	auto B = Matrix(n, n, hostB);
	
	clock_t start = clock();
	auto C = A * B;
	C.cpu();
	int took = (int) (((clock() - start) / (float) CLOCKS_PER_SEC) * 1000);
	
	printf("GPU took: %'6d (ms)\n", took);
	
	auto Acpu = A.cpu();
	auto Bcpu = B.cpu();
	start = clock();
	auto Ccpu = Acpu * Bcpu;
	took = (int) (((clock() - start) / (float) CLOCKS_PER_SEC) * 1000);
	
	printf("CPU took: %'6d (ms)\n", took);
}

const Matrix call() {
	float arr[6] = { -10, 20, -20, 0, 1, 10 };
	Matrix a = Matrix(2, 2, arr);
	
	return a;
}

Layer layer1() {
	float biasArr[] = { -30, 10 };
	float params[] = { 20, 20, -20, -20 };
	
	Matrix w = Matrix(2, 2, params);
	Matrix t = transp(w);
	
	Vector bias = Vector(2, biasArr);
	
	Layer l = Layer(2, 2);
	l.setWeights(t);
	l.setBias(bias);
	
	return l;
}

Layer layer2() {
	float biasArr[] = { -10 };
	float params[] = { 20, 20 };
	
	Matrix w = Matrix(1, 2, params);
	Matrix t = transp(w);
	
	Vector bias = Vector(1, biasArr);
	
	Layer l = Layer(2, 1);
	l.setWeights(t);
	l.setBias(bias);
	
	return l;
}

void net() {
	float raw[] = { 0, 0, 0, 1, 1, 0, 1, 1 };
	
	Matrix input = Matrix(4, 2, raw);
	
	Layer l1 = layer1();
	Layer l2 = layer2();
	
	printf("Creating network\n");
	Network n = Network(2);
	printf("Creating network\n");
	n[0] = l1;
	n[1] = l2;
	
	Matrix ans = n.activate(input);
	ans.print();
	
	float j = n.j(input, ans);
	
	printf("\n\n%12.8f\n\n", j);
	fflush(stdout);
}

void testAdd() {
	lina::init();
	setlocale(LC_NUMERIC, "");
	
	srand(time(nullptr));
	
	size_t n = 10000;
	float* a = (float*) malloc(sizeof(float) * n * n);
	
	float* b = (float*) malloc(sizeof(float) * n * n);
	
	//rand_fill(n * n, a, -1, 1);
	//rand_fill(n * n, b, -1, 1);
	printf("Random finish\n");
	fflush(stdout);
	
	Matrix x = Matrix(n, n, a);
	Matrix y = Matrix(n, n, b);
	Matrix z = Matrix(n, n);
	float alpha = 1;
	const float* xptr = x.getDevPtr();
	const float* yptr = y.getDevPtr();
	float* zptr = z.getDevPtr();
	
	//warm
	for (int i = 0; i < 10; i++) {
		cublasScopy(cublas_handle, n * n, yptr, 1, zptr, 1);
		cublasSaxpy(cublas_handle, n * n, &alpha, xptr, 1, zptr, 1);
		matrixAdd(x.rows(), x.cols(), x.getDevPtr(), y.getDevPtr(), z.getDevPtr());
		z.cpu();
	}
	
	printf("Test finish\n");
	fflush(stdout);
	
	clock_t now;
	unsigned int iter = 100;
	float took;
	
	//=======================================================
	now = clock();
	for (int i = 0; i < iter; i++) {
		matrixAdd(x.rows(), x.cols(), x.getDevPtr(), y.getDevPtr(), z.getDevPtr());
	}
	z.cpu();
	took = (((clock() - now) / (float) CLOCKS_PER_SEC) * 1000);
	
	printf("\n");
	printf("Naive Took : %10.4f (ms)\n\n", took);
	//=======================================================
	
	//=======================================================
	
	now = clock();
	
	for (int i = 0; i < iter; i++) {
		cublasScopy(cublas_handle, n * n, yptr, 1, zptr, 1);
		cublasSaxpy(cublas_handle, n * n, &alpha, xptr, 1, zptr, 1);
	}
	z.cpu();
	took = (((clock() - now) / (float) CLOCKS_PER_SEC) * 1000);
	
	printf("CUBLAS Took: %10.4f (ms)\n\n", took);
	//=======================================================
}

void printVector(vector<string>& v) {
	
	for (int i = 0; i < v.size(); i++) {
		cout << v.at(i) << " ";
	}
	
}

int columns(string& data) {
	long long idx = data.find('\n', 0);
	if (idx == -1) {
		throw Exception("Could not find the '\\n' character.");
	}
	
	int start = 0;
	int length = idx - start;
	string line = data.substr(start, length);
	
	if (length == 0) {
		throw Exception("The first line is empty.");
	}
	
	int counter = 0;
	istringstream iss(line);
	string token;
	while (getline(iss, token, ',')) {
		counter++;
	}
	
	return counter;
}

void test1() {
	srand(time(NULL));
	string s = ffull("files/iris.data");
	
	Grid g(s);
	g.shuffle();
	g.print();
	
	cout << endl;
	cout << endl;
	
	Matrix X = g.toMatrix(0, (size_t) 4).stdScale();
	
	Matrix Y = g.toMatrix(4);
	
	Network n(2, 4, 3);
	n.rand();
	
	int iter = 10000;
	
	cout << endl;
	n.rand();
	for (int i = 0; i <= iter; i++) {
		n.gdescent(X, Y);
		
		if (i % (iter / 10) == 0) {
			float j = n.j(X, Y);
			printf("iter: %6d        j:  %12.8f\n", i, j);
		}
	}
}

void execute(Matrix& X, Matrix& Y, int iter) {
	Network n(3, X.cols(), Y.cols());
	
	n.rand();
	fflush(stdout);
	for (int i = 0; i <= iter; i++) {
		n.gdescent(X, Y);
		if (iter < 10 || i % (iter / 10) == 0) {
			float j = n.j(X, Y);
			printf("iter: %6d        j:  %12.8f\n", i, j);
		}
	}
	
	CPUMatrix out = n.activate(X).insertCol(0, Y.col(0)).cpu();
	
	size_t m = Y.rows();
	size_t hit = 0;
	size_t wrong = 0;
	size_t miss = 0;
	float confident = 0.5;
	for (size_t i = 0; i < m; i++) {
		float expected = out(i, 0);
		float ans = out(i, 1);
		
		if (expected == 1.0) {
			if (ans >= confident) {
				hit++;
			} else if (ans < (1 - confident)) {
				wrong++;
			} else {
				miss++;
			}
		} else if (expected == 0.0) {
			if (ans < (1 - confident)) {
				hit++;
			} else if (ans >= confident) {
				wrong++;
			} else {
				miss++;
			}
		} else {
			wrong++;
		}
		
		//printf("%d   -  %0.2f\n", (int) expected, ans);
	}
	
	printf("\n hit  : %6d", (int) hit);
	printf("\n wrong: %6d", (int) wrong);
	printf("\n miss : %6d", (int) miss);
	
	cout << endl;
	cout << endl;
	size_t total = hit + wrong;
	printf("\n hit   %: %6d%% ", (int) (hit / (float) total * 100));
	printf("\n wrong %: %6d%%", (int) (wrong / (float) total * 100));
}

void test2() {
	srand(time(NULL));
	string s = ffull("files/adult.data");
	
	Grid g(s);
	g.shuffle();
	//g.print();
	
	Matrix X = g.toMatrix(0, 13, true);
	Matrix Y = g.toMatrix(14).cols(0, 1);
	
	execute(X, Y, 300);
}

void test3() {
	srand(time(NULL));
	string s = ffull("files/cancer.data");
	
	Grid g(s);
	g.shuffle();
	g.replace(10, "2", "0");
	g.replace(10, "4", "1");
	
	
	Matrix X = g.toMatrix(0, 10, true);
	Matrix Y = g.toMatrix(10).cols(0, 1);
	
	execute(X, Y, 10000);
}

void test4() {
	srand(time(NULL));
	string s = ffull("files/winequality-white.data");
	
	Grid g(s, ';');
	g.shuffle();
	g.replace(11, "0", "0");
	g.replace(11, "1", "0");
	g.replace(11, "2", "0");
	g.replace(11, "3", "0");
	g.replace(11, "4", "0");
	g.replace(11, "5", "0");
	g.replace(11, "6", "0");
	g.replace(11, "7", "0");
	g.replace(11, "8", "0");
	g.replace(11, "9", "1");
	g.replace(11, "10", "0");
	
	g.print();
	
	Matrix X = g.toMatrix(0, 11, true);
	Matrix Y = g.toMatrix(11).cols(0, 1);
	
	execute(X, Y, 1000);
}

int main(void) {
	lina::init();
	
	test4();
	
	return 0;
}

