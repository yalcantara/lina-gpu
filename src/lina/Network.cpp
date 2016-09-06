/*
 * Network.cpp
 *
 *  Created on: May 21, 2016
 *      Author: yaison
 */

#include "Network.h"
#include "utils.h"

namespace lina {

Network::Network(size_t depth) :
		depth(depth) {
	layers = (Layer*) calloc(depth, sizeof(Layer));
}

Network::Network(size_t depth, size_t in, size_t out) :
		depth(depth) {
	layers = (Layer*) calloc(depth, sizeof(Layer));
	
	if (depth == 1) {
		layers[0] = Layer(in, out);
		return;
	}
	
	layers[0] = Layer(in, in * 2);
	
	for (size_t i = 1; i < depth - 1; i++) {
		layers[i] = Layer(in * 2, in * 2);
	}
	
	layers[depth - 1] = Layer(in * 2, out);
}

void Network::rand() {
	for (int i = 0; i < depth; i++) {
		Layer& l = layers[i];
		l.rand();
	}
}

size_t Network::out() const {
	return layers[depth - 1].out();
}

size_t Network::params() const {
	
	size_t count = 0;
	
	for (int i = 0; i < depth; i++) {
		count += layers[i].params();
	}
	
	return count;
}

void Network::update(vector<Matrix>& grad) {
	for (size_t i = 0; i < depth; i++) {
		Layer& L = layers[i];
		
		Matrix& g = grad[i];
		L.update(g);
	}
}

const Matrix Network::activate(const Matrix& x) const {
	
	size_t L = depth;
	
	Matrix a = layers[0].activate(x);
	
	for (size_t i = 1; i < L; i++) {
		Layer& li = layers[i];
		
		a = li.activate(a);
	}
	
	return a;
}

void Network::sgdescent(Matrix& X, Matrix& Y) {
	size_t m = Y.rows();
	
	for (size_t i = 0; i < m; i += 30) {
		if (i + 30 < m) {
			Matrix x = X.rows(i, i + 30);
			Matrix y = Y.rows(i, i + 30);
			
			gdescent(x, y);
		}
	}
}

void Network::gdescent(Matrix& X, Matrix& Y) {
	vector<Matrix> grad = backprop(X, Y);
	update(grad);
}

vector<Matrix> Network::backprop(Matrix& X, Matrix& Y) const {
	
	const int L = depth;
	
	const size_t m = Y.rows();
	
	Layer& crt = layers[0];
	
	//Last layer A - Y
	vector<Matrix> A(L + 1);
	
	const int ALength = L + 1;
	
	//=========================================================================
	//A[0] must be equal to X
	A[0] = X;
	// our layers starts with the hidden layer having 1 less index than the
	// A (a) array.
	for (int l = 0; l < L; l++) {
		A[l + 1] = layers[l].activate(A[l]);
	}
	
	//=========================================================================
	// for a' (adx) we don't need the last layer, making indexing a bit awkward.
	vector<Matrix> Adx(L - 1);
	for (int l = 0; l < L - 1; l++) {
		Adx[l] = layers[l].gradient(A[l]);
	}
	
	//=========================================================================
	//δ of the last layer is just A - Y
	vector<Matrix> E(L);
	E[L - 1] = A[ALength - 1] - Y;
	//let's go backwards, starting at L-2 because we already
	//set L-1 to the error array.
	for (int l = L - 2; l >= 0; l--) {
		Matrix& e = E[l + 1];
		const Matrix W = layers[l + 1].getWeights();
		
		E[l] = (e * transp(W)).eleMult(Adx[l]);
		
	}
	
	//=========================================================================
	vector<Matrix> grad(L);
	for (int l = 0; l < L; l++) {
		
		//The math resolves to: Δ = (1 / m) * δ(trans) * [1 a]
		grad[l] = (1.0 / m) * transp(E[l]) * A[l].insertCol(0, 1);
		
	}
	
	return grad;
}

float Network::j(const Matrix& x, const Matrix& y) const {
	//h for hypothesis
	Matrix h = activate(x);
	const size_t m = y.rows();
	
	float cost = 1.0 / (2 * m) * sum((h - y) ^ 2);
	
	return cost;
	
}

Layer& Network::operator[](unsigned int idx) {
	return layers[idx];
}

void Network::print() const {
	
	for (size_t i = 0; i < depth; i++) {
		layers[i].print();
	}
}

Network::~Network() {
	if (layers) {
		free(layers);
	}
}

} /* namespace lina */
