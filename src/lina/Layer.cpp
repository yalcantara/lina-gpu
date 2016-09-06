/*
 * Layer.cpp
 *
 *  Created on: May 18, 2016
 *      Author: yaison
 */

#include "Layer.h"

#include <stdlib.h>
#include <string>
#include "Matrix.h"
#include "CPUMatrix.h"
#include "CPUVector.h"
#include "Exception.h"
#include "utils.h"

using namespace std;

namespace lina {

Layer::Layer(const Layer& other):_in(other._in), _out(other._out), weight(other.weight), bias(other.bias),func(func){
}

Layer::Layer(size_t in, size_t out) :
		_in(in), _out(out), bias(out), weight(in, out), func(SIGMOID) {
}

Layer::Layer(size_t in, size_t out, Func func) :
		_in(in), _out(out), bias(out), weight(in, out), func(func) {
}

void Layer::rand(){
	weight = randMatrix(in(), out());
	bias = randVector(out());
}

void Layer::setFunction(const Func& func){
	this->func = func;
}

void Layer::setWeights(const Matrix& params) {
	
	if (params.rows() != _in) {
		throw Exception(
				"The row size does not match the number of input size. Expected " + to_string(_in)
						+ " but got " + to_string(params.rows()) + " instead.");
	}
	
	if (params.cols() != _out) {
		throw Exception(
				"The column size does not match the number of input size. Expected " + to_string(_in)
						+ " but got " + to_string(params.cols()) + " instead.");
	}
	
	weight.copy(params);
}

const Matrix Layer::getWeights()const{
	const Matrix ret(weight);
	return ret;
}

size_t Layer::in()const{
	return _in;
}

size_t Layer::out()const{
	return _out;
}

size_t Layer::params()const{
	return weight.size() + bias.size();
}

void Layer::update(Matrix& grad){
	const Vector v = grad.col(0);
	
	const Matrix w = grad.cols(1, grad.cols());
	
	bias = bias - v;
	weight = weight - transp(w);
	
}


Layer& Layer::operator=(const Layer& other){
	this->_in = other.in();
	this->_out = other.out();
	
	this->weight = other.weight;
	this->bias = other.bias;
	this->func = other.func;
	
	return *this;
}

void Layer::setBias(const Vector& bias) {
	if (bias.size() != _out) {
		throw Exception(
				"The bias length does not match the number of output size. Expected "
						+ to_string(_in) + " but got " + to_string(bias.size()) + " instead.");
	}
	
	this->bias.copy(bias);
}

const Matrix Layer::activate(const Matrix& x) const {
	Matrix xw = x * weight;
	
	Matrix ans = func.fx(xw, bias);
	
	return ans;
}

const Matrix Layer::gradient(const Matrix& x) const {
	Matrix xw = x * weight;
	
	Matrix ans = func.dx(xw, bias);
	
	return ans;
}

void Layer::print() const {
	CPUMatrix w = weight.cpu();
	CPUVector b = bias.cpu();
	
	printf("\nLayer (%d, %d)\n", (int) _in, (int) _out);
	cout << "Bias:     "; //10 chars
	for (size_t i = 0; i < _out; i++) {
		
		printf("%10.4f", b[i]);
		if (i + 1 < _out) {
			printf(" | ");
		}
	}
	
	cout << endl;
	cout << "----------"; //10 chars
	for (size_t i = 0; i < _out; i++) {
		cout << "----------";
		if (i + 1 < _out) {
			printf("---");
		}
	}
	cout << endl;
	cout << "Weights:  "; //10 chars
			
	for (size_t i = 0; i < _in; i++) {
		for (size_t j = 0; j < _out; j++) {
			float v = w(i, j);
			printf("%10.4f", v);
			if (j + 1 < _out) {
				cout << " | ";
			}
		}
		
		if (i + 1 < _in) {
			cout << endl;
			cout << "          "; //10 chars
		}
	}
	
	cout << endl << endl;
	fflush(stdout);
}

Layer::~Layer() {
	// TODO Auto-generated destructor stub
}

} /* namespace lina */
