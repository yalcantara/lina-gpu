/*
 * Layer.h
 *
 *  Created on: May 18, 2016
 *      Author: yaison
 */

#ifndef LAYER_H_
#define LAYER_H_

#include <stdlib.h>
#include "Matrix.h"
#include "Func.h"

namespace lina {

class Layer {
private:
	size_t _in;
	size_t _out;
	Vector bias;
	Matrix weight;
	Func func;
	
public:
	//including the bias unit
	Layer(const Layer& other);
	Layer(size_t in, size_t out);
	Layer(size_t in, size_t out, Func func);
	
	void rand();
	Layer& operator=(const Layer& other);
	
	size_t in()const;
	size_t out()const;
	size_t params()const;
	void update(Matrix& grad);
	void setBias(const Vector& bias);
	void setWeights(const Matrix& params);
	const Matrix getWeights()const;
	void setFunction(const Func& func);
	const Matrix activate(const Matrix& x)const;
	const Matrix gradient(const Matrix& x) const ;
	void print()const;
	virtual ~Layer();
};

} /* namespace lina */

#endif /* LAYER_H_ */
