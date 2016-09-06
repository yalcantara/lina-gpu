/*
 * Network.h
 *
 *  Created on: May 21, 2016
 *      Author: yaison
 */

#ifndef NETWORK_H_
#define NETWORK_H_

#include <stdlib.h>
#include <vector>
#include "Layer.h"

using namespace std;
namespace lina {

class Network {

private:

public:
	
	Layer* layers;
	
	const size_t depth;
	Network(size_t depth);
	Network(size_t depth, size_t in,  size_t out);
	
	void rand();
	
	size_t out()const;
	size_t params() const;
	
	void update(vector<Matrix>& grad);
	
	const Matrix activate(const Matrix& x)const;
	void sgdescent(Matrix& X, Matrix& Y);
	void gdescent(Matrix& X, Matrix& Y);
	vector<Matrix> backprop(Matrix& X, Matrix& Y) const;
	float j(const Matrix& x, const Matrix& y)const;
	Layer& operator[](unsigned int idx);
	void print()const;
	virtual ~Network();
};

} /* namespace lina */

#endif /* NETWORK_H_ */
