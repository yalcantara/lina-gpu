/*
 * Vector.h
 *
 *  Created on: May 20, 2016
 *      Author: yaison
 */

#ifndef VECTOR_H_
#define VECTOR_H_

#include <stdlib.h>

#include "CPUVector.h"


namespace lina {

class Vector {
protected:
	float* devPtr;
	size_t length;
public:
	Vector(const Vector& other);
	Vector(size_t length);
	Vector(size_t length, float* vals);
	size_t size()const;
	Vector& operator=(const Vector& other);
	const Vector operator-(const Vector& b)const;
	void copy(const Vector& src);
	const CPUVector cpu()const;
	float* getDevPtr()const;
	
	void print()const;
	virtual ~Vector();
};

} /* namespace lina */

#endif /* VECTOR_H_ */
