/*
 * Func.h
 *
 *  Created on: May 21, 2016
 *      Author: yaison
 */

#ifndef FUNC_H_
#define FUNC_H_

#include "Matrix.h"

namespace lina {

class Func {
private:
	const Matrix (*fxPtr)(const Matrix& x, const Vector& bias);
	const Matrix (*dxPtr)(const Matrix& x, const Vector& bias);
public:
	Func(const Func& other);
	Func(const Matrix (*)(const Matrix&, const Vector&), const Matrix (*)(const Matrix&, const Vector&) );
	
	const Matrix fx(const Matrix& x, const Vector& bias)const;
	const Matrix dx(const Matrix& x, const Vector& bias)const;
	
	virtual ~Func();
};

} /* namespace lina */

#endif /* FUNC_H_ */
