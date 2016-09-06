/*
 * Func.cpp
 *
 *  Created on: May 21, 2016
 *      Author: yaison
 */

#include "Func.h"
#include "Exception.h"

#include <string.h>

namespace lina {

Func::Func(const Func& other):fxPtr(other.fxPtr), dxPtr(other.dxPtr){
	
}

Func::Func(const Matrix (*fxPtr)(const Matrix&, const Vector&),
		const Matrix (*dxPtr)(const Matrix&, const Vector&)) :
		fxPtr(fxPtr), dxPtr(dxPtr) {
}

const Matrix Func::fx(const Matrix& x, const Vector& bias) const {
	if (x.cols() != bias.size()) {
		throw Exception(
				"The number of columns does not match the bias length. Columns " + to_string(x.cols())
						+ ", bias length " + to_string(bias.size()) + ".");
	}
	return fxPtr(x, bias);
}
const Matrix Func::dx(const Matrix& x, const Vector& bias) const {
	if (x.cols() != bias.size()) {
		throw Exception(
				"The number of columns does not match the bias length. Columns " + to_string(x.cols())
						+ ", bias length " + to_string(bias.size()) + ".");
	}
	return dxPtr(x, bias);
}

Func::~Func() {
	// TODO Auto-generated destructor stub
}

} /* namespace lina */
