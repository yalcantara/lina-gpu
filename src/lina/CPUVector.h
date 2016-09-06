/*
 * CPUVector.h
 *
 *  Created on: May 20, 2016
 *      Author: yaison
 */

#ifndef CPUVECTOR_H_
#define CPUVECTOR_H_

#include <stdlib.h>

namespace lina {

class CPUVector {
	
private:
	float* vals;
public:
	const size_t length;
	CPUVector(size_t length);
	float operator[](size_t idx)const;
	void print()const;
	float* getHostPtr()const;
	virtual ~CPUVector();
};

} /* namespace lina */

#endif /* CPUVECTOR_H_ */
