/*
 * CPUVector.cpp
 *
 *  Created on: May 20, 2016
 *      Author: yaison
 */

#include "CPUVector.h"

#include <stdlib.h>
#include <algorithm>
#include "utils.h"

using namespace std;

namespace lina {

CPUVector::CPUVector(size_t length) :
		length(length) {
	vals = (float*) malloc(sizeof(float) * length);
}

float CPUVector::operator[](size_t idx)const{
	return vals[idx];
}

void CPUVector::print() const {
	size_t l = min((size_t) VECTOR_PRINT_MAX, length);
	
	if(l > VECTOR_PRINT_MAX){
			printf("Vector  %d   (truncated)\n", (int) l);
		}else{
			printf("Vector  %d\n", (int) l);
		}
	
	for (size_t i = 0; i < l; i++) {
		printf("%10.4f", vals[i]);
		
		printf("\n");
	}
	printf("\n");
	fflush(stdout);
}

float* CPUVector::getHostPtr()const{
	return vals;
}

CPUVector::~CPUVector() {
	if(vals){
		free(vals);
	}
}

} /* namespace lina */
