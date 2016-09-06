/*
 * Exception.cpp
 *
 *  Created on: May 18, 2016
 *      Author: yaison
 */

#include "Exception.h"
#include <string>

namespace lina {

Exception::Exception(string msg):msg(msg) {
	
}


const char* Exception::what() const throw(){
	return msg.c_str();
}


Exception::~Exception() {
	// TODO Auto-generated destructor stub
}

} /* namespace lina */
