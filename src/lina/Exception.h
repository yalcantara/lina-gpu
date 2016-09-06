/*
 * Exception.h
 *
 *  Created on: May 18, 2016
 *      Author: yaison
 */

#ifndef EXCEPTION_H_
#define EXCEPTION_H_

#include <exception>
#include <string>

using namespace std;
namespace lina {

class Exception: public exception {
private:
	const string msg;
public:
	Exception(string msg);
	virtual const char* what() const throw();
	virtual ~Exception();
};

} /* namespace lina */

#endif /* EXCEPTION_H_ */
