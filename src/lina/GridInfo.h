/*
 * GridInfo.h
 *
 *  Created on: Jul 3, 2016
 *      Author: yaison
 */

#ifndef GRIDINFO_H_
#define GRIDINFO_H_

#include <stdlib.h>
#include <vector>
#include <string>

#include "GridColInfo.h"

using namespace std;

namespace lina {

class GridInfo {
private:
	GridColInfo* colInfo;
	const size_t _cols;
	
	vector<string> getColumnValues(vector<string> data, size_t cols, size_t col);
	
public:
	GridInfo(size_t cols);
	
	void fill(vector<string>& data);
	void fill(vector<string>& data, size_t col);
	
	size_t rows()const;
	size_t cols()const;
	size_t diffCount(size_t col)const;
	const vector<string> diffWords(size_t col)const;
	long int diffIdx(size_t col, string& word) const;
	
	bool hasMissing(size_t col) const;
	bool isWord(size_t col)const;
	bool isInteger(size_t col)const;
	bool isNumeric(size_t col)const;
	bool isFloat(size_t col)const;
	bool isBoolean(size_t col)const;
	
	double sum(size_t col)const;
	double max(size_t col)const;
	double min(size_t col)const;
	double avg(size_t col)const;
	double stdev(size_t col)const;
	
	virtual ~GridInfo();
};

} /* namespace lina */

#endif /* GRIDINFO_H_ */
