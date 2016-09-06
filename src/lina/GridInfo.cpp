/*
 * GridInfo.cpp
 *
 *  Created on: Jul 3, 2016
 *      Author: yaison
 */

#include "GridInfo.h"

#include <iostream>
#include <math.h>

using namespace std;
namespace lina {

GridInfo::GridInfo(size_t cols) :
		_cols(cols), colInfo(nullptr) {
	
	colInfo = (GridColInfo*) calloc(cols, sizeof(GridColInfo));
}

vector<string> GridInfo::getColumnValues(vector<string> data, size_t cols, size_t col) {
	
	size_t rows = (size_t) ceil(data.size() / (float) cols);
	
	vector<string> vals;
	for (size_t i = 0; i < rows; i++) {
		string val = data.at(i * cols + col);
		vals.push_back(val);
	}
	
	return vals;
}

void GridInfo::fill(vector<string>& data) {
	
	for (size_t j = 0; j < _cols; j++) {
		vector<string> vals = getColumnValues(data, _cols, j);
		colInfo[j].fill(vals);
	}
}

void GridInfo::fill(vector<string>& data, size_t col) {
	
	vector<string> vals = getColumnValues(data, _cols, col);
	colInfo[col].fill(vals);
	
}

size_t GridInfo::rows() const {
	return colInfo[0].count();
}
size_t GridInfo::cols() const {
	return _cols;
}
size_t GridInfo::diffCount(size_t col) const {
	return colInfo[col].diffCount();
}
const vector<string> GridInfo::diffWords(size_t col) const {
	return colInfo[col].diffWords();
}

long int GridInfo::diffIdx(size_t col, string& word) const {
	return colInfo[col].diffIdx(word);
}

bool GridInfo::hasMissing(size_t col) const {
	if (rows() == 0)
		return false;
	
	return colInfo[col].missing() > 0;
}

bool GridInfo::isWord(size_t col) const {
	if (rows() == 0)
		return false;
	
	return (colInfo[col].words() + colInfo[col].missing()) == colInfo[col].count();
}

bool GridInfo::isInteger(size_t col) const {
	if (rows() == 0)
		return false;
	
	return (colInfo[col].integers() + colInfo[col].missing()) == colInfo[col].count();
	
}

bool GridInfo::isNumeric(size_t col) const {
	if (rows() == 0)
		return false;
	
	return colInfo[col].words() == 0;
}

bool GridInfo::isFloat(size_t col) const {
	if (rows() == 0)
		return false;
	
	return (colInfo[col].floats() + colInfo[col].missing()) == colInfo[col].count();
}

bool GridInfo::isBoolean(size_t col) const {
	if (rows() == 0)
		return false;
	
	return (colInfo[col].ones() + colInfo[col].zeros() + colInfo[col].missing())
			== colInfo[col].count();
}

double GridInfo::sum(size_t col) const {
	return colInfo[col].sum();
}

double GridInfo::max(size_t col) const {
	return colInfo[col].max();
}

double GridInfo::min(size_t col) const {
	return colInfo[col].min();
}

double GridInfo::avg(size_t col) const {
	return colInfo[col].avg();
}

double GridInfo::stdev(size_t col) const {
	return colInfo[col].stdev();
}

GridInfo::~GridInfo() {
	if (colInfo) {
		free(colInfo);
	}
}

} /* namespace lina */
