/*
 * Grid.h
 *
 *  Created on: Jul 2, 2016
 *      Author: yaison
 */

#ifndef GRID_H_
#define GRID_H_

#include <vector>
#include <string>

#include "GridInfo.h"
#include "Matrix.h"

using namespace std;

namespace lina {

class Grid {
	
private:
	const size_t _cols;
	GridInfo _info;
	vector<string> data;
	size_t calculateColumns(string& raw, char delimiter);
	void toVector(float* vals, size_t row, size_t start, size_t end, bool stdScale)const; 
	
public:
	Grid(string& raw);
	Grid(string& raw, char delimiter);
	size_t cols()const;
	size_t rows()const;
	void shuffle();
	void replace(size_t col, string oldVal, string newVal);
	void swap(size_t from, size_t to);
	GridInfo info();
	const string get(size_t row, size_t col)const;
	void set(size_t row, size_t col, string& val);
	const Matrix toMatrix(size_t col)const;
	const Matrix toMatrix(size_t col, bool stdScale)const;
	const Matrix toMatrix(size_t start, size_t end)const;
	const Matrix toMatrix(size_t start, size_t end, bool stdScale)const;
	void addRow(vector<string> row);
	void print()const;
	virtual ~Grid();
};

} /* namespace lina */

#endif /* GRID_H_ */
