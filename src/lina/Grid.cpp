/*
 * Grid.cpp
 *
 *  Created on: Jul 2, 2016
 *      Author: yaison
 */

#include "Grid.h"

#include <iostream>
#include <sstream>
#include <string>
#include <cstring>
#include "utils.h"

#include "Exception.h"

using namespace std;

namespace lina {

Grid::Grid(string& raw) :Grid(raw, ',') {
	
}

Grid::Grid(string& raw, char delimiter) :
		_cols(calculateColumns(raw, delimiter)), _info(_cols) {
	
	int start = 0;
	int idx = 0;
	for (int i = 0;; i++) {
		idx = raw.find('\n', idx);
		
		int length = idx - start;
		string line = raw.substr(start, length);
		
		if (length != 0) {
			
			long long missing = line.find("?");
			if (missing >= 0) {
				//cout << "ignoring line '" << line <<"'" <<endl;
			} else {
				
				istringstream iss(line);
				string token;
				int counter = 0;
				while (getline(iss, token, delimiter)) {
					string trimmed = trim(token);
					
					data.push_back(trimmed);
					counter++;
				}
				
				if (counter > 0) {
					for (int i = counter; i < _cols; i++) {
						data.push_back("(empty)");
					}
				}
			}
		}
		
		if (idx < 0) {
			break;
		}
		start = idx + 1;
		idx++;
	}
	
	_info.fill(data);
}

void Grid::shuffle() {
	
	size_t m = rows();
	
	//inspired from Java's Collection.shuffle
	for (size_t i = m; i > 1; i--) {
		size_t to = round((rand() / (double) RAND_MAX) * (i - 1));
		
		swap(i - 1, to);
	}
}

void Grid::replace(size_t col, string oldVal, string newVal) {
	
	size_t m = rows();
	bool modified = false;
	for (size_t i = 0; i < m; i++) {
		string val = get(i, col);
		
		if (val == oldVal) {
			set(i, col, newVal);
			modified = true;
		}
	}
	
	if(modified){
		_info.fill(data, col);
	}
}

void Grid::swap(size_t from, size_t to) {
	vector<string> tmp(_cols);
	
	for (size_t j = 0; j < _cols; j++) {
		tmp[j] = get(to, j);
	}
	
	for (size_t j = 0; j < _cols; j++) {
		string val = get(from, j);
		set(to, j, val);
	}
	
	for (size_t j = 0; j < _cols; j++) {
		string val = tmp[j];
		set(from, j, val);
	}
}

size_t Grid::calculateColumns(string& raw, char delimiter) {
	long long idx = raw.find('\n', 0);
	if (idx == -1) {
		throw Exception("Could not find the '\\n' character.");
	}
	
	int start = 0;
	int length = idx - start;
	string line = raw.substr(start, length);
	
	if (length == 0) {
		throw Exception("The first line is empty.");
	}
	
	int counter = 0;
	istringstream iss(line);
	string token;
	while (getline(iss, token, delimiter)) {
		counter++;
	}
	
	return counter;
}

const string Grid::get(size_t row, size_t col) const {
	
	if (row >= rows()) {
		throw Exception(
				"The row parameter is out of bounds. Expected < " + to_string(rows()) + " but got "
						+ to_string(row) + " instead.");
	}
	
	if (col >= cols()) {
		throw Exception(
				"The col parameter is out of bounds. Expected < " + to_string(cols()) + " but got "
						+ to_string(col) + " instead.");
	}
	
	const string val = data.at(row * _cols + col);
	
	return val;
}

void Grid::set(size_t row, size_t col, string& val) {
	data[row * _cols + col] = val;
}

size_t Grid::cols() const {
	return _cols;
}

size_t Grid::rows() const {
	return data.size() / _cols;
}

const Matrix Grid::toMatrix(size_t col) const {
	return toMatrix(col, false);
}

const Matrix Grid::toMatrix(size_t col, bool stdScale) const {
	return toMatrix(col, col + 1, stdScale);
}

const Matrix Grid::toMatrix(size_t start, size_t end) const {
	return toMatrix(start, end, false);
}

const Matrix Grid::toMatrix(size_t start, size_t end, bool stdScale) const {
	size_t total = 0;
	
	for (size_t i = start; i < end; i++) {
		if (_info.isNumeric(i)) {
			total++;
		} else if (_info.isWord(i)) {
			total += _info.diffCount(i);
		} else {
			throw Exception(
					"Invalid data for column " + to_string(i)
							+ ", it isn't fully numeric or fully word.");
		}
		
	}
	
	vector<float> v(total * rows());
	float* vals = v.data();
	
	for (size_t i = 0; i < rows(); i++) {
		float* ptr = vals + i * total;
		toVector(ptr, i, start, end, stdScale);
	}
	
	const Matrix mtr(rows(), total, vals);
	
	return mtr;
}

void Grid::toVector(float* vals, size_t row, size_t start, size_t end, bool stdScale) const {
	
	size_t colIdx = 0;
	for (size_t i = start; i < end; i++) {
		string val = data.at(row * _cols + i);
		
		if (_info.isNumeric(i)) {
			float f = stringToFloat(val);
			if (stdScale && _info.isBoolean(i) == false) {
				double stdev = _info.stdev(i);
				double mean = _info.avg(i);
				float ans = (float) ((f - mean) / stdev);
				vals[colIdx] = ans;
			} else {
				vals[colIdx] = f;
			}
			colIdx++;
		} else if (_info.isWord(i)) {
			long int idx = _info.diffIdx(i, val);
			
			if (idx < 0) {
				throw Exception(
						"Word '" + val + "' not found for column idx " + to_string(i) + ".");
			}
			
			vals[colIdx + idx] = 1.0;
			colIdx += _info.diffCount(i);
		} else {
			throw Exception(
					"Invalid data for column " + to_string(i)
							+ ", it isn't fully numeric or fully word.");
		}
	}
}

void Grid::print() const {
	
	size_t m = rows();
	size_t n = cols();
	
	for (size_t i = 0; i < m; i++) {
		for (size_t j = 0; j < n; j++) {
			string val = get(i, j);
			cout << val;
			
			if (j + 1 < n) {
				cout << ", ";
			}
		}
		
		if (i + 1 < m) {
			cout << endl;
		}
	}
	cout << endl;
}

Grid::~Grid() {
	data.clear();
}

} /* namespace lina */
