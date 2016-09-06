/*
 * GridColInfo.cpp
 *
 *  Created on: Jul 3, 2016
 *      Author: yaison
 */

#include "GridColInfo.h"

#include <math.h>
#include "utils.h"

namespace lina {

GridColInfo::GridColInfo() :
		_count(0), _words(0), _integers(0), _floats(0), _missing(0), _oneCount(0), _zeroCount(0), _diffWords(), _max(NAN), _min(
		NAN), _avg(
		NAN), _sum(0.0), _stdev(
		NAN) {
	
}

void GridColInfo::fill(vector<string>& vals) {
	
	size_t n = vals.size();
	
	if (n == 0) {
		return;
	}
	
	size_t intCount = 0;
	size_t floatCount = 0;
	size_t _diffCount = 0;
	size_t wordCount = 0;
	size_t missingCount = 0;
	size_t oneCount = 0;
	size_t zeroCount = 0;
	
	double colMax = NAN;
	double colMin = NAN;
	double colSum = 0.0;
	
	double __mean = 0.0;
	double __m2 = 0.0;
	for (size_t i = 0; i < n; i++) {
		string v = vals.at(i);
		
		if (v == "?") {
			//easy case, val is NULL ^_^
			missingCount++;
		} else {
			const char* val = v.c_str();
			char* end;
			float num = strtof(val, &end);
			if ((end != NULL) && (end[0] == '\0')) {
				if (ceil(num) == num) {
					intCount++;
					if(num == 1.0){
						oneCount++;
					}else if(num == 0.0){
						zeroCount++;
					}
				} else {
					floatCount++;
				}
				colMax = higher(colMax, num);
				colMin = lower(colMin, num);
				
				colSum += num;
				
				//Online standard deviation algorithm
				double delta = num - __mean;
				__mean += delta / n;
				__m2 += delta * (num - __mean);
			} else {
				//not a number
				wordCount++;
				if (contains(_diffWords, v) == false) {
					_diffCount++;
					_diffWords.push_back(v);
				}
			}
		}
	}
	
	_count = n;
	_words = wordCount;
	_integers = intCount;
	_floats = floatCount;
	_missing = missingCount;
	_oneCount = oneCount;
	_zeroCount = zeroCount;
	
	if (intCount > 0 || floatCount > 0) {
		_max = colMax;
		_min = colMin;
		_avg = __mean;
		_sum = colSum;
		_stdev = (__m2 / (n - 1));
	}
}

size_t GridColInfo::count() const {
	return _count;
}

size_t GridColInfo::words() const {
	return _words;
}

const vector<string> GridColInfo::diffWords() const {
	const vector<string> ret = _diffWords;
	return ret;
}

size_t GridColInfo::diffCount() const {
	return _diffWords.size();
}

long int GridColInfo::diffIdx(string& val) const {
	return indexOf(_diffWords, val);
}

size_t GridColInfo::integers() const {
	return _integers;
}

size_t GridColInfo::floats() const {
	return _floats;
}

size_t GridColInfo::numbers() const {
	return _integers + _floats;
}

size_t GridColInfo::missing() const {
	return _missing;
}

size_t GridColInfo::ones() const {
	return _oneCount;
}

size_t GridColInfo::zeros() const {
	return _zeroCount;
}

double GridColInfo::max() const {
	return _max;
}

double GridColInfo::min() const {
	return _min;
}

double GridColInfo::sum() const {
	return _sum;
}

double GridColInfo::avg() const {
	return _avg;
}

double GridColInfo::stdev() const {
	return _stdev;
}

void GridColInfo::print() const {
	cout << "----------------------------------------------------" << endl;
	printf("Count       :  %10d\n", (int) _count);
	printf("Words       :  %10d\n", (int) _words);
	printf("Diff Words  :  %10d\n", (int) diffCount());
	printf("Integers    :  %10d\n", (int) _integers);
	printf("Floats      :  %10d\n", (int) _floats);
	printf("Missing     :  %10d\n", (int) _missing);
	printf("\n");
	printf("Max         :  %10.4f\n", _max);
	printf("Min         :  %10.4f\n", _min);
	printf("Avg         :  %10.4f\n", _avg);
	printf("Sum         :  %10.4f\n", _sum);
	printf("Std Dev     :  %10.4f\n", _stdev);
	cout << "----------------------------------------------------" << endl;
	fflush(stdout);
	
}

GridColInfo::~GridColInfo() {
	// TODO Auto-generated destructor stub
}

} /* namespace lina */
