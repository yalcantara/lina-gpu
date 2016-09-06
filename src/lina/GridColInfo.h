/*
 * GridColInfo.h
 *
 *  Created on: Jul 3, 2016
 *      Author: yaison
 */

#ifndef GRIDCOLINFO_H_
#define GRIDCOLINFO_H_


#include <vector>
#include <string>

using namespace std;

namespace lina {

class GridColInfo {
	
private:
	
	vector<string> _diffWords;
	
	size_t _count;
	size_t _words;
	size_t _integers;
	size_t _floats;
	size_t _missing;
	size_t _oneCount;
	size_t _zeroCount;
	
	double _max;
	double _min;
	double _sum;
	double _avg;
	double _stdev;
	
	
public:
	GridColInfo();
	void fill(vector<string>& vals);
	
	size_t count()const;
	size_t words()const;
	const vector<string> diffWords()const;
	size_t diffCount()const;
	long int diffIdx(string& word)const;
	size_t integers()const;
	size_t floats()const;
	size_t numbers()const;
	size_t missing()const;
	size_t ones()const;
	size_t zeros()const;
	
	
	double max()const;
	double min()const;
	double sum()const;
	double avg()const;
	double stdev()const;
	
	
	
	void print()const;
	virtual ~GridColInfo();
};

} /* namespace lina */

#endif /* GRIDCOLINFO_H_ */
