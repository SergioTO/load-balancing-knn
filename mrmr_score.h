#pragma once

class mrmr_score
{
public:
	int index;
	double score_correlation;
	double f_value;
	
	double score_final;

	//FCQ F - test correlation quotient
	void calculate_score_final_FCQ(){score_final= f_value / score_correlation; }
	//FCD F - test correlation difference
	void calculate_score_final_FCD() { score_final = f_value - score_correlation; }
};