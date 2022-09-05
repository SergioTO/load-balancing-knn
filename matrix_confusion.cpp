#include "pch.h"
#include "matrix_confusion.h"
matrix_confusion::matrix_confusion() {}

	void matrix_confusion::init_matrix_confusion(int k, int different_class, int number_characteristics_use)
	{

		this->k = k;
		this->different_class = different_class;
		this->number_characteristics_use = number_characteristics_use;
		for (int i = 0; i < different_class; i++)
		{
			matrix.push_back(vector<int>());
			for (int j = 0; j < different_class; j++)
			{
				matrix[i].push_back(0);
			}
		}

	}

	void matrix_confusion::recalculate_classified()
	{
	correctly_classified=0;
	incorrectly_classified=0;
		for(int i=0;i<different_class;i++)
		{
			for(int j=0;j<different_class;j++)
			{
				if(i==j){correctly_classified+=matrix[i][j];}
				else{		incorrectly_classified+=matrix[i][j];}
	
			}

		}
	}	

	double matrix_confusion::accuracy()
	{
		return (correctly_classified / double(correctly_classified + incorrectly_classified) * 100);
	}
	void matrix_confusion::fill_matrix(int real, int assigned)
	{

		matrix[real][assigned] += 1;
		if (real == assigned) { correctly_classified++; }
		else { incorrectly_classified++; }

	}

	//void fill_matrix() 
	//{
	//	for (auto i = 0;i< assigned_class.size(); i++) 
	//	{
	//		auto real=real_class[i];
	//		auto assigned = assigned_class[i];
	//		matrix[real][assigned] += 1;
	//	}
	//}
	//void fill_messure()
	//{
	//	for (int i = 0; i < different_class; i++) 
	//	{
	//		for (int j = 0; j < different_class; j++) 
	//		{
	//			if (i == j) { correctly_classified += matrix[i][j]; }
	//			else { incorrectly_classified += matrix[i][j]; }
	//		}
	//	}

	//	accuracy = correctly_classified / (correctly_classified+(double)incorrectly_classified);
	//}
	void matrix_confusion::fusion_matrix(matrix_confusion* m)
	{
		correctly_classified += m->correctly_classified;
		incorrectly_classified += m->incorrectly_classified;
		for (int i = 0; i < different_class; i++) //Filas
		{
			for (int j = 0; j < different_class; j++) //Columnas
			{
				matrix[j][i] += m->matrix[j][i];
			}
		}
	}

	void matrix_confusion::printMatrix()
	{
		cout << endl << "Confusion matrix" << endl << endl;
		for (int i = 0; i < different_class; i++) //Filas
		{
			for (int j = 0; j < different_class; j++) //Columnas
			{

				cout << std::setw(10) << matrix[j][i];
			}
			cout << endl;
		}
	}

	void matrix_confusion::printMessures()
	{
		cout << endl << "Mesures with number_characteristics_use: " << number_characteristics_use << endl;
		cout << "Mesures with K: " << k << endl << endl;


		cout << "Results KNN" << endl;
		cout << "    Correctly Classified Instances     " << std::setw(10) << correctly_classified << "   " << std::setw(10) << correctly_classified / double(correctly_classified + incorrectly_classified) * 100 << "%" << endl;;
		cout << "    Incorrectly Classified Instances   " << std::setw(10) << incorrectly_classified << "   " << std::setw(10) << incorrectly_classified / double(correctly_classified + incorrectly_classified) * 100 << "%" << endl;;
		cout << endl;
	}


