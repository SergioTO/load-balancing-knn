
#include "pch.h"
#include "container_matrix_confusion.h"

	container_matrix_confusion::~container_matrix_confusion()
	{

		for (int i = 0; i < number_differents_features; i++)
		{
			delete[] m_c[i];
		}
		delete[] m_c;
	}
	container_matrix_confusion::container_matrix_confusion()
	{
	}

	void container_matrix_confusion::init(int number_differents_features, int number_differents_K, int different_class, int desired_features_start, int desired_k_start) {
		this->number_differents_features = number_differents_features;
		this->number_differents_K = number_differents_K;
		this->different_class = different_class;
		this->desired_features_start = desired_features_start;
		this->desired_k_start = desired_k_start;

		//m_c = new shared_ptr<matrix_confusion>*[number_differents_features];
		m_c = new matrix_confusion * [number_differents_features];
		for (int i = 0; i < number_differents_features; i++)
		{
			//m_c[i] = new shared_ptr<matrix_confusion>[number_differents_K];
			m_c[i] = new matrix_confusion[number_differents_K];
			for (int k = 0; k < number_differents_K; k++)
			{

				m_c[i][k].init_matrix_confusion((desired_k_start + k), different_class, (desired_features_start + i));
			}
		}
	}

	void container_matrix_confusion::print_best_matrix()
	{
		cout << endl;
		matrix_confusion* better_matrix = NULL;
		get_better_matrix(better_matrix);

		better_matrix->printMatrix();
		better_matrix->printMessures();
		better_matrix = NULL;
		//for (int i = 0; i < matrices.size(); i++)
		//{
		//	matrices[i].fill_matrix();
		//	matrices[i].fill_messure();
		//	matrices[i].printMatrix();
		//	matrices[i].printMessures();
		//}
	}


	vector<matrix_confusion> container_matrix_confusion::get_better_matrix_for_every_number_of_features()
	{
		vector<matrix_confusion> m(number_differents_features);
		matrix_confusion* better_matrix = NULL;

		for (int i = 0; i < number_differents_features; i++)
		{
			for (int k = 0; k < number_differents_K; k++)
			{
				//matrices[i].fill_matrix();
				//matrices[i].fill_messure();
				if (better_matrix == NULL || better_matrix->correctly_classified < m_c[i][k].correctly_classified)
				{
					better_matrix = &m_c[i][k];
				}
			}
			matrix_confusion n = *better_matrix;
			m[i] = n;
			better_matrix = NULL;
		}
		return m;
	}

	void container_matrix_confusion::get_better_matrix(matrix_confusion*& better_matrix)
	{
		for (int i = 0; i < number_differents_features; i++)
		{
			for (int k = 0; k < number_differents_K; k++)
			{
				//matrices[i].fill_matrix();
				//matrices[i].fill_messure();
				if (better_matrix == NULL || better_matrix->correctly_classified < m_c[i][k].correctly_classified)
				{
					better_matrix = &m_c[i][k];
				}
			}
		}
	}

	matrix_confusion* container_matrix_confusion::get_better_matrix()
	{
		matrix_confusion* better_matrix = NULL;
		get_better_matrix(better_matrix);
		return better_matrix;
	}


	void container_matrix_confusion::fusion_container_matrix(container_matrix_confusion* s, int tam)
	{
		for (int t = 0; t < tam; t++) {
			for (int i = 0; i < number_differents_features; i++)
			{
				for (int k = 0; k < number_differents_K; k++)
				{
					matrix_confusion* m = &(s[t]).m_c[i][k];
					m_c[i][k].fusion_matrix(m);
				}
			}
		}
	}



