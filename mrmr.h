#pragma once
#include "common_include.h"
#include "point.h"
#include "point_result.h"
#include <math.h>   
#include "mrmr_score.h"



class mrmr {
public:

	struct {
		int index_feature;
		double score;
		bool operator()(point_result& p1, point_result& p2) const { return p1.distance < p2.distance; }
	} comparison;

	int* occurency_class;
	double *variance_features; //variance_features[4], varianza de la columna/feature 4 (es decir, la columna quinta)
	double *standard_deviation_features; //standard_deviation_features[4], la desviación estandar de la columna/feature 4 (es decir, la columna quinta)
	double *mean_features; //mean_features[4], media de la columna/feature 4 (es decir, la columna quinta)
	double *F_statis_score; //F_statis_score[4], el score de fstatis de la columna/feature 4 (es decir, la columna quinta), respecto a la clase objetivo


	//Array multidimensional cada punto, ejemplo: data[2][5] indica, Feature 2, valor del 5 dato (es decir, el valor que toma el dato 5 (el sexto) para la feature/columna/atrbiuto 2 (la tercera)
	double **matrix_correlation;
	double **data;

	int *target_class; //array con los labels



	int total_size;
	int total_number_features;
	int desired_number_features;
	int values_target_class;
	bool execute_parallel = false;

	vector<int> features_finall;

	mrmr(int values_target_class, int total_number_features, int desired_number_features, vector<point> &vector, int parallel)
	{
		if (parallel) { execute_parallel = true; }
		this->total_size = vector.size();
		this->total_number_features = total_number_features;
		this->desired_number_features = desired_number_features;
		this->values_target_class = values_target_class;


		occurency_class = new int[values_target_class];
		variance_features = new double[total_number_features];
		standard_deviation_features = new double[total_number_features];
		mean_features = new double[total_number_features];
		F_statis_score = new double[total_number_features];
		target_class = new int[total_size];


		create_target_class(vector);
		crear_occurency_class(values_target_class);
		crear_data(vector, total_size, total_number_features);
		create_matrix_correlation(total_number_features);
		features_finall = run();
		printed_features();


	}
	~mrmr() 
	{

		delete[] occurency_class;
		delete[] variance_features; 
		delete[] standard_deviation_features; 
		delete[] mean_features; 
		delete[] F_statis_score;




		for (int i = 0; i < total_number_features; i++) {
			delete[] matrix_correlation[i];
		}
		delete[] matrix_correlation;

		for (int i = 0; i < total_number_features; i++) {
			delete[] data[i];
		}
		delete[] data;

		delete[] target_class; 
	}
	void printed_features() 
	{
		cout << "MRMR with " << features_finall.size() << " features" << endl;
		for (int i = 0; i < features_finall.size(); i++)
		{
			cout << features_finall[i] << " ";
		}
		cout << endl;
	}

	void create_target_class(std::vector<point> & vector)
	{
		#pragma omp parallel for if(execute_parallel==true)
		for (int i = 0; i < total_size; i++)
		{
			target_class[i] = vector[i].class_of_the_point;
		}
	}

	void crear_occurency_class(int values_target_class)
	{
		//Las veces que ocurre cada uno
		#pragma omp parallel for if(execute_parallel==true)
		for (int i = 0; i < values_target_class; i++) {
			occurency_class[i] = 0; 	//Inicializa a 0
			for (int j = 0; j < total_size; j++)
				if (target_class[j] == i)
				{
					occurency_class[i] += 1;
				}
		}
	}

	void crear_data(vector<point> &vector, int size, int size_columns)
	{
		data = new double*[size_columns];
		#pragma omp parallel for if(execute_parallel==true)
		for (int i = 0; i < size_columns; i++) {
			data[i] = new double[size];//lo incializo aquí en vez de arriba
			for (int j = 0; j < size; j++)
			{
				double value = vector[j].characteristics[i];
				data[i][j] = value;
			}
		}

	}
	void create_matrix_correlation(int number_features)
	{
		//Crear matriz
		matrix_correlation = new double*[number_features];

		//Inicializar matriz
		#pragma omp parallel for if(execute_parallel==true)
		for (int i = 0; i < number_features; i++) {
			matrix_correlation[i] = new double[number_features]; // these are our columns //lo incializo aquí en vez de arriba
			for (int j = 0; j < number_features; j++) {
				matrix_correlation[i][j] = -1;
			}
		}
	}

	vector<int>  run()
	{


		//Aquí paralelo
		#pragma omp parallel for if(execute_parallel==true)
		for (int i = 0; i < total_number_features; i++)
		{
			mean_features[i] = mean(data[i], total_size);
			variance_features[i] = variance(data[i], total_size, mean_features[i]);
			standard_deviation_features[i] = sqrt(variance_features[i]);
			double f = Fstatis(data[i], target_class, total_size, mean_features[i]);
			F_statis_score[i] = f;
		}

		vector<int> vector_features_selected;
		vector_features_selected.reserve(desired_number_features);
		vector<mrmr_score> vector_features_not_selected;
		vector_features_not_selected.reserve(total_number_features);

		//NO paralelo
		for (int i = 0; i < total_number_features; i++)
		{
			mrmr_score aux;
			aux.index = i;
			aux.f_value = F_statis_score[i];
			aux.score_correlation = -1; //No es necesario, pero le damos un valor
			vector_features_not_selected.push_back(aux);
		}

		int last_index_feature_added = max_value_index_in_vector(F_statis_score, total_number_features);
		vector_features_selected.push_back(last_index_feature_added);

		//int last_index_feature_added = vector_features_selected.back();

		vector_features_not_selected.erase(vector_features_not_selected.begin() + last_index_feature_added);




		//Start with one because already have the first feature (max F_statis_socre)
		for (int k = 1; k < desired_number_features; k++) {

			#pragma omp parallel for if(execute_parallel==true)
			for (int i = 0; i < vector_features_not_selected.size(); i++)
			{
				int index_feature_not_selected = vector_features_not_selected[i].index;
				//double correlation_sum = 0;
				double c = correlation(data[index_feature_not_selected], data[last_index_feature_added], total_size, mean_features[index_feature_not_selected], mean_features[last_index_feature_added], standard_deviation_features[index_feature_not_selected], standard_deviation_features[last_index_feature_added]);
				c = abs(c);
				if (c < 0.001)
				{
					c = 0.001;
				}
				matrix_correlation[index_feature_not_selected][last_index_feature_added] = c;
				matrix_correlation[last_index_feature_added][index_feature_not_selected] = c;
			}

			#pragma omp parallel for  if(execute_parallel==true)
			for (int i = 0; i < vector_features_not_selected.size(); i++)
			{
				double sum_correlation = 0;
				int index_feature_not_selected = vector_features_not_selected[i].index;
				for (int j = 0; j < vector_features_selected.size(); j++)
				{
					int index_feature_selected = vector_features_selected[j];
					sum_correlation += matrix_correlation[index_feature_not_selected][index_feature_selected];
				}
				double mrmr_score_correlation = sum_correlation / vector_features_selected.size();
				vector_features_not_selected[i].score_correlation = mrmr_score_correlation;
				vector_features_not_selected[i].calculate_score_final_FCQ();
			}

			//Elige la mejor caracteristica y la añade vector_features_selected, a continuación la borra de vector_features_not_selected
			//El bucle continua
			{
			int temp_index = max_score_index(vector_features_not_selected); //Seleccionamos el mejor con el mrmr score
			last_index_feature_added = vector_features_not_selected[temp_index].index; //Obtenemos el valor de la columna/feature
			vector_features_selected.push_back(last_index_feature_added); //lo añadimos a seleccionado
			vector_features_not_selected.erase(vector_features_not_selected.begin() + temp_index);//Lo eliminamos de no seleccionado
			}
			//continuamos para añadir el siguiente
		}
		//Elegir la mejor feature

		return vector_features_selected;
	}



	int max_value_index_in_vector(double *data, int size)
	{
		double max = -DBL_MAX;
		int index = -1;
		for (int i = 0; i < size; i++)
		{
			if (data[i] > max) { max = data[i]; index = i; };
		}
		return index;
	}

	int max_score_index(vector<mrmr_score> &data)
	{
		double max = -DBL_MAX;
		int index = -1;
		for (int i = 0; i < data.size(); i++)
		{
			if (data[i].score_final > max) { max = data[i].score_final; index = i; };
		}
		return index;
	}




	double Fstatis(const double *data, const int *target_class_h, const int size, double g_mean)
	{

		double result;
		double* g_mean_target_class = new double[values_target_class]; //Este valor no es variable global porque nunca más se volvera a computar

		double primera_parte = 0;
		double segunda_parte = 0;
		double segunda_parte_primera = 0;

		vector<double*> valores_por_clase;
		int reservar = (size / values_target_class) + 500;
		valores_por_clase.reserve(reservar);

		int k = 0;
		for (int i = 0; i < values_target_class; i++) {
			k = 0;
			double *valor = new double[occurency_class[i]];
			for (int j = 0; j < size; j++)
				if (target_class_h[j] == i)
				{
					valor[k] = data[j];
					k++;
				}
			valores_por_clase.push_back(valor);
		}


		//Para cada clase calculmaos la media de  todos los valores de datos que pertenecen a esa clase
		for (int i = 0; i < values_target_class; i++) {
			g_mean_target_class[i] = mean(valores_por_clase[i], occurency_class[i]);
		}

		for (int i = 0; i < values_target_class; i++) {
			primera_parte += occurency_class[i] * pow((g_mean_target_class[i] - g_mean), 2) / (values_target_class - 1);
		}
		//Aquí calculamos la varianza NO de todos los valores de una feature, sino la varianza solo para los valores que están en cada clase de la columna objetivo
		for (int i = 0; i < values_target_class; i++) {
			segunda_parte_primera += (occurency_class[i] - 1) * variance(valores_por_clase[i], occurency_class[i], g_mean_target_class[i]);
		}
		segunda_parte = segunda_parte_primera / (size - values_target_class);

		result = primera_parte / segunda_parte;

		
		delete[] g_mean_target_class;
		for (auto v : valores_por_clase)
		{
			delete[] v;
		}
		return result;
	}

private:
	double mean(const double *data, const int size)
	{
		double mean_result = 0;
		for (int i = 0; i < size; i++)
		{
			mean_result += data[i];
		}
		mean_result = mean_result / size;
		return mean_result;
	}

	double variance(const double *data, int size, const double mean)
	{
		double variance_result = 0;
		for (int i = 0; i < size; i++)
		{
			variance_result += (data[i] - mean) * (data[i] - mean);
		}
		variance_result = variance_result / (size - 1); //Varianza de una muestra
		//variance_result = variance_result / (size - 1); //Varianza de una poblacion
		return variance_result;
	}
	double covariance(const double *data_x, const double *data_y, const int size, const double mean_x, const double mean_y)
	{
		double covariance_result = 0;
		for (int i = 0; i < size; i++)
		{
			covariance_result += (data_x[i] - mean_x)*(data_y[i] - mean_y);
		}
		covariance_result = covariance_result / (size - 1);
		return covariance_result;
	}
	//Pearson correlation

	double correlation(const double *data_x, const double *data_y, const int size, const double mean_x, const double mean_y, const double standard_deviation_x, const double standard_deviation_y)
	{
		double correlation_result = covariance(data_x, data_y, size, mean_x, mean_y)
			/ (standard_deviation_x * standard_deviation_y);
		return correlation_result;
	}

};
