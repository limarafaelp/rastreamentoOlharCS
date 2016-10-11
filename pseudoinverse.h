#pragma once

#include <windows.h>
#include <wchar.h>
#include "pxcsensemanager.h"
#include "pxcfacemodule.h" //Necessario para o novo driver do realsense
#include "pxchandmodule.h" //idem
#include "pxchanddata.h"   //idem

#include "util_render.h"  //SDK provided utility class used for rendering (packaged in libpxcutils.lib)

#include <conio.h>
#include <iostream>
#include <ctime>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <iostream>
#include <stdio.h>
#include <conio.h>
#include <wchar.h>
#include <vector>
#include "pxcsession.h"
#include "pxccapture.h"
#include "util_render.h"
#include "util_cmdline.h"
#include "pxcprojection.h"
#include "pxcmetadata.h"
#include "util_cmdline.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Eigen>
//#include <Eigen/src/Core/Block.h>
//#include<Eigen/src/Core/DenseBase.h>
#include <opencv2/core/eigen.hpp>

//#include <armadillo>
//C:\armadillo - 7.400.4
//#include <C:/armadillo-7.400.4/include/armadillo>
#include <armadillo>
//#include <$(ARMADILLO_ROOT)/include/armadillo>
using namespace Eigen;
using namespace std;
using namespace cv;
using namespace arma;

/*void printMatrix(Mat A)
{
	int m = A.rows;
	int n = A.cols;
	for (int i = 0; i < m; i++)
	{
		printf("[ ");
		for (int j = 0; j < n - 1; j++)
			printf("%.7f\t", A.at<double>(i, j));

		printf("%.7f ]\n", A.at<double>(i, n - 1));
	}
	printf("\n");
}*/

/*void printEigenMatrix(Eigen::MatrixXf *A)
{
	int m = A->rows();
	int n = A->cols();
	for (int i = 0; i < m; i++)
	{
		printf("[ ");
		for (int j = 0; j < n - 1; j++)
		{
			printf("%.7f\t", (*A)(i, j));
		}
		printf("%.7f\ ]\n", (*A)(i, n - 1));
	}
	printf("\n");
	//A->lpNorm<Infinity>();
}*/

/*template<typename Number, typename T1>
void foo(Number &x, const Eigen::DenseBase<T1>& matrix)
{
	MatrixXf AA = matrix.eval();
	cout << AA.lpNorm<2>();
	//matrix.template lpNorm<2>();
	//return matrix.lpNorm<2>();
}*/

//estava const Ref <Eigen::MatrixXf>&A,
//estava Eigen::MatrixBase<T1> & iA,
/*template<typename T1, typename T2> //typename
void pinvEigen(
	Eigen::DenseBase<T1> & A,
	Eigen::DenseBase<T2> & iA,
	//Eigen::MatrixBase<T1> & A,
	//Eigen::MatrixBase<T2> & iA,
	bool iA_isNULL)
{
	const double eps = 0.0000000001;
	int m = A.rows();
	int n = A.cols();

	if (iA_isNULL) //Deve ser chamada também quando n == 1
	{
		cout << "NORMA DE A" << endl;
		MatrixXf AA = A.leftCols(1).eval();
		double x = AA.squaredNorm();

		iA = AA.transpose().eval();
		if (x != 0)
			iA /= x;
	}
	if(n != 1)
	{
		int k0;
		if (iA_isNULL)
		{
			k0 = 1;
		}
		else
		{
			k0 = n - 1;
		}
		
		for (int k = k0; k < n; k++)
		{
			//cout << "iA ate agora:" << endl << iA << endl;
			auto A1 = A.leftCols(k);
			Ref <const MatrixXf> iA_mat = iA;
			Ref <const MatrixXf> a = A.col(k);
			//Ref <const MatrixXf> g = iA_mat*a;
			MatrixXf g = iA_mat*a;
			
			//Eigen::DenseBase<T1> a = A.col(k);
			MatrixXf c;

			float gg = g.squaredNorm();
			//cout << "iA.shape = (" << iA_mat.rows() << ", " << iA_mat.cols() << ")" << endl;
			//cout << "a shape = (" << a.rows() << ", " << a.cols() << "), A[:,:k].shape = (" << A.rows() << ", "<< k <<"), g.shape = (" << g.rows() << ", " << g.cols() << ")";
			//cout << "Ate aqui OK" << endl;
			MatrixXf bloco_sup = a - A.leftCols(k)*g;	
			//cout << "e ate aqui" << endl;
			MatrixXf aux = A.leftCols(k)*g - a;
			//cout << "e ate aqui de novo" << endl;
			if (aux.lpNorm<Eigen::Infinity>() < eps)
			{	
				c = (1 / (1 + gg))*g.transpose()*iA_mat;
			}
			else
			{
				pinvEigen(bloco_sup, c, false);
			}
			cout << "Nao funciona apos esta linha" << endl;
			iA.resize(iA.rows() + 1, iA.cols());
			iA << bloco_sup, c;
			//iA_mat.resize(iA_mat.rows() + 1, iA_mat.cols());
			//iA_mat << bloco_sup, c;
			//iA << iA_mat;
		}
	}
}*/
/*
Mat pinv(Mat A, Mat *iA_1 = NULL)
{
	// Calcula a pseudoinversa da matriz A = [a1 | ... | ak] //
	 // onde iA_1 é a pseudoinversa de [a1 | ... | ak-1]       //
	 // só funciona para matrizes do tipo double (CV_64FC1)   //
	Mat iA;

	const double eps = 0.0000000001;
	int m = A.rows;
	int n = A.cols;

	if (n == 1)
	{
		double x = cv::norm(A, NORM_L2SQR);
		cv::transpose(A, iA);
		if (x != 0)
			iA /= x;
		return iA;
	}

	int k0;
	if (iA_1 == NULL)
	{
		k0 = 1;
		Rect col0 = Rect(0, 0, 1, m);
		iA = pinv(A(col0));
	}
	else
	{
		k0 = n - 1;
		iA = (*iA_1);
	}

	for (int k = k0; k < n; k++)
	{
		Rect cols_k = Rect(0, 0, k, m);
		Rect col_k = Rect(k, 0, 1, m);
		
		Mat A1 = A(cols_k);
		Mat a = A(col_k);
		Mat g = iA*a;
		double gg = g.dot(g);

		Mat c;
		if (norm(A1*g - a, NORM_INF) < eps)
		{
			transpose(g, g);
			c = (1. / (1 + gg))*g*iA;
			transpose(g, g);
		}
		else
			c = pinv(a - A1*g);

		vconcat(iA - g*c, c, iA);
	}
	return iA;
}
*/

arma::mat pseudoinversa(const arma::mat &A, const arma::mat &iA_ant, bool iA_isnull)
{
	double eps = 0.0000000001; //1e-10
	int m = A.n_rows;
	int n = A.n_cols;

	arma::mat iA;
	int k0;
	if (iA_isnull)
	{
		arma::mat col0 = A.col(0) % A.col(0);
		double x = arma::accu(col0);
		
		if (x == 0)
			iA = col0.t();
		else
			iA = (1 / x)*col0.t();

		k0 = 1;
	}
	else
	{
		k0 = n - 1;
		iA = iA_ant;
	}

	for (int k = k0; k < n; k++)
	{
		const arma::mat &A1 = A(span(0, m - 1), span(0, k - 1));
		arma::mat a = A(span(0, m - 1), k);
		
		iA.print("iA");
		a.print("a");
		cout << a.t()*a << endl;
		//double g = arma::accu(iA % a); //Nao consegui multiplicar matrizes como iA*a
		//cout << g;
		//g.print("iA*a");
		//arma::mat g = arma::mat(iA*a, iA.n_rows, iA.n_cols);
		//arma::mat g = iA*a;
		/*double gg = arma::dot(g, g);

		arma::mat c;
		if (arma::abs(A1*g - a).max() < eps)
			c = (1 / (1 + gg))*g.t()*iA;
		else
			c = pseudoinversa(a - A1*g, iA_ant, true);
		
		// iA = np.bmat([[iA - g*c],[c]]) em python
		iA = arma::join_vert(iA - g*c, c);
		*/
	}
	return iA;
}
arma::mat setValue(const arma::mat & A, float value)
{
	//A = value;
	//cout << A.n_cols;
	//A = arma::ones(A.n_rows, A.n_rows, 1)*value;
	//A.print("submatriz de A:");
	cout << "submatriz (" << A.n_rows << ", " << A.n_cols << ")" << endl;
	return arma::ones(A.n_rows, A.n_rows, 1)*value;
}

arma::mat cv2arma(cv::Mat &M_cv)
{	//Por enquanto so converte para double
	arma::mat M(reinterpret_cast <double*>(M_cv.data), M_cv.cols, M_cv.rows);
	return M.t();
}
void teste()
{
	double data[14] = { 2,2,3,1,2,3,4,-1,2,0,0,7,5,4 };
	cv::Mat opencv_mat = cv::Mat(7, 2, CV_64FC1, data);

	cout << opencv_mat;

	arma::mat arma_mat = cv2arma(opencv_mat);
	arma_mat.print("A, em armadillo:");

	arma::mat iA = pseudoinversa(arma_mat, arma_mat, true);
	//arma::mat iA = setValue(arma_mat, 3);
	iA.print("iA;");

	//pseudoinverse(arma_mat, arma_mat, true);
}