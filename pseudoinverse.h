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

using namespace Eigen;
using namespace std;
using namespace cv;

void printMatrix(Mat A)
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
}

void printEigenMatrix(Eigen::MatrixXf *A)
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
}

template<typename Number, typename T1>
void foo(Number &x, const Eigen::DenseBase<T1>& matrix)
{
	MatrixXf AA = matrix.eval();
	cout << AA.lpNorm<2>();
	//matrix.template lpNorm<2>();
	//return matrix.lpNorm<2>();
}

//estava const Ref <Eigen::MatrixXf>&A,
//estava Eigen::MatrixBase<T1> & iA,
template<typename T1, typename T2> //typename
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

	if (iA_isNULL) //Deve ser chamada tamb�m quando n == 1
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
}

Mat pinv(Mat A, Mat *iA_1 = NULL)
{
	// Calcula a pseudoinversa da matriz A = [a1 | ... | ak] //
	 // onde iA_1 � a pseudoinversa de [a1 | ... | ak-1]       //
	 // s� funciona para matrizes do tipo double (CV_64FC1)   //
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


void teste()
{
	double data[14] = { 2,2,3,1,2,3,4,-1,2,0,0,7,5,4 };
	Mat A = Mat(7, 2, CV_64FC1, data);

	Eigen::MatrixXf A_eigen, iA_eigen;
	cv2eigen(A, A_eigen);

	cout << "MATRIZ A" << endl;
	cout << A_eigen << endl;
	cout << "------------------------------------------------------" << endl << endl;

	pinvEigen(A_eigen, iA_eigen, true); //A_eigen.block<A_eigen.rows(), A_eigen.cols()>(0,0)

	cout << "PSEUDOINVERSA DE A" << endl;
	cout << iA_eigen << endl;
	cout << "------------------------------------------------------" << endl << endl;
/*
	cout << "A" << endl;
	cout << A_eigen;
	cout << "pinv(A)" << endl;
	cout << iA_eigen;
*/
	//printMatrix(A);
	//time_t begin, end;
	//begin = clock();
	//Mat iA = pinv(A, NULL);
	//end = clock();
	//double dt = (double(end - begin)) / CLOCKS_PER_SEC;
	//printf("%f segundos", dt);
	//printMatrix(iA);

	/*Eigen::MatrixXf iA_eigen;
	cv2eigen(iA, iA_eigen);
	printEigenMatrix(&iA_eigen);
	*/
	//Ref<MatrixXd> sub = iA_eigen.block(1, 2, 1, 1);
	//iA_eigen.block<1, 2>(1, 1);
}