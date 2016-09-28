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
		printf("%.7f\ ]\n", (*A)(i, n-1));
	}
	printf("\n");
	//A->lpNorm<Infinity>();
}
/*Mat pinv(Mat A, Mat *iA_1 = NULL)
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
}*/

Eigen::MatrixXf pinv(
	const Ref <const MatrixXf>& A,
	const Ref <const MatrixXf> &iA_1 = NULL)
{
	Ref <MatrixXf> &iA;

	const double eps = 0.0000000001;
	int m = A->rows();
	int n = A->cols();

	if (n == 1)
	{
		double x = A->squaredNorm();
		iA = A->transpose();
		if (x != 0)
			iA /= x;
		return iA;
	}

	int k0;
	if (iA_1 == NULL)
	{
		k0 = 1;
		iA = pinv(A.topLeftCorner(1, m));
	}
	else
	{
		k0 = n - 1;
		iA = iA_1;
	}

	for (int k = k0; k < n; k++)
	{
		const Ref <const Matrix> A1 = A.topLeftCorner(k, m);
		const Ref <const Matrix> a = A.col(k);
		MatrixXf g = iA*a;
		double gg = g.squaredNorm();

		MatrixXf c;
		if ((A1*g - a).lpNorm <Infinity>() < eps)
		{
			g = g.transpose();
			c = (1. / (1 + gg))*g*iA;
			g = g.transpose();
		}
		else
			c = pinv(a - A1*g);
		MatrixXf ret(iA.rows() + 1, iA.cols());
		ret << a - A1*g, c;
		return ret
	}
}
void teste()
{
	/*Mat I(Mat::eye(cv::Size(3,3), CV_64FC1)); //CV_64FC1
	I.at <double>(1, 0) = 3;
	printf("HERE!");*/

	//double data[4] = { 1,3,2,3 };
	//Mat A = Mat(4, 1, CV_64FC1, data);
	double data[14] = { 2,2,3,1,2,3,4,-1,2,0,0,7,5,4 };
	Mat A = Mat(2, 7, CV_64FC1, data);
	printMatrix(A);
	time_t begin, end;
	begin = clock();
	Mat iA = pinv(A, NULL);
	end = clock();
	double dt = (double(end - begin)) / CLOCKS_PER_SEC;
	printf("%f segundos", dt);
	printMatrix(iA);

	Eigen::MatrixXf iA_eigen;
	cv2eigen(iA, iA_eigen);
	printEigenMatrix(&iA_eigen);

	//Ref<MatrixXd> sub = iA_eigen.block(1, 2, 1, 1);
	//iA_eigen.block<1, 2>(1, 1);
}