/*******************************************************************************

Intel Realsene SDK
The program starts Color and Depth Stream using Intel Realsense SDK
and converting its frame from PXCImage to Mat variable.
Easy for Image processing in Intel Realsense SDK Camera.

*******************************************************************************/


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

#include "pseudoinverse.h"
using namespace std;
using namespace cv;

bool eyeRegion(cv::Mat *frame, cv::Mat *eyeRect, PXCFaceData::LandmarkPoint *points, pxcI32 npoints)
{
	int width = 400;
	int height = 100;

	if (npoints == 0)
		return false;

	bool confidence = (int) points[14].confidenceWorld == 100 && (int) points[22].confidenceWorld == 100
		&& (int) points[26].confidenceWorld == 100 && (int) points[31].confidenceWorld == 100;

	if (!confidence)
		return false;

	auto p14 = cv::Point(points[14].image.x, points[14].image.y);
	auto p22 = cv::Point(points[22].image.x, points[22].image.y);
	auto p26 = cv::Point(points[26].image.x, points[26].image.y);
	auto p31 = cv::Point(points[31].image.x, points[31].image.y);

	double n0 = pow(p22.x - p14.x, 2) + pow(p22.y - p14.y, 2);
	double n1 = pow(p26.x - p31.x, 2) + pow(p26.y - p31.y, 2);
	double ratio = n1 / n0;
	printf("%f\n", ratio);
	if (ratio < 0.10 || ratio > 0.6)
		return false;

	// Nao sei somar vetores aqui
	cv::Point v1 = cv::Point(p22.x - p14.x, p22.y - p14.y);
	cv::Point v2 = cv::Point(-v1.y, v1.x);

	// Nem multiplicar por escalar
	double l1 = .1, l2 = .25;
	v1 = cv::Point((int)(((int)((double)v1.x)*l1)), (((int)((double)v1.y)*l1)));
	v2 = cv::Point((int)(((int)((double)v2.x)*l2)), (((int)((double)v2.y)*l2)));

	cv::Point r1 = cv::Point(p14.x - v2.x - v1.x, p14.y - v2.y - v1.y);
	cv::Point r2 = cv::Point(p14.x + v2.x - v1.x, p14.y + v2.y - v1.y);
	cv::Point r3 = cv::Point(p22.x + v2.x + v1.x, p22.y + v2.y + v1.y);
	cv::Point r4 = cv::Point(p22.x - v2.x + v1.x, p22.y - v2.y + v1.y);

	std::vector<Point2f> input;
	std::vector <Point2f> output;

	input.push_back(Point2f(r1.x, r1.y));
	input.push_back(Point2f(r2.x, r2.y));
	input.push_back(Point2f(r3.x, r3.y));
	input.push_back(Point2f(r4.x, r4.y));

	output.push_back(Point2f(    0,      0));
	output.push_back(Point2f(    0, height));
	output.push_back(Point2f(width, height));
	output.push_back(Point2f(width,      0));

	Mat lambda; 
	lambda = findHomography(input, output);
	*eyeRect = cv::Mat(height, width, PXCImage::PIXEL_FORMAT_RGB24);

	cv::Rect roi = cv::Rect(0,0,width,height);
	cv::warpPerspective(*frame, *eyeRect, lambda, eyeRect->size());
	Mat m = (*eyeRect)(roi);
	//Mat *eyeRect1 = NULL;
	//*eyeRect1 = (*eyeRect)(roi);
	m.copyTo(*eyeRect);

	cv::line(*frame, r1, r2, Scalar(255, 0, 0));
	cv::line(*frame, r2, r3, Scalar(255, 0, 0));
	cv::line(*frame, r3, r4, Scalar(255, 0, 0));
	cv::line(*frame, r4, r1, Scalar(255, 0, 0));
}


void drawRect(int row, int col) {
	//http://stackoverflow.com/questions/2981621/how-to-draw-to-screen-in-c
	HDC screenDC = ::GetDC(0);

	int width = GetSystemMetrics(SM_CXSCREEN);
	int height = GetSystemMetrics(SM_CYSCREEN);
	
	int ncol = 4, nrow = 4;
	int dy = height / (nrow + 2);
	int dx = width / (ncol + 2);

	int rx = dx*(col + 1);
	int ry = dy*(row + 1);
	int l = 40;

	//::Rectangle(screenDC, 0, 0, width, height);
	RECT rect = { 0,0,width, height };
	HBRUSH brush = CreateSolidBrush(RGB(30, 40, 40));
	FillRect(screenDC, &rect, brush);

	rect = { rx, ry, rx + l, ry + l };
	brush = CreateSolidBrush(RGB(255, 0, 0));
	FillRect(screenDC, &rect, brush);

	::ReleaseDC(0, screenDC);
}

Mat drawCircle(int i, int j, int m, int n)
{
	int width  = GetSystemMetrics(SM_CXSCREEN);
	int height = GetSystemMetrics(SM_CXSCREEN);

	Mat frame = cv::Mat::zeros(height, width, CV_8U);
	frame.setTo(200);

	int rad = 60;
	int radInc = 20;
	int hStep = width / (n + 2);
	int vStep = height/(m + 2);
	int x0 = hStep;
	int y0 = vStep;
	for (int circ = 0; circ < 4; circ++) {
		int x = x0 + j * hStep;
		int y = y0 + i * vStep;
		cv::circle(frame, cv::Point(x, y), rad, 255 * (circ % 2), -1, CV_AA);
		rad -= radInc;
		if (rad < 1) rad = 1;
	}
	return frame;
}

//x bool EnterFullscreen()
//x {
//x 	DEVMODE newSettings;
//x 	// now fill the DEVMODE with standard settings, mainly monitor frequenzy
//x 	EnumDisplaySettings(NULL, 0, &newSettings);
//x 	// set desired screen size and resolution	
//x 	newSettings.dmPelsWidth = 800;
//x 	newSettings.dmPelsHeight = 600;
//x 	newSettings.dmBitsPerPel = 16;
//x 	//set those flags to let the next function know what we want to change
//x 	newSettings.dmFields = DM_BITSPERPEL | DM_PELSWIDTH | DM_PELSHEIGHT;
//x 	// and apply the new settings
//x 	if (ChangeDisplaySettings(&newSettings, CDS_FULLSCREEN)
//x 		!= DISP_CHANGE_SUCCESSFUL)
//x 		return false; // in case of error
//x 	else return true;
//x }

int wmain(int argc, WCHAR* argv[])
{
	clock_t t1 = clock();
	teste();
	//printf("HELLOOOOOOO!");
	clock_t t2 = clock();
	double diff = double(t2 - t1) / CLOCKS_PER_SEC;
	printf("tempo para calcular inversa: %f", diff);
	return 1111;

	int flag = false; //MUDAR ISSO
	int width = 640, height = 480;
	cv::Mat frame1;
	PXCFaceData::LandmarkPoint *landmarks1 = NULL; //marcadores do primeiro frame com todos landmarks detectados

	cout << "Intel Realsense SDK Hacking using Opencv" << endl;
	cout << "Intel Realsense Camera SDK Frame Capture in opencv Mat Variable       -- by Deepak" << endl;
	cout << "Compiled with OpenCV version " << CV_VERSION << endl << endl;

	PXCSenseManager *psm = 0;
	psm = PXCSenseManager::CreateInstance();
	if (!psm) {
		wprintf_s(L"Unable to create the PXCSenseManager\n");
		return 1;
	}

	psm->EnableStream(PXCCapture::STREAM_TYPE_COLOR, width, height); //depth resolution
	psm->EnableFace();
	psm->Init();

	///////////// OPENCV
	IplImage *image = 0;
	CvSize gab_size;
	gab_size.height = height;
	gab_size.width = width;
	image = cvCreateImage(gab_size, 8, 3);

	PXCImage::ImageData data;
	unsigned char *rgb_data;

	PXCImage::ImageInfo rgb_info;

	int i = 0, j = 0;
	int ncol = 4, nrow = 4;
	bool recordingVideo = false;
	time_t begin, end;

	int frameCount = 0;
	cv::VideoWriter outputVideo;
	for (;;)
	{
		PXCFaceData::LandmarkPoint *headPoints = NULL;
		pxcI32 npoints(0);

		if (psm->AcquireFrame(true)<PXC_STATUS_NO_ERROR) break;

		////Matriz de rotacao
		PXCFaceModule *fmod = psm->QueryFace();

		if (fmod != NULL) {
			PXCFaceData *fdata = fmod->CreateOutput();
			fdata->Update();
			if (fdata)
			{
				pxcI32 nfaces = fdata->QueryNumberOfDetectedFaces();
				if (nfaces > 0)
				{
					PXCFaceData::Face *face = fdata->QueryFaceByIndex(0);
					PXCFaceData::LandmarksData *ldata = face->QueryLandmarks();

					if (ldata)
					{
						npoints = ldata->QueryNumPoints();
						headPoints = new PXCFaceData::LandmarkPoint[npoints];

						ldata->QueryPoints(headPoints);
					}
				}
				fdata->Release();
			}
		}

		PXCCapture::Sample *sample = psm->QuerySample();
		PXCImage *colorIm;

		// retrieve the image or frame by type from the sample
		colorIm = sample->color;
		PXCImage *color_image = colorIm;
		color_image->AcquireAccess(PXCImage::ACCESS_READ_WRITE, PXCImage::PIXEL_FORMAT_RGB24, &data);
		rgb_data = data.planes[0];
		
		for (int y = 0; y<height; y++)
		{
			for (int x = 0; x<width; x++)
			{
				for (int k = 0; k<3; k++)
				{
					image->imageData[y * width * 3 + x * 3 + k] = rgb_data[y * width * 3 + x * 3 + k];
				}
			}
		}

		color_image->ReleaseAccess(&data);
		cv::Mat rgb = cvarrToMat(image);

		//Mat rgbOut = Mat(rgb.rows, rgb.cols * 2, CV_8UC3);
		Mat eyeRect;
		if (eyeRegion(&rgb, &eyeRect, headPoints, npoints))
		{
			//eyeRect.copyTo(rgbOut(cv::Rect(rgb.cols, 0, eyeRect.cols, eyeRect.rows)));
			if (!recordingVideo)
			{
				recordingVideo = true;
				int fps = 15;
				string video = "video" + to_string(i) + "_" + to_string(j) + ".avi";
				outputVideo.open(video, CV_FOURCC('M', 'P', '4', '2'), 5, cv::Size(eyeRect.cols, eyeRect.rows));
				if (!outputVideo.isOpened())
				{
					printf("Output video could not be opened!");
					break;
				}

				cv::namedWindow("Name", CV_WINDOW_NORMAL);
				cv::setWindowProperty("Name", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
				cv::imshow("Name", drawCircle(i, j, nrow, ncol));

				begin = clock();
			}
			Mat rgbOut = Mat(eyeRect.rows, eyeRect.cols, CV_8UC3);
			eyeRect.copyTo(rgbOut);
			//outputVideo.write(rgbOut);
			outputVideo << rgbOut;
			//frameCount++;
			end = clock();
			double dt = (double(end - begin)) / CLOCKS_PER_SEC;
			if (dt >= 2)
			{
				outputVideo.release();
				recordingVideo = false;
				j++;
				if (j == ncol)
				{
					i++;
					j = 0;
					if (i == nrow)
						break;
				}

			}
		}
			

		if (cvWaitKey(10) >= 0)
			break;


		psm->ReleaseFrame();

	}
	cvReleaseImage(&image);
	psm->Release();
	return 0;
}