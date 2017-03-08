// GroundTruthVerification.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <string>  
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\opencv.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <sstream>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;

int upperH = 31, lowerH = 10;
int upperS = 255, lowerS = 50;
int upperV = 255, lowerV = 120;

//category
/*
白天陰天 1
白天晴天 2
白天雨天 3
黃昏 4
晚上晴天 5
晚上雨天 6
黃昏門檻 黃色25%以上
*/

double DetectDusk(Mat image)
{
	cv::Vec3b pixel;

	int h;
	int s;
	int v;
	int range = 30;

	double trueCount = 0;
	double falseCount = 0;
	double cb, cr;
	double persentage = 0.0;
	Mat hsv;
	cvtColor(image, hsv, CV_BGR2HSV);

	for (int i = 0; i < hsv.rows / 2; i++)
	for (int j = hsv.cols / 3; j < hsv.cols * 0.67; j++)
	{

		h = hsv.at<cv::Vec3b>(i, j)[0];
		s = hsv.at<cv::Vec3b>(i, j)[1];
		v = hsv.at<cv::Vec3b>(i, j)[2];

		if ((h >= lowerH && h <= upperH) && (s >= lowerS && s <= upperS) && (v >= lowerV && v <= upperV))
		{
			hsv.at<cv::Vec3b>(i, j)[0] = 255;
			hsv.at<cv::Vec3b>(i, j)[1] = 255;
			hsv.at<cv::Vec3b>(i, j)[2] = 255;
			trueCount++;
		}
		else
		{
			hsv.at<cv::Vec3b>(i, j) = 0;
			falseCount++;
		}
	}
	cout << "trueCount :" << trueCount << endl;
	cout << "falseCount :" << falseCount << endl;
	persentage = (trueCount / (trueCount + falseCount)) * 100.0;
	cout << "Persentage : " << persentage << endl;
	//imshow("Result", image);

	return persentage;
}



string daylightRainDropsCascadeName = "raindrops_cascade_daytime_300_100_30.xml";
string nightRainDropsCascadeName = "raindrops_cascade_night_300_100_21_haar.xml";
CascadeClassifier daylightRainDropsCascade;
CascadeClassifier nightRainDropsCascade;

int detectRainDrops(Mat image, string weather)
{


	std::vector<Rect> drops;
	Mat gray;
	Mat imageRoi = image.clone();
	imageRoi = image(Rect(0, 0, image.cols, image.rows / 2));
	cvtColor(imageRoi, gray, CV_BGR2GRAY);
	equalizeHist(gray, gray);

	if (weather == "daytime"){
		daylightRainDropsCascade.detectMultiScale(gray, drops, 1.1, 8, 0, Size(12, 12), Size(36, 36));
	}
	else if (weather == "night"){
		nightRainDropsCascade.detectMultiScale(gray, drops, 1.1, 1, 0, Size(1, 1), Size(48, 48));
	}
	else {
		return -1;
	}


	cout << "drops count : " << drops.size() << endl;

	for (size_t i = 0; i < drops.size(); i++)
	{
		Point center(drops[i].x + drops[i].width*0.5, drops[i].y + drops[i].height*0.5);
		ellipse(image, center, Size(drops[i].width*0.5, drops[i].height*0.5), 0, 0, 360, Scalar(255, 0, 255), 1, 8, 0);
	}
	//-- Show what you got
	//namedWindow("Raindrops", 0);
	//imshow("Raindrops", image);
	return drops.size();
}



double detectBrightness(Mat image){
	double average = 0;
	double sum = 0;
	int pointCount = 0;
	Mat hsv;

	cvtColor(image, hsv, CV_BGR2HSV);

	for (int j = 0; j < hsv.rows / 2; j = j + 10){
	//cout << "J = " << j << endl;
		for (int i = hsv.cols / 3; i < (int)(hsv.cols*0.67); i = i + 10){
			pointCount++;
			cv::Vec3b pixel = hsv.at<Vec3b>(j, i);
			sum = sum + (int)pixel[2];
		}
	}

	average = sum / pointCount;

	cout << "Point Count : " << pointCount << endl;
	cout << "Average : " << average << endl;

	

	return average;
}

float calcBlurriness(const Mat src)
{
	Mat Gx, Gy;
	Sobel(src, Gx, CV_32F, 1, 0);
	Sobel(src, Gy, CV_32F, 0, 1);
	cv::Mat FM = Gx.mul(Gx) + Gy.mul(Gy);
	double focusMeasure = cv::mean(FM).val[0];
	return focusMeasure;
	/*double normGx = norm(Gx);
	double normGy = norm(Gy);
	double sumSq = normGx * normGx + normGy * normGy;
	return static_cast<float>(1. / (sumSq / src.size().area() + 1e-6));*/
}


double varianceOfLaplacian(const cv::Mat& src)
{
	cv::Mat lap;
	cv::Laplacian(src, lap, CV_64F);

	cv::Scalar mu, sigma;
	cv::meanStdDev(lap, mu, sigma);

	double focusMeasure = sigma.val[0] * sigma.val[0];
	return focusMeasure;
}


int main()
{
	string imagePath = "frames/";
	string imageExtension = ".jpg";
	Mat src, srcHSV, dst;
	ofstream outFile("result.txt");
	double average = 0;
	double duskPersentage = 0;
	int dropCount = 0;


	//-- 1. Load the cascades
	if (!daylightRainDropsCascade.load(daylightRainDropsCascadeName)) {
		cout << "Cannot load daylight rain drop cascade file !!!" << endl;
		system("pause");
		return -1;
	}

	if (!nightRainDropsCascade.load(nightRainDropsCascadeName)) {
		cout << "Cannot load night rain drop cascade file !!!" << endl;
		system("pause");
		return -1;
	}



	for(int i = 1; i <= 2400; i++){
		stringstream ss;
		stringstream ss2;
		ss << imagePath << i << imageExtension;
		
		//src = imread(ss.str());
		//src = imread("frames/12.jpg");
		//imshow("Src", src);		
		
		
		average = detectBrightness(src);
		if (average >= 200){ //晴天門檻
			duskPersentage = DetectDusk(src);
			if (duskPersentage >= 25){ //(晴天)黃昏
				ss2 << i << imageExtension << "," << 4;
				outFile << ss2.str() << endl;
			}
			else{ //白天晴天
				ss2 << i << imageExtension << "," << 2;
				outFile << ss2.str() << endl;
			}

		}
		else if (average >= 90 && average < 200){ //陰天門檻
			duskPersentage = DetectDusk(src);
			if (duskPersentage >= 15){ //(陰天)黃昏
				ss2 << i << imageExtension << "," << 4;
				outFile << ss2.str() << endl;
			}
			else{ //白天陰天
				dropCount = detectRainDrops(src, "daytime");
				if (dropCount >= 100){ //白天雨天
					ss2 << i << imageExtension << "," << 3;
					outFile << ss2.str() << endl;
				}
				else{
					ss2 << i << imageExtension << "," << 1;
					outFile << ss2.str() << endl;
				}
			}
		}
		else if (average < 90){ //晚上門檻
			dropCount = detectRainDrops(src, "night");
			if (dropCount >= 20){ //晚上雨天
				ss2 << i << imageExtension << "," << 6;
				outFile << ss2.str() << endl;
			}
			else{ //晚上無雨(晚上晴天)
				ss2 << i << imageExtension << "," << 5;
				outFile << ss2.str() << endl;
			}
		}
		
		cout << ss2.str() << endl;
		cout << "=====================" << endl;
		waitKey(0);
	}

	outFile.close();



	return 0;
}

