#include <math.h>
#include<stdio.h>
#include <opencv2/opencv.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
using namespace cv;
using namespace std;


//计算距离函数
double getDistance(Point A, Point B)
{
	double dis;
	dis=(A.x-B.x)*(A.x-B.x)+(A.y-B.y)*(A.y-B.y);
	return sqrt(dis);

}


int main( int argc, char** argv )
{
	//读取图片
	Mat srcImage=imread("大风车2.png");
	imshow("原始图",srcImage);
	//分离颜色通道
	vector<Mat> imgChannels;
	split(srcImage,imgChannels);


	//过滤颜色
	//Mat midImage2 = imgChannels.at(2)-imgChannels.at(0);
	Mat midImage2 = imgChannels.at(2)-imgChannels.at(0);
	
	//二值化
	threshold(midImage2,midImage2,100,255,THRESH_BINARY);
    //imshow("二值化图像",midImage2);
	
	//图像形态学操作：膨胀和开操作
	int elementsize=2;
	Mat kernel= getStructuringElement(MORPH_RECT,Size(elementsize*2+1,elementsize*2+1),Point(-1,-1));
   	dilate(midImage2,midImage2,kernel);
	//imshow("膨胀图像",midImage2);
	//elementsize =3;
	kernel= getStructuringElement(MORPH_RECT,Size(elementsize*2+1,elementsize*2+1),Point(-1,-1));
	morphologyEx(midImage2,midImage2,CV_MOP_CLOSE,kernel);
	//imshow("开操作",midImage2);


	
	//寻找轮廓
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	Mat dstImage = Mat::zeros(srcImage.rows, srcImage.cols, CV_8UC3);
	findContours( midImage2, contours, hierarchy,CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE );
	
	RotatedRect rect_tmp2;
	bool findTarget=0;

	if(hierarchy.size())
		for(int i=0;i>=0;i=hierarchy[i][0])
		{
			rect_tmp2=minAreaRect(contours[i]);
			Point2f P[4];

			rect_tmp2.points(P);

			Point2f srcRect[4];
			Point2f dstRect[4];

			double width;
			double height;

			width = getDistance(P[0],P[1]);
			height = getDistance(P[1],P[2]);

			if(width>height)
			{
				srcRect[0]=P[0];
				srcRect[1]=P[1];
				srcRect[2]=P[2];
				srcRect[3]=P[3];
			
			}
			else
			{
				double tem = width;
			    width = height;
			    height = tem;
				srcRect[0]=P[1];
				srcRect[1]=P[2];
				srcRect[2]=P[3];
				srcRect[3]=P[0];
			
			
			}
			Scalar color(rand()&255,rand()&255,rand()&255);
			drawContours(srcImage,contours,i,color,2,8,hierarchy);
			//drawContours(srcImage,contours,i,color,2,8,hierarchy);
			
			double area=height*width;
			if(area>5000)
			{
				cout <<hierarchy[i]<<endl;

				dstRect[0]=Point2f(0,0);
				dstRect[1]=Point2f(width,0);
				dstRect[2]=Point2f(width,height);
				dstRect[3]=Point2f(0,height);
				//透视变换 ，矫正
				Mat transform = getPerspectiveTransform(srcRect,dstRect);
				Mat perspectMat;
				warpPerspective(midImage2,perspectMat,transform,midImage2.size(),INTER_LINEAR);

				imshow("warpdst",perspectMat);

				Mat testim;
				testim = perspectMat(Rect(0,0,width,height));
				imshow("testim",testim);






			}
		
		
		
		
		
		
		}
		imshow("111",srcImage);
		//imshow( "轮廓图", dstImage );
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	/*
	//画出轮廓
	int index = 0;
	for( ; index >= 0; index = hierarchy[index][0] )
	{
		
		drawContours( dstImage, contours, index, Scalar(0,0,255), 1, 8, hierarchy );
	}

	imshow( "轮廓图", dstImage );
	*/

	waitKey(0);

}
