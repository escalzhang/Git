#include <time.h>
#include <math.h>
#include <stdio.h>
#include <iostream>
#include <string.h>

#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
using namespace cv;
using namespace std;
clock_t start_time;
clock_t end_time;
clock_t mid_time;
//计算距离函数
double getDistance(Point A,Point B)
{
    double dis;
    dis =(A.x-B.x)*(A.x-B.x)+(A.y-B.y)*(A.y-B.y);
    return sqrt(dis);

}



//主函数
int main( int argc, char** argv )
{

    //读取视频
    VideoCapture capture;
    capture.open("1.avi");

int minv=0;
Mat src;
double sss[40];

    //初始化一些数据
    //src=imread("1.png");

    while(1){
    start_time=clock();
    for(int k=0;k<=40;k++)
    sss[k]=999999;
     capture>>src;

    GaussianBlur(src,src,Size(3,3),0,0);
    //分离颜色通道
        vector <Mat> Channels;
        split(src,Channels);
    mid_time=clock();
        //如果敌人为蓝色
        //Mat mid = Channels.at(0)-Channels.at(2);

        //如果敌人为红色
        Mat mid  = Channels.at(2)-Channels.at(0);
         //二值化
     threshold(mid,mid,100,255,THRESH_BINARY);
 //   imshow("腐蚀",mid);
    //图像形态学操作
           int elementsize=2;
         Mat kernel= getStructuringElement(MORPH_RECT,Size(elementsize+1,elementsize+1),Point(-1,-1));
         //膨胀
     erode(mid,mid,kernel);

     kernel= getStructuringElement(MORPH_RECT,Size(elementsize*2+1,elementsize*2+1),Point(-1,-1));
     //闭操作
           // morphologyEx(mid,mid,CV_MOP_CLOSE,kernel);
        //imshow("close",mid);
     vector<vector<Point> > contours;
        vector<Vec4i> hierarchy;
        Mat dst = Mat::zeros(src.rows, src.cols, CV_8UC3);
    findContours( mid, contours, hierarchy,CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE );
        RotatedRect rect_tmp2;
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
double area=height*width;
    if(area>500)
    sss[i]=area;






    //画出轮廓的外接矩形
           // for(int j=0;j<4;j++)
             //{
               //  line(src,P[j],P[(j+1)%4],Scalar (255,200,200),3,4);


             //}








        }
//drawContours(src,contours,-1,Scalar(255,255,0),2,8,hierarchy);

        for(int t1=0;t1<40;t1++)
{
    if(sss[t1]<sss[minv])
    minv =t1;


}

      if(hierarchy[minv][2]>=0)
               {
                   RotatedRect rect_final =minAreaRect(contours[hierarchy[minv][2]]);

                   Point center_point =rect_final.center;
                   circle(src,center_point,2,Scalar(255,100,0),2);
                   cout<<"center:("<<center_point.x<<","<<center_point.y<<")";

               }

   imshow("avi",src);
    end_time=clock() ;
    cout<<"time:"<<(double)(end_time-start_time)/CLOCKS_PER_SEC<<endl;
   // cout<<"midtime:"<<(double)(mid_time-start_time)/CLOCKS_PER_SEC<<endl;

    waitKey(1);


    }





}



