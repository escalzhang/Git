//只适用与C++11版本下

#include <math.h>
#include<stdio.h>
#include <iostream>
#include<string.h>
#include<stdlib.h>
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

//模板匹配函数
double  templateMatch(Mat image,Mat tepl,Point &Point,int method)
{
    imshow("11",tepl);
    int result_cols = image.cols - tepl.cols + 1;
    int result_rows = image.rows - tepl.rows + 1;
    //    cout <<result_cols<<" "<<result_rows<<endl;
        cv::Mat result = cv::Mat( result_cols, result_rows, CV_32FC1 );
        cv::matchTemplate(image, tepl, result, method );

        double minVal, maxVal;
        cv::Point minLoc, maxLoc;
        cv::minMaxLoc( result, &minVal, &maxVal, &minLoc, &maxLoc, Mat() );

        switch(method)
        {
        case CV_TM_SQDIFF:
        case CV_TM_SQDIFF_NORMED:
            Point = minLoc;
            return minVal;

        default:
            Point = maxLoc;
            return maxVal;

        }



}



int main( int argc, char** argv )
{
    VideoCapture capture;
    capture.open("2.avi");

    while(1)
    {


    //输入模板图片
    Mat templ[9];
    templ[1]=imread("template1",IMREAD_GRAYSCALE);
    templ[2]=imread("template2",IMREAD_GRAYSCALE);
    templ[3]=imread("template3",IMREAD_GRAYSCALE);
    templ[4]=imread("template4",IMREAD_GRAYSCALE);
    templ[5]=imread("template5",IMREAD_GRAYSCALE);
    templ[6]=imread("template6",IMREAD_GRAYSCALE);
    templ[7]=imread("template7",IMREAD_GRAYSCALE);
    templ[8]=imread("template8",IMREAD_GRAYSCALE);




    for(int num=1;num<=8;num++)
    {

        if(templ[num].empty())
        {

            cout<<"read error"<<endl;
        }

    }

  /*

*/

               int cnnt =0;
    Mat srcImage;
    capture>>srcImage;

    //读取图片
    //srcImage=imread("1.png");


    //分离颜色通道
    vector<Mat> imgChannels;
    split(srcImage,imgChannels);

    //过滤颜色
    //敌人为蓝色
    //Mat midImage2 = imgChannels.at(0)-imgChannels.at(2);

    //敌人为红色
    Mat midImage2 = imgChannels.at(2)-imgChannels.at(0);
    //imshow("112222",midImage2);

    //二值化
    threshold(midImage2,midImage2,100,255,THRESH_BINARY);
        imshow("二值化图像",midImage2);

    //图像形态学操作：膨胀和开操作
    int elementsize=2;
    Mat kernel= getStructuringElement(MORPH_RECT,Size(elementsize+1,elementsize+1),Point(-1,-1));
       dilate(midImage2,midImage2,kernel);
    //imshow("膨胀图像",midImage2);
    //elementsize =3;
    kernel= getStructuringElement(MORPH_RECT,Size(elementsize*2+1,elementsize*2+1),Point(-1,-1));
    morphologyEx(midImage2,midImage2,CV_MOP_CLOSE,kernel);
    imshow("开操作",midImage2);



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
            //drawContours(srcImage,contours,i,color,2,8,hierarchy);
            //drawContours(srcImage,contours,i,color,2,8,hierarchy);

            double area=height*width;
            if(area>5000)
            {
             //画出轮廓的外接矩形
                for(int j=0;j<4;j++)
                {
                    line(srcImage,P[j],P[(j+1)%4],Scalar (255,200,200),3,LINE_AA);


                }
                imshow("112222",srcImage);
                cout <<hierarchy[i]<<endl;

                dstRect[0]=Point2f(0,0);
                dstRect[1]=Point2f(width,0);
                dstRect[2]=Point2f(width,height);
                dstRect[3]=Point2f(0,height);
                //透视变换 ，矫正
                Mat transform = getPerspectiveTransform(srcRect,dstRect);
                Mat perspectMat;
                warpPerspective(midImage2,perspectMat,transform,midImage2.size(),INTER_LINEAR);

                //imshow("warpdst",perspectMat);

                Mat testim;
                testim = perspectMat(Rect(0,0,width,height));


                //保存模板图片
                string s="leaf"+to_string((int)cnnt);

                imwrite("./img/"+s+".jpg",testim);

                //imshow("testim",testim);

                if(testim.empty())
                {
                    cout<<"filed open"<<endl;
                    return -1;

                }

                Point matchLoc;
                double value;
                Mat tmp1;
                resize(testim,tmp1,Size(42,20));
                imwrite("./tmp/"+s+".jpg",tmp1);
                cnnt++;
                //imshow("temp1",tmp1);

        //将获取的图像和模板图片进行对比
                vector<double> Vvalue1;
                vector<double> Vvalue2;

                   int c=1;
                for(;c<7;c++)
                {
                    value =templateMatch(tmp1,templ[c],matchLoc,5);
                   Vvalue1.push_back(value);

                }
int b;
                for(b=7; b<=8;b++)
                {
                    value =templateMatch(tmp1,templ[b],matchLoc,5);
                    Vvalue2.push_back(value);


                }

                int maxv1 =0,maxv2=0;

                for(int t1=0;t1<6;t1++)
                {
                   // cout <<Vvalue1[t1]<<endl;

                    if(Vvalue1[t1]>Vvalue1[maxv1])
                    {
                        maxv1 =t1;
                    }

                }
                cout<<endl;

                for(int t2=0;t2<2;t2++)
                {
                    //cout<<Vvalue2[t2]<<endl;
                    if(Vvalue2[t2]>Vvalue2[maxv2])
                    {
                        maxv2 =t2;
                    }


                }
                //cout <<endl;
                cout <<Vvalue1[maxv1]<<endl;
                cout <<Vvalue2[maxv2]<<endl;

                cout<<"--------------------------------------------------------------"<<endl;
        //根据对比完的条件画出目标
                /*if(Vvalue1[maxv1]>1||Vvalue1[maxv1]<0)
                 {Vvalue1[maxv1]=1;}

                 if(Vvalue2[maxv2]>1||Vvalue2[maxv2]<0)
                   {Vvalue2[maxv2]=0;}
                */
                if(Vvalue1[maxv1]>Vvalue2[maxv2]&&Vvalue1[maxv1]>0.6)
                {
                    for(int j1=0;j1<4;j1++)
                    {
                        line(srcImage,P[j1],P[(j1+1)%4],Scalar (0,255,0),3,LINE_AA);


                    }




                }


                cout <<Vvalue1[maxv1]<<endl;
                cout <<Vvalue2[maxv2]<<endl;






            }




        }


    //imshow("final",srcImage);
    //waitKey(0);

   imshow("avi",srcImage);
    waitKey(30);

}






}

