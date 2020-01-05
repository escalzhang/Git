//只适用与C++11版本下
#include <time.h>
#include <math.h>
#include <stdio.h>
#include <iostream>
#include <string.h>
#include  <chrono>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
using namespace cv;
using namespace std;
clock_t start_time;
clock_t end_time;

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
//min2
/*static bool CircleInfo2(std::vector<cv::Point2f>& pts, cv::Point2f& center, float& radius)
{
    center = cv::Point2d(0, 0);
    radius = 0.0;
    if (pts.size() < 3) return false;;

    double sumX = 0.0;
    double sumY = 0.0;
    double sumX2 = 0.0;
    double sumY2 = 0.0;
    double sumX3 = 0.0;
    double sumY3 = 0.0;
    double sumXY = 0.0;
    double sumX1Y2 = 0.0;
    double sumX2Y1 = 0.0;
    const double N = (double)pts.size();
    for (int i = 0; i < pts.size(); ++i)
    {
        double x = pts.at(i).x;
        double y = pts.at(i).y;
        double x2 = x * x;
        double y2 = y * y;
        double x3 = x2 *x;
        double y3 = y2 *y;
        double xy = x * y;
        double x1y2 = x * y2;
        double x2y1 = x2 * y;

        sumX += x;
        sumY += y;
        sumX2 += x2;
        sumY2 += y2;
        sumX3 += x3;
        sumY3 += y3;
        sumXY += xy;
        sumX1Y2 += x1y2;
        sumX2Y1 += x2y1;
    }
    double C = N * sumX2 - sumX * sumX;
    double D = N * sumXY - sumX * sumY;
    double E = N * sumX3 + N * sumX1Y2 - (sumX2 + sumY2) * sumX;
    double G = N * sumY2 - sumY * sumY;
    double H = N * sumX2Y1 + N * sumY3 - (sumX2 + sumY2) * sumY;

    double denominator = C * G - D * D;
    if (std::abs(denominator) < DBL_EPSILON) return false;
    double a = (H * D - E * G) / (denominator);
    denominator = D * D - G * C;
    if (std::abs(denominator) < DBL_EPSILON) return false;
    double b = (H * C - E * D) / (denominator);
    double c = -(a * sumX + b * sumY + sumX2 + sumY2) / N;

    center.x = a / (-2);
    center.y = b / (-2);
    radius = std::sqrt(a * a + b * b - 4 * c) / 2;
    return true;
}

*/
int main( int argc, char** argv )
{
   VideoCapture capture;
    capture.open("1.avi");
Mat drawcircle;
Mat templ[9];
templ[1]=imread("template1",IMREAD_GRAYSCALE);
templ[2]=imread("template2",IMREAD_GRAYSCALE);
templ[3]=imread("template3",IMREAD_GRAYSCALE);
templ[4]=imread("template4",IMREAD_GRAYSCALE);
templ[5]=imread("template5",IMREAD_GRAYSCALE);
templ[6]=imread("template6",IMREAD_GRAYSCALE);
templ[7]=imread("template7",IMREAD_GRAYSCALE);
templ[8]=imread("template8",IMREAD_GRAYSCALE);
   while(1)
    {
       start_time=clock();

 auto t1 = chrono::high_resolution_clock::now();

    //输入模板图片





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
   drawcircle=Mat(srcImage.rows,srcImage.cols, CV_8UC3, Scalar(0, 0, 0));
    //读取图片
   // srcImage=imread("1.png");


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
      //  imshow("二值化图像",midImage2);

    //图像形态学操作：膨胀和开操作
    int elementsize=2;
    Mat kernel= getStructuringElement(MORPH_RECT,Size(elementsize+1,elementsize+1),Point(-1,-1));
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
auto t3 = chrono::high_resolution_clock::now();
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
            //Scalar color(rand()&255,rand()&255,rand()&255);
            //drawContours(srcImage,contours,i,color,2,8,hierarchy);
            //drawContours(srcImage,contours,i,color,2,8,hierarchy);
auto t4 = chrono::high_resolution_clock::now();
            double area=height*width;
            if(area>5000)
            {
             //画出轮廓的外接矩形
               /* for(int j=0;j<4;j++)
                {
                    line(srcImage,P[j],P[(j+1)%4],Scalar (255,200,200),3,LINE_AA);


                }
                */
              //  imshow("112222",srcImage);
          //      cout <<hierarchy[i]<<endl;

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
                //string s="leaf"+to_string((int)cnnt);

               // imwrite("./img/"+s+".jpg",testim);

                //imshow("testim",testim);

               // if(testim.empty())
                //{
                //    cout<<"filed open"<<endl;
                //    return -1;

                //}

                Point matchLoc;
                double value;
                Mat tmp1;
                resize(testim,tmp1,Size(42,20));
                //imwrite("./tmp/"+s+".jpg",tmp1);
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
                //cout <<Vvalue1[maxv1]<<endl;
                //cout <<Vvalue2[maxv2]<<endl;

              //  cout<<"--------------------------------------------------------------"<<endl;
        //根据对比完的条件画出目标
                /*if(Vvalue1[maxv1]>1||Vvalue1[maxv1]<0)
                 {Vvalue1[maxv1]=1;}

                 if(Vvalue2[maxv2]>1||Vvalue2[maxv2]<0)
                   {Vvalue2[maxv2]=0;}
                */
                if(Vvalue1[maxv1]>Vvalue2[maxv2]&&Vvalue1[maxv1]>0.6)
                {
                   /* for(int j1=0;j1<4;j1++)
                    {
                        line(srcImage,P[j1],P[(j1+1)%4],Scalar (0,255,0),3,LINE_AA);


                    }
                    */



                    if(hierarchy[i][2]>=0)
                    {

                        RotatedRect rect_tmp=minAreaRect(contours[hierarchy[i][2]]);
                        Point2f Pnt[4];
                        rect_tmp.points(Pnt);
                        /* const float maxHWRatio=0.7153846;
                        const float maxArea=1300;
                        const float minArea=1000;

                        float width=rect_tmp.size.width;
                        float height=rect_tmp.size.height;
                        if(height>width)
                        {
                            float ttt=height;
                            height =width;
                            width =ttt;


                        }
                        float area =width*height;

                        if(height/width>maxHWRatio||area>maxArea||area<minArea)
                        {

                            cout <<"hw "<<height/width<<"area "<<area<<endl;

                           for(int j=0;j<4;++j)
                            {
                                line(srcImage,Pnt[j],Pnt[(j+1)%4],Scalar(255,0,255),4);

                            }


                            //circle(drawcircle,centerP,1,Scalar(0,0,255),1);


                           // continue;



                        }*/


                        Point centerP=rect_tmp.center;
                        //打击点
                        circle(srcImage,centerP,2,Scalar(255,0,255),2);
                        cout << "center：" << "x:"<<centerP.x<<" "<<"y:"<< centerP.y << endl;


                    }





                }


              //  cout <<Vvalue1[maxv1]<<endl;
              //  cout <<Vvalue2[maxv2]<<endl;


              //  cout << "two period: " << (static_cast<chrono::duration<double, std::milli>>(t4 - t1)).count() << " ms" << endl;




            }




        }


    //imshow("final",srcImage);
    //waitKey(0);

   imshow("avi",srcImage);
   end_time=clock() ;
   auto t2 = chrono::high_resolution_clock::now();
   // cout << "Total period: " << (static_cast<chrono::duration<double, std::milli>>(t2 - t1)).count() << " ms" << endl;
   // cout << "one period: " << (static_cast<chrono::duration<double, std::milli>>(t3 - t1)).count() << " ms" << endl;

    cout<<"time:"<<(double)(end_time-start_time)/CLOCKS_PER_SEC<<endl;

  waitKey(1);
}

//   imshow("2",drawcircle);




}

