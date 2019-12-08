#include<iostream>
#include<opencv2/opencv.hpp>
#include "opencv2/highgui/highgui.hpp"   
#include "opencv2/imgproc/imgproc.hpp"
   
using namespace cv;
using namespace std;

Mat src,dst;
int iLowH = 50;   
     int iHighH = 77;   
     int iLowS = 149;    
     int iHighS = 255;   
     int iLowV = 97;   
     int iHighV = 125;   
int i=0;
 int x_final=0;int x_n[2];
 int y_final=0;int y_n[2];
void track_on(int ,void*);
int main(int argc, char** argv ){
src=imread("/home/zxc/桌面/C++/picture/gg.jpeg");
if(!src.data){
printf("could not load image...\n");
return -1;
}
namedWindow("green round",CV_WINDOW_AUTOSIZE);
namedWindow("round",CV_WINDOW_AUTOSIZE);
namedWindow("Trackbar",CV_WINDOW_AUTOSIZE);


//Create trackbars in "Control" window   
  
        createTrackbar("LowH", "Trackbar", &iLowH, 179,track_on); //Hue (0 - 179) 
  
        createTrackbar("HighH", "Trackbar", &iHighH, 179,track_on);   

        createTrackbar("LowS", "Trackbar", &iLowS, 255,track_on);  

//Saturation (0 - 255)   
  
        createTrackbar("HighS", "Trackbar", &iHighS, 255,track_on);   

        createTrackbar("LowV", "Trackbar", &iLowV, 255,track_on);  
  
//Value (0 - 255)   
  
        createTrackbar("HighV", "Trackbar", &iHighV, 255,track_on);   
track_on(0,0);
  
        //开操作 (去除一些噪点)  
         Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));  
         morphologyEx(dst, dst, MORPH_OPEN, element);  
         //闭操作 (连接一些连通域)  
         morphologyEx(dst, dst, MORPH_CLOSE, element);  
// 寻找轮廓
	vector<vector<Point>> contours;	
	vector<Vec4i> hireachy;
	findContours(dst, contours, hireachy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point());

	cout << "----------vector<vector<Point>> contours------------" << endl;
 	for (size_t i = 0; i < contours.size(); i++)
	{
		cout << "第" << i << "行：";
		for (size_t j = 0; j < contours[i].size(); j++)
		{
			cout<<contours[i][j]<<endl;
		}
		cout << endl;
	}
	cout << endl;

	Point circle_center;              //定义圆心坐标
	for (auto t = 0; t < contours.size(); ++t)
	{
		// 面积过滤
		double area = contourArea(contours[t]);     //计算点集所围区域的面积
		if (area < 100)            //晒选出轮廓面积大于100的轮廓
			continue;
		// 横纵比过滤
		Rect rect = boundingRect(contours[t]);            // 求点集的最小直立外包矩形

                cout<<"rect:"<<endl;
                cout<<rect<<"\n"<<endl;

		float ratio = float(rect.width) / float(rect.height);        //求出宽高比
 
		if (ratio < 1.1 && ratio > 0.9)       //因为圆的外接直立矩形肯定近似于一个正方形，因此宽高比接近1.0
		{ 
			drawContours(src, contours, t, Scalar(0, 0, 255), 1, LINE_AA, Mat(), 0, Point());    //画圆轮廓
                         x_n[i] = rect.x + rect.width / 2;  x_final=x_final+ x_n[i];
			 y_n[i] = rect.y + rect.height / 2; y_final=y_final+ y_n[i];  
                         i++;                                      
		}

	}
                        int x=x_final/2;int y=y_final/2;
                        circle_center = Point(x, y);          //得到圆心坐标
                        cout << "圆心坐标：" << "宽:"<<circle_center.x<<" "<<"高:"<< circle_center.y << endl;
			circle(src, circle_center, 2, Scalar(0, 255, 255), 2, 8, 0);  //画圆心

imshow("green round",src); 
imshow("round",dst);


waitKey(0);
    return 0;     

}






void track_on(int ,void*){
Mat imgHSV;   
  
        vector<Mat> hsvSplit;   
  
        cvtColor(src, imgHSV, COLOR_BGR2HSV);  
  
//Convert the captured frame from BGR to HSV   
  
         //因为我们读取的是彩色图，直方图均衡化需要在HSV空间做   
  
        split(imgHSV, hsvSplit);   
  
        equalizeHist(hsvSplit[2],hsvSplit[2]);   
  
        merge(hsvSplit,imgHSV);   
  
       // Mat imgThresholded;   
  
        inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), dst); //Threshold the image   
       
}
