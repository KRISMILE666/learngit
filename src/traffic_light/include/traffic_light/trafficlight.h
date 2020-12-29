#ifndef TRAFFICLIGHT_H
#define TRAFFICLIGHT_H

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <stdio.h>
using namespace cv;
using namespace std;

extern vector<vector<Point>> contours;
extern vector<Rect> boundRect;

//计算图片亮度
float light(Mat img)
{
    Scalar scalar = mean(img);
    return scalar.val[0];
}

//根据图片亮度调整图片，避免过暗或过亮
Mat ContrastAndBright(Mat srcImage , float img_light)
{
    Mat dstImage;
    int g_nContrastValue = 128;
    int g_nBrightValue = 100;
    float alpha=g_nContrastValue*0.01;          //调整因子alpha，beta
    float beta;
    if(img_light-g_nBrightValue>0)
        beta = (g_nBrightValue-img_light);
    if(img_light-g_nBrightValue<=0)
        beta = (g_nBrightValue-img_light);
    srcImage.convertTo(dstImage, -1, alpha,beta);
    return dstImage;
}

//提取黑色像素
Mat get_black(Mat hsv_img)
{
    Mat img_h,img_s,img_v;
    vector<Mat> hsv_vec;
    cv::split(hsv_img, hsv_vec);
    img_h = hsv_vec[0];
    img_s = hsv_vec[1];
    img_v = hsv_vec[2];

    img_h.convertTo(img_h,CV_32F);
    img_s.convertTo(img_s,CV_32F);
    img_v.convertTo(img_v,CV_32F);
    //计算每个通道的最大值
    double max_h,max_s,max_v;
    minMaxIdx(img_h,0,&max_h);
    minMaxIdx(img_s,0,&max_s);
    minMaxIdx(img_v,0,&max_v);
    //各通道归一化
    img_h /= max_h;
    img_s /= max_s;
    img_v /= max_v;

    Mat blackTempMat=(img_v<0.2);
    //imshow("black",blackTempMat);
    return blackTempMat;
}
//绿色阈值提取
Mat get_green(Mat hsv_img)
{
    Mat greenTempMat;
    inRange(hsv_img,Scalar(46,100,100),Scalar(95,255,255),greenTempMat);
    cv::GaussianBlur(greenTempMat,greenTempMat,cv::Size(9,9),2,2);                  //高斯滤波
    //imshow("greenTempMat",greenTempMat);
    return greenTempMat;
}
//红色阈值提取
Mat get_red(Mat hsv_img)
{
    Mat lowTempMat;
    Mat upperTempMat;
    Mat redTempMat;
    inRange(hsv_img,Scalar(0,100,100),Scalar(10,255,255),lowTempMat);
    inRange(hsv_img,Scalar(160,120,120),Scalar(180,255,255),upperTempMat);
    addWeighted(lowTempMat,1.0,upperTempMat,1.0,0.0,redTempMat);                    //图像叠加
    GaussianBlur(redTempMat,redTempMat,cv::Size(9,9),2,2);                          //高斯滤波
    //imshow("redTempMat",redTempMat);
    return redTempMat;
}

//形态学处理
Mat morphological_progress(Mat black_mask)
{
    Mat element= getStructuringElement(MORPH_RECT,Size(1*1+1,1*1+1),Point(1,1));    //element
    Mat element1= getStructuringElement(MORPH_RECT,Size(2*2+1,2*2+1),Point(2,2));   //element1
    Mat dilate_black_mask,erode_black_mask,black_mask2;

    dilate(black_mask,dilate_black_mask,element1);                                  //腐蚀
    //imshow("dilate_img",dilate_black_mask);
    //erode(black_mask,erode_black_mask,element);                                   //膨胀
    //imshow("erode_img",erode_black_mask);
    return dilate_black_mask;
}

//填洞
void fillHole(const Mat srcBw, Mat &dstBw)                                          //原始图像srcBw，结果图像dstBw
{
    Size m_Size = srcBw.size();
    Mat Temp=Mat::zeros(m_Size.height+2,m_Size.width+2,srcBw.type());               //延展图像
    srcBw.copyTo(Temp(Range(1, m_Size.height + 1), Range(1, m_Size.width + 1)));
    cv::floodFill(Temp, Point(0, 0), Scalar(255));                                  //填充区域

    Mat cutImg;                                                                     //裁剪延展的图像
    Temp(Range(1, m_Size.height + 1), Range(1, m_Size.width + 1)).copyTo(cutImg);
    dstBw = srcBw | (~cutImg);
}

//寻找外轮廓
int find_contours(Mat black_img)
{
    cv::findContours(black_img,contours,CV_RETR_TREE,CV_CHAIN_APPROX_SIMPLE);
    cv::Mat result(black_img.size(),CV_8U,Scalar(255));
    drawContours(result,contours,-1,Scalar(100),2);
    //imshow("result",result);
    //cv::waitKey(0);
    //移除过长或过短轮廓
    unsigned int cmin=20;
    unsigned int cmax=1000;
    auto itc=contours.begin();
    while(itc!=contours.end())
    {
        if(itc->size()<cmin||itc->size()>cmax)
            itc=contours.erase(itc);
        else
            ++itc;
    }
    return contours.size();
}

//在图像上显示轮廓信息
void draw_contours(Mat black_img,vector<vector<Point>> contours)
{
    //显示结果
    vector<vector<Point>> conPoint(contours.size());
    for(unsigned int i=0;i<contours.size();i++)
    {
        approxPolyDP(Mat(contours[i]),conPoint[i],2,true);
        boundRect.push_back(boundingRect(Mat(conPoint[i])));
    }
    RNG rng(12345);
    Mat resultMat = Mat::zeros(black_img.size(),CV_8UC3);
    for(unsigned int i=0;i<contours.size();i++)
    {
        Scalar color=Scalar(rng.uniform(0,255),rng.uniform(0,255),rng.uniform(0,255));
        drawContours(resultMat,conPoint,i,color,1,8,vector<Vec4i>(),0,Point());
        rectangle(resultMat,boundRect[i].tl(),boundRect[i].br(),color,2,8,0);
    }
    //imshow("resultMat",resultMat);
}

//霍夫变换检测圆形，红色圆
void circle_red(Mat img,Mat th_img,int a,int b)
{
    vector<Vec3f> circles;
    HoughCircles(th_img,circles,CV_HOUGH_GRADIENT,1,float(th_img.rows)/8,100,20,0,float(th_img.cols)/2);//霍夫圆检测
    //cout<<circles.size()<<endl;
    if(!circles.empty())
    {
        //在原图上画出圆
        for(size_t i = 0;i<circles.size();i++)
        {
            Point center(cvRound(circles[i][0]+a),cvRound(circles[i][1])+b);
            int radius=cvRound(circles[i][2]);
            circle(img,center,3,Scalar(0,255,0),-1,8,0);
            circle(img,center,radius,Scalar(0,0,255),1,8,0);
            cv::putText(img,"red_light",center,FONT_HERSHEY_PLAIN,1.5,Scalar(0,0,255),2,8,false);
        }
    }
}
//霍夫变换检测圆形，绿色圆
void circle_green(Mat img,Mat th_img,int a,int b)
{
    vector<Vec3f> circles;
    HoughCircles(th_img,circles,CV_HOUGH_GRADIENT,1,float(th_img.rows)/8,100,20,0,float(th_img.cols)/2);//霍夫圆检测
    //cout<<circles.size()<<endl;
    if(!circles.empty())
    {
        //在原图上画出圆
        for(size_t i = 0;i<circles.size();i++)
        {
            Point center(cvRound(circles[i][0]+a),cvRound(circles[i][1])+b);
            int radius=cvRound(circles[i][2]);
            circle(img,center,3,Scalar(0,255,0),-1,8,0);
            circle(img,center,radius,Scalar(0,0,255),1,8,0);
            cv::putText(img,"green_light",center,FONT_HERSHEY_PLAIN,1.5,Scalar(0,255,0),2,8,false);
        }
    }
}



#endif // TRAFFICLIGHT_H
