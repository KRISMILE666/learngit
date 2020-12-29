// Lane_detection
// 20191023
// Edit by LXY
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <map>
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

using namespace std;
using namespace cv;

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
    int g_nBrightValue = 130;
    float alpha=g_nContrastValue*0.01;          //调整因子alpha，beta
    float beta;
    if(img_light-g_nBrightValue>0)
        beta = (g_nBrightValue-img_light)*2;
    if(img_light-g_nBrightValue<=0)
        beta = (g_nBrightValue-img_light)*2;
    srcImage.convertTo(dstImage, -1, alpha,beta);
    return dstImage;
}
//提取白色像素
Mat get_white(Mat hsv_img)
{
    Mat whiteTempMat;
    Scalar lower_white = Scalar(0,0,190);       //低阈值
    Scalar upper_white = Scalar(180,40,255);    //高阈值
    cv::inRange(hsv_img,lower_white,upper_white,whiteTempMat);
    //imshow("white",whiteTempMat);
    return whiteTempMat;
}
//提取黄色像素
Mat get_yellow(Mat hsv_img)
{
    Mat yellowTempMat;
    Scalar lower_yellow = Scalar(20,43,100);   //低阈值
    Scalar upper_yellow = Scalar(40,255,255);  //高阈值
    cv::inRange(hsv_img,lower_yellow,upper_yellow,yellowTempMat);
    //imshow("yellow",yellowTempMat);
    return yellowTempMat;
}
//map排序，升序
int map_ascending_sort(map<int, int> iMap)
{
    vector<pair<int,int>> vtMap;
    for(auto it = iMap.begin(); it != iMap.end(); it++)
        vtMap.push_back(make_pair(it->first, it->second));
    sort(vtMap.begin(), vtMap.end(),
         [](const pair<int,int> &x, const pair<int,int> &y) -> int {
        return x.second < y.second;         //升序
    });
    auto it = vtMap.begin();
    return it->first;
}
//map排序，降序
int map_descending_sort(map<int, int> iMap)
{
    vector<pair<int,int>> vtMap;
    for(auto it = iMap.begin(); it != iMap.end(); it++)
        vtMap.push_back(make_pair(it->first, it->second));
    sort(vtMap.begin(), vtMap.end(),
         [](const pair<int,int> &x, const pair<int,int> &y) -> int {
        return x.second > y.second;         //降序
    });
    auto it = vtMap.begin();
    return it->first;
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "lane_detect");
    ros::NodeHandle nh;
    //image_transport::ImageTransport transport(nh);
    //image_transport::Subscriber image_sub; 
    //image_sub=transport.advertise("galaxy_camera/image_raw", 1);
    //读入图片,根据实际情况修改图片路径
    cv::Mat img1 = cv::imread("/home/wangpeng/learngit/src/lane_detect/img/abroad/3.jpg");
    if(img1.empty())
    {
        ROS_INFO("Couldn't find the image!");
        return -1;
    }
    ROS_INFO("The original image is displayed in \"original_img\" window!");
    cv::namedWindow("origin",CV_WINDOW_AUTOSIZE);
    cv::imshow("origin",img1);
    //imshow("original_img", img1);
    //计算图片亮度
    float img_light = light(img1);
    //根据图片亮度调整图片，避免过暗或过亮
    cv::Mat img = ContrastAndBright(img1,img_light);
    //imshow("img",img);
    //感兴趣区域提取
    int img_w = img.cols;
    int img_h = img.rows;
    int x0 = 0;
    int y0 = (img_h-1)/2;
    int delta_w = img_w-1;
    int delta_h = (img_h-1)/2;
    cv::Mat ROI_img;
    Rect rect(x0, y0, delta_w, delta_h);
    img(rect).copyTo(ROI_img);
    //imshow("ROI_img",ROI_img);
    //颜色空间转换，BGR2HSV
    cv::Mat hsv_img;
    cvtColor(ROI_img,hsv_img,CV_BGR2HSV);
    //imshow("hsv_img",hsv_img);
    //提取有效像素
    cv::Mat white_mask;
    cv::Mat yellow_mask;
    cv::Mat result_mask;
    yellow_mask = get_yellow(hsv_img);//提取黄色像素
    white_mask = get_white(hsv_img);//提取白色像素
    cv::bitwise_or(yellow_mask,white_mask,result_mask);//白色和黄色像素叠加
    //imshow("result_mask",result_mask);

    //图像灰度化,BGR2GRAY
    cv::Mat Gray_img;
    cvtColor(ROI_img, Gray_img, CV_BGR2GRAY);
    //imshow("Gray_img", Gray_img);
    cv::Mat gray_img = Gray_img/4;//使灰度图像变暗
    //imshow("gray_img", gray_img);

    //图像叠加
    cv::Mat gray_result;
    cv::bitwise_or(result_mask,gray_img,gray_result);
    //imshow("gray_result",gray_result);

    //高斯滤波,选择核大小5x5
    GaussianBlur(gray_result,gray_result,Size(5,5),0,0);
    //imshow("gray_result1",gray_result);

    //边缘检测,选择边缘检测上下临界值分别为70，150
    cv::Mat edge_img;
    Canny(gray_result,edge_img,70,150);
    //imshow("edge_img", edge_img);


    //霍夫变换检测直线
    /* 坐标分辨率 1
     * 角度分辨率 CV_PI/180
     * 阈值 100，hough变换图像空间值最大点，大于此阈值的交点，被认为是一条直线
     * 最小直线长度 60
     * 最大直线间隙 100
     */
    vector<Vec4f> lines;
    vector<Vec4f> lines_left;
    vector<Vec4f> lines_right;
    HoughLinesP(edge_img, lines, 1, CV_PI/180, 100, 60,100);
    for(size_t i = 0;i<lines.size();i++)
    {
        int x1,y1,x2,y2;
        int delta_y,delta_x;
        float k;
        Vec4i l = lines[i];
        x1=l[0];y1=l[1];x2=l[2];y2=l[3];
        delta_x = x2-x1;
        delta_y = y2-y1;
        k = float(delta_y)/delta_x;//计算直线斜率k
        //如果k>0.3&&k<5，则为右车道
        if(k>0.3&&k<5)
        {
            if(x2>img_w/2)
                lines_right.push_back(l);
            line(ROI_img,Point(x1,y1),Point(x2,y2),Scalar(0,0,255),3,CV_AA);//在图像上画出车道线,右车道为红色
            //line(img1,Point(x1+x0,y1+y0),Point(x2+x0,y2+y0),Scalar(0,0,255),3,CV_AA);
        }
        //如果k<-0.3&&k>-5，则为左车道
        if(k<-0.5&&k>-5)
        {
            if(x1<img_w/2)
                lines_left.push_back(l);
            line(ROI_img,Point(x1,y1),Point(x2,y2),Scalar(0,255,0),3,CV_AA);//左车道为绿色
            //line(img1,Point(x1+x0,y1+y0),Point(x2+x0,y2+y0),Scalar(0,255,0),3,CV_AA);
        }
    }

    //在原图上显示直线
    float delta_xr1,delta_xr2;
    float delta_xl1,delta_xl2;

    if(!lines_right.empty())
    {
        //保存右车道端点信息：横坐标，对应直线编号
        map<int,int> x_right;
        for(size_t i = 0;i<lines_right.size();i++)
        {
            x_right[i] = lines_right[i][2] ;
        }

        //保存右车道对应直线编号的xr1，yr1，xr2，yr2
        int key_point_right = map_ascending_sort(x_right);

        int xr1 = lines_right[key_point_right][0];
        int yr1 = lines_right[key_point_right][1];
        int xr2 = lines_right[key_point_right][2];
        int yr2 = lines_right[key_point_right][3];
        float kr = float(yr2-yr1)/(xr2-xr1);        //右车道斜率kr
        //cout<<"kr="<<kr<<endl;
        //line(img1,Point(xr1+x0,yr1+y0),Point(xr2+x0,yr2+y0),Scalar(0,0,255),3,CV_AA);

        //延长车道，使显示更加美观
        delta_xr1 = float(yr1-0)/kr;            //Δxr1
        delta_xr2 = float(ROI_img.rows-yr2)/kr; //Δxr2

        xr1 = xr1-delta_xr1;                    //向左延长
        yr1 = 0;
        xr2 = xr2+delta_xr2;                    //向右延长
        yr2 = ROI_img.rows;

        line(img1,Point(xr1+x0,yr1+y0),Point(xr2+x0,yr2+y0),Scalar(0,0,255),3,CV_AA);
    }


    if(!lines_left.empty())
    {
        //保存左车道端点信息：横坐标，对应直线编号
        map<int,int> x_left;
        for(size_t i = 0;i<lines_left.size();i++)
        {
            x_left[i]= lines_left[i][0] ;
        }

        //保存左车道对应直线编号的xl1，yl1，xl2，yl2
        int key_point_left = map_descending_sort(x_left);
        int xl1 = lines_left[key_point_left][0];
        int yl1 = lines_left[key_point_left][1];
        int xl2 = lines_left[key_point_left][2];
        int yl2 = lines_left[key_point_left][3];
        float kl = float(yl2-yl1)/(xl2-xl1);
        //cout<<"kl="<<kl<<endl;
        line(img1,Point(xl1+x0,yl1+y0),Point(xl2+x0,yl2+y0),Scalar(0,255,0),3,CV_AA);

        //延长车道，使显示更加美观
        delta_xl1 = float(yl1-ROI_img.rows)/kl; //Δxl1
        delta_xl2 = float(0-yl2)/kl;            //Δxl2

        xl1 = xl1-delta_xl1;             //向左延长
        yl1 = ROI_img.rows;
        xl2 = xl2+delta_xl2;             //向右延长
        yl2 = 0;

        line(img1,Point(xl1+x0,yl1+y0),Point(xl2+x0,yl2+y0),Scalar(0,255,0),3,CV_AA);
    }

    //imshow("result_roi",ROI_img);
    ROS_INFO("The result image is displayed in \"result_img\" window!");
    //imshow("result_img",img1);//在原图上显示结果
    cv::namedWindow("result",CV_WINDOW_AUTOSIZE);
    cv::imshow("result",img1);
    waitKey(0);
    
    ros::spin();
    return 0;
}
