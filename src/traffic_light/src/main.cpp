#include "../include/traffic_light/trafficlight.h"
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

vector<vector<Point>> contours;
vector<Rect> boundRect;

int main(int argc, char** argv)
{
    ros::init(argc, argv, "traffic_light");
    //读取图像
    cv::Mat img1 = cv::imread("/home/wangpeng/learngit/src/traffic_light/img/27.jpg");
    if(img1.empty() )
    {
        ROS_ERROR("Read the picture failed!");
        return -1;
    }

    //imshow("original_img",img1);                    //原图像
    	/*注释
	参数1：窗口的名字
	参数2：窗口类型，CV_WINDOW_AUTOSIZE 时表明窗口大小等于图片大小。不可以被拖动改变大小。
	CV_WINDOW_NORMAL 时，表明窗口可以被随意拖动改变大小。
	*/
    cv::namedWindow("origin",CV_WINDOW_AUTOSIZE);
    cv::imshow("origin",img1);
    float img_light = light(img1);
    cv::Mat img = ContrastAndBright(img1,img_light);
    //imshow("img",img);                              //亮度调整后的img
    cv::Mat roi_img;
    Rect rect(0,0,img.cols,(img.rows-1)*3/3);
    img(rect).copyTo(roi_img);
    //imshow("roi_img",roi_img);                      //感兴趣区域，可根据每张图片不同调整

//    Mat gray_img;
//    cvtColor(roi_img,gray_img,COLOR_BGR2GRAY);      //图像灰度化

    cv::Mat hsv_img;
    cvtColor(roi_img,hsv_img,COLOR_BGR2HSV);        //BGR2HSV
    cv::Mat black_mask = get_black(hsv_img);            //黑色像素提取

    //形态学梯度处理
    cv::Mat black_mask0,black_mask1;
    black_mask0=morphological_progress(black_mask); //腐蚀、膨胀操作
    fillHole(black_mask0,black_mask1);              //填洞操作
    //imshow("fillHole_img",black_mask1);

    cv::Mat element= getStructuringElement(MORPH_RECT,Size(2*2+1,2*2+1),Point(2,2));
    erode(black_mask1,black_mask1,element);         //膨胀操作
    //imshow("black_mask1",black_mask1);

    GaussianBlur(black_mask1,black_mask1,Size(5,5),0.1,0);//高斯滤波

    //寻找轮廓
    int contours_size=find_contours(black_mask1);   //找到的轮廓数量
    //cout<<"Number of contours: "<<contours_size<<endl;
    ROS_INFO("Number of contours: %d",contours_size);
    //画出矩形
    draw_contours(black_mask1,contours);            //根据轮廓，拟合矩形框

    //寻找roi，将符合条件的矩形框，作为新的感兴趣区域进行处理
    for(unsigned int i=0;i<boundRect.size();i++)
    {
        cv::Mat roi_light;
        //rectangle(img,boundRect[i].tl(),boundRect[i].br(),Scalar(0,0,255),2,8,0); //在原图像上画出感兴趣区域
        //感兴趣区域矩形框坐标
        float a=boundRect[i].tl().x;                //左上顶点坐标tl
        float b=boundRect[i].tl().y;      
        float c=boundRect[i].br().x;                //右下顶点坐标br
        float d=boundRect[i].br().y;
        float width=c-a;                            //矩形框-宽
        float height=d-b;                           //矩形框-高
        Rect rect_light(a,b,width,height);
        hsv_img(rect_light).copyTo(roi_light);      //在每个矩形框中，搜索红绿灯
        //imshow("roi_light",roi_light);
        //cv::waitKey(0);

        //绿色像素提取
        cv::Mat greenTempMat = get_green(roi_light);
        //imshow("green",greenTempMat);
        circle_green(img,greenTempMat,a,b);         //检测绿色圆

        //红色像素提取
        cv::Mat redTempMat = get_red(roi_light);
        //imshow("red",redTempMat);
        circle_red(img,redTempMat,a,b);             //检测红色圆
    }

    //Show the image
    ROS_INFO("The result is displaced in the result_img window!");
    //imshow("result_img",img);
    cv::namedWindow("result",CV_WINDOW_AUTOSIZE);
    cv::imshow("result",img);

    cv::waitKey(0);
    ros::spin();
    return 0;
}



