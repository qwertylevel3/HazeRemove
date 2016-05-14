#include <stdio.h>
#include<vector>
#include<iostream>
#include<algorithm>
#include <opencv2/opencv.hpp>
#include"opencv2/ximgproc/edge_filter.hpp"


using namespace cv;

struct Node
{
    Node(double x,double y,double value)
    {
        this->x=x;
        this->y=y;
        this->value=value;
    }
    double x;
    double y;
    double value;

};
bool cmp(Node a,Node b)
{
    return a.value>b.value;
}

cv::Mat getMinChannel(cv::Mat img)
{

    cv::Mat imgGray(img.rows,img.cols,CV_8U,cv::Scalar(255));

    for(int i=0;i<img.rows;i++)
    {
        for(int j=0;j<img.cols;j++)
        {
            uchar localMin=255;
            Vec3i bgr=img.at<Vec3b>(i,j);
            for(int k=0;k<3;k++)
            {
                if(bgr.val[k]<localMin)
                {
                    localMin=bgr.val[k];
                }
            }
            imgGray.at<uchar>(i,j) =localMin;
        }
    }
    return imgGray;
}

cv::Mat getDarkChannel(cv::Mat img,int blockSize=3)
{
    if(blockSize %2==0 || blockSize<3)
    {
        std::cout<<"block size is not odd or too small"<<std::endl;
        exit(0);
    }

    int addSize=(blockSize-1)/2;

    int newHeight=img.rows+blockSize-1;
    int newWidth=img.cols+blockSize-1;

    cv::Mat imgMiddle(newHeight,newWidth,CV_8U,cv::Scalar(255));
    for(int i=addSize;i<newHeight-addSize;i++)
    {
        for(int j=addSize;j<newWidth-addSize;j++)
        {
            imgMiddle.at<uchar>(i,j)=img.at<uchar>(i-addSize,j-addSize);
        }
    }

    cv::Mat imgDark(img.rows,img.cols,CV_8U,cv::Scalar(0));


    for(int i=addSize;i<newHeight-addSize;i++)
    {
        for(int j=addSize;j<newWidth-addSize;j++)
        {
            uchar localMin=255;

            for(int k=i-addSize;k<i+addSize+1;k++)
            {
                for(int l=j-addSize;l<j+addSize+1;l++)
                {
                    if(imgMiddle.at<uchar>(k,l)<localMin)
                    {
                        localMin=imgMiddle.at<uchar>(k,l);
                    }
                }
            }
            imgDark.at<uchar>(i-addSize,j-addSize)=localMin;
        }
    }
    return imgDark;
}

int getAtomsphericLight(cv::Mat darkChannel,
                        cv::Mat img,
                        bool meanMode=false,double percent=0.001)
{
    int size=darkChannel.rows*darkChannel.cols;
    int height=darkChannel.rows;
    int width=darkChannel.cols;

    std::vector<Node> nodes;

    for(int i=0;i<height;i++)
    {
        for(int j=0;j<width;j++)
        {
            Node oneNode(i,j,darkChannel.at<uchar>(i,j));
            nodes.push_back(oneNode);
        }
    }
    std::sort(nodes.begin(),nodes.end(),cmp);

    int atomsphericLight=0;

    //todo....too small picture
    //.....

    if(meanMode)
    {
        int sum=0;
        for(int i=0;i<percent*size;i++)
        {
            for(int j=0;j<3;j++)
            {
                Vec3i bgr=img.at<Vec3b>(nodes[i].x,nodes[i].y);

                sum=sum+bgr.val[j];
            }
        }
        atomsphericLight=int (sum/(int(percent*size)*3));
        return atomsphericLight;
    }

    for(int i=0;i<percent*size;i++)
    {
        for(int j=0;j<3;j++)
        {
            Vec3i bgr=img.at<Vec3b>(nodes[i].x,nodes[i].y);

            if(bgr.val[j]>atomsphericLight)
            {
                atomsphericLight=bgr.val[j];
            }
        }
    }
    return atomsphericLight;

}

cv::Mat getRecoverScene(cv::Mat img,
                     double omega=0.95,double t0=0.1,
                     int blockSize=15,bool meanMode=false,
                     double percent=0.001)
{
    cv::Mat imgGray=getMinChannel(img);
    cv::Mat imgDark=getDarkChannel(imgGray,blockSize);
    int atomsphericLight=getAtomsphericLight(imgDark,img,meanMode,percent);


    //debug...
    imshow("imgGray",imgGray);
    imshow("imgDark",imgDark);
    std::cout<<"atomsphericLight:"<<getAtomsphericLight(imgDark,img);


    cv::Mat transmission(imgDark.rows,imgDark.cols,CV_8U,cv::Scalar(0));

    for(int i=0;i<transmission.rows;i++)
    {
        for(int j=0;j<transmission.cols;j++)
        {
            transmission.at<uchar>(i,j)=(1-omega*double(imgDark.at<uchar>(i,j))/atomsphericLight)*255;
        }
    }
    imshow("transmission1",transmission);

    cv::ximgproc::guidedFilter(imgGray,transmission,transmission,4,0.01);

    imshow("transmission2",transmission);


    cv::Mat tempTransmission(imgDark.rows,imgDark.cols,CV_64F,cv::Scalar(0));


    for(int i=0;i<transmission.rows;i++)
    {
        for(int j=0;j<transmission.cols;j++)
        {
            tempTransmission.at<double>(i,j)=double(transmission.at<uchar>(i,j))/255.0;
            if(tempTransmission.at<double>(i,j)<0.1)
            {
                tempTransmission.at<double>(i,j)=0.1;
            }
        }
    }
    imshow("temp",tempTransmission);


    cv::Mat sceneRadiance(img.rows,img.cols,CV_8UC3);

    for(int i=0;i<3;i++)
    {
        for(int j=0;j<sceneRadiance.rows;j++)
        {
            for(int k=0;k<sceneRadiance.cols;k++)
            {
                sceneRadiance.at<Vec3b>(j,k).val[i]=(double(img.at<Vec3b>(j,k).val[i]-atomsphericLight))/tempTransmission.at<double>(j,k)+atomsphericLight;

            }
        }

        for(int j=0;j<sceneRadiance.rows;j++)
        {
            for(int k=0;k<sceneRadiance.cols;k++)
            {
                if(sceneRadiance.at<Vec3b>(j,k).val[i]>255)
                {
                    sceneRadiance.at<Vec3b>(j,k).val[i]=255;
                }
                if(sceneRadiance.at<Vec3b>(j,k).val[i]<0)
                {
                    sceneRadiance.at<Vec3b>(j,k).val[i]=0;
                }
            }

        }
    }
    return sceneRadiance;
}

int main(int argc,char * argv[])
{
    if(!argv[1])
    {
        return 0;
    }

    cv::Mat img=imread(argv[1],CV_LOAD_IMAGE_COLOR);
    if(!img.data)
    {
        std::cout<<"read image error"<<std::endl;
        return 0;
    }
    imshow("ori",img);

    cv::Mat sceneRecover=getRecoverScene(img);
    imshow("after",sceneRecover);

    waitKey(0);
    cv::destroyAllWindows();
    return 0;
}

