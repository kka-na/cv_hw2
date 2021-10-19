#ifndef Warping_h
#define Warping_h

#include <iostream>
#include <cmath>
 
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d.hpp"

using namespace cv;
using namespace std;

class Warping{
public:
     Mat forwardWarping(Mat imgL, Mat imgR, Mat matHomo){
        /*
            |___img___|    |___points___|
            | p  p  p |    | x1 x2 ..   |
            | p  p  p | -> | y1 y2 ..   |
            | p  p  p |    | 1  1  ..   |
        */
        Mat ptsR = allPoint2Mat(imgR); 
        /*     |_______H_______|   |_X_|
               | h11  h12  h13 |   | X |
          HX = | h21  h22  h23 | * | Y | => "Forward Warping ! "
               | h31  h32   1  |   | 1 |
        |
        |_______p2_______|   |___ht____|   |_______p1t_______|   |__________ht*p1t_________|   |__________________(ht*p1t)/last row_________________|
        |  wx1' wx2' ..  |   | a  b  c |   | x1 x2 ..        |   | ax1+by1+c ax2+by2+c ..  |   | (ax1+by1+c)/(gx1+hy1+1) (ax2+by2+c)/(gx2+hy2+1) .. |
        |  wy1' wy2' ..  | = | d  e  f | * | y1 y2 ..        | = | dx1+ey1+f dx2+ey2+f ..  |   |                                                    |
        |   w    w   ..  |   | g  h  i |   | 1  1  ..        |   | gx1+hy1+1 gx2+hy2+1 ..  |   | (dx1+ey1+f)/(gx1+hy1+1) (dx2+ey2+f)/(gx2+hy2+1) .. |
        */
        Mat ptsAfterHOMO = matHomo * ptsR; 
        divide(ptsAfterHOMO.row(0), ptsAfterHOMO.row(2), ptsAfterHOMO.row(0));
        divide(ptsAfterHOMO.row(1), ptsAfterHOMO.row(2), ptsAfterHOMO.row(1));

        pair<int,int> xMinMax = getMinMax(ptsAfterHOMO, 0);
        pair<int,int> yMinMax = getMinMax(ptsAfterHOMO, 1);

        int w,h;
        if(xMinMax.second > imgL.cols) w = xMinMax.second + xMinMax.first;
        else w = imgL.cols + xMinMax.first;
        if(yMinMax.second > imgL.rows) h = yMinMax.second + yMinMax.first;
        else h = imgL.rows + yMinMax.first;
        Mat result = Mat::zeros(h,w, imgL.type());
        /*
        Mat roi(result, Rect(xMinMax.first, yMinMax.first, imgL.cols, imgL.rows));
        imgL.copyTo(roi);
        */
        for(int r=0; r<imgR.rows; r++){
            for(int c=0; c<imgR.cols; c++){
                int x = int(ptsAfterHOMO.at<double>(0,r*imgL.cols+c));// + xMinMax.first);
                int y = int(ptsAfterHOMO.at<double>(1,r*imgL.cols+c));// + yMinMax.first);
                if(x > 0 && y > 0){
                    for(int i=0; i<3; i++){
                        result.at<Vec3b>(y,x)[i] = imgR.at<Vec3b>(r,c)[i];
                    }
                }
            }
        }
        return result;
    }

    Mat allPoint2Mat(Mat imgR){
        Mat des; 
        for(int r=0; r<int(imgR.rows); r++){
            for(int c=0; c<int(imgR.cols); c++){
                double datas[] = {double(c), double(r), 1};
                Mat oneRow(1,3,CV_64FC1, datas);
                des.push_back(oneRow);
            }
        }
        return des.t();
    }

    pair<int, int> getMinMax(Mat matPts, int rowIdx){
         int tempMin = 99999999;
         int tempMax = -99999999;
         for(int i=0; i<matPts.cols; i++){
             if(matPts.at<double>(rowIdx,i) < tempMin)
                tempMin = int(matPts.at<double>(rowIdx,i));
         }
         if(tempMin < 0 )tempMin = 0;
         for(int i=0; i<matPts.cols; i++){
             if(matPts.at<double>(rowIdx,i) > tempMax)
                tempMax = int(matPts.at<double>(rowIdx,i));
         }
         return make_pair(tempMin, tempMax);
    }

    Mat backwardWarping(Mat imgL, Mat imgR, Mat matHomo){
        /*         |_________H________|   |_X'_|
                   | h'11  h'12  h'13 |   | X' |
        inv(H)X' = | h'21  h'22  h'23 | * | Y' | => "Backward Warping ! "
                   | h'31  h'32   1   |   | 1  |
        */
        //Same with Forwarding 
        Mat ptsR = allPoint2Mat(imgR);
        Mat ptsAfterHOMO = matHomo * ptsR; 
        divide(ptsAfterHOMO.row(0), ptsAfterHOMO.row(2), ptsAfterHOMO.row(0));
        divide(ptsAfterHOMO.row(1), ptsAfterHOMO.row(2), ptsAfterHOMO.row(1));

        pair<int,int> xMinMax = getMinMax(ptsAfterHOMO, 0);
        pair<int,int> yMinMax = getMinMax(ptsAfterHOMO, 1);

        int w,h;
        if(xMinMax.second > imgL.cols) w = xMinMax.second;// + xMinMax.first;
        else w = imgL.cols;//+ xMinMax.first;
        if(yMinMax.second > imgL.rows) h = yMinMax.second;// + yMinMax.first;
        else h = imgL.rows;// + xMinMax.first;

        int wWp = xMinMax.second-xMinMax.first;
        int hWp = yMinMax.second-yMinMax.first;

        Mat ptsWP = allPoint2MatbyIdx(xMinMax, yMinMax);
        Mat invHOMO = matHomo.inv();
        Mat ptsAfterInvHOMO = invHOMO * ptsWP;
        divide(ptsAfterInvHOMO.row(0), ptsAfterInvHOMO.row(2), ptsAfterInvHOMO.row(0));
        divide(ptsAfterInvHOMO.row(1), ptsAfterInvHOMO.row(2), ptsAfterInvHOMO.row(1));
        
        
        Mat result = Mat::zeros(h,w, imgL.type());
        /*
        Mat roi(result, Rect(xMinMax.first, yMinMax.first, imgL.cols, imgL.rows));
        imgL.copyTo(roi);
        */
        for(int r=0; r<hWp; r++){
            for(int c=0; c<wWp; c++){
                //using Bilinear Interpolation , can get
                double x = double(ptsAfterInvHOMO.at<double>(0,r*wWp+c));
                double y = double(ptsAfterInvHOMO.at<double>(1,r*wWp+c));
                if((x<imgL.cols-1) && (y<imgL.rows-1) && (x>1) && (y>1)){
                    vector<double> biRGB = doBilinearInt(imgR, x, y);
                    for(int i=0; i<3; i++){
                        //result.at<Vec3b>(r+2*(yMinMax.first),c+2*(xMinMax.first))[i] = biRGB[i];
                        result.at<Vec3b>(r+yMinMax.first,c+xMinMax.first)[i] = biRGB[i];
                    }
                }
            }
        }
        return result;
        
    }

    Mat allPoint2MatbyIdx(pair<int, int> xMinMax, pair<int, int> yMinMax){
        Mat des; 
        for(int r=yMinMax.first; r<yMinMax.second; r++){
            for(int c=xMinMax.first; c<int(xMinMax.second); c++){
                double datas[] = {double(c), double(r), 1};
                Mat oneRow(1,3,CV_64FC1, datas);
                des.push_back(oneRow);
            }
        }
        return des.t();
    }

    vector<double> doBilinearInt(Mat imgR, double x, double y){
        /* Bilinear Interpolation ( find xy with four points )
        (x1,y2)   (x2,y2)
            (x,y)
        (x1,y1)   (x2,y1)
        p(x,y) ~ {(y2-y)/(y2-y1)}*p(x,y1) + {(y-y1)/(y2-y1)}*p(x,y2)
               ~ {(y2-y)/(y2-y1)}*[{(x2-x)/(x2-x1)}*p(x1,y1) + {(x-x1)/(x2-x1)}*p(x2,y1)] 
                 + {(y-y1)/(y2-y1)}*[{(x2-x)/(x2-x1)}*p(x1,y2) + {(x-x1)/(x2-x1)}*p(x2,y2)]
               = {1/(x2-x1)(y2-y1)}*[{p(x1,y1)(x2-x)(y2-y)}+{p(x2,y1)(x-x1)(y2-y)}
                                     +{p(x1,y2)(x2-x)(y-y1)}+{p(x2,y2)(x-x1)(y-y1)}]
        */
        double x1 = floor(x); double y1 = floor(y);
        double x2 = ceil(x); double y2 = ceil(y);
        vector<double> p11, p21, p12, p22;
        for(int i=0; i<3; i++){
            p11.push_back(imgR.at<Vec3b>(y1,x1)[i]);
            p21.push_back(imgR.at<Vec3b>(y1,x2)[i]);
            p12.push_back(imgR.at<Vec3b>(y2,x1)[i]);
            p22.push_back(imgR.at<Vec3b>(y2,x2)[i]);
        }
        vector<double> p;
        /*
        p = {1/(x2-x1)(y2-y1)}*[{p(x1,y1)(x2-x)(y2-y)}+{p(x2,y1)(x-x1)(y2-y)}
                                     +{p(x1,y2)(x2-x)(y-y1)}+{p(x2,y2)(x-x1)(y-y1)}]
          = frac * [{pp11} + {pp21} + {pp12} + {pp22}]
        */
        for(int i=0; i<3; i++){
            double frac = 1 / ((x2-x1) * (y2-y1)); 
            double pp11 = p11[i] * ((x2-x) * (y2-y));
            double pp21 = p21[i] * ((x-x1) * (y2-y));
            double pp12 = p12[i] * ((x2-x) * (y-y1));
            double pp22 = p22[i] * ((x-x1) * (y-y1));
            p.push_back(frac * (pp11 + pp21 + pp12 + pp22));
        }
        return p;
    }

};

#endif //Warping_h