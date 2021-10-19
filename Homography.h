#ifndef Homography_h
#define Homography_h

#include <iostream>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d.hpp"

using namespace cv;
using namespace std;

class Homography{
public: 
    bool showProcess = false;
    Mat getHomoGraphywithRANSAC(int iteration, vector<Point2f> ptL, vector<Point2f> ptR, vector<DMatch> matches){
        vector<int> finalInlier; 
        Mat finalHOMO;
        /*
        |_________________A_____________________||__h__| = |__B__|
        | x1  y1  1   0   0   0   -x1x1' -y1x1' || h11 |   | x1' |
        | 0   0   0   x1  y1  1   -x1y1' -y1y1' || h12 |   | y1' |
        | x2  y2  2   0   0   0   -x2x2' -y2x2' || h13 |   | x2' |
        | 0   0   0   x2  y2  2   -x2y2' -y2y2' || h21 | = | y2' |
        | x3  y3  3   0   0   0   -x3x3' -y3x3' || h22 |   | x3' |
        | 0   0   0   x3  y3  3   -x3y3' -y3y3' || h23 |   | y3' |
        | x4  y4  4   0   0   0   -x4x4' -y4x4' || h31 |   | x4' |
        | 0   0   0   x4  y4  4   -x4y4' -y4y4' || h32 |   | y4' | 

         Ah = B 
          
        */
        //matrix for finding inliers by compare distance 
        /*
        |__p'__|   |____H____||__p__|
        |  wx' |   | *  *  * ||  x  |
        |  wy' | = | *  *  * ||  y  |
        |   w  |   | *  *  * ||  1  |
        p' = p2 , p = p1 
        */
        Mat p1 = point2Mat(ptL);
        Mat p2 = point2Mat(ptR);
        double temp_best_percent = -1;
        double dist_th = 0.3; //distance threshold between p' and homo*p 
        double percent_th = 0.8; //
        
        int inlierCnt;

        Mat nowHOMO, tempHOMO;
        vector<int> foundInlier;

        for (int i=0; i<iteration; i++){
            if(showProcess) cout<<"\n["<<i<<"]";
            vector<int> randIdx = getRandIdx(matches); if(showProcess) cout<<" |get RandIdx| ( ";
            for(int ii = 0; ii<randIdx.size(); ii++){
                if(showProcess) cout<<randIdx[ii]<<" ";
            }if(showProcess) cout<<") ";
            tempHOMO = getHOMO(ptL, ptR, randIdx); if(showProcess) cout<<" |get HOMO| ";
            /*
            |__h__|
            | h11 |
            | h12 |
            | h13 |
            | h21 |
            | h22 |
            | h23 |
            | h31 |
            | h32 | 
            */
            tempHOMO.push_back(Mat(1,1,CV_64FC1).setTo(Scalar(1))); //add scale factor 
            /*
            |__h__|
            | h11 |
            | h12 |
            | h13 |
            | h21 |
            | h22 |
            | h23 |
            | h31 |
            | h32 |
            |  1  | 
            */
            tempHOMO = tempHOMO.reshape(1,3); if(showProcess) cout<<" |reshape| ";
            /*
            |_______h_______|
            | h11  h12  h13 |
            | h21  h22  h23 |
            | h31  h32   1  |
            */

            /*
            min||Ah-b||^2 
            */
            /*  
            |_______p2_______|   |___ht____|   |_______p1t_______|   |__________ht*p1t_________|   |__________________(ht*p1t)/last row_________________|
            |  wx1' wx2' ..  |   | a  b  c |   | x1 x2 ..        |   | ax1+by1+c ax2+by2+c ..  |   | (ax1+by1+c)/(gx1+hy1+1) (ax2+by2+c)/(gx2+hy2+1) .. |
            |  wy1' wy2' ..  | = | d  e  f | * | y1 y2 ..        | = | dx1+ey1+f dx2+ey2+f ..  |   |                                                    |
            |   w    w   ..  |   | g  h  i |   | 1  1  ..        |   | gx1+hy1+1 gx2+hy2+1 ..  |   | (dx1+ey1+f)/(gx1+hy1+1) (dx2+ey2+f)/(gx2+hy2+1) .. |
            */
            nowHOMO = tempHOMO*p1; 
            divide(nowHOMO.row(0), nowHOMO.row(2), nowHOMO.row(0));
            divide(nowHOMO.row(1), nowHOMO.row(2), nowHOMO.row(1));
            divide(nowHOMO.row(2), nowHOMO.row(2), nowHOMO.row(2)); if(showProcess) cout<<" |get x', y'| "; 
            if(showProcess) cout<<"(x1',y1')=("<<nowHOMO.at<double>(0,0)<<", "<<nowHOMO.at<double>(1,0)<<") "; 
            /*
              dst(p, p') = root((p'1-p1)^2+(p'2-p2)^2)
            */
            subtract(nowHOMO, p2, nowHOMO); 
            pow(nowHOMO, 2, nowHOMO); 
            Mat dist(1, p1.rows, CV_64FC1 );
            reduce(nowHOMO, dist, 0, REDUCE_SUM); //dim = 0 means that the matrix is reduced to a single row 
            sqrt(dist, dist); if(showProcess) cout<<" |get distance between p, p'| "; 
            if(showProcess) cout<<" d(p1, p1')="<<dist.at<double>(0,0)<<" ";

            inlierCnt = 0; 
            foundInlier.clear();
            
            for(int j=0; j<dist.cols; j++){
                //compare p' and Hp  
                if(dist.at<double>(0,j) < dist_th){ 
                    inlierCnt ++;
                    foundInlier.push_back(j);
                }
            }if(showProcess) cout<<" |found inliers = "<<inlierCnt<<"| ";
            
            double inlierRatio = inlierCnt/(double)dist.cols + 1e-5;
            if(inlierRatio > percent_th){
                finalInlier = foundInlier;
                if(showProcess) cout<<" |iteration stop| ";
                break;
            }
           
            if(inlierRatio > temp_best_percent){
                if(showProcess) cout<<" |update best| ";
                finalInlier = foundInlier;
                temp_best_percent = inlierRatio;
            }
        }

        finalHOMO = getHOMO(ptL, ptR, finalInlier);
        finalHOMO.push_back(Mat(1,1,CV_64FC1).setTo(Scalar(1)));
        finalHOMO = finalHOMO.reshape(1,3);
        return finalHOMO;
    }
    
    Mat point2Mat(vector<Point2f> pointV){
        int cols = pointV.size(); 
        int rows = 3; 
        Mat des; 
        for(int i=0; i<cols; i++){
            double datas[] = {pointV[i].x, pointV[i].y, 1.0}; 
            Mat oneRow(1, 3, CV_64FC1, datas);
            des.push_back(oneRow);
        }
        return des.t();
    }

    Mat getHOMO(vector<Point2f> ptL, vector<Point2f> ptR, vector<int> idxes){
        Mat tempHOMO; 
        Mat A, B;
        for(int i=0; i<int(idxes.size()); i++){
            int idx = idxes[i];
            double xa = ptL[idx].x; double ya = ptL[idx].y; 
            double xb = ptR[idx].x; double yb = ptR[idx].y;
            double dataA[] = { xa, ya, 1, 0,  0,  0, -xa*xb, -ya*xb, 
                               0,  0,  0, xa, ya, 1, -xa*yb, -ya*yb}; 
            double dataB[] = { xb, yb };
            Mat cutA = Mat(2, 8, CV_64FC1, dataA); 
            Mat cutB = Mat(2, 1, CV_64FC1, dataB);
            A.push_back(cutA); 
            B.push_back(cutB);
        }
        solve(A, B, tempHOMO, DECOMP_SVD);
        return tempHOMO;
    }

    vector<int> getRandIdx(vector<DMatch> p1){
        vector<int> randIdx;
        int cnt = 0;
        int matchSize = p1.size();
        while(true){
            if( cnt >= 4 ) break;
            int idx = rand() % (matchSize-1);
            randIdx.push_back(idx);
            cnt++;
        }
        return randIdx;
    }
};
#endif //Homography_H