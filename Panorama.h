#ifndef Panorama_h
#define Panorama_h

#include <iostream>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d.hpp"

#include "Homography.h"
#include "Warping.h"

using namespace cv;
using namespace std;

class Panorama{
public :
    //result Checing
    Mat resSIFT;
    Mat resMatches;
    Mat resGoodMatches;
    Mat resForWarp;
    Mat resBackWarp;
    vector<Mat> processes[2];    
    
    Mat makePanoramaFiveImgs(vector<Mat> imgs, int th, int min_match){
        Mat img1, img2, img3, img4, img5;
        img1 = imgs[0]; img2=imgs[1]; img3 =imgs[2]; img4 = imgs[3]; img5 = imgs[4];
        Mat result23 = makePanorama(img2, img3, th, min_match); 
        processes[0].push_back(resSIFT); processes[0].push_back(resMatches); processes[0].push_back(resGoodMatches); processes[0].push_back(resForWarp); processes[0].push_back(resBackWarp);
        flip(result23, result23, 1); flip(img1, img1, 1);
        Mat result123 = makePanorama(result23, img1, th, min_match); flip(result123, result123, 1);
        processes[1].push_back(resSIFT); processes[1].push_back(resMatches); processes[1].push_back(resGoodMatches); processes[1].push_back(resForWarp); processes[1].push_back(resBackWarp);
        /*
        Mat result34 = makePanorama(img3, img4, th, min_match);
        Mat result345 = makePanorama(result34, img5,  th, min_match);

        flip(result345, result345, 1); flip(result123, result123, 1);
        Mat result = makePanorama(result345, result123, th, min_match);
        flip(result, result, 1);
        */
        return result123;
    }
    

    Mat makePanorama(Mat imgL, Mat imgR, int th, int min_match){
        Mat imgLG, imgRG;
        cvtColor(imgL, imgLG, COLOR_BGR2GRAY); cvtColor(imgR, imgRG, COLOR_BGR2GRAY);

        //1. Extract SIFT keypoints
        pair<vector<KeyPoint>, Mat> siftL = getSIFTKeypoints(imgLG);
        pair<vector<KeyPoint>, Mat> siftR = getSIFTKeypoints(imgRG);
        vector<KeyPoint> kptL = siftL.first; Mat imgKptL = siftL.second;
        vector<KeyPoint> kptR = siftR.first; Mat imgKptR = siftR.second; 
        //Display 
        Mat imgKptR_copy = imgKptR.clone(); 
        resize(imgKptR_copy, imgKptR_copy, imgKptL.size()); hconcat(imgKptL, imgKptR_copy, resSIFT);

        //2. using SIFT keypoints extract Descriptor & Brute force matching 
        Mat imgLDesc = getSIFTDescriptor(imgLG, kptL); 
        Mat imgRDesc = getSIFTDescriptor(imgRG, kptR); 
        BFMatcher matcher(NORM_L2);
        vector<DMatch> matches; 
        matcher.match(imgLDesc, imgRDesc, matches);
        //Display
        drawMatches(imgLG, kptL, imgRG, kptR, matches, resMatches, Scalar::all(-1), Scalar::all(-1), vector<char>(),DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    
        //3. keep Good Matching Result 
        double distMin = matches[0].distance;
        double dist;
        for(int i=0; i<imgLDesc.rows; i++){
            dist = matches[i].distance;
            if(dist<distMin) distMin = dist;
        }
        vector<DMatch> goodMatch1;
        do{  
            vector<DMatch> goodMatch2; 
            for(int i=0; i<imgLDesc.rows; i++){
                if(matches[i].distance < th * distMin)
                    goodMatch2.push_back(matches[i]);
            }
            goodMatch1 = goodMatch2;
            th -= 1;
        }while(th != 1 && goodMatch1.size() > min_match);

        drawMatches(imgLG, kptL, imgRG, kptR, goodMatch1, resGoodMatches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        
        //4. using RANSAC and get Homography, and change perspective view
        vector<Point2f> ptL, ptR;
        for(int i=0; i<goodMatch1.size(); i++){
            ptL.push_back(kptL[goodMatch1[i].queryIdx].pt);
            ptR.push_back(kptR[goodMatch1[i].trainIdx].pt);
        }
        Homography myHomography;
        Mat matHomo = myHomography.getHomoGraphywithRANSAC(200, ptR, ptL, goodMatch1);

        Warping myWarping; 
        Mat imgForWarp = myWarping.forwardWarping(imgL, imgR, matHomo);
        resForWarp = imgForWarp;

        Mat imgBackWarp = myWarping.backwardWarping(imgL, imgR, matHomo);
        resBackWarp = imgBackWarp;

        //5. Combine with original image and converted img
        Mat imgPano;
        imgPano = imgBackWarp.clone();
        Mat roi(imgPano, Rect(0,0,imgL.cols, imgL.rows));
        imgL.copyTo(roi);

        int cutX = 0, cutY = 0;
        for(int y = 0; y<imgPano.rows; y++){
            for(int x =0; x<imgPano.cols; x++){
                if(imgPano.at<Vec3b>(y,x)[0] == 0 && imgPano.at<Vec3b>(y,x)[1] == 0 && imgPano.at<Vec3b>(y,x)[2] == 0){
                    continue;
                }
                if(cutX < x ) cutX = x;
                if(cutY < y ) cutY = y;
            }
        }
        Mat imgPanoCut;
        imgPanoCut = imgPano(Range(0, cutY), Range(0, cutX));

        return imgPanoCut;

    }

    pair<vector<KeyPoint>, Mat> getSIFTKeypoints(Mat imgGray){
        std::vector<KeyPoint> kpts; 
        Mat imgKpts; 
        Ptr<cv::SiftFeatureDetector>detector = cv::SiftFeatureDetector::create();
        detector->detect(imgGray, kpts);
        drawKeypoints(imgGray, kpts, imgKpts,Scalar::all(-1), DrawMatchesFlags::DEFAULT);
        return make_pair(kpts, imgKpts);
    }

    Mat getSIFTDescriptor(Mat imgGray, vector<KeyPoint> kpts){
        Mat imgDesc; 
        Ptr<cv::SiftDescriptorExtractor> extractor = cv::SiftDescriptorExtractor::create(100,4,3,false,true);
        extractor->compute(imgGray, kpts, imgDesc);
        return imgDesc;
    }

   
    


};
#endif //Panorama_h