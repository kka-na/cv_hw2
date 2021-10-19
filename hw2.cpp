#include "Panorama.h"

vector<Mat> resizeImgs(vector<Mat>, int);
Mat fiveHconcat(Mat, Mat, Mat, Mat, Mat);
Mat sixVconcat(Mat, Mat, Mat, Mat, Mat, Mat);
Mat threeVconcat(Mat, Mat, Mat);

int main(){
	cout<<"\nComputer Vision HW2\n22212231 김가나 과제\n";
	int question_num = 0;
	cout<<"\n\n፨ 문제 번호 입력. ፨"<<endl;
	cout<<"[ Question #1]\nMake Panorama Image\n";
  
	Panorama pn;
	
    vector<Mat> labs; 
    labs.push_back(imread("../image/lab1.JPEG")); labs.push_back(imread("../image/lab2.JPEG")); labs.push_back(imread("../image/lab3.JPEG"));
    labs.push_back(imread("../image/lab4.JPEG")); labs.push_back(imread("../image/lab5.JPEG")); 
    vector<Mat> inha; 
    inha.push_back(imread("../image/inha1.JPEG")); inha.push_back(imread("../image/inha2.JPEG")); inha.push_back(imread("../image/inha3.JPEG"));
    inha.push_back(imread("../image/inha4.JPEG")); inha.push_back(imread("../image/inha5.JPEG")); 

    vector<Mat> inhu; 
    inhu.push_back(imread("../image/inhu1.JPEG")); inhu.push_back(imread("../image/inhu2.JPEG")); inhu.push_back(imread("../image/inhu3.JPEG"));
    inhu.push_back(imread("../image/inhu4.JPEG")); inhu.push_back(imread("../image/inhu5.JPEG")); 

    while(true){
		cout<<"\n፨ Quit to Enter 0 ፨"<<endl;
		cout<<"▶▷▶ ";
		cin>>question_num;
		if(question_num == 1){
            int img_num;
            cout<<"\n፨ Enter the image number. Inha : 1, Inhu : 2 ፨"<<endl;
            cout<<"▶▷▶ ";
            cin>>img_num;
            vector<Mat> imgs;
            string img_name;
            if(img_num == 1 ){
                imgs = inha;
                img_name = "Inha";
            }
            else if(img_num == 2){
                imgs = inhu;
                img_name = "Inhu";
            }

            imgs = resizeImgs(imgs, 4);
            Mat origin = fiveHconcat(imgs[0], imgs[1], imgs[2], imgs[3], imgs[4]); //original image
			Mat resPano = pn.makePanoramaFiveImgs(imgs, 2,50); //단계별로 구현한 함수 이용 
            imshow("Panorama with Five Images", resPano);                            
			imwrite("../result/result_"+img_name+".jpg", resPano);

            for(int i=0; i<2; i++){
                Mat resProcess = sixVconcat(origin, pn.processes[i][0], pn.processes[i][1],pn.processes[i][2],pn.processes[i][3], pn.processes[i][4]);
			    imwrite("../result/process"+to_string(i+1)+"_"+img_name+".jpg", resProcess);
            }
			
     
		}else if(question_num == 0){
			cout<<"End HW2... "<<endl<<endl;
			break;
		}else{
			cout<<"Enter Nubmer again"<<endl;
		}
		waitKey(0);
		destroyAllWindows();
	}

	return 0;

}

vector<Mat> resizeImgs(vector<Mat> imgs, int sizeIdx){
    for(int i=0; i<int(imgs.size()); i++){
        Size size = imgs[i].size() / sizeIdx;
        resize(imgs[i], imgs[i], size );
    }
    return imgs;
}

Mat fiveHconcat(Mat img1, Mat img2, Mat img3, Mat img4, Mat img5){
    Mat dst1, dst2, dst3, dst4;
    resize(img2, img2, img1.size()); resize(img3, img3, img1.size()); 
    resize(img4, img4, img1.size()); resize(img5, img5, img1.size());
    hconcat(img1, img2, dst1);
    hconcat(img3, img4, dst2);
    hconcat(dst1, dst2, dst3);
    hconcat(dst3, img5, dst4);
    return dst4;
}

Mat threeVconcat(Mat img1, Mat img2, Mat img3){
    Mat dst1, dst2;
    resize(img2, img2, img1.size()); resize(img3, img3, img1.size());
    vconcat(img1, img2, dst1);
    vconcat(dst1, img3, dst2);
    return dst2;
}
Mat sixVconcat(Mat img1, Mat img2, Mat img3, Mat img4, Mat img5, Mat img6){
    Mat dst1, dst2, dst3, dst4, dst5;
    resize(img2, img2, img1.size()); resize(img3, img3, img1.size()); 
    resize(img4, img4, img1.size()); resize(img5, img5, img1.size()); resize(img6, img6, img1.size());
    vconcat(img1, img2, dst1);
    vconcat(img3, img4, dst2);
    vconcat(dst1, dst2, dst3);
    vconcat(img5, img6, dst4);
    vconcat(dst3, dst4, dst5);
    return dst5;
}