#include <fstream>    
#include <opencv2/opencv.hpp>  
#include <string>  
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>
#include <unistd.h>
#include <stdio.h>
#include <iostream>
#include <sys/types.h>
#include <fcntl.h>
#include <dirent.h>
#include <vector>
#include <sstream>
using namespace cv;
using namespace std;
using namespace ml;
#define PosSamNO 265    //正样本个数    
#define NegSamNO 129   //负样本个数    
#define HardExampleNO 129  //难例个数  

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/ml.hpp"
#include "opencv2/objdetect.hpp"
#include <time.h>
using namespace cv;
using namespace cv::ml;
using namespace std;
void train_svm_hog()
{
	//HOG检测器，用来计算HOG描述子的  
	//检测窗口(48,48),块尺寸(16,16),块步长(8,8),cell尺寸(8,8),直方图bin个数16   
	cv::HOGDescriptor hog(cvSize(48,48),cvSize(16,16),cvSize(8,8),cvSize(8,8),9);
	int DescriptorDim = 0;//HOG描述子的维数，由图片大小、检测窗口大小、块大小、细胞单元中直方图bin个数决定    
	//以下是设置SVM训练模型的配置
	Ptr<SVM> svm = SVM::create();
	svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::LINEAR);
	svm->setGamma(1);
	svm->setC(0.01);
	svm->setCoef0(0);
	svm->setNu(0.5);
	svm->setP(0.1);
	svm->setTermCriteria(cvTermCriteria(CV_TERMCRIT_ITER, 100, 00001));

	std::string ImgName;
	std::string res;
	
	//所有训练样本的特征向量组成的矩阵，行数等于所有样本的个数，列数等于HOG描述子维数   
	cv::Mat sampleFeatureMat;
	//训练样本的类别向量，行数等于所有样本的个数，列数等于1；1表示有目标，-1表示无目标   
	cv::Mat sampleLabelMat;


	//依次读取正样本图片，生成HOG描述子    
	for (int num = 0; num < PosSamNO ; num++)
	{
		string res;
		stringstream ss;
		ss<<num;
		ss>>res;
		ImgName = "/home/guojiaxin/图片/samples/good/" + res+".jpg";
		std::cout << "Processing：" << ImgName << std::endl;
		
		cv::Mat image = cv::imread(ImgName);
		cv::resize(image, image, cv::Size(48, 48));
		
		//HOG描述子向量   
		std::vector<float> descriptors;
		//计算HOG描述子，检测窗口移动步长(8,8)  
		hog.compute(image, descriptors, cv::Size(8, 8));
		std::cout << "descriptor dimention：" << descriptors.size() << std::endl;
		

		//处理第一个样本时初始化特征向量矩阵和类别矩阵，因为只有知道了特征向量的维数才能初始化特征向量矩阵    
		if (0 == num)
		{
			//HOG描述子的维数   
			DescriptorDim = descriptors.size();
			//初始化所有训练样本的特征向量组成的矩阵，行数等于所有样本的个数，列数等于HOG描述子维数sampleFeatureMat    
			sampleFeatureMat = cv::Mat::zeros(PosSamNO + NegSamNO + HardExampleNO, DescriptorDim, CV_32FC1);
			//初始化训练样本的类别向量，行数等于所有样本的个数，列数等于1   
			sampleLabelMat = cv::Mat::zeros(PosSamNO + NegSamNO + HardExampleNO, 1, CV_32SC1);
		}
		std::cout << "descriptor dimention：" << descriptors.size() << std::endl;
		
		//将计算好的HOG描述子复制到样本特征矩阵sampleFeatureMat    
		for (int i = 0; i < DescriptorDim; i++)
		{
			//第num个样本的特征向量中的第i个元素   
			sampleFeatureMat.at<float>(num, i) = descriptors[i];
		}
		//正样本类别为1，有目标      
		sampleLabelMat.at<float>(num, 0) = 1;
	}

	//依次读取负样本图片，生成HOG描述子    
	for (int num = 0; num < NegSamNO ; num++)
	{
		string res;
		stringstream ss;
		ss<<num;
		ss>>res;
		ImgName = "/home/guojiaxin/图片/samples/bad/" +res+".jpg";
       
		std::cout << "Processing：" << ImgName << std::endl;
		cv::Mat src = cv::imread(ImgName);
		cv::resize(src, src, cv::Size(48, 48));
		//HOG描述子向量  
		std::vector<float> descriptors;
		//计算HOG描述子，检测窗口移动步长(8,8)   
		hog.compute(src, descriptors, cv::Size(8, 8));
		std::cout << "descriptor dimention：" << descriptors.size() << std::endl;
		//将计算好的HOG描述子复制到样本特征矩阵sampleFeatureMat    
		for (int i = 0; i < descriptors.size(); i++)
		{
			//第PosSamNO+num个样本的特征向量中的第i个元素  
			
			//std::cout << "Processing：" << ImgName << std::endl;
			
			sampleFeatureMat.at<float>(num + PosSamNO, i) = descriptors[i];
		}
		std::cout << "Processing：" << ImgName << std::endl;
		//负样本类别为-1，无目标  
		sampleLabelMat.at<float>(num + PosSamNO, 0) = -1;
	}

	//处理HardExample负样本    
	if (HardExampleNO > 0)
	{
		//HardExample负样本的文件列表   
		//std::ifstream finHardExample("hard_samples_d.txt");
		//依次读取HardExample负样本图片，生成HOG描述子    
		for (int num = 0; num < HardExampleNO ; num++)
		{
			string res;
			stringstream ss;
			ss<<num;
			ss>>res;
			ImgName = "/home/guojiaxin/图片/samples/bad/" +res+".jpg";
       
		
			std::cout << "Processing：" << ImgName << std::endl;

			cv::Mat src = cv::imread(ImgName, cv::IMREAD_GRAYSCALE);
			cv::resize(src, src, cv::Size(48, 48));
			//HOG描述子向量    
			std::vector<float> descriptors;
			//计算HOG描述子，检测窗口移动步长(8,8)   
			hog.compute(src, descriptors, cv::Size(8, 8));

			//将计算好的HOG描述子复制到样本特征矩阵sampleFeatureMat    
			for (int i = 0; i < DescriptorDim; i++)
			{
				//第PosSamNO+num个样本的特征向量中的第i个元素  
				sampleFeatureMat.at<float>(num + PosSamNO + NegSamNO, i) = descriptors[i];
			}
			//负样本类别为-1，无目标   
			sampleLabelMat.at<float>(num + PosSamNO + NegSamNO, 0) = -1;
		}
	}

	//训练SVM分类器    
	//迭代终止条件，当迭代满1000次或误差小于FLT_EPSILON时停止迭代    
	std::cout << "开始训练SVM分类器" << std::endl;
	// cv::Ptr<cv::ml::TrainData> td = cv::ml::TrainData::create(sampleFeatureMat, cv::ml::SampleTypes::ROW_SAMPLE, sampleLabelMat);
	// //训练分类器    
	Ptr<TrainData> td = TrainData::create(sampleFeatureMat, ROW_SAMPLE, sampleLabelMat);
	
	svm->train(td);
	std::cout << "训练完成" << std::endl;
	//将训练好的SVM模型保存为xml文件  
	svm->save("SVM_HOG.xml");

	return;
}



void svm_hog_detect()
{
	//HOG检测器，用来计算HOG描述子的    
	//检测窗口(48,48),块尺寸(16,16),块步长(8,8),cell尺寸(8,8),直方图bin个数9    
	cv::HOGDescriptor hog(cvSize(48,48),cvSize(16,16),cvSize(8,8),cvSize(8,8),9);

	//HOG描述子的维数，由图片大小、检测窗口大小、块大小、细胞单元中直方图bin个数决定   
	int DescriptorDim;

	cv::String detector = "SVM_HOG.xml";
	//从XML文件读取训练好的SVM模型  
	cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::load(detector);
	//std::cout << "Degree = " << svm->getDegree() << std::endl;
	if (svm->empty())
	{
		std::cout << "load svm detector failed!!!" << std::endl;
		return;
	}
	std::cout << "descriptor dimention："  << std::endl;
		
	//特征向量的维数，即HOG描述子的维数    
	DescriptorDim = svm->getVarCount();

	//获取svecsmat，元素类型为float  
	cv::Mat svecsmat = svm->getSupportVectors();
	//特征向量维数  
	int svdim = svm->getVarCount();
	int numofsv = svecsmat.rows;

	//alphamat和svindex必须初始化，否则getDecisionFunction()函数会报错  
	cv::Mat alphamat = cv::Mat::zeros(numofsv, svdim, CV_32F);
	cv::Mat svindex = cv::Mat::zeros(1, numofsv, CV_64F);

	cv::Mat Result;
	double rho = svm->getDecisionFunction(0, alphamat, svindex);
	//将alphamat元素的数据类型重新转成CV_32F  
	alphamat.convertTo(alphamat, CV_32F);
	Result = -1 * alphamat * svecsmat;

	std::vector<float> vec;
	for (int i = 0; i < svdim; ++i)
	{
		vec.push_back(Result.at<float>(0, i));
	}
	vec.push_back(rho);

	//saving HOGDetectorForOpenCV.txt  
	std::ofstream fout("HOGDetectorForOpenCV.txt");
	for (int i = 0; i < vec.size(); ++i)
	{
		fout << vec[i] << std::endl;
	}

	hog.setSVMDetector(vec);

	/**************读入视频进行HOG检测******************/
	// capture >> frame;
	for (int i = 1; i < 4; ++i){
		stringstream ImgName;
        
        //  stringstream str;
        // str << "/home/guojiaxin/视频/FRM/bak/frm" << currentFrame << ".png";        /*图片存储位置*/

        // cout << str.str( ) << endl;
     
		ImgName<<"/home/guojiaxin/桌面/视觉组选拔小任务/"<<i<<".jpg";
		cout << ImgName.str( ) << endl;
		// std::cout << "Processing：" << ImgName << std::endl;

		cv::Mat frame = cv::imread(ImgName.str( ));
		
		std::vector<cv::Rect> found, found_1, found_filtered;
		//对图片进行多尺度检测  detectMultiScale
		hog.detectMultiScale(frame, found, 0, cv::Size(8,8), cv::Size(48, 48), 1.4077, 2);

		for (int i = 0; i < found.size(); i++)
		{
			if (found[i].x > 0 && found[i].y > 0 && (found[i].x + found[i].width) < frame.cols
				&& (found[i].y + found[i].height) < frame.rows)
				found_1.push_back(found[i]);
		}

		//找出所有没有嵌套的矩形框r,并放入found_filtered中,如果有嵌套的话,  
		//则取外面最大的那个矩形框放入found_filtered中    
		for (int i = 0; i < found_1.size(); i++)
		{
			cv::Rect r = found_1[i];
			int j = 0;
			for (; j < found_1.size(); j++)
				if (j != i && (r & found_1[j]) == found_1[j])
					break;
			if (j == found_1.size())
				found_filtered.push_back(r);
		}

		//画矩形框，因为hog检测出的矩形框比实际目标框要稍微大些,所以这里需要做一些调整，可根据实际情况调整    
		for (int i = 0; i < found_filtered.size(); i++)
		{
			cv::Rect r = found_filtered[i];
			//将检测矩形框缩小后绘制，根据实际情况调整  
			r.x += cvRound(r.width*0.1);
			r.width = cvRound(r.width*0.8);
			r.y += cvRound(r.height*0.1);
			r.height = cvRound(r.height*0.8);
		}

		for (int i = 0; i < found_filtered.size(); i++)
		{
			cv::Rect r = found_filtered[i];

			cv::rectangle(frame, r.tl(), r.br(), cv::Scalar(0, 0, 255), 2);

		}
		cv::imshow("detect result", frame);

		//保存检测结果  
		stringstream str;
	    str << "/home/guojiaxin/视频/nanli/"<<i<<".jpg";        /*图片存储位置*/
		cv::imshow("detect result", frame);
	    cout << str.str( ) << endl;
	    imwrite( str.str( ), frame );}
	    return;

	// cv::VideoCapture capture("/home/guojiaxin/CLionProjects/untitled/a.mp4");

	// if (!capture.isOpened())
	// {
	// 	std::cout << "Read video Failed !" << std::endl;
	// 	return;
	// }

	// cv::Mat frame;

	// int frame_num = capture.get(cv::CAP_PROP_FRAME_COUNT);
	// std::cout << "total frame number is: " << frame_num << std::endl;


	// int width = capture.get(cv::CAP_PROP_FRAME_WIDTH);
	// int height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);

	// cv::VideoWriter out;

	// //用于保存检测结果  
	// out.open("test_result.avi", CV_FOURCC('D', 'I', 'V', 'X'), 25.0, cv::Size(720, 480), true);

	// for (int i = 0; i < 1000; ++i)
	// {
	// 	capture >> frame;

	// 	//cv::resize(frame, frame, cv::Size(width / 2, height / 2));

	// 	//目标矩形框数组   
	// 	std::vector<cv::Rect> found, found_1, found_filtered;
	// 	//对图片进行多尺度检测  detectMultiScale
	// 	hog.detectMultiScale(frame, found, 0, cv::Size(8,8), cv::Size(24, 24), 1.33, 2);

	// 	for (int i = 0; i < found.size(); i++)
	// 	{
	// 		if (found[i].x > 0 && found[i].y > 0 && (found[i].x + found[i].width) < frame.cols
	// 			&& (found[i].y + found[i].height) < frame.rows)
	// 			found_1.push_back(found[i]);
	// 	}

	// 	//找出所有没有嵌套的矩形框r,并放入found_filtered中,如果有嵌套的话,  
	// 	//则取外面最大的那个矩形框放入found_filtered中    
	// 	for (int i = 0; i < found_1.size(); i++)
	// 	{
	// 		cv::Rect r = found_1[i];
	// 		int j = 0;
	// 		for (; j < found_1.size(); j++)
	// 			if (j != i && (r & found_1[j]) == found_1[j])
	// 				break;
	// 		if (j == found_1.size())
	// 			found_filtered.push_back(r);
	// 	}

	// 	//画矩形框，因为hog检测出的矩形框比实际目标框要稍微大些,所以这里需要做一些调整，可根据实际情况调整    
	// 	for (int i = 0; i < found_filtered.size(); i++)
	// 	{
	// 		cv::Rect r = found_filtered[i];
	// 		//将检测矩形框缩小后绘制，根据实际情况调整  
	// 		r.x += cvRound(r.width*0.1);
	// 		r.width = cvRound(r.width*0.8);
	// 		r.y += cvRound(r.height*0.1);
	// 		r.height = cvRound(r.height*0.8);
	// 	}

	// 	for (int i = 0; i < found_filtered.size(); i++)
	// 	{
	// 		cv::Rect r = found_filtered[i];

	// 		cv::rectangle(frame, r.tl(), r.br(), cv::Scalar(0, 0, 255), 2);

	// 	}
	// 	cv::imshow("detect result", frame);

	// 	//保存检测结果  
	// 	out << frame;

	// 	if (cv::waitKey(30) == 'q')
	// 	{
	// 		break;
	// 	}
	// }
	// capture.release();
	// out.release();
	// return;
}


int main()
{
	train_svm_hog();
	svm_hog_detect();
	return 0;
}