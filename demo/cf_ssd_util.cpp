
#include <sstream>
#include <sys/stat.h>
#include <dirent.h>

#include <iostream>
#include <fstream>
#include "cf_ssd_util.hh"
/*
*递归获取文件夹下的所有图片的文件名
*/
int get_images_path(std::string dataset, std::vector<std::string> &imgsPathList)
{
	DIR *dirHandle =  opendir(dataset.c_str());
	if (dirHandle == NULL) {
		printf("Open dir[%s] failed.\r\n", dataset.c_str());
		return -1;
	}

	struct dirent *filePtr = NULL;
	while ((filePtr = readdir(dirHandle)) != NULL) {
		if (strcmp(filePtr->d_name, ".") == 0 || strcmp(filePtr->d_name, "..") == 0) //skip current dir and parent dir
			continue;

		std::string fileName(dataset + "/" + filePtr->d_name);

		struct stat sb;
		if (stat(fileName.c_str(), &sb) == -1)
			continue;

		if (S_ISREG(sb.st_mode)) {	
			std::cout << fileName << std::endl;
			// if (fileName.rfind(".tiff") == std::string::npos)
			if (fileName.rfind(".jpg") == std::string::npos \
				&& fileName.rfind(".tiff") == std::string::npos)
				continue;
								//for regular file
			imgsPathList.push_back(fileName);
		} else if (S_ISDIR(sb.st_mode))	{						//for directory
			std::vector<std::string> subImageList;
			get_images_path(fileName, subImageList);

			for (auto subImage : subImageList)
				imgsPathList.push_back(subImage);
		} else if(S_ISLNK(sb.st_mode)) {						//for A symbolic link
		} else if(S_ISCHR(sb.st_mode)) {						//for a character device
		} else if(S_ISSOCK(sb.st_mode)) {						//for a local-domain socket
		} else if(S_ISBLK(sb.st_mode)) {						//for block device
		}
	}
	closedir(dirHandle);
	return 0;
}

/*
*从图片库中随机获取一定数量的图片，并进行解码
*/
std::vector<std::string> decode_image_opencvMap(const std::string dataset, 
									std::vector<cv::Mat> &imgsMatList, bool Isrand, unsigned int cnt)
{
    std::vector<std::string> imageList;
    get_images_path(dataset, imageList);
	std::vector<std::string> imageName;
	unsigned int toDecode = 0;

	if (!imageList.size())
		return imageName;

	if (cnt == 0) 
		toDecode = imageList.size();
	else
		toDecode = cnt;

	for (int i = 0; i < toDecode; i++) {
		unsigned randNum = i;
		if (Isrand)
			randNum = rand()%(imageList.size());
		// std::cout << "rand Num : " << randNum << std::endl;

		cv::Mat bitMap = cv::imread(imageList[randNum].c_str());
		if (bitMap.empty()) {
			// std::cout << "Decode " << imageList[randNum] << "failed." << std::endl;
			--i;
			continue;
		}
        imgsMatList.push_back(bitMap);
		imageName.push_back(imageList[randNum]);
	}
	return imageName;
}

int video_capture(const std::string &videoName)
{
	cv::VideoCapture cap(videoName);
	if (!cap.isOpened()) {
		std::cout << "Open video file failed." << std::endl;
		return -1;
	}
	cv::Mat frame;
	do {
		cap >> frame;
		cv::imshow("frame", frame);
		if (cv::waitKey(30) >= 0)
			break;
	} while (!frame.empty());
	return 0;
}

void plot_face_rect(const std::string savePath, std::string imgName, const cv::Mat & img, 
                            const facesPerImg_t &faces)
{
    std::size_t index = imgName.find_last_of('/');
    std::string name = imgName.substr(index, imgName.length());

    cv::Mat mat = img.clone();
	if (!faces.hasFace)
		return;
	
    for (auto bbox : faces.faces) {

	cv::rectangle(mat, cv::Point(bbox.x1 * img.cols, bbox.y1 * img.rows), \
					cv::Point(bbox.x2 * img.cols, bbox.y2 * img.rows), \
					cv::Scalar(255, 255, 0), 2, 2, 0);
	std::stringstream ss;
    std::string txt;
	ss.clear();
	ss << bbox.score;
	ss >> txt;
	cv::putText(mat, txt, cv::Point(bbox.x1 * img.cols, bbox.y1 * img.rows -2), cv::FONT_HERSHEY_PLAIN, 1.5, cv::Scalar(255, 255, 0), 2);

    }
    cv::imwrite(savePath + name, mat);
}