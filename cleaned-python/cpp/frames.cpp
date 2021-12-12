#include <stdio.h>

#include "opencv2/core.hpp"
//#include "opencv2/video.hpp"
//#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
//#include "opencv2/highgui.hpp"


int main(int argc, char **argv)
{
	if (argc != 2) {
		fprintf(stderr, "expected one file argument\n");
		return 1;
	}
	cv::VideoCapture cptr(argv[1]);
	
	if (!cptr.isOpened())
		exit(5);

	cv::Mat frame;
	cptr >> frame;

	printf("channels: %d\tdepth: %d\telemSize: %d\telemSize1: %d\n",
			frame.channels(), frame.depth(), frame.elemSize(),
			frame.elemSize1());
	
}
