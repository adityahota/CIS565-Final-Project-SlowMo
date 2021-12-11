#include <stdio.h>

#include "opencv/core.hpp"
#include "opencv2/video.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"


int main(int argc, char **argv)
{
	if (argc != 2) {
		fprintf(stderr, "expected one file argument\n");
		return 1;
	}
	VideoCapture cptr(argv[1]);
	
	if (!cptr.isOpened())
		CV_Error(CV_StsError, "failed to open file");

	Mat frame;
	cap >> frame;

	printf("channels: %d\tdepth: %d\telemSize: %d\telemSize1: %d\n",
			frame.channels(), frame.depth(), frame.elemSize(),
			frame.elemSize1())
	
}
