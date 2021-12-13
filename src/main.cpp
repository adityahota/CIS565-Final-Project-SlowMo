#include "includes.h"

#include "includes.h"
#include "flavr.h"
#include "bstem.h"

int main()
{
    string stacked_frames_file = "/home/aditya/Documents/Development/cis565/Final-Project/CUDA-Convolution2D/example-frames/frames_post__1x3x4x256x448.bin";

    // Read stacked input size
    Dims5 input_dims5 = filename2dims5(stacked_frames_file);

    return 0;
}
