#include "includes.h"

#include "includes.h"
#include "flavr.h"
#include "layers/bstem.h"
#include "layers/bblock.h"

void dumpTensor2Bin(char *fName, int size, float *t)
{
    FILE *f = fopen(fName, "wb");
    fwrite(t, size, sizeof(float), f);
    fclose(f);
    std::cout << "written to file" << std::endl;
}

int main()
{
    cudaSetDevice(0);
    cudnnHandle_t h;
    checkCUDNN(cudnnCreate(&h));

    std::string input_frames_file = "example-frames/frames_post__1x3x4x256x448.bin";

    // Read stacked input size
    Dims5 input_frames_dims5 = filename2dims5(input_frames_file);
    std::cout << "Stacked input frames dimensions: ";
    printDims5(input_frames_dims5);

    // Read stacked input frames into memory
    SizedArrayFloat input_frames = readTensor2FloatBuffer(input_frames_file);

    // Allocate space on GPU for input frames and copy to device
    size_t input_frames_data_size = dims5ToSize(input_frames_dims5) * sizeof(float);
    float *dev_input_frames;
    cudaMalloc(&dev_input_frames, input_frames_data_size);
    cudaMemcpy(dev_input_frames, input_frames.arr, input_frames_data_size, cudaMemcpyHostToDevice);

    /*
     * LAYER CREATION
     */
    // Create encoder stem
    BStem *stem = new BStem(input_frames_dims5, "tensor_bins/module.encoder.stem.0.weight__64x3x3x7x7.bin");
    std::cout << "Encoder output dimensions: ";
    printDims5(stem->getOutputDims());

    // Create

    // Create

    // Run encoder stem
    float *dev_output;
    stem->run(h, nullptr, dev_input_frames, nullptr, &dev_output, nullptr);

    // Copy stem data back to the user
    float *host_output = new float[dims5ToSize(stem->getOutputDims())];
    cudaMemcpy(host_output, dev_output, dims5ToSize(stem->getOutputDims()) * sizeof(float), cudaMemcpyDeviceToHost);

    //
    //
    //
    // Free unneeded data
    cudaFree(dev_input_frames);
    cudaFree(dev_output);

    dumpTensor2Bin("temp/main_out.bin", dims5ToSize(stem->getOutputDims()), host_output);

    std::cerr << "Exiting..." << std::endl;
    return 0;
}
