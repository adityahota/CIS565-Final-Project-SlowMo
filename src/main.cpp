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

    // Create layer1
    //      0: Basic Block
    Dims3 layer1_conv1_str = mkDims3(1, 1, 1);
    Dims3 layer1_conv2_str = mkDims3(1, 1, 1);
    BBlock *layer1_block0 = new BBlock(stem->getOutputDims(),
                                       "tensor_bins/module.encoder.layer1.0.conv1.0.weight__64x64x3x3x3.bin", layer1_conv1_str,
                                       "tensor_bins/module.encoder.layer1.0.conv2.0.weight__64x64x3x3x3.bin", layer1_conv2_str,
                                       "tensor_bins/module.encoder.layer1.0.fg.attn_layer.0.weight__64x64x1x1x1.bin",
                                       "tensor_bins/module.encoder.layer1.0.fg.attn_layer.0.bias__64.bin",
                                       false, "", mkDims3(0, 0, 0));
    std::cout << "Layer 1 block 0 output dimensions: ";
    printDims5(layer1_block0->getOutputDims());

    //      1: Basic Block
    BBlock *layer1_block1 = new BBlock(layer1_block0->getOutputDims(),
                                       "tensor_bins/module.encoder.layer1.1.conv1.0.weight__64x64x3x3x3.bin", layer1_conv1_str,
                                       "tensor_bins/module.encoder.layer1.1.conv2.0.weight__64x64x3x3x3.bin", layer1_conv2_str,
                                       "tensor_bins/module.encoder.layer1.1.fg.attn_layer.0.weight__64x64x1x1x1.bin",
                                       "tensor_bins/module.encoder.layer1.1.fg.attn_layer.0.bias__64.bin",
                                       false, "", mkDims3(0, 0, 0));
    std::cout << "Layer 1 block 1 output dimensions: ";
    printDims5(layer1_block1->getOutputDims());

    // Run encoder stem
    // checkCUDAError("cuda main stem");
    float *dev_output_stem;
    stem->run(h, nullptr, dev_input_frames, nullptr, &dev_output_stem, nullptr);

    // Run layer 1
    //      0: Basic Block
    // checkCUDAError("cuda main basic block 0");
    float *dev_output_layer1_block0;
    layer1_block0->run(h, nullptr, dev_output_stem, nullptr, &dev_output_layer1_block0, nullptr);

    //      1: Basic Block
    // checkCUDAError("cuda main basic block 1");
    float *dev_output_layer1_block1;
    layer1_block1->run(h, nullptr, dev_output_layer1_block0, nullptr, &dev_output_layer1_block1, nullptr);

    // Copy layer 1 data back to the user
    int output_elements = dims5ToSize(layer1_block0->getOutputDims());
    float *host_output = new float[output_elements];
    cudaMemcpy(host_output, dev_output_layer1_block1, output_elements * sizeof(float), cudaMemcpyDeviceToHost);

    //
    //
    //
    // Free unneeded data
    cudaFree(dev_input_frames);
    cudaFree(dev_output_stem);
    cudaFree(dev_output_layer1_block0);
    cudaFree(dev_output_layer1_block1);
    delete[] host_output;

    dumpTensor2Bin("temp/main_out.bin", output_elements, host_output);

    std::cerr << "Exiting..." << std::endl;
    return 0;
}
