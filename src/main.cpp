#include "includes.h"

#include "includes.h"
#include "flavr.h"
#include "layers/bstem.h"
#include "layers/bblock.h"

void writeTensor2Bin(char *fName, int size, float *t)
{
    FILE *f = fopen(fName, "wb");
    fwrite(t, size, sizeof(float), f);
    fclose(f);
    std::cout << fName << " written to file" << std::endl;
}

int main(int argc, char *argv[])
{
    cudaSetDevice(0);
    cudnnHandle_t h;
    checkCUDNN(cudnnCreate(&h));

    std::string input_frames_file = argv[1];

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

    //
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

    //
    // Create layer2
    //      0: Basic Block (with downsampling)
    Dims3 layer2_block0_conv1_str = mkDims3(1, 2, 2);
    Dims3 layer2_block0_conv2_str = mkDims3(1, 1, 1);
    Dims3 layer2_downsample_str = mkDims3(1, 2, 2);
    BBlock *layer2_block0 = new BBlock(layer1_block1->getOutputDims(),
                                       "tensor_bins/module.encoder.layer2.0.conv1.0.weight__128x64x3x3x3.bin", layer2_block0_conv1_str,
                                       "tensor_bins/module.encoder.layer2.0.conv2.0.weight__128x128x3x3x3.bin", layer2_block0_conv2_str,
                                       "tensor_bins/module.encoder.layer2.0.fg.attn_layer.0.weight__128x128x1x1x1.bin",
                                       "tensor_bins/module.encoder.layer2.0.fg.attn_layer.0.bias__128.bin",
                                       true, "tensor_bins/module.encoder.layer2.0.downsample.0.weight__128x64x1x1x1.bin", layer2_downsample_str);
    std::cout << "Layer 2 block 0 output dimensions: ";
    printDims5(layer2_block0->getOutputDims());

    //      1: Basic Block
    Dims3 layer2_block1_conv1_str = mkDims3(1, 1, 1);
    Dims3 layer2_block1_conv2_str = mkDims3(1, 1, 1);
    BBlock *layer2_block1 = new BBlock(layer2_block0->getOutputDims(),
                                       "tensor_bins/module.encoder.layer2.1.conv1.0.weight__128x128x3x3x3.bin", layer2_block1_conv1_str,
                                       "tensor_bins/module.encoder.layer2.1.conv2.0.weight__128x128x3x3x3.bin", layer2_block1_conv2_str,
                                       "tensor_bins/module.encoder.layer2.1.fg.attn_layer.0.weight__128x128x1x1x1.bin",
                                       "tensor_bins/module.encoder.layer2.1.fg.attn_layer.0.bias__128.bin",
                                       false, "", mkDims3(0, 0, 0));
    std::cout << "Layer 2 block 1 output dimensions: ";
    printDims5(layer2_block1->getOutputDims());

    //
    // Create layer3
    //      0: Basic Block (with downsampling)
    Dims3 layer3_block0_conv1_str = mkDims3(1, 2, 2);
    Dims3 layer3_block0_conv2_str = mkDims3(1, 1, 1);
    Dims3 layer3_downsample_str = mkDims3(1, 2, 2);
    BBlock *layer3_block0 = new BBlock(layer2_block1->getOutputDims(),
                                       "tensor_bins/module.encoder.layer3.0.conv1.0.weight__256x128x3x3x3.bin", layer3_block0_conv1_str,
                                       "tensor_bins/module.encoder.layer3.0.conv2.0.weight__256x256x3x3x3.bin", layer3_block0_conv2_str,
                                       "tensor_bins/module.encoder.layer3.0.fg.attn_layer.0.weight__256x256x1x1x1.bin",
                                       "tensor_bins/module.encoder.layer3.0.fg.attn_layer.0.bias__256.bin",
                                       true, "tensor_bins/module.encoder.layer3.0.downsample.0.weight__256x128x1x1x1.bin", layer3_downsample_str);
    std::cout << "Layer 3 block 0 output dimensions: ";
    printDims5(layer3_block0->getOutputDims());

    //      1: Basic Block
    Dims3 layer3_block1_conv1_str = mkDims3(1, 1, 1);
    Dims3 layer3_block1_conv2_str = mkDims3(1, 1, 1);
    BBlock *layer3_block1 = new BBlock(layer3_block0->getOutputDims(),
                                       "tensor_bins/module.encoder.layer3.1.conv1.0.weight__256x256x3x3x3.bin", layer3_block1_conv1_str,
                                       "tensor_bins/module.encoder.layer3.1.conv2.0.weight__256x256x3x3x3.bin", layer3_block1_conv2_str,
                                       "tensor_bins/module.encoder.layer3.1.fg.attn_layer.0.weight__256x256x1x1x1.bin",
                                       "tensor_bins/module.encoder.layer3.1.fg.attn_layer.0.bias__256.bin",
                                       false, "", mkDims3(0, 0, 0));
    std::cout << "Layer 3 block 1 output dimensions: ";
    printDims5(layer3_block1->getOutputDims());

    //
    // Create layer4
    //      0: Basic Block (with downsampling)
    Dims3 layer4_block0_conv1_str = mkDims3(1, 1, 1);
    Dims3 layer4_block0_conv2_str = mkDims3(1, 1, 1);
    Dims3 layer4_downsample_str = mkDims3(1, 1, 1);
    BBlock *layer4_block0 = new BBlock(layer3_block1->getOutputDims(),
                                       "tensor_bins/module.encoder.layer4.0.conv1.0.weight__512x256x3x3x3.bin", layer4_block0_conv1_str,
                                       "tensor_bins/module.encoder.layer4.0.conv2.0.weight__512x512x3x3x3.bin", layer4_block0_conv2_str,
                                       "tensor_bins/module.encoder.layer4.0.fg.attn_layer.0.weight__512x512x1x1x1.bin",
                                       "tensor_bins/module.encoder.layer4.0.fg.attn_layer.0.bias__512.bin",
                                       true, "tensor_bins/module.encoder.layer4.0.downsample.0.weight__512x256x1x1x1.bin", layer4_downsample_str);
    std::cout << "Layer 4 block 0 output dimensions: ";
    printDims5(layer4_block0->getOutputDims());

    //      1: Basic Block
    Dims3 layer4_block1_conv1_str = mkDims3(1, 1, 1);
    Dims3 layer4_block1_conv2_str = mkDims3(1, 1, 1);
    BBlock *layer4_block1 = new BBlock(layer4_block0->getOutputDims(),
                                       "tensor_bins/module.encoder.layer4.1.conv1.0.weight__512x512x3x3x3.bin", layer4_block1_conv1_str,
                                       "tensor_bins/module.encoder.layer4.1.conv2.0.weight__512x512x3x3x3.bin", layer4_block1_conv2_str,
                                       "tensor_bins/module.encoder.layer4.1.fg.attn_layer.0.weight__512x512x1x1x1.bin",
                                       "tensor_bins/module.encoder.layer4.1.fg.attn_layer.0.bias__512.bin",
                                       false, "", mkDims3(0, 0, 0));
    std::cout << "Layer 4 block 1 output dimensions: ";
    printDims5(layer4_block1->getOutputDims());

    /*
     * LAYER RUNNING
     */
    //
    // Run encoder stem
    //
    float *dev_output_stem;
    stem->run(h, nullptr, dev_input_frames, nullptr, &dev_output_stem, nullptr);

    //
    // Run layer 1
    //      0: Basic Block
    float *dev_output_layer1_block0;
    layer1_block0->run(h, nullptr, dev_output_stem, nullptr, &dev_output_layer1_block0, nullptr);

    //      1: Basic Block
    float *dev_output_layer1_block1;
    layer1_block1->run(h, nullptr, dev_output_layer1_block0, nullptr, &dev_output_layer1_block1, nullptr);
    checkCUDAError("cuda main block1L1");

    //
    // Run layer 2
    //      0: Basic Block
    float *dev_output_layer2_block0;
    layer2_block0->run(h, nullptr, dev_output_layer1_block1, nullptr, &dev_output_layer2_block0, nullptr);

    //      1: Basic Block
    float *dev_output_layer2_block1;
    layer2_block1->run(h, nullptr, dev_output_layer2_block0, nullptr, &dev_output_layer2_block1, nullptr);

    //
    // Run layer 3
    //      0: Basic Block
    float *dev_output_layer3_block0;
    layer3_block0->run(h, nullptr, dev_output_layer2_block1, nullptr, &dev_output_layer3_block0, nullptr);

    //      1: Basic Block
    float *dev_output_layer3_block1;
    layer3_block1->run(h, nullptr, dev_output_layer3_block0, nullptr, &dev_output_layer3_block1, nullptr);

    //
    // Run layer 4
    //      0: Basic Block
    float *dev_output_layer4_block0;
    layer4_block0->run(h, nullptr, dev_output_layer3_block1, nullptr, &dev_output_layer4_block0, nullptr);

    //      1: Basic Block
    float *dev_output_layer4_block1;
    layer4_block1->run(h, nullptr, dev_output_layer4_block0, nullptr, &dev_output_layer4_block1, nullptr);

    /*
     * DATA TRANSFER TO HOST
     */
    // Copy stem data back to the user
    int output_elements_stem = dims5ToSize(stem->getOutputDims());
    float *host_stem = new float[output_elements_stem];
    cudaMemcpy(host_stem, dev_output_stem, output_elements_stem * sizeof(float), cudaMemcpyDeviceToHost);

    // Copy layer 1 data back to the user
    int output_elements_x1 = dims5ToSize(layer1_block1->getOutputDims());
    float *host_x1 = new float[output_elements_x1];
    cudaMemcpy(host_x1, dev_output_layer1_block1, output_elements_x1 * sizeof(float), cudaMemcpyDeviceToHost);

    // Copy layer 2 data back to the user
    int output_elements_x2 = dims5ToSize(layer2_block1->getOutputDims());
    float *host_x2 = new float[output_elements_x2];
    cudaMemcpy(host_x2, dev_output_layer2_block1, output_elements_x2 * sizeof(float), cudaMemcpyDeviceToHost);

    // Copy layer 3 data back to the user
    int output_elements_x3 = dims5ToSize(layer3_block1->getOutputDims());
    float *host_x3 = new float[output_elements_x3];
    cudaMemcpy(host_x3, dev_output_layer3_block1, output_elements_x3 * sizeof(float), cudaMemcpyDeviceToHost);

    // Copy layer 4 data back to the user
    int output_elements_x4 = dims5ToSize(layer4_block1->getOutputDims());
    float *host_x4 = new float[output_elements_x4];
    cudaMemcpy(host_x4, dev_output_layer4_block1, output_elements_x4 * sizeof(float), cudaMemcpyDeviceToHost);

    writeTensor2Bin("temp/x0_cudnn.bin", output_elements_stem, host_stem);
    writeTensor2Bin("temp/x1_cudnn.bin", output_elements_x1, host_x1);
    writeTensor2Bin("temp/x2_cudnn.bin", output_elements_x2, host_x2);
    writeTensor2Bin("temp/x3_cudnn.bin", output_elements_x3, host_x3);
    writeTensor2Bin("temp/x4_cudnn.bin", output_elements_x4, host_x4);

    // Free data
    cudaFree(dev_input_frames);
    cudaFree(dev_output_stem);
    cudaFree(dev_output_layer1_block0);
    cudaFree(dev_output_layer1_block1);
    cudaFree(dev_output_layer2_block0);
    cudaFree(dev_output_layer2_block1);
    cudaFree(dev_output_layer3_block0);
    cudaFree(dev_output_layer3_block1);
    cudaFree(dev_output_layer4_block0);
    cudaFree(dev_output_layer4_block1);
    delete[] host_stem;
    delete[] host_x1;
    delete[] host_x2;
    delete[] host_x3;
    delete[] host_x4;

    std::cerr << "Exiting..." << std::endl;
    return 0;
}
