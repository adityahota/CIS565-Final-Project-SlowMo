#if 0

#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>

void test_parse(std::string const &filename)
{
    std::vector<int> dim = std::vector<int>();
    auto idx = filename.find("__");
    auto subStr = filename.substr(idx + 2);
    std::istringstream ss(subStr);
    char c;
    do
    {
        int val;
        ss >> val;
        dim.push_back(val);
    } while (ss >> c, c == 'x');
    for (auto const x : dim)
    {
        std::cout << x << ' ';
    }
    std::cout << std::endl;
}

int main()
{
    std::ifstream f("tensorNames.txt");
    std::string line;
    while (std::getline(f, line))
    {
        test_parse(line);
    }
    f.close();
    return 0;
}
#endif

#if 0

#include "includes.h"
#include "flavr.h"

std::vector<VidFrame> grab_vid_frames(std::string f_name)
{
    return std::vector<VidFrame>(); // TODO
}

int num_frames_new(int num_frames_orig, int expansion_coeff)
{
    return 0; // TODO
}

int main(int argc, char *argv[])
{
    std::string in_file = "";  // TODO: parsy parsy
    std::string out_file = ""; // TODO: parsy parsy
    // number of frames to add between every 2 source frames
    int expansion_coeff = 0;
    // TODO: get the model/interpolation factor
    // TODO: everything exists and valid yada yada

    // TODO: Load Model params
    // TODO: GPU side model
    // TODO: memcpys???
    auto cuFlavr = Flavr();

    std::cout << "Done Loading Network" << std::endl;

    // TODO: load input video
    auto orig_frames = grab_vid_frames(in_file);
    // TODO: make output structure
    auto output_size = num_frames_new(orig_frames.size(), expansion_coeff);
    // auto result_frames = std::vector<VidFrame>(output_size);
    std::vector<VidFrame> result_frames;
    result_frames.reserve(output_size);

    // TODO: sliding window of 4 frames feed to model, get back result, add to output
    //  TODO: what do about first couple and last couple frames??
    for (int i = 0; i < orig_frames.size(); i++)
    {
        //! Will error as bounds are wrong
        auto input_packet_start = &(orig_frames.data()[i]);
        auto guessed_frames = cuFlavr.runModel(input_packet_start);
        result_frames.insert(result_frames.end(), guessed_frames.begin(), guessed_frames.end());
        // TODO progress bar
    }

    // TODO: Write output video to file
    // TODO: cleanup

    return 0;
}
#endif

#if 0
#include <vector>
#include <stdarg.h>
#include <iostream>

std::vector<int> variDims(int vec_size, ...)
{
    std::vector<int> v = std::vector<int>();
    va_list ap;
    va_start(ap, vec_size);
    for (int i = 0; i < vec_size; i++)
    {
        int val = va_arg(ap, int);
        v.push_back(val);
    }
    va_end(ap);
    return v;
}
void dispVec(std::vector<int> v)
{
    std::cout << "num elems is " << v.size();
    std::cout << ", ";
    for (auto i : v)
    {
        std::cout << i << ", ";
    }
    std::cout << std::endl;
}

int main(int argc, char *argv[])
{
    auto a = variDims(2, 1, 12);
    dispVec(a);
    auto b = variDims(5, 1, 3, 4, 256, 448);
    dispVec(b);
    return 0;
}
#endif

#if 0
#include <iostream>
#include "includes.h"
#include "flavr.h"
#include "layers/Conv3d.h"

int main()
{
    Dims5 conv1_dim_in = mkDims5(1, 3, 4, 256, 448);
    Dims3 conv1_pad = mkDims3(1, 3, 3);
    Dims3 conv1_str = mkDims3(1, 2, 2);
    Dims3 conv1_dil = mkDims3(1, 1, 1);
    Conv3d *conv1 = new Conv3d("/home/aditya/Downloads/module.encoder.stem.0.weight__64x3x3x7x7.bin", conv1_dim_in,
                               conv1_pad, conv1_str, conv1_dil);

    Dims5 out_dim = conv1->getOutputDim();
    for (int i : out_dim.dims)
    {
        std::cout << i << " ";
    }
    std::cout << std::endl;

    Dims5 conv2_dim_in = mkDims5(1, 64, 1, 1, 1);
    Dims3 conv2_pad = mkDims3(0, 0, 0);
    Dims3 conv2_str = mkDims3(1, 1, 1);
    Dims3 conv2_dil = mkDims3(1, 1, 1);
    Conv3dBias *conv2 = new Conv3dBias("/home/aditya/Downloads/module.encoder.layer1.0.fg.attn_layer.0.weight__64x64x1x1x1.bin",
                                       "/home/aditya/Downloads/module.encoder.layer1.0.fg.attn_layer.0.bias__64.bin",
                                       conv2_dim_in, conv2_pad, conv2_str, conv2_dil);

    out_dim = conv2->getOutputDim();
    for (int i : out_dim.dims)
    {
        std::cout << i << " ";
    }
    std::cout << std::endl;

    Dims5 pool1_dim_in = mkDims5(1, 64, 4, 256, 448);
    Dims3 pool1_win = mkDims3(GET_DIM5_D(pool1_dim_in), GET_DIM5_H(pool1_dim_in), GET_DIM5_W(pool1_dim_in));
    Dims3 pool1_str = pool1_win;
    Dims3 pool1_pad = mkDims3(0, 0, 0);
    AvgPool3d *pool1 = new AvgPool3d(pool1_dim_in, pool1_win, pool1_pad, pool1_str);

    out_dim = pool1->getOutputDim();
    for (int i : out_dim.dims)
    {
        std::cout << i << " ";
    }
    std::cout << std::endl;
}

#endif

#if 0
#include "layers/layer_utils.h"
int main()
{
    std::string fname = "tensor_bins/module.outconv.1.weight__3x64x7x7.bin";
    auto dims = filename2dims(fname);
    auto kern = readTensor2FloatBuffer(fname);
    std::cout << "num dims is " << dims.dims.size() << std::endl;
    assert(dims.dims.size() == 4);
    for (int i = 0; i < dims.dims[0]; i++)
    {
        for (int j = 0; j < dims.dims[1]; j++)
        {
            for (int k = 0; k < dims.dims[2]; k++)
            {
                for (int s = 0; s < dims.dims[3]; s++)
                {
                    std::cout << (kern.arr)[s +
                                            k * dims.dims[3] +
                                            j * dims.dims[2] * dims.dims[3] +
                                            i * dims.dims[1] * dims.dims[2] * dims.dims[3]]
                              << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
    }
    free(kern.arr);
    return 0;
}
#endif

#if 1
#include <iostream>
#include "includes.h"
#include "flavr.h"
#include "layers/Conv3d.h"

int main()
{
    auto tens = readTensor2FloatBuffer("tensor_bins/module.encoder.stem.0.weight__64x3x3x7x7.bin");
    std::cout << "read in; count is " << tens.count << std::endl;
    float *dev_tens;
    cudaMalloc(&dev_tens, tens.count * sizeof(float));
    cudaMemcpy(dev_tens, tens.arr, tens.count * sizeof(float), cudaMemcpyHostToDevice);
    cudnnTensorDescriptor_t tensDesc;
    checkCUDNN(cudnnCreateTensorDescriptor(&tensDesc));
    int dimsTens[] = {64, 3, 3, 7, 7};
    int strideTens[] = {3 * 3 * 7 * 7, 3 * 7 * 7, 7 * 7, 7, 1};
    checkCUDNN(cudnnSetTensorNdDescriptor(tensDesc,
                                          CUDNN_DATA_FLOAT, 5,
                                          dimsTens, strideTens));
    auto sig = new Sigmoid();
    float foo = 0.f;
    float *fooo = &foo;
    cudnnHandle_t h;
    checkCUDNN(cudnnCreate(&h));
    sig->run(h, &tensDesc, dev_tens, nullptr, &fooo, nullptr);
    cudaMemcpy(tens.arr, fooo, tens.count * sizeof(float), cudaMemcpyDeviceToHost);

    // FILE *f = fopen("tmpTen.bin", "wb");
    // fwrite(tens.arr, sizeof(float), tens.count, f);
    FILE *f = fopen("stemSigmoidCudnn.txt", "w");
    for (int i = 0; i < tens.count; i++)
    {
        fprintf(f, "%f\n", tens.arr[i]);
    }
    std::cout << "done writing" << std::endl;
    fclose(f);
    cudnnDestroy(h);
    delete sig;
    cudnnDestroyTensorDescriptor(tensDesc);
    cudaFree(dev_tens);
    delete[] tens.arr;
    // Dims5 conv1_dim_in = mkDims5(1, 3, 4, 256, 448);
    // Dims3 conv1_pad = mkDims3(1, 3, 3);
    // Dims3 conv1_str = mkDims3(1, 2, 2);
    // Dims3 conv1_dil = mkDims3(1, 1, 1);
    // Conv3d *conv1 = new Conv3d("tensor_bins/module.encoder.stem.0.weight__64x3x3x7x7.bin", conv1_dim_in,
    //                            conv1_pad, conv1_str, conv1_dil);

    // Dims5 out_dim = conv1->getOutputDim();
    // for (int i : out_dim.dims)
    // {
    //     std::cout << i << " ";
    // }
    // std::cout << std::endl;

    // // Dims5 conv2_dim_in = mkDims5(1, 64, 1, 1, 1);
    // // Dims3 conv2_pad = mkDims3(0, 0, 0);
    // // Dims3 conv2_str = mkDims3(1, 1, 1);
    // // Dims3 conv2_dil = mkDims3(1, 1, 1);
    // // Conv3dBias *conv2 = new Conv3dBias("/home/aditya/Downloads/module.encoder.layer1.0.fg.attn_layer.0.weight__64x64x1x1x1.bin",
    // //                                    "/home/aditya/Downloads/module.encoder.layer1.0.fg.attn_layer.0.bias__64.bin",
    // //                                    conv2_dim_in, conv2_pad, conv2_str, conv2_dil);

    // // out_dim = conv2->getOutputDim();
    // // for (int i : out_dim.dims)
    // // {
    // //     std::cout << i << " ";
    // // }
    // // std::cout << std::endl;

    // // Dims5 pool1_dim_in = mkDims5(1, 64, 4, 256, 448);
    // // Dims3 pool1_win = mkDims3(GET_DIM5_D(pool1_dim_in), GET_DIM5_H(pool1_dim_in), GET_DIM5_W(pool1_dim_in));
    // // Dims3 pool1_str = pool1_win;
    // // Dims3 pool1_pad = mkDims3(0, 0, 0);
    // // AvgPool3d *pool1 = new AvgPool3d(pool1_dim_in, pool1_win, pool1_pad, pool1_str);

    // // out_dim = pool1->getOutputDim();
    // // for (int i : out_dim.dims)
    // // {
    // //     std::cout << i << " ";
    // // }
    // // std::cout << std::endl;
    return 0;
}

#endif