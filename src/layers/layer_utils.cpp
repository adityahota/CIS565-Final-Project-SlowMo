#include "layer_utils.h"

TsrDims filename2dims(std::string const &filename)
{
    TsrDims td;
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
    td.dims = dim;
    td.layout = true;
    return td;
}

Dims5 filename2dims5(std::string const &filename)
{
    Dims5 dims;
    std::vector<int> dim = std::vector<int>();
    auto idx = filename.find("__");
    auto subStr = filename.substr(idx + 2);
    std::istringstream ss(subStr);
    char c;
    int i = 0;
    do
    {
        int val;
        ss >> val;
        dims.dims[i++] = val;
    } while (ss >> c, c == 'x');

    return dims;
}

Dims4 filename2dims4(std::string const &filename)
{
    Dims4 dims;
    std::vector<int> dim = std::vector<int>();
    auto idx = filename.find("__");
    auto subStr = filename.substr(idx + 2);
    std::istringstream ss(subStr);
    char c;
    int i = 0;
    do
    {
        int val;
        ss >> val;
        dims.dims[i++] = val;
    } while (ss >> c, c == 'x');

    return dims;
}

SizedArrayFloat readTensor2FloatBuffer(std::string const &fName)
{
    // Read tensor file into the filter array
    float *buf;
    int len;
    std::ifstream is;
    is.open(fName, std::ios::binary);
    is.seekg(0, std::ios::end);
    len = is.tellg();
    is.seekg(0, std::ios::beg);
    int numFloats = len / sizeof(float);
    buf = new float[numFloats];
    is.read((char *)buf, len);
    is.close();
    SizedArrayFloat sa;
    sa.arr = buf;
    sa.count = numFloats;
    return sa;
}

Dims5 mkDims5(int d[5])
{
    Dims5 x;
    for (int i = 0; i < 5; i++)
    {
        x.dims[i] = d[i];
    }
    return x;
}

Dims5 mkDims5(int a, int b, int c, int d, int e)
{
    Dims5 x;
    x.dims[0] = a;
    x.dims[1] = b;
    x.dims[2] = c;
    x.dims[3] = d;
    x.dims[4] = e;
    return x;
}

Dims3 mkDims3(int d[3])
{
    Dims3 x;
    for (int i = 0; i < 3; i++)
    {
        x.dims[i] = d[i];
    }
    return x;
}

Dims3 mkDims3(int a, int b, int c)
{
    Dims3 x;
    x.dims[0] = a;
    x.dims[1] = b;
    x.dims[2] = c;
    return x;
}
