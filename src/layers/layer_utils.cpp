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

Dims3 mkDims3(int d[3])
{
    Dims3 x;
    for (int i = 0; i < 3; i++)
    {
        x.dims[i] = d[i];
    }
    return x;
}

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
