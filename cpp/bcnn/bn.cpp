#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>
#include <time.h> 
#include <random>
#include <cstring>
#include <omp.h>

#if defined(_WIN32)
#define PATH_SEPARATOR "\\\\"
#else
#define PATH_SEPARATOR "/"
#endif

#include <vector>

struct Mat
{
public:
    std::vector<float> tensor;

    int dim;
    int channel;
    int height;
    int width;

    Mat() : dim(1), channel(3), height(150), width(150) {
        tensor.resize(dim * channel * height * width);
    }

    // 多态构造函数
    Mat(int d, int c, int h, int w) : dim(d), channel(c), height(h), width(w) {
        tensor.resize(d * c * h * w);
    }

    float& operator[](size_t index)
    {
        return tensor[index];
    }

    const float& operator[](size_t index) const
    {
        return tensor[index];
    }
};

double get_current_time()
{
    auto now = std::chrono::high_resolution_clock::now();
    auto usec = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch());
    return usec.count() / 1000.0;
}

bool readBinaryFile(const std::string& filepath, std::vector<float>& buffer)
{
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "Failed to open file: " << filepath << std::endl;
        return false;
    }

    file.seekg(0, std::ios::end);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    size_t numFloats = size / sizeof(float);
    buffer.resize(numFloats);
    if (file.read(reinterpret_cast<char*>(buffer.data()), size))
    {
        return true;
    }
    else
    {
        std::cerr << "Failed to read file: " << filepath << std::endl;
        return false;
    }
}

Mat bn1_input(1, 32, 150, 150);
Mat bn1_output(1, 32, 150, 150);

void pretensor( Mat& input)
{
    for (int i = 0; i < input.channel; ++i)
    {
        for (int j = 0; j < input.height; ++j)
        {
            for (int k = 0; k < input.width; ++k)
            {
                int index = i * input.height * input.width + j * input.width + k;
                float value = std::sin(static_cast<float>(index));
                input[index] = value;
            }
        }
    }
}

void printMat(Mat& mat)
{
    for (int d = 0; d < mat.dim; ++d)
    {
        for (int c = 0; c < mat.channel; ++c)
        {
            for (int h = 0; h < mat.height; ++h)
            {
                for (int w = 0; w < mat.width; ++w)
                {
                    int index =
                        d * mat.channel * mat.height * mat.width + c * mat.height * mat.width + h * mat.width + w;
                    printf("%.5lf ", mat[index]);
                }
                std::puts("");
            }
            std::puts("");
        }

        std::puts("");
    }
    std::puts("");
}

double bn(const Mat& input, Mat&output,std::vector<float> weight,std::vector<float> bias, std::vector<float> rm, std::vector<float> rv)
{
    double start = get_current_time();
    double eps = 1e-5;
    //#pragma omp parallel for
    //#pragma parallel for
    for(int c = 0;c<input.channel;++c)
    {
        //#pragma omp parallel for
        for(int h = 0;h<input.height;++h)
        {
            for(int w = 0;w<input.width;++w)
            {
                int index = c * input.height * input.width + h * input.width + w;
                double x_hat = (input[index]-rm[c])/(sqrt(rv[c]+eps));
                output[index] = x_hat*weight[c]+bias[c];
            }
        }
    }
    double end = get_current_time();
    return (end-start);
}

int main()
{
    std::vector<float> bn1_weight(64);
    std::vector<float> bn1_bias(64);
    std::vector<float> bn1_mm(64);
    std::vector<float> bn1_mv(64);
    std::string bn1_weight_path = ".\\src" PATH_SEPARATOR "bn1.weight.bin";
    std::string bn1_bias_path = ".\\src" PATH_SEPARATOR "bn1.bias.bin";
    readBinaryFile(bn1_weight_path, bn1_weight);
    readBinaryFile(bn1_bias_path, bn1_bias);
    std::string bn1_rm_path = ".\\src" PATH_SEPARATOR "bn1.running_mean.bin";
    std::string bn1_rv_path = ".\\src" PATH_SEPARATOR "bn1.running_var.bin";
    readBinaryFile(bn1_rm_path, bn1_mm);
    readBinaryFile(bn1_rv_path, bn1_mv);
    pretensor(bn1_input);
    for(int cc =0;cc<5;++cc)
        std::cout<< bn(bn1_input,bn1_output,bn1_weight,bn1_bias,bn1_mm, bn1_mv)<<std::endl;
    //printMat(bn1_output);
    return 0;
}