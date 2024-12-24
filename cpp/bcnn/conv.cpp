#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>
#include <time.h>
#include <random>
#include <cstring>
#if defined(_WIN32)
#define PATH_SEPARATOR "\\\\"
#else
#define PATH_SEPARATOR "/"
#endif

struct Mat
{
public:
    std::vector<float> tensor;

    int dim;
    int channel;
    int height;
    int width;

    Mat() : dim(1), channel(3), height(150), width(150)
    {
        tensor.resize(dim * channel * height * width);
    }

    // 多态构造函数
    Mat(int d, int c, int h, int w) : dim(d), channel(c), height(h), width(w)
    {
        tensor.resize(d * c * h * w);
    }

    float &operator[](size_t index)
    {
        return tensor[index];
    }

    const float &operator[](size_t index) const
    {
        return tensor[index];
    }
};

void pretensor(Mat &input)
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

void printMat(Mat &mat)
{
    for (int d = 0; d < mat.dim; ++d)
    {
        for (int c = 0; c < mat.channel; ++c)
        {
            for (int h = 0; h < mat.height; ++h)
            {
                for (int w = 0; w < mat.width; ++w)
                {
                    int index = d * mat.channel * mat.height * mat.width + c * mat.height * mat.width + h * mat.width + w;
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

/*
void conv2d(const Mat input, Mat& output, std::vector<float> weight, std::vector<float> bias,std::vector<int> kernel_size,std::vector<int> stride,int padding)
{
    int dim = input.dim;
    int channel = input.channel;
    int height = input.height;
    int width = input.width;
    int weight_pos = 0;
    float sum = 0;
    int cnt[100];
    int kernel_max = kernel_size[0]*kernel_size[1];
    memset(cnt, 0, sizeof cnt);
    for (int i = 0; i < output.channel; ++i)
    {

        int dx[9] = { 0, 1, 2, width, width + 1, width + 2, 2 * width, 2 * width + 1, 2 * width + 2 };
        for (int d = 0; d < dim; ++d)
        {
            for (int c = 0; c < channel; ++c)
            {
                for (int h = 0; h < height; h += stride[0])
                {
                    // std::cout << weight_pos<<' ';
                    //  if ((h + stride[0]) > height)
                    //  continue;
                    if (h + kernel_size[0] > height)
                        continue;
                    for (int w = 0; w < width; w += stride[1])
                    {
                        // if ((w + stride[1]) > width)
                        // continue;
                        if (w + kernel_size[1] > width)
                            continue;
                        int index = d * channel * height * width + c * height * width + h * width + w;
                        // std::cout << index << std::endl;
                        sum = 0;
                        for (int i = 0; i < kernel_max; ++i)
                        {
                            // std::cout << i<<' '<<dx[i] << ' ' << index + dx[i] << std::endl;
                            // std::cout << weight_pos + i << ' ' << index + dx[i] << std::endl;
                            sum += (input[index + dx[i]] * weight[weight_pos + i]);
                        }
                        // puts("");
                        // std::cout<<sum<<' ';
                        output[cnt[c]++] += sum;
                        // std::cout<<output[cnt[c] - 1] << ' ';
                    }
                }
                weight_pos += kernel_max;
            }
        }
    }
    for (int i = 0; i < output.channel; ++i)
    {
        for (int j = 0; j < output.height * output.width; ++j)
        {
            output[i * output.height * output.width + j] += bias[i];
        }
    }
}

*/
void conv2d(const Mat &input, Mat &output, const std::vector<float> &weight, const std::vector<float> &bias,
            const std::vector<int> &kernel_size, const std::vector<int> &stride, int padding)
{
    int in_channel = input.channel;
    int in_height = input.height;
    int in_width = input.width;

    // 创建扩展后的输入图像
    Mat padded_input(input.dim, in_channel, in_height + 2 * padding, in_width + 2 * padding);
    std::fill(padded_input.tensor.begin(), padded_input.tensor.end(), 0); // 用0填充padding区域
    // 复制原始图像到扩展图像的中间（未填充区域）
    for (int c = 0; c < in_channel; ++c)
    {
        for (int h = 0; h < in_height; ++h)
        {
            for (int w = 0; w < in_width; ++w)
            {
                padded_input[c * (in_height + 2 * padding) * (in_width + 2 * padding) + (h + padding) * (in_width + 2 * padding) + w + padding] = input[c * in_height * in_width + h * in_width + w];
            }
        }
    }
    int kernel_max = kernel_size[0] * kernel_size[1];
    int weight_pos = 0;
    float sum = 0;
    int cnt[100000]; // 这个数组应该根据output的尺寸动态分配
    memset(cnt, 0, sizeof(cnt));
    int new_height = padded_input.height;
    int new_width = padded_input.width;
    int dx[25];
    if(kernel_max == 9)
        int dx[9] = {0, 1, 2, new_width, new_width + 1, new_width + 2, 2 * new_width, 2 * padded_input.width + 1, 2 * padded_input.width + 2};
    else if(kernel_max == 25)
        int dx[25] = {0,1,2,3,4,
                    new_width,new_width+1,new_width+2,new_width+3,new_width+4,
                    2*new_width,2*new_width+1,2*new_width+2,2*new_width+3,2*new_width+4,
                    3*new_width,3*new_width+1,3*new_width+2,3*new_width+3,3*new_width+4,
                    4*new_width,4*new_width+1,4*new_width+2,4*new_width+3,4*new_width+4,
        };
    for (int d = 0; d < output.dim; ++d)
    {
        for (int c = 0; c < output.channel; ++c)
        {
            for (int h = 0; h < output.height; h += stride[0])
            {
                if (h + kernel_size[0] > output.height)
                    continue;
                for (int w = 0; w < output.width; w += stride[1])
                {
                    if (w + kernel_size[1] > output.width)
                        continue;
                    int index = d*input.channel*new_height*new_width+c*new_height*new_width+h*new_width+w;
                    sum = 0;
                    for (int i = 0; i < kernel_max; ++i)
                    {
                        sum += (padded_input[index + dx[i]] * weight[weight_pos + i]);
                    }
                    output[cnt[c]++] += sum;
                    std::cout<<sum<<std::endl;
                }
                return;//目前是好的
            }
            weight_pos += kernel_max;
        }
    }
    for (int i = 0; i < output.channel; ++i)
    {
        for (int j = 0; j < output.height * output.width; ++j)
        {
            output[i * output.height * output.width + j] += bias[i];
        }
    }
}

bool readBinaryFile(const std::string &filepath, std::vector<float> &buffer)
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
    if (file.read(reinterpret_cast<char *>(buffer.data()), size))
    {
        return true;
    }
    else
    {
        std::cerr << "Failed to read file: " << filepath << std::endl;
        return false;
    }
}

int main()
{
    std::vector<float> conv1_weight(32 * 3 * 5 * 5);
    std::vector<float> conv1_bias(32);
    std::string conv1_weight_path = ".\\src" PATH_SEPARATOR "conv1.weight.bin";
    std::string conv1_bias_path = ".\\src" PATH_SEPARATOR "conv1.bias.bin";
    readBinaryFile(conv1_weight_path, conv1_weight);
    readBinaryFile(conv1_bias_path, conv1_bias);
    int padding = 2;
    std::vector<int> kernel_size = {5, 5};
    std::vector<int> stride = {1, 1};
    Mat input(1, 3, 150, 150);
    pretensor(input);
    Mat output(1, 32, 150, 150);
    // printMat(input);
    for(int i = 0;i<100;++i)
    {
        std::cout<<conv1_weight[i]<<' ';
    }
    return 0;
    conv2d(input, output, conv1_weight, conv1_bias, kernel_size, stride, padding);
    // printMat(output);
    return 0;
}
