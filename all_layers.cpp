#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
#include <ctime>

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

    Mat(int d, int c, int h, int w) : dim(d), channel(c), height(h), width(w)
    {
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

//weight和bias参数
std::vector<float> conv1_weight(864);
std::vector<float> conv1_bias(32);
std::vector<float> conv2_weight(9216);
std::vector<float> conv2_bias(32);
std::vector<float> conv3_weight(18432);
std::vector<float> conv3_bias(64);
std::vector<float> linear1_weight(2367488);
std::vector<float> linear1_bias(128);
std::vector<float> linear2_weight(128);
std::vector<float> linear2_bias(1);

Mat conv1_input(1, 3, 150, 150);
Mat conv1_output(1, 32, 148, 148);
Mat relu1_output(1, 32, 148, 148);
Mat mp1_output(1, 32, 74, 74);
Mat conv2_output(1, 32, 72, 72);
Mat relu2_output(1, 32, 72, 72);
Mat mp2_output(1, 32, 36, 36);
Mat conv3_output(1, 64, 34, 34);
Mat relu3_output(1, 64, 34, 34);
Mat mp3_output(1, 64, 17, 17);
Mat flatten_output(1, 1, 1, 18496);
Mat linear1_output(1,1,1,128);
Mat relu4_output(1, 1, 1, 128);
Mat linear2_output(1,1,1,1);

std::vector<int> conv_kernel_size = { 3,3 };
std::vector<int> conv_stride = { 1,1 };
int conv_kernel_max = conv_kernel_size[0] * conv_kernel_size[1];
std::vector<int> mp_kernel_size = { 2,2 };
int mp_kernel_max = mp_kernel_size[0] * mp_kernel_size[1];
std::vector<int> mp_stride = { 2,2 };
float ans = 0;
//double time = 0;

std::vector<float> layer1_output(33375000);

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

void preread()
{
    std::string conv1_weight_path = "D:\\code\\VS2022\\Dogs vs Cats" PATH_SEPARATOR "layer1.0.weight.bin";
    std::string conv1_bias_path = "D:\\code\\VS2022\\Dogs vs Cats" PATH_SEPARATOR "layer1.0.bias.bin";
    std::string conv2_weight_path = "D:\\code\\VS2022\\Dogs vs Cats" PATH_SEPARATOR "layer2.0.weight.bin";
    std::string conv2_bias_path = "D:\\code\\VS2022\\Dogs vs Cats" PATH_SEPARATOR "layer2.0.bias.bin";
    std::string conv3_weight_path = "D:\\code\\VS2022\\Dogs vs Cats" PATH_SEPARATOR "layer3.0.weight.bin";
    std::string conv3_bias_path = "D:\\code\\VS2022\\Dogs vs Cats" PATH_SEPARATOR "layer3.0.bias.bin";
    std::string linear1_weight_path = "D:\\code\\VS2022\\Dogs vs Cats" PATH_SEPARATOR "fc1.weight.bin";
    std::string linear1_bias_path = "D:\\code\\VS2022\\Dogs vs Cats" PATH_SEPARATOR "fc1.bias.bin";
    std::string linear2_weight_path = "D:\\code\\VS2022\\Dogs vs Cats" PATH_SEPARATOR "fc2.weight.bin";
    std::string linear2_bias_path = "D:\\code\\VS2022\\Dogs vs Cats" PATH_SEPARATOR "fc2.bias.bin";
    readBinaryFile(conv1_weight_path, conv1_weight);
    readBinaryFile(conv1_bias_path, conv1_bias);
    readBinaryFile(conv2_weight_path, conv2_weight);
    readBinaryFile(conv2_bias_path, conv2_bias);
    readBinaryFile(conv3_weight_path, conv3_weight);
    readBinaryFile(conv3_bias_path, conv3_bias);
    readBinaryFile(linear1_weight_path, linear1_weight);
    readBinaryFile(linear1_bias_path, linear1_bias);
    readBinaryFile(linear2_weight_path, linear2_weight);
    readBinaryFile(linear2_bias_path, linear2_bias);

    

}

int conv2d(const Mat input, Mat& output, std::vector<float> weight, std::vector<float> bias)
{
    int dim = input.dim;
    int channel = input.channel;
    int height = input.height;
    int width = input.width;
    int weight_pos = 0;
    float sum = 0;
    int cnt[100];
    memset(cnt, 0, sizeof cnt);
    for (int i = 0; i < output.channel; ++i)
    {
        
        int dx[9] = { 0, 1, 2, width, width + 1, width + 2, 2 * width, 2 * width + 1, 2 * width + 2 };
        for (int d = 0; d < dim; ++d)
        {
            for (int c = 0; c < channel; ++c)
            {
                for (int h = 0; h < height; h += conv_stride[0])
                {
                    // std::cout << weight_pos<<' ';
                    //  if ((h + stride[0]) > height)
                    //  continue;
                    if (h + conv_kernel_size[0] > height)
                        continue;
                    for (int w = 0; w < width; w += conv_stride[1])
                    {
                        // if ((w + stride[1]) > width)
                        // continue;
                        if (w + conv_kernel_size[1] > width)
                            continue;
                        int index = d * channel * height * width + c * height * width + h * width + w;
                        // std::cout << index << std::endl;
                        sum = 0;
                        for (int i = 0; i < conv_kernel_max; ++i)
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
                weight_pos += conv_kernel_max;
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
    return 0;
}

int relu(const Mat& input, Mat& output)
{
    int sum = (input.dim * input.channel * input.height * input.width);
    for (int i = 0; i < sum; ++i)
    {
        if (input[i] < 0)
            output[i] = 0;
        else
            output[i] = input[i];
    }
    return 0;
}

int maxpool2d(const Mat& input, Mat& output)
{
    int input_h = input.height;
    int input_w = input.width;
    int out_h = output.height;
    int out_w = output.width;

    int dx[4] = { 0, 1, input.width, (input.width + 1) };
    int cnt = 0;
    for (int d = 0; d < input.dim; ++d)
    {
        for (int c = 0; c < input.channel; ++c)
        {
            // 对每一个批次的每一个通道做mp2d
            for (int h = 0; h < input_h; h += mp_stride[1]) // 第一个
            {
                // std::cout << (h + stride[1]) << std::endl;
                if ((h + mp_stride[1]) > input_h)
                    continue;
                for (int w = 0; w < input_w; w += mp_stride[0])
                {
                    if ((w + mp_stride[0]) > input_w)
                        continue;
                    float max = -1000000;
                    int index = (d * input.channel * input_h * input_w) + (c * input_h * input_w) + (h * input_w) + w;
                    // std::cout << input[index] << std::endl;
                    // std::cout<<index<<std::endl;
                    for (int i = 0; i < mp_kernel_max; ++i)
                    {
                        if (input[index + dx[i]] > max)
                            max = input[index + dx[i]];
                        // std::cout << max << std::endl;
                    }
                    output[cnt++] = max;
                    // printf("%d\\n", cnt);
                }
            }
        }
    }
    
    return 0;
}

int flatten(const Mat& input, Mat & output)
{
    int sum = (input.dim * input.channel * input.height * input.width);
    for (int i = 0; i < sum; ++i)
    {
        output[i] = input[i];
    }
    return 0;
}

int linear(const Mat& input, Mat& output, std::vector<float> weight, std::vector<float> bias)
{
    // double start = easynn::get_current_time();

    for (int i = 0; i < output.width; i++)
    {
        float sum = 0;
        sum = bias[i];
        for (int j = 0; j < input.width; j++)
        {
            sum += weight[i * input.width + j] * input[j];
        }
        output[i] = sum;
    }

    //double end = easynn::get_current_time();
    // printf("%-25s,in_channels:%-4d, out_channels:%-4d, input_h:%-4d ,input_w:%-4d ,out_h:%-4d ,out_w:%-4d
    // ,time=%fms\\n",  input.c, output.c, input.h, input.w, output.h, output.w, end - start);
    return 0;
}

// Sigmoid函数的定义
float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

int forward()
{
    
    conv2d(conv1_input, conv1_output, conv1_weight, conv1_bias);
    relu(conv1_output, relu1_output);
    maxpool2d(relu1_output, mp1_output);
    
    conv2d(mp1_output, conv2_output, conv2_weight, conv2_bias);
    relu(conv2_output, relu2_output);
    maxpool2d(relu2_output, mp2_output);

    conv2d(mp2_output, conv3_output, conv3_weight, conv3_bias);
    relu(conv3_output, relu3_output);
    maxpool2d(relu3_output, mp3_output);

    flatten(mp3_output, flatten_output);
    linear(flatten_output, linear1_output, linear1_weight, linear1_bias);

    relu(linear1_output, relu4_output);
    linear(relu4_output, linear2_output, linear2_weight, linear2_bias);
    //std::cout << linear2_output[0] << std::endl;//sigmoid之后会比较接近，所以可以先看sigmoid之前的数据
    ans = sigmoid(linear2_output[0]);
    return 0;
    //return time;

}

int main()
{
    preread();
    pretensor(conv1_input); 
    forward();
    std::cout << ans;
    return 0;
}