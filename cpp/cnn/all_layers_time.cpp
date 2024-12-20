#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
#include <chrono>
#include <time.h> 
#include <random>
double get_current_time()
{
    auto now = std::chrono::high_resolution_clock::now();
    auto usec = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch());
    return usec.count() / 1000.0;
}
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

std::vector<Mat> mats(250);

void get_mat(int num)
{
    unsigned int fixed_seed = 42;  // 你可以选择任何固定的数字作为种子
    std::mt19937 gen(fixed_seed);  // 使用固定种子初始化生成器

    // 定义随机数分布范围
    std::uniform_real_distribution<> dist_real(0.0, 1.0); // 均匀分布的实数范围[0.0, 1.0)

    // 创建一个向量来存储Mat对象
    

    // 循环生成250个Mat对象
    for (int i = 0; i < num; ++i) {
        // 创建一个新的Mat对象，维度为（1,3,150,150）
        mats[i] = Mat(1, 3, 150, 150);

        // 用随机浮点数填充Mat对象
        for (int d = 0; d < mats[i].dim; ++d) {
            for (int c = 0; c < mats[i].channel; ++c) {
                for (int h = 0; h < mats[i].height; ++h) {
                    for (int w = 0; w < mats[i].width; ++w) {
                        int index = d * mats[i].channel * mats[i].height * mats[i].width +
                            c * mats[i].height * mats[i].width +
                            h * mats[i].width + w;
                        mats[i][index] = dist_real(gen);
                    }
                }
            }
        }
    }
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
double all_time[250] ;
double conv1_time[250];
double relu1_time[250];
double mp1_time[250];
double conv2_time[250];
double relu2_time[250];
double mp2_time[250];
double conv3_time[250];
double relu3_time[250];
double mp3_time[250];
double flatten1_time[250];
double linear1_time[250];
double relu4_time[250];
double linear2_time[250];


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

double conv2d(const Mat input, Mat& output, std::vector<float> weight, std::vector<float> bias)
{
    double start = get_current_time();
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
    double end = get_current_time();
    return (end-start);
}

double relu(const Mat& input, Mat& output)
{
    double start = get_current_time();
    int sum = (input.dim * input.channel * input.height * input.width);
    for (int i = 0; i < sum; ++i)
    {
        if (input[i] < 0)
            output[i] = 0;
        else
            output[i] = input[i];
    }
    double end = get_current_time();
    return (end - start);
}

double maxpool2d(const Mat& input, Mat& output)
{
    double start = get_current_time();
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
    
    double end = get_current_time();
    return (end - start);
}

double flatten(const Mat& input, Mat & output)
{
    double start = get_current_time();
    int sum = (input.dim * input.channel * input.height * input.width);
    for (int i = 0; i < sum; ++i)
    {
        output[i] = input[i];
    }
    double end = get_current_time();
    return (end - start);
}

double linear(const Mat& input, Mat& output, std::vector<float> weight, std::vector<float> bias)
{
    double start = get_current_time();

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
    double end = get_current_time();
    return (end - start);
}

// Sigmoid函数的定义
float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

int forward(Mat & input, int i)
{
    all_time[i] = 0;
    
    
    
    conv1_time[i] = conv2d(input, conv1_output, conv1_weight, conv1_bias);
    all_time[i] += conv1_time[i];
    relu1_time[i] += relu(conv1_output, relu1_output);
    all_time[i] += relu1_time[i];
    mp1_time[i] += maxpool2d(relu1_output, mp1_output);
    all_time[i] += mp1_time[i];

    conv2_time[i] += conv2d(mp1_output, conv2_output, conv2_weight, conv2_bias);
    all_time[i] += conv2_time[i];
    relu2_time[i] += relu(conv2_output, relu2_output);
    all_time[i] += relu2_time[i];
    mp2_time[i] += maxpool2d(relu2_output, mp2_output);
    all_time[i] += mp2_time[i];

    conv3_time[i] += conv2d(mp2_output, conv3_output, conv3_weight, conv3_bias);
    all_time[i] += conv3_time[i];
    relu3_time[i] += relu(conv3_output, relu3_output);
    all_time[i] += relu3_time[i];
    mp3_time[i] += maxpool2d(relu3_output, mp3_output);
    all_time[i] += mp3_time[i];

    flatten1_time[i] += flatten(mp3_output, flatten_output);
    all_time[i] += flatten1_time[i];
    linear1_time[i] += linear(flatten_output, linear1_output, linear1_weight, linear1_bias);
    all_time[i] += linear1_time[i];

    relu4_time[i] += relu(linear1_output, relu4_output);
    all_time[i] += relu4_time[i];
    linear2_time[i] += linear(relu4_output, linear2_output, linear2_weight, linear2_bias);
    all_time[i] += linear2_time[i];
    printf("conv1: %.3lf, relu1: %.3lf, mp1: %.3lf\nconv2: %.3lf, relu2: %.3lf, mp2: %.3lf\nconv3: %.3lf, relu3: %.3lf, mp3: %.3lf\nflatten1: %.3lf, linear1: %.3lf, relu4: %.3lf, linear2: %.3lf\n", conv1_time[i], relu1_time[i], mp1_time[i], conv2_time[i], relu2_time[i], mp2_time[i], conv3_time[i], relu3_time[i], mp3_time[i], flatten1_time[i], linear1_time[i], relu4_time[i], linear2_time[i]);
    //std::cout << linear2_output[0] << std::endl;//sigmoid之后会比较接近，所以可以先看sigmoid之前的数据
    printf("all time:%.3lf\n", all_time[i]);
    ans = sigmoid(linear2_output[0]);
    printf("predict output:%f\n\n", ans);
    return 0;
    //return time;

}

double calculateAverage(const double* array, int begin, int end) {
    double sum = 0.0;
    for (int i = begin; i < end; ++i) {
        sum += array[i];
    }
    return sum / (end-begin);
}

int main()
{
    preread();
    //pretensor(conv1_input); 
    get_mat(250);
    for(int i = 0;i<250;++i)
    {
        forward(mats[i], i);
    }
    double avgConv1Time = calculateAverage(conv1_time, 50,250);
    double avgRelu1Time = calculateAverage(relu1_time, 50,250);
    double avgMp1Time = calculateAverage(mp1_time, 50,250);
    double avgConv2Time = calculateAverage(conv2_time,50, 250);
    double avgRelu2Time = calculateAverage(relu2_time, 50,250);
    double avgMp2Time = calculateAverage(mp2_time, 50,250);
    double avgConv3Time = calculateAverage(conv3_time, 50,250);
    double avgRelu3Time = calculateAverage(relu3_time,50, 250);
    double avgMp3Time = calculateAverage(mp3_time, 50,250);
    double avgFlatten1Time = calculateAverage(flatten1_time,50, 250);
    double avgLinear1Time = calculateAverage(linear1_time, 50,250);
    double avgRelu4Time = calculateAverage(relu4_time, 50,250);
    double avgLinear2Time = calculateAverage(linear2_time, 50,250);
    double avgAllTime = calculateAverage(all_time, 50,250);

    printf("Average conv1 time: %.3lf, Average relu1 time: %.3lf, Average mp1 time: %.3lf\n", avgConv1Time, avgRelu1Time, avgMp1Time);
    printf("Average conv2 time: %.3lf, Average relu2 time: %.3lf, Average mp2 time: %.3lf\n", avgConv2Time, avgRelu2Time, avgMp2Time);
    printf("Average conv3 time: %.3lf, Average relu3 time: %.3lf, Average mp3 time: %.3lf\n", avgConv3Time, avgRelu3Time, avgMp3Time);

    printf("Average flatten1 time: %.3lf ms, ", avgFlatten1Time);
    printf("Average linear1 time: %.3lf ms, ", avgLinear1Time);
    printf("Average relu4 time: %.3lf ms, ", avgRelu4Time);
    printf("Average linear2 time: %.3lf ms\n", avgLinear2Time);
    printf("Average all time: %.3lf ms\n", avgAllTime);
    return 0;
}
