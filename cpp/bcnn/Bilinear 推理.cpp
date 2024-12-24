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

void printMat(Mat& mat) {
    for (int c = 0; c < mat.channel; ++c) {
        for (int h = 0; h < mat.height; ++h) {
            for (int w = 0; w < mat.width; ++w) {
                int index = c * mat.height * mat.width + h * mat.width + w;
                printf("%.5lf ", mat[index]);
            }
            std::puts("");
        }
        std::puts("");
    }
    std::puts("");
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
std::vector<float> conv1_weight(32*3*5*5);
std::vector<float> conv1_bias(32);
std::vector<float> conv2_weight(32*32*5*5);
std::vector<float> conv2_bias(32);

std::vector<float> conv3_weight(64*32*5*5);
std::vector<float> conv3_bias(64);
std::vector<float> conv4_weight(64*64*5*5);
std::vector<float> conv4_bias(64);

std::vector<float> conv5_weight(128*64*3*3);
std::vector<float> conv5_bias(128);
std::vector<float> conv6_weight(128*128*3*3);
std::vector<float> conv6_bias(128);

std::vector<float> conv7_weight(128*128*3*3);
std::vector<float> conv7_bias(128);
std::vector<float> conv8_weight(128*128*3*3);
std::vector<float> conv8_bias(128);

std::vector<float> linear1_weight(16384);
std::vector<float> linear1_bias(1);


Mat conv1_input(1, 3, 150, 150);
Mat conv1_output(1, 32, 150, 150);
Mat relu1_output(1, 32, 148, 148);
Mat conv2_output(1, 32, 72, 72);
Mat relu2_output(1, 32, 72, 72);
Mat bn1_output(1,32,74,74);
Mat mp1_output(1, 32, 74, 74);

Mat conv3_output(1, 32, 148, 148);
Mat relu3_output(1, 32, 148, 148);
Mat conv4_output(1, 32, 72, 72);
Mat relu4_output(1, 32, 72, 72);
Mat bn2_output(1,32,74,74);
Mat mp2_output(1, 32, 74, 74);

Mat conv5_output(1, 32, 148, 148);
Mat relu5_output(1, 32, 148, 148);
Mat conv6_output(1, 32, 72, 72);
Mat relu6_output(1, 32, 72, 72);
Mat bn3_output(1,32,74,74);
Mat mp3_output(1, 32, 74, 74);

Mat conv7_output(1, 32, 148, 148);
Mat relu7_output(1, 32, 148, 148);
Mat conv8_output(1, 32, 72, 72);
Mat relu8_output(1, 32, 72, 72);
Mat bn4_output(1,32,74,74);
Mat mp4_output(1, 32, 74, 74);

Mat avg_output(1,128,4,4);

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
double conv2_time[250];
double relu2_time[250];
double bn1_time[250];
double mp1_time[250];

double conv3_time[250];
double relu3_time[250];
double conv4_time[250];
double relu4_time[250];
double bn2_time[250];
double mp2_time[250];

double conv5_time[250];
double relu5_time[250];
double conv6_time[250];
double relu6_time[250];
double bn3_time[250];
double mp3_time[250];

double conv7_time[250];
double relu7_time[250];
double conv8_time[250];
double relu8_time[250];
double bn4_time[250];
double mp4_time[250];

double avg1_time[250];

double calculateAverage(const double* array, int begin, int end) {
    double sum = 0.0;
    for (int i = begin; i < end; ++i) {
        sum += array[i];
    }
    return sum / (end-begin);
}

float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

int forward(Mat & input, int i)
{
    all_time[i] = 0;

}

// void preread()
// {
//     std::string conv1_weight_path = "D:\\code\\VS2022\\Dogs vs Cats" PATH_SEPARATOR "layer1.0.weight.bin";
//     std::string conv1_bias_path = "D:\\code\\VS2022\\Dogs vs Cats" PATH_SEPARATOR "layer1.0.bias.bin";
//     std::string conv2_weight_path = "D:\\code\\VS2022\\Dogs vs Cats" PATH_SEPARATOR "layer2.0.weight.bin";
//     std::string conv2_bias_path = "D:\\code\\VS2022\\Dogs vs Cats" PATH_SEPARATOR "layer2.0.bias.bin";
//     std::string conv3_weight_path = "D:\\code\\VS2022\\Dogs vs Cats" PATH_SEPARATOR "layer3.0.weight.bin";
//     std::string conv3_bias_path = "D:\\code\\VS2022\\Dogs vs Cats" PATH_SEPARATOR "layer3.0.bias.bin";
//     std::string linear1_weight_path = "D:\\code\\VS2022\\Dogs vs Cats" PATH_SEPARATOR "fc1.weight.bin";
//     std::string linear1_bias_path = "D:\\code\\VS2022\\Dogs vs Cats" PATH_SEPARATOR "fc1.bias.bin";
//     std::string linear2_weight_path = "D:\\code\\VS2022\\Dogs vs Cats" PATH_SEPARATOR "fc2.weight.bin";
//     std::string linear2_bias_path = "D:\\code\\VS2022\\Dogs vs Cats" PATH_SEPARATOR "fc2.bias.bin";
//     readBinaryFile(conv1_weight_path, conv1_weight);
//     readBinaryFile(conv1_bias_path, conv1_bias);
//     readBinaryFile(conv2_weight_path, conv2_weight);
//     readBinaryFile(conv2_bias_path, conv2_bias);
//     readBinaryFile(conv3_weight_path, conv3_weight);
//     readBinaryFile(conv3_bias_path, conv3_bias);
//     readBinaryFile(linear1_weight_path, linear1_weight);
//     readBinaryFile(linear1_bias_path, linear1_bias);
//     readBinaryFile(linear2_weight_path, linear2_weight);
//     readBinaryFile(linear2_bias_path, linear2_bias);

// }

int main()
{
    //preread();
    return 0;
}