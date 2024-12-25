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

std::vector<float> bn1_weight(32);
std::vector<float> bn1_bias(32);
std::vector<float> bn1_mm(32);
std::vector<float> bn1_mv(32);

std::vector<float> conv3_weight(64*32*5*5);
std::vector<float> conv3_bias(64);
std::vector<float> conv4_weight(64*64*5*5);
std::vector<float> conv4_bias(64);

std::vector<float> bn2_weight(64);
std::vector<float> bn2_bias(64);
std::vector<float> bn2_mm(64);
std::vector<float> bn2_mv(64);

std::vector<float> conv5_weight(128*64*3*3);
std::vector<float> conv5_bias(128);
std::vector<float> conv6_weight(128*128*3*3);
std::vector<float> conv6_bias(128);

std::vector<float> bn3_weight(128);
std::vector<float> bn3_bias(128);
std::vector<float> bn3_mm(128);
std::vector<float> bn3_mv(128);

std::vector<float> conv7_weight(128*128*3*3);
std::vector<float> conv7_bias(128);
std::vector<float> conv8_weight(128*128*3*3);
std::vector<float> conv8_bias(128);

std::vector<float> bn4_weight(128);
std::vector<float> bn4_bias(128);
std::vector<float> bn4_mm(128);
std::vector<float> bn4_mv(128);

std::vector<float> linear1_weight(16384);
std::vector<float> linear1_bias(1);


Mat conv1_input(1, 3, 150, 150);
Mat conv1_output(1, 32, 150, 150);
Mat relu1_output(1, 32, 150, 150);
Mat conv2_output(1, 32, 150, 150);
Mat relu2_output(1, 32, 150, 150);
Mat bn1_output(1,32,150,150);
Mat mp1_output(1, 32, 75, 75);

Mat conv3_output(1, 64, 75, 75);
Mat relu3_output(1, 64, 75, 75);
Mat conv4_output(1, 64, 75, 75);
Mat relu4_output(1, 64, 75, 75);
Mat bn2_output(1,64,75,75);
Mat mp2_output(1, 64, 37, 37);

Mat conv5_output(1, 128, 37, 37);
Mat relu5_output(1, 128, 37, 37);
Mat conv6_output(1, 128, 37, 37);
Mat relu6_output(1, 128, 37, 37);
Mat bn3_output(1,128,37,37);
Mat mp3_output(1, 128, 18, 18);

Mat conv7_output(1, 128, 18, 18);
Mat relu7_output(1, 128, 18, 18);
Mat conv8_output(1, 128, 18, 18);
Mat relu8_output(1, 128, 18, 18);
Mat bn4_output(1,128,18,18);
Mat mp4_output(1, 128, 9, 9);

Mat avg_output(1,128,4,4);

Mat view1_output(1,1,128,16);
Mat bmm_output(1,1,128,128);
Mat view2_output(1,1,1,128*128);
Mat ssr_output(1,1,1,128*128);
Mat l2_output (1,1,1,128*128);

Mat linear1_output(1,1,1,1);

std::vector<int> kernel_size22 = { 2,2 };
std::vector<int> kernel_size33 = { 3,3 };
std::vector<int> kernel_size55 = { 5,5 };

std::vector<int> stride11 = { 1,1 };
std::vector<int> stride22 = { 2,2 };

int padding1 = 1;
int padding2 = 2;

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

Mat padd(const Mat input,int this_padding)
{
    if(this_padding == 0)
        return input;
    int new_height = input.height + 2*this_padding;
    int new_width = input.width + 2*this_padding;
    Mat new_mat(input.dim,input.channel,new_height,new_width);
    std::fill(new_mat.tensor.begin(), new_mat.tensor.end(), 0);
    for (int c = 0; c < input.channel; ++c)
    {
        for (int h = 0; h < input.height; ++h)
        {
            for (int w = 0; w < input.width; ++w)
            {
                new_mat[c * new_height * new_width + (h + this_padding) * new_width + w + this_padding] = input[c * input.height * input.width + h * input.width + w];
            }
        }
    }
    return new_mat;
}

void conv2d(const Mat &input, Mat &output, const std::vector<float> &weight, const std::vector<float> &bias,
            const std::vector<int> &conv_kernel_size, const std::vector<int> &conv_stride, int conv_padding)
{

    int weight_pos = 0;
    int conv_kernel_max = conv_kernel_size[0]*conv_kernel_size[1];
    Mat padded_mat = padd(input,conv_padding);
    float sum = 0;
    int cnt[1000];
    memset(cnt, 0, sizeof cnt);
    int dx[25];
    memset (dx,0,sizeof dx);
    if(conv_kernel_max == 9) 
    {
        // 直接初始化数组的前9个元素
        dx[0] = 0;
        dx[1] = 1;
        dx[2] = 2;
        dx[3] = padded_mat.width;
        dx[4] = padded_mat.width + 1;
        dx[5] = padded_mat.width + 2;
        dx[6] = 2 * padded_mat.width;
        dx[7] = 2 * padded_mat.width + 1;
        dx[8] = 2 * padded_mat.width + 2;
    }
    if(conv_kernel_max == 25)
    {
        dx[0] = 0; dx[1] = 1; dx[2] = 2; dx[3] = 3; dx[4] = 4;
        dx[5] = padded_mat.width; dx[6] = padded_mat.width + 1; dx[7] = padded_mat.width + 2; dx[8] = padded_mat.width + 3; dx[9] = padded_mat.width + 4;
        dx[10] = 2 * padded_mat.width; dx[11] = 2 * padded_mat.width + 1; dx[12] = 2 * padded_mat.width + 2; dx[13] = 2 * padded_mat.width + 3; dx[14] = 2 * padded_mat.width + 4;
        dx[15] = 3 * padded_mat.width; dx[16] = 3 * padded_mat.width + 1; dx[17] = 3 * padded_mat.width + 2; dx[18] = 3 * padded_mat.width + 3; dx[19] = 3 * padded_mat.width + 4;
        dx[20] = 4 * padded_mat.width; dx[21] = 4 * padded_mat.width + 1; dx[22] = 4 * padded_mat.width + 2; dx[23] = 4 * padded_mat.width + 3; dx[24] = 4 * padded_mat.width + 4;
    }
    
    for (int i = 0; i < output.channel; ++i)
    {
        for (int d = 0; d < padded_mat.dim; ++d)
        {
            for (int c = 0; c < padded_mat.channel; ++c)
            {
                for (int h = 0; h < padded_mat.height; h += conv_stride[0])
                {
                    // std::cout << weight_pos<<' ';
                    //  if ((h + stride[0]) > height)
                    //  continue;
                    if (h + conv_kernel_size[0] > padded_mat.height)
                        continue;
                    for (int w = 0; w < padded_mat.width; w += conv_stride[1])
                    {
                        // if ((w + stride[1]) > width)
                        // continue;
                        if (w + conv_kernel_size[1] > padded_mat.width)
                            continue;
                        int index = d * padded_mat.channel * padded_mat.height * padded_mat.width + c * padded_mat.height * padded_mat.width + h * padded_mat.width + w;
                        // std::cout << index << std::endl;
                        sum = 0;
                        for (int i = 0; i < conv_kernel_max; ++i)
                        {
                            // std::cout << i<<' '<<dx[i] << ' ' << index + dx[i] << std::endl;
                            // std::cout << weight_pos + i << ' ' << index + dx[i] << std::endl;
                            sum += (padded_mat[index + dx[i]] * weight[weight_pos + i]);
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
}

void relu(const Mat &input,Mat &output)
{
    int sum = (input.dim * input.channel * input.height * input.width);
    for (int i = 0; i < sum; ++i)
    {
        if (input[i] < 0)
            output[i] = 0;
        else
            output[i] = input[i];
    }
}

void bn(const Mat& input, Mat&output,std::vector<float> weight,std::vector<float> bias, std::vector<float> rm, std::vector<float> rv)
{
    double eps = 1e-5;
    for(int c = 0;c<input.channel;++c)
    {
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
}

void mp(const Mat& input, Mat& output,std::vector<int> mp_kernel_size,std::vector<int> mp_stride)
{
    int input_h = input.height;
    int input_w = input.width;
    int out_h = output.height;
    int out_w = output.width;

    int mp_kernel_max = mp_kernel_size[0] * mp_kernel_size[1];
    int dx[4] = { 0, 1, input.width ,(input.width + 1) };
    int cnt = 0;
    for (int d = 0; d < input.dim; ++d)
    {
        for (int c = 0; c < input.channel; ++c)
        {
            //对每一个批次的每一个通道做mp2d
            for (int h = 0; h < input_h; h += mp_stride[1]) // 第一个
            {
                //std::cout << (h + stride[1]) << std::endl;
                if ((h + mp_stride[1]) > input_h) continue;
                for (int w = 0; w < input_w; w += mp_stride[0])
                {
                    if ((w + mp_stride[0]) > input_w) continue;
                    float max = -1000000;
                    int index = (d * input.channel * input_h * input_w) + (c * input_h * input_w) + (h * input_w) + w;
                    //std::cout << input[index] << std::endl;
                    //std::cout<<index<<std::endl;
                    for (int i = 0; i < mp_kernel_max; ++i)
                    {
                        if (input[index + dx[i]] > max)
                            max = input[index + dx[i]];
                        //std::cout << max << std::endl;

                    }
                    output[cnt++] = max;
                    //printf("%d\n", cnt);
                }
            }
        }
    }
}

void avgp(const Mat& input, Mat& output,std::vector<int> avgp_kernel_size,std::vector<int> avgp_stride) {
    int input_h = input.height;
    int input_w = input.width;
    int out_h = output.height;
    int out_w = output.width;

    for (int d = 0; d < input.dim; ++d) {
        for (int c = 0; c < input.channel; ++c) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    float sum = 0.0;
                    int count = 0;
                    for (int kh = 0; kh < avgp_kernel_size[0]; ++kh) {
                        for (int kw = 0; kw < avgp_kernel_size[1]; ++kw) {
                            int h = oh * avgp_stride[1] + kh;
                            int w = ow * avgp_stride[0] + kw;
                            if (h < input_h && w < input_w) {
                                int index = (d * input.channel * input_h * input_w) + (c * input_h * input_w) + (h * input_w) + w;
                                sum += input[index];
                                count++;
                            }
                        }
                    }
                    output[(d * output.channel * out_h * out_w) + (c * out_h * out_w) + (oh * out_w) + ow] = sum / count;
                }
            }
        }
    }
}

double view(const Mat& input, Mat & output)
{
    double start = get_current_time();
    int sum = (input.dim * input.channel * input.height * input.width);
    for (int i = 0; i < sum; ++i)
    {
        output[i] = input[i];
        //if (i)
        //    output[i] = input[i];
        //else
        //    output[i] = input[i];

    }
    double end = get_current_time();
    return (end - start);
}

void bmm(const Mat& input,Mat& output) 
{
    for(int i = 0;i<input.height;++i)
    {
        for(int j = 0;j<input.height;++j)
        {
            double sum = 0;
            for(int k = 0;k<input.width;++k)
            {
                int index1 = i*input.width+k;
                int index2 = j*input.width+k;
                sum += input[index1]*input[index2];
            }
            int index = i*input.height+j;
            output[index] = sum/(input.width);
            //output[index] = sum;
        }
    }
}

void SignSquareRoot(Mat& input, Mat& output) 
{
    for (int d = 0; d < input.dim; ++d) {
        for (int c = 0; c < input.channel; ++c) {
            for (int h = 0; h < input.height; ++h) {
                for (int w = 0; w < input.width; ++w) {
                    int index = (d * input.channel * input.height * input.width) + (c * input.height * input.width) + (h * input.width) + w;
                    float value = input[index];
                    output[index] = std::copysign(std::sqrt(std::abs(value)+1e-10), value);
                }
            }
        }
    }
}

void L2Normalization(Mat& input, Mat& output) 
{
    double sum = 0;
    for(int w = 0;w<input.width;++w)
    {
        sum += input[w]*input[w];
    }
    sum = sqrt(sum+1e-10);
    for(int w = 0;w<input.width;++w)
    {
        output[w] = input[w]/sum;
    }

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

float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

void preread()
{
    std::string conv1_weight_path = ".\\src" PATH_SEPARATOR "conv1.weight.bin";
    std::string conv1_bias_path = ".\\src" PATH_SEPARATOR "conv1.bias.bin";
    readBinaryFile(conv1_weight_path, conv1_weight);
    readBinaryFile(conv1_bias_path, conv1_bias);

    std::string conv2_weight_path = ".\\src" PATH_SEPARATOR "conv2.weight.bin";
    std::string conv2_bias_path = ".\\src" PATH_SEPARATOR "conv2.bias.bin";
    readBinaryFile(conv2_weight_path, conv2_weight);
    readBinaryFile(conv2_bias_path, conv2_bias);

    std::string conv3_weight_path = ".\\src" PATH_SEPARATOR "conv3.weight.bin";
    std::string conv3_bias_path = ".\\src" PATH_SEPARATOR "conv3.bias.bin";
    readBinaryFile(conv3_weight_path, conv3_weight);
    readBinaryFile(conv3_bias_path, conv3_bias);

    std::string conv4_weight_path = ".\\src" PATH_SEPARATOR "conv4.weight.bin";
    std::string conv4_bias_path = ".\\src" PATH_SEPARATOR "conv4.bias.bin";
    readBinaryFile(conv4_weight_path, conv4_weight);
    readBinaryFile(conv4_bias_path, conv4_bias);

    std::string conv5_weight_path = ".\\src" PATH_SEPARATOR "conv5.weight.bin";
    std::string conv5_bias_path = ".\\src" PATH_SEPARATOR "conv5.bias.bin";
    readBinaryFile(conv5_weight_path, conv5_weight);
    readBinaryFile(conv5_bias_path, conv5_bias);

    std::string conv6_weight_path = ".\\src" PATH_SEPARATOR "conv6.weight.bin";
    std::string conv6_bias_path = ".\\src" PATH_SEPARATOR "conv6.bias.bin";
    readBinaryFile(conv6_weight_path, conv6_weight);
    readBinaryFile(conv6_bias_path, conv6_bias);

    std::string conv7_weight_path = ".\\src" PATH_SEPARATOR "conv7.weight.bin";
    std::string conv7_bias_path = ".\\src" PATH_SEPARATOR "conv7.bias.bin";
    readBinaryFile(conv7_weight_path, conv7_weight);
    readBinaryFile(conv7_bias_path, conv7_bias);

    std::string conv8_weight_path = ".\\src" PATH_SEPARATOR "conv8.weight.bin";
    std::string conv8_bias_path = ".\\src" PATH_SEPARATOR "conv8.bias.bin";
    readBinaryFile(conv8_weight_path, conv8_weight);
    readBinaryFile(conv8_bias_path, conv8_bias);

    std::string linear1_weight_path = ".\\src" PATH_SEPARATOR "linear1.weight.bin";
    std::string linear1_bias_path = ".\\src" PATH_SEPARATOR "linear1.bias.bin";
    readBinaryFile(linear1_weight_path, linear1_weight);
    readBinaryFile(linear1_bias_path, linear1_bias);

    
    std::string bn1_weight_path = ".\\src" PATH_SEPARATOR "bn1.weight.bin";
    std::string bn1_bias_path = ".\\src" PATH_SEPARATOR "bn1.bias.bin";
    readBinaryFile(bn1_weight_path, bn1_weight);
    readBinaryFile(bn1_bias_path, bn1_bias);
    std::string bn1_rm_path = ".\\src" PATH_SEPARATOR "bn1.running_mean.bin";
    std::string bn1_rv_path = ".\\src" PATH_SEPARATOR "bn1.running_var.bin";
    readBinaryFile(bn1_rm_path, bn1_mm);
    readBinaryFile(bn1_rv_path, bn1_mv);

    std::string bn2_weight_path = ".\\src" PATH_SEPARATOR "bn2.weight.bin";
    std::string bn2_bias_path = ".\\src" PATH_SEPARATOR "bn2.bias.bin";
    readBinaryFile(bn2_weight_path, bn2_weight);
    readBinaryFile(bn2_bias_path, bn2_bias);
    std::string bn2_rm_path = ".\\src" PATH_SEPARATOR "bn2.running_mean.bin";
    std::string bn2_rv_path = ".\\src" PATH_SEPARATOR "bn2.running_var.bin";
    readBinaryFile(bn2_rm_path, bn2_mm);
    readBinaryFile(bn2_rv_path, bn2_mv);

    std::string bn3_weight_path = ".\\src" PATH_SEPARATOR "bn3.weight.bin";
    std::string bn3_bias_path = ".\\src" PATH_SEPARATOR "bn3.bias.bin";
    readBinaryFile(bn3_weight_path, bn3_weight);
    readBinaryFile(bn3_bias_path, bn3_bias);
    std::string bn3_rm_path = ".\\src" PATH_SEPARATOR "bn3.running_mean.bin";
    std::string bn3_rv_path = ".\\src" PATH_SEPARATOR "bn3.running_var.bin";
    readBinaryFile(bn3_rm_path, bn3_mm);
    readBinaryFile(bn3_rv_path, bn3_mv);

    std::string bn4_weight_path = ".\\src" PATH_SEPARATOR "bn4.weight.bin";
    std::string bn4_bias_path = ".\\src" PATH_SEPARATOR "bn4.bias.bin";
    readBinaryFile(bn4_weight_path, bn4_weight);
    readBinaryFile(bn4_bias_path, bn4_bias);
    std::string bn4_rm_path = ".\\src" PATH_SEPARATOR "bn4.running_mean.bin";
    std::string bn4_rv_path = ".\\src" PATH_SEPARATOR "bn4.running_var.bin";
    readBinaryFile(bn4_rm_path, bn4_mm);
    readBinaryFile(bn4_rv_path, bn4_mv);

}

int forward(Mat & input, int i)
{
    conv2d(conv1_input,conv1_output,conv1_weight,conv1_bias,kernel_size55,stride11,padding2);
    relu(conv1_output,relu1_output);
    conv2d(relu1_output,conv2_output,conv2_weight,conv2_bias,kernel_size55,stride11,padding2);
    relu(conv2_output,relu2_output);
    bn(relu2_output,bn1_output,bn1_weight,bn1_bias,bn1_mm, bn1_mv);
    mp(bn1_output,mp1_output,kernel_size22,stride22);

    conv2d(mp1_output,conv3_output,conv3_weight,conv3_bias,kernel_size55,stride11,padding2);
    relu(conv3_output,relu3_output);
    conv2d(relu3_output,conv4_output,conv4_weight,conv4_bias,kernel_size55,stride11,padding2);
    relu(conv4_output,relu4_output);
    bn(relu4_output,bn2_output,bn2_weight,bn2_bias,bn2_mm, bn2_mv);
    mp(bn2_output,mp2_output,kernel_size22,stride22);

    conv2d(mp2_output,conv5_output,conv5_weight,conv5_bias,kernel_size33,stride11,padding1);
    relu(conv5_output,relu5_output);
    conv2d(relu5_output,conv6_output,conv6_weight,conv6_bias,kernel_size33,stride11,padding1);
    relu(conv6_output,relu6_output);

    bn(relu6_output,bn3_output,bn3_weight,bn3_bias,bn3_mm, bn3_mv);
    mp(bn3_output,mp3_output,kernel_size22,stride22);

    conv2d(mp3_output,conv7_output,conv7_weight,conv7_bias,kernel_size33,stride11,padding1);
    relu(conv7_output,relu7_output);
    conv2d(relu7_output,conv8_output,conv8_weight,conv8_bias,kernel_size33,stride11,padding1);
    relu(conv8_output,relu8_output);
    bn(relu8_output,bn4_output,bn4_weight,bn4_bias,bn4_mm, bn4_mv);
    mp(bn4_output,mp4_output,kernel_size22,stride22);

    avgp(mp4_output,avg_output,kernel_size22,stride22);

    view(avg_output,view1_output);
    bmm(view1_output,bmm_output);
    view(bmm_output,view2_output);
    SignSquareRoot(view2_output,ssr_output);
    L2Normalization(ssr_output,l2_output);
    linear(l2_output,linear1_output,linear1_weight,linear1_bias);

    std::cout<<sigmoid(linear1_output[0]);


    return 0;
}

int main()
{
    preread();
    pretensor(conv1_input);
    forward(conv1_input,0);
    return 0;
}