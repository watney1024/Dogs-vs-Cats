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

Mat conv2_input(1, 3, 150, 150); 
Mat conv2_output(1, 32, 150, 150);


std::vector<int> conv_kernel_size = { 5,5 };
std::vector<int> conv_stride = { 1,1 };
int padding = 2;

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

double get_current_time()
{
    auto now = std::chrono::high_resolution_clock::now();
    auto usec = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch());
    return usec.count() / 1000.0;
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

double conv2d(const Mat &input, Mat &output, const std::vector<float> &weight, const std::vector<float> &bias,
            const std::vector<int> &kernel_size, const std::vector<int> &stride, int this_padding)
{
    double start = get_current_time();
    int weight_pos = 0;
    int conv_kernel_max = kernel_size[0]*kernel_size[1];
    Mat padded_mat = padd(input,padding);
    float sum = 0;
    int cnt[100];
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
    int oc = output.channel;
    int pd = padded_mat.dim;
    int pc = padded_mat.channel;
    int ph = padded_mat.height;
    int pw = padded_mat.width;
    //#pragma omp parallel for
    //#pragma parallel for
    for (int i = 0; i < oc; ++i)
    {
        //#pragma omp parallel for
        for (int d = 0; d < pd; ++d)
        {
            //#pragma omp parallel for
            for (int c = 0; c < pc; ++c)
            {
                //#pragma omp parallel for
                for (int h = 0; h < ph; h += conv_stride[0])
                {
                    // std::cout << weight_pos<<' ';
                    //  if ((h + stride[0]) > height)
                    //  continue;
                    if (h + conv_kernel_size[0] > ph)
                        continue;
                    //#pragma omp parallel for
                    for (int w = 0; w < pw; w += conv_stride[1])
                    {
                        // if ((w + stride[1]) > width)
                        // continue;
                        if (w + conv_kernel_size[1] > pw)
                            continue;
                        int index = d * pc * ph * pw + c * ph * pw + h * pw + w;
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
    double end = get_current_time();
    return (end-start);
}


int main()
{
    //omp_set_num_threads(2);
    std::vector<float> conv2_weight(32 * 3 * 5 * 5);
    std::vector<float> conv2_bias(32);
    std::string conv2_weight_path = ".\\src" PATH_SEPARATOR "conv1.weight.bin";
    std::string conv2_bias_path = ".\\src" PATH_SEPARATOR "conv1.bias.bin";
    readBinaryFile(conv2_weight_path, conv2_weight);
    readBinaryFile(conv2_bias_path, conv2_bias);
    pretensor(conv2_input);
    
    //for (int cc = 0;cc<5;++cc)
        //std::cout<<conv2d(conv2_input,conv2_output,conv2_weight,conv2_bias,conv_kernel_size,conv_stride,padding)<<std::endl;
    conv2d(conv2_input,conv2_output,conv2_weight,conv2_bias,conv_kernel_size,conv_stride,padding);
    printMat(conv2_output);
    return 0;
}