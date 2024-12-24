#include <vector>
#include <iostream>
#include <cmath>
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

void pretensor(Mat& input) {
    for (int i = 0; i < input.channel; ++i) {
        for (int j = 0; j < input.height; ++j) {
            for (int k = 0; k < input.width; ++k) {
                int index = i * input.height * input.width + j * input.width + k;
                float value = std::sin(static_cast<float>(index));
                input[index] = value;
            }
        }
    }
}

// 批量矩阵乘法算子(由于batch_size是1，其实就是矩阵乘法)
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

// 示例用法
int main() 
{
    Mat input(1,1,128,16);
    Mat output(1,1,128,128);
    pretensor(input);
    //printMat(input);
    bmm(input,output);
    printMat(output);
    return 0;
}