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

void pretensor_simple(Mat& input) {
    for (int i = 0; i < input.channel; ++i) {
        for (int j = 0; j < input.height; ++j) {
            for (int k = 0; k < input.width; ++k) {
                int index = i * input.height * input.width + j * input.width + k;
                input[index] = index;
            }
        }
    }
}

// L2 Normalization 算子(对应torch中dim=1情况)
void L2Normalization(Mat& input, Mat& output) 
{
    for (int h = 0; h < input.height; ++h) 
    {
        for(int w = 0;w<input.width;++w)
        {
            double sum = 0;
            for (int c = 0; c < input.channel; ++c)
            {
                int index = c * input.height * input.width + h * input.width + w;
                sum += (input[index]*input[index]);
            }
            sum = sqrt(sum);
            for (int c = 0; c < input.channel; ++c)
            {
                int index = c * input.height * input.width + h * input.width + w;
                output[index] = input[index]/sum;
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

int main() {
    Mat input(1,3,150,150);
    Mat output(1,3,150,150);
    pretensor(input);
    L2Normalization(input, output);
    //printMat(input);
    printMat(output);
    return 0;
}