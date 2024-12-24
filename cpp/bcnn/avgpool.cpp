#include <cmath>
#include <iostream>
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

std::vector<int> padding;
std::vector<int> kernel_size;
std::vector<int> stride;
std::vector<int> dilation;

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

void avgpool2d(const Mat& input, Mat& output) {
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
                    for (int kh = 0; kh < kernel_size[0]; ++kh) {
                        for (int kw = 0; kw < kernel_size[1]; ++kw) {
                            int h = oh * stride[1] + kh;
                            int w = ow * stride[0] + kw;
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


int main()
{
    padding.assign(0, 0);     
    kernel_size.assign(2, 2); 
    stride.assign(2, 2);      

    Mat mp1_input(1, 3, 150, 150);
    Mat mp1_output(1, 3, 75, 75);

    pretensor(mp1_input);
    avgpool2d(mp1_input, mp1_output);//用sin生成的数据测试
    //printMat(mp1_input);
    printMat(mp1_output);

    return 0;
}
