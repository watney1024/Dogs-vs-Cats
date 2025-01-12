# 猫狗分类

## 2024年10月27日
上传了新文件，增加了测时间的功能

## 2024年12月13日
上传了debug过程的文档

## 2024年12月15日

上传了项目汇报的PPT，其中还有一些实验没有做完

## 2024年12月17日

修改了PPT的p2和p3，上传了Bilinear的代码

## 2024年12月20日

添加了bcnn的算子的列表

## 2024年12月21日

提交了auto_training的代码,可以通过.sh脚本实现自动训练，并且把数据保存为log文件

## 2024年12月22日

把auto_taining改完了，只要运行suto.sh脚本就可以自动运行，使用步骤如下

conda activate pytorch #激活虚拟环境

chmod +x auto.sh

./auto.sh & #执行sh脚本，可以在这个里面修改参数  


## 2024年12月24日

写了并且测试完了l2-normalization，sign-squared-root，avgpool和bmm算子。
目前还差conv(之前没加padding)，和bn算子没写。

## 2024年12月25日

把所有算子都写完了，并且最终结果和python上一样(c++的output: 0.153745，python的output: 0.153746)。  
在写bn算子的时候我发现bn要在relu前面使用，这样效果更好，修改后跑了下正确率可以提高3%左右。  
修改前val_acc:0.7811，修改后val_acc:0.8087  
把所有的速度都测完了(还未加openmp)，记录在了bcnn文件夹下的算子列表.md中，整个推理用时在9153ms左右，卷积操作用时占了大头。

## 2024年12月29日

使用openmp在for循环前面加了# pragma parallel for,但是效果不大，运行时间在9258ms左右,可能是哪里设置的有问题。

## 2024年12月30日

才发现昨天的并行语句有问题，应该是#pragma omp parallel for,写成了#pragma parallel for, 虽然没报错, 但是并没有并行。  
又试了下#pragma opm parallel for,故意打错的指令，但是不报错，还能正常运行，但是不起作用，还是有点坑的。

## 2025年1月11日

上传了科研训练的总结，以及通过hook可视化中间层的文件。

## 2025年1月12日

给conv算子加上了并行，在并行的时候发现如果不符合规则会有额外的运行开销，包括并行里面有continue，多个线程同时访问变量（包括自增等）。解决方法就是首先修改掉continue部分，使得循环次数确定，之后在并行过程中多使用“绝对位置”，少用“相对位置”，例如

```c++
output[cnt[c]++] += sum;
```

由于多个线程同时访问，使得自增操作会有问题，解决方法要么使用critical锁住线程（但是会有额外时间开销），要么就是把cnt[c]修改成

```c++
int output_index = i * output.height * output.width + h * output.width + w;
output[output_index] += (padded_mat[index + dx[m]] * weight[weight_pos + m]);
```

这样可以避免这种情况。这样子推理整体运行时间从9秒降到了3秒。
