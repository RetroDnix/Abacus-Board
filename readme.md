# 一、简介

为了方便二次开发与应用，我们基于一批优秀开源项目开发了适用于“珠算”代码大模型的微调适配平台，可以通过零代码的方式实现模型的微调、评测与推理功能。

## 相关链接

珠算大模型：https://github.com/HIT-SCIR/Abacus

微调适配平台：https://github.com/RetroDnix/Abacus-Board

如果你遇到了Bug，可以在[微调适配平台的代码仓库](https://github.com/RetroDnix/Abacus-Board)中向我们提出issue。


# 二、安装

微调适配平台使用的主要Python库及其版本如下表所示：

|  库 | 版本 | 说明 |
| ----------- | ----------- | ------ |
| Pytorch | 2.3.0 | 深度学习基础库 |
| LLamafactory | 0.9.0 | 用于模型微调 |
| accelerate | 0.34.2 | 用于加速模型训练 |
| OpenCompass | 0.2.3 | 用于模型评估 |
| VLLM | 0.4.3 | 用于模型推理 |
| Streamlit | latest | 用于生成UI界面 |

由于上述库之间依赖关系较难处理、且部分库配置比较麻烦。为了简化使用过程，我们推荐直接使用通过Docker镜像使用微调适配平台。同时，我们给出构建Docker镜像使用的dockerfile以及相关数据文件，你也可以选择自行配置环境。

## 1、通过Docker使用

我们提供的Docker镜像基于NVIDIA的cuda镜像构建。对应的版本为cuda12.4.1、ubuntu22.04。在开始之前，请确保你已经安装了Docker与NVIDIA-Container-Toolkit。


### 拉取微调适配平台镜像与示例数据集
docker pull retrodnix/abacus-board:v1.3

```
# 拉取源码、dockerfile、示例数据集等，非必须
git clone https://github.com/RetroDnix/Abacus-Board
```

### 运行镜像

```
docker run \
    --gpus all \
    --network host \
    -it \
    -v ./saves:/workspace/Abacus-Board/saves \
    -v ./data-example:/workspace/Abacus-Board/data \
    retrodnix/abacus-board:v1.3 
```

参数说明：
- `--gpus all`: 在运行镜像时使用所有GPU
- `--network host`: host网络模式
- `-v ./saves:/workspace/Abacus-Board/saves`: 将./saves替换为你的模型权重储存路径
- `-v ./data:/workspace/Abacus-Board/data`: 将./data替换为你的微调数据集储存路径。我们使用与LLamafactory相同的数据集格式，你可以直接使用克隆的`Abacus-Board`仓库中的`data-example`文件夹作为你的数据集路径。

运行以上命令后，在镜像的交互式终端中通过`streamlit run main.py`启动微调适配平台。平台会启动并映射到你主机的8888端口中。此时你可以通过访问`localhost:8888`来使用平台。

## 2、自行构建Docker镜像

如果你存在系统、cuda版本不匹配等情况。你可以选择自行构建环境。

### 拉取相关文件

```
git clone https://github.com/RetroDnix/Abacus-Board
```

### 运行构建

在这一步开始前，你需要安装Docker与NVIDIA-Container-Toolkit，并且根据需要修改dockerfile。比如将第一行的FROM语句改为你想使用的CUDA镜像。之后，使用以下命令开始构建：

```
cd Abacus-Board
docker build -t abacus-board:v2 .
```

你可以将`abacus-board:v2`更改为想使用的名称和版本号。

如果以上方式均不符合你的需求，你可以考虑根据dockerfile中的命令，在物理机的虚拟环境中完成环境的搭建。