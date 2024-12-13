# 使用说明

## 模型微调(HF)

- 理论上支持所有Llamafactory支持的模型, 也就是各种的HF格式模型

### 关键参数

- 模型检查点:以下模型检查点已测试
    - 9g_2b_sft_hf(九格2B)
    - 9g_8b_sft_hf(九格8B)
    - t6-iter_031500(珠算模型)
- 训练方式: 
    - 选择Lora_drop时, 使用Lora_drop训练方式
    - 9g_2b_sft_hf(九格2B)与t6-iter_031500(珠算模型)支持Lora_Drop方式
- 预处理工作线程数
    - 当使用九格8B的时候, 必须设置为1, 否则会报错
- 数据集
    - 可以选择identity\alpaca_en_demo\alpaca_zh_demo等作演示用
- CUDA_VISIBLE_DEVICES
    - 根据当前节点的GPU空闲情况选择

## 模型微调(BMTrain)

- 支持pt格式的九格2B和九格8B模型

### 关键参数

- 模型变体: 根据当前是2B还是8B模型选择
- 模型检查点: 以下模型检查点已测试
    - 9g_2b_sft_hf(九格2B)
    - 9g_8b_sft_hf(九格8B)
- 数据集
    - 可以选择alpaca\skypile\slimpajama三个数据集

## 模型测评

- 理论上支持所有VLLM支持的模型

### 关键参数

- 模型检查点:以下模型检查点已测试
    - 9g_2b_sft_hf(九格2B)
    - 9g_8b_sft_hf(九格8B)
    - t6-iter_031500(珠算模型)

## 模型推理

- 理论上支持所有VLLM支持的模型
- 点击"加载模型"来加载一个模型到内存中
- 点击"卸载模型"来从内存中卸载模型

### 关键参数

- 模型检查点:以下模型检查点已测试
    - 9g_2b_sft_hf(九格2B)
    - 9g_8b_sft_hf(九格8B)
    - t6-iter_031500(珠算模型)