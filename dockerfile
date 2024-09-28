# CUDA基础镜像
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# 安装基础包
RUN apt update && \
    apt install -y \
        git wget build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev \
        libreadline-dev libffi-dev libsqlite3-dev libbz2-dev liblzma-dev && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /temp

# 下载python
RUN wget https://mirrors.huaweicloud.com/python/3.10.15/Python-3.10.15.tgz && \
    tar -xvf Python-3.10.15.tgz

# 编译&安装python
RUN cd Python-3.10.15 && \
    ./configure --enable-optimizations && \
    make -j8 && \
    make install

WORKDIR /workspace

RUN rm -r /temp && \
    ln -s /usr/local/bin/python3 /usr/local/bin/python && \
    pip3 config set global.index-url https://mirrors.aliyun.com/pypi/simple/

# 安装微调平台
RUN git clone https://github.com/RetroDnix/Abacus-Board Abacus-Board

WORKDIR /workspace/Abacus-Board

# 安装vllm 0.4.3
RUN pip3 install vllm==0.4.3 streamlit --no-cache-dir 

# 安装Llamafactory
RUN git clone https://github.com/hiyouga/LLaMA-Factory.git && \
    cd LLaMA-Factory && \
    git checkout v0.9.0 && \
    pip3 install -e ".[torch,metrics]" torch==2.3.0 --no-cache-dir

# 安装opencompass
# 该opencompass拷贝在0.2.3版本的基础上进行了小幅度修改
RUN cd opencompass && \
    pip3 install -e .[full,vllm] torch==2.3.0 --no-cache-dir 

# 准备eval-plus
RUN git clone --recurse-submodules git@github.com:open-compass/human-eval.git && \
    cd human-eval && \
    pip3 install -e . --no-cache-dir && \
    pip3 install -e evalplus --no-cache-dir 

# 将humaneval+与mbpp+数据拷贝到缓存目录,这是为了避免因为计算节点断网导致数据下载失败
RUN mkdir -p ~/.cache/evalplus && \
    cp EvalPlusData/HumanEvalPlus-v0.1.9.jsonl ~/.cache/evalplus && \
    cp EvalPlusData/MbppPlus-v0.1.0.jsonl ~/.cache/evalplus && \

CMD ["streamlit","run","main.py"]