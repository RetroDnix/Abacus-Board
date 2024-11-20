# CUDA基础镜像
FROM nvidia/cuda:12.2.2-cudnn-runtime-ubuntu22.04

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
RUN git clone https://github.com/RetroDnix/Abacus-Board Abacus-Board && git checkout develop

WORKDIR /workspace/Abacus-Board

RUN cd LLaMA-Factory && \
    pip3 install -e ".[torch,metrics]" torch==2.3.0 accelerate==0.34.2 --no-cache-dir

RUN git clone https://github.com/thunlp/OpenDelta && \
    cd OpenDelta && \
    python setup.py install

RUN pip3 install bmtrain==1.0.0 --no-cache-dir

# flash-attn vllm
# flash_attn-2.5.9.post1+cu122torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
# https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.9.post1/flash_attn-2.5.9.post1+cu122torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# vllm-0.5.0.dev0+cu122-cp310-cp310-linux_x86_64.whl
# https://qy-obs-6d58.obs.cn-north-4.myhuaweicloud.com/vllm-0.5.0.dev0%2Bcu122-cp310-cp310-linux_x86_64.whl

RUN pip3 install ./wheels/* --no-cache-dir

RUN pip3 install streamlit --no-cache-dir

RUN pip3 install einops pytrie transformers matplotlib h5py sentencepiece --no-cache-dir

RUN pip3 install protobuf==3.20.0 tensorboard tensorboardX --no-cache-dir

# 安装opencompass
# 该opencompass拷贝在0.2.3版本的基础上进行了小幅度修改
RUN cd opencompass && \
    pip3 install -e .[full,vllm] torch==2.3.0 --no-cache-dir 

# 准备eval-plus
RUN git clone --recurse-submodules https://github.com/open-compass/human-eval.git && \
    cd human-eval && \
    pip3 install -e . --no-cache-dir && \
    pip3 install -e evalplus --no-cache-dir 

# 将humaneval+与mbpp+数据拷贝到缓存目录,这是为了避免因为计算节点断网导致数据下载失败
RUN mkdir -p ~/.cache/evalplus && \
    cp EvalPlusData/HumanEvalPlus-v0.1.9.jsonl ~/.cache/evalplus && \
    cp EvalPlusData/MbppPlus-v0.1.0.jsonl ~/.cache/EvalPlusData

RUN git pull && git checkout develop

CMD ["streamlit","run","main.py","--server.port=8888","--server.address=127.0.0.1"]
