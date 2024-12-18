# CUDA基础镜像
FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04

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
COPY ./Abacus-Board /workspace/Abacus-Board

WORKDIR /workspace/Abacus-Board

RUN pip3 install vllm==0.4.3 --no-cache-dir

RUN cd LLaMA-Factory && \
    pip3 install -e ".[torch,metrics]" torch==2.3.0 accelerate==0.34.2 --no-cache-dir

RUN cd OpenDelta && \
    python setup.py install

RUN pip3 install bmtrain==1.0.0 --no-cache-dir

RUN cd LoRA && \
    python setup.py install

RUN pip3 install ./wheels/* --no-cache-dir

RUN pip3 install streamlit --no-cache-dir

RUN pip3 install einops pytrie transformers matplotlib h5py sentencepiece --no-cache-dir

RUN pip3 install protobuf==3.20.0 tensorboard tensorboardX --no-cache-dir

# 安装opencompass
# 该opencompass拷贝在0.2.3版本的基础上进行了小幅度修改
RUN cd opencompass && \
    pip3 install -e .[full,vllm] torch==2.3.0 --no-cache-dir 

# 准备eval-plus
RUN cd human-eval && \
    pip3 install -e . --no-cache-dir && \
    pip3 install -e evalplus --no-cache-dir 

# 将humaneval+与mbpp+数据拷贝到缓存目录,这是为了避免因为计算节点断网导致数据下载失败
RUN mkdir -p ~/.cache/evalplus && \
    cp EvalPlusData/HumanEvalPlus-v0.1.9.jsonl ~/.cache/evalplus && \
    cp EvalPlusData/MbppPlus-v0.1.0.jsonl ~/.cache/EvalPlusData

RUN rm /usr/local/lib/python3.10/site-packages/transformers/models/llama/modeling_llama.py

COPY ./modeling_llama.py /usr/local/lib/python3.10/site-packages/transformers/models/llama/

CMD ["streamlit","run","main.py","--server.port=8888"]
