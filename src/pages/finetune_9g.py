import streamlit as st
from src.utils.filesystem import getDS_9g
from src.components.top import top_page
from src.widgets.stated_widgets import number_input, text_input, slider, toggle, selectbox
from src.utils.mem_gc import torch_gc, abort_process
from src.runner_9g import train_9g, validate_args_9g

from datetime import datetime
from signal import SIGTERM
import os
from copy import deepcopy
from subprocess import Popen

TRAINER = "trainer_9g"

def finetune_9g():
    state = st.session_state
    
    if "train_args_9g" not in state:
        default_name = "Finetune_9g_%s"%datetime.now().strftime("%y%m%d-%H%M")
        state["train_args_9g"] = {
            #训练的名称，模型和log等信息会存储在该文件夹中
            "model_unique": default_name,
            #config位置，在configs/目录中
            "config": default_name,
            #训练batch size
            "batch_size":1,
            "dataset_dir":"./data-example",
            "dataset":"",
            "dataset_language":"zh",
            "output_dir":"",
            #多久存一次
            "save":True,
            "save_iters":500,
            #总的iteration
            "train_iters":10000,
            #在dataset_config/目录下，数据集的设置
            "dataset_config":"fm9g_sft",
            #dataloder 的加载线程的设置，如果配置较好，可以适量提高
            "dataloader_num_threads":1,
            "dataloader_prefetch":1,
            "dataloader_prefetch_factor":1,
            "dataloader_num_workers":1,
            "parallel_load_datastate":"8",
            #学习率
            "lr":1e-2,
            #warmup的次数
            "warmup_iters":20,
            #学习率下降方法
            "lr_scheduler":"cosine",
            #drop的比例
            "drop_iters":0.1,
            #drop比例
            "drop_begin":-1,
            "drop_rate":0.5,
            #是否use checkpoint，建议使用
            "use_checkpoint":"1",
            "n_gpus":1,
            "cuda_visible_devices":"0",
        }

    train_args = state["train_args_9g"]
    
    if "trainer" not in state:
        state["trainer"] = None
    
    if "run_every" not in state:
        state["run_every"] = 2
    
    if "cached_plot" not in state:
        state["cached_plot"] = None
    
    if "cached_log" not in state:
        state["cached_log"] = ""

    st.markdown("##### 训练名称")
    model_unique = text_input(
        "本次训练的唯一标识,训练的名称，模型和log等信息会存储在该文件夹中",
        train_args,
        key="model_unique",
        prefix="_finetune_9g_"
    )
    st.divider()

    st.markdown("##### 数据集")
    st.caption('''请选择存放微调数据集的文件夹, 文件夹应包含一个格式为"{input:"",output:""}"的jsonl数据集''')
    col_ds_path, col_ds_lan, col_ds = st.columns([2, 2, 6])
    with col_ds_path:
        text_input("数据集储存路径", train_args, key="dataset_dir")
    with col_ds_lan:
        selectbox(
            label="数据集语言",
            data=train_args,
            options=["zh","en"],
            key="dataset_language",
            prefix="_finetune_9g_"
        )
    with col_ds:
        all_datasets, message = getDS_9g(train_args.get("dataset_dir", "./data-example"))
        train_args["dataset"] = st.selectbox(
            label="选择数据集",
            options=all_datasets,
            placeholder="未选择",
        )
    if message != "":
        st.error(message, icon=":material/warning:")

    st.divider()
    
    st.markdown(
        "##### 结果输出",
    )
    col_output_dir, col_output_name = st.columns([3, 7])
    with col_output_dir:
        output_path = st.text_input("输出路径", "./saves")
    with col_output_name:
        output_name = st.text_input(
            "结果保存名称",
            train_args.get("model_unique","train_9g"),
            key="output_dir",
        )
        train_args["output_dir"] = os.path.join(output_path, output_name)
    st.divider()

    st.markdown(
        "##### 训练设置",
    )
    col_learning_rate, col_batch_size, col_num_train_epochs = st.columns(3)
    with col_learning_rate:
        number_input(
            "学习率", 0.0, 1.0, data=train_args, key="lr", step=1e-5, format="%.5f",prefix="_finetune_9g_"
        )
    with col_batch_size:
        number_input(
            "训练的批次大小", 0, 16, data=train_args, key="batch_size",prefix="_finetune_9g_"
        )
    with col_num_train_epochs:
        number_input(
            "训练步数", 0, 100000, data=train_args, key="train_iters", step=1, prefix="_finetune_9g_"
        )

    col_scheduler, col_save_iters = st.columns(2)
    with col_scheduler:
        schedulers = ["cosine"]
        selectbox(
            label="学习率调度器",
            options=schedulers,
            data=train_args,
            key="lr_scheduler"
        )
    with col_save_iters:
        number_input("保存间隔", 0, 10000, data=train_args, key="save_iters",prefix="_finetune_9g_")

    slider("预热步数", 0, 1000, data=train_args, key="warmup_iters", step=1)

    st.divider()
    
    with st.sidebar:
        model_variant = st.selectbox("模型变体", options=["fm9g_2b", "fm9g_8b"], key="finetune_9g_config")
        
        ckpt_path, ckpt = top_page("finetune_ckpt_parm")
        train_args["model_name_or_path"] = None if ckpt == None else os.path.join(ckpt_path, ckpt)
        
        st.markdown("##### 资源分配")
        
        col_cuda_visible_devices, col_n_gpus = st.columns(2)
        with col_cuda_visible_devices:
            text_input(
                "CUDA_VISIBLE_DEVICES", 
                data = train_args,
                key="cuda_visible_devices",
                prefix="_finetune_9g_",
            )
        with col_n_gpus:
            number_input(
                label="GPU数量",
                data=train_args,
                key="n_gpus",
                min_value=1,
                max_value=8,
                prefix="_finetune_9g_",
            )
        number_input("预处理工作线程数", 0, 128, data=train_args, key="dataloader_num_workers")
        
        st.divider()
        
        start_training = st.button("开始微调", use_container_width=True)
        
        if start_training:
            if state.get(TRAINER, None) is not None:
                st.error("请等待当前训练完成再继续操作。", icon=":material/warning:")
            elif validate_args_9g(train_args) != "":
                st.error(validate_args_9g(train_args), icon=":material/warning:")
            else:
                trainer = train_9g(train_args, model_variant)
                if(trainer == None):
                    st.error("模型微调出错")
                else:
                    state[TRAINER] = trainer
                    st.toast("开始模型微调", icon=":material/info:")
                    print("开始模型微调")
                    st.rerun(scope="app")
        
        if state.get(TRAINER, None) is not None:
            trainer = state[TRAINER]
            if st.button("停止微调", key="stop", use_container_width=True, type="primary"):
                state[TRAINER] = None
                abort_process(trainer.pid)
                torch_gc()
                st.rerun(scope="app")
        
        # st.html(body = '''    
        #     <div style="text-align: center;color: gray; font-size: 12px;">
        #         本页面使用
        #         <a href="https://streamlit.io/" target="_blank">Streamlit</a>
        #         与
        #         <a href="https://github.com/hiyouga/LLaMA-Factory" target="_blank">Llamafactory</a>
        #         构建。
        #     </div>
        # ''')

    
    # @st.fragment(run_every=state["run_every"])
    # def show_train_state():
    #     state = st.session_state
    #     trainer = state.get(TRAINER, None)
        
    #     if trainer is not None:
    #         st.info("模型微调正在运行中...", icon=":material/info:")
    #         return_dict = next(trainer)
    #         new_plot = return_dict.get("loss_viewer", None)
    #         new_log = return_dict.get("output", "")
    #         state["cached_plot"] = state["cached_plot"] if new_plot == None else new_plot
    #         state["cached_log"] = state["cached_log"] if new_log == "" else new_log
            
    #         if return_dict.get("end", False):
    #             state[TRAINER] = None
    #             st.rerun(scope="app")
                
    #         with st.expander("模型微调日志", expanded=True, icon=":material/monitoring:"):
    #             if return_dict.get("progress", None) != None:
    #                 label = return_dict["progress"][0]
    #                 precentage = return_dict["progress"][1]
    #                 st.progress(precentage / 100, label)
                    
    #             if state["cached_plot"] != None:
    #                 st.pyplot(state["cached_plot"])
                
    #             with st.container(height=500):
    #                 st.text(state["cached_log"])
        
        
    # if state.get("trainer", None) is not None:
    #     show_train_state()
    # else:
    #     if state["cached_plot"] != None:
    #         st.success("模型微调完成", icon=":material/check:")
    #     else: 
    #         st.info("空闲", icon=":material/info:")
    #     with st.expander("模型微调日志", expanded=True, icon=":material/monitoring:"):            
    #         with st.container(height=500):
    #             st.text(state["cached_log"])