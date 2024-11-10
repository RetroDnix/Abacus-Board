import streamlit as st
from src.utils.filesystem import getDS
from src.components.top import top_page
from src.widgets.stated_widgets import number_input, text_input, slider, toggle, selectbox
from src.runner_9g import gen_cmd_9g, validate_args_9g

from datetime import datetime
from signal import SIGTERM
import os
from copy import deepcopy
from subprocess import Popen

def finetune_9g():
    state = st.session_state
    
    if "train_args_9g" not in state:
        state["train_args_9g"] = {
            #训练的名称，模型和log等信息会存储在该文件夹中
            "model_unique":"Finetune_9g_%s"%datetime.now().strftime("%y%m%d-%H%M"),
            #config位置，在configs/目录中
            "config":"thisConfig",
            #训练batch size
            "batch_size":1,
            "dataset_dir":"./data-example",
            "dataset":"",
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
            "lr_scheduler":"Cosine",
            #drop的比例
            "drop_iters":0.1,
            #drop比例
            "drop_begin":-1,
            "drop_rate":0.5,
            #是use checkpoint，建议使用
            "use_checkpoint":"0",
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
        
    # if "finetune_cuda_visible_devices" not in state:
    #     state["finetune_cuda_visible_devices"] = "0"

    st.markdown("##### 训练名称")
    model_unique = text_input(
        "本次训练的唯一标识,训练的名称，模型和log等信息会存储在该文件夹中",
        train_args,
        key="model_unique",
        prefix="_finetune_9g_"
    )
    st.divider()

    st.markdown("##### 数据集")
    st.caption("选择微调时使用的数据集, 目前支持Alpaca与ShareGPT格式的数据集")
    col_ds_path, col_ds = st.columns([3, 7])
    with col_ds_path:
        text_input("数据集储存路径", train_args, key="dataset_dir")
    with col_ds:
        all_datasets, message = getDS(train_args.get("dataset_dir", "./data-example"))
        state["_dataset"] = [s for s in train_args["dataset"].split(",") if s != ""]
        def save_dataset():
            train_args["dataset"] = ",".join(state["_dataset"])
        st.multiselect(
            label="选择数据集",
            options=all_datasets,
            placeholder="未选择",
            key="_dataset",
            on_change=save_dataset,
        )
    if message != "":
        st.error(message, icon=":material/warning:")

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
        schedulers = ["Cosine"]
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
        ckpt_path, ckpt = top_page("finetune_ckpt_parm")
        train_args["model_name_or_path"] = None if ckpt == None else os.path.join(ckpt_path, ckpt)
        
        st.markdown("##### 资源分配")
        state["_finetune_9g_cuda_visible_devices"] = state.get("finetune_9g_cuda_visible_devices", "0")
        def save_cuda():
            state["finetune_9g_cuda_visible_devices"] = state["_finetune_9g_cuda_visible_devices"]
        st.text_input(
            "CUDA_VISIBLE_DEVICES", 
            key="_finetune_9g_cuda_visible_devices",
            on_change= save_cuda
        )
        number_input("预处理工作线程数", 0, 128, data=train_args, key="dataloader_num_workers")
        
        st.divider()
        
        col_show_instruction, col_start_training = st.columns(2)
        with col_show_instruction:
            show_cmd = st.button("预览命令", use_container_width=True)
        
        with col_start_training:
            start_training = st.button("开始微调", use_container_width=True)
        
        if show_cmd:
            if state.get("trainer", None) is not None:
                st.error("请等待当前训练完成再继续操作。", icon=":material/warning:")
            else:
                msg = validate_args_9g(train_args)
                if msg != "":
                    print(msg)
                    st.error(msg, icon=":material/warning:")
        
        if start_training:
            if state.get("trainer", None) is not None:
                st.error("请等待当前训练完成再继续操作。", icon=":material/warning:")
            elif validate_args_9g(train_args) != "":
                st.error(validate_args_9g(train_args), icon=":material/warning:")
            else:
                env = deepcopy(os.environ)
                env["CUDA_VISIBLE_DEVICES"] = state["finetune_9g_cuda_visible_devices"]
                trainer = Popen(gen_cmd_9g(train_args), env=env, shell=True, preexec_fn=os.setsid)
                st.toast("开始模型微调", icon=":material/info:")
                print("开始模型微调")
                st.rerun(scope="app")
        
        # if state.get("trainer", None) is not None:
        #     trainer = state["trainer"]
        #     runner = state.get("runner", None)
        #     if st.button("停止微调", key="stop", use_container_width=True, type="primary"):
        #         state["trainer"] = None
        #         if runner is not None:
        #             abort_process(runner.pid)
        #             torch_gc()
        #         state["runner"] = None
        #         st.rerun(scope="app")
        
        # st.html(body = '''    
        #     <div style="text-align: center;color: gray; font-size: 12px;">
        #         本页面使用
        #         <a href="https://streamlit.io/" target="_blank">Streamlit</a>
        #         与
        #         <a href="https://github.com/hiyouga/LLaMA-Factory" target="_blank">Llamafactory</a>
        #         构建。
        #     </div>
        # ''')
    
    if show_cmd:
        if state["trainer"] == None and validate_args_9g(train_args) == "":
            print("ww")
            st.markdown("```bash\n{}\n```".format(gen_cmd_9g(train_args)))
    
    @st.fragment(run_every=state["run_every"])
    def show_train_state():
        state = st.session_state
        trainer = state.get("trainer", None)
        
        if trainer is not None:
            st.info("模型微调正在运行中...", icon=":material/info:")
            return_dict = next(trainer)
            new_plot = return_dict.get("loss_viewer", None)
            new_log = return_dict.get("output", "")
            state["cached_plot"] = state["cached_plot"] if new_plot == None else new_plot
            state["cached_log"] = state["cached_log"] if new_log == "" else new_log
            
            if return_dict.get("end", False):
                state["runner"] = None
                state["trainer"] = None
                st.rerun(scope="app")
                
            with st.expander("模型微调日志", expanded=True, icon=":material/monitoring:"):
                if return_dict.get("progress", None) != None:
                    label = return_dict["progress"][0]
                    precentage = return_dict["progress"][1]
                    st.progress(precentage / 100, label)
                    
                if state["cached_plot"] != None:
                    st.pyplot(state["cached_plot"])
                
                with st.container(height=500):
                    st.text(state["cached_log"])
        
        
    if state.get("trainer", None) is not None:
        show_train_state()
    else:
        if state["cached_plot"] != None:
            st.success("模型微调完成", icon=":material/check:")
        else: 
            st.info("空闲", icon=":material/info:")
        with st.expander("模型微调日志", expanded=True, icon=":material/monitoring:"):
            if state["cached_plot"] != None:
                st.pyplot(state["cached_plot"])
            
            with st.container(height=500):
                st.text(state["cached_log"])