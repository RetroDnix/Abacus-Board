import streamlit as st
from src.utils.filesystem import getDS
from src.utils.gen_cmd import gen_cmd,validate_args
from src.components.top import top_page
from src.runner import Runner

from datetime import datetime
from signal import SIGTERM
import os

def finetune_page():
    state = st.session_state
    
    if "train_args" not in state:
        state["train_args"] = {
            "finetuning_type":"full",
            "stage":"sft",
            "dataset_dir": "./data",
            "dataset":"",
            "cutoff_len":1024,
            "max_samples":100000,
            "learning_rate":5e-5,
            "per_device_train_batch_size":4,
            "num_train_epochs":3.0,
            "max_grad_norm":1.0,
            "gradient_accumulation_steps":4,
            "lr_scheduler_type":"linear",
            "logging_steps":50,
            "save_steps":500,
            "warmup_ratio":0.00,
            "val_size":0.05,
            "per_device_eval_batch_size":4,
            "eval_steps":100,
            "do_train":True,
            "template":"default",
            "flash_attn":"auto",
            "optim":"adamw_torch",
            "packing":False,
            "plot_loss":True,
            "include_num_input_tokens_seen":True,
            "preprocessing_num_workers":8,
            "eval_on_start":True
        }
    
    if "lora_args" not in state:
        state["lora_args"] = {
            "lora_rank":8,
            "lora_alpha":16,
            "lora_dropout":0.0,
            "use_rslora":False,
            "use_dora":False,
            "pissa_convert":False,
            "pissa_init":False,
            "create_new_adapter":False
        }
    
    if "freeze_args" not in state:
        state["freeze_args"] = {
            "freeze_trainable_layers":4
        }

    train_args = state["train_args"]
    lora_args = state["lora_args"] 
    freeze_args = state["freeze_args"]
    
    if "trainer" not in state:
        state["trainer"] = None
    
    if "run_every" not in state:
        state["run_every"] = 2
    
    if "cached_plot" not in state:
        state["cached_plot"] = None
    
    if "cached_log" not in state:
        state["cached_log"] = ""
        
    if "cuda_visible_devices" not in state:
        state["cuda_visible_devices"] = ""

    st.markdown("##### 微调方法")
    st.caption("选择微调时使用的训练方式与微调方法")
    col_finetuning_type, col_stage = st.columns([3, 7])
    with col_finetuning_type:
        finetune_methods = ["full", "freeze", "LoRA"]
        train_args["finetuning_type"] = st.selectbox(
            label = "训练方式", 
            options = finetune_methods,
            index = finetune_methods.index(train_args["finetuning_type"])
        )
    with col_stage:
        Stage = {"Supervised Finetune": "sft", "Pretrain": "pt"}
        stage_index = {"sft":0,"pt":1}
        train_args["stage"] = Stage[st.selectbox(
            label = "微调阶段", 
            options = ["Supervised Finetune", "Pretrain"],
            index = stage_index[train_args["stage"]],
        )]
    st.divider()

    st.markdown("##### 数据集")
    st.caption("选择微调时使用的数据集, 目前支持Alpaca与ShareGPT格式的数据集")
    col_ds_path, col_ds = st.columns([3, 7])
    with col_ds_path:
        train_args["dataset_dir"] = st.text_input("数据集储存路径", train_args["dataset_dir"], key="ds_path")
    with col_ds:
        all_datasets, message = getDS(train_args.get("dataset_dir", "./data"))
        # default = [s for s in train_args["dataset"].split(",") if s != ""]
        train_args["dataset"] = ",".join(st.multiselect(
            label="选择数据集",
            options=all_datasets,
            # default=default,
            placeholder="未选择",
            key="ds",
        ))
    if message != "":
        st.error(message, icon=":material/warning:")

    col_cutoff_len, col_max_samples = st.columns(2)
    with col_cutoff_len:
        train_args["cutoff_len"] = st.number_input("截断长度", 0, 4096, train_args["cutoff_len"])
    with col_max_samples:
        train_args["max_samples"] = st.number_input("最大样本数", 0, 1000000, train_args["max_samples"])

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
            "%s_%s_%s"
            % (
                train_args.get("finetuning_type", "UnknownFtType"),
                train_args.get("stage", "UnknownStage"),
                datetime.now().strftime("%y%m%d-%H%M"),
            ),
            key="output_dir",
        )
        train_args["output_dir"] = os.path.join(output_path, output_name)
    st.divider()

    st.markdown(
        "##### 训练设置",
    )
    col_learning_rate, col_batch_size, col_num_train_epochs = st.columns(3)
    with col_learning_rate:
        train_args["learning_rate"] = st.number_input(
            "学习率", 0.0, 1.0, train_args["learning_rate"], step=1e-5, format="%.5f"
        )
    with col_batch_size:
        train_args["per_device_train_batch_size"] = st.number_input(
            "每个设备的批次大小(训练)", 0, 16, train_args["per_device_train_batch_size"]
        )
    with col_num_train_epochs:
        train_args["num_train_epochs"] = st.number_input(
            "训练轮数", 0.0, 100.0, train_args["num_train_epochs"], step=0.1, format="%.1f"
        )

    col_max_grad_norm, col_grad_accu, col_lr_scheduler = st.columns(3)
    with col_max_grad_norm:
        train_args["max_grad_norm"] = st.number_input(
            "最大梯度范数", 0.0, 10.0, train_args["max_grad_norm"], step=0.1, format="%.1f"
        )
    with col_grad_accu:
        train_args["gradient_accumulation_steps"] = st.number_input(
            "梯度累积", 1, 1024, train_args["gradient_accumulation_steps"]
        )
    with col_lr_scheduler:
        schedulers = ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant"]
        train_args["lr_scheduler_type"] = st.selectbox(
            label="学习率调度器",
            options=schedulers,
            index=schedulers.index(train_args["lr_scheduler_type"])
        )

    col_logging_steps, col_save_steps, col_precision = st.columns(
        3, vertical_alignment="center"
    )
    with col_logging_steps:
        train_args["logging_steps"] = st.number_input("日志步数", 0, 1000, train_args["logging_steps"])
    with col_save_steps:
        train_args["save_steps"] = st.number_input("保存步数", 0, 1000, train_args["save_steps"])
    with col_precision:
        precisions = ["bf16", "fp16", "fp32", "pure_bp16"]
        state["_precison"] = st.selectbox(
            label="精度", 
            options=precisions, 
            index=precisions.index(state.get("_precison", "bf16")), 
        )
        precision = state["_precison"]
        if precision != "fp32":
            train_args[precision] = True

    train_args["warmup_ratio"] = st.slider(
        "预热步数比例", 0.00, 1.00, train_args["warmup_ratio"], step=0.01, format="%.2f"
    )

    with st.expander("lora训练设置"):
        col_lora_rank, col_lora_alpha, col_lora_dropout = st.columns(3)
        with col_lora_rank:
            lora_args["lora_rank"] = st.number_input("LoRA 矩阵的秩大小", 0, 1024, lora_args["lora_rank"], key="lora_rank")
        with col_lora_alpha:
            lora_args["lora_alpha"] = st.number_input("LoRA 缩放系数大小", 0, 2048, lora_args["lora_alpha"], step=1, key="lora_alpha")
        with col_lora_dropout:
            lora_args["lora_dropout"] = st.slider("LoRA 权重随机丢弃的概率", 0.0, 1.0, lora_args["lora_dropout"], step=0.01, format="%.2f", key="lora_dropout")
        col_use_rslora, col_use_dora, col_use_pissa, col_new_adpter = st.columns(4)
        with col_use_rslora:
            lora_args["use_rslora"] = st.toggle("使用RS-LoRA", value=lora_args["use_rslora"], key="use_rslora")
        with col_use_dora:
            lora_args["use_dora"] = st.toggle("使用D-LoRA", value=lora_args["use_dora"], key="use_dora")
        with col_use_pissa:
            lora_args["pissa_convert"] = lora_args["pissa_init"] = st.toggle("使用PiSSA", value=lora_args["pissa_convert"], key="use_pissa")
        with col_new_adpter:
            lora_args["create_new_adapter"] = st.toggle("新建Lora适配器", value=lora_args["create_new_adapter"], key="new_adapter")
    
    with st.expander("部分参数训练训练设置"):
        st.caption("最末尾（+）/最前端（-）可训练隐藏层的数量。")
        freeze_args["freeze_trainable_layers"] = st.slider("可训练层数", -128, 128, freeze_args["freeze_trainable_layers"], step=1, label_visibility="hidden", key="freeze_trainable_layers")
    
    st.divider()

    st.markdown(
        "##### 评估设置",
    )
    col_do_eval, col_val_size = st.columns([1,2])
    with col_do_eval:
        train_args["do_eval"] = st.toggle("使用验证", train_args["do_eval"])
    with col_val_size:
        train_args["val_size"] = st.slider("验证集比例", 0.00, 1.00, train_args["val_size"])
    col_eval_batch_size, col_eval_steps = st.columns(2)
    with col_eval_batch_size:
        train_args["per_device_eval_batch_size"] = st.number_input(
            "每个设备的批次大小(验证)", 0, 16, train_args["per_device_eval_batch_size"]
        )
    with col_eval_steps:
        train_args["eval_steps"] = st.number_input("验证步数", 0, 1000, train_args["eval_steps"])
    
    st.divider()
    
    with st.sidebar:
        ckpt_path, ckpt = top_page("finetune_ckpt_parm")
        train_args["model_name_or_path"] = None if ckpt == None else os.path.join(ckpt_path, ckpt)
        
        st.markdown("##### 资源分配")
        state["finetune_cuda_visible_devices"] = st.text_input("CUDA_VISIBLE_DEVICES", value=state.get("finetune_cuda_visible_devices",""))
        train_args["preprocessing_num_workers"] = st.number_input("预处理工作线程数", 0, 128, train_args["preprocessing_num_workers"])
        
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
                msg = validate_args(train_args)
                if msg != "":
                    st.error(msg, icon=":material/warning:")
        
        if start_training:
            if state.get("trainer", None) is not None:
                st.error("请等待当前训练完成再继续操作。", icon=":material/warning:")
            elif validate_args(train_args) != "":
                st.error(validate_args(train_args), icon=":material/warning:")
            else:
                runner = Runner()
                trainer = runner.launch(train_args, freeze_args, lora_args, state["finetune_cuda_visible_devices"])
                if trainer != None:
                    state["runner"] = runner
                    state["trainer"] = trainer
                    state["cached_log"] = ""
                    state["cached_plot"] = None
                st.toast("开始模型微调", icon=":material/info:")
                print("开始模型微调")
                st.rerun(scope="app")
        
        if state.get("trainer", None) is not None:
            trainer = state["trainer"]
            runner = state.get("runner", None)
            if st.button("停止微调", key="stop", use_container_width=True, type="primary"):
                state["trainer"] = None
                if runner is not None:
                    runner.terminate()
                state["runner"] = None
                st.rerun(scope="app")
        
        st.html(body = '''    
            <div style="text-align: center;color: gray; font-size: 12px;">
                本页面使用
                <a href="https://streamlit.io/" target="_blank">Streamlit</a>
                与
                <a href="https://github.com/hiyouga/LLaMA-Factory" target="_blank">Llamafactory</a>
                构建。
            </div>
        ''')
    
    if show_cmd:
        if state["trainer"] == None and validate_args(train_args) == "":
            st.markdown("```bash\n{}\n```".format(gen_cmd(train_args,freeze_args,lora_args)))
    
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
                
            with st.expander("Llamafactory日志", expanded=True, icon=":material/monitoring:"):
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
        with st.expander("OpenCompass日志", expanded=True, icon=":material/monitoring:"):
            if state["cached_plot"] != None:
                st.pyplot(state["cached_plot"])
            
            with st.container(height=500):
                st.text(state["cached_log"])