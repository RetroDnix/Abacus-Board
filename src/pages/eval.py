import streamlit as st
from subprocess import Popen
from src.eval_runner import start_eval
from src.components.top import top_page
from src.widgets.stated_widgets import number_input, text_input
import os
from glob import glob
import random
import pandas as pd
from copy import deepcopy
from signal import SIGTERM

def eval_page():
    random.seed()
    state = st.session_state
    
    if "opc_instance" not in state:
        state["opc_instance"] = None
    
    if "opc_log" not in state:
        state["opc_log"] = ""
    
    if "eval_result" not in state:
        state["eval_result"] = None
    
    if "eval_args" not in state:
        state["eval_args"] = {
            "abbr": "ModelEvaluate",
            "real_abbr": "",
            "dataset": [],
            "max_out_len": 2048,
            "max_seq_len": 4096,
            "batch_size": 16,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 2048,
            "num_gpus": 1,
            "num_procs": 8,
            "cuda_visible_devices":"0"
        }
        
    eval_args = state["eval_args"]
    
    all_datasets = ["HumanEval", "HumanEval+", "MBPP[sanitized]","MBPP+", "MMLU", "HellaSwag", "ARC-e", "BBH", "C-Eval", "CMMLU", "GSM8K"]
    
    text_input("任务名称", data = eval_args, key="abbr")
    def save_dataset():
        eval_args["dataset"] = state["ds_eval"]
    st.multiselect(
        label="选择数据集",
        options=all_datasets,
        placeholder="未选择",
        key="ds_eval",
        on_change=save_dataset
    )              
    
    st.divider()
    
    st.markdown("###### 评测参数")
    
    col_max_out_len, col_max_seq_len, col_batch_size = st.columns(3)
    with col_max_out_len:
        number_input("最大输出长度", 0, 4096, data=eval_args, key="max_out_len", prefix="_eval_")
    with col_max_seq_len:
        number_input("最大序列长度", 0, 4096, data=eval_args, key="max_seq_len", prefix="_eval_")
    with col_batch_size:
        number_input("批次大小", 0, 128, data=eval_args, key="batch_size", prefix="_eval_")
    
    col_temperature, col_top_p, col_max_tokens = st.columns(3)
    with col_temperature:
        number_input("Temperature", 0.0, 1.0, data=eval_args, key="temperature", prefix="_eval_")
    with col_top_p:
        number_input("Top-p", 0.0, 1.0, data=eval_args, key="top_p", prefix="_eval_")
    with col_max_tokens:
        number_input("Max tokens", 0, 4096, data=eval_args, key="max_tokens", prefix="_eval_")
    
    st.divider()
    
    @st.fragment(run_every=2)
    def update_log():
        work_dir = "./opencompass/outputs/%s"%eval_args["real_abbr"]
        if state.get("opc_instance", None) is not None:
            instance = state["opc_instance"]
            if instance.poll() == None:
                st.info("模型测评正在运行中...", icon=":material/info:")
                files = glob(work_dir + "/**/*.out", recursive=True)
                if len(files) > 0:
                    with open(files[0],"r",encoding="utf-8") as log:
                        state["opc_log"] = log.read()
            else: 
                state["opc_instance"] = None
                state["opc_log"] += "已结束\n"
                files = glob(work_dir + "/**/*.csv", recursive=True)
                if len(files) > 0:
                    state["eval_result"] = pd.read_csv(files[0])
                st.rerun(scope="app")
                    
        with st.expander("OpenCompass日志", expanded=True, icon=":material/monitoring:"):
            with st.container(height=500):
                st.text(state["opc_log"])
        
        if state.get("eval_result", None) is not None:
            st.markdown("###### 评测结果")
            st.table(state["eval_result"])
    
    if state.get("opc_instance", None) is not None:
        update_log()
    else:
        if state.get("eval_result", None) is not None:
            st.success("模型评测完成", icon=":material/check:")
            # print("模型评测完成")
        else: st.info("空闲", icon=":material/info:")
        
        with st.expander("OpenCompass日志", expanded=True, icon=":material/monitoring:"):
            with st.container(height=500):
                st.text(state["opc_log"])
        
        if state.get("eval_result", None) is not None:
            st.markdown("###### 评测结果")
            st.table(state["eval_result"])
    
    with st.sidebar:
        ckpt_path, ckpt = top_page("eval_ckpt_parm")
        
        st.markdown("##### 资源分配")
        text_input("CUDA_VISIBLE_DEVICES", data=eval_args, key="cuda_visible_devices", prefix="_eval_")
        col_num_gpus, col_num_procs = st.columns(2)
        with col_num_gpus:
            number_input("GPU数量", 0, 8, data=eval_args, key="num_gpus", prefix="_eval_")
        
        with col_num_procs:
            number_input("CPU核心数量", 0, 128, data=eval_args, key="num_procs", prefix="_eval_")
        
        st.divider()
        
        col_start_evaluate, col_stop_evaluate = st.columns(2)
        with col_start_evaluate:
            start_evaluate = st.button("开始评测", use_container_width=True)
        with col_stop_evaluate:
            stop_evaluate = st.button("停止评测", use_container_width=True, type="primary")
        
        if start_evaluate:
            if eval_args["abbr"] == "":
                st.error("任务名称不能为空", icon=":material/warning:")
            elif len(eval_args["dataset"]) == 0:
                st.error("请选择至少一个数据集", icon=":material/warning:")
            elif ckpt == None or ckpt == "":
                st.error("请输入模型检查点", icon=":material/warning:")    
            elif not os.path.exists(os.path.join(ckpt_path, ckpt)):
                st.error("检查点路径不存在", icon=":material/warning:")
            elif state["opc_instance"] is not None:
                st.error("请等待当前评测任务结束", icon=":material/warning:")
            else:
                state["eval_result"] = None
                state["opc_log"] = ""
                eval_args["real_abbr"] = eval_args["abbr"] + '_' + str(random.randint(1, 1000000))
                eval_args["ckpt_full_path"] = os.path.join(ckpt_path, ckpt)
                cmd = start_eval(eval_args)
                env = deepcopy(os.environ)
                env["CUDA_VISIBLE_DEVICES"] = eval_args["cuda_visible_devices"]
                state["opc_instance"] = Popen(cmd, shell=True, env=env, preexec_fn=os.setsid)
                st.toast("开始模型评估", icon=":material/info:")
                print("开始模型评估")
                st.rerun(scope="app")
        
        if stop_evaluate:
            instance = state.get("opc_instance", None)
            if instance != None:
                instance.terminate()
                instance.wait()
                os.killpg(instance.pid, SIGTERM)
                state["opc_log"] += "terminated\n"
                state["opc_instance"] = None
                st.toast("评测任务已终止", icon=":material/info:")
                print("评测任务已终止")
                st.rerun(scope="app")
            else:
                st.toast("没有正在进行的评测任务", icon=":material/error:")

        st.html(body = '''    
            <div style="text-align: center;color: gray; font-size: 12px;">
                本页面使用
                <a href="https://streamlit.io/" target="_blank">Streamlit</a>
                与
                <a href="https://github.com/open-compass/opencompass" target="_blank">OpenCompass</a>
                构建。
            </div>
        ''')