from openai import OpenAI
import streamlit as st
from subprocess import Popen, PIPE, STDOUT
from src.components.top import top_page
from src.widgets.stated_widgets import number_input, text_input, slider, toggle, selectbox
import os
from copy import deepcopy
import select  
from signal import SIGTERM

def infer_page():
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"
    
    with st.sidebar:
        state = st.session_state
        
        ckpt_path, ckpt = top_page("infer_ckpt_parm")
        
        if "infer_args" not in state:
            state["infer_args"] = {
                "max_tokens": 2048, 
                "temperature": 0.3, 
                "top_p": 1.0,
                "tensor_parallel_size": 1,
                "gpu_memory_utilization": 0.9,
                "cuda_visible_devices":"0"
            }
        infer_args = state["infer_args"]
        
        if "vllm_instance" not in state:
            state["vllm_instance"] = None
        
        if "vllm_log" not in state:
            state["vllm_log"] = ""
        
        if "openai_model" not in state:
            state["openai_model"] = "vllm-model"

        if "messages" not in state:
            state.messages = [
                # {"role": "system", "content": system_prompt}
            ]
            
        if "client" not in state:
            state.client = None
        
        st.markdown("##### 推理参数")
        col_max_tokens, col_temperature, col_top_p = st.columns(3)
        with col_max_tokens:
            number_input("最大标记数", min_value=1, max_value=4096, data=infer_args, key="max_tokens", prefix="_infer_")
        with col_temperature:
            number_input("温度", min_value=0.0, max_value=1.0, data=infer_args, key="temperature", prefix="_infer_")
        with col_top_p:
            number_input("Top-p", min_value=0.0, max_value=1.0, data=infer_args, key="top_p", prefix="_infer_")
        
        state["_infer_system_prompt"] = infer_args.get("infer_system_prompt", "")
        def save_sys_prompt():
            infer_args["infer_system_prompt"] = state["_infer_system_prompt"]
        st.text_area(label="System Prompt", key="_infer_system_prompt", on_change=save_sys_prompt)
        
        st.markdown("##### 资源分配")
        text_input("CUDA_VISIBLE_DEVICES", data=infer_args, key="cuda_visible_devices", prefix="_infer_")
        col_tensor_parallel_size, col_gpu_memory_utilization = st.columns(2)
        with col_tensor_parallel_size:
            number_input("Tensor并行大小", min_value=1, max_value=8, data=infer_args, key="tensor_parallel_size", prefix="_infer_")
        with col_gpu_memory_utilization:
            number_input("GPU内存利用率目标", min_value=0.0, max_value=1.0, data=infer_args, key="gpu_memory_utilization", prefix="_infer_")
        
        st.divider()
        
        col_load, col_unload = st.columns(2)
        with col_load:
            load_model = st.button("加载模型",key="load",use_container_width=True)
        with col_unload:
            unload_model = st.button("卸载模型",key="unload",use_container_width=True)
    
        if st.button("清空对话历史", key="clear", use_container_width=True, type="primary"):
            state.messages = [
                # {"role": "system", "content": system_prompt}
            ]
    
        if load_model:
            if state.get("vllm_instance", None) is None:
                if ckpt == None or not os.path.exists(os.path.join(ckpt_path, ckpt)):
                    st.error("检查点路径不存在", icon=":material/error:")
                else:
                    env = deepcopy(os.environ)
                    env["CUDA_VISIBLE_DEVICES"] = infer_args["cuda_visible_devices"]
                    state["infer_ckpt_full_path"] = os.path.join(ckpt_path, ckpt)
                    cmd = "vllm serve --trust-remote-code %s --gpu-memory-utilization %s --tensor-parallel-size %s" % (
                        state["infer_ckpt_full_path"],
                        infer_args["gpu_memory_utilization"],
                        infer_args["tensor_parallel_size"],
                    )
                    state["vllm_instance"] = Popen(cmd, stdout=PIPE, env=env, shell=True, preexec_fn=os.setsid)
                    state["vllm_log"] = ""
                    st.toast("开始加载模型", icon=":material/info:")
                    print("开始加载模型")
            else: 
                st.error("请先卸载当前模型", icon=":material/error:")
        
        if unload_model:
            if state.get("vllm_instance", None) is not None:
                instance = state.get("vllm_instance")
                if instance != None:
                    instance.terminate()
                    instance.wait()
                    # os.killpg(instance.pid, SIGTERM) 
                    state["vllm_log"] += "terminated\n"
                    state["vllm_instance"] = None
                state["client"] = None
                st.toast("模型已卸载",icon=":material/info:")
                print("模型已卸载")
                st.rerun(scope="app")
        
        st.html(body = '''    
            <div style="text-align: center;color: gray; font-size: 12px;">
                本页面使用
                <a href="https://streamlit.io/" target="_blank">Streamlit</a>
                与
                <a href="https://github.com/vllm-project/vllm" target="_blank">VLLM</a>
                构建。
            </div>
        ''')

    @st.fragment(run_every=2)
    def update_log():
        if state["vllm_instance"] is not None:
            instance = state["vllm_instance"]
            if instance.poll() == None:
                output = instance.stdout
                readable, _, _ = select.select([output], [], [], 0.1)
                while output in readable:
                    line = output.readline().decode('utf-8')
                    # 加载完成
                    if line.find("Application startup complete.") != -1:
                        state.client = OpenAI(
                            api_key=openai_api_key,
                            base_url=openai_api_base,
                        )
                        print("模型已加载")
                        st.rerun(scope='app')
                    state["vllm_log"] += line
                    readable, _, _ = select.select([output], [], [], 0.1)
            else: 
                state["vllm_instance"] = None
                state["vllm_log"] += "terminated\n"
                state["client"] = None
                st.toast("模型异常终止", icon=":material/error:")
                print("模型异常终止")
        
        with st.expander("VLLM日志", expanded=True, icon=":material/monitoring:"):
            with st.container(height=250):
                st.text(state["vllm_log"])
    
    update_log()

    if state.client != None:
        st.success("模型已加载", icon=":material/check:")
        client = state.client
        state = st.session_state
        for message in state.messages:
            if message["role"] == "system":
                continue
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        if prompt := st.chat_input("有什么问题都可以问我~"):
            state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                sysprompt = infer_args.get("infer_system_prompt","")
                cur_messages=[ {"role": m["role"], "content": m["content"]} for m in state.messages ]
                if sysprompt != "":
                    cur_messages = [{"role":"system", "content": sysprompt}] + cur_messages
                stream = client.chat.completions.create(
                    model=state["infer_ckpt_full_path"],
                    messages = cur_messages,
                    stream=True,
                    temperature=infer_args["temperature"],
                    max_tokens=infer_args["max_tokens"],
                    top_p=infer_args["top_p"],
                )
                response = st.write_stream(stream)
            st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.info("模型未加载，当模型载入后，会在此处显示聊天窗口", icon=":material/info:")
    