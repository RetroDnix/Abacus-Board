import streamlit as st
from src.utils.filesystem import getCKPTS

def getIndex(options, value):
    if value in options:
        return options.index(value)
    else:
        return None

def top_page(key="ckpt_parm"):
    # col_ckpt_path, col_ckpt = st.columns([3,7])
    
    if key not in st.session_state:
        st.session_state[key] = {"ckpt_path": "./saves", "ckpt": "", "ckpt_options": []}
    ckpt_parm = st.session_state[key]
    
    # with col_ckpt_path:
    ckpt_parm["ckpt_path"] = st.text_input("检查点储存路径", "./saves", key="ckpt_path")
    ckpt_parm["ckpt_options"], message = getCKPTS(ckpt_parm["ckpt_path"])
    
    # with col_ckpt:
    ckpt_parm["ckpt"] = st.selectbox(
        label="选择模型检查点",
        options=ckpt_parm["ckpt_options"],
        index=getIndex(ckpt_parm["ckpt_options"], ckpt_parm["ckpt"]),
        placeholder="未选择",
        key="ckpt",
    )
    
    if message != "":
        st.error(message, icon=":material/warning:")
    return ckpt_parm["ckpt_path"], ckpt_parm["ckpt"]