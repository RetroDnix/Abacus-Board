import streamlit as st
from src.utils.filesystem import getCKPTS
from src.widgets.stated_widgets import text_input, selectbox

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
    
    text_input("检查点储存路径", data=ckpt_parm, key="ckpt_path", prefix=key)
    ckpt_parm["ckpt_options"], message = getCKPTS(ckpt_parm["ckpt_path"])
    
    # with col_ckpt:
    print(ckpt_parm["ckpt_options"])
    selectbox(
        label="选择模型检查点",
        options=ckpt_parm["ckpt_options"],
        placeholder="未选择",
        data=ckpt_parm,
        key="ckpt",
        prefix=key
    )
    
    if message != "":
        st.error(message, icon=":material/warning:")
    return ckpt_parm["ckpt_path"], ckpt_parm["ckpt"]