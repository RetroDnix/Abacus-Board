import streamlit as st

from src.pages.finetune import finetune_page
from src.pages.eval import eval_page
from src.pages.infer import infer_page

p_finetune = st.Page(finetune_page, title="模型微调", icon=":material/cookie:")
p_eval = st.Page(eval_page, title="模型评估", icon=":material/sticky_note_2:")
p_infer = st.Page(infer_page, title="模型推理", icon=":material/analytics:")

st.set_page_config(
    page_title='''"珠算"大模型微调适配平台''',
    page_icon="🧊",
    layout="centered",    # 'wide' or 'centered'
    menu_items={
        'About': 'http://ir.hit.edu.cn/',
    }
)
pg = st.navigation([p_finetune, p_eval, p_infer])
pg.run()