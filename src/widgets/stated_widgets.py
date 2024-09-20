import streamlit as st
from typing import Dict, Union
    
def load_value(key:str, ds:Dict, prefix:str):
    st.session_state[prefix + key] = ds[key]
def store_value(key:str, ds:Dict, prefix:str):
    ds[key] = st.session_state[prefix + key]
    

def number_input(
    label: str,
    min_value: Union[int,float],
    max_value: Union[int,float],
    data: Dict,
    key: str,
    prefix:str = "_",
    step=None,
    format=None,
    help="",
    placeholder="None",
    disabled=False,
    label_visibility="visible"
):
    load_value(key, data, prefix)
    st.number_input(
        label=label, 
        min_value=min_value, 
        max_value=max_value, 
        step=step, 
        format=format,
        key=prefix + key,
        help=help,
        placeholder=placeholder,
        disabled=disabled,
        label_visibility=label_visibility,
        on_change=store_value,
        args=[key,data, prefix]
    )

def text_input(
    label: str,
    data: Dict,
    key: str,
    prefix:str = "_",
    help="",
    placeholder="",
    disabled=False,
    label_visibility="visible"
):
    load_value(key, data, prefix)
    st.text_input(
        label=label,
        key=prefix + key,
        help=help,
        placeholder=placeholder,
        disabled=disabled,
        label_visibility=label_visibility,
        on_change=store_value,
        args=[key,data, prefix]
    )

def slider(
    label: str,
    min_value: Union[int,float],
    max_value: Union[int,float],
    data: Dict,
    key: str,
    prefix:str = "_",
    step=None,
    format=None,
    help="",
    disabled=False,
    label_visibility="visible"
):
    load_value(key, data, prefix)
    st.slider(
        label=label,
        min_value=min_value,
        max_value=max_value,
        step=step,
        format=format,
        key=prefix + key,
        help=help,
        disabled=disabled,
        label_visibility=label_visibility,
        on_change=store_value,
        args=[key,data, prefix]
    )

def toggle(
    label: str,
    data: Dict,
    key: str,
    prefix:str = "_",
    help="",
    disabled=False,
    label_visibility="visible"
):
    load_value(key, data, prefix)
    st.toggle(
        label=label,
        key=prefix + key,
        help=help,
        disabled=disabled,
        label_visibility=label_visibility,
        on_change=store_value,
        args=[key,data, prefix]
    )

def selectbox(
    label: str,
    options: list,
    data: Dict,
    key: str,
    prefix:str = "_",
    help="",
    placeholder="请选择",
    disabled=False,
    label_visibility="visible"
):
    load_value(key, data, prefix)
    st.selectbox(
        label=label,
        options=options,
        key=prefix + key,
        help=help,
        disabled=disabled,
        label_visibility=label_visibility,
        on_change=store_value,
        args=[key, data, prefix],
        placeholder=placeholder
    )
