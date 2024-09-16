import os,json
from typing import List, Tuple

def getCKPTS(ckpt_path:str)->List:
    if os.path.exists(ckpt_path):
        return sorted([dir for dir in os.listdir(ckpt_path) if os.path.isdir(os.path.join(ckpt_path, dir))]),""
    else: return [],"检查点路径不存在"

def getDS(ds_path:str)->Tuple[List,str]:
    ds_path = os.path.join(ds_path,"dataset_info.json")
    if os.path.exists(ds_path):
        return json.load(open(ds_path)).keys(),""
    else: 
        print(f"未找到dataset_info.json文件")
        return [],"未找到dataset_info.json文件"
