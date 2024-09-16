from typing import Dict, Any, Union
import os

def validate_args(train_args: Dict[str, Any]) -> str:
    if(train_args["model_name_or_path"] == None):
        return "请选择模型检查点"
    if(train_args["dataset"] == ""):
        return "请选择数据集"
    if(train_args["output_dir"] == ""):
        return "请选择输出路径"
    return ""
    

def clean_cmd(args) -> Dict[str, Any]:
    no_skip_keys = ["packing"]
    return {k: v for k, v in args.items() if (k in no_skip_keys) or (v is not None and v is not False and v != "")}

def gen_cmd(train_args, freeze_args, lora_args) -> str:
    if train_args["finetuning_type"] == "LoRA":
        train_args.update(lora_args)
    elif train_args["finetuning_type"] == "freeze":
        train_args.update(freeze_args)
    
    cmd_lines = ["llamafactory-cli train "]
    for k, v in clean_cmd(train_args).items():
        cmd_lines.append("    --{} {} ".format(k, str(v)))

    if os.name == "nt":
        cmd_text = "`\n".join(cmd_lines)
    else:
        cmd_text = "\\\n".join(cmd_lines)
    
    return cmd_text