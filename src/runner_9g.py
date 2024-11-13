import os, json
from typing import Dict, Any, Literal
from subprocess import run, Popen
from copy import deepcopy

def validate_args_9g(train_args: Dict[str, Any]) -> str:
    if train_args["model_name_or_path"] == None:
        return "请选择模型检查点"
    if train_args["dataset"] == "":
        return "请选择数据集"
    if train_args["output_dir"] == "":
        return "请选择输出路径"
    return ""

def get_absolutely_path(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))

def train_9g(
    train_args: Dict[str, Any], model_variant: Literal["fm9g_2b", "fm9g_8b"]
) -> bool:
    WORK_DIR = f"./BMTrainer/FM_9G/apps/{model_variant}"
    NAME = train_args["model_unique"]

    train_config_dir = f"{WORK_DIR}/train_configs/{NAME}.json"
    dataset_config_dir = f"{WORK_DIR}/dataset_configs/{NAME}.json"
    origin_ds_dir = os.path.join(train_args["dataset_dir"], train_args["dataset"])
    transformed_ds_dir = f"dataset_bin/{NAME}"
    
    ds = os.listdir(origin_ds_dir)
    dsfiles = [os.path.join(origin_ds_dir, f) for f in ds if f.endswith(".jsonl")]
    if len(dsfiles) == 0: return None
    else: origin_ds_file = dsfiles[0]

    os.makedirs(f"{WORK_DIR}/train_configs", exist_ok=True)
    os.makedirs(f"{WORK_DIR}/dataset_configs", exist_ok=True)
    os.makedirs("dataset_bin", exist_ok=True)

    cmd = [
        "python",
        "BMTrainer/quick_start_clean/convert_json2index.py",
        "--path",
        origin_ds_file,
        "--language",
        train_args["dataset_language"],
        "--output",
        transformed_ds_dir
    ]
    result = run(cmd)
    if result.returncode != 0:
        return None
    
    meta = json.load(open(os.path.join(transformed_ds_dir, "meta.json")))
    dataset_config = {
        "dataset_name": train_args["dataset"],
        "task_name": train_args["dataset"],
        "abs_weight": 1.0,
        "path": get_absolutely_path(os.path.join(train_args["dataset_dir"], train_args["dataset"])),
        "transforms": "0124_hq_data/general/script_cpmc.py",
        "allow_repeat": True,
        "nlines": meta["nlines"],
        "ave_tokens_per_line": meta["avg_token_per_line"],
        "total_tokens": meta["avg_token_per_line"] * meta["nlines"] / 1e9,
    }
    json.dump(dataset_config, open(dataset_config_dir, "w"), ensure_ascii=False, indent=4)

    real_train_args = {
        "model_unique": NAME,
        "load": get_absolutely_path(train_args["model_name_or_path"]),
        "batch_size": train_args["batch_size"],
        "save": True,
        "save_iters": train_args["save_iters"],
        "save_path": get_absolutely_path(train_args["output_dir"]),
        "train_iters": train_args["train_iters"],
        "dataset_config": get_absolutely_path(dataset_config_dir),
        "dataloader_num_threads": 1,
        "dataloader_prefetch": 1,
        "dataloader_prefetch_factor": 1,
        "dataloader_num_workers": train_args["dataloader_num_workers"],
        "parallel_load_datastate": "8",
        "lr": train_args["lr"],
        "warmup_iters": train_args["warmup_iters"],
        "lr_scheduler": train_args["lr_scheduler"],
        "drop_iters": train_args["drop_iters"],
        "drop_begin": train_args["drop_begin"],
        "drop_rate": train_args["drop_rate"],
        "use_checkpoint": "1",
    }
    json.dump({"pretrain":real_train_args}, open(train_config_dir, "w"), ensure_ascii=False, indent=4)
    env = deepcopy(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = train_args["cuda_visible_devices"]
    print(os.listdir(WORK_DIR))
    return Popen(["./pretrain_dragonfly.sh", NAME, str(train_args["n_gpus"])], cwd=WORK_DIR, env=env)
"""
model_unique
load
dataset_config
GPU_PER_NODE


train:
    batch_size
    train_iters
    max_length
    n_gpus/GPU_PER_NODE
    lr
    warmup_iters
    lr_scheduler
    drop_begin
    drop_rate
    drop_iters



config(generated)
"""
