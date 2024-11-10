import os
from typing import Dict, Any

def gen_cmd_9g(train_args):
    args = {}
    args["model_unique"] = "2b_0701"
    # args["resume_ckpt"]=""
    args["config"] = "2.4b"
    args["flash"] = "cuda"
    args["batch_size"] = "1"
    args["max_length"] = "4096"
    args["save_iters"] = "500"
    args["train_iters"] = "10"
    args["dataset_config"] = "datasets_info"
    args["local"] = "False"
    args["dataloader"] = "indexed"
    args["save"] = "True"
    args["dataloader_num_threads"] = 1
    args["dataloader_prefetch"] = 1
    args["dataloader_prefetch_factor"] = 1
    args["dataloader_num_workers"] = 1
    args["lr"] = "1e-5"
    args["warmup_iters"] = "20"
    args["drop_iters"] = "0.1"
    args["tokenizer_path"] = "./tokenizer/tokenizer.model"
    args["load_grad"] = "False"
    args["grad_ckpt_num"] = "160"
    args["exp_group"] = ""
    args["ignore_cuda_oom"] = "1"
    args["tensorboard_all_tasks"] = "0"
    args["stop_when_end"] = "0"
    args["only_run_dataloader"] = "0"
    args["eps"] = "1e-6"
    args["inspect_iters"] = "100"
    args["strict_state_dict"] = "1"
    args["only_load_model"] = "1"
    args["lr_scheduler"] = "cosine"
    args["resume_no_optimze"] = "0"
    args["tp_size"] = "1"
    args["parallel_load_datastate"] = "8"
    args["async_save"] = "False"
    args["load_dataloader_ckpt"] = "0"
    args["drop_begin"] = "-1"
    args["drop_rate"] = "0.5"
    args["use_checkpoint"] = "0"

    # 基于微调的预训练模型路径
    args["load"] = "../models/sft_2b/"

    args.update(train_args)

    OPTS = []
    OPTS.append(f"--model-config model_configs/{args['config']}.json")
    OPTS.append(f"--batch-size {args['batch_size']}")
    OPTS.append(f"--train-iters {args['train_iters']}")
    OPTS.append(f"--save-iters {args['save_iters']}")
    OPTS.append("--save-name fm9g_live_checkpoint")
    OPTS.append(f"--max-length {args['max_length']}")
    OPTS.append(f"--lr {args['lr']}")
    OPTS.append(f"--inspect-iters {args['inspect_iters']}")
    OPTS.append(f"--warmup-iters {args['warmup_iters']}")
    OPTS.append(f"--drop-iters {args['drop_iters']}")
    OPTS.append(f"--lr_scheduler {args['lr_scheduler']}")
    OPTS.append("--offload")
    # OPTS.append("--vocab ./tokenizer/vocab.txt")
    OPTS.append(f"--flash {args['flash']}")
    OPTS.append(f"--tensorboard_all_tasks {args['tensorboard_all_tasks']}")
    OPTS.append(f"--ignore_cuda_oom {args['ignore_cuda_oom']}")
    OPTS.append(f"--stop_when_end {args['stop_when_end']}")
    OPTS.append(f"--only_run_dataloader {args['only_run_dataloader']}")
    OPTS.append(f"--eps {args['eps']}")
    OPTS.append(f"--strict_state_dict {args['strict_state_dict']}")
    OPTS.append(f"--only_load_model {args['only_load_model']}")
    OPTS.append(f"--resume_no_optimze {args['resume_no_optimze']}")
    OPTS.append(f"--tokenizer_path {args['tokenizer_path']}")
    OPTS.append("--weight-decay 0.1")
    OPTS.append(f"--tp-size {args['tp_size']}")
    OPTS.append(f"--parallel_load_datastate {args['parallel_load_datastate']}")
    OPTS.append(f"--load_dataloader_ckpt {args['load_dataloader_ckpt']}")
    OPTS.append(f"--drop_begin {args['drop_begin']}")
    OPTS.append(f"--drop_rate {args['drop_rate']}")
    OPTS.append(f"--use_checkpoint {args['use_checkpoint']}")
    OPTS.append(f"--load {args['load']}")

    if args["dataloader"] == "indexed":
        OPTS.append(f"--dataloader_num_threads {args['dataloader_num_threads']}")
        OPTS.append(f"--dataloader_prefetch {args['dataloader_prefetch']}")
        OPTS.append(f"--dataloader_num_workers {args['dataloader_num_workers']}")
        OPTS.append(f"--dataloader_prefetch_factor {args['dataloader_prefetch_factor']}")

    if args["save"] == "True":
        OPTS.append(f"--save ./data/checkpoints/{args['model_unique']}/")
        OPTS.append(f"--save-model ./not_exist/{args['model_unique']}/")
    else:
        print("won't save model")

    # logs，/local/logs 等价于 ./datalogs（软链）
    os.makedirs(f"./data/checkpoints/logs/{args['model_unique']}", exist_ok=True)
    OPTS.append(f"--log-dir ./data/checkpoints/logs/{args['model_unique']}")
    OPTS.append(f"--tensorboard ./data/tensorboard/{args['exp_group']}{args['model_unique']}/")

    if args["local"] == "True":
        current_dir = os.getcwd()
        OPTS.append(f"--dataset ./datasets/{args['dataset_config']}.json")
    else:
        current_dir = os.getcwd()
        OPTS.append(f"--dataset ./datasets/{args['dataset_config']}.json")
        print(f"Platform config: {os.getenv('PLATFORM_CONFIG_PATH')}")

    OPTS = "\n    ".join(OPTS)

    GPUS_PER_NODE = 8
    NNODES = 1
    RANK = 0
    MASTER_ENDPOINT = "localhost"
    MASTER_PORT = 9001
    CMD = f"torchrun\n    --nnodes={NNODES}\n    --nproc_per_node={GPUS_PER_NODE}\n    --node_rank={RANK}\n    --rdzv_id=1\n    --rdzv_backend=c10d\n    --rdzv_endpoint={MASTER_ENDPOINT}:{MASTER_PORT}\n    {os.getenv('PRETRAIN_ENTRY')} {OPTS}"
    return CMD

def validate_args_9g(train_args: Dict[str, Any]) -> str:
    if(train_args["model_name_or_path"] == None):
        return "请选择模型检查点"
    if(train_args["dataset"] == ""):
        return "请选择数据集"
    return ""
    