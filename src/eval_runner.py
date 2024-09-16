template = """
from opencompass.models import HuggingFace, VLLM
from mmengine.config import read_base

with read_base():
    from .datasets.humaneval.humaneval_gen_8e312c import humaneval_datasets
    from .datasets.humaneval_plus.humaneval_plus_gen_8e312c import humaneval_plus_datasets
    from .datasets.mbpp.deprecated_sanitized_mbpp_gen_1e1056 import sanitized_mbpp_datasets
    from .datasets.mbpp_plus.deprecated_mbpp_plus_gen_94815c import mbpp_plus_datasets
    
    from .datasets.mmlu.mmlu_gen_4d595a import mmlu_datasets
    from .datasets.hellaswag.hellaswag_gen_6faab5 import hellaswag_datasets
    from .datasets.ARC_e.ARC_e_gen_1e0de5 import ARC_e_datasets
    from .datasets.bbh.bbh_gen_5bf00b import bbh_datasets
    from .datasets.ceval.ceval_gen_5f30c7 import ceval_datasets
    from .datasets.cmmlu.cmmlu_gen_c13365 import cmmlu_datasets
    from .datasets.gsm8k.gsm8k_gen_1d7fe4 import gsm8k_datasets
    
    from .summarizers.groups.bbh import bbh_summary_groups
    from .summarizers.groups.mmlu import mmlu_summary_groups
    from .summarizers.groups.cmmlu import cmmlu_summary_groups
    from .summarizers.groups.ceval import ceval_summary_groups

datasets = []

%s

_meta_template = dict(
    round=[
        dict(role="HUMAN", begin='<用户>'),
        dict(role="BOT", begin="<AI>", generate=True),
    ],
)

models = [
    dict(
        type=VLLM,
        abbr="%s",
        path="%s",
        model_kwargs=dict(gpu_memory_utilization=0.9,tensor_parallel_size=%s),
        meta_template=_meta_template,
        max_out_len=%s,
        max_seq_len=%s,
        batch_size=%s,
        generation_kwargs=dict(temperature=%s, top_p=%s, max_tokens=%s, stop=['<用户>']),
        run_cfg=dict(num_gpus=%s, num_procs=%s),
    )
]

summerize_groups = []

%s

summarizer = dict(
    summary_groups=summerize_groups,
)

"""
import os
def start_eval(eval_args):
    filename = "opencompass/configs/eval_%s.py"%eval_args["real_abbr"]
    options = eval_args["dataset"]
    
    dataset_dict = {
        "HumanEval":"humaneval_datasets",
        "HumanEval+":"humaneval_plus_datasets",
        "MBPP[sanitized]":"sanitized_mbpp_datasets",
        "MBPP+":"mbpp_plus_datasets",
        "MMLU":"mmlu_datasets",
        "HellaSwag":"hellaswag_datasets",
        "ARC-e":"ARC_e_datasets",
        "BBH":"bbh_datasets", 
        "C-Eval":"ceval_datasets", 
        "CMMLU":"cmmlu_datasets", 
        "GSM8K":"gsm8k_datasets"
    }
    
    summerizer_dict = {
        "BBH":"bbh_summary_groups",
        "MMLU":"mmlu_summary_groups",
        "CMMLU":"cmmlu_summary_groups",
        "C-Eval":"ceval_summary_groups"
    }
    
    dataset_str = ""
    summerizer_str = ""
    for option in options:
        dataset_str += "datasets += %s\n"%dataset_dict[option]
        if option in summerizer_dict:
            summerizer_str += "summerize_groups += %s\n"%summerizer_dict[option]
    
    config = template%(
        dataset_str,
        eval_args["real_abbr"],
        os.path.abspath(eval_args["ckpt_full_path"]),
        eval_args["num_gpus"],
        eval_args["max_out_len"],
        eval_args["max_seq_len"],
        eval_args["batch_size"],
        eval_args["temperature"],
        eval_args["top_p"],
        eval_args["max_tokens"],
        eval_args["num_gpus"],
        eval_args["num_procs"],
        summerizer_str
    )
    with open(filename,"w",encoding="utf-8") as f:
        f.write(config)
    
    return '''cd opencompass && mkdir -p "./outputs/%s" && python run.py "configs/eval_%s.py" -w "./outputs/%s"'''%(eval_args["real_abbr"], eval_args["real_abbr"], eval_args["real_abbr"])
    
    