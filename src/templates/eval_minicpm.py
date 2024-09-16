from opencompass.models import HuggingFace, VLLM
from mmengine.config import read_base

with read_base():
    from .datasets.humaneval.humaneval_gen_8e312c import humaneval_datasets
    from .datasets.mbpp.mbpp_gen_no_instruction import mbpp_datasets
    from .datasets.mmlu.mmlu_gen_4d595a import mmlu_datasets
    from .datasets.hellaswag.hellaswag_gen_6faab5 import hellaswag_datasets
    from .datasets.gsm8k.gsm8k_gen_1d7fe4 import gsm8k_datasets

datasets = []
datasets += humaneval_datasets
datasets += mbpp_datasets
datasets += mmlu_datasets
datasets += hellaswag_datasets
datasets += gsm8k_datasets

_meta_template = dict(
    round=[
        dict(role="HUMAN", begin='<用户>'),
        dict(role="BOT", begin="<AI>", generate=True),
    ],
)

models = [
    dict(
        type=VLLM,
        abbr="freeze_sft_240901-1506",
        path="/home/zhenghuanyang/evaluate/opencompass/saves/freeze_sft_240901-1506",
        model_kwargs=dict(gpu_memory_utilization=0.9,tensor_parallel_size=1),
        meta_template=_meta_template,
        max_out_len=1024,
        max_seq_len=2048,
        batch_size=32,
        generation_kwargs=dict(temperature=0, top_p=1, max_tokens=2048, stop=['<用户>']),
        # run_cfg=dict(num_gpus=1, num_procs=1),
        end_str='<用户>',
    )
]