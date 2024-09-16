import os, json
from copy import deepcopy
from subprocess import Popen, TimeoutExpired
from typing import Any, Dict, List
from src.utils.gen_cmd import gen_cmd
from src.utils.plot import gen_loss_plot


RUNNING_LOG = "running_log.txt"
TRAINER_LOG = "trainer_log.jsonl"
TRAINING_ARGS = "training_args.yaml"
TRAINING_ARGS_NAME = "training_args.bin"

class Runner():
    def __init__(self) -> None:
        self.aborted = False
    def launch(self, train_args, freeze_args, lora_args, cuda_visible_devices):
        os.makedirs(train_args["output_dir"], exist_ok=True)

        env = deepcopy(os.environ)
        env["LLAMABOARD_ENABLED"] = "1"
        env["LLAMABOARD_WORKDIR"] = train_args["output_dir"]
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        
        self.train_args = train_args
        self.freeze_args = freeze_args
        self.lora_args = lora_args

        self.trainer = Popen(gen_cmd(train_args, freeze_args, lora_args), env=env, shell=True, preexec_fn=os.setsid)
        yield from self.monitor()
            
    def monitor(self):
        self.is_running = True
        running_log = ""
        output_path = self.train_args["output_dir"]

        while self.trainer is not None:
            if self.aborted:
                yield {
                    "end": True,
                    "output": "微调异常退出",
                    "progress": None,
                }
            else:
                running_log, running_progress, running_loss = get_trainer_info(output_path)
                return_dict = {
                    "output": running_log,
                    "progress": running_progress,
                }
                if running_loss is not None:
                   return_dict["loss_viewer"] = running_loss

                yield return_dict

            
            try:
                self.trainer.wait(0.1)
                self.trainer = None
            except TimeoutExpired:
                continue

        if os.path.exists(os.path.join(output_path, TRAINING_ARGS_NAME)):
            finish_info = running_log + "\n微调成功结束"
        else:
            finish_info = running_log + "\n微调结束,过程中存在错误"
        
        self.is_running = False
        return_dict = {
            "end": True,
            "output": finish_info,
            "progress": None,
        }
        yield return_dict


def get_trainer_info(output_path: os.PathLike):
    r"""
    Gets training infomation for monitor.
    """
    running_log = ""
    running_progress = None
    running_loss = None

    running_log_path = os.path.join(output_path, RUNNING_LOG)
    if os.path.isfile(running_log_path):
        with open(running_log_path, "r", encoding="utf-8") as f:
            running_log = f.read()

    trainer_log_path = os.path.join(output_path, TRAINER_LOG)
    if os.path.isfile(trainer_log_path):
        trainer_log: List[Dict[str, Any]] = []
        with open(trainer_log_path, "r", encoding="utf-8") as f:
            for line in f:
                trainer_log.append(json.loads(line))

        if len(trainer_log) != 0:
            latest_log = trainer_log[-1]
            percentage = latest_log["percentage"]
            label = "Running {:d}/{:d}: {} < {}".format(
                latest_log["current_steps"],
                latest_log["total_steps"],
                latest_log["elapsed_time"],
                latest_log["remaining_time"],
            )
            running_progress = (label, percentage)
            running_loss = gen_loss_plot(trainer_log)

    return running_log, running_progress, running_loss

