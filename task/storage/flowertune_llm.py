import logging
import math
import os
import warnings
from collections import OrderedDict
from typing import Tuple

import numpy as np
import torch
from datasets import Dataset
from flwr.common import ndarrays_to_parameters
from flwr.common.typing import NDArrays
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import SizePartitioner, SquarePartitioner
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from peft.peft_model import PeftModel
from peft.utils import prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    TrainingArguments,
)
from trl import SFTTrainer

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["RAY_DISABLE_DOCKER_CPU_WARNING"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)
logger = logging.getLogger(__name__)


model_cfg = {
    "name": "openlm-research/open_llama_3b_v2",
    "quantization": 4,
    "use_gradient_checkpointing": True,
    "lora.peft-lora-r": 32,
    "lora.peft-lora-alpha": 64,
}

TRAINING_CFG = {
    "save-every-round": 1,
    "learning-rate-max": 5e-5,
    "learning-rate-min": 1e-6,
    "seq-length": 128,
}

TRAINING_ARGS = {
    "disable_tqdm": True,
    "output_dir": "./results/",
    "per_device_train_batch_size": 64,
    "gradient_accumulation_steps": 1,
    "logging_steps": 10,
    "num_train_epochs": 1,
    "max_steps": 10,
    "save_steps": 1000,
    "save_total_limit": 10,
    "gradient_checkpointing": True,
    "lr_scheduler_type": "constant",
    "auto_find_batch_size": False,
}


def formatting_prompts_func(example):
    output_texts = []
    mssg = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    for i in range(len(example["instruction"])):
        text = f"{mssg}\n### Instruction:\n{example['instruction'][i]}\n### Response: {example['response'][i]}"
        output_texts.append(text)
    return output_texts


def get_tokenizer_and_data_collator_and_prompt_formatting(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=True, padding_side="right"
    )
    tokenizer.pad_token = tokenizer.eos_token
    response_template_with_context = "\n### Response:"
    response_template_ids = tokenizer.encode(
        response_template_with_context, add_special_tokens=False
    )[2:]
    data_collator = DataCollatorForLanguageModeling(
        response_template_ids, tokenizer=tokenizer
    )

    return tokenizer, data_collator, formatting_prompts_func


def get_model() -> PeftModel:
    """Load model with quantization config and LoRA."""
    if model_cfg["quantization"] == 4:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    elif model_cfg["quantization"] == 8:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        raise ValueError(
            f"Use 4-bit or 8-bit quantization. You passed: {model_cfg['quantization']}"
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name"],
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
    )

    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=model_cfg["use_gradient_checkpointing"]
    )

    peft_config = LoraConfig(
        r=model_cfg["lora.peft-lora-r"],
        lora_alpha=model_cfg["lora.peft-lora-alpha"],
        lora_dropout=0.075,
        task_type="CAUSAL_LM",
    )

    return get_peft_model(model, peft_config)


def size_based_split(num_clients, num_quality=0):
    size_min, size_max = [
        max(256, 52000 / (num_clients * 3)),
        int(52000 / (num_clients * 2)),
    ]
    sizes = [np.random.randint(size_min, size_max) for _ in range(num_clients)]
    remaining_points = 52000 - np.sum(sizes[num_quality:])
    for i in range(num_quality):
        sizes[i] = int(remaining_points / num_quality)
    partitioner = SizePartitioner(
        partition_sizes=sizes,
    )
    return FederatedDataset(
        dataset="vicgalle/alpaca-gpt4",
        partitioners={"train": partitioner},
        trust_remote_code=True,
    )


def square_split(num_clients):
    partitioner = SquarePartitioner(
        num_partitions=num_clients,
    )
    return FederatedDataset(
        dataset="vicgalle/alpaca-gpt4",
        partitioners={"train": partitioner},
        trust_remote_code=True,
    )


def load_data(partition_id: int, fds: FederatedDataset) -> Dataset:
    client_trainset = fds.load_partition(partition_id, "train")
    client_trainset = client_trainset.rename_column("output", "response")
    return client_trainset


def load_global_test_data(fds: FederatedDataset):
    global_test = fds.load_split("train")
    testinds = np.random.choice(len(global_test), size=int(len(global_test) / 20))
    global_test = global_test.select(testinds)
    global_test = global_test.rename_column("output", "response")
    return global_test


def cosine_annealing(
    current_round: int,
    total_round: int,
    lrate_max: float = 0.001,
    lrate_min: float = 0.0,
) -> float:
    """Implement cosine annealing learning rate schedule."""
    cos_inner = math.pi * current_round / total_round
    return lrate_min + 0.5 * (lrate_max - lrate_min) * (1 + math.cos(cos_inner))


tokenizer, data_collator, formatting_prompts_func = (
    get_tokenizer_and_data_collator_and_prompt_formatting(model_cfg["name"])
)


def train(model: PeftModel, trainloader: Dataset, current_round: int = 1, max_round: int = 10, **kwargs) -> None:
    new_lr = cosine_annealing(
        current_round,
        max_round,
        TRAINING_CFG["learning-rate-max"],
        TRAINING_CFG["learning-rate-min"],
    )

    training_arguments = TrainingArguments(**TRAINING_ARGS)
    training_arguments.learning_rate = new_lr
    training_arguments.output_dir = "./results/"

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_arguments,
        max_seq_length=TRAINING_CFG["seq-length"],
        train_dataset=trainloader,
        formatting_func=formatting_prompts_func,
    )
    trainer.train()


def test(
    model: PeftModel,
    testloader: Dataset,
) -> Tuple[float, float]:
    """Validate the model on the entire test set."""
    eval_args = TrainingArguments(**TRAINING_ARGS)
    eval_args.output_dir = "./results/trash"
    eval_args.per_device_eval_batch_size = 32
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=eval_args,
        max_seq_length=TRAINING_CFG["seq-length"],
        eval_dataset=testloader,
        formatting_func=formatting_prompts_func,
    )

    eval_loss = trainer.evaluate()["eval_loss"]
    del model
    torch.cuda.empty_cache()
    return 0, -eval_loss


def ndarrays_from_model(model: torch.nn.ModuleList) -> NDArrays:
    """Return the parameters of the current net."""
    state_dict = get_peft_model_state_dict(model)
    return [val.cpu().numpy() for _, val in state_dict.items()]


def ndarrays_to_model(model: torch.nn.ModuleList, params: NDArrays):
    """Change the parameters of the model using the given ones."""
    peft_state_dict_keys = get_peft_model_state_dict(model).keys()
    params_dict = zip(peft_state_dict_keys, params)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    set_peft_model_state_dict(model, state_dict)


def get_initial_parameters():
    init_model = get_model()
    initial_parameters = ndarrays_to_parameters(ndarrays_from_model(init_model))
    del init_model
    torch.cuda.empty_cache()
    return initial_parameters
