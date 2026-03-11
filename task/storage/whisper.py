import random
from collections import OrderedDict
from typing import List, Tuple

import numpy as np
import torch
from datasets import Dataset, concatenate_datasets
from flwr.common import NDArrays
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import GroupedNaturalIdPartitioner
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")

fds = None  # Cache FederatedDataset


def get_encoding_fn(processor):
    """Return a function to pre-process/encode the SpeechCommands dataset.

    Working with the 12-class version: reassign labels accordingly.
    """

    def prepare_dataset(batch):
        audio = batch["audio"]
        data = {}
        data["data"] = processor(
            audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt"
        ).input_features

        # All unknown keywords are assigned label 11. Silence clips get label 10.
        data["targets"] = (
            11
            if batch["is_unknown"]
            else (10 if batch["label"] == 35 else batch["label"])
        )
        return data

    return prepare_dataset


def prepare_silences_dataset(train_dataset, ratio_silence: float = 0.1) -> Dataset:
    """Generate silence clips from background audio for training."""
    silences = train_dataset.filter(lambda x: x["label"] == 35)
    num_silence_total = int(len(train_dataset) * ratio_silence)
    num_silence_per_bkg = num_silence_total // len(silences)

    silence_to_add = []
    for sil in silences:
        sil_array = sil["audio"]["array"]
        sr = sil["audio"]["sampling_rate"]
        for _ in range(num_silence_per_bkg):
            random_offset = random.randint(0, len(sil_array) - sr - 1)
            sil_array_crop = sil_array[random_offset : random_offset + sr]

            entry = sil
            silence_to_add.append(entry)
            silence_to_add[-1]["audio"]["array"] = sil_array_crop

    return Dataset.from_list(silence_to_add)


def load_data(partition_id: int, remove_cols: str):
    """Load and preprocess a client partition."""
    global fds
    if fds is None:
        partitioner = GroupedNaturalIdPartitioner(
            partition_by="speaker_id", group_size=5
        )
        fds = FederatedDataset(
            dataset="speech_commands",
            subset="v0.02",
            partitioners={"train": partitioner},
            trust_remote_code=True,
        )

    partition = fds.load_partition(partition_id)
    encoding_fn = get_encoding_fn(processor)

    remove_cols = remove_cols.split(",")
    partition = partition.map(encoding_fn, num_proc=2, remove_columns=remove_cols)

    # Add ~10% silence training examples
    partitioner = fds.partitioners["train"]
    ratio_silences_for_client = 0.1 * (len(partition) / len(partitioner.dataset))
    silence_dataset = prepare_silences_dataset(
        partitioner.dataset, ratio_silences_for_client
    )
    if len(silence_dataset) > 0:
        silence_enc = silence_dataset.map(encoding_fn)
        partition = concatenate_datasets([partition, silence_enc])

    return partition


def get_model(device, num_classes, compile: bool = True):
    """Create model: Whisper-tiny Encoder + classification head."""
    encoder = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-tiny"
    ).get_encoder()
    encoder = encoder.to(device)
    if compile:
        encoder = torch.compile(encoder)

    # Classification head (~782K parameters, only part trained in federation)
    classifier = torch.nn.Sequential(
        torch.nn.Conv1d(1500, 128, kernel_size=1),
        torch.nn.ReLU(),
        torch.nn.Flatten(1),
        torch.nn.Linear(128 * 384, num_classes),
    ).to(device)
    return encoder, classifier


def set_params(model: torch.nn.ModuleList, params: List[NDArrays]):
    """Set model weights from a list of NumPy ndarrays."""
    params_dict = zip(model.state_dict().keys(), params)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def get_params(module: torch.nn.ModuleList):
    return [val.cpu().numpy() for _, val in module.state_dict().items()]


class RunningAvg:
    def __init__(self):
        self.n = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.n += 1

    def __call__(self):
        return self.total / self.n


def construct_balanced_sampler(trainset):
    hist, _ = np.histogram(trainset["targets"], bins=12)
    hist_mask = hist > 0
    w_per_class = len(trainset) / (hist + 1)
    w_per_class += 1
    w_per_class *= hist_mask
    w_ss = [w_per_class[t] for t in trainset["targets"]]
    return WeightedRandomSampler(w_ss, len(w_ss))


def train_one_epoch(
    model,
    classifier,
    optimizer,
    criterion,
    dataloader,
    device,
    disable_tqdm: bool = False,
):
    """Train the classification head."""
    model.eval()
    classifier.train()
    classifier.to(device)
    loss_avg, acc_avg = RunningAvg(), RunningAvg()
    avg_loss, avg_acc = 0.0, 0.0
    with tqdm(total=len(dataloader.dataset), disable=disable_tqdm) as t:
        for b in dataloader:
            optimizer.zero_grad()
            data = b["data"].squeeze().to(device)
            labels = b["targets"].to(device)
            with torch.no_grad():
                res = model(data)[0]

            resres = classifier(res)

            loss = criterion(resres.float(), labels)
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(resres.data, 1)
            correct = (predicted == labels).sum().item()
            acc = correct / data.shape[0]
            loss_ = loss.cpu().item()

            loss_avg.update(loss_)
            acc_avg.update(acc)

            t.update(data.shape[0])
            avg_loss, avg_acc = loss_avg(), acc_avg()
            t.set_postfix({"avg_loss": f"{avg_loss:.4f}", "avg_acc": f"{avg_acc:.4f}"})

    return avg_loss, avg_acc


def eval_model(model, classifier, criterion, dataloader, device):
    """Evaluate the model on a validation/test set."""
    model.eval()
    classifier.eval()
    classifier.to(device)
    correct = 0
    loss_ = 0
    total = 0
    with torch.no_grad():
        for b in dataloader:
            data = b["data"].squeeze().to(device)
            labels = b["targets"].to(device)
            res = model(data)[0]
            resres = classifier(res)

            loss = criterion(resres.float(), labels)
            _, predicted = torch.max(resres.data, 1)
            correct += (predicted == labels).sum().item()
            total += data.shape[0]
            loss_ += loss.cpu().item()

    accuracy = correct / total
    loss = loss_ / total

    return loss, accuracy
