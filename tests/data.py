import datasets
import numpy as np
import torch
import torchvision
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import LinearPartitioner
from torch.utils.data import DataLoader


class CUDA_VisionDataSet:
    """
    Loads the dataset fully onto the GPU before iterating the torch.DataLoader.
    Reason: torch.DataLoader usually iterates on CPU, and sends to GPU in batches.
    Here, the dataset is loaded onto the GPU when called, and any subsequent torch.DataLoader iterations happen on GPU.

    Can significantly speed up training if the dataset is big.
    Needs substantial VRAM, don't use if dataset size is close or bigger to VRAM.
    Inputs:
        dataset: the huggingface dataset to be used, may be the full dataset or a client dataset.
        transform: any train or test transform to apply.
        image_name: the column name of the image. not consistent across huggingface datasets.
        target_name: label column name.

    Output:
        a torch.utils.data.Dataset which fully loads onto the GPU.

    Remark: We convert to RGB (on CPU) to handle grayscale images in RGB datasets
            (i.e. a mostly RGB dataset may include some grayscale images which can bug out the process)
            Do comment that line out if using a grayscale dataset, as computation will be significantly
            increased else.
            CUDA_VisionDataSet is a specific feature for flwr experiments. Do not use if not sure about GPU VRAM,
            if using very big datasets, or if just testing.
    """

    def __init__(
        self,
        dataset: datasets.Dataset,
        transform: torchvision.transforms.Compose = torchvision.transforms.Compose([]),
        image_name: str = "image",
        target_name: str = "label",
    ):
        self.data = dataset[image_name]
        self.targets = dataset[target_name]
        self.len = len(self.targets)

        self.train_data = torch.stack(
            [
                (
                    transform(entry).to_numpy()
                    if isinstance(entry, torch.Tensor)
                    else transform(
                        torchvision.transforms.functional.to_tensor(
                            entry.convert("RGB")
                        )
                    )
                )
                for entry in self.data
            ],
            dim=0,
        ).to("cuda")
        self.targets = torch.tensor(
            (np.array(self.targets).reshape(self.len)), dtype=torch.long
        ).to("cuda")
        del dataset

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.train_data[idx], self.targets[idx]


"""
For testing, use a LinearPartitioner with 10 clients on CIFAR-100 with a ResNet18 model.
"""


def split_data(num_clients):

    partitioner = LinearPartitioner(
        num_partitions=num_clients,
    )

    FDS = FederatedDataset(
        dataset="cifar10",
        partitioners={"train": partitioner},
    )

    trainloaders = [
        DataLoader(
            CUDA_VisionDataSet(
                FDS.load_partition(partition_id=partition_id),
                image_name="img",
                target_name="label",
            ),
            batch_size=256,
            shuffle=True,
            drop_last=True,
        )
        for partition_id in range(num_clients)
    ]

    testloader = DataLoader(
        CUDA_VisionDataSet(
            FDS.load_split("test"),
            image_name="img",
            target_name="label",
        ),
        batch_size=256,
        shuffle=True,
        drop_last=True,
    )

    return trainloaders, testloader
