"""
Kor NLI, STS Dataset

Author: DaeHyeon Gi <spliter2157@gmail.com>
"""
from typing import Optional
from collections import defaultdict

from torch.utils.data import Dataset
from transformers import BertTokenizer

from common.data import KorNLIDataset, KorSTSDataset, PatentPairDataset


class TripletDataset(Dataset):
    def __init__(self, phase: str) -> None:
        super().__init__()
        nli_dataset = KorNLIDataset()
        self.data = self.compose_triplet(dataset=getattr(nli_dataset, phase)())

    def compose_triplet(self, dataset: object) -> list[tuple[str, str, str]]:
        data_dict = defaultdict(dict)
        for idx, d in enumerate(dataset):
            if idx % 3 == 0:
                anchor = d[0]
            if d[2] == 0:
                data_dict[anchor]["entailment"] = d[1]
            elif d[2] == 2:
                data_dict[anchor]["contradiction"] = d[1]
        triplet_data = []
        for anchor, hypothesis in data_dict.items():
            if "entailment" in hypothesis and "contradiction" in hypothesis:
                triplet_data.append((anchor, hypothesis["entailment"], hypothesis["contradiction"]))
                # triplet_data.append((anchor, hypothesis["entailment"]))
        return triplet_data

    def __getitem__(self, index: int) -> tuple[str, str, str]:
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)


class STSDataset(Dataset):
    def __init__(self, phase: str) -> None:
        super().__init__()
        self.dataset = KorSTSDataset()
        self.data = getattr(self.dataset, phase)()

    def __getitem__(self, index: int) -> tuple[str, str, float]:
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)


class PatentDataset(Dataset):
    def __init__(self, phase: str) -> None:
        super().__init__()
        self.dataset = PatentPairDataset(
            train_pair_file="/data/dh/personel/train.json",
            val_pair_file="/data/dh/personel/validation.json"
        )
        self.data = getattr(self.dataset, phase)()

    def __getitem__(self, index: int) -> tuple[str, str, Optional[float]]:
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)
