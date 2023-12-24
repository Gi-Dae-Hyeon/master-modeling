"""
Kor NLI, STS Dataset

Author: DaeHyeon Gi <spliter2157@gmail.com>
"""
from collections import defaultdict
from typing import List, Tuple

from torch.utils.data import Dataset

from common.data import KorNLIDataset, KorSTSDataset


class TripletDataset(Dataset):
    def __init__(self, phase: str) -> None:
        super().__init__()
        nli_dataset = KorNLIDataset()
        self.data = self.compose_triplet(dataset=getattr(nli_dataset, phase)())

    def compose_triplet(self, dataset: object) -> List[Tuple[str, str, str]]:
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
        return triplet_data

    def __getitem__(self, index: int) -> Tuple[str, str, str]:
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)


class STSDataset(Dataset):
    def __init__(self, phase: str) -> None:
        super().__init__()
        self.dataset = KorSTSDataset()
        self.data = getattr(self.dataset, phase)()

    def __getitem__(self, index: int) -> Tuple[str, str, float]:
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)
