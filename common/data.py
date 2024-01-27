"""
NLP Dataset Modules
Author: DaeHyeon Gi <spliter2157@gmail.com>
"""

import json
from pathlib import Path

import datasets
from tqdm import tqdm


class KorNLIDataset:
    """
    한국어 자연어 추론 데이터셋.

    HuggingFace datasets 라이브러리에서 한국어 자연어 추론 데이터셋을 로드합니다.
    학습 데이터셋을 반환하는 메서드를 제공합니다.

    Methods:
        get_train_dataset: premise, hypothesis, label을 포함하는 튜플 리스트를 반환합니다.
        get_validation_dataset: premise, hypothesis, label을 포함하는 튜플 리스트를 반환합니다.
        get_test_dataset: premise, hypothesis, label을 포함하는 튜플 리스트를 반환합니다.
    """

    def __init__(self) -> None:
        self.nli = datasets.load_dataset("kor_nlu", "nli")
        self.multi_nli = datasets.load_dataset("kor_nli", "multi_nli")

    def train(self) -> list[tuple[str, str, int]]:
        """
        학습 데이터셋을 반환합니다.

        이 메서드는 SNLI, MultiNLI 학습 데이터를 조합하여
        premise, hypothesis, label을 포함하는 튜플 리스트를 반환합니다.

        Returns:
            premise, hypothesis, label을 포함하는 튜플 리스트.
        """
        train_dataset = []
        nli_data = self.nli["train"]
        for prem, hypo, label in zip(nli_data["premise"], nli_data["hypothesis"], nli_data["label"]):
            if prem is not None and hypo is not None:
                train_dataset.append((prem, hypo, label))

        m_nli_data = self.multi_nli["train"]
        for prem, hypo, label in zip(m_nli_data["premise"], m_nli_data["hypothesis"], m_nli_data["label"]):
            if prem is not None and hypo is not None:
                train_dataset.append((prem, hypo, label))
        return train_dataset

    def validation(self) -> list[tuple[str, str, int]]:
        """
        검증 데이터셋을 반환합니다.

        이 메서드는 XNLI 데이터셋의 검증 데이터를 사용하여
        premise, hypothesis, label을 포함하는 튜플 리스트를 반환합니다.

        Returns:
            premise, hypothesis, label을 포함하는 튜플 리스트.
        """
        validation_dataset = []
        data = self.nli["validation"]
        for prem, hypo, label in zip(data["premise"], data["hypothesis"], data["label"]):
            if prem is not None and hypo is not None:
                validation_dataset.append((prem, hypo, label))
        return validation_dataset

    def test(self) -> list[tuple[str, str, int]]:
        """
        테스트 데이터셋을 반환합니다.

        이 메서드는 XNLI 데이터셋의 테스트 데이터를 사용하여
        premise, hypothesis, label을 포함하는 튜플 리스트를 반환합니다.

        Returns:
            premise, hypothesis, label을 포함하는 튜플 리스트.
        """
        test_dataset = []
        data = self.nli["test"]
        for prem, hypo, label in zip(data["premise"], data["hypothesis"], data["label"]):
            if prem is not None and hypo is not None:
                test_dataset.append((prem, hypo, label))
        return test_dataset


class KorSTSDataset:
    """
    한국어 문장 유사도 데이터셋.

    HuggingFace datasets 라이브러리에서 한국어 문장 유사도 데이터셋을 로드합니다.
    학습, 검증, 테스트 데이터셋을 반환하는 메서드를 제공합니다.

    Methods:
        get_train_dataset: sentence1, sentence2, score를 포함하는 튜플 리스트를 반환합니다.
        get_validation_dataset: sentence1, sentence2, score를 포함하는 튜플 리스트를 반환합니다.
        get_test_dataset: sentence1, sentence2, score를 포함하는 튜플 리스트를 반환합니다.
    """

    def __init__(self) -> None:
        self.sts = datasets.load_dataset("kor_nlu", "sts")

    def train(self) -> list[tuple[str, str, float]]:
        """
        학습 데이터셋을 반환합니다.

        이 메서드는 STS 학습 데이터를 사용하여
        sentence1, sentence2, score를 포함하는 튜플 리스트를 반환합니다.

        Returns:
            sentence1, sentence2, score를 포함하는 튜플 리스트.
        """
        train_dataset = []
        data = self.sts["train"]
        for prem, hypo, label in zip(data["sentence1"], data["sentence2"], data["score"]):
            if prem is not None and hypo is not None:
                train_dataset.append((prem, hypo, label / 5))
        return train_dataset

    def validation(self) -> list[tuple[str, str, float]]:
        """
        검증 데이터셋을 반환합니다.

        이 메서드는 STS 검증 데이터를 사용하여
        sentence1, sentence2, score를 포함하는 튜플 리스트를 반환합니다.

        Returns:
            sentence1, sentence2, score를 포함하는 튜플 리스트.
        """
        validation_dataset = []
        data = self.sts["validation"]
        for prem, hypo, label in zip(data["sentence1"], data["sentence2"], data["score"]):
            if prem is not None and hypo is not None:
                validation_dataset.append((prem, hypo, label / 5))
        return validation_dataset

    def test(self) -> list[tuple[str, str, float]]:
        """
        테스트 데이터셋을 반환합니다.

        이 메서드는 STS 테스트 데이터를 사용하여
        sentence1, sentence2, score를 포함하는 튜플 리스트를 반환합니다.

        Returns:
            sentence1, sentence2, score를 포함하는 튜플 리스트.
        """
        test_dataset = []
        data = self.sts["test"]
        for prem, hypo, label in zip(data["sentence1"], data["sentence2"], data["score"]):
            if prem is not None and hypo is not None:
                test_dataset.append((prem, hypo, label / 5))
        return test_dataset


class PatentPairDataset:
    def __init__(
        self,
        train_pair_file: str,
        val_pair_file: str,
    ) -> None:
        self.train_dataset_path = Path(train_pair_file)
        self.val_dataset_path = Path(val_pair_file)

    def train(self) -> list[tuple[str, str]]:
        ret = []
        with Path(self.train_dataset_path).open("r", encoding="utf-8") as fp:
            self.train_dataset = json.load(fp)
        for data in tqdm(self.train_dataset, desc="Loading Train Dataset"):
            ret.append((data["query"], data["candidate"]))
        return ret

    def validation(self) -> list[tuple[str, str, float]]:
        ret = []
        with Path(self.val_dataset_path).open("r", encoding="utf-8") as fp:
            self.val_dataset = json.load(fp)[:100000]
        for data in tqdm(self.val_dataset, desc="Loading Val Dataset"):
            ret.append((data["query"], data["candidate"], data["score"]))
        return ret
