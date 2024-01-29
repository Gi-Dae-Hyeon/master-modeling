import logging
from typing import Any, Optional, Callable

import torch
import lightning.pytorch as pl
from lightning.fabric.utilities.rank_zero import _get_rank
from lightning.pytorch.callbacks.callback import Callback
from lightning.pytorch.utilities.rank_zero import rank_prefixed_message, rank_zero_warn
from rank_bm25 import BM25Plus

log = logging.getLogger(__name__)


class HardnessAnealingSampler(Callback):
    mode_dict = {"min": torch.lt, "max": torch.gt}
    order_dict = {"min": "<", "max": ">"}
    def __init__(
        self,
        monitor: str,
        mode: str,
        patience: int,
        min_delta: float,
        verbose: bool = True
    ) -> None:
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.wait_count = 0
        self.stopped_epoch = 0
        self.hardness = 1
        self.best_score = torch.inf if mode == "max" else -torch.inf

    def __call__(
        self,
        query: list[str],
        corpus: list[str],
        vectors: torch.Tensor
    ) -> torch.Tensor:
        if len(corpus) != vectors.size()[0]:
            raise ValueError()
        tokenized_corpus = [doc.split(" ") for doc in set(corpus)]
        bm25 = BM25Plus(tokenized_corpus)

        scores = [bm25.get_scores(q.split()) for q in query]

        negative_samples = torch.Tensor()
        for score in scores:
            sorted_indices = torch.argsort(torch.Tensor(score), descending=True)
            negative_idx = sorted_indices[-self.hardness]
            negative_samples = torch.cat([negative_samples, vectors[negative_idx]])
        return negative_samples.unsqueeze(1).size()

    @property
    def state_key(self) -> str:
        return self._generate_state_key(
            monitor=self.monitor,
            mode=self.mode,
        )

    @property
    def monitor_op(self) -> Callable:
        return self.mode_dict[self.mode]

    def setup(self, trainer: pl.Trainer, *args, **kwargs) -> None:
        if self._check_on_train_epoch_end is None:
            # if the user runs validation multiple times per training epoch or multiple training epochs without
            # validation, then we run after validation instead of on train epoch end
            self._check_on_train_epoch_end = trainer.val_check_interval == 1.0 and trainer.check_val_every_n_epoch == 1

    def _validate_condition_metric(self, logs: dict[str, torch.Tensor]) -> bool:
        monitor_val = logs.get(self.monitor)

        error_msg = (
            f"Hardness Annealing Sampler conditioned on metric `{self.monitor}` which is not available."
            " Pass in or modify your `HardnessAnnealingSampler` callback to use any of the following:"
            f' `{"`, `".join(list(logs.keys()))}`'
        )

        if monitor_val is None:
            if self.strict:
                raise RuntimeError(error_msg)
            if self.verbose > 0:
                rank_zero_warn(error_msg, category=RuntimeWarning)
            return False
        return True

    def state_dict(self) -> dict[str, Any]:
        return {
            "wait_count": self.wait_count,
            "best_score": self.best_score,
            "patience": self.patience,
            "hardness": self.hardness,
            "stopped_epoch": self.stopped_epoch
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.wait_count = state_dict["wait_count"]
        self.best_score = state_dict["best_score"]
        self.patience = state_dict["patience"]
        self.hardness = state_dict["hardness"]
        self.stopped_epoch = state_dict["stopped_epoch"]

    def on_train_epoch_end(self, trainer: pl.Trainer, *args, **kwargs) -> None:
        if not self._check_on_train_epoch_end or self._should_skip_check(trainer):
            return
        self._run_hardness_check(trainer)

    def on_validation_end(self, trainer: pl.Trainer, *args, **kwargs) -> None:
        if self._check_on_train_epoch_end or self._should_skip_check(trainer):
            return
        self._run_hardness_check(trainer)

    def _run_hardness_check(self, trainer: pl.Trainer) -> None:
        logs = trainer.callback_metrics
        if trainer.fast_dev_run or not self._validate_condition_metric(logs):
            return
        current = logs[self.monitor].squeeze()
        harder, reason = self._evaluate_hardness_criteria(current)
        if harder:
            self.hardness += 1
        if reason and self.verbose:
            self._log_info(trainer, reason, self.log_rank_zero_only)

    def _evaluate_hardness_criteria(self, current: torch.Tensor) -> tuple(bool, Optional[str]):
        harder = False
        reason = None

        if self.monitor_op(current - self.min_delta, self.best_score.to(current.device)):
            harder = False
            reason = self._improvement_message(current)
            self.best_score = current
            self.wait_count = 0
        else:
            self.wait_count += 1
            if self.wait_count >= self.patience:
                harder = True
                reason = (
                    f"Monitored metric {self.monitor} did not improve in the last {self.wait_count} records",
                    f"\tBest Score: {self.best_score:.3f}. Signaling Trainer to HARDER!!"
                )
        return harder, reason

    def _improvement_message(self, current: torch.Tensor) -> str:
        if torch.isfinite(self.best_score):
            msg = (
                f"Metric {self.monitor} improved by {abs(self.best_score - current):.3f} >="
                f" min_delta = {abs(self.min_delta)}. New best score: {current:.3f}"
            )
        else:
            msg = f"Metric {self.monitor} improved. New best score: {current:.3f}"
        return msg

    @staticmethod
    def _log_info(trainer: Optional[pl.Trainer], message: str, log_rank_zero_only: bool) -> None:
        rank = _get_rank(
            strategy=(
                trainer.strategy if trainer is not None else None
            ),
        )
        if trainer is not None and trainer.world_size <= 1:
            rank = None
        message = rank_prefixed_message(message, rank)
        if rank is None or not log_rank_zero_only or rank == 0:
            log.info(message)
