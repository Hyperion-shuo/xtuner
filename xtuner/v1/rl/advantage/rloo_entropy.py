import random
from typing import TYPE_CHECKING, cast

import torch

from xtuner.v1.rl.advantage.base import AdvantageEstimator
from xtuner.v1.rl.advantage.config import BaseAdvantageConfig

if TYPE_CHECKING:
    from xtuner.v1.data_proto.rl_data import RolloutState


class OverlongRLOOGroupEntropyEstimator(AdvantageEstimator):
    """RLOO with group entropy filtering.
    """

    def __init__(
        self, 
        entropy_upper_bound: float = 0.65, 
        entropy_lower_bound: float = 0.4, 
        tau_upper: float = 0.0, 
        tau_lower: float = 0.0, 
        coeff_min_upper: float = 0.2, 
        coeff_min_lower: float = 0.5,
        overlong_filer: bool = True,
    ) -> None:
        self.entropy_upper_bound = entropy_upper_bound
        self.entropy_lower_bound = entropy_lower_bound
        self.tau_upper = tau_upper
        self.tau_lower = tau_lower
        self.coeff_min_upper = coeff_min_upper
        self.coeff_min_lower = coeff_min_lower
        self.overlong_filer = overlong_filer

    def _compute_overlong_mask(self, group: list["RolloutState"]) -> torch.Tensor:
        overlong_mask = []
        overlong_indices = []  # 收集超长样本的索引

        for i, d in enumerate(group):
            if d.finish_reason == "stop":
                overlong_mask.append(1.0)
            else:
                overlong_indices.append(i)
                overlong_mask.append(0.0)  # 先默认设为0

        # 如果有超长样本，随机选择一个保留
        if overlong_indices:
            selected_idx = random.choice(overlong_indices)
            overlong_mask[selected_idx] = 1.0

        overlong_mask = torch.tensor(overlong_mask, dtype=torch.float32)
        return overlong_mask

    def _compute_advantages(self, rewards: torch.Tensor, group: list["RolloutState"]) -> torch.Tensor:
        if self.overlong_filer:
            overlong_mask = self._compute_overlong_mask(group).to(rewards.device)
        else:
            overlong_mask = torch.ones(len(rewards), dtype=torch.float32, device=rewards.device)
        masked_rewards = rewards * overlong_mask
        valid_k = overlong_mask.sum()
        baseline = (masked_rewards.sum() - masked_rewards) / torch.clamp(valid_k - 1, min=1.0)
        return (rewards - baseline) * overlong_mask

    def _process_entropy(self, advantages: torch.Tensor, group: list["RolloutState"]) -> torch.Tensor:
        sum_entropy: torch.Tensor | None = None
        total_tokens = 0
        for d in group:
            raw_logprobs = d.logprobs
            logprobs = (
                raw_logprobs
                if isinstance(raw_logprobs, torch.Tensor)
                else torch.tensor(raw_logprobs, dtype=torch.float32)
            )
            entropy = -(logprobs).sum()
            sum_entropy = entropy if sum_entropy is None else sum_entropy + entropy
            total_tokens += len(cast(list, d.response_ids))

        avg_entropy = sum_entropy / max(total_tokens, 1) if sum_entropy is not None else torch.tensor(0.0)
        if isinstance(avg_entropy, float):
            avg_entropy = torch.tensor(avg_entropy, device=advantages.device)

        if avg_entropy > self.entropy_upper_bound:
            delta = (avg_entropy - self.entropy_upper_bound) / self.entropy_upper_bound
            s = torch.sigmoid(
                torch.tensor(-delta / max(self.tau_upper, 1e-8), device=advantages.device)
            ).item()
            entropy_coeff = self.coeff_min_upper + (1 - self.coeff_min_upper) * s / 0.5
            advantages = torch.where(advantages < 0, advantages * entropy_coeff, advantages)

        elif avg_entropy < self.entropy_lower_bound:
            delta = (self.entropy_lower_bound - avg_entropy) / self.entropy_lower_bound
            s = torch.sigmoid(
                torch.tensor(-delta / max(self.tau_lower, 1e-8), device=advantages.device)
            ).item()
            entropy_coeff = self.coeff_min_lower + (1 - self.coeff_min_lower) * s / 0.5
            advantages = torch.where(advantages > 0, advantages * entropy_coeff, advantages)
        return advantages

    def compute(self, rewards: torch.Tensor, group: list["RolloutState"]) -> torch.Tensor:
        advantages = self._compute_advantages(rewards, group)
        return self._process_entropy(advantages, group)

    def __repr__(self) -> str:
        return "OverlongRLOOGroupEntropyEstimator()"

class OverlongRLOOGroupEntropyAdvantageConfig(BaseAdvantageConfig):
    """Configuration for :class:`~intern_s1_delivery.advantage.rloo_entropy.OverlongRLOOGroupEntropyEstimator`."""

    entropy_upper_bound: float = 0.65
    entropy_lower_bound: float = 0.4
    tau_upper: float = 0.0
    tau_lower: float = 0.0
    coeff_min_upper: float = 0.2
    coeff_min_lower: float = 0.5
    overlong_filer: bool = True

    def build(self) -> OverlongRLOOGroupEntropyEstimator:
        return OverlongRLOOGroupEntropyEstimator(
            entropy_upper_bound=self.entropy_upper_bound,
            entropy_lower_bound=self.entropy_lower_bound,
            tau_upper=self.tau_upper,
            tau_lower=self.tau_lower,
            coeff_min_upper=self.coeff_min_upper,
            coeff_min_lower=self.coeff_min_lower,
            overlong_filer=self.overlong_filer,
        )
