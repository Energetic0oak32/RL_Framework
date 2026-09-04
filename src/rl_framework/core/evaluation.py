from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean, pstdev


@dataclass(slots=True)
class EvaluationResult:
    """
    Resultado agregado de uma avaliação.
    """

    episode_rewards: list[float]
    episode_lengths: list[int]

    @property
    def episodes(self) -> int:
        return len(
            self.episode_rewards
        )

    @property
    def mean_reward(self) -> float:
        return fmean(
            self.episode_rewards
        )

    @property
    def reward_std(self) -> float:

        if len(self.episode_rewards) < 2:
            return 0.0

        return pstdev(
            self.episode_rewards
        )

    @property
    def mean_episode_length(self) -> float:
        return fmean(
            self.episode_lengths
        )