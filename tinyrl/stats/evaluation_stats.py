from typing import List
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class EvluationStats:
    average_rewards_: List[float] = field(default_factory=list)
    std_rewards_: List[float] = field(default_factory=list)
    
    def update(self, rewards: np.ndarray) -> None:
        average_reward = float(np.mean(rewards).item())
        std_reward = float(np.std(rewards).item())
        self.average_rewards_.append(average_reward)
        self.std_rewards_.append(std_reward)

    def to_csv(self, path: str) -> None:
        data = {
            "average_rewards": self.average_rewards_,
            "std_rewards": self.std_rewards_,
        }
        pd.DataFrame.from_dict(data).to_csv(path, index=False)

    @property
    def last_average_reward(self) -> float:
        return self.average_rewards_[-1]
    
    @property
    def last_std_reward(self) -> float:
        return self.std_rewards_[-1]
    