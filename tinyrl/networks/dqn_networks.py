import torch
import torch.nn as nn
from gymnasium.spaces import Box

import numpy as np

from tinyrl.networks.base import Critic



class QNetwork(Critic):
    """
    Deep Q-Network described in the deepmind paper: "Playing Atari with Deep Reinforcement Learning".

    References:
    ---------- 
    Human-level control through deep reinforcement learning; Mnih, et al. (2015).
    https://www.nature.com/articles/nature14236
    """ 
    def __init__(self, state_ch: int, action_dim: int):
        super().__init__()

        self.state_ch = state_ch
        self.action_dim = action_dim

        self.q = nn.Sequential(
            # [batch, state_ch, 84, 84] -> [batch, 32, 18, 18] 
            nn.Conv2d(in_channels=state_ch, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(True),
            
            # [batch, 32, 18, 18] -> [batch, 64, 9, 9]
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(True),
            
            # [batch, 64, 9, 9] -> [batch, 64, 7, 7]
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(True),
            
            # [batch, 64, 7, 7] -> [batch, 64 * 7 * 7] 
            nn.Flatten(start_dim=1),
            
            # [batch, 64 * 7 * 7] -> [batch, 512] 
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(True),
            
            # [batch, 512] -> [batch, action_dim] 
            nn.Linear(512, action_dim)
        ) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:            # [C, H, W], else [batch, C, H, W]
            x = x.unsqueeze(0)      # [1, C, H, W]
        return self.q(x)

    def predict(self, x: torch.Tensor | np.ndarray) -> tuple[int, float]:
        if isinstance(x, np.ndarray): 
            x = torch.from_numpy(x).to(torch.float32).div_(255.0) 
        if x.dim() == 3:
            x = x.unsqueeze(0)
        q_values = self.q(x).squeeze(0)             # [action_dim,]
        q_max = q_values.max()                      # [1,]
        action = torch.argmax(q_values, dim=-1)     # [1,]
        return (int(action.item()), float(q_max.item()))
