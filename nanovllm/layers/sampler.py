import torch
from torch import nn


class Sampler(nn.Module):

    @torch.compile
    def _greedy(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.argmax(dim=-1)

    @torch.compile
    def _sample(self, logits: torch.Tensor, temperatures: torch.Tensor) -> torch.Tensor:
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
        return probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)

    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor) -> torch.Tensor:
        if temperatures.max().item() < 1e-7:
            return self._greedy(logits)
        return self._sample(logits, temperatures)
