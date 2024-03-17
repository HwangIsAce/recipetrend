from typing import Optional, Tuple

import torch

from dataclasses import dataclass

from transformers.utils import ModelOutput

@dataclass
class CustomBaseModelOutputWithPoolingAndCrossAttentions(ModelOutput):
    output_aug_l1: torch.FloatTensor = None
    output_aug_l2: torch.FloatTensor = None
    output_aug_h1: torch.FloatTensor = None
    output_aug_h2: torch.FloatTensor = None
    output_aug_b1: torch.FloatTensor = None
    output_aug_b2: torch.FloatTensor = None
    # last_hidden_state: torch.FloatTensor = None
    logits: torch.FloatTensor = None
    labels: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None