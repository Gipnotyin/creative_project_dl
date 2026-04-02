from __future__ import annotations

from typing import Dict

import timm
import torch
import torch.nn as nn


class SimilisMultiTaskModel(nn.Module):
    def __init__(self, backbone_name: str, num_classes: Dict[str, int], pretrained: bool = True):
        super().__init__()

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )

        if hasattr(self.backbone, "num_features"):
            in_features = self.backbone.num_features
        else:
            raise ValueError(f"Cannot infer num_features for backbone: {backbone_name}")

        # Безопасные внутренние имена модулей
        self.field_to_head_name = {field: f"{field}_head" for field in num_classes.keys()}

        self.heads = nn.ModuleDict(
            {
                self.field_to_head_name[field]: nn.Linear(in_features, n_classes)
                for field, n_classes in num_classes.items()
            }
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.backbone(x)

        outputs = {}
        for field, head_name in self.field_to_head_name.items():
            outputs[field] = self.heads[head_name](features)

        return outputs
