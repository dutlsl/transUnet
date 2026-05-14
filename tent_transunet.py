import torch
import torch.nn as nn

class TentTransUNet(nn.Module):
    def __init__(self, model, optimizer):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.model = self._configure_model(self.model)

    def _configure_model(self, model):
        """TransUNet용 TENT 로직: BatchNorm, LayerNorm, GroupNorm 타겟팅"""
        model.train()
        model.requires_grad_(False) # 먼저 모든 파라미터를 프리징

        for m in model.modules():
            # U-Mamba(InstanceNorm)와 달리 TransUNet은 아래 3개 레이어를 사용합니다.
            if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                m.requires_grad_(True)
                if hasattr(m, 'weight') and m.weight is not None:
                    m.weight.requires_grad_(True)
                if hasattr(m, 'bias') and m.bias is not None:
                    m.bias.requires_grad_(True)
        return model

    @torch.jit.export
    def softmax_entropy(self, x: torch.Tensor) -> torch.Tensor:
        """엔트로피 계산 함수"""
        return -(x.softmax(1) * x.log_softmax(1)).sum(1)

    def forward(self, x):
        logits = self.model(x)
        loss = self.softmax_entropy(logits).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return logits