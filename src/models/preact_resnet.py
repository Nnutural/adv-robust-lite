from __future__ import annotations


def preact_resnet18(num_classes: int = 10):
    import torch.nn as nn

    from torchvision.models.resnet import BasicBlock, ResNet

    class PreActBasicBlock(BasicBlock):
        def forward(self, x):
            identity = x
            out = self.bn1(x)
            out = self.relu(out)
            if self.downsample is not None:
                identity = self.downsample(out)
            out = self.conv1(out)
            out = self.bn2(out)
            out = self.relu(out)
            out = self.conv2(out)
            return out + identity

    model = ResNet(PreActBasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model
