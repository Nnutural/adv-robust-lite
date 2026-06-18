from __future__ import annotations


def preact_resnet18(num_classes: int = 10):
    import torch.nn as nn

    class PreActBasicBlock(nn.Module):
        expansion = 1

        def __init__(self, in_planes: int, planes: int, stride: int = 1):
            super().__init__()
            self.bn1 = nn.BatchNorm2d(in_planes)
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.relu = nn.ReLU(inplace=True)
            if stride != 1 or in_planes != planes * self.expansion:
                self.shortcut = nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=False)
            else:
                self.shortcut = nn.Identity()

        def forward(self, x):
            out = self.relu(self.bn1(x))
            shortcut = self.shortcut(out) if not isinstance(self.shortcut, nn.Identity) else x
            out = self.conv1(out)
            out = self.relu(self.bn2(out))
            out = self.conv2(out)
            return out + shortcut

    class PreActResNet(nn.Module):
        def __init__(self, block, num_blocks, num_classes: int = 10):
            super().__init__()
            self.in_planes = 64
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
            self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
            self.bn_final = nn.BatchNorm2d(512 * block.expansion)
            self.relu = nn.ReLU(inplace=True)
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        def _make_layer(self, block, planes: int, num_blocks: int, stride: int):
            strides = [stride] + [1] * (num_blocks - 1)
            layers = []
            for s in strides:
                layers.append(block(self.in_planes, planes, s))
                self.in_planes = planes * block.expansion
            return nn.Sequential(*layers)

        def forward(self, x):
            out = self.conv1(x)
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = self.relu(self.bn_final(out))
            out = self.avgpool(out).flatten(1)
            return self.fc(out)

    return PreActResNet(PreActBasicBlock, [2, 2, 2, 2], num_classes=num_classes)
