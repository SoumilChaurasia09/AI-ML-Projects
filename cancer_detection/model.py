import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class CancerCNN(nn.Module):
    """
    Modified ResNet-18 for 28x28 medical image classification.
    Instead of the standard ResNet-18 first layer (7x7 conv with stride 2 and maxpool),
    we use a 3x3 conv with stride 1 and omit the maxpool. This prevents aggressive 
    resolution downsampling, retaining spatial information in 28x28 inputs.
    """
    def __init__(self, in_channels=3, num_classes=2, pretrained=False):
        super(CancerCNN, self).__init__()
        self.in_planes = 64

        # Adjusted first conv layer for 28x28 inputs (replaces 7x7 stride 2 + maxpool)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Standard ResNet layers: BasicBlock counts are [2, 2, 2, 2]
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)  # Output: 14x14
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)  # Output: 7x7
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)  # Output: 4x4
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * BasicBlock.expansion, num_classes)

        if pretrained:
            self.load_pretrained_weights()

    def load_pretrained_weights(self):
        import torchvision.models as models
        try:
            # Try loading via modern weights argument
            pretrained_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        except Exception:
            # Fallback for older torchvision versions
            pretrained_model = models.resnet18(pretrained=True)
            
        pretrained_dict = pretrained_model.state_dict()
        model_dict = self.state_dict()
        
        matched_dict = {}
        for k, v in pretrained_dict.items():
            if k in model_dict:
                # Skip first conv layer and classifier head due to shape differences
                if "conv1" not in k and "fc" not in k and "linear" not in k:
                    if model_dict[k].shape == v.shape:
                        matched_dict[k] = v
                        
        model_dict.update(matched_dict)
        self.load_state_dict(model_dict)
        print(f"Transfer Learning: Loaded {len(matched_dict)} pretrained ResNet-18 tensors.")


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

if __name__ == "__main__":
    # Sanity check with 28x28 inputs
    model = CancerCNN(in_channels=1, num_classes=2)
    x = torch.randn(2, 1, 28, 28)
    out = model(x)
    print("ResNet-18 upgrade verification: SUCCESS. Output shape:", out.shape)
