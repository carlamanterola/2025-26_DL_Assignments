import torch
import torch.nn as nn
import torchvision.models as models


# ─────────────────────────────────────────────
#  Model 1 — CNN from Scratch (AlexNet-like)
# ─────────────────────────────────────────────
class ScratchCNN(nn.Module):
    """
    Simple AlexNet-inspired CNN trained from scratch on CIFAR-10.
    CIFAR-10 images are 32x32, so we use smaller kernels/strides than
    the original AlexNet which was designed for 224x224.
    """
    def __init__(self, num_classes=10):
        super(ScratchCNN, self).__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),   # 32x32 → 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),          # 32x32 → 16x16

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 16x16 → 16x16
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),          # 16x16 → 8x8

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1), # 8x8 → 8x8
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # Block 4
            nn.Conv2d(256, 256, kernel_size=3, padding=1), # 8x8 → 8x8
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),          # 8x8 → 4x4
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)   # flatten
        x = self.classifier(x)
        return x


# ─────────────────────────────────────────────
#  Model 2 — VGG16 Transfer Learning
# ─────────────────────────────────────────────
def get_vgg_transfer(num_classes=10, freeze_features=True):
    """
    VGG16 pre-trained on ImageNet, adapted for CIFAR-10.

    WHY freeze_features=True at first:
        The convolutional backbone already has rich ImageNet features.
        We first train only the new classifier head (fast convergence).
        Then optionally unfreeze for fine-tuning (see train_part2.py).

    PyTorch transfer learning mechanics:
        - weights='IMAGENET1K_V1' loads ImageNet pre-trained weights
        - param.requires_grad = False prevents those layers from updating
        - Only the new classifier head (requires_grad=True by default) trains
        - The optimizer only receives parameters where requires_grad=True
    """
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

    # Freeze all convolutional feature layers
    if freeze_features:
        for param in model.features.parameters():
            param.requires_grad = False

    # Replace the classifier head
    # Original VGG16 classifier expects 25088 input features (for 224x224 input)
    # We upsample CIFAR-10 images to 224x224 in the transforms (see train_part2.py)
    model.classifier = nn.Sequential(
        nn.Linear(25088, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(4096, 1024),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(1024, num_classes),   # 10 classes for CIFAR-10
    )

    return model


def unfreeze_vgg_top_layers(model, n_blocks=1):
    """
    Unfreeze the last n convolutional blocks of VGG16 for fine-tuning.

    WHY: After the head converges, unfreezing the top conv layers lets the
    network adapt low-level features closer to the target domain (CIFAR-10),
    improving accuracy at the cost of more compute.

    VGG16 features layout (by index):
        0-4   → Block 1 (conv1_1, relu, conv1_2, relu, maxpool)
        5-9   → Block 2
        10-16 → Block 3
        17-23 → Block 4
        24-30 → Block 5  ← unfreeze this first
    """
    blocks = {
        1: range(24, 31),  # Block 5
        2: range(17, 31),  # Blocks 4+5
        3: range(10, 31),  # Blocks 3+4+5
    }
    n_blocks = min(n_blocks, 3)
    for idx in blocks[n_blocks]:
        for param in model.features[idx].parameters():
            param.requires_grad = True
    print(f"Unfroze top {n_blocks} VGG block(s) for fine-tuning.")
    return model