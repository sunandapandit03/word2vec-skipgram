"""
model.py — The Brain
=====================
This file defines the neural network architecture.

WHAT IS A CNN?
  A CNN (Convolutional Neural Network) is a type of AI model that
  looks at images in layers — first detecting simple things like edges,
  then shapes, then complex patterns like "this is a crack" or "this is rust".

WHY EFFICIENTNET-B3?
  Instead of training from scratch (which needs millions of images),
  we use EfficientNet-B3 which was already trained on 1.2 million images
  by Google. It already "knows" what cracks, textures, and surfaces look like.
  We just teach it the final step: "and THIS kind of crack means danger".

THE 4-HEAD DESIGN:
  After the CNN extracts features from the image, we split into 4 branches
  (called "heads"), each answering a different question:

  Head 1 — Is there a crack? (yes/no)
  Head 2 — What type of structure? (deck / pavement / wall)
  Head 3 — How bad is it? (0-100 health score)
  Head 4 — How long until failure? (days, as a number)

  All 4 heads share the same "vision" from the CNN backbone.
  This is more efficient than training 4 separate models.
"""

import torch
import torch.nn as nn
from torchvision import models

# ── These are the labels your model will output ───────────────────────────────
DAMAGE_CLASSES    = ["non-cracked", "cracked"]       # Head 1 outputs
STRUCTURE_CLASSES = ["deck", "pavement", "wall"]      # Head 2 outputs
RISK_LEVELS       = ["low", "medium", "high", "critical"]  # computed after


class BridgeFailureModel(nn.Module):
    """
    The complete bridge failure prediction neural network.

    Input:  A 300×300 RGB image tensor
    Output: 4 predictions (damage, structure, severity, timeline)
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()

        # ── STEP 1: Load the EfficientNet-B3 backbone ─────────────────────────
        # pretrained=True means we download Google's pre-trained weights.
        # This gives us a massive head-start — the model already understands
        # visual patterns from 1.2M images before seeing a single bridge photo.
        weights  = models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
        backbone = models.efficientnet_b3(weights=weights)

        # We keep the feature extraction layers but remove EfficientNet's
        # original final classifier (it was trained for 1000 ImageNet classes,
        # we don't need that). We replace it with our own 4-head classifier.
        self.feature_extractor = nn.Sequential(
            backbone.features,   # the CNN layers that "see" the image
            backbone.avgpool,    # compresses spatial info into one vector
            nn.Flatten(),        # turns the 2D output into a 1D list of numbers
        )
        # After flattening, we get a vector of 1536 numbers per image.
        # These 1536 numbers encode everything the CNN learned about that image.
        feature_dim = 1536

        # ── STEP 2: Shared bottleneck layer ───────────────────────────────────
        # All 4 heads share this layer. It compresses 1536 features → 512.
        # Think of it like a "summary" — distilling the most important info
        # before each head uses it for its specific prediction.
        #
        # BatchNorm1d: normalizes the numbers so training is more stable
        # ReLU:        activation function — introduces non-linearity
        # Dropout(0.4): randomly zeros 40% of neurons during training
        #               → prevents the model from memorizing instead of learning
        self.shared = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
        )

        # ── STEP 3: Head 1 — Crack detection ──────────────────────────────────
        # Binary classification: is there a crack or not?
        # Output: 2 numbers (logits). The higher one is the prediction.
        # e.g. [0.2, 3.8] → cracked (index 1 is higher)
        self.damage_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2),  # 2 outputs: non-cracked=0, cracked=1
        )

        # ── STEP 4: Head 2 — Structure type ───────────────────────────────────
        # 3-class classification: deck, pavement, or wall?
        # Output: 3 numbers. Highest one = predicted structure type.
        self.structure_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 3),  # 3 outputs: deck=0, pavement=1, wall=2
        )

        # ── STEP 5: Head 3 — Severity score ───────────────────────────────────
        # Regression: outputs a health score between 0 and 100.
        # Sigmoid() squashes any number into the range [0, 1].
        # We then multiply by 100 in forward() to get [0, 100].
        # High score = healthy bridge. Low score = serious damage.
        self.severity_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid(),       # output range: 0.0 to 1.0
        )

        # ── STEP 6: Head 4 — Days to failure ──────────────────────────────────
        # Regression: outputs estimated days until critical failure.
        # Softplus() ensures the output is always positive (can't have
        # negative days). We train in log-space (log(days+1)) so the model
        # handles the wide range (1 day to 1000+ days) without huge loss values.
        self.timeline_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Softplus(),      # output: always positive, represents log(days+1)
        )

    def forward(self, x: torch.Tensor) -> dict:
        """
        The forward pass — what happens when an image goes through the model.

        Args:
            x: image tensor of shape (batch_size, 3, 300, 300)
               3 = RGB channels, 300x300 = image dimensions

        Returns:
            dict with 4 prediction tensors, one per head
        """
        # Pass image through the backbone → get 1536 feature numbers per image
        features = self.feature_extractor(x)       # shape: (batch, 1536)

        # Compress to 512 shared features
        shared = self.shared(features)              # shape: (batch, 512)

        # Each head uses the same shared features to make its prediction
        return {
            "damage_logits":    self.damage_head(shared),          # (batch, 2)
            "structure_logits": self.structure_head(shared),       # (batch, 3)
            "severity":         self.severity_head(shared) * 100,  # (batch, 1) → 0-100
            "timeline_log":     self.timeline_head(shared),        # (batch, 1) → log(days+1)
        }


# ── Helper functions ──────────────────────────────────────────────────────────

def build_model(pretrained: bool = True) -> BridgeFailureModel:
    """
    Creates the model and freezes the backbone.

    WHY FREEZE THE BACKBONE INITIALLY?
    The EfficientNet backbone already has excellent pre-trained weights.
    If we let it change too fast at the start, it "forgets" what it learned.
    So in Phase 1 of training, only the 4 heads learn.
    In Phase 2 (after epoch 10), we unfreeze the last few backbone layers
    for fine-tuning — gentle adjustments on top of what it already knows.
    """
    model = BridgeFailureModel(pretrained=pretrained)

    # Freeze ALL backbone parameters — only heads will train initially
    for param in model.feature_extractor.parameters():
        param.requires_grad = False

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model ready. Total params: {total:,} | Trainable: {trainable:,}")
    return model


def unfreeze_backbone(model: BridgeFailureModel, layers: int = 3):
    """
    Called at epoch 10 to start fine-tuning the backbone.
    We only unfreeze the LAST few blocks — the earlier layers detect
    basic edges and shapes which are universal and shouldn't change.
    The later layers detect higher-level features specific to our images.
    """
    blocks = list(model.feature_extractor[0].children())
    for block in blocks[-layers:]:
        for param in block.parameters():
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Unfrozen last {layers} backbone blocks. Trainable params now: {trainable:,}")
