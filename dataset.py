"""
dataset.py — Data Loading & Preprocessing
==========================================
This file does two things:
  1. Reads your bridge_images table from MySQL
  2. Prepares each image for the model (resize, normalize, augment)

WHY DO WE PREPROCESS IMAGES?
  Neural networks don't understand raw pixels well unless they're standardized.
  EfficientNet-B3 was trained on images that were:
    - Resized to 300×300 pixels
    - Normalized with specific mean/std values (ImageNet statistics)
  If we feed it different-sized or unnormalized images, accuracy drops badly.

WHY DATA AUGMENTATION?
  Your dataset has a finite number of images. Augmentation creates new
  "virtual" versions of each image (flipped, rotated, color-adjusted).
  This teaches the model to recognize cracks regardless of the camera angle,
  lighting, or orientation — making it more robust in the real world.

WHY RULE-BASED SEVERITY/TIMELINE?
  Your MySQL table only has: image_path, structure_type, condition_type.
  It does NOT have severity scores or days_to_failure — a civil engineer
  would need to add those. For now, we use a rule table based on the
  structure type + condition. Replace these with real expert labels later.
"""

import os
import mysql.connector
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split


# ── Image size ────────────────────────────────────────────────────────────────
# EfficientNet-B3 was designed for 300×300 input. Using other sizes
# can work but 300×300 is optimal for this model.
IMAGE_SIZE = 300

# ImageNet normalization constants — these are fixed values from how
# EfficientNet was originally trained. Do not change these.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ── TRAINING transforms (used during training only) ───────────────────────────
# These augmentations randomly modify each image every time it's loaded.
# The model sees a "different" version of each image each epoch.
TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    # Flip horizontally 50% of the time — cracks look the same mirrored
    transforms.RandomHorizontalFlip(),

    # Flip vertically 50% of the time — useful for overhead/floor shots
    transforms.RandomVerticalFlip(),

    # Rotate up to ±15 degrees — cameras aren't always perfectly level
    transforms.RandomRotation(15),

    # Adjust brightness/contrast/saturation — simulate different lighting
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),

    # Convert PIL image to PyTorch tensor (values 0.0 to 1.0)
    transforms.ToTensor(),

    # Normalize using ImageNet statistics so the model sees familiar value ranges
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# ── VALIDATION transforms (used during testing/inference only) ────────────────
# NO augmentation here — we want consistent, reproducible results.
# Just resize and normalize.
VAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ── Label encoding ────────────────────────────────────────────────────────────
# Neural networks work with numbers, not strings.
# These maps convert your database text values to integers.

CONDITION_MAP = {
    "non-cracked": 0,
    "cracked":     1,
}

STRUCTURE_MAP = {
    "deck":     0,
    "pavement": 1,
    "wall":     2,
}

# ── Rule-based severity + days_to_failure ─────────────────────────────────────
# Since your DB doesn't have these columns, we derive them from
# structure_type + condition_type using engineering judgment.
#
# Format: (structure, condition) → (severity_score, days_to_failure)
# severity_score: 0 = destroyed, 100 = perfect condition
# days_to_failure: how many days until urgent intervention is needed
#
# TODO: Replace these with real expert assessments when available.
SEVERITY_RULES = {
    ("deck",     "cracked"):     (35, 60),    # deck cracks = highest risk
    ("deck",     "non-cracked"): (85, 730),
    ("pavement", "cracked"):     (50, 150),
    ("pavement", "non-cracked"): (90, 1095),
    ("wall",     "cracked"):     (45, 120),
    ("wall",     "non-cracked"): (88, 900),
}


# ── MySQL data loader ─────────────────────────────────────────────────────────

def load_from_mysql(host: str     = "localhost",
                    user: str     = "root",
                    password: str = None,
                    database: str = "bridge_db") -> pd.DataFrame:
    """
    Connects to MySQL, reads the bridge_images table, and returns a DataFrame.

    The password is read from the DB_PASSWORD environment variable if not
    passed directly — this keeps secrets out of the code.
    """
    password = password or os.environ.get("DB_PASSWORD", "")

    print(f"Connecting to MySQL at {host}...")
    conn = mysql.connector.connect(
        host=host, user=user, password=password, database=database
    )

    df = pd.read_sql(
        "SELECT image_path, structure_type, condition_type FROM bridge_images",
        conn
    )
    conn.close()

    # Normalize to lowercase and remove whitespace
    # (MySQL values might be "Cracked" or "cracked " — we handle both)
    df["structure_type"] = df["structure_type"].str.lower().str.strip()
    df["condition_type"] = df["condition_type"].str.lower().str.strip()

    # Normalize condition to use hyphens (match our CONDITION_MAP keys)
    df["condition_type"] = df["condition_type"].str.replace(" ", "-")

    # Derive severity and days_to_failure from the rule table
    df["severity_score"]  = df.apply(
        lambda r: SEVERITY_RULES.get(
            (r.structure_type, r.condition_type), (50, 365)
        )[0], axis=1
    )
    df["days_to_failure"] = df.apply(
        lambda r: SEVERITY_RULES.get(
            (r.structure_type, r.condition_type), (50, 365)
        )[1], axis=1
    )

    print(f"Loaded {len(df):,} records from MySQL.")
    print(df[["structure_type", "condition_type"]].value_counts().to_string())
    return df


# ── PyTorch Dataset ───────────────────────────────────────────────────────────

class BridgeDataset(Dataset):
    """
    A PyTorch Dataset wraps our data so PyTorch's DataLoader can
    efficiently batch, shuffle, and parallelize loading.

    __len__     → how many images are in this dataset?
    __getitem__ → give me image number N and its labels
    """

    def __init__(self, df: pd.DataFrame,
                 transform=None,
                 image_root: str = ""):
        """
        Args:
            df         : DataFrame with columns from load_from_mysql()
            transform  : image preprocessing pipeline (train or val)
            image_root : prefix path if image_path in DB is relative
                         e.g. image_root="archive" + image_path="deck/Cracked/img1.jpg"
                         → opens "archive/deck/Cracked/img1.jpg"
        """
        self.df         = df.reset_index(drop=True)
        self.transform  = transform or VAL_TRANSFORMS
        self.image_root = image_root

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        # Build full file path and open the image
        img_path = os.path.join(self.image_root, row["image_path"])
        image    = Image.open(img_path).convert("RGB")
        # .convert("RGB") ensures we always get 3 channels,
        # even if the image is grayscale or has an alpha channel.

        # Apply preprocessing (resize, normalize, augment)
        if self.transform:
            image = self.transform(image)

        # Convert labels from strings to numbers
        condition = CONDITION_MAP.get(row["condition_type"], 0)
        structure = STRUCTURE_MAP.get(row["structure_type"], 0)
        severity  = float(row["severity_score"])
        days      = float(row["days_to_failure"])

        # Return a dictionary — the training loop will unpack these
        return {
            "image":           image,
            # torch.long = integer tensor (required by CrossEntropyLoss)
            "damage_label":    torch.tensor(condition, dtype=torch.long),
            "structure_label": torch.tensor(structure, dtype=torch.long),
            # torch.float32 = decimal tensor (required by MSELoss)
            "severity":        torch.tensor([severity], dtype=torch.float32),
            "days_to_failure": torch.tensor([days],     dtype=torch.float32),
        }


# ── DataLoader factory ────────────────────────────────────────────────────────

def get_dataloaders(db_config: dict,
                    image_root:  str   = "",
                    batch_size:  int   = 32,
                    num_workers: int   = 4,
                    val_split:   float = 0.2):
    """
    Full pipeline: MySQL → DataFrame → split → Dataset → DataLoader

    WHAT IS A DATALOADER?
    PyTorch's DataLoader batches your data and loads it in parallel.
    Instead of loading images one by one (slow), it loads batch_size=32
    images at a time using num_workers=4 background threads.

    TRAIN/VAL SPLIT:
    We split 80% for training, 20% for validation.
    stratify=condition_type ensures both splits have the same ratio of
    cracked/non-cracked images (otherwise you might get all non-cracked
    in the val set by accident).
    """
    df = load_from_mysql(**db_config)

    train_df, val_df = train_test_split(
        df,
        test_size=val_split,
        stratify=df["condition_type"],  # keep class balance in both splits
        random_state=42                 # fixed seed = reproducible split
    )

    print(f"Train: {len(train_df):,} images | Val: {len(val_df):,} images")

    train_dataset = BridgeDataset(train_df, transform=TRAIN_TRANSFORMS, image_root=image_root)
    val_dataset   = BridgeDataset(val_df,   transform=VAL_TRANSFORMS,   image_root=image_root)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,          # shuffle every epoch so model doesn't memorize order
        num_workers=num_workers,
        pin_memory=True,       # faster GPU transfer
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,         # no need to shuffle validation
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
