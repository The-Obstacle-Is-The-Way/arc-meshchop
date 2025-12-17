# Training Configuration & Hyperparameters

> Reference: "State-of-the-Art Stroke Lesion Segmentation at 1/1000th of Parameters"

## Training Infrastructure

| Component | Specification |
|-----------|---------------|
| Framework | PyTorch |
| Configuration | Typer CLI |
| GPU | NVIDIA A100 (80 GB) in paper |
| Precision | FP16 on CUDA, FP32 on MPS/CPU |
| Memory management | Layer checkpointing (for large models) |

---

## Baseline Model Configuration

These settings are used for **all baseline models** (U-MAMBA, MedNeXt, SegResNet, etc.):

### Optimizer: AdamW

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001,           # Learning rate
    weight_decay=3e-5,  # L2 regularization
    eps=1e-4            # Numerical stability
)
```

| Parameter | Value | Notes |
|-----------|-------|-------|
| Learning rate | 0.001 | Maximum LR for OneCycleLR |
| Weight decay | 3e-5 | Regularization strength |
| Epsilon | 1e-4 | Higher than default (1e-8) for FP16 stability |
| Batch size | 1 | Limited by 256^3 volume size |
| Epochs | 50 | |

### Learning Rate Scheduler: OneCycleLR

```python
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.001,
    total_steps=num_epochs * steps_per_epoch,
    pct_start=0.01,  # 1% warmup
)
```

**OneCycleLR Behavior:**
1. Start at `1/100th of max LR` = 0.00001
2. Warmup: "quickly increases to the maximum" over first 1% of training
3. Annealing: "gradually decreases" back to minimum (cosine by default)

---

## MeshNet-Specific Configuration

MeshNet uses **hyperparameter optimization** rather than fixed values.

### Hyperparameter Search Framework

**Tool:** Optuna with `SuccessiveHalvingPruner` (ASHA)
**Algorithm:** ASHA (Asynchronous Successive Halving Algorithm)

**ASHA Behavior:**
- Initially evaluates many configurations with few epochs
- Promotes promising configurations to higher fidelity (more epochs)
- Efficiently prunes poor configurations early

### Search Space

| Hyperparameter | Distribution | Range |
|----------------|--------------|-------|
| Channels (X) | Uniform integer | [5, 21] |
| Learning rate | Log-uniform | [1e-4, 4e-2] |
| Weight decay | Log-uniform | [1e-4, 4e-2] |
| Background weight | Uniform | [0, 1] |
| Warmup percentage | Categorical | {0.02, 0.1, 0.2} |
| Epochs (fidelity) | Fidelity | [15, 50] |

### MeshNet Training with Restarts

**10 restarts per configuration:**
- Train each configuration 10 times with different random seeds
- Select best model based on validation performance
- Accounts for training variance

---

## Loss Function

### Cross-Entropy with Class Weighting

```python
loss_fn = torch.nn.CrossEntropyLoss(
    weight=torch.tensor([0.5, 1.0]),  # [background, lesion]
    label_smoothing=0.01
)
```

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Background weight | 0.5 | Down-weight dominant class |
| Lesion weight | 1.0 | Full weight for minority class |
| Label smoothing | 0.01 | Regularization, prevents overconfidence |

### Why Class Weighting?

Stroke lesions typically occupy a small fraction of brain volume:
- **Class imbalance:** ~90-99% background, ~1-10% lesion
- Without weighting, model would predict "all background" for easy accuracy
- Higher lesion weight forces model to learn lesion features

---

## Precision and Memory Management

### Mixed-Precision Training

Cross-platform `torch.amp` (works on CUDA, MPS, CPU):

```python
from torch.amp import autocast, GradScaler

device_type = "cuda"  # or "mps" or "cpu"
scaler = GradScaler(device_type) if device_type == "cuda" else None

with autocast(device_type=device_type, enabled=(device_type == "cuda")):
    output = model(input)
    loss = loss_fn(output, target)

if scaler:
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
else:
    loss.backward()
    optimizer.step()
```

**Platform behavior:**
- **CUDA:** FP16 with GradScaler (2x memory reduction, tensor cores)
- **MPS (Apple Silicon):** FP32 (MPS doesn't benefit from FP16 for this workload)
- **CPU:** FP32 fallback

---

## Training Loop Pseudocode

```python
model = MeshNet(channels=26, num_classes=2)
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=3e-5, eps=1e-4)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=0.001, total_steps=50*len(train_loader), pct_start=0.01
)
criterion = torch.nn.CrossEntropyLoss(
    weight=torch.tensor([0.5, 1.0]).to(device), label_smoothing=0.01
)

# Cross-platform AMP setup
device_type = "cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu"
)
use_amp = device_type == "cuda"
scaler = torch.amp.GradScaler(device_type) if use_amp else None

for epoch in range(50):
    model.train()
    for batch in train_loader:
        images, masks = batch
        images = images.to(device)
        masks = masks.to(device).long()

        optimizer.zero_grad()

        with torch.amp.autocast(device_type=device_type, enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, masks.squeeze(1))

        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        scheduler.step()
```

---

## Key Training Considerations

### What the Paper Does

1. **Baselines:** Fixed hyperparameters, single training run
2. **MeshNet:** Hyperparameter search on inner folds of first outer fold only
3. **Final MeshNet:** Use optimized hyperparameters across all outer folds
4. **Reporting:** 10 restarts, report mean +/- std of best restart per fold

### Reproducibility Notes

- Set random seeds for reproducibility
- Save model checkpoints at best validation DICE
- Log all hyperparameters (we use Typer CLI args)
- Document exact preprocessing steps

---

## Hardware Requirements

### From Paper

| Resource | Specification |
|----------|---------------|
| GPU | NVIDIA A100 (80 GB) |

### Estimated Minimums (Not in Paper)

| Resource | Estimated Minimum | Rationale |
|----------|-------------------|-----------|
| GPU VRAM | ~4-8 GB (FP16) | MeshNet-26 has only 147K params |
| System RAM | 32+ GB | For data loading |
| Storage | 50+ GB | ARC dataset + checkpoints |

See [TRAIN-001](../issues/TRAIN-001-runtime-estimates.md) for detailed runtime estimates by hardware.

---

## References

- AdamW: Loshchilov & Hutter, "Decoupled Weight Decay Regularization"
- OneCycleLR: Smith & Topin, "Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates"
- ASHA: Li et al., "A System for Massively Parallel Hyperparameter Tuning"
