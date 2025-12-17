# Training Configuration & Hyperparameters

> Extracted from: "State-of-the-Art Stroke Lesion Segmentation at 1/1000th of Parameters"

## Training Infrastructure

| Component | Specification |
|-----------|---------------|
| Framework | PyTorch |
| Configuration | Typer CLI (paper used Hydra) |
| GPU | NVIDIA A100 (80 GB) |
| Precision | Half-precision (FP16) on CUDA, FP32 on MPS/CPU |
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
| Batch size | 1 | Limited by 256³ volume size |
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

**OneCycleLR Behavior (from paper):**
1. Start at `1/100th of max LR` = 0.00001
2. Warmup: "quickly increases to the maximum" over first 1% of training
3. Annealing: "gradually decreases" back to minimum

```
LR Schedule:
        ↗ peak (0.001)
       /  \
      /    \
     /      \
    /        \_____ final
start (0.00001)
|←1%→|←----- gradual decrease -----→|
```

> **Note:** The paper says the LR "gradually decreases" but does not specify the annealing strategy. PyTorch's OneCycleLR uses cosine annealing by default.

---

## MeshNet-Specific Configuration

MeshNet uses **hyperparameter optimization** rather than fixed values.

### Hyperparameter Search Framework

**Tool:** Orion (asynchronous black-box optimization)
**Algorithm:** ASHA (Asynchronous Successive Halving Algorithm)

> **Implementation Note:** Our implementation uses **Optuna** with `SuccessiveHalvingPruner` (ASHA) instead of Orion, for Python 3.12+ compatibility. The ASHA algorithm behavior is identical.

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

**Note on fidelity:** ASHA uses epochs as fidelity - poor configs are stopped early at 15 epochs, promising ones continue to 50.

### Optimized Values

> **⚠️ NOT DISCLOSED:** The paper does not disclose the actual optimized hyperparameter values found by the HPO search. Only the search space is provided. The final values would need to be determined by running the optimization or obtained from the authors.

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

### Label Smoothing

Instead of hard targets [0, 1], uses soft targets [0.005, 0.995]:
- Prevents overconfident predictions
- Acts as regularization
- Improves calibration

---

## Precision and Memory Management

### Mixed-Precision Training

Our implementation uses cross-platform `torch.amp` (not CUDA-only `torch.cuda.amp`):

```python
# Cross-platform AMP (works on CUDA, MPS, CPU)
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

**Cautions:**
- GradScaler only used on CUDA (prevents underflow)
- Some operations may need FP32 (BatchNorm accumulation)

### Layer Checkpointing (Gradient Checkpointing)

```python
from torch.utils.checkpoint import checkpoint

# Instead of:
x = layer1(x)
x = layer2(x)

# Use:
x = checkpoint(layer1, x)
x = checkpoint(layer2, x)
```

**Trade-off:**
- Saves memory by not storing all activations
- Increases computation (recomputes activations during backward)
- Used for models that don't fit in GPU memory

---

## Training Loop Pseudocode

```python
# Cross-platform device setup (must be defined first)
device_type = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device(device_type)
use_amp = device_type == "cuda"
scaler = torch.amp.GradScaler(device_type) if use_amp else None

model = MeshNet(channels=26, num_classes=2)
model = model.to(device)  # Use .to(device) instead of .cuda().half() - autocast handles precision

optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=3e-5, eps=1e-4)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, total_steps=50*len(train_loader), pct_start=0.01)
criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.5, 1.0]).to(device), label_smoothing=0.01)

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

    # Validation
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            # Compute validation metrics
            pass
```

---

## Hyperparameter Search Code Structure

```python
from orion.client import create_experiment

experiment = create_experiment(
    name='meshnet_hpo',
    space={
        'channels': 'uniform(5, 21, discrete=True)',
        'lr': 'loguniform(1e-4, 4e-2)',
        'weight_decay': 'loguniform(1e-4, 4e-2)',
        'bg_weight': 'uniform(0, 1)',
        'warmup_pct': 'choices([0.02, 0.1, 0.2])',
    },
    algorithms={
        'asha': {
            'seed': 42,
            'num_rungs': 4,
            'num_brackets': 1,
        }
    }
)

while not experiment.is_done:
    trial = experiment.suggest()
    if trial is None:
        break

    # Train model with trial.params
    dice_score = train_and_evaluate(
        channels=trial.params['channels'],
        lr=trial.params['lr'],
        weight_decay=trial.params['weight_decay'],
        bg_weight=trial.params['bg_weight'],
        warmup_pct=trial.params['warmup_pct'],
    )

    experiment.observe(trial, [{'name': 'dice', 'type': 'objective', 'value': -dice_score}])  # Minimize negative DICE
```

---

## Key Training Considerations

### What the Paper Does

1. **Baselines:** Fixed hyperparameters, single training run
2. **MeshNet:** Hyperparameter search on inner folds of first outer fold only
3. **Final MeshNet:** Use optimized hyperparameters across all outer folds
4. **Reporting:** 10 restarts, report mean ± std of best restart per fold

### Reproducibility Notes

- Set random seeds for reproducibility
- Save model checkpoints at best validation DICE
- Log all hyperparameters (we use Typer CLI args; paper used Hydra)
- Document exact preprocessing steps

---

## Hardware Requirements

### From Paper

| Resource | Specification |
|----------|---------------|
| GPU | NVIDIA A100 (80 GB) |

> **Note:** The paper states "Each experiment was conducted on a single NVIDIA A100 GPU with 80 GB memory."

### Estimated Minimums (Not in Paper)

The following are reasonable estimates for MeshNet-only reproduction, **not stated in the paper**:

| Resource | Estimated Minimum | Rationale |
|----------|-------------------|-----------|
| GPU VRAM | ~4-8 GB (FP16) | MeshNet-26 has only 147K params |
| System RAM | 32+ GB | For data loading |
| Storage | 50+ GB | ARC dataset + checkpoints |

**Note:** Larger baseline models (MedNeXt, U-MAMBA) require layer checkpointing even on A100 80GB.

---

## References

- AdamW: Loshchilov & Hutter, "Decoupled Weight Decay Regularization"
- OneCycleLR: Smith & Topin, "Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates"
- ASHA: Li et al., "A System for Massively Parallel Hyperparameter Tuning"
- Orion: Bouthillier et al., "Epistimio/orion: Asynchronous Distributed Hyperparameter Optimization"
