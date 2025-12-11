# Training Configuration & Hyperparameters

> Extracted from: "State-of-the-Art Stroke Lesion Segmentation at 1/1000th of Parameters"

## Training Infrastructure

| Component | Specification |
|-----------|---------------|
| Framework | PyTorch |
| Configuration | Hydra |
| GPU | NVIDIA A100 (80 GB) |
| Precision | Half-precision (FP16) |
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
    anneal_strategy='cos'
)
```

**OneCycleLR Behavior:**
1. Start at `max_lr / 100` = 0.00001
2. Warmup: Increase to `max_lr` = 0.001 over first 1% of training
3. Annealing: Cosine decay back down to minimum

```
LR Schedule:
        ↗ peak (0.001)
       /  \
      /    \
     /      \
    /        \_____ final
start (0.00001)
|←1%→|←----- 99% cosine decay -----→|
```

---

## MeshNet-Specific Configuration

MeshNet uses **hyperparameter optimization** rather than fixed values.

### Hyperparameter Search Framework

**Tool:** Orion (asynchronous black-box optimization)
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

**Note on fidelity:** ASHA uses epochs as fidelity - poor configs are stopped early at 15 epochs, promising ones continue to 50.

### Optimized Values (Inferred from Results)

Based on the paper's best-performing MeshNet-26:

| Hyperparameter | Likely Optimal Value |
|----------------|---------------------|
| Channels | 26 |
| Learning rate | ~0.01-0.02 (higher than baselines) |
| Weight decay | ~0.01-0.02 |
| Background weight | ~0.5 |
| Warmup percentage | 0.02 or 0.1 |

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

### Half-Precision Training (FP16)

```python
scaler = torch.cuda.amp.GradScaler()

with torch.cuda.amp.autocast():
    output = model(input)
    loss = loss_fn(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**Benefits:**
- 2x memory reduction
- Faster computation on tensor cores
- Enables full 256³ volume training

**Cautions:**
- May need gradient scaling to prevent underflow
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
model = MeshNet(channels=26, num_classes=2)
model = model.cuda().half()  # FP16

optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=3e-5, eps=1e-4)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, total_steps=50*len(train_loader), pct_start=0.01)
criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.5, 1.0]).cuda(), label_smoothing=0.01)
scaler = torch.cuda.amp.GradScaler()

for epoch in range(50):
    model.train()
    for batch in train_loader:
        images, masks = batch
        images = images.cuda().half()
        masks = masks.cuda().long()

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, masks.squeeze(1))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
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
- Log all hyperparameters with Hydra/MLflow/W&B
- Document exact preprocessing steps

---

## Hardware Requirements

### Minimum (for MeshNet)

| Resource | Requirement |
|----------|-------------|
| GPU | 8 GB VRAM (FP16) |
| RAM | 32 GB |
| Storage | 50 GB (dataset + checkpoints) |

### Recommended (for baselines)

| Resource | Requirement |
|----------|-------------|
| GPU | 80 GB VRAM (A100) or multiple GPUs |
| RAM | 64 GB |
| Storage | 100 GB |

**Note:** Larger models (MedNeXt, U-MAMBA) require layer checkpointing even on A100 80GB.

---

## References

- AdamW: Loshchilov & Hutter, "Decoupled Weight Decay Regularization"
- OneCycleLR: Smith & Topin, "Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates"
- ASHA: Li et al., "A System for Massively Parallel Hyperparameter Tuning"
- Orion: Bouthillier et al., "Epistimio/orion: Asynchronous Distributed Hyperparameter Optimization"
