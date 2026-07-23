# vision-baselines

Reproducible PyTorch training pipelines for standard image-classification baselines.

The goal is a clean reference harness — deterministic seeding, standard augmentation, cosine learning-rate schedule, per-epoch CSV metrics, best-checkpoint saving — that new models and datasets can drop into without rewriting the training loop.

## Usage

```
pip install -r requirements.txt

python train.py --model lenet --epochs 30
python train.py --model resnet18 --epochs 60
```

Each run writes `results/<run-name>/metrics.csv` and `best.pt`.

## Models

- `lenet` — small LeNet-style CNN, a fast sanity-check baseline
- `resnet18` — torchvision ResNet-18 with a CIFAR-sized stem (3x3 conv, no initial max-pool — the ImageNet stem downsamples 32x32 inputs too aggressively)

## Results

Updated as runs complete on this hardware.

| Model | Dataset | Epochs | Test accuracy |
|---|---|---|---|
| LeNet | CIFAR-10 | 30 | run pending |
| ResNet-18 | CIFAR-10 | 60 | run pending |

## Roadmap

- [ ] mixed-precision training
- [ ] CIFAR-100 config
- [ ] fine-tuning pretrained backbones
