# DeePiCt — Linux aarch64 + CUDA 13 + GB10 (sm_121)

Upstream: https://github.com/ZauggGroup/DeePiCt

DeePiCt ships two independent networks. They land in very different places on
aarch64, so they are assessed separately.

| branch | stack | result |
|---|---|---|
| **3D CNN** (`3d_cnn/`) | PyTorch + MONAI | 🟢 **PASS — native, no container** |
| **2D CNN** (`2d_cnn/`) | Keras 2.3.1 / TensorFlow 2.0.0 / Python 3.6 | 🔴 **blocked** |

## 3D CNN — 🟢 PASS, no source patches

`3d_cnn/environment.yaml` leaves `pytorch` and `torchvision` unpinned; the only
hard pins are `python=3.7`, `h5py==2.10.0` and `mrcfile==1.1.2`, none of which
matter. A current stack works directly:

```bash
mamba create -y -p ./env -c conda-forge python=3.11
./env/bin/pip install torch torchvision            # 2.13.0+cu130 aarch64
./env/bin/pip install h5py mrcfile monai scikit-learn scikit-image scipy numpy \
                      pandas pyyaml tqdm tensorboardX matplotlib seaborn imageio
export PYTHONPATH=$PWD/3d_cnn/src
```

Resolved: torch 2.13.0+cu130, monai 1.6.0, h5py 3.16.0 — i.e. torch 2.13 and
h5py 3.x instead of the 2020-era pins, with no code changes.

### Smoke test — DeePiCt's own training routines

`_matrix/deepict-smoke/run_train.py` drives `networks.unet.UNet3D`,
`networks.loss.DiceCoefficientLoss` and `networks.routines.train` on synthetic
32^3 patches containing a bright sphere in noise (48 training / 16 validation).

| epoch | val loss | val Dice |
|---|---|---|
| 1 | 0.8508 | 0.0549 |
| 3 | 0.1012 | 0.9038 |
| 5 | 0.0656 | 0.9308 |
| 7 | **0.0390** | **0.9574** |
| 8 | 0.0816 | 0.9137 |

Validation Dice **0.055 -> 0.914** (peak 0.957). The network, the Dice loss and
the training loop are all numerically correct on aarch64.

### Three upstream bugs hit along the way (none aarch64-specific)

1. **`UNet3D(final_activation=...)` needs a module, not a string.**
   `final_activation='sigmoid'` fails the constructor's own assertion
   (`AssertionError: Activation must be torch module`); pass `nn.Sigmoid()`.

2. **`networks.routines.train(lr_scheduler=None)` is not optional.** The
   parameter defaults to `None` but the function ends with an unconditional
   `lr_scheduler.step(train_loss)`, so the default value crashes with
   `AttributeError: 'NoneType' object has no attribute 'step'`. Supply e.g.
   `ReduceLROnPlateau`.

3. **`networks.metrics.dice` cannot be used with `networks.routines.validate` on
   GPU.** `validate` calls `metric(prediction, y.long()).item()` with CUDA
   tensors, but `metrics.dice` is written with `np.sum` / `np.multiply`.
   `validate` also returns only the loss, not the `(loss, metric)` pair its
   body suggests. The smoke test therefore computes Dice with a small
   torch-native helper.

## 2D CNN — 🔴 blocked, same wall as crYOLO

`2d_cnn/envs/keras-env.yaml` is a fully build-pinned 2019 environment:

```yaml
  - python=3.6.10=h0371630_0
  - cudatoolkit=10.0.130=0
  - cudnn=7.6.5=cuda10.0_0
  - snakemake=5.3.0=py36_1
  - pip:
    - keras==2.3.1
    - tensorflow==2.0.0
    - tensorflow-gpu==2.0.0
    - h5py==2.10.0
```

- **Python 3.6.10** with exact conda build strings — nothing resolves for
  linux-aarch64.
- `tensorflow-gpu==2.0.0` never existed as an aarch64 wheel, and the NGC
  container route does not help: `nvcr.io/nvidia/tensorflow:25.02-tf2-py3` is
  Python **3.12**, so this pin set cannot be installed there either.
- `cudatoolkit=10.0` predates the Blackwell architecture entirely; GB10 is
  sm_121.

This is the same structural blocker as crYOLO: not a porting problem but a
frozen 2019 dependency set. Porting the 2D branch would mean rewriting it
against a modern Keras — out of scope here.

**Substitutes available on this host:** the DeePiCt 3D branch itself,
MemBrain-seg (membrane segmentation, PyTorch) and DeepFinder (particle
segmentation/picking, TF container) all cover the same ground and are verified
working.
