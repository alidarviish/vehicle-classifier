"""Real-torch checks of the ResNet18 setup (src/resnet.py and the parameter groups of src/train.py).

No training run, no data, no download:
- the network is built with pretrained=False (random weights). The checks are about structure,
  freezing, BatchNorm modes, optimizer groups and output shape, which do not depend on weight values;
- inputs are random tensors of shape 2 x 3 x 224 x 224; nothing is saved.
Check 7 calls optimizer.step() on these throwaway models, only to confirm that Adam leaves
frozen layer4 unchanged during the warm-up epochs.

Run from the repository root with the Python used for training:
    python scripts/verify_resnet.py
Exit code 0 = every check passed, 1 = at least one check failed.
"""

import io
import platform
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))  # so "from src..." works with "python scripts/verify_resnet.py"

import torch
import torchvision
from PIL import Image
from torch import nn

from src.dataset import CLASS_TO_IDX
from src.resnet import (BACKBONE_PARTS, PRETRAINED_WEIGHTS, ResNet18Classifier, build_resnet18, count_params,
                        trainable_parts, unfreeze_layer4)
from src.train import (BATCH_SIZE, FT_WARMUP_EPOCHS, IMAGE_SIZE, LEARNING_RATE, OPTIMIZERS,
                       RESNET_HEAD_LR, RESNET_IMAGE_SIZE, RESNET_LAYER4_LR, RESNET_NORM_MEAN,
                       RESNET_NORM_STD, RESNET_TRANSFORM, WEIGHT_DECAY, resnet_transform)

SEED = 42
NUM_CLASSES = 8
BATCH = 2
TORCHVISION_RESNET18_TOTAL = 11_689_512   # torchvision resnet18 with its 1000-class head
FC_PARAMS = 4_104                         # 512 * 8 + 8
LAYER4_PARAMS = 8_393_728
TOTAL_PARAMS = 11_180_616                 # TORCHVISION_RESNET18_TOTAL - 513_000 + FC_PARAMS
FE_COUNTS = (4_104, 11_176_512)           # (trainable, frozen): fc only
FT_COUNTS = (8_397_832, 2_782_784)        # (trainable, frozen): fc + layer4
ALWAYS_FROZEN = ("conv1", "bn1", "layer1", "layer2", "layer3")


# ---------- helpers ----------

def new_model():
    """Same seed before every build, so all models start from identical weights."""
    torch.manual_seed(SEED)
    return build_resnet18(num_classes=NUM_CLASSES, pretrained=False)


def random_batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    images = torch.randn(BATCH, 3, RESNET_IMAGE_SIZE, RESNET_IMAGE_SIZE, generator=g)
    labels = torch.tensor([0, 1])
    return images, labels


def param_groups(model, fine_tuning):
    """The same groups as src/train.py builds for --model resnet18 (fc; plus layer4 in fine-tuning)."""
    groups = [{"params": list(model.net.fc.parameters()), "lr": RESNET_HEAD_LR}]
    if fine_tuning:
        groups.append({"params": list(model.net.layer4.parameters()), "lr": RESNET_LAYER4_LR})
    return groups


def numel(module):
    return sum(p.numel() for p in module.parameters())


def part(model, name):
    return getattr(model.net, name)


def trainable_param_names(module):
    return [n for n, p in module.named_parameters() if p.requires_grad]


def modules_in_mode(module, training):
    """Names of submodules (with own parameters or buffers) whose .training equals `training`."""
    names = []
    for name, m in module.named_modules():
        has_state = any(True for _ in m.parameters(recurse=False)) or any(True for _ in m.buffers(recurse=False))
        if has_state and m.training == training:
            names.append(name or "<self>")
    return names


def batchnorms(module):
    return [m for m in module.modules() if isinstance(m, nn.BatchNorm2d)]


def snapshot(named_tensors):
    return {n: t.detach().clone() for n, t in named_tensors}


def changed(before, after):
    return [n for n in before if not torch.equal(before[n], after[n])]


def short(names, limit=5):
    return ", ".join(names[:limit]) + (f" ... (+{len(names) - limit})" if len(names) > limit else "")


# ---------- checks: each returns a list of problems (empty = PASS) ----------

def check_1_build():
    p = []
    m = new_model()
    if not isinstance(m, ResNet18Classifier):
        p.append(f"build_resnet18 returned {type(m).__name__}, not ResNet18Classifier")
    if not isinstance(m.net, torchvision.models.ResNet):
        p.append(f"model.net is {type(m.net).__name__}, not torchvision ResNet")
    fc = m.net.fc
    if not (isinstance(fc, nn.Linear) and fc.in_features == 512 and fc.out_features == NUM_CLASSES
            and fc.bias is not None):
        p.append(f"head is {fc}, expected Linear(512, {NUM_CLASSES}) with bias")
    reference = numel(torchvision.models.resnet18(weights=None))
    if reference != TORCHVISION_RESNET18_TOTAL:
        p.append(f"torchvision resnet18 has {reference:,} parameters, expected {TORCHVISION_RESNET18_TOTAL:,}")
    return p


def check_2_counts():
    p = []
    m = new_model()
    if numel(m) != TOTAL_PARAMS:
        p.append(f"total {numel(m):,}, expected {TOTAL_PARAMS:,}")
    if numel(m.net.fc) != FC_PARAMS:
        p.append(f"fc {numel(m.net.fc):,}, expected {FC_PARAMS:,}")
    if numel(m.net.layer4) != LAYER4_PARAMS:
        p.append(f"layer4 {numel(m.net.layer4):,}, expected {LAYER4_PARAMS:,}")
    if count_params(m) != FE_COUNTS:
        p.append(f"fc only: (trainable, frozen) = {count_params(m)}, expected {FE_COUNTS}")
    unfreeze_layer4(m)
    if count_params(m) != FT_COUNTS:
        p.append(f"fc + layer4: (trainable, frozen) = {count_params(m)}, expected {FT_COUNTS}")
    return p


def check_3_feature_extraction_modes():
    p = []
    m = new_model()
    m.train()
    if trainable_parts(m) != ["fc"]:
        p.append(f"trainable parts {trainable_parts(m)}, expected ['fc']")
    trainable = trainable_param_names(m)
    if any(not n.startswith("net.fc.") for n in trainable):
        p.append(f"trainable parameters outside fc: {short([n for n in trainable if not n.startswith('net.fc.')])}")
    in_train = modules_in_mode(m, True)
    if in_train != ["net.fc"]:
        p.append(f"after train(), modules with parameters/buffers in train mode: {in_train}, expected ['net.fc']")
    for name in BACKBONE_PARTS:
        bad = [f"{name}.{b}" for b in modules_in_mode(part(m, name), True)]
        if bad:
            p.append(f"frozen {name} not in eval mode: {short(bad)}")
    bn_train = [n for n, b in m.named_modules() if isinstance(b, nn.BatchNorm2d) and b.training]
    if bn_train:
        p.append(f"BatchNorm in train mode: {short(bn_train)}")
    m.eval()
    still_train = modules_in_mode(m, True)
    if still_train:
        p.append(f"after eval(), still in train mode: {short(still_train)}")
    return p


def check_4_fine_tuning_before_unfreeze():
    p = []
    fe = new_model()
    ft = new_model()
    if not all(torch.equal(a, b) for a, b in zip(fe.state_dict().values(), ft.state_dict().values())):
        p.append("feature-extraction and fine-tuning models do not start from identical weights")
    OPTIMIZERS["adam"](param_groups(ft, fine_tuning=True), weight_decay=WEIGHT_DECAY)
    ft.train()
    if trainable_parts(ft) != ["fc"]:
        p.append(f"epochs 1-{FT_WARMUP_EPOCHS}: trainable parts {trainable_parts(ft)}, expected ['fc']")
    if trainable_param_names(ft.net.layer4):
        p.append(f"layer4 has trainable parameters: {short(trainable_param_names(ft.net.layer4))}")
    bad = modules_in_mode(ft.net.layer4, True)
    if bad:
        p.append(f"layer4 modules in train mode: {short(bad)}")
    if any(b.training for b in batchnorms(ft.net.layer4)):
        p.append("layer4 BatchNorm in train mode")
    return p


def check_5_fine_tuning_after_unfreeze():
    p = []
    m = new_model()
    unfreeze_layer4(m)
    m.train()
    if trainable_parts(m) != ["layer4", "fc"]:
        p.append(f"trainable parts {trainable_parts(m)}, expected ['layer4', 'fc']")
    frozen_layer4 = [n for n, q in m.net.layer4.named_parameters() if not q.requires_grad]
    if frozen_layer4:
        p.append(f"layer4 parameters still frozen: {short(frozen_layer4)}")
    if not all(b.training for b in batchnorms(m.net.layer4)):
        p.append("not every layer4 BatchNorm is in train mode")
    if not m.net.fc.training:
        p.append("fc not in train mode")
    for name in ALWAYS_FROZEN:
        if trainable_param_names(part(m, name)):
            p.append(f"{name} has trainable parameters")
        bad = modules_in_mode(part(m, name), True)
        if bad:
            p.append(f"{name} not in eval mode: {short(bad)}")
    return p


def check_6_forward_shape():
    p = []
    if RESNET_IMAGE_SIZE != 224:
        p.append(f"RESNET_IMAGE_SIZE in src/train.py is {RESNET_IMAGE_SIZE}, expected 224")
    if IMAGE_SIZE != 128:
        p.append(f"baseline IMAGE_SIZE in src/train.py is {IMAGE_SIZE}, expected 128 (must stay unchanged)")
    tensor = RESNET_TRANSFORM(Image.new("RGB", (300, 180)))   # any input size -> 3 x 224 x 224
    if tuple(tensor.shape) != (3, RESNET_IMAGE_SIZE, RESNET_IMAGE_SIZE):
        p.append(f"RESNET_TRANSFORM output {tuple(tensor.shape)}, expected (3, {RESNET_IMAGE_SIZE}, {RESNET_IMAGE_SIZE})")
    images, _ = random_batch()
    for label, m in [("feature extraction", new_model()), ("fine-tuning after unfreeze", new_model())]:
        if label.startswith("fine"):
            unfreeze_layer4(m)
        for mode in ("train", "eval"):
            getattr(m, mode)()
            out = m(images)
            if tuple(out.shape) != (BATCH, NUM_CLASSES):
                p.append(f"{label}, {mode}(): output shape {tuple(out.shape)}, expected ({BATCH}, {NUM_CLASSES})")
    return p


def check_7_optimizer():
    p = []
    m = new_model()
    opt = OPTIMIZERS["adam"](param_groups(m, fine_tuning=True), weight_decay=WEIGHT_DECAY)
    lrs = [g["lr"] for g in opt.param_groups]
    if lrs != [1e-3, 1e-4]:
        p.append(f"group learning rates {lrs}, expected [0.001, 0.0001]")
    if [id(q) for q in opt.param_groups[0]["params"]] != [id(q) for q in m.net.fc.parameters()]:
        p.append("group 0 is not exactly the fc parameters")
    if [id(q) for q in opt.param_groups[1]["params"]] != [id(q) for q in m.net.layer4.parameters()]:
        p.append("group 1 is not exactly the layer4 parameters")

    # warm-up epoch: one backward pass, before unfreezing
    images, labels = random_batch()
    m.train()
    before = snapshot(m.named_parameters())
    opt.zero_grad()
    nn.CrossEntropyLoss()(m(images), labels).backward()
    with_grad = [n for n, q in m.named_parameters() if q.grad is not None]
    if [n for n in with_grad if n.startswith("net.layer4.")]:
        p.append("layer4 has gradients before unfreeze")
    if [n for n in with_grad if not n.startswith("net.fc.")]:
        p.append(f"gradients outside fc before unfreeze: {short([n for n in with_grad if not n.startswith('net.fc.')])}")
    if not all(q.grad is not None for q in m.net.fc.parameters()):
        p.append("fc has no gradient")
    moved = changed(before, snapshot(m.named_parameters()))
    if moved:
        p.append(f"weights changed without optimizer.step(): {short(moved)}")

    # one step on the throwaway model: only fc may change, layer4 must be skipped by Adam
    opt.step()
    moved = changed(before, snapshot(m.named_parameters()))
    if not moved or any(not n.startswith("net.fc.") for n in moved):
        p.append(f"after one step before unfreeze, changed weights: {short(moved) or 'none'}, expected fc only")
    if any(len(opt.state[q]) > 0 for q in m.net.layer4.parameters()):
        p.append("Adam created state for frozen layer4 parameters")

    # after unfreeze: layer4 gets gradients, conv1/bn1/layer1-3 still none
    unfreeze_layer4(m)
    m.train()
    opt.zero_grad()
    nn.CrossEntropyLoss()(m(images), labels).backward()
    if not all(q.grad is not None for q in m.net.layer4.parameters()):
        p.append("after unfreeze, some layer4 parameters have no gradient")
    frozen_grads = [n for n, q in m.named_parameters() if q.grad is not None
                    and n.split(".")[1] in ALWAYS_FROZEN]
    if frozen_grads:
        p.append(f"after unfreeze, gradients in always-frozen parts: {short(frozen_grads)}")
    return p


def check_8_batchnorm_stats():
    p = []
    images, _ = random_batch()

    # feature extraction: no running statistic may change
    m = new_model()
    before = snapshot(m.named_buffers())
    m.train()
    m(images)
    moved = changed(before, snapshot(m.named_buffers()))
    if moved:
        p.append(f"feature extraction: frozen BatchNorm statistics changed: {short(moved)}")

    # fine-tuning after unfreeze: frozen parts unchanged, layer4 must update (shows the check can detect changes)
    m = new_model()
    unfreeze_layer4(m)
    before = snapshot(m.named_buffers())
    m.train()
    m(images)
    moved = changed(before, snapshot(m.named_buffers()))
    frozen_moved = [n for n in moved if n.split(".")[1] in ALWAYS_FROZEN]
    if frozen_moved:
        p.append(f"fine-tuning: statistics of frozen parts changed: {short(frozen_moved)}")
    layer4_buffers = [n for n in before if n.startswith("net.layer4.")]
    if sorted(n for n in moved if n.startswith("net.layer4.")) != sorted(layer4_buffers):
        p.append("control failed: not every layer4 BatchNorm statistic changed in train mode, "
                 "so the comparison may not detect changes")
    return p


def check_9_checkpoint_round_trip():
    """The evaluation path of scripts/analyze_baseline.py, in memory only (io.BytesIO, nothing on disk).

    torch.load is called exactly as analyze_baseline.py calls it:
    torch.load(file, map_location="cpu", weights_only=True), so only tensors and plain Python values
    are accepted. If the load fails here, analyze_baseline.py would fail on a real ResNet checkpoint too.
    The checkpoint has the same keys and value types as src/train.py writes for --model resnet18
    (all metadata plain Python values; val_f1 is stored as float(best_f1)).
    """
    p = []
    m = new_model()
    unfreeze_layer4(m)
    torch.manual_seed(123)
    m.net.fc.reset_parameters()   # saved weights differ from a fresh build, so a missing load is detected
    trainable, frozen = count_params(m)
    checkpoint = {
        "model_state": m.state_dict(),
        "class_to_idx": CLASS_TO_IDX,
        "architecture": "ResNet18",
        "image_size": RESNET_IMAGE_SIZE,
        "transform": {"resize": [RESNET_IMAGE_SIZE, RESNET_IMAGE_SIZE], "normalize_mean": RESNET_NORM_MEAN,
                      "normalize_std": RESNET_NORM_STD, "augmentation": "none", "augmentation_params": None},
        "seed": SEED,
        "dropout": 0.0,
        "pooling": None,
        "optimizer": "adam",
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "batch_mode": "standard",
        "train_subset": None,
        "train_subset_sha256": None,
        "n_train": 2633,
        "scheduler": "none",
        "scheduler_params": None,
        "weight_decay": WEIGHT_DECAY,
        "loss_name": "ce",
        "epoch": 6,
        "val_f1": float(0.5),
        "pretrained_weights": PRETRAINED_WEIGHTS,
        "resnet_mode": "fine_tuning",
        "warmup_epochs": FT_WARMUP_EPOCHS,
        "trainable_parts": trainable_parts(m),
        "trainable_params": trainable,
        "frozen_params": frozen,
        "param_group_lrs": {"fc": RESNET_HEAD_LR, "layer4": RESNET_LAYER4_LR},
    }
    buffer = io.BytesIO()
    torch.save(checkpoint, buffer)
    buffer.seek(0)
    loaded = torch.load(buffer, map_location="cpu", weights_only=True)   # same call as analyze_baseline.py; errors are not caught

    for key in checkpoint:
        if key == "model_state":
            continue
        if key not in loaded or loaded[key] != checkpoint[key]:
            p.append(f"metadata '{key}' not loaded back unchanged: {loaded.get(key, '<missing>')!r}")
    if list(loaded["model_state"]) != list(checkpoint["model_state"]):
        p.append("model_state keys differ after torch.save / torch.load")
    if loaded["image_size"] != 224 or loaded["transform"]["resize"] != [224, 224]:
        p.append(f"checkpoint records image_size {loaded['image_size']} / resize {loaded['transform']['resize']}, "
                 "expected 224 / [224, 224]")

    # analyze_baseline.py rebuilds the ResNet transform from the checkpoint: 224 for new checkpoints,
    # 128 for the earlier exploratory ones
    image = Image.new("RGB", (300, 180))
    for size in (loaded["transform"]["resize"][0], 128):
        rebuilt = resnet_transform(size, loaded["transform"]["normalize_mean"], loaded["transform"]["normalize_std"])
        if tuple(rebuilt(image).shape) != (3, size, size):
            p.append(f"transform rebuilt from checkpoint metadata (resize {size}) gives {tuple(rebuilt(image).shape)}")

    other = new_model()
    other.load_state_dict(loaded["model_state"], strict=True)
    images, _ = random_batch()
    m.eval()
    other.eval()
    with torch.no_grad():
        if not torch.allclose(m(images), other(images), rtol=0, atol=1e-6):
            p.append("reloaded model gives different outputs in eval mode")

    # strict loading must reject a state with a missing key
    broken = dict(loaded["model_state"])
    removed = "net.fc.bias"
    del broken[removed]
    try:
        new_model().load_state_dict(broken, strict=True)
    except RuntimeError as error:
        if removed not in str(error):
            p.append(f"strict load failed, but the error does not name the missing key '{removed}': {error}")
    else:
        p.append(f"load_state_dict(strict=True) accepted a state without '{removed}'")
    return p


CHECKS = [
    ("1 build ResNet18Classifier (torch/torchvision, pretrained=False)", check_1_build),
    ("2 parameter counts", check_2_counts),
    ("3 feature extraction: only fc trainable / in train mode, frozen BatchNorm in eval", check_3_feature_extraction_modes),
    (f"4 fine-tuning epochs 1-{FT_WARMUP_EPOCHS}: only fc trainable, layer4 frozen / eval", check_4_fine_tuning_before_unfreeze),
    ("5 fine-tuning after unfreeze: layer4 + fc trainable, layer4 BatchNorm train, conv1/bn1/layer1-3 frozen / eval", check_5_fine_tuning_after_unfreeze),
    (f"6 transform and forward {BATCH}x3x{RESNET_IMAGE_SIZE}x{RESNET_IMAGE_SIZE} -> {BATCH}x{NUM_CLASSES}",
     check_6_forward_shape),
    ("7 optimizer groups, learning rates and gradients", check_7_optimizer),
    ("8 BatchNorm running statistics of frozen parts unchanged", check_8_batchnorm_stats),
    ("9 checkpoint round trip in memory (torch.save / torch.load as in analyze_baseline.py, strict=True, size 224)",
     check_9_checkpoint_round_trip),
]


def main():
    print(f"python {platform.python_version()} | torch {torch.__version__} | torchvision {torchvision.__version__} | CPU")
    print("no training, no data, no download (pretrained=False); nothing is saved\n")
    failed = 0
    for name, fn in CHECKS:
        try:
            problems = fn()
        except Exception:
            problems = ["exception:\n" + traceback.format_exc()]
        if problems:
            failed += 1
            print(f"[FAIL] {name}")
            for problem in problems:
                print(f"       - {problem}")
        else:
            print(f"[PASS] {name}")
    print(f"\n{len(CHECKS) - failed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
