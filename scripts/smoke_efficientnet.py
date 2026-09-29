"""Smoke test for the EfficientNet-B0 comparison experiment (no training run, writes nothing in the repository).

Run from the repository root:
    python scripts/smoke_efficientnet.py

Needs the training environment (torch, torchvision, scikit-learn) and configs/local_paths.json.
Uses synthetic tensors, one batch each of the train and validation splits (read-only), and a
temporary folder outside the repository. Test and Neysan images are never read. The first run
downloads the torchvision EfficientNet-B0 ImageNet weights into the torch cache (not into the
repository). Exit code 0 = all checks passed.
"""
import csv
import hashlib
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))  # so "from src ..." and "from scripts ..." work with "python scripts/smoke_efficientnet.py"

import torch
from torch import nn

import src.train as T
from src import efficientnet as E

results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -> {detail}" if detail else ""))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


FINAL_CKPT = REPO / "checkpoints" / "resnet224_ft_aug_best.pt"
FINAL_SHA = "c26280e97f7324ee1ec85d7bf78a58ea0b4d2acb4cf3e975dbd52316366281a1"
TEST_CSV = REPO / "decisions" / "test_frozen.csv"
protected_before = {p: sha(p) for p in (FINAL_CKPT, TEST_CSV)}


def effnet_snapshot():
    """sha256 of every existing effnet_b0* file in checkpoints/ and reports/ (incl. reports/analysis/effnet_b0*/)."""
    snap = {}
    for base in (REPO / "checkpoints", REPO / "reports"):
        if not base.is_dir():
            continue
        for p in base.rglob("effnet_b0*"):
            files = [p] if p.is_file() else [f for f in p.rglob("*") if f.is_file()]
            for f in files:
                snap[f.relative_to(REPO).as_posix()] = sha(f)
    return snap


effnet_before = effnet_snapshot()   # artifacts of earlier real runs are allowed; only changes by this test fail
print(f"existing effnet_b0* artifacts before the smoke test: {len(effnet_before)}")
check("final ResNet checkpoint sha256 before", protected_before[FINAL_CKPT] == FINAL_SHA)

# 1. model structure, head and parameter counts (pretrained weights)
model = E.build_efficientnet_b0(num_classes=8, pretrained=True)
head = model.net.classifier
print("classifier:", head)
check("classifier = Dropout(p=0.2) -> Linear(1280, 8)",
      isinstance(head[0], nn.Dropout) and head[0].p == 0.2 and isinstance(head[1], nn.Linear)
      and head[1].in_features == 1280 and head[1].out_features == 8 and len(head) == 2)
trainable, frozen = E.count_params(model)
print(f"trainable {trainable:,} | frozen {frozen:,} | total {trainable + frozen:,}")
check("trainable = classifier only = 1280*8 + 8 = 10,248", trainable == 10248, f"{trainable:,}")
check("frozen = all of net.features", frozen == sum(p.numel() for p in model.net.features.parameters()), f"{frozen:,}")
check("trainable_parts == ['classifier']", E.trainable_parts(model) == ["classifier"], str(E.trainable_parts(model)))

# 2. train/eval modes: frozen features stay in eval (BatchNorm, stochastic depth), head dropout trains
model.train()
check("model.train(): features in eval mode (all submodules)", all(not m.training for m in model.net.features.modules()))
check("model.train(): classifier (dropout) in train mode", model.net.classifier.training and head[0].training)
model.eval()
check("model.eval(): everything in eval mode", all(not m.training for m in model.modules()))

# 3. forward on a synthetic batch
with torch.no_grad():
    out = model(torch.randn(2, 3, 224, 224))
check("forward [2, 3, 224, 224] -> [2, 8]", list(out.shape) == [2, 8], str(list(out.shape)))

# 4. one optimizer step on synthetic data: only the classifier changes
torch.manual_seed(0)
opt = torch.optim.Adam([{"params": list(model.net.classifier.parameters()), "lr": T.EFFNET_HEAD_LR}])
feat_before = {k: v.clone() for k, v in model.net.features.state_dict().items()}
head_before = model.net.classifier[1].weight.clone()
model.train()
loss = nn.CrossEntropyLoss()(model(torch.randn(4, 3, 224, 224)), torch.tensor([0, 1, 2, 7]))
opt.zero_grad(); loss.backward(); opt.step()
check("one step: classifier weights changed", not torch.equal(head_before, model.net.classifier[1].weight))
check("one step: features weights AND BatchNorm running stats unchanged",
      all(torch.equal(feat_before[k], v) for k, v in model.net.features.state_dict().items()))

# 5. pretrained=False builds the same network without download (used by analysis)
empty = E.build_efficientnet_b0(num_classes=8, pretrained=False)
check("pretrained=False: same state_dict keys and shapes",
      {k: tuple(v.shape) for k, v in empty.state_dict().items()} == {k: tuple(v.shape) for k, v in model.state_dict().items()})


# 6. CLI guards (train() is replaced by a stub, nothing is trained or written)
def cli(argv):
    captured, real = {}, T.train
    T.train = lambda *a, **k: captured.update(args=a)
    old = sys.argv
    sys.argv = ["src.train"] + argv
    try:
        T.main()
        return "ok", captured.get("args")
    except SystemExit as e:
        return f"rejected (exit {e.code})", None
    finally:
        T.train, sys.argv = real, old


eff = ["--model", "efficientnet_b0"]
status, args = cli(["--run-name", "effnet_b0_fe_aug", *eff, "--augmentation", "full_aug"])
check("CLI efficientnet_b0 + full_aug + run name effnet_b0_fe_aug accepted", status == "ok", status)
if args:
    check("  train() gets full_aug, adam, wd 0, scheduler none, ce, standard, no subset, model efficientnet_b0",
          args[4] == "full_aug" and args[7] == "adam" and args[8] == 0.0 and args[9] == "none" and args[10] is None
          and args[11] == "standard" and args[12] == "ce" and args[13] == "efficientnet_b0" and args[14] is None)
for label, argv in [
    ("run name without effnet_b0_ prefix", ["--run-name", "resnet224_ft_aug", *eff, "--augmentation", "full_aug"]),
    ("--resnet-mode with efficientnet_b0", ["--run-name", "effnet_b0_x", *eff, "--resnet-mode", "fine_tuning"]),
    ("--augmentation crop_jitter", ["--run-name", "effnet_b0_x", *eff, "--augmentation", "crop_jitter"]),
    ("--dropout 0.5", ["--run-name", "effnet_b0_x", *eff, "--dropout", "0.5"]),
    ("--loss bce", ["--run-name", "effnet_b0_x", *eff, "--loss", "bce"]),
]:
    status, _ = cli(argv)
    check(f"CLI rejects {label}", status.startswith("rejected"), status)
status, _ = cli(["--run-name", "x", "--model", "resnet18"])
check("regression: resnet18 without --resnet-mode still rejected", status.startswith("rejected"), status)
status, _ = cli(["--run-name", "x", "--model", "resnet18", "--resnet-mode", "fine_tuning", "--augmentation", "full_aug"])
check("regression: resnet18 fine_tuning + full_aug still accepted", status == "ok", status)


# 7. train(): overwrite / name guards and the transforms it passes on (stopped before any training)
class Stop(Exception):
    pass


real_reports, real_ckpt, real_loaders = T.REPORTS_DIR, T.CHECKPOINT_DIR, T.build_loaders
with tempfile.TemporaryDirectory() as tmp:
    T.REPORTS_DIR, T.CHECKPOINT_DIR = Path(tmp) / "reports", Path(tmp) / "checkpoints"
    seen = {}

    def fake_loaders(manifest, config, train_tf, subset, batch_mode, val_tf):
        seen.update(train=train_tf, val=val_tf)
        raise Stop

    T.build_loaders = fake_loaders
    try:
        try:
            T.train("resnet224_ft_aug", T.DEFAULT_MANIFEST, T.DEFAULT_CONFIG, augmentation="full_aug",
                    model_name="efficientnet_b0")
            check("train() rejects a run name without prefix", False)
        except ValueError:
            check("train() rejects a run name without prefix", True)
        try:
            T.train("effnet_b0_fe_aug", T.DEFAULT_MANIFEST, T.DEFAULT_CONFIG, augmentation="full_aug",
                    model_name="efficientnet_b0")
        except Stop:
            pass
        check("train() uses RESNET_TRAIN_TRANSFORMS['full_aug'] for train and RESNET_TRANSFORM for val",
              seen.get("train") is T.RESNET_TRAIN_TRANSFORMS["full_aug"] and seen.get("val") is T.RESNET_TRANSFORM)
        T.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        (T.CHECKPOINT_DIR / "effnet_b0_fe_aug_best.pt").write_bytes(b"x")
        try:
            T.train("effnet_b0_fe_aug", T.DEFAULT_MANIFEST, T.DEFAULT_CONFIG, augmentation="full_aug",
                    model_name="efficientnet_b0")
            check("train() refuses to overwrite an existing checkpoint", False)
        except FileExistsError:
            check("train() refuses to overwrite an existing checkpoint", True)
    finally:
        T.REPORTS_DIR, T.CHECKPOINT_DIR, T.build_loaders = real_reports, real_ckpt, real_loaders

# 8. one real batch of train and validation with the EfficientNet transforms (no Test / Neysan)
test_sha = {r["sha256"] for r in csv.DictReader(open(TEST_CSV, encoding="utf-8"))}
ney_sha = {r["sha256"] for r in csv.DictReader(open(REPO / "decisions/neysan_eval.csv", encoding="utf-8"))}
split = {r["image_path"]: r for r in csv.DictReader(open(T.DEFAULT_MANIFEST, encoding="utf-8"))}
tl, vl = T.build_loaders(T.DEFAULT_MANIFEST, T.DEFAULT_CONFIG, T.RESNET_TRAIN_TRANSFORMS["full_aug"],
                         None, "standard", T.RESNET_TRANSFORM)
used = tl.dataset.image_paths + vl.dataset.image_paths
check("loaders: train 2633 / val 659, only train/val rows, no Test / Neysan content",
      len(tl.dataset) == 2633 and len(vl.dataset) == 659
      and all(split[p]["split"] in ("train", "val") for p in used)
      and not {split[p]["sha256"] for p in used} & (test_sha | ney_sha))
x, _ = next(iter(tl)); xv, _ = next(iter(vl))
model.eval()
with torch.no_grad():
    o1, o2 = model(x), model(xv)
check("real batches [32, 3, 224, 224] -> [32, 8]", list(x.shape) == [32, 3, 224, 224] and list(o1.shape) == [32, 8]
      and list(o2.shape) == [32, 8])

# 9. analysis branch: evaluate_checkpoint() with a synthetic checkpoint and a tiny synthetic dataset
import src.dataset as D
from scripts import analyze_baseline as A


class TinyVal(torch.utils.data.Dataset):
    def __init__(self, split, transform, manifest, config):
        self.items = [(torch.randn(3, 224, 224), i % 8) for i in range(4)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


with tempfile.TemporaryDirectory() as tmp:
    ck = {"model_state": empty.state_dict(), "class_to_idx": D.CLASS_TO_IDX, "architecture": "EfficientNet-B0",
          "transform": {"resize": [224, 224], "normalize_mean": T.RESNET_NORM_MEAN,
                        "normalize_std": T.RESNET_NORM_STD, "augmentation": "full_aug"}}
    path = Path(tmp) / "effnet_b0_smoke_best.pt"
    torch.save(ck, path)
    real_ds = D.VehicleDataset
    D.VehicleDataset = TinyVal
    try:
        y_true, y_pred, classes, _ = A.evaluate_checkpoint(path, T.DEFAULT_CONFIG)
        check("analyze_baseline.evaluate_checkpoint loads an EfficientNet-B0 checkpoint (weights_only)",
              len(y_true) == len(y_pred) == 4 and list(classes) == list(D.CLASSES))
    finally:
        D.VehicleDataset = real_ds
check("EVAL_TRANSFORM_NAMES has EfficientNet-B0", A.EVAL_TRANSFORM_NAMES.get("EfficientNet-B0") == "RESNET_TRANSFORM")

# 10. protected files unchanged, no EfficientNet file written in the repository
check("final ResNet checkpoint and decisions/test_frozen.csv unchanged",
      all(sha(p) == h for p, h in protected_before.items()))
effnet_after = effnet_snapshot()
check("smoke test created, changed or deleted no effnet_b0* artifact in checkpoints/ or reports/",
      effnet_after == effnet_before,
      f"before {len(effnet_before)} | after {len(effnet_after)} | "
      f"new {sorted(set(effnet_after) - set(effnet_before))} | removed {sorted(set(effnet_before) - set(effnet_after))} | "
      f"changed {sorted(k for k in effnet_before.keys() & effnet_after.keys() if effnet_before[k] != effnet_after[k])}")
print(f"\n{sum(results)}/{len(results)} checks passed (no training; Test and Neysan not read)")
sys.exit(0 if all(results) else 1)
