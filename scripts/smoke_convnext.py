"""Smoke test for the ConvNeXt-Tiny comparison experiment (no training run, writes nothing in the repository).

Run from the repository root:
    python scripts/smoke_convnext.py

Needs the training environment (torch, torchvision, scikit-learn) and configs/local_paths.json.
Uses synthetic tensors, one batch each of the train and validation splits (read-only), and a
temporary folder outside the repository. Test and Neysan images are never read. The first run
downloads the torchvision ConvNeXt-Tiny ImageNet weights into the torch cache (not into the
repository). Exit code 0 = all checks passed.
"""
import csv
import hashlib
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))  # so "from src ..." and "from scripts ..." work with "python scripts/smoke_convnext.py"

import torch
from torch import nn
from torchvision.ops import StochasticDepth

import src.train as T
from src import convnext as C

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


def convnext_snapshot():
    """sha256 of every existing convnext_t* file in checkpoints/ and reports/ (incl. reports/analysis/convnext_t*/)."""
    snap = {}
    for base in (REPO / "checkpoints", REPO / "reports"):
        if not base.is_dir():
            continue
        for p in base.rglob("convnext_t*"):
            files = [p] if p.is_file() else [f for f in p.rglob("*") if f.is_file()]
            for f in files:
                snap[f.relative_to(REPO).as_posix()] = sha(f)
    return snap


convnext_before = convnext_snapshot()   # artifacts of earlier real runs are allowed; only changes by this test fail
print(f"existing convnext_t* artifacts before the smoke test: {len(convnext_before)}")
check("final ResNet checkpoint sha256 before", protected_before[FINAL_CKPT] == FINAL_SHA)

# 1. model structure, head and parameter counts (pretrained weights)
model = C.build_convnext_tiny(num_classes=8, pretrained=True)
head = model.net.classifier
print("classifier:", head)
check("classifier = LayerNorm2d(768) -> Flatten -> Linear(768, 8)",
      len(head) == 3 and type(head[0]).__name__ == "LayerNorm2d" and tuple(head[0].normalized_shape) == (768,)
      and isinstance(head[1], nn.Flatten) and isinstance(head[2], nn.Linear)
      and head[2].in_features == 768 and head[2].out_features == 8)
trainable, frozen = C.count_params(model)
features_params = sum(p.numel() for p in model.net.features.parameters())
print(f"trainable {trainable:,} | frozen {frozen:,} | total {trainable + frozen:,}")
check("trainable = whole classifier = LayerNorm 2*768 + Linear 768*8+8 = 7,688", trainable == 7688, f"{trainable:,}")
check("frozen = all of net.features", frozen == features_params, f"{frozen:,}")
check("trainable_parts == ['classifier']", C.trainable_parts(model) == ["classifier"], str(C.trainable_parts(model)))
check("PRETRAINED_WEIGHTS == 'IMAGENET1K_V1'", C.PRETRAINED_WEIGHTS == "IMAGENET1K_V1")

# 2. train/eval modes: frozen features (incl. stochastic depth) stay in eval, classifier trains
sd = [m for m in model.net.features.modules() if isinstance(m, StochasticDepth)]
model.train()
check("model.train(): features in eval mode (all submodules)", all(not m.training for m in model.net.features.modules()))
check("model.train(): every StochasticDepth module in the frozen backbone is off (eval)",
      len(sd) > 0 and all(not m.training for m in sd), f"{len(sd)} StochasticDepth modules")
check("model.train(): classifier in train mode", model.net.classifier.training)
x_syn = torch.randn(2, 3, 224, 224)
with torch.no_grad():
    f1, f2 = model.net.features(x_syn), model.net.features(x_syn)
check("model.train(): frozen backbone is deterministic (no stochastic depth active)", torch.equal(f1, f2))
model.eval()
check("model.eval(): everything in eval mode", all(not m.training for m in model.modules()))

# 3. forward on a synthetic batch
with torch.no_grad():
    out = model(x_syn)
check("forward [2, 3, 224, 224] -> [2, 8]", list(out.shape) == [2, 8], str(list(out.shape)))

# 4. one optimizer step on synthetic data: only the classifier changes
torch.manual_seed(0)
opt = torch.optim.Adam([{"params": list(model.net.classifier.parameters()), "lr": T.CONVNEXT_HEAD_LR}])
feat_before = {k: v.clone() for k, v in model.net.features.state_dict().items()}
head_before = {k: v.clone() for k, v in model.net.classifier.state_dict().items()}
model.train()
loss = nn.CrossEntropyLoss()(model(torch.randn(4, 3, 224, 224)), torch.tensor([0, 1, 2, 7]))
opt.zero_grad(); loss.backward(); opt.step()
head_after = model.net.classifier.state_dict()
check("one step: classifier Linear and LayerNorm weights changed",
      not torch.equal(head_before["2.weight"], head_after["2.weight"])
      and not torch.equal(head_before["0.weight"], head_after["0.weight"]))
check("one step: features weights unchanged",
      all(torch.equal(feat_before[k], v) for k, v in model.net.features.state_dict().items()))

# 5. pretrained=False builds the same network without download (used by analysis)
empty = C.build_convnext_tiny(num_classes=8, pretrained=False)
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


cnx = ["--model", "convnext_tiny"]
status, args = cli(["--run-name", "convnext_t_fe_none", *cnx])
check("CLI convnext_tiny + run name convnext_t_fe_none accepted", status == "ok", status)
if args:
    check("  train() gets augmentation none, adam, wd 0, scheduler none, ce, standard, no subset, model convnext_tiny",
          args[4] == "none" and args[7] == "adam" and args[8] == 0.0 and args[9] == "none" and args[10] is None
          and args[11] == "standard" and args[12] == "ce" and args[13] == "convnext_tiny" and args[14] is None)
for label, argv in [
    ("run name without convnext_t_ prefix", ["--run-name", "effnet_b0_x", *cnx]),
    ("--resnet-mode with convnext_tiny", ["--run-name", "convnext_t_x", *cnx, "--resnet-mode", "fine_tuning"]),
    ("--effnet-mode with convnext_tiny", ["--run-name", "convnext_t_x", *cnx, "--effnet-mode", "fine_tuning"]),
    ("--effnet-top-lr with convnext_tiny", ["--run-name", "convnext_t_x", *cnx, "--effnet-top-lr", "5e-5"]),
    ("--augmentation crop_jitter", ["--run-name", "convnext_t_x", *cnx, "--augmentation", "crop_jitter"]),
    ("--dropout 0.5", ["--run-name", "convnext_t_x", *cnx, "--dropout", "0.5"]),
    ("--loss bce", ["--run-name", "convnext_t_x", *cnx, "--loss", "bce"]),
    ("--batch-mode balanced", ["--run-name", "convnext_t_x", *cnx, "--batch-mode", "balanced"]),
]:
    status, _ = cli(argv)
    check(f"CLI rejects {label}", status.startswith("rejected"), status)
status, _ = cli(["--run-name", "x", "--model", "resnet18"])
check("regression: resnet18 without --resnet-mode still rejected", status.startswith("rejected"), status)
status, _ = cli(["--run-name", "x", "--model", "resnet18", "--resnet-mode", "fine_tuning", "--augmentation", "full_aug"])
check("regression: resnet18 fine_tuning + full_aug still accepted", status == "ok", status)
status, _ = cli(["--run-name", "effnet_b0_ft_none", "--model", "efficientnet_b0", "--effnet-mode", "fine_tuning"])
check("regression: efficientnet_b0 fine_tuning still accepted", status == "ok", status)
status, _ = cli(["--run-name", "convnext_t_x", "--model", "efficientnet_b0"])
check("regression: efficientnet_b0 with a convnext_t_ run name still rejected", status.startswith("rejected"), status)


# 7. train(): name / overwrite guards and the transforms it passes on (stopped before any training)
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
            T.train("resnet224_ft_aug", T.DEFAULT_MANIFEST, T.DEFAULT_CONFIG, model_name="convnext_tiny")
            check("train() rejects a run name without prefix", False)
        except ValueError:
            check("train() rejects a run name without prefix", True)
        try:
            T.train("convnext_t_fe_none", T.DEFAULT_MANIFEST, T.DEFAULT_CONFIG, model_name="convnext_tiny")
        except Stop:
            pass
        check("train() augmentation none: RESNET_TRANSFORM for train and val",
              seen.get("train") is T.RESNET_TRANSFORM and seen.get("val") is T.RESNET_TRANSFORM)
        steps = [type(t).__name__ for t in T.RESNET_TRANSFORM.transforms]
        check("RESNET_TRANSFORM = Resize(224, 224) -> ToTensor -> Normalize(ImageNet), no augmentation",
              steps == ["Resize", "ToTensor", "Normalize"] and tuple(T.RESNET_TRANSFORM.transforms[0].size) == (224, 224)
              and list(T.RESNET_TRANSFORM.transforms[2].mean) == [0.485, 0.456, 0.406]
              and list(T.RESNET_TRANSFORM.transforms[2].std) == [0.229, 0.224, 0.225], str(steps))
        T.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        (T.CHECKPOINT_DIR / "convnext_t_fe_none_best.pt").write_bytes(b"x")
        try:
            T.train("convnext_t_fe_none", T.DEFAULT_MANIFEST, T.DEFAULT_CONFIG, model_name="convnext_tiny")
            check("train() refuses to overwrite an existing checkpoint", False)
        except FileExistsError:
            check("train() refuses to overwrite an existing checkpoint", True)
        T.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        (T.REPORTS_DIR / "convnext_t_hist_only_history.csv").write_text("x", encoding="utf-8")
        try:
            T.train("convnext_t_hist_only", T.DEFAULT_MANIFEST, T.DEFAULT_CONFIG, model_name="convnext_tiny")
            check("train() refuses to overwrite an existing history", False)
        except FileExistsError:
            check("train() refuses to overwrite an existing history", True)
    finally:
        T.REPORTS_DIR, T.CHECKPOINT_DIR, T.build_loaders = real_reports, real_ckpt, real_loaders

# 7b. train(): history and checkpoint metadata, with build_loaders and run_epoch replaced by stubs
#     (synthetic tensors, no real training, no image read, files only in a temporary folder)
from torch.utils.data import DataLoader, TensorDataset


def tiny_loaders(manifest, config, train_tf, subset, batch_mode, val_tf):
    ds = TensorDataset(torch.randn(2, 3, 224, 224), torch.tensor([0, 1]))
    return DataLoader(ds, batch_size=2), DataLoader(ds, batch_size=2)


calls = {"n": 0}


def fake_epoch(model, loader, criterion, optimizer=None, loss_name="ce"):
    calls["n"] += 1
    f1 = calls["n"] / 1000   # strictly increasing, so a checkpoint is written every epoch
    return {"loss": 1.0, "accuracy": f1, "precision": f1, "recall": f1, "f1": f1}


real_reports, real_ckpt, real_loaders, real_epoch = T.REPORTS_DIR, T.CHECKPOINT_DIR, T.build_loaders, T.run_epoch
with tempfile.TemporaryDirectory() as tmp:
    T.REPORTS_DIR, T.CHECKPOINT_DIR = Path(tmp) / "reports", Path(tmp) / "checkpoints"
    T.build_loaders, T.run_epoch = tiny_loaders, fake_epoch
    try:
        hist = T.train("convnext_t_smoke", T.DEFAULT_MANIFEST, T.DEFAULT_CONFIG, epochs=2, model_name="convnext_tiny")
        ck = torch.load(T.CHECKPOINT_DIR / "convnext_t_smoke_best.pt", map_location="cpu", weights_only=True)
        check("history trainable_params = 7,688 in every epoch, no lr_top column",
              [r["trainable_params"] for r in hist] == [7688, 7688] and all("lr_top" not in r for r in hist))
        check("checkpoint metadata: architecture, transform, weights, mode, parts, counts, lr groups",
              ck["architecture"] == "ConvNeXt-Tiny" and ck["pretrained_weights"] == "IMAGENET1K_V1"
              and ck["convnext_mode"] == "feature_extraction" and ck["trainable_parts"] == ["classifier"]
              and ck["trainable_params"] == 7688 and ck["frozen_params"] == features_params
              and ck["param_group_lrs"] == {"classifier": 1e-3} and ck["transform"]["resize"] == [224, 224]
              and ck["transform"]["augmentation"] == "none" and ck["transform"]["normalize_mean"] == [0.485, 0.456, 0.406]
              and ck["image_size"] == 224 and ck["seed"] == 42,
              str({k: ck[k] for k in ("architecture", "convnext_mode", "trainable_parts", "trainable_params",
                                      "frozen_params", "param_group_lrs")}))
    finally:
        T.REPORTS_DIR, T.CHECKPOINT_DIR, T.build_loaders, T.run_epoch = real_reports, real_ckpt, real_loaders, real_epoch

# 8. one real batch of train and validation with the ConvNeXt transform (no Test / Neysan)
test_sha = {r["sha256"] for r in csv.DictReader(open(TEST_CSV, encoding="utf-8"))}
ney_sha = {r["sha256"] for r in csv.DictReader(open(REPO / "decisions/neysan_eval.csv", encoding="utf-8"))}
split = {r["image_path"]: r for r in csv.DictReader(open(T.DEFAULT_MANIFEST, encoding="utf-8"))}
tl, vl = T.build_loaders(T.DEFAULT_MANIFEST, T.DEFAULT_CONFIG, T.RESNET_TRANSFORM, None, "standard", T.RESNET_TRANSFORM)
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
    ck = {"model_state": empty.state_dict(), "class_to_idx": D.CLASS_TO_IDX, "architecture": "ConvNeXt-Tiny",
          "transform": {"resize": [224, 224], "normalize_mean": T.RESNET_NORM_MEAN,
                        "normalize_std": T.RESNET_NORM_STD, "augmentation": "none"}}
    path = Path(tmp) / "convnext_t_smoke_best.pt"
    torch.save(ck, path)
    real_ds = D.VehicleDataset
    D.VehicleDataset = TinyVal
    try:
        y_true, y_pred, classes, _ = A.evaluate_checkpoint(path, T.DEFAULT_CONFIG)
        check("analyze_baseline.evaluate_checkpoint loads a ConvNeXt-Tiny checkpoint (weights_only)",
              len(y_true) == len(y_pred) == 4 and list(classes) == list(D.CLASSES))
    finally:
        D.VehicleDataset = real_ds
check("EVAL_TRANSFORM_NAMES has ConvNeXt-Tiny", A.EVAL_TRANSFORM_NAMES.get("ConvNeXt-Tiny") == "RESNET_TRANSFORM")

# 10. protected files unchanged, no ConvNeXt artifact created or changed in the repository
check("final ResNet checkpoint and decisions/test_frozen.csv unchanged",
      all(sha(p) == h for p, h in protected_before.items()))
convnext_after = convnext_snapshot()
check("smoke test created, changed or deleted no convnext_t* artifact in checkpoints/ or reports/",
      convnext_after == convnext_before,
      f"before {len(convnext_before)} | after {len(convnext_after)} | "
      f"new {sorted(set(convnext_after) - set(convnext_before))} | removed {sorted(set(convnext_before) - set(convnext_after))}")
print(f"\n{sum(results)}/{len(results)} checks passed (no training; Test and Neysan not read)")
sys.exit(0 if all(results) else 1)
