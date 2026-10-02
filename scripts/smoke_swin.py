"""Smoke test for the Swin-Tiny feature-extraction experiment (src/swin.py, --model swin_tiny in src/train.py).

No training run, no data, nothing written in the repository:
- the pretrained network is built once (the first run downloads the torchvision Swin-Tiny ImageNet weights
  into the torch cache, outside the repository); inputs are random tensors, one Adam step is taken on a
  throwaway model only;
- src.train.train() is only called with src.train.build_loaders replaced by a function that raises, so no
  image is read and no epoch can run; the guard checks with dummy files use a temporary folder;
- every file opened by this process is recorded (sys.addaudithook): no image, Test, Neysan or Final Test
  file may be opened. Test and Neysan files are never opened.
The expected parameter counts are the values of the design review. If the installed torch / torchvision
gives a different structure or different counts, the check fails and prints the real values.
It also runs scripts/verify_resnet.py (must report 9/9) and scripts/smoke_resnet_l3l4.py (must pass),
checks that the final checkpoint checkpoints/resnet224_ft_aug_best.pt is unchanged, that no file in
checkpoints/ or reports/ was created or changed, and that `git diff --check` is clean.

Run from the repository root with the Python used for training:
    python scripts/smoke_swin.py
Exit code 0 = all checks passed.
"""
import sys

OPENED = []   # every path this process opens, recorded before torch is imported


def _record_open(event, args):
    if event == "open" and args and isinstance(args[0], (str, bytes)):
        OPENED.append(args[0].decode(errors="replace") if isinstance(args[0], bytes) else args[0])


sys.addaudithook(_record_open)

import hashlib
import inspect
import io
import subprocess
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))  # so "from src ..." works with "python scripts/smoke_swin.py"

import torch
import torchvision
from PIL import Image
from torch import nn
from torchvision import models
from torchvision.ops import StochasticDepth

import src.train as T
from src import swin
from src.dataset import CLASS_TO_IDX, DEFAULT_CONFIG, DEFAULT_MANIFEST
from src.model import POOLING_TYPES

FINAL_RUN = "resnet224_ft_aug"
FINAL_CKPT = REPO / "checkpoints" / f"{FINAL_RUN}_best.pt"
FINAL_SHA = "c26280e97f7324ee1ec85d7bf78a58ea0b4d2acb4cf3e975dbd52316366281a1"
NEW_RUN = "swin_t_fe_none"
SEED = 42
# design review values (torchvision swin_t, 8-class head); a mismatch fails and prints the real numbers
EXPECTED_TRAINABLE = 6_152          # head: 768 * 8 + 8
EXPECTED_FROZEN = 27_519_354        # features + norm
EXPECTED_TOTAL = 27_525_506
EXPECTED_META_NUM_PARAMS = 28_288_354   # Swin_T_Weights.IMAGENET1K_V1.meta["num_params"] (1000-class head)
EXPECTED_STAGE_DEPTHS = [2, 2, 6, 2]    # Swin blocks in features[1], [3], [5], [7]
EXPECTED_STOCHASTIC_DEPTH = 12          # one StochasticDepth per Swin block
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff")

results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -> {detail}" if detail else ""))


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot():
    """(size, mtime) of every file under checkpoints/ and reports/."""
    snap = {}
    for base in (REPO / "checkpoints", REPO / "reports"):
        if base.is_dir():
            for f in base.rglob("*"):
                if f.is_file():
                    st = f.stat()
                    snap[f.relative_to(REPO).as_posix()] = (st.st_size, st.st_mtime_ns)
    return snap


def run_artifacts(run_name):
    """(size, mtime) of the run's checkpoint and history, None if missing (a real run may already exist)."""
    state = {}
    for path in (REPO / "checkpoints" / f"{run_name}_best.pt", REPO / "reports" / f"{run_name}_history.csv"):
        st = path.stat() if path.exists() else None
        state[path.name] = (st.st_size, st.st_mtime_ns) if st else None
    return state


def random_batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(2, 3, 224, 224, generator=g), torch.tensor([0, 1])


def last_line(text):
    lines = text.strip().splitlines()
    return lines[-1] if lines else ""


# ---------- 17 (before). protected files ----------
final_before = sha(FINAL_CKPT)
files_before = snapshot()
new_run_before = run_artifacts(NEW_RUN)
check("17a. final checkpoint resnet224_ft_aug_best.pt SHA256 before = expected", final_before == FINAL_SHA,
      final_before[:16])

# ---------- 1. versions ----------
print(f"torch {torch.__version__} | torchvision {torchvision.__version__} | python {sys.version.split()[0]}")
check("1. torch / torchvision versions printed", True, f"torch {torch.__version__}, torchvision {torchvision.__version__}")

# ---------- 2. weights enum: transforms() and meta ----------
weights = models.Swin_T_Weights.IMAGENET1K_V1
try:
    preset = weights.transforms()
    preset_ok = callable(preset)
except Exception as error:   # noqa: BLE001 - reported as a failed check
    preset, preset_ok = repr(error), False
meta = getattr(weights, "meta", None)
meta_ok = isinstance(meta, dict) and len(meta.get("categories", [])) == 1000
num_params = meta.get("num_params") if isinstance(meta, dict) else None
check("2a. Swin_T_Weights.IMAGENET1K_V1.transforms() exists", preset_ok)
print(f"    pretrained preset transform (evidence only, NOT used by the project): {preset}")
check("2b. Swin_T_Weights.IMAGENET1K_V1.meta exists (1000 categories)", meta_ok,
      f"num_params {num_params}, min_size {meta.get('min_size') if isinstance(meta, dict) else None}")
check(f"2c. meta num_params = {EXPECTED_META_NUM_PARAMS:,} (design review)", num_params == EXPECTED_META_NUM_PARAMS,
      f"actual {num_params}")

# ---------- 3. pretrained build really loads the ImageNet weights ----------
torch.manual_seed(SEED)
m = swin.build_swin_tiny(num_classes=len(CLASS_TO_IDX), pretrained=True)
reference = weights.get_state_dict(progress=False)
own = m.net.state_dict()
backbone_keys = [k for k in reference if not k.startswith("head.")]
mismatch = [k for k in backbone_keys if k not in own or not torch.equal(own[k], reference[k])]
check("3. pretrained build: every backbone tensor equals the IMAGENET1K_V1 weights",
      backbone_keys and not mismatch, f"{len(backbone_keys)} tensors compared, {len(mismatch)} differ {mismatch[:3]}")

# ---------- 4. real structure ----------
net = m.net
depths = [len(net.features[i]) for i in (1, 3, 5, 7)] if len(net.features) == 8 else None
merging = [type(net.features[i]).__name__ for i in (2, 4, 6)] if len(net.features) == 8 else None
structure = {
    "class": type(net).__name__, "features": len(net.features), "stage_depths": depths, "merging": merging,
    "norm": (type(net.norm).__name__, tuple(getattr(net.norm, "normalized_shape", ()))),
    "head": (type(net.head).__name__, getattr(net.head, "in_features", None), getattr(net.head, "out_features", None),
             getattr(net.head, "bias", None) is not None),
}
print(f"    structure: {structure}")
check("4a. torchvision SwinTransformer: 8 feature parts, stage depths [2, 2, 6, 2], PatchMerging x3, norm LayerNorm(768)",
      structure["class"] == "SwinTransformer" and structure["features"] == 8
      and depths == EXPECTED_STAGE_DEPTHS and merging == ["PatchMerging"] * 3
      and structure["norm"] == ("LayerNorm", (swin.FEATURE_DIM,)), f"{structure}")
check("4b. head = Linear(768, 8) with bias; FEATURE_DIM 768, PRETRAINED_WEIGHTS IMAGENET1K_V1",
      structure["head"] == ("Linear", 768, 8, True) and swin.FEATURE_DIM == 768
      and swin.PRETRAINED_WEIGHTS == "IMAGENET1K_V1", f"{structure['head']}")
m.eval()
with torch.no_grad():
    out_shape = tuple(m(random_batch(0)[0]).shape)
check("4c. forward of 2 x 3 x 224 x 224 gives logits 2 x 8", out_shape == (2, 8), f"{out_shape}")

# ---------- 5. real parameter counts ----------
trainable, frozen = swin.count_params(m)
total = trainable + frozen
stock_total = sum(p.numel() for p in models.swin_t().parameters())   # torchvision model, 1000-class head
print(f"    real counts: trainable {trainable:,} | frozen {frozen:,} | total {total:,} "
      f"| torchvision swin_t (1000 classes) {stock_total:,}")
check(f"5a. trainable = {EXPECTED_TRAINABLE:,} (design review)", trainable == EXPECTED_TRAINABLE, f"actual {trainable:,}")
check(f"5b. frozen = {EXPECTED_FROZEN:,} (design review)", frozen == EXPECTED_FROZEN, f"actual {frozen:,}")
check(f"5c. total = {EXPECTED_TOTAL:,} (design review)", total == EXPECTED_TOTAL, f"actual {total:,}")
check(f"5d. torchvision swin_t total (1000 classes) = {EXPECTED_META_NUM_PARAMS:,}",
      stock_total == EXPECTED_META_NUM_PARAMS, f"actual {stock_total:,}")

# ---------- 6. only the head is trainable ----------
trainable_names = [n for n, p in m.named_parameters() if p.requires_grad]
check("6. only net.head.weight / net.head.bias trainable; trainable_parts = ['head']",
      trainable_names == ["net.head.weight", "net.head.bias"] and swin.trainable_parts(m) == ["head"],
      f"{trainable_names}, {swin.trainable_parts(m)}")

# ---------- 7. modes after model.train() ----------
m.train()
sd_modules = [mod for mod in net.features.modules() if isinstance(mod, StochasticDepth)]
backbone_training = [n for n, mod in net.features.named_modules() if mod.training] + \
                    [f"norm.{n}" for n, mod in net.norm.named_modules() if mod.training]
check("7a. after model.train(): model and head in train mode", m.training and net.head.training)
check("7b. after model.train(): every module of features and norm in eval mode",
      not backbone_training, f"in train mode: {backbone_training[:5]}")
check(f"7c. stochastic depth off: {EXPECTED_STOCHASTIC_DEPTH} StochasticDepth modules, none training",
      len(sd_modules) == EXPECTED_STOCHASTIC_DEPTH and not any(mod.training for mod in sd_modules),
      f"{len(sd_modules)} modules, probabilities {[round(mod.p, 4) for mod in sd_modules]}")

# ---------- 8. two forwards in train mode give identical outputs ----------
images, labels = random_batch(1)
with torch.no_grad():
    first, second = m(images), m(images)
check("8a. two forwards in train mode: identical outputs", torch.equal(first, second),
      f"max diff {(first - second).abs().max().item():.3g}")
torch.manual_seed(SEED)
control = models.swin_t()
control.train()   # plain torchvision model, nothing frozen: stochastic depth active
with torch.no_grad():
    c1, c2 = control(images), control(images)
check("8b. control: a plain swin_t in train mode gives different outputs (the check above can detect stochastic depth)",
      not torch.equal(c1, c2), f"max diff {(c1 - c2).abs().max().item():.3g}")
del control

# ---------- 10. optimizer: exactly one group (head, lr 1e-3) ----------
groups, group_lrs = T.swin_param_groups(m)
optimizer = T.OPTIMIZERS["adam"](groups, weight_decay=T.WEIGHT_DECAY)
check("10. optimizer: exactly 1 parameter group, lr 1e-3, weight_decay 0, holds exactly the head parameters",
      len(optimizer.param_groups) == 1 and optimizer.param_groups[0]["lr"] == 1e-3 == T.SWIN_HEAD_LR
      and optimizer.param_groups[0]["weight_decay"] == 0.0 and group_lrs == {"head": 1e-3}
      and [id(p) for p in optimizer.param_groups[0]["params"]] == [id(p) for p in net.head.parameters()],
      f"{len(optimizer.param_groups)} groups, {group_lrs}")

# ---------- 9. one training step: only the head gets gradients and changes ----------
before = {n: p.detach().clone() for n, p in m.named_parameters()}
m.train()
optimizer.zero_grad(set_to_none=True)
loss = nn.CrossEntropyLoss()(m(images), labels)
loss.backward()
optimizer.step()
with_grad = [n for n, p in m.named_parameters() if p.grad is not None]
moved = [n for n, p in m.named_parameters() if not torch.equal(before[n], p.detach())]
check("9a. training step on 2 x 3 x 224 x 224: finite loss, gradients only in the head",
      torch.isfinite(loss).item() and with_grad == ["net.head.weight", "net.head.bias"], f"loss {loss.item():.4f}, {with_grad}")
check("9b. training step: only head weights changed (features and norm unchanged)",
      moved == ["net.head.weight", "net.head.bias"], f"changed: {moved[:5]}")

# ---------- 11. checkpoint round trip ----------
meta_keys = T.swin_checkpoint_metadata(m, group_lrs, "none")
checkpoint = {"model_state": m.state_dict(), "class_to_idx": CLASS_TO_IDX, "seed": T.SEED,
              "learning_rate": T.LEARNING_RATE, "batch_size": T.BATCH_SIZE, "optimizer": "adam",
              "weight_decay": T.WEIGHT_DECAY, "scheduler": "none", "scheduler_params": None, "loss_name": "ce",
              "epoch": 1, "val_f1": 0.5, **meta_keys}
buffer = io.BytesIO()
torch.save(checkpoint, buffer)
buffer.seek(0)
loaded = torch.load(buffer, map_location="cpu", weights_only=True)   # as analyze_baseline.py loads
fresh = swin.build_swin_tiny(num_classes=len(CLASS_TO_IDX), pretrained=False)
result = fresh.load_state_dict(loaded["model_state"], strict=True)
m.eval()
fresh.eval()
with torch.no_grad():
    same = torch.equal(m(images), fresh(images))
check("11. checkpoint round trip (weights_only=True, strict=True): no missing/unexpected keys, identical outputs",
      not result.missing_keys and not result.unexpected_keys and same)

# ---------- 12. metadata and transform round trip ----------
expected_transform = {"resize": [224, 224], "normalize_mean": [0.485, 0.456, 0.406],
                      "normalize_std": [0.229, 0.224, 0.225], "augmentation": "none", "augmentation_params": None}
check("12a. metadata: architecture, swin_mode, pretrained weights, parts, counts, lrs restored unchanged",
      all(loaded[k] == v for k, v in checkpoint.items() if k != "model_state")
      and loaded["architecture"] == "Swin-Tiny" and loaded["swin_mode"] == "feature_extraction"
      and loaded["pretrained_weights"] == "IMAGENET1K_V1" and loaded["trainable_parts"] == ["head"]
      and (loaded["trainable_params"], loaded["frozen_params"]) == (trainable, frozen)
      and loaded["param_group_lrs"] == {"head": 1e-3} and loaded["image_size"] == 224
      and loaded["warmup_epochs"] is None, f"{ {k: v for k, v in meta_keys.items() if k != 'transform'} }")
recorded = loaded["transform"]
rebuilt = T.resnet_transform(recorded["resize"][0], recorded["normalize_mean"], recorded["normalize_std"])
probe = Image.new("RGB", (300, 180), (120, 60, 200))
check("12b. transform: recorded = Resize 224x224 + ImageNet mean/std, augmentation none; rebuilt = RESNET_TRANSFORM",
      recorded == expected_transform and repr(rebuilt) == repr(T.RESNET_TRANSFORM)
      and torch.equal(rebuilt(probe), T.RESNET_TRANSFORM(probe)), f"{recorded}")
project_steps = [type(t).__name__ for t in T.RESNET_TRANSFORM.transforms]
check("12c. project transform unchanged: Resize((224, 224)) -> ToTensor -> Normalize(ImageNet); 'none' = RESNET_TRANSFORM",
      project_steps == ["Resize", "ToTensor", "Normalize"] and T.RESNET_TRANSFORM.transforms[0].size == (224, 224)
      and T.RESNET_TRAIN_TRANSFORMS["none"] is T.RESNET_TRANSFORM, f"{project_steps}")
import scripts.analyze_baseline as AB   # noqa: E402
source = inspect.getsource(AB.evaluate_checkpoint)
check("12d. analyze_baseline: EVAL_TRANSFORM_NAMES['Swin-Tiny'] = RESNET_TRANSFORM; Swin-Tiny branch uses the recorded transform",
      AB.EVAL_TRANSFORM_NAMES.get("Swin-Tiny") == "RESNET_TRANSFORM"
      and 'checkpoint["architecture"] == "Swin-Tiny"' in source and "build_swin_tiny(" in source)


# ---------- 13 / 14. run-name prefix and overwrite guard (no data read, no epoch run) ----------
class LoadersCalled(Exception):
    pass


captured = {}


def no_loaders(*args, **kwargs):
    captured["args"], captured["kwargs"] = args, kwargs
    raise LoadersCalled("build_loaders reached")


def call_train(run_name, **kw):
    """train() with build_loaders replaced; returns the exception type raised (None if it returned)."""
    try:
        T.train(run_name, DEFAULT_MANIFEST, DEFAULT_CONFIG, model_name="swin_tiny", **kw)
    except (FileExistsError, ValueError, LoadersCalled) as error:
        return type(error)
    return None


original_loaders = T.build_loaders
T.build_loaders = no_loaders
try:
    bad_names = ["swint_fe_none", "swin_fe_none", "convnext_t_fe_none", FINAL_RUN]
    rejected = {name: call_train(name) for name in bad_names}
    check("13. train(): run names without the 'swin_t_' prefix rejected (ValueError, before any data)",
          all(r is ValueError for r in rejected.values()) and T.SWIN_RUN_PREFIX == "swin_t_", f"{rejected}")
    saved_dirs = (T.CHECKPOINT_DIR, T.REPORTS_DIR)
    with tempfile.TemporaryDirectory() as tmp:
        T.CHECKPOINT_DIR, T.REPORTS_DIR = Path(tmp) / "checkpoints", Path(tmp) / "reports"
        T.CHECKPOINT_DIR.mkdir()
        T.REPORTS_DIR.mkdir()
        try:
            (T.CHECKPOINT_DIR / "swin_t_guard_ckpt_best.pt").write_bytes(b"x")
            (T.REPORTS_DIR / "swin_t_guard_hist_history.csv").write_text("x", encoding="utf-8")
            r_ck = call_train("swin_t_guard_ckpt")
            r_hi = call_train("swin_t_guard_hist")
            captured.clear()
            r_new = call_train(NEW_RUN)
            leftover = sorted(p.name for p in Path(tmp).rglob("*") if p.is_file())
        finally:
            T.CHECKPOINT_DIR, T.REPORTS_DIR = saved_dirs
    check("14a. overwrite guard (temporary folder): existing checkpoint only -> refused", r_ck is FileExistsError)
    check("14b. overwrite guard (temporary folder): existing history only -> refused", r_hi is FileExistsError)
    passed_args = captured.get("args", ())
    check(f"14c. '{NEW_RUN}' passes the guard and stops at the replaced data loader; "
          "train and val transform = RESNET_TRANSFORM (no augmentation)",
          r_new is LoadersCalled and len(passed_args) >= 6
          and passed_args[2] is T.RESNET_TRANSFORM and passed_args[5] is T.RESNET_TRANSFORM)
    check("14d. overwrite guard: nothing written besides the two dummy files",
          leftover == ["swin_t_guard_ckpt_best.pt", "swin_t_guard_hist_history.csv"], f"{leftover}")
finally:
    T.build_loaders = original_loaders

# CLI defaults for `--run-name swin_t_fe_none --model swin_tiny` (train() replaced, nothing runs)
original_train, original_argv = T.train, sys.argv
calls = []
T.train = lambda *a, **k: calls.append(inspect.signature(original_train).bind(*a, **k).arguments)
sys.argv = ["src.train", "--run-name", NEW_RUN, "--model", "swin_tiny"]
try:
    T.main()
finally:
    T.train, sys.argv = original_train, original_argv
cfg = calls[0] if calls else {}
want = {"run_name": NEW_RUN, "model_name": "swin_tiny", "epochs": 20, "augmentation": "none", "dropout": 0.0,
        "pooling": "max", "optimizer_name": "adam", "weight_decay": 0.0, "scheduler_name": "none",
        "train_subset": None, "batch_mode": "standard", "loss_name": "ce", "resnet_mode": None}
check("14e. CLI '--run-name swin_t_fe_none --model swin_tiny': epochs 20, augmentation none, adam, wd 0, scheduler none, "
      "CE, standard batches; seed 42, batch 32",
      len(calls) == 1 and all(cfg.get(k) == v for k, v in want.items()) and T.SEED == 42 and T.BATCH_SIZE == 32,
      f"{ {k: cfg.get(k) for k in want} }")

# ---------- 15. invalid CLI combinations exit with code 2 ----------
other_pooling = next(p for p in POOLING_TYPES if p != "max")
cli_cases = [
    ("run name without prefix", "swint_fe_none", []),
    ("ConvNeXt run name", "convnext_t_fe_none", []),
    ("--resnet-mode", NEW_RUN, ["--resnet-mode", "feature_extraction"]),
    ("--resnet-ft-layers", NEW_RUN, ["--resnet-ft-layers", "layer4"]),
    ("--convnext-mode", NEW_RUN, ["--convnext-mode", "feature_extraction"]),
    ("--effnet-mode", NEW_RUN, ["--effnet-mode", "feature_extraction"]),
    ("--effnet-top-lr", NEW_RUN, ["--effnet-top-lr", "1e-4"]),
    ("--augmentation crop_jitter", NEW_RUN, ["--augmentation", "crop_jitter"]),
    ("--dropout", NEW_RUN, ["--dropout", "0.3"]),
    (f"--pooling {other_pooling}", NEW_RUN, ["--pooling", other_pooling]),
    ("--loss bce", NEW_RUN, ["--loss", "bce"]),
    ("--batch-mode balanced", NEW_RUN, ["--batch-mode", "balanced"]),
    ("--train-subset", NEW_RUN, ["--train-subset", "smoke_swin_does_not_exist.csv"]),
]
for label, run_name, extra in cli_cases:
    proc = subprocess.run([sys.executable, "-m", "src.train", "--run-name", run_name, "--model", "swin_tiny", *extra],
                          cwd=REPO, capture_output=True, text=True, timeout=600)
    check(f"15. CLI rejects --model swin_tiny with {label} (exit code 2, before training)",
          proc.returncode == 2 and "error:" in proc.stderr, f"exit {proc.returncode}: {last_line(proc.stderr)}")

# ---------- 18. regression: scripts/verify_resnet.py ----------
proc = subprocess.run([sys.executable, str(REPO / "scripts" / "verify_resnet.py")], cwd=REPO,
                      capture_output=True, text=True, timeout=1800)
n_pass, n_fail = proc.stdout.count("[PASS]"), proc.stdout.count("[FAIL]")
check("18. scripts/verify_resnet.py: 9/9 PASS, exit code 0", proc.returncode == 0 and n_pass == 9 and n_fail == 0,
      f"exit {proc.returncode}, {n_pass} PASS, {n_fail} FAIL")

# ---------- 19. regression: scripts/smoke_resnet_l3l4.py ----------
proc = subprocess.run([sys.executable, str(REPO / "scripts" / "smoke_resnet_l3l4.py")], cwd=REPO,
                      capture_output=True, text=True, timeout=3600)
n_pass, n_fail = proc.stdout.count("[PASS]"), proc.stdout.count("[FAIL]")
check("19. scripts/smoke_resnet_l3l4.py: all PASS, exit code 0", proc.returncode == 0 and n_pass > 0 and n_fail == 0,
      f"exit {proc.returncode}, {n_pass} PASS, {n_fail} FAIL")

# ---------- 20. git diff --check ----------
try:
    proc = subprocess.run(["git", "--no-optional-locks", "diff", "--check"], cwd=REPO, capture_output=True, text=True)
    untracked = subprocess.run(["git", "--no-optional-locks", "ls-files", "--others", "--exclude-standard"], cwd=REPO,
                               capture_output=True, text=True).stdout.split()
    bad_untracked = []
    for name in untracked:
        if name.endswith(".py"):
            data = (REPO / name).read_bytes()
            if b"\r" in data or any(line.rstrip(b" \t") != line for line in data.split(b"\n")) or not data.endswith(b"\n"):
                bad_untracked.append(name)
    check("20. git diff --check clean; new .py files: no trailing whitespace, no CR, final newline",
          proc.returncode == 0 and not proc.stdout.strip() and not bad_untracked,
          f"{last_line(proc.stdout)} {bad_untracked}".strip())
except FileNotFoundError:
    check("20. git diff --check clean", False, "git not found")

# ---------- 16. files opened by this process ----------
def forbidden(path):
    lower = path.replace("\\", "/").lower()
    if lower.endswith(IMAGE_SUFFIXES):
        return True
    if "site-packages" in lower or lower.startswith(sys.prefix.replace("\\", "/").lower()):
        return False
    return any(word in lower for word in ("neysan", "test_frozen", "final_test", "/test/", "evaluate_test"))


flagged = sorted({p for p in OPENED if forbidden(p)})
check("16. no image, Test, Neysan or Final Test file opened by this process", not flagged, f"{flagged[:5]}")

# ---------- 17 (after). protected files ----------
final_after = sha(FINAL_CKPT)
check("17b. final checkpoint SHA256 after = before", final_after == final_before, final_after[:16])
files_after = snapshot()
diff = sorted(set(files_before.items()) ^ set(files_after.items()))
check("17c. no file created or changed in checkpoints/ or reports/", not diff, f"{diff[:3]}")
check(f"17d. artifacts of '{NEW_RUN}' not created or changed by this test (an earlier real run is allowed)",
      run_artifacts(NEW_RUN) == new_run_before, f"{new_run_before}")

print(f"\nreal parameter counts: trainable {trainable:,} | frozen {frozen:,} | total {total:,} "
      f"| torchvision swin_t 1000 classes {stock_total:,} | meta num_params {num_params}")
print(f"final checkpoint SHA256 before {final_before}\nfinal checkpoint SHA256 after  {final_after}")
print(f"\n{sum(results)}/{len(results)} checks passed (no training, no data, Test and Neysan not read)")
sys.exit(0 if all(results) else 1)
