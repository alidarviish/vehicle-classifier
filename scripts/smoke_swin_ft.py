"""Smoke test for Swin-Tiny partial fine-tuning (--model swin_tiny --swin-mode fine_tuning in src/train.py).

No training run, no data, no download, nothing written in the repository:
- networks are built with pretrained=False (random weights); the checks are about structure, parameter counts,
  freezing, train/eval modes, gradient flow, optimizer groups, checkpoint metadata and the run guards, which
  do not depend on weight values; inputs are random tensors, Adam steps are taken on throwaway models only;
- src.train.train() is only called with src.train.build_loaders replaced by a function that raises, so no
  image is read and no epoch can run; the guard checks with dummy files use a temporary folder;
- every file opened by this process is recorded (sys.addaudithook): no image, Test, Neysan or Final Test
  file may be opened.
The expected parameter counts are the values of the design review; a mismatch fails and prints the real values.
It also runs scripts/smoke_swin.py (must report 53/53; it runs scripts/smoke_resnet_l3l4.py, 45/45, and
scripts/verify_resnet.py, 9/9), checks that the final checkpoint checkpoints/resnet224_ft_aug_best.pt is
unchanged, that no file in checkpoints/ or reports/ was created or changed, and that `git diff --check` is clean.

Run from the repository root with the Python used for training:
    python scripts/smoke_swin_ft.py
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
import re
import subprocess
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))  # so "from src ..." works with "python scripts/smoke_swin_ft.py"

import torch
from torch import nn
from torchvision.ops import StochasticDepth

import src.train as T
from src import swin
from src.dataset import CLASS_TO_IDX, DEFAULT_CONFIG, DEFAULT_MANIFEST

FINAL_RUN = "resnet224_ft_aug"
FINAL_CKPT = REPO / "checkpoints" / f"{FINAL_RUN}_best.pt"
FINAL_SHA = "c26280e97f7324ee1ec85d7bf78a58ea0b4d2acb4cf3e975dbd52316366281a1"
FE_RUN = "swin_t_fe_none"
NEW_RUN = "swin_t_ft_none"
SEED = 42
# design review values (torchvision swin_t, 8-class head); a mismatch fails and prints the real numbers
PART_COUNTS = {"features.0": 4_896, "features.1": 224_694, "features.2": 74_496, "features.3": 891_756,
               "features.4": 296_448, "features.5": 10_658_952, "features.6": 1_182_720, "features.7": 14_183_856,
               "norm": 1_536, "head": 6_152}
TOTAL = 27_525_506
WARMUP_COUNTS = (6_152, 27_519_354)        # (trainable, frozen): head only
FT_COUNTS = (15_374_264, 12_151_242)       # head + features[6:8] + norm
TOP_GROUP_PARAMS = 15_368_112              # features[6] + features[7] + norm
FT_PARTS = ["features.6", "features.7", "norm", "head"]
PARTS = list(PART_COUNTS)
FROZEN_PARTS = [f"features.{i}" for i in range(6)]
SD_FROZEN, SD_TOP = 10, 2                  # StochasticDepth modules in features[0:6] / features[7]
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


def new_model():
    torch.manual_seed(SEED)
    return swin.build_swin_tiny(num_classes=len(CLASS_TO_IDX), pretrained=False)


def random_batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(2, 3, 224, 224, generator=g), torch.tensor([0, 1])


def part_of(name):
    """'features.5' for 'net.features.5.3.attn.qkv.weight', 'norm' / 'head' otherwise."""
    pieces = name.split(".")
    return f"features.{pieces[2]}" if pieces[1] == "features" else pieces[1]


def part_counts(model):
    counts = dict.fromkeys(PARTS, 0)
    for name, p in model.named_parameters():
        counts[part_of(name)] += p.numel()
    return counts


def one_step(model, optimizer, seed):
    """One Adam step in train mode; returns (parts with a gradient, parts whose weights changed)."""
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    model.train()
    optimizer.zero_grad(set_to_none=True)
    images, labels = random_batch(seed)
    nn.CrossEntropyLoss()(model(images), labels).backward()
    optimizer.step()
    with_grad = sorted({part_of(n) for n, p in model.named_parameters() if p.grad is not None}, key=PARTS.index)
    moved = sorted({part_of(n) for n, p in model.named_parameters() if not torch.equal(before[n], p.detach())},
                   key=PARTS.index)
    return with_grad, moved


def training_modules(module):
    return [n for n, m in module.named_modules() if m.training]


def last_line(text):
    lines = text.strip().splitlines()
    return lines[-1] if lines else ""


# ---------- 0. protected files before ----------
final_before = sha(FINAL_CKPT)
files_before = snapshot()
fe_before, ft_before = run_artifacts(FE_RUN), run_artifacts(NEW_RUN)
check("0. final checkpoint resnet224_ft_aug_best.pt SHA256 before = expected", final_before == FINAL_SHA,
      final_before[:16])
print(f"    existing artifacts: {FE_RUN} {fe_before} | {NEW_RUN} {ft_before}")

# ---------- 1. constants and API ----------
check("1a. SWIN_MODES, SWIN_HEAD_LR 1e-3, SWIN_TOP_LR 1e-4, warm-up 5 epochs, prefix 'swin_t_'",
      T.SWIN_MODES == ("feature_extraction", "fine_tuning") and T.SWIN_HEAD_LR == 1e-3 and T.SWIN_TOP_LR == 1e-4
      and T.FT_WARMUP_EPOCHS == 5 and T.SWIN_RUN_PREFIX == "swin_t_")
check("1b. swin.UNFREEZE_STAGES = (6, 7); top_stage_parameters / unfreeze_top_stages exist",
      swin.UNFREEZE_STAGES == (6, 7) and callable(swin.top_stage_parameters) and callable(swin.unfreeze_top_stages))
check("1c. train() and CLI default: swin_mode 'feature_extraction'",
      inspect.signature(T.train).parameters["swin_mode"].default == "feature_extraction")

# ---------- 2. parameter counts per part ----------
m = new_model()
counts = part_counts(m)
print(f"    real parameter counts per part: { {k: f'{v:,}' for k, v in counts.items()} }")
check("2a. parameters per part = design review (features[0..7], norm, head)", counts == PART_COUNTS,
      f"differs: { {k: (v, PART_COUNTS[k]) for k, v in counts.items() if v != PART_COUNTS[k]} }")
check(f"2b. total = {TOTAL:,}", sum(counts.values()) == TOTAL, f"actual {sum(counts.values()):,}")

# ---------- 3. warm-up (epochs 1-5) ----------
groups, group_lrs = T.swin_param_groups(m, "fine_tuning")
optimizer = T.OPTIMIZERS["adam"](groups, weight_decay=T.WEIGHT_DECAY)
check(f"3a. warm-up counts: trainable {WARMUP_COUNTS[0]:,} / frozen {WARMUP_COUNTS[1]:,}",
      swin.count_params(m) == WARMUP_COUNTS, f"actual {swin.count_params(m)}")
check("3b. warm-up trainable parts = ['head']", swin.trainable_parts(m) == ["head"], f"{swin.trainable_parts(m)}")

# ---------- 4. optimizer groups ----------
ids = [[id(p) for p in g["params"]] for g in optimizer.param_groups]
expected_ids = [[id(p) for p in m.net.head.parameters()],
                [id(p) for i in (6, 7) for p in m.net.features[i].parameters()] + [id(p) for p in m.net.norm.parameters()]]
flat = [i for g in ids for i in g]
sizes = [sum(p.numel() for p in g["params"]) for g in optimizer.param_groups]
check("4a. optimizer: exactly 2 groups, lrs [1e-3, 1e-4], weight_decay 0; names {head, features.6-7+norm}",
      len(optimizer.param_groups) == 2 and [g["lr"] for g in optimizer.param_groups] == [1e-3, 1e-4]
      and all(g["weight_decay"] == 0.0 for g in optimizer.param_groups)
      and group_lrs == {"head": 1e-3, T.SWIN_TOP_GROUP: 1e-4} and T.SWIN_TOP_GROUP == "features.6-7+norm",
      f"{group_lrs}")
check(f"4b. group 1 = exactly the head ({PART_COUNTS['head']:,}), group 2 = exactly features[6], features[7], norm "
      f"({TOP_GROUP_PARAMS:,}); no parameter in two groups",
      ids == expected_ids and len(flat) == len(set(flat)) and sizes == [PART_COUNTS["head"], TOP_GROUP_PARAMS],
      f"group sizes {sizes}")
fe_groups, fe_lrs = T.swin_param_groups(new_model())
check("4c. feature extraction (default) unchanged: 1 group, head 1e-3",
      len(fe_groups) == 1 and fe_lrs == {"head": 1e-3})

# ---------- 5. warm-up train/eval and training step ----------
m.train()
sd_all = [mod for mod in m.net.features.modules() if isinstance(mod, StochasticDepth)]
check("5a. warm-up after model.train(): every module of features and norm in eval, head in train",
      not training_modules(m.net.features) and not training_modules(m.net.norm) and m.net.head.training,
      f"in train mode: {(training_modules(m.net.features) + training_modules(m.net.norm))[:5]}")
check(f"5b. warm-up: {SD_FROZEN + SD_TOP} StochasticDepth modules, none training",
      len(sd_all) == SD_FROZEN + SD_TOP and not any(mod.training for mod in sd_all), f"{len(sd_all)} modules")
with_grad, moved = one_step(m, optimizer, seed=0)
check("5c. warm-up step: gradients only in the head", with_grad == ["head"], f"{with_grad}")
check("5d. warm-up step: only head weights changed", moved == ["head"], f"{moved}")

# ---------- 6. after the unfreeze (epoch 6 on) ----------
swin.unfreeze_top_stages(m)
check(f"6a. after unfreeze: trainable {FT_COUNTS[0]:,} / frozen {FT_COUNTS[1]:,} (total {TOTAL:,})",
      swin.count_params(m) == FT_COUNTS and sum(swin.count_params(m)) == TOTAL, f"actual {swin.count_params(m)}")
check(f"6b. after unfreeze: trainable parts = {FT_PARTS}", swin.trainable_parts(m) == FT_PARTS,
      f"{swin.trainable_parts(m)}")
check("6c. after unfreeze: features[0:6] still fully frozen",
      not any(p.requires_grad for i in range(6) for p in m.net.features[i].parameters()))
m.train()
sd_frozen = [mod for i in range(6) for mod in m.net.features[i].modules() if isinstance(mod, StochasticDepth)]
sd_top = [mod for mod in m.net.features[7].modules() if isinstance(mod, StochasticDepth)]
check("6d. after model.train(): every module of features[0:6] in eval",
      not any(training_modules(m.net.features[i]) for i in range(6)))
check(f"6e. stochastic depth of the frozen sections inactive ({SD_FROZEN} modules, none training)",
      len(sd_frozen) == SD_FROZEN and not any(mod.training for mod in sd_frozen), f"{len(sd_frozen)} modules")
check("6f. features[6], features[7] (every module) and norm in train mode; norm trainable",
      all(mod.training for i in (6, 7) for mod in m.net.features[i].modules())
      and m.net.norm.training and swin.is_trainable(m.net.norm))
check(f"6g. stochastic depth of stage 4 active in train mode ({SD_TOP} modules, all training)",
      len(sd_top) == SD_TOP and all(mod.training for mod in sd_top),
      f"{len(sd_top)} modules, p {[round(mod.p, 4) for mod in sd_top]}")
with_grad, moved = one_step(m, optimizer, seed=1)
check("6h. after unfreeze, step: gradients only in features.6, features.7, norm, head", with_grad == FT_PARTS,
      f"{with_grad}")
check("6i. after unfreeze, step: only features.6, features.7, norm, head changed; features[0:6] unchanged",
      moved == FT_PARTS, f"{moved}")
m.eval()
images, _ = random_batch(2)
with torch.no_grad():
    first, second = m(images), m(images)
check("6j. model.eval(): no module in train mode, two forwards identical",
      not training_modules(m) and torch.equal(first, second))

# ---------- 7. checkpoint metadata and round trip ----------
meta = T.swin_checkpoint_metadata(m, group_lrs, "none", "fine_tuning")
checkpoint = {"model_state": m.state_dict(), "class_to_idx": CLASS_TO_IDX, "seed": T.SEED, "epoch": 6,
              "val_f1": 0.5, **meta}
buffer = io.BytesIO()
torch.save(checkpoint, buffer)
buffer.seek(0)
loaded = torch.load(buffer, map_location="cpu", weights_only=True)   # as analyze_baseline.py loads
check("7a. metadata: swin_mode fine_tuning, warmup_epochs 5, unfrozen_stages [6, 7], parts, counts, group lrs",
      loaded["architecture"] == "Swin-Tiny" and loaded["swin_mode"] == "fine_tuning" and loaded["warmup_epochs"] == 5
      and loaded["unfrozen_stages"] == [6, 7] and loaded["trainable_parts"] == FT_PARTS
      and (loaded["trainable_params"], loaded["frozen_params"]) == FT_COUNTS
      and loaded["param_group_lrs"] == {"head": 1e-3, "features.6-7+norm": 1e-4}
      and loaded["pretrained_weights"] == "IMAGENET1K_V1",
      f"{ {k: v for k, v in meta.items() if k != 'transform'} }")
check("7b. metadata round trip (weights_only=True): all keys restored unchanged; transform = project transform, none",
      all(loaded[k] == v for k, v in checkpoint.items() if k != "model_state")
      and loaded["transform"] == {"resize": [224, 224], "normalize_mean": [0.485, 0.456, 0.406],
                                  "normalize_std": [0.229, 0.224, 0.225], "augmentation": "none",
                                  "augmentation_params": None})
fresh = new_model()
result = fresh.load_state_dict(loaded["model_state"], strict=True)
fresh.eval()
with torch.no_grad():
    same = torch.equal(m(images), fresh(images))
check("7c. checkpoint round trip (strict=True): no missing/unexpected keys, identical eval outputs",
      not result.missing_keys and not result.unexpected_keys and same)
fe = new_model()
fe_meta = T.swin_checkpoint_metadata(fe, {"head": 1e-3}, "none")
check("7d. feature extraction metadata unchanged (default): swin_mode feature_extraction, warmup / unfrozen None, "
      "parts ['head'], 6,152 / 27,519,354",
      fe_meta["swin_mode"] == "feature_extraction" and fe_meta["warmup_epochs"] is None
      and fe_meta["unfrozen_stages"] is None and fe_meta["trainable_parts"] == ["head"]
      and (fe_meta["trainable_params"], fe_meta["frozen_params"]) == WARMUP_COUNTS)


# ---------- 8. train() guards (no data read, no epoch run) ----------
class LoadersCalled(Exception):
    pass


captured = {}


def no_loaders(*args, **kwargs):
    captured["args"] = args
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
    check("8a. train(): unknown swin_mode rejected (ValueError)", call_train(NEW_RUN, swin_mode="full") is ValueError)
    rejected = {name: call_train(name, swin_mode="fine_tuning")
                for name in ("swint_ft_none", "convnext_t_ft_none", FINAL_RUN)}
    check("8b. train(): run names without the 'swin_t_' prefix rejected (ValueError)",
          all(r is ValueError for r in rejected.values()), f"{rejected}")
    for run, state in ((FE_RUN, fe_before), (NEW_RUN, ft_before)):
        exists = any(v is not None for v in state.values())
        expected = FileExistsError if exists else LoadersCalled
        got = {mode: call_train(run, swin_mode=mode) for mode in T.SWIN_MODES}
        check(f"8c. real folders: '{run}' {'exists -> refused' if exists else 'does not exist -> passes the guard'} "
              "in both modes", all(r is expected for r in got.values()), f"{got}")
    saved_dirs = (T.CHECKPOINT_DIR, T.REPORTS_DIR)
    with tempfile.TemporaryDirectory() as tmp:
        T.CHECKPOINT_DIR, T.REPORTS_DIR = Path(tmp) / "checkpoints", Path(tmp) / "reports"
        T.CHECKPOINT_DIR.mkdir()
        T.REPORTS_DIR.mkdir()
        try:
            (T.CHECKPOINT_DIR / f"{NEW_RUN}_best.pt").write_bytes(b"x")
            (T.REPORTS_DIR / f"{FE_RUN}_history.csv").write_text("x", encoding="utf-8")
            r_ft = call_train(NEW_RUN, swin_mode="fine_tuning")
            r_fe = call_train(FE_RUN, swin_mode="feature_extraction")
            captured.clear()
            r_new = call_train("swin_t_ft_smoke_new", swin_mode="fine_tuning")
            leftover = sorted(p.name for p in Path(tmp).rglob("*") if p.is_file())
        finally:
            T.CHECKPOINT_DIR, T.REPORTS_DIR = saved_dirs
    check(f"8d. temporary folder: existing checkpoint of '{NEW_RUN}' -> refused", r_ft is FileExistsError)
    check(f"8e. temporary folder: existing history of '{FE_RUN}' -> refused", r_fe is FileExistsError)
    passed = captured.get("args", ())
    check("8f. temporary folder: new swin_t_ run passes the guard, stops at the replaced data loader; "
          "train and val transform = RESNET_TRANSFORM",
          r_new is LoadersCalled and len(passed) >= 6 and passed[2] is T.RESNET_TRANSFORM
          and passed[5] is T.RESNET_TRANSFORM)
    check("8g. temporary folder: nothing written besides the two dummy files",
          leftover == [f"{FE_RUN}_history.csv", f"{NEW_RUN}_best.pt"], f"{leftover}")
finally:
    T.build_loaders = original_loaders

# ---------- 9. CLI ----------
original_train, original_argv = T.train, sys.argv
configs = {}
T.train = lambda *a, **k: configs.setdefault(sys.argv[2], inspect.signature(original_train).bind(*a, **k).arguments)
try:
    for run, extra in ((NEW_RUN, ["--swin-mode", "fine_tuning"]), (FE_RUN, [])):
        sys.argv = ["src.train", "--run-name", run, "--model", "swin_tiny", *extra]
        T.main()
finally:
    T.train, sys.argv = original_train, original_argv
want = {"model_name": "swin_tiny", "epochs": 20, "augmentation": "none", "optimizer_name": "adam",
        "weight_decay": 0.0, "scheduler_name": "none", "loss_name": "ce", "batch_mode": "standard",
        "train_subset": None}
ft_cfg, fe_cfg = configs.get(NEW_RUN, {}), configs.get(FE_RUN, {})
check(f"9a. CLI '--run-name {NEW_RUN} --model swin_tiny --swin-mode fine_tuning': swin_mode fine_tuning, "
      "20 epochs, augmentation none, adam, wd 0, scheduler none, CE",
      ft_cfg.get("swin_mode") == "fine_tuning" and all(ft_cfg.get(k) == v for k, v in want.items()),
      f"{ {k: ft_cfg.get(k) for k in [*want, 'swin_mode']} }")
check(f"9b. CLI '--run-name {FE_RUN} --model swin_tiny' (no --swin-mode): swin_mode feature_extraction",
      fe_cfg.get("swin_mode") == "feature_extraction" and all(fe_cfg.get(k) == v for k, v in want.items()))
cli_cases = [
    ("--swin-mode with convnext_tiny", ["--run-name", "convnext_t_smoke", "--model", "convnext_tiny",
                                       "--swin-mode", "fine_tuning"]),
    ("--swin-mode with resnet18", ["--run-name", "smoke_cli", "--model", "resnet18", "--resnet-mode",
                                  "fine_tuning", "--swin-mode", "fine_tuning"]),
    ("--swin-mode with efficientnet_b0", ["--run-name", "effnet_b0_smoke", "--model", "efficientnet_b0",
                                         "--swin-mode", "fine_tuning"]),
    ("--swin-mode with baseline", ["--run-name", "smoke_cli", "--swin-mode", "feature_extraction"]),
    ("unknown --swin-mode value", ["--run-name", NEW_RUN, "--model", "swin_tiny", "--swin-mode", "full"]),
    ("swin fine_tuning without the swin_t_ prefix", ["--run-name", "swint_ft_none", "--model", "swin_tiny",
                                                     "--swin-mode", "fine_tuning"]),
    ("swin fine_tuning with --convnext-mode", ["--run-name", NEW_RUN, "--model", "swin_tiny", "--swin-mode",
                                               "fine_tuning", "--convnext-mode", "fine_tuning"]),
    ("swin fine_tuning with --effnet-top-lr", ["--run-name", NEW_RUN, "--model", "swin_tiny", "--swin-mode",
                                               "fine_tuning", "--effnet-top-lr", "1e-4"]),
    ("swin fine_tuning with --resnet-ft-layers", ["--run-name", NEW_RUN, "--model", "swin_tiny", "--swin-mode",
                                                  "fine_tuning", "--resnet-ft-layers", "layer4"]),
    ("swin fine_tuning with --loss bce", ["--run-name", NEW_RUN, "--model", "swin_tiny", "--swin-mode",
                                          "fine_tuning", "--loss", "bce"]),
    ("swin fine_tuning with --augmentation crop_jitter", ["--run-name", NEW_RUN, "--model", "swin_tiny",
                                                          "--swin-mode", "fine_tuning", "--augmentation",
                                                          "crop_jitter"]),
]
for label, argv in cli_cases:
    proc = subprocess.run([sys.executable, "-m", "src.train", *argv], cwd=REPO, capture_output=True, text=True,
                          timeout=600)
    check(f"9c. CLI rejects {label} (exit code 2, before training)",
          proc.returncode == 2 and "error:" in proc.stderr, f"exit {proc.returncode}: {last_line(proc.stderr)}")

# ---------- 10. regression: smoke_swin.py (includes smoke_resnet_l3l4.py and verify_resnet.py) ----------
proc = subprocess.run([sys.executable, str(REPO / "scripts" / "smoke_swin.py")], cwd=REPO,
                      capture_output=True, text=True, timeout=7200)
line18 = next((l for l in proc.stdout.splitlines() if "] 18. " in l), "")
line19 = next((l for l in proc.stdout.splitlines() if "] 19. " in l), "")
summary = re.search(r"(\d+)/(\d+) checks passed", proc.stdout)
check("10a. scripts/smoke_swin.py: 53/53, exit code 0",
      proc.returncode == 0 and summary is not None and summary.groups() == ("53", "53"),
      f"exit {proc.returncode}, {summary.group(0) if summary else 'no summary'}")
check("10b. scripts/smoke_resnet_l3l4.py (run by smoke_swin.py): 45 PASS, 0 FAIL",
      line19.startswith("[PASS]") and "45 PASS, 0 FAIL" in line19, line19.split("->")[-1].strip())
check("10c. scripts/verify_resnet.py (run by smoke_swin.py): 9 PASS, 0 FAIL",
      line18.startswith("[PASS]") and "9 PASS, 0 FAIL" in line18, line18.split("->")[-1].strip())

# ---------- 11. git diff --check ----------
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
    check("11. git diff --check clean; new .py files: no trailing whitespace, no CR, final newline",
          proc.returncode == 0 and not proc.stdout.strip() and not bad_untracked,
          f"{last_line(proc.stdout)} {bad_untracked}".strip())
except FileNotFoundError:
    check("11. git diff --check clean", False, "git not found")


# ---------- 12. files opened by this process ----------
def forbidden(path):
    lower = path.replace("\\", "/").lower()
    if lower.endswith(IMAGE_SUFFIXES):
        return True
    if "site-packages" in lower or lower.startswith(sys.prefix.replace("\\", "/").lower()):
        return False
    return any(word in lower for word in ("neysan", "test_frozen", "final_test", "/test/", "evaluate_test"))


flagged = sorted({p for p in OPENED if forbidden(p)})
check("12. no image, Test, Neysan or Final Test file opened by this process", not flagged, f"{flagged[:5]}")

# ---------- 13. protected files after ----------
final_after = sha(FINAL_CKPT)
check("13a. final checkpoint SHA256 after = before", final_after == final_before, final_after[:16])
files_after = snapshot()
diff = sorted(set(files_before.items()) ^ set(files_after.items()))
check("13b. no file created or changed in checkpoints/ or reports/", not diff, f"{diff[:3]}")
check(f"13c. artifacts of '{FE_RUN}' and '{NEW_RUN}' not created or changed by this test",
      run_artifacts(FE_RUN) == fe_before and run_artifacts(NEW_RUN) == ft_before)

print(f"\nfinal checkpoint SHA256 before {final_before}\nfinal checkpoint SHA256 after  {final_after}")
print(f"\n{sum(results)}/{len(results)} checks passed (no training, no data, Test and Neysan not read)")
sys.exit(0 if all(results) else 1)
