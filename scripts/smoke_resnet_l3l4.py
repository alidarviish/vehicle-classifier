"""Smoke test for the ResNet18 layer3 + layer4 fine-tuning option (--resnet-ft-layers layer3_layer4).

No training run, no data, no download, nothing written in the repository:
- networks are built with pretrained=False (random weights); the checks are about trainability, BatchNorm
  modes, optimizer groups, checkpoint metadata and the overwrite guard, which do not depend on weight values;
- inputs are random tensors; one Adam step is taken on throwaway models only;
- src.train.train() is only called with src.train.build_loaders replaced by a function that raises, so no
  image is read and no epoch can run; the guard checks with temporary files use a temporary folder;
- Test and Neysan files are never opened.
It also runs scripts/verify_resnet.py (must report 9/9) and checks that the final checkpoint
checkpoints/resnet224_ft_aug_best.pt is unchanged and that no file in checkpoints/ or reports/ was created
or changed.

Run from the repository root with the Python used for training:
    python scripts/smoke_resnet_l3l4.py
Exit code 0 = all checks passed.
"""
import hashlib
import io
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))  # so "from src ..." works with "python scripts/smoke_resnet_l3l4.py"

import torch
from torch import nn

import src.train as T
from src.dataset import CLASS_TO_IDX, DEFAULT_CONFIG, DEFAULT_MANIFEST
from src.resnet import build_resnet18, count_params, trainable_parts, unfreeze_layer4, unfreeze_layers

FINAL_RUN = "resnet224_ft_aug"
FINAL_CKPT = REPO / "checkpoints" / f"{FINAL_RUN}_best.pt"
FINAL_SHA = "c26280e97f7324ee1ec85d7bf78a58ea0b4d2acb4cf3e975dbd52316366281a1"
NEW_RUN = "resnet224_ft_l3l4_aug"
SEED = 42
FE_COUNTS = (4_104, 11_176_512)          # (trainable, frozen): fc only (warm-up)
L4_COUNTS = (8_397_832, 2_782_784)       # fc + layer4 (default fine-tuning, final model)
L3L4_COUNTS = (10_497_544, 683_072)      # fc + layer3 + layer4
TOTAL = 11_180_616
FROZEN_L3L4 = ("conv1", "bn1", "layer1", "layer2")

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
    return build_resnet18(num_classes=len(CLASS_TO_IDX), pretrained=False)


def random_batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(2, 3, 224, 224, generator=g), torch.tensor([0, 1])


def params_of(model, part):
    return dict(getattr(model.net, part).named_parameters())


def bn_modules(model, part):
    return [m for m in getattr(model.net, part).modules() if isinstance(m, nn.BatchNorm2d)] \
        if isinstance(getattr(model.net, part), nn.Module) else []


def bn_state(model, parts):
    return {f"{part}.{i}.{k}": v.detach().clone() for part in parts
            for i, m in enumerate(bn_modules(model, part)) for k, v in
            (("running_mean", m.running_mean), ("running_var", m.running_var))}


def one_step(model, optimizer, seed=0):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    images, labels = random_batch(seed)
    nn.CrossEntropyLoss()(model(images), labels).backward()
    optimizer.step()


def changed(before, model, parts):
    return {part: any(not torch.equal(before[part][n], p.detach()) for n, p in params_of(model, part).items())
            for part in parts}


PARTS = ("conv1", "bn1", "layer1", "layer2", "layer3", "layer4", "fc")

# ---------- 0. protected files ----------
final_before = sha(FINAL_CKPT)
files_before = snapshot()
new_run_before = run_artifacts(NEW_RUN)
check("final checkpoint resnet224_ft_aug_best.pt SHA256 before", final_before == FINAL_SHA, final_before[:16])

# ---------- 1. API and constants ----------
check("RESNET_FT_LAYERS = {layer4: (layer4,), layer3_layer4: (layer3, layer4)}, default layer4",
      T.RESNET_FT_LAYERS == {"layer4": ("layer4",), "layer3_layer4": ("layer3", "layer4")}
      and T.RESNET_DEFAULT_FT_LAYERS == "layer4")
check("learning rates unchanged: head 1e-3, backbone layers 1e-4, warm-up 5",
      T.RESNET_HEAD_LR == 1e-3 and T.RESNET_LAYER4_LR == 1e-4 and T.FT_WARMUP_EPOCHS == 5)
bad = []
for names in [(), ("fc",), ("layer5",), ("conv1",)]:
    try:
        unfreeze_layers(new_model(), names)
        bad.append(names)
    except ValueError:
        pass
check("unfreeze_layers rejects empty / unknown / non-layer names", not bad, f"accepted: {bad}" if bad else "")

# ---------- 2. layer3_layer4: warm-up (epochs 1-5) ----------
m = new_model()
groups, group_lrs = T.resnet_param_groups(m, "fine_tuning", "layer3_layer4")
check("warm-up counts: trainable 4,104 / frozen 11,176,512 (total 11,180,616)",
      count_params(m) == FE_COUNTS and sum(count_params(m)) == TOTAL, f"{count_params(m)}")
check("warm-up trainable parts = ['fc']", trainable_parts(m) == ["fc"], f"{trainable_parts(m)}")
check("optimizer groups: exactly 3, lrs [fc 1e-3, layer3 1e-4, layer4 1e-4]",
      len(groups) == 3 and [g["lr"] for g in groups] == [1e-3, 1e-4, 1e-4]
      and group_lrs == {"fc": 1e-3, "layer3": 1e-4, "layer4": 1e-4}, f"{group_lrs}")
ids = [[id(p) for p in g["params"]] for g in groups]
expected = [[id(p) for p in getattr(m.net, part).parameters()] for part in ("fc", "layer3", "layer4")]
flat = [i for g in ids for i in g]
check("each group holds exactly the parameters of fc / layer3 / layer4; no parameter in two groups",
      ids == expected and len(flat) == len(set(flat)))
check("grouped parameters = 10,497,544 (every parameter that will be trained)",
      sum(p.numel() for g in groups for p in g["params"]) == L3L4_COUNTS[0])
optimizer = T.OPTIMIZERS["adam"](groups, weight_decay=T.WEIGHT_DECAY)
m.train()
check("warm-up: every backbone BatchNorm in eval mode",
      all(not bn.training for part in PARTS[:-1] for bn in bn_modules(m, part)))
before = {part: {n: p.detach().clone() for n, p in params_of(m, part).items()} for part in PARTS}
bn_before = bn_state(m, PARTS[:-1])
one_step(m, optimizer, seed=0)
grads = {part: any(p.grad is not None for p in getattr(m.net, part).parameters()) for part in PARTS}
moved = changed(before, m, PARTS)
check("warm-up step: only fc has gradients", grads == {p: p == "fc" for p in PARTS}, f"{grads}")
check("warm-up step: only fc weights updated", moved == {p: p == "fc" for p in PARTS}, f"{moved}")
bn_after = bn_state(m, PARTS[:-1])
check("warm-up step: running stats of all backbone BatchNorm unchanged",
      all(torch.equal(bn_before[k], bn_after[k]) for k in bn_before))

# ---------- 3. layer3_layer4: after the warm-up (epoch 6 on) ----------
unfreeze_layers(m, T.RESNET_FT_LAYERS["layer3_layer4"])
check("after unfreeze: trainable 10,497,544 / frozen 683,072", count_params(m) == L3L4_COUNTS, f"{count_params(m)}")
check("after unfreeze: trainable parts = ['layer3', 'layer4', 'fc']",
      trainable_parts(m) == ["layer3", "layer4", "fc"], f"{trainable_parts(m)}")
m.train()
check("after unfreeze: BatchNorm of layer3 and layer4 in train mode",
      all(bn.training for part in ("layer3", "layer4") for bn in bn_modules(m, part)))
check("after unfreeze: BatchNorm of bn1 / layer1 / layer2 still in eval mode",
      all(not bn.training for part in ("bn1", "layer1", "layer2") for bn in bn_modules(m, part)))
before = {part: {n: p.detach().clone() for n, p in params_of(m, part).items()} for part in PARTS}
bn_before = bn_state(m, PARTS[:-1])
one_step(m, optimizer, seed=1)
grads = {part: any(p.grad is not None for p in getattr(m.net, part).parameters()) for part in PARTS}
moved = changed(before, m, PARTS)
want = {p: p in ("layer3", "layer4", "fc") for p in PARTS}
check("after unfreeze: gradients in layer3, layer4, fc only", grads == want, f"{grads}")
check("after unfreeze: weights of layer3, layer4, fc updated; conv1 / bn1 / layer1 / layer2 unchanged",
      moved == want, f"{moved}")
check("after unfreeze: frozen parameters have no gradient",
      all(p.grad is None for part in FROZEN_L3L4 for p in getattr(m.net, part).parameters()))
bn_after = bn_state(m, PARTS[:-1])
frozen_keys = [k for k in bn_before if k.split(".")[0] in FROZEN_L3L4]
open_keys = [k for k in bn_before if k.split(".")[0] in ("layer3", "layer4")]
check("after unfreeze: running stats of frozen BatchNorm (bn1, layer1, layer2) unchanged",
      all(torch.equal(bn_before[k], bn_after[k]) for k in frozen_keys))
check("after unfreeze: running stats of layer3 / layer4 BatchNorm updated",
      all(any(not torch.equal(bn_before[k], bn_after[k]) for k in open_keys if k.startswith(part + "."))
          for part in ("layer3", "layer4")))

# ---------- 4. checkpoint metadata and round trip ----------
meta = T.resnet_checkpoint_metadata(m, "fine_tuning", "layer3_layer4", group_lrs)
check("metadata: resnet_ft_layers, unfrozen_layers, trainable_parts, param_group_lrs, counts",
      meta["resnet_ft_layers"] == "layer3_layer4" and meta["unfrozen_layers"] == ["layer3", "layer4"]
      and meta["trainable_parts"] == ["layer3", "layer4", "fc"]
      and meta["param_group_lrs"] == {"fc": 1e-3, "layer3": 1e-4, "layer4": 1e-4}
      and (meta["trainable_params"], meta["frozen_params"]) == L3L4_COUNTS
      and meta["resnet_mode"] == "fine_tuning" and meta["warmup_epochs"] == 5, f"{meta}")
ckpt = {"model_state": m.state_dict(), "class_to_idx": CLASS_TO_IDX, "architecture": "ResNet18", **meta}
buffer = io.BytesIO()
torch.save(ckpt, buffer)
buffer.seek(0)
loaded = torch.load(buffer, map_location="cpu", weights_only=True)   # as analyze_baseline.py / predict.py load
check("checkpoint round trip (weights_only=True): all metadata restored unchanged",
      all(loaded[k] == v for k, v in ckpt.items() if k != "model_state"))
other = new_model()
other.load_state_dict(loaded["model_state"], strict=True)
m.eval()
other.eval()
images, _ = random_batch(2)
with torch.no_grad():
    same = torch.allclose(m(images), other(images), rtol=0, atol=1e-6)
check("checkpoint round trip: strict load into a fresh ResNet18, identical outputs", same)

# ---------- 5. default behaviour unchanged ----------
d = new_model()
d_groups, d_lrs = T.resnet_param_groups(d, "fine_tuning")
check("default fine-tuning: 2 groups [fc 1e-3, layer4 1e-4]",
      len(d_groups) == 2 and d_lrs == {"fc": 1e-3, "layer4": 1e-4}
      and [id(p) for p in d_groups[1]["params"]] == [id(p) for p in d.net.layer4.parameters()], f"{d_lrs}")
unfreeze_layers(d, T.RESNET_FT_LAYERS[T.RESNET_DEFAULT_FT_LAYERS])
w = new_model()
unfreeze_layer4(w)
check("default fine-tuning after unfreeze: 8,397,832 / 2,782,784, parts ['layer4', 'fc']; unfreeze_layer4 identical",
      count_params(d) == L4_COUNTS == count_params(w) and trainable_parts(d) == ["layer4", "fc"] == trainable_parts(w))
d_meta = T.resnet_checkpoint_metadata(d, "fine_tuning", "layer4", d_lrs)
check("default metadata: resnet_ft_layers 'layer4', unfrozen_layers ['layer4']",
      d_meta["resnet_ft_layers"] == "layer4" and d_meta["unfrozen_layers"] == ["layer4"])
f = new_model()
f_groups, f_lrs = T.resnet_param_groups(f, "feature_extraction")
f_meta = T.resnet_checkpoint_metadata(f, "feature_extraction", "layer4", f_lrs)
check("feature extraction: 1 group [fc 1e-3]; resnet_ft_layers / unfrozen_layers / warmup_epochs None",
      len(f_groups) == 1 and f_lrs == {"fc": 1e-3} and f_meta["resnet_ft_layers"] is None
      and f_meta["unfrozen_layers"] is None and f_meta["warmup_epochs"] is None)


# ---------- 6. rejected configurations and overwrite guard (no data read, no epoch run) ----------
class LoadersCalled(Exception):
    pass


def no_loaders(*args, **kwargs):
    raise LoadersCalled("build_loaders reached")


def call_train(run_name, **kw):
    """train() with build_loaders replaced; returns the exception type raised (None if it returned)."""
    try:
        T.train(run_name, DEFAULT_MANIFEST, DEFAULT_CONFIG, model_name="resnet18", augmentation="full_aug", **kw)
    except (FileExistsError, ValueError, LoadersCalled) as error:
        return type(error)
    return None


original_loaders = T.build_loaders
T.build_loaders = no_loaders
try:
    check("train(): layer3_layer4 with feature_extraction rejected (ValueError)",
          call_train(NEW_RUN, resnet_mode="feature_extraction", resnet_ft_layers="layer3_layer4") is ValueError)
    check("train(): unknown resnet_ft_layers rejected (ValueError)",
          call_train(NEW_RUN, resnet_mode="fine_tuning", resnet_ft_layers="layer2_layer3_layer4") is ValueError)
    try:
        T.train("smoke_x", DEFAULT_MANIFEST, DEFAULT_CONFIG, model_name="baseline", resnet_ft_layers="layer3_layer4")
        other_model = None
    except (ValueError, LoadersCalled) as error:
        other_model = type(error)
    check("train(): resnet_ft_layers with another model rejected (ValueError)", other_model is ValueError)
    for ft in ("layer4", "layer3_layer4"):
        check(f"overwrite guard: existing run '{FINAL_RUN}' refused (resnet_ft_layers={ft}), before any data or file",
              call_train(FINAL_RUN, resnet_mode="fine_tuning", resnet_ft_layers=ft) is FileExistsError)
    check(f"overwrite guard: existing run '{FINAL_RUN}' refused in feature_extraction mode too",
          call_train(FINAL_RUN, resnet_mode="feature_extraction") is FileExistsError)
    saved_dirs = (T.CHECKPOINT_DIR, T.REPORTS_DIR)
    with tempfile.TemporaryDirectory() as tmp:
        T.CHECKPOINT_DIR, T.REPORTS_DIR = Path(tmp) / "checkpoints", Path(tmp) / "reports"
        T.CHECKPOINT_DIR.mkdir()
        T.REPORTS_DIR.mkdir()
        try:
            (T.CHECKPOINT_DIR / "guard_ckpt_best.pt").write_bytes(b"x")
            (T.REPORTS_DIR / "guard_hist_history.csv").write_text("x", encoding="utf-8")
            r_ck = call_train("guard_ckpt", resnet_mode="fine_tuning", resnet_ft_layers="layer3_layer4")
            r_hi = call_train("guard_hist", resnet_mode="fine_tuning", resnet_ft_layers="layer3_layer4")
            r_new = call_train(NEW_RUN, resnet_mode="fine_tuning", resnet_ft_layers="layer3_layer4")
            leftover = sorted(p.name for p in Path(tmp).rglob("*") if p.is_file())
        finally:
            T.CHECKPOINT_DIR, T.REPORTS_DIR = saved_dirs
    check("overwrite guard (temporary folder): existing checkpoint only -> refused", r_ck is FileExistsError)
    check("overwrite guard (temporary folder): existing history only -> refused", r_hi is FileExistsError)
    check("overwrite guard (temporary folder): new run name passes the guard (stops at the replaced data loader)",
          r_new is LoadersCalled)
    check("overwrite guard (temporary folder): nothing written besides the two dummy files",
          leftover == ["guard_ckpt_best.pt", "guard_hist_history.csv"], f"{leftover}")
finally:
    T.build_loaders = original_loaders

cli_cases = [
    ("--resnet-ft-layers with feature_extraction",
     ["--model", "resnet18", "--resnet-mode", "feature_extraction", "--resnet-ft-layers", "layer3_layer4"]),
    ("--resnet-ft-layers with convnext_tiny",
     ["--model", "convnext_tiny", "--resnet-ft-layers", "layer3_layer4"]),
]
for label, extra in cli_cases:
    run_name = "convnext_t_smoke_reject" if "convnext_tiny" in extra else "smoke_cli_reject"
    proc = subprocess.run([sys.executable, "-m", "src.train", "--run-name", run_name, *extra], cwd=REPO,
                          capture_output=True, text=True, timeout=600)
    check(f"CLI rejects {label} (exit code 2, before training)",
          proc.returncode == 2 and "--resnet-ft-layers is only used with" in proc.stderr,
          f"exit {proc.returncode}: {proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else ''}")

# ---------- 7. regression: scripts/verify_resnet.py ----------
proc = subprocess.run([sys.executable, str(REPO / "scripts" / "verify_resnet.py")], cwd=REPO,
                      capture_output=True, text=True, timeout=1800)
n_pass = proc.stdout.count("[PASS]")
n_fail = proc.stdout.count("[FAIL]")
check("scripts/verify_resnet.py: 9/9 PASS, exit code 0", proc.returncode == 0 and n_pass == 9 and n_fail == 0,
      f"exit {proc.returncode}, {n_pass} PASS, {n_fail} FAIL")

# ---------- 8. protected files after ----------
final_after = sha(FINAL_CKPT)
check("final checkpoint SHA256 after = before", final_after == final_before, final_after[:16])
files_after = snapshot()
diff = sorted(set(files_before.items()) ^ set(files_after.items()))
check("no file created or changed in checkpoints/ or reports/", not diff, f"{diff[:3]}")
check(f"artifacts of '{NEW_RUN}' not created or changed by this test (an earlier real run is allowed)",
      run_artifacts(NEW_RUN) == new_run_before, f"{new_run_before}")

print(f"\nfinal checkpoint SHA256 before {final_before}\nfinal checkpoint SHA256 after  {final_after}")
print(f"\n{sum(results)}/{len(results)} checks passed (no training, no data, Test and Neysan not read)")
sys.exit(0 if all(results) else 1)
