"""Build dataset v3: our data + half of the mentor test data, in one simple folder.

    python scripts/archive/build_dataset_v3.py

Inputs (read only, nothing is changed):
    data/split_manifest.csv + data/raw/...   our Train (2633) and Validation (659)
    data/clean/test/                          our old Test (400), now used for training
    data/mentor_test/TestingData/test0..4     the mentor test folders
    data/neysan/                              our Neysan images (only for the duplicate check)

What it does:
    1. Splits the 961 test0 images 50/50 per class: one half becomes the new final test,
       the other half goes 80/20 to Train / Validation.
       Near-duplicate images (pHash distance <= 5) always stay on the same side, and a test0
       image that looks like one of our own images is never put in the test half.
    2. The 157 extra Neysan images of the mentor tests are split 50/50 the same way.
    3. Writes data/dataset_v3/ :
           train/<class>/        val/<class>/
           test/test0..test4/<folder>/   the mentor scenarios, only with test-half images
           neysan_extra/         Neysan images of the training half (for the Neysan detector)
           split_v3.csv          one row per image
           review/               mentor training images the current model disagrees with
    4. Writes data/split_manifest_v3.csv (Train + Validation, same format as split_manifest.csv).
"""
import csv
import hashlib
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from PIL import Image

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"
MENTOR = DATA / "mentor_test" / "TestingData"
OUT = DATA / "dataset_v3"
CACHE = DATA / "phash_cache.csv"
SEED = 42
TEST_SHARE = 0.5         # share of test0 used as the final test
VAL_SHARE = 0.2          # share of the rest used as Validation
MAX_DISTANCE = 5         # pHash Hamming distance for "near duplicate"
CLASSES = ["ambulance", "autobus", "kamyun", "kamyunet", "minibus", "savari", "taxi", "vanet"]
IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".bmp"}


SHA_CACHE = DATA / "sha_cache.csv"
_sha = {}
if SHA_CACHE.exists():
    with open(SHA_CACHE, encoding="utf-8") as f:
        _sha = {(r["path"], int(r["size"])): r["sha256"] for r in csv.DictReader(f)}


def sha256(path):
    """SHA256 of a file; cached by path and size so re-runs are fast."""
    key = (str(path), path.stat().st_size)
    if key not in _sha:
        _sha[key] = hashlib.sha256(path.read_bytes()).hexdigest()
    return _sha[key]


def save_sha_cache():
    with open(SHA_CACHE, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "size", "sha256"])
        w.writerows((p, n, h) for (p, n), h in _sha.items())


# --- perceptual hash (pHash): 32x32 grayscale -> DCT -> 8x8 low frequencies -> 64 bits
N = 32
DCT = np.array([[np.cos(np.pi * (2 * x + 1) * u / (2 * N)) for x in range(N)] for u in range(N)])


def phash(path):
    img = np.asarray(Image.open(path).convert("L").resize((N, N), Image.LANCZOS), dtype=float)
    low = (DCT @ img @ DCT.T)[:8, :8]
    bits = (low > np.median(low)).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def load_cache():
    if not CACHE.exists():
        return {}
    with open(CACHE, encoding="utf-8") as f:
        return {r["sha256"]: int(r["phash"]) for r in csv.DictReader(f)}


def save_cache(cache):
    with open(CACHE, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sha256", "phash"])
        w.writerows(cache.items())


def copy(src, dst):
    """Copy one file; skip it if an identical-size copy is already there (safe to re-run)."""
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def files(folder):
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_TYPES)


# --- 1. collect every image ------------------------------------------------------------
def collect():
    """Returns our images, the test0 images, the extra Neysan images and the mentor folders."""
    ours = []
    roots = {"v1": DATA / "raw" / "dataset", "v2": DATA / "raw" / "datasetv2_TrainUclean"}
    with open(DATA / "split_manifest.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            source, rest = r["image_path"].split("/", 1)
            ours.append({"path": roots[source] / rest, "sha256": r["sha256"], "label": r["label"],
                         "origin": "ours_" + r["split"], "name": r["image_path"].replace("/", "_")})
    for p in files(DATA / "clean" / "test"):
        label = p.relative_to(DATA / "clean" / "test").parts[0]
        ours.append({"path": p, "sha256": sha256(p), "label": label, "origin": "ours_test",
                     "name": "old_test_" + p.name})

    mentor = {}                                    # test name -> {sha256: (folder, path)}
    for t in sorted(d.name for d in MENTOR.iterdir() if d.is_dir()):
        mentor[t] = {}
        for p in files(MENTOR / t):
            mentor[t][sha256(p)] = (p.parent.name, p)
    test0 = [{"path": p, "sha256": h, "label": folder, "origin": "mentor", "name": "mentor_" + p.name}
             for h, (folder, p) in mentor["test0"].items()]
    extra = {}
    for t, items in mentor.items():
        for h, (folder, p) in items.items():
            if h not in mentor["test0"]:
                extra[h] = p
    neysan = [{"path": p, "sha256": h, "label": "neysan", "origin": "mentor_neysan",
               "name": "mentor_" + p.name} for h, p in extra.items()]
    our_neysan = [{"path": p, "sha256": sha256(p)} for p in files(DATA / "neysan")]
    return ours, test0, neysan, our_neysan, mentor


def add_phashes(rows):
    cache = load_cache()
    for i, r in enumerate(rows):
        if r["sha256"] not in cache:
            cache[r["sha256"]] = phash(r["path"])
            if len(cache) % 500 == 0:
                save_cache(cache)
                print("  pHash", i + 1, "/", len(rows), flush=True)
        r["phash"] = cache[r["sha256"]]
    save_cache(cache)


def near(a, b):
    return bin(a ^ b).count("1") <= MAX_DISTANCE


# --- 2. split ----------------------------------------------------------------------------
def groups_of(rows):
    """Union near-duplicate images into groups (they must stay on the same side)."""
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            if near(rows[i]["phash"], rows[j]["phash"]):
                parent[find(i)] = find(j)
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[find(i)].append(r)
    return list(groups.values())


def split_groups(groups, share, rng):
    """Per class, put groups on side A until it holds `share` of that class. Returns (A, B)."""
    by_class = defaultdict(list)
    for g in groups:
        by_class[g[0]["label"]].append(g)
    a, b = [], []
    for label in sorted(by_class):
        gs = by_class[label]
        rng.shuffle(gs)
        target = round(share * sum(len(g) for g in gs))
        taken = 0
        for g in gs:
            if taken < target and not any(r.get("forced_train") for r in g):
                a += g
                taken += len(g)
            else:
                b += g
    return a, b


def main():
    rng = random.Random(SEED)
    print("reading images ...", flush=True)
    ours, test0, neysan, our_neysan, mentor = collect()
    save_sha_cache()
    print("pHash ...", flush=True)
    add_phashes(ours + test0 + neysan + our_neysan)

    # a mentor image that looks like one of our images must not be in the test half
    known = [r["phash"] for r in ours]
    known_neysan = [r["phash"] for r in our_neysan]
    for r in test0:
        r["forced_train"] = any(near(r["phash"], k) for k in known)
    for r in neysan:
        r["forced_train"] = any(near(r["phash"], k) for k in known_neysan + known)

    test_half, rest = split_groups(groups_of(test0), TEST_SHARE, rng)
    val_part, train_part = split_groups(groups_of(rest), VAL_SHARE, rng)
    neysan_test, neysan_train = split_groups(groups_of(neysan), TEST_SHARE, rng)

    # --- 3. write the dataset ---------------------------------------------------------
    rows = []
    for r in ours:
        rows.append({**r, "split": "val" if r["origin"] == "ours_val" else "train"})
    rows += [{**r, "split": "train"} for r in train_part]
    rows += [{**r, "split": "val"} for r in val_part]
    for r in rows:
        r["new_path"] = f"{r['split']}/{r['label']}/{r['name']}"
    rows += [{**r, "split": "test", "new_path": f"test/test0/{r['label']}/{r['path'].name}"} for r in test_half]
    rows += [{**r, "split": "test", "new_path": f"test/(scenarios)/{r['path'].name}"} for r in neysan_test]
    rows += [{**r, "split": "neysan_extra", "new_path": f"neysan_extra/{r['name']}"} for r in neysan_train]

    names = Counter(r["new_path"] for r in rows)
    clash = [n for n, c in names.items() if c > 1]
    if clash:
        sys.exit(f"file name clash: {clash[:5]}")

    print("copying ...", flush=True)
    for r in rows:
        if "(scenarios)" in r["new_path"]:
            continue
        copy(r["path"], OUT / r["new_path"])

    # the mentor scenarios (test2, test3, test4) with only the test-half images
    test_shas = {r["sha256"] for r in test_half} | {r["sha256"] for r in neysan_test}
    scenario_counts = {}
    for t, items in mentor.items():
        n = 0
        for h, (folder, p) in items.items():
            if h in test_shas:
                copy(p, OUT / "test" / t / folder / p.name)
                n += 1
        scenario_counts[t] = n

    with open(OUT / "split_v3.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["path", "sha256", "label", "split", "origin", "phash", "near_our_data"])
        for r in rows:
            if "(scenarios)" not in r["new_path"]:
                w.writerow([r["new_path"], r["sha256"], r["label"], r["split"], r["origin"],
                            f"{r['phash']:016x}", r.get("forced_train", "")])

    with open(DATA / "split_manifest_v3.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "sha256", "label", "split", "source", "origin_split", "folder_label", "dup_group"])
        for r in rows:
            if r["split"] in ("train", "val"):
                w.writerow(["v3/" + r["new_path"], r["sha256"], r["label"], r["split"], "v3",
                            r["origin"], r["label"], ""])

    # --- 4. review list: mentor training images the current model gets wrong ----------
    review = []
    pred_file = DATA / "mentor_test_results" / "predictions.csv"
    if pred_file.exists():
        with open(pred_file, encoding="utf-8-sig") as f:
            preds = {r["image_path"].replace("\\", "/"): r for r in csv.DictReader(f)}
        new_path = {r["sha256"]: r["new_path"] for r in rows}
        for r in train_part + val_part:
            p = preds.get("TestingData/test0/" + r["label"] + "/" + r["path"].name)
            if p and p["predicted_class"] != r["label"]:
                review.append((r, p))
        (OUT / "review").mkdir(exist_ok=True)
        with open(OUT / "review" / "review.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["path", "label", "model_says", "confidence", "keep (yes / no / new label)"])
            for r, p in sorted(review, key=lambda x: -float(x[1]["confidence"])):
                w.writerow([new_path[r["sha256"]], r["label"], p["predicted_class"],
                            p["confidence"], ""])

    # --- summary --------------------------------------------------------------------
    print()
    print(f"{'split':14s}" + "".join(f"{c:>10s}" for c in CLASSES) + f"{'total':>8s}")
    for s in ["train", "val", "test"]:
        part = [r for r in rows if r["split"] == s and r["label"] != "neysan"]
        cnt = Counter(r["label"] for r in part)
        print(f"{s:14s}" + "".join(f"{cnt[c]:10d}" for c in CLASSES) + f"{len(part):8d}")
    print()
    print("mentor test0 :", len(test0), "| test half", len(test_half), "| train", len(train_part),
          "| val", len(val_part), "| looks like our data (kept out of test)", sum(r["forced_train"] for r in test0))
    print("mentor Neysan:", len(neysan), "| test half", len(neysan_test), "| neysan_extra", len(neysan_train))
    print("test scenarios:", scenario_counts)
    print("review list  :", len(review), "images ->", OUT / "review" / "review.csv")


if __name__ == "__main__":
    main()
