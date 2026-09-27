"""First data audit: counts and exact duplicates, read only from data/manifest.csv.

Nothing is changed: no raw file is touched, no label is changed,
no image is removed. folder_label is used exactly as it is in the manifest.

Usage:
    python scripts/audit_dataset.py
"""

import csv
from collections import Counter, defaultdict
from pathlib import Path

MANIFEST_PATH = Path("data/manifest.csv")
REPORT_DIR = Path("reports/audit")
TEXT_REPORT = REPORT_DIR / "dataset_audit.txt"
DUPLICATES_CSV = REPORT_DIR / "exact_duplicates.csv"


def read_manifest(path):
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def count_table(rows, keys):
    """Count rows by the given columns, e.g. keys = ["source", "origin_split"]."""
    counts = Counter()
    for row in rows:
        group = " / ".join(row[k] for k in keys)
        counts[group] += 1
    lines = []
    for group in sorted(counts):
        lines.append(f"  {group:<40} {counts[group]:>6}")
    lines.append(f"  {'TOTAL':<40} {sum(counts.values()):>6}")
    return lines


def find_duplicate_groups(rows):
    """Group rows by sha256 and keep only hashes that appear more than once."""
    by_hash = defaultdict(list)
    for row in rows:
        by_hash[row["sha256"]].append(row)
    groups = []
    for sha, members in sorted(by_hash.items()):
        if len(members) > 1:
            groups.append((sha, members))
    return groups


def describe_relation(members):
    """Say where the copies of one duplicate group live."""
    sources = sorted({m["source"] for m in members})
    places = sorted({f"{m['source']}/{m['origin_split']}" for m in members})
    if len(sources) > 1:
        return "cross-source: " + " <-> ".join(places)
    if len(places) == 1:
        return "within " + places[0]
    splits = sorted({m["origin_split"] for m in members})
    return " <-> ".join(splits) + f" ({sources[0]})"


def main():
    rows = read_manifest(MANIFEST_PATH)
    groups = find_duplicate_groups(rows)

    report = []
    report.append("Dataset audit (read only, from data/manifest.csv)")
    report.append(f"images in manifest: {len(rows)}")
    report.append("")
    report.append("Images per source")
    report += count_table(rows, ["source"])
    report.append("")
    report.append("Images per source / origin_split")
    report += count_table(rows, ["source", "origin_split"])
    report.append("")
    report.append("Images per source / origin_split / folder_label")
    report += count_table(rows, ["source", "origin_split", "folder_label"])
    report.append("")

    # Exact duplicates: same sha256 = same file bytes.
    relation_counts = Counter()
    conflict_groups = []
    csv_rows = []
    report.append(f"Exact duplicate groups (same sha256): {len(groups)}")
    report.append(f"files involved: {sum(len(m) for _, m in groups)}")
    report.append("")
    for number, (sha, members) in enumerate(groups, start=1):
        group_id = f"DUP{number:03d}"
        relation = describe_relation(members)
        labels = sorted({m["folder_label"] for m in members})
        is_conflict = len(labels) > 1
        relation_counts[relation] += 1
        if is_conflict:
            conflict_groups.append((group_id, sha, members))

        report.append(f"{group_id}  {relation}  labels={','.join(labels)}"
                      + ("  LABEL CONFLICT" if is_conflict else ""))
        for m in members:
            report.append(f"    {m['image_path']}")
            csv_rows.append({
                "group_id": group_id,
                "sha256": sha,
                "n_files": len(members),
                "relation": relation,
                "label_conflict": is_conflict,
                "image_path": m["image_path"],
                "source": m["source"],
                "origin_split": m["origin_split"],
                "folder_label": m["folder_label"],
            })
    report.append("")
    report.append("Duplicate groups per relation")
    for relation in sorted(relation_counts):
        report.append(f"  {relation:<40} {relation_counts[relation]:>6}")
    report.append("")
    report.append(f"Exact duplicate label conflicts (same sha256, different folder_label): {len(conflict_groups)}")
    for group_id, sha, members in conflict_groups:
        report.append(f"  {group_id}  sha256={sha}")
        for m in members:
            report.append(f"    {m['image_path']}  (folder_label={m['folder_label']})")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with open(TEXT_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")
    with open(DUPLICATES_CSV, "w", newline="", encoding="utf-8") as f:
        columns = ["group_id", "sha256", "n_files", "relation", "label_conflict",
                   "image_path", "source", "origin_split", "folder_label"]
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(csv_rows)

    print("\n".join(report))
    print(f"\nwrote {TEXT_REPORT} and {DUPLICATES_CSV}")


if __name__ == "__main__":
    main()
