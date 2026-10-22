#!/usr/bin/env python3
"""Convert and independently verify all seven Unitree task groups."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from check_unitree_zarr import check
from convert_unitree_to_zarr import convert


WORKSPACE = Path(__file__).resolve().parents[2]
DATASETS = (
    ("dex3_pick/left", "left_pick"),
    ("dex3_pick/right", "right_pick"),
    ("dex3_place/left", "left_place"),
    ("dex3_place/right", "right_place"),
    ("dex3_stack", "stack"),
    ("dex3_handover/left_to_right_handover_high", "left_to_right_handover_high"),
    ("dex3_handover/right_to_left_handover_low", "right_to_left_handover_low"),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=WORKSPACE / "tsingtec_raw_data")
    parser.add_argument("--output-root", type=Path, default=WORKSPACE / "tsingtec_training_data")
    args = parser.parse_args()
    raw_root, output_root = args.raw_root.resolve(), args.output_root.resolve()
    # Preflight the complete plan before starting; never replace existing data.
    for source, name in DATASETS:
        if not list((raw_root / source).glob("episode_*/data.json")):
            raise ValueError(f"No source episodes: {raw_root / source}")
        if (output_root / name).exists():
            raise FileExistsError(output_root / name)
    output_root.mkdir(parents=True, exist_ok=True)
    summary = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
               "image_source": "colors.color_1", "datasets": [], "complete": False}
    for source, name in DATASETS:
        output = output_root / name
        print(f"\n=== Converting {source} -> {output} ===", flush=True)
        report = convert(raw_root / source, output)
        print(f"=== Verifying all joints and all RGB images: {name} ===", flush=True)
        result = check(raw_root / source, output, all_images=True)
        (output / "verification_report.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        summary["datasets"].append({"name": name, "source": str(raw_root / source),
                                    "output": str(output), "n_episodes": report["n_episodes"],
                                    "n_frames": report["n_frames"], "verification_passed": result["passed"]})
        (output_root / "conversion_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary.update(complete=True,
                   n_episodes=sum(d["n_episodes"] for d in summary["datasets"]),
                   n_frames=sum(d["n_frames"] for d in summary["datasets"]))
    (output_root / "conversion_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
