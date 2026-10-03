"""Prepare the same four CogAge activities separately for every available subject."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    from . import prepare_dataset as pipeline
except ImportError:
    import prepare_dataset as pipeline


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = SCRIPT_DIR.parent / "Sample"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "processed_datasets_by_subject"
DEFAULT_ACTIVITIES = ("Bending", "Walking", "Sitting", "Standing")
SUBJECT_FOLDER_PATTERN = re.compile(r"^(S\d+)_(.+)_([12])_extracted$")


def discover_subject_recordings(data_root: Path) -> dict[str, dict[str, dict[str, Path]]]:
    """Discover canonical subject/activity folders and preserve folder-based splits."""
    subjects: dict[str, dict[str, dict[str, Path]]] = {}
    for folder in sorted(data_root.iterdir(), key=lambda item: item.name.casefold()):
        if not folder.is_dir():
            continue
        match = SUBJECT_FOLDER_PATTERN.fullmatch(folder.name)
        if match is None:
            continue
        subject, activity, recording = match.groups()
        split = "train" if recording == "1" else "test"
        subjects.setdefault(subject, {}).setdefault(activity, {})[split] = folder
    return subjects


def build_inventory(
    subjects: dict[str, dict[str, dict[str, Path]]],
    selected_activities: list[str],
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    rows: list[dict[str, Any]] = []
    paired_activities: dict[str, list[str]] = {}
    for subject in sorted(subjects):
        paired_activities[subject] = []
        for activity in selected_activities:
            recordings = subjects[subject].get(activity, {})
            train_folder = recordings.get("train")
            test_folder = recordings.get("test")
            if train_folder and test_folder:
                status = "paired"
                paired_activities[subject].append(activity)
            elif train_folder:
                status = "missing_test"
            elif test_folder:
                status = "missing_train"
            else:
                status = "missing_both"
            rows.append(
                {
                    "subject": subject,
                    "activity": activity,
                    "train_folder": train_folder.name if train_folder else "",
                    "test_folder": test_folder.name if test_folder else "",
                    "status": status,
                }
            )
    return pd.DataFrame(rows), paired_activities


def _save_subject_dataset(
    output_dir: Path,
    subject: str,
    paired_activities: list[str],
    class_mapping: dict[str, int],
    event_records: list[dict[str, Any]],
    x_train_raw: np.ndarray,
    y_train: np.ndarray,
    train_metadata: list[dict[str, Any]],
    x_test_raw: np.ndarray,
    y_test: np.ndarray,
    test_metadata: list[dict[str, Any]],
    feature_names: list[str],
    audit_frame: pd.DataFrame,
    cleaning_stats: Counter[str],
    scaler: StandardScaler,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    x_train = scaler.transform(x_train_raw.reshape(-1, x_train_raw.shape[-1])).reshape(x_train_raw.shape)
    x_test = (
        scaler.transform(x_test_raw.reshape(-1, x_test_raw.shape[-1])).reshape(x_test_raw.shape)
        if len(x_test_raw)
        else x_test_raw.copy()
    )
    x_train = x_train.astype(np.float32, copy=False)
    x_test = x_test.astype(np.float32, copy=False)

    pipeline.save_dataset(
        output_dir,
        x_train,
        y_train,
        x_test,
        y_test,
        train_metadata,
        test_metadata,
        feature_names,
        class_mapping,
        scaler,
        x_train_raw,
        Counter(y_test.tolist()),
    )
    audit_frame.to_csv(output_dir / "raw_file_audit.csv", index=False)

    train_counts = Counter(y_train.tolist())
    test_counts = Counter(y_test.tolist())
    rows = [
        f"CogAge Atomic Behaviour dataset preparation: {subject}",
        "",
        f"Subject: {subject}",
        f"Paired activities processed: {', '.join(paired_activities)}",
        f"Shared class mapping: {json.dumps(class_mapping, sort_keys=True)}",
        f"Resample frequency: {pipeline.RESAMPLE_MS}",
        f"Window size: {pipeline.WINDOW_SIZE}",
        f"Feature count: {len(feature_names)}",
        f"X_train shape: {x_train.shape}",
        f"X_test shape: {x_test.shape}",
        "Split policy: this subject's _1_extracted folders feed train; _2_extracted folders feed test.",
        "Scaler: fitted only on this subject's training windows; then applied to that subject's train and test windows.",
        f"Raw sensor files audited: {len(audit_frame)}",
        f"Malformed rows: {int(audit_frame['malformed_rows'].sum()) if not audit_frame.empty else 0}",
        f"Missing raw measurement values: {int(audit_frame['missing_values'].sum()) if not audit_frame.empty else 0}",
        f"Resampled NaN cells interpolated or dropped: {cleaning_stats['resampled_nan_cells_before_interpolation']}",
        f"Rows removed after interpolation/synchronization: {cleaning_stats['rows_removed_after_interpolation_or_sync']}",
        f"Incomplete train windows discarded: {cleaning_stats['discarded_incomplete_windows_train']} ({cleaning_stats['discarded_tail_rows_train']} tail rows)",
        f"Incomplete test windows discarded: {cleaning_stats['discarded_incomplete_windows_test']} ({cleaning_stats['discarded_tail_rows_test']} tail rows)",
        "",
        "Window counts per shared class ID:",
    ]
    for activity, label in sorted(class_mapping.items(), key=lambda item: item[1]):
        rows.append(
            f"  Class {label} ({activity}): train={train_counts[label]}, test={test_counts[label]}"
        )

    rows.extend(["", "Original folders used:"])
    used_folders = sorted({record["original_folder"] for record in event_records})
    rows.extend(f"  {folder}" for folder in used_folders)
    rows.extend(
        [
            "",
            "The shared four-class mapping is retained even if a class is unavailable for this subject; unavailable classes have zero windows.",
            "The original _2_extracted split is kept as test and is never mixed into train or validation.",
        ]
    )
    (output_dir / "dataset_summary.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")


def process_subject(
    subject: str,
    subject_recordings: dict[str, dict[str, Path]],
    paired_activities: list[str],
    class_mapping: dict[str, int],
    output_dir: Path,
) -> dict[str, Any]:
    audit_records: list[dict[str, Any]] = []
    event_records: list[dict[str, Any]] = []
    cleaning_stats: Counter[str] = Counter()

    for activity in paired_activities:
        label = class_mapping[activity]
        for split in ("train", "test"):
            folder = subject_recordings[activity][split]
            print(f"  Processing {folder.name} -> {split.upper()}")
            records = pipeline.process_activity(
                activity,
                folder,
                label,
                split,
                audit_records,
                cleaning_stats,
            )
            for record in records:
                record["subject"] = subject
            event_records.extend(records)

    training_events = [record for record in event_records if record["split"] == "train"]
    feature_names = pipeline._feature_intersection(training_events)
    if not feature_names:
        return {
            "subject": subject,
            "status": "skipped_no_training_features",
            "paired_activities": ",".join(paired_activities),
            "train_windows": 0,
            "test_windows": 0,
            "output_dir": "",
        }

    x_train_raw, y_train, train_metadata = pipeline.build_train_dataset(
        event_records, feature_names, cleaning_stats
    )
    x_test_raw, y_test, test_metadata = pipeline.build_test_dataset(
        event_records, feature_names, cleaning_stats
    )
    if not len(x_train_raw):
        return {
            "subject": subject,
            "status": "skipped_no_complete_training_windows",
            "paired_activities": ",".join(paired_activities),
            "train_windows": 0,
            "test_windows": len(x_test_raw),
            "output_dir": "",
        }

    scaler = StandardScaler()
    scaler.fit(x_train_raw.reshape(-1, x_train_raw.shape[-1]))
    _save_subject_dataset(
        output_dir,
        subject,
        paired_activities,
        class_mapping,
        event_records,
        x_train_raw,
        y_train,
        train_metadata,
        x_test_raw,
        y_test,
        test_metadata,
        feature_names,
        pd.DataFrame(audit_records),
        cleaning_stats,
        scaler,
    )
    return {
        "subject": subject,
        "status": "prepared",
        "paired_activities": ",".join(paired_activities),
        "train_windows": len(x_train_raw),
        "test_windows": len(x_test_raw),
        "output_dir": str(output_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--activities",
        default=",".join(DEFAULT_ACTIVITIES),
        help="Exactly four comma-separated canonical activity names.",
    )
    parser.add_argument(
        "--subjects",
        default="",
        help="Optional comma-separated subject IDs, e.g. S2,S3. Default: discover every subject.",
    )
    args = parser.parse_args()
    data_root = args.data_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    selected_activities = [value.strip() for value in args.activities.split(",") if value.strip()]
    requested_subjects = [value.strip().upper() for value in args.subjects.split(",") if value.strip()]

    if len(selected_activities) != 4 or len(set(selected_activities)) != 4:
        raise ValueError("Exactly four distinct activities are required to preserve the assignment labels.")
    if not data_root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {data_root}")

    discovered = discover_subject_recordings(data_root)
    if requested_subjects:
        unknown = sorted(set(requested_subjects) - set(discovered))
        if unknown:
            raise ValueError(f"Requested subjects not found: {unknown}")
        discovered = {subject: discovered[subject] for subject in requested_subjects}

    inventory, paired_activities = build_inventory(discovered, selected_activities)
    class_mapping = {activity: index for index, activity in enumerate(selected_activities)}
    print("Original split rule: _1_extracted -> TRAIN; _2_extracted -> TEST")
    print(f"Shared class mapping: {class_mapping}")
    print("\nFolder-pair inventory:")
    print(inventory.to_string(index=False))

    run_root = output_root / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_root.mkdir(parents=True, exist_ok=False)
    inventory.to_csv(run_root / "subject_activity_inventory.csv", index=False)

    results: list[dict[str, Any]] = []
    for subject in sorted(discovered):
        available_pairs = paired_activities[subject]
        if not available_pairs:
            result = {
                "subject": subject,
                "status": "skipped_no_activity_with_both_splits",
                "paired_activities": "",
                "train_windows": 0,
                "test_windows": 0,
                "output_dir": "",
            }
            print(
                f"\n{subject}: skipped, because none of the selected activities has both "
                "the original training and testing folders."
            )
            results.append(result)
            continue

        print(f"\nPreparing {subject}; complete activity pairs: {available_pairs}")
        result = process_subject(
            subject,
            discovered[subject],
            available_pairs,
            class_mapping,
            run_root / subject,
        )
        results.append(result)
        print(
            f"{subject}: {result['status']}; train windows={result['train_windows']}; "
            f"test windows={result['test_windows']}"
        )

    results_frame = pd.DataFrame(results)
    results_frame.to_csv(run_root / "subject_preparation_summary.csv", index=False)
    (run_root / "class_mapping.json").write_text(
        json.dumps(class_mapping, indent=2), encoding="utf-8"
    )
    print(f"\nPer-subject datasets and inventory saved under: {run_root}")
    print("Subject-level summary:")
    print(results_frame.to_string(index=False))


if __name__ == "__main__":
    main()
