"""Prepare four S1 CogAge activities without crossing the original train/test split."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = SCRIPT_DIR.parent / "Sample"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "processed_dataset"
DEFAULT_ACTIVITIES = ("Bending", "Walking", "Sitting", "Standing")
RESAMPLE_MS = "20ms"
WINDOW_SIZE = 128
FOLDER_PATTERN = re.compile(r"^S1_(.+)_([12])_extracted$")
EVENT_FILE_PATTERN = re.compile(r"^(.+)_Event_(\d+)\.data$")


def parse_activity_folder(folder_name: str) -> tuple[str, str] | None:
    """Return (activity, split) for canonical S1 folders; ignore _extracted_r variants."""
    match = FOLDER_PATTERN.fullmatch(folder_name)
    if not match:
        return None
    activity, recording = match.groups()
    return activity, "train" if recording == "1" else "test"


def discover_activities(data_root: Path) -> dict[str, dict[str, Path]]:
    """Discover canonical activity folders and their original recording splits."""
    activities: dict[str, dict[str, Path]] = defaultdict(dict)
    for folder in sorted(data_root.iterdir(), key=lambda item: item.name.casefold()):
        if not folder.is_dir():
            continue
        parsed = parse_activity_folder(folder.name)
        if parsed is None:
            continue
        activity, split = parsed
        activities[activity][split] = folder
    return dict(activities)


def print_activity_inventory(activities: dict[str, dict[str, Path]]) -> None:
    print("Available canonical S1 activities (_extracted_r variants excluded):")
    for activity in sorted(activities, key=str.casefold):
        folders = activities[activity]
        train_name = folders["train"].name if "train" in folders else "MISSING"
        test_name = folders["test"].name if "test" in folders else "MISSING"
        print(f"  {activity}: {train_name} -> TRAIN; {test_name} -> TEST")
    print(
        "Convention check: every canonical folder ending _1_extracted is TRAIN; "
        "every canonical folder ending _2_extracted is TEST."
    )


def _inspect_raw_file(path: Path) -> dict[str, Any]:
    """Read one raw file and report its dimensions, timestamp, and data quality."""
    nonblank_lines = 0
    field_counts: Counter[int] = Counter()
    try:
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            for line in stream:
                if line.strip():
                    nonblank_lines += 1
                    field_counts[len(line.rstrip("\r\n").split(";"))] += 1
        frame = pd.read_csv(path, sep=";", header=None, on_bad_lines="skip")
    except Exception as exc:
        return {
            "file": path.name,
            "rows": 0,
            "columns": 0,
            "malformed_rows": nonblank_lines,
            "missing_values": 0,
            "bad_timestamps": 0,
            "timestamp_format": "unreadable",
            "median_interval_ms": np.nan,
            "error": str(exc),
        }

    modal_width = field_counts.most_common(1)[0][0] if field_counts else frame.shape[1]
    malformed_rows = sum(count for width, count in field_counts.items() if width != modal_width)
    if frame.empty or frame.shape[1] < 2:
        return {
            "file": path.name,
            "rows": len(frame),
            "columns": frame.shape[1],
            "malformed_rows": malformed_rows + max(0, nonblank_lines - len(frame)),
            "missing_values": int(frame.isna().sum().sum()),
            "bad_timestamps": len(frame),
            "timestamp_format": "unreadable",
            "median_interval_ms": np.nan,
            "error": "No timestamp and measurement columns",
        }

    timestamps = pd.to_numeric(frame.iloc[:, 0], errors="coerce")
    bad_timestamps = int(timestamps.isna().sum())
    numeric_values = frame.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    missing_values = int(numeric_values.isna().sum().sum())
    valid_ts = timestamps.dropna()
    positive_deltas = valid_ts.diff().dropna()
    positive_deltas = positive_deltas[positive_deltas > 0]
    median_interval = float(positive_deltas.median()) if not positive_deltas.empty else np.nan
    first_timestamp = valid_ts.iloc[0] if not valid_ts.empty else np.nan
    try:
        timestamp_format = (
            f"epoch milliseconds ({pd.to_datetime(first_timestamp, unit='ms')})"
            if pd.notna(first_timestamp)
            else "invalid"
        )
    except (ValueError, OverflowError, TypeError):
        timestamp_format = "invalid epoch milliseconds"

    return {
        "file": path.name,
        "rows": len(frame),
        "columns": frame.shape[1],
        "malformed_rows": malformed_rows + max(0, nonblank_lines - len(frame)),
        "missing_values": missing_values,
        "bad_timestamps": bad_timestamps,
        "timestamp_format": timestamp_format,
        "median_interval_ms": median_interval,
        "error": "",
    }


def load_sensor_file(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load semicolon-separated values, convert epoch-ms timestamps, and name values."""
    audit = _inspect_raw_file(path)
    try:
        raw = pd.read_csv(path, sep=";", header=None, on_bad_lines="skip")
    except Exception:
        return pd.DataFrame(), audit
    if raw.empty or raw.shape[1] < 2:
        return pd.DataFrame(), audit

    timestamps = pd.to_numeric(raw.iloc[:, 0], errors="coerce")
    values = raw.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    valid_timestamp = timestamps.notna()
    timestamps = timestamps.loc[valid_timestamp]
    values = values.loc[valid_timestamp].copy()
    sensor_match = EVENT_FILE_PATTERN.fullmatch(path.name)
    if sensor_match is None:
        return pd.DataFrame(), audit
    sensor = sensor_match.group(1)
    values.columns = [f"{sensor}_value_{index}" for index in range(1, values.shape[1] + 1)]
    values.index = pd.to_datetime(timestamps.to_numpy(), unit="ms", errors="coerce")
    values = values.loc[~values.index.isna()].sort_index()
    values.index.name = "timestamp"
    return values, audit


def resample_sensor(frame: pd.DataFrame) -> pd.DataFrame:
    """Put a sensor on a shared 20 ms grid by averaging values within each bin."""
    if frame.empty:
        return frame
    # Resampling aligns different sensor sampling rates on the same 20 ms time grid.
    return frame.resample(RESAMPLE_MS).mean()


def interpolate_sensor(frame: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """Linearly interpolate internal gaps and drop any remaining incomplete rows."""
    if frame.empty:
        return frame, 0, 0
    missing_before = int(frame.isna().sum().sum())
    interpolated = frame.interpolate(method="linear", limit_area="inside")
    rows_before_drop = len(interpolated)
    clean = interpolated.dropna(how="any")
    return clean, missing_before, rows_before_drop - len(clean)


def _event_files(folder: Path) -> dict[int, dict[str, Path]]:
    events: dict[int, dict[str, Path]] = defaultdict(dict)
    for path in sorted(folder.glob("*.data")):
        match = EVENT_FILE_PATTERN.fullmatch(path.name)
        if match:
            sensor, event_number = match.groups()
            events[int(event_number)][sensor] = path
    return dict(events)


def load_event(
    folder: Path,
    event_number: int,
    sensor_files: dict[str, Path],
    audit_records: list[dict[str, Any]],
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    """Load and synchronize one Event; never joins data across Event boundaries."""
    sensor_frames: list[pd.DataFrame] = []
    missing_sensors: list[str] = []
    missing_values = 0
    dropped_rows = 0
    for sensor, path in sorted(sensor_files.items()):
        frame, audit = load_sensor_file(path)
        audit_records.append(
            {
                "original_folder": folder.name,
                "event": f"Event_{event_number}",
                "sensor": sensor,
                **audit,
            }
        )
        if frame.empty:
            missing_sensors.append(sensor)
            continue
        frame = resample_sensor(frame)
        frame, missing_count, dropped_count = interpolate_sensor(frame)
        missing_values += missing_count
        dropped_rows += dropped_count
        if frame.empty:
            missing_sensors.append(sensor)
            continue
        sensor_frames.append(frame)

    if not sensor_frames:
        return None, {
            "missing_sensors": missing_sensors,
            "missing_values": missing_values,
            "dropped_rows": dropped_rows,
            "reason": "no usable sensor streams",
        }

    # An inner join retains only timestamps shared by the sensor streams in this Event.
    union_timestamps = set().union(*(set(frame.index) for frame in sensor_frames))
    combined = pd.concat(sensor_frames, axis=1, join="inner").sort_index()
    dropped_rows += len(union_timestamps) - len(combined)
    before_drop = len(combined)
    combined = combined.dropna(how="any")
    dropped_rows += before_drop - len(combined)
    if combined.empty:
        return None, {
            "missing_sensors": missing_sensors,
            "missing_values": missing_values,
            "dropped_rows": dropped_rows,
            "reason": "sensor streams have no complete synchronized timestamps",
        }
    return combined, {
        "missing_sensors": missing_sensors,
        "missing_values": missing_values,
        "dropped_rows": dropped_rows,
        "reason": "",
    }


def create_windows(
    frame: pd.DataFrame,
    feature_names: list[str],
) -> tuple[list[np.ndarray], int]:
    """Create complete non-overlapping windows from a single event."""
    if not feature_names:
        return [], len(frame) // WINDOW_SIZE
    usable = frame.reindex(columns=feature_names).dropna(how="any")
    count = len(usable) // WINDOW_SIZE
    windows = [
        usable.iloc[index * WINDOW_SIZE : (index + 1) * WINDOW_SIZE].to_numpy(dtype=np.float64)
        for index in range(count)
    ]
    return windows, len(usable) % WINDOW_SIZE


def process_activity(
    activity: str,
    folder: Path,
    label: int,
    split: str,
    audit_records: list[dict[str, Any]],
    cleaning_stats: Counter[str],
) -> list[dict[str, Any]]:
    """Load each available event and retain its folder, split, and activity identity."""
    event_records: list[dict[str, Any]] = []
    events = _event_files(folder)
    if not events:
        print(f"  WARNING: no Event_*.data files in {folder.name}")
        return event_records

    all_sensors = sorted({sensor for event in events.values() for sensor in event})
    for event_number in sorted(events):
        available = events[event_number]
        missing = sorted(set(all_sensors) - set(available))
        if missing:
            print(f"  {folder.name}/Event_{event_number}: missing files for {missing}")
        frame, stats = load_event(folder, event_number, available, audit_records)
        cleaning_stats["resampled_nan_cells_before_interpolation"] += stats["missing_values"]
        cleaning_stats["rows_removed_after_interpolation_or_sync"] += stats["dropped_rows"]
        if frame is None:
            print(f"  {folder.name}/Event_{event_number}: skipped ({stats['reason']})")
            continue
        if stats["missing_sensors"]:
            print(
                f"  {folder.name}/Event_{event_number}: unusable/missing sensors "
                f"{stats['missing_sensors']}"
            )
        event_records.append(
            {
                "subject": "S1",
                "activity": activity,
                "label": label,
                "original_folder": folder.name,
                "split": split,
                "train_or_test": split,
                "activity_instance": 1 if split == "train" else 2,
                "event": f"Event_{event_number}",
                "event_number": event_number,
                "frame": frame,
            }
        )
    return event_records


def _feature_intersection(event_records: list[dict[str, Any]]) -> list[str]:
    if not event_records:
        return []
    common = set(event_records[0]["frame"].columns)
    for record in event_records[1:]:
        common.intersection_update(record["frame"].columns)
    return sorted(common, key=str.casefold)


def _window_records(
    event_records: list[dict[str, Any]],
    split: str,
    feature_names: list[str],
    cleaning_stats: Counter[str],
) -> tuple[list[np.ndarray], np.ndarray, list[dict[str, Any]]]:
    windows: list[np.ndarray] = []
    labels: list[int] = []
    metadata: list[dict[str, Any]] = []
    for record in event_records:
        if record["split"] != split:
            continue
        event_windows, _ = create_windows(record["frame"], feature_names)
        row_count = len(record["frame"].reindex(columns=feature_names).dropna(how="any"))
        tail_rows = row_count % WINDOW_SIZE
        cleaning_stats[f"discarded_incomplete_windows_{split}"] += int(tail_rows > 0)
        cleaning_stats[f"discarded_tail_rows_{split}"] += tail_rows
        for window_id, window in enumerate(event_windows):
            windows.append(window)
            labels.append(record["label"])
            metadata.append(
                {
                    "subject": record["subject"],
                    "activity": record["activity"],
                    "train_or_test": record["train_or_test"],
                    "activity_instance": record["activity_instance"],
                    "event": record["event"],
                    "window_id": window_id,
                    "label": record["label"],
                    "original_folder": record["original_folder"],
                    "split": record["split"],
                }
            )
    if windows:
        values = np.stack(windows).astype(np.float32, copy=False)
    else:
        values = np.empty((0, WINDOW_SIZE, len(feature_names)), dtype=np.float32)
    return values, np.asarray(labels, dtype=np.int64), metadata


def build_train_dataset(
    event_records: list[dict[str, Any]],
    feature_names: list[str],
    cleaning_stats: Counter[str],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    return _window_records(event_records, "train", feature_names, cleaning_stats)


def build_test_dataset(
    event_records: list[dict[str, Any]],
    feature_names: list[str],
    cleaning_stats: Counter[str],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    return _window_records(event_records, "test", feature_names, cleaning_stats)


def _audit_summary(audit_frame: pd.DataFrame) -> None:
    print("\nRaw sensor-file inspection (audit CSV contains one row per file):")
    if audit_frame.empty:
        print("  No .data files were inspected.")
        return
    summary = audit_frame.groupby("sensor", sort=True).agg(
        files=("file", "count"),
        rows_min=("rows", "min"),
        rows_max=("rows", "max"),
        columns_min=("columns", "min"),
        columns_max=("columns", "max"),
        malformed_rows=("malformed_rows", "sum"),
        missing_values=("missing_values", "sum"),
        bad_timestamps=("bad_timestamps", "sum"),
        median_interval_ms=("median_interval_ms", "median"),
    )
    print(summary.to_string())


def _create_plots(
    output_dir: Path,
    x_train_raw: np.ndarray,
    y_train: np.ndarray,
    class_mapping: dict[str, int],
    feature_names: list[str],
    test_counts: Counter[int],
) -> None:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    train_counts = Counter(y_train.tolist())
    class_ids = list(class_mapping.values())
    class_names = [name for name, _ in sorted(class_mapping.items(), key=lambda item: item[1])]
    positions = np.arange(len(class_ids))
    width = 0.38
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(positions - width / 2, [train_counts[i] for i in class_ids], width, label="Train")
    axis.bar(positions + width / 2, [test_counts[i] for i in class_ids], width, label="Test")
    axis.set_xticks(positions, class_names, rotation=20, ha="right")
    axis.set_ylabel("Windows")
    axis.set_title("Window counts by activity and original split")
    axis.legend()
    figure.tight_layout()
    figure.savefig(figures_dir / "class_counts.png", dpi=160)
    plt.close(figure)

    if not feature_names or not len(x_train_raw):
        return
    selected_feature = next(
        (index for index, name in enumerate(feature_names) if "PhoneAccelerometer" in name),
        0,
    )
    feature_label = feature_names[selected_feature]
    figure, axes = plt.subplots(len(class_mapping), 1, figsize=(11, 9), sharex=True)
    if len(class_mapping) == 1:
        axes = [axes]
    for axis, (activity, label) in zip(axes, sorted(class_mapping.items(), key=lambda item: item[1])):
        matches = np.flatnonzero(y_train == label)
        if len(matches):
            axis.plot(
                np.arange(WINDOW_SIZE) * 20,
                x_train_raw[matches[0], :, selected_feature],
                linewidth=0.9,
            )
            axis.set_ylabel(activity)
        else:
            axis.text(0.5, 0.5, "No training windows", ha="center", va="center", transform=axis.transAxes)
        axis.grid(alpha=0.25)
    axes[-1].set_xlabel("Time within window (ms)")
    figure.suptitle(f"Training examples: {feature_label}")
    figure.tight_layout()
    figure.savefig(figures_dir / "sensor_time_series.png", dpi=160)
    plt.close(figure)


def save_dataset(
    output_dir: Path,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    train_metadata: list[dict[str, Any]],
    test_metadata: list[dict[str, Any]],
    feature_names: list[str],
    class_mapping: dict[str, int],
    scaler: StandardScaler,
    x_train_raw: np.ndarray,
    test_counts: Counter[int],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "X_train.npy", x_train)
    np.save(output_dir / "y_train.npy", y_train)
    np.save(output_dir / "X_test.npy", x_test)
    np.save(output_dir / "y_test.npy", y_test)
    pd.DataFrame(train_metadata).to_csv(output_dir / "train_metadata.csv", index=False)
    pd.DataFrame(test_metadata).to_csv(output_dir / "test_metadata.csv", index=False)
    (output_dir / "feature_names.json").write_text(
        json.dumps(feature_names, indent=2), encoding="utf-8"
    )
    (output_dir / "class_mapping.json").write_text(
        json.dumps(class_mapping, indent=2), encoding="utf-8"
    )
    joblib.dump(scaler, output_dir / "scaler.joblib")
    _create_plots(output_dir, x_train_raw, y_train, class_mapping, feature_names, test_counts)


def create_summary(
    output_dir: Path,
    activities: dict[str, dict[str, Path]],
    selected_activities: list[str],
    class_mapping: dict[str, int],
    event_records: list[dict[str, Any]],
    train_metadata: list[dict[str, Any]],
    test_metadata: list[dict[str, Any]],
    x_train: np.ndarray,
    x_test: np.ndarray,
    feature_names: list[str],
    audit_frame: pd.DataFrame,
    cleaning_stats: Counter[str],
) -> None:
    lines = [
        "CogAge Atomic Behaviour S1 dataset preparation summary",
        "",
        f"Selected activities: {', '.join(selected_activities)}",
        f"Class mapping: {json.dumps(class_mapping, sort_keys=True)}",
        f"Resample frequency: {RESAMPLE_MS}",
        f"Window size: {WINDOW_SIZE} timesteps",
        "Split policy: only _1_extracted folders feed train; only _2_extracted folders feed test.",
        "Scaler: StandardScaler fitted on training-window timesteps only, then applied to train and test.",
        f"Feature count: {len(feature_names)}",
        f"X_train shape: {x_train.shape}",
        f"X_test shape: {x_test.shape}",
        f"Inspected raw sensor files: {len(audit_frame)}",
        f"Malformed rows reported by audit: {int(audit_frame['malformed_rows'].sum()) if not audit_frame.empty else 0}",
        f"Missing raw measurement values (audit): {int(audit_frame['missing_values'].sum()) if not audit_frame.empty else 0}",
        f"Resampled NaN cells interpolated or dropped: {cleaning_stats['resampled_nan_cells_before_interpolation']}",
        f"Rows removed after interpolation/synchronization: {cleaning_stats['rows_removed_after_interpolation_or_sync']}",
        f"Incomplete train windows discarded: {cleaning_stats['discarded_incomplete_windows_train']} ({cleaning_stats['discarded_tail_rows_train']} tail rows)",
        f"Incomplete test windows discarded: {cleaning_stats['discarded_incomplete_windows_test']} ({cleaning_stats['discarded_tail_rows_test']} tail rows)",
        "",
        "Selected original folders:",
    ]
    for activity in selected_activities:
        for split in ("train", "test"):
            lines.append(f"  {activities[activity][split].name} -> {split.upper()}")

    lines.extend(["", "Class distributions (windows):"])
    for split, metadata in (("TRAIN", train_metadata), ("TEST", test_metadata)):
        count_by_class = Counter(item["label"] for item in metadata)
        lines.append(f"  {split}:")
        for activity, label in sorted(class_mapping.items(), key=lambda item: item[1]):
            lines.append(f"    Class {label} ({activity}): {count_by_class[label]}")

    lines.extend(["", "Raw recordings, events, and windows per class:"])
    for activity, label in sorted(class_mapping.items(), key=lambda item: item[1]):
        class_events = [record for record in event_records if record["activity"] == activity]
        recordings = len({record["original_folder"] for record in class_events})
        event_count = len({(record["original_folder"], record["event"]) for record in class_events})
        train_count = sum(item["label"] == label for item in train_metadata)
        test_count = sum(item["label"] == label for item in test_metadata)
        lines.append(
            f"  {activity}: recordings={recordings}, events={event_count}, "
            f"train_windows={train_count}, test_windows={test_count}"
        )
    lines.extend(
        [
            "",
            "Per-file raw inspection: raw_file_audit.csv (rows, columns, malformed rows, missing values, timestamp conversion, sampling interval).",
            "Plots: figures/class_counts.png and figures/sensor_time_series.png.",
            "Validation data is not generated; the original test recording remains untouched.",
        ]
    )
    (output_dir / "dataset_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def _validate_selection(
    selected_activities: list[str], activities: dict[str, dict[str, Path]]
) -> None:
    if len(selected_activities) != 4 or len(set(selected_activities)) != 4:
        raise ValueError("Exactly four distinct activities must be selected.")
    missing = [
        f"{activity} ({split})"
        for activity in selected_activities
        for split in ("train", "test")
        if split not in activities.get(activity, {})
    ]
    if missing:
        raise ValueError("Selected activities must have canonical train and test folders: " + ", ".join(missing))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--activities",
        type=str,
        default=",".join(DEFAULT_ACTIVITIES),
        help="Exactly four comma-separated canonical S1 activity names.",
    )
    args = parser.parse_args()
    data_root = args.data_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    selected_activities = [name.strip() for name in args.activities.split(",") if name.strip()]
    if not data_root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {data_root}")

    activities = discover_activities(data_root)
    print_activity_inventory(activities)
    _validate_selection(selected_activities, activities)
    class_mapping = {activity: index for index, activity in enumerate(selected_activities)}
    print("\nSelected four-class mapping and original split folders:")
    for activity in selected_activities:
        print(f"  Class {class_mapping[activity]}: {activity}")
        print(f"    {activities[activity]['train'].name} -> TRAIN")
        print(f"    {activities[activity]['test'].name} -> TEST")

    audit_records: list[dict[str, Any]] = []
    event_records: list[dict[str, Any]] = []
    cleaning_stats: Counter[str] = Counter()
    for activity in selected_activities:
        label = class_mapping[activity]
        for split in ("train", "test"):
            folder = activities[activity][split]
            print(f"\nProcessing {folder.name} -> {split.upper()}")
            event_records.extend(
                process_activity(activity, folder, label, split, audit_records, cleaning_stats)
            )

    audit_frame = pd.DataFrame(audit_records)
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_frame.to_csv(output_dir / "raw_file_audit.csv", index=False)
    _audit_summary(audit_frame)

    training_events = [record for record in event_records if record["split"] == "train"]
    feature_names = _feature_intersection(training_events)
    if not feature_names:
        raise RuntimeError("No common sensor features are available across the selected events.")
    all_observed = set().union(*(set(record["frame"].columns) for record in event_records))
    excluded_features = sorted(all_observed - set(feature_names), key=str.casefold)
    if excluded_features:
        print(f"\nExcluded features absent from at least one synchronized event: {excluded_features}")

    x_train_raw, y_train, train_metadata = build_train_dataset(
        event_records, feature_names, cleaning_stats
    )
    x_test_raw, y_test, test_metadata = build_test_dataset(
        event_records, feature_names, cleaning_stats
    )
    if not len(x_train_raw) or not len(x_test_raw):
        raise RuntimeError(
            f"Expected non-empty train and test windows; got {len(x_train_raw)} train, "
            f"{len(x_test_raw)} test. See raw_file_audit.csv and dataset_summary.txt."
        )

    # Fit only on training timesteps; the original test recording never affects scaler parameters.
    scaler = StandardScaler()
    scaler.fit(x_train_raw.reshape(-1, x_train_raw.shape[-1]))
    x_train = scaler.transform(x_train_raw.reshape(-1, x_train_raw.shape[-1])).reshape(x_train_raw.shape)
    x_test = scaler.transform(x_test_raw.reshape(-1, x_test_raw.shape[-1])).reshape(x_test_raw.shape)

    test_counts = Counter(y_test.tolist())
    save_dataset(
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
        test_counts,
    )
    create_summary(
        output_dir,
        activities,
        selected_activities,
        class_mapping,
        event_records,
        train_metadata,
        test_metadata,
        x_train,
        x_test,
        feature_names,
        audit_frame,
        cleaning_stats,
    )
    print(f"\nSaved processed dataset to: {output_dir}")
    print(f"X_train: {x_train.shape}; y_train: {y_train.shape}")
    print(f"X_test:  {x_test.shape}; y_test:  {y_test.shape}")


if __name__ == "__main__":
    main()