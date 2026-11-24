"""Build Stage 4 NPY train/val data from ranked Jan--Mar 2024 PGNs.

Each month contributes exactly 100,000 accepted games, split 90,000/10,000.
Samples are buffered in small batches and appended to private NPY files; only
headers are finalized once row counts are known. No full sample corpus or
second disk copy is needed. A compatible manifest is published only on success.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import time

import numpy as np

from scripts.stage4.pgn_reader_300k import MONTHS, accept_candidate, iter_candidates
from scripts.stage4.pgn_samples_300k import extract_samples


GAMES_PER_MONTH = 100_000
TRAIN_PER_MONTH = 90_000
MAX_PLIES_PER_GAME = 32
ARRAYS = {
    "states": (np.dtype("float32"), (8, 8, 14)),
    "targets": (np.dtype("int32"), ()),
    "legal_masks": (np.dtype("bool"), (4672,)),
}


def _atomic_json(path, payload):
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _npy_header(dtype, trailing_shape, rows):
    """NumPy reserves growth-axis digits in this standard NPY 2.0 header."""
    stream = io.BytesIO()
    np.lib.format.write_array_header_2_0(stream, {
        "descr": np.lib.format.dtype_to_descr(dtype),
        "fortran_order": False,
        "shape": (rows, *trailing_shape),
    })
    return stream.getvalue()


class _NpyStream:
    """Append C-order rows, then update the fixed-size header in place."""
    def __init__(self, output, split, key, max_rows):
        self.dtype, self.trailing_shape = ARRAYS[key]
        self.final_path = output / f"{split}_{key}.npy"
        self.temporary = output / f".{split}_{key}.npy.partial"
        self.rows = 0
        self.max_rows = max_rows
        header = _npy_header(self.dtype, self.trailing_shape, 0)
        self.header_size = len(header)
        # Check digit-width transitions before writing any sample data.
        counts = [0, max_rows, *(10 ** i for i in range(len(str(max_rows))))]
        if any(len(_npy_header(self.dtype, self.trailing_shape, n)) != self.header_size
               for n in counts):
            raise RuntimeError("NumPy NPY header does not reserve a fixed growth axis")
        self.stream = self.temporary.open("xb")
        self.stream.write(header)

    def append(self, array):
        array = np.ascontiguousarray(array, dtype=self.dtype)
        if array.shape[1:] != self.trailing_shape:
            raise ValueError("Unexpected sample array shape")
        if self.rows + len(array) > self.max_rows:
            raise ValueError("Sample count exceeds the configured game/ply quota")
        self.stream.write(array.tobytes(order="C"))
        self.rows += len(array)
        self.stream.flush()

    def finalize(self):
        header = _npy_header(self.dtype, self.trailing_shape, self.rows)
        if len(header) != self.header_size:
            raise RuntimeError("Final NPY header length changed")
        expected_bytes = (self.header_size + self.rows * self.dtype.itemsize
                          * int(np.prod(self.trailing_shape, dtype=np.int64)))
        if self.stream.tell() != expected_bytes:
            raise RuntimeError("NPY byte count differs from the row count")
        self.stream.seek(0)
        self.stream.write(header)
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.stream.close()
        os.replace(self.temporary, self.final_path)

    def close(self):
        self.stream.close()


class _SplitWriter:
    def __init__(self, output, split, batch_samples, max_rows):
        self.batch_samples = batch_samples
        self.buffer = []
        self.rows = 0
        self.arrays = {}
        try:
            for key in ARRAYS:
                self.arrays[key] = _NpyStream(output, split, key, max_rows)
        except BaseException:
            self.close()
            raise

    def append_game(self, samples):
        start = self.rows
        self.buffer.extend(samples)
        self.rows += len(samples)
        if len(self.buffer) >= self.batch_samples:
            self.flush()
        return start, self.rows

    def flush(self):
        if not self.buffer:
            return
        # Construct one field at a time; at most batch_samples + 31 samples.
        for key, attribute in (("states", "state"), ("targets", "target"),
                               ("legal_masks", "legal_mask")):
            array = np.asarray([getattr(s, attribute) for s in self.buffer],
                               dtype=ARRAYS[key][0])
            self.arrays[key].append(array)
            del array
        self.buffer.clear()

    def finalize(self):
        self.flush()
        for array in self.arrays.values():
            if array.rows != self.rows:
                raise RuntimeError("Inconsistent sample counts across arrays")
            array.finalize()

    def close(self):
        for array in self.arrays.values():
            array.close()


def _split_plans(seed):
    rng = np.random.default_rng(seed)
    plans = {}
    for month in MONTHS:
        plan = np.full(GAMES_PER_MONTH, "train", dtype="<U5")
        plan[rng.permutation(GAMES_PER_MONTH)[TRAIN_PER_MONTH:]] = "val"
        plans[month] = plan
    return plans


def build_dataset(pgn_paths, output_dir, seed=42, batch_samples=2048):
    """Create a new dataset directory; refuse existing outputs, no resume."""
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite dataset: {output}")
    if set(pgn_paths) != set(MONTHS):
        raise ValueError(f"Expected exactly these months: {MONTHS}")
    if batch_samples <= 0:
        raise ValueError("batch_samples must be positive")
    paths = {month: Path(pgn_paths[month]).resolve() for month in MONTHS}
    sources = {}
    for month, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        sources[month] = {"path": str(path), "bytes": path.stat().st_size,
                          "sha256": _sha256(path)}
    output.mkdir(parents=True)
    stats = {month: {
        "accepted_games": 0, "sample_rejected_games": 0,
        "num_games": {"train": 0, "val": 0},
        "num_samples": {"train": 0, "val": 0},
    } for month in MONTHS}
    writers = {}
    started = time.perf_counter()
    status_path = output / "build_status.json"
    try:
        _atomic_json(status_path, {"status": "building", "by_month": stats})
        for split, games in (("train", TRAIN_PER_MONTH),
                             ("val", GAMES_PER_MONTH - TRAIN_PER_MONTH)):
            writers[split] = _SplitWriter(output, split, batch_samples,
                                         len(MONTHS) * games * MAX_PLIES_PER_GAME)
        plans = _split_plans(seed)
        # Sampling and split RNGs are separate; rejected games cannot shift splits.
        sample_rng = np.random.default_rng(seed)
        accepted = set()
        candidates = iter_candidates(
            paths, seed, accepted,
            month_done=lambda month: stats[month]["accepted_games"] >= GAMES_PER_MONTH,
        )
        try:
            with (output / "games.jsonl").open("x", encoding="utf-8") as game_log:
                for candidate in candidates:
                    month = candidate.month
                    samples = extract_samples(candidate.game, sample_rng)
                    if not samples:
                        stats[month]["sample_rejected_games"] += 1
                        continue
                    index = stats[month]["accepted_games"]
                    split = str(plans[month][index])
                    start, stop = writers[split].append_game(samples)
                    if not accept_candidate(candidate, accepted):
                        raise RuntimeError("Already-accepted fingerprint reached the writer")
                    stats[month]["accepted_games"] += 1
                    stats[month]["num_games"][split] += 1
                    stats[month]["num_samples"][split] += len(samples)
                    game_log.write(json.dumps({
                        "month": month, "source_offset": candidate.offset,
                        "fingerprint": candidate.fingerprint, "split": split,
                        "sample_row_start": start, "sample_row_stop": stop,
                        "ply_indices": [s.ply_index for s in samples],
                    }) + "\n")
                    if stats[month]["accepted_games"] % 1000 == 0:
                        game_log.flush()
                        _atomic_json(status_path, {"status": "building", "by_month": stats})
                        print(f'{month}: {stats[month]["accepted_games"]}/100000 accepted games',
                              flush=True)
        finally:
            candidates.close()
        deficits = {month: GAMES_PER_MONTH - row["accepted_games"]
                    for month, row in stats.items() if row["accepted_games"] < GAMES_PER_MONTH}
        if deficits:
            raise ValueError(f"Insufficient accepted games; missing quota by month: {deficits}")
        for month, row in stats.items():
            if row["num_games"] != {"train": TRAIN_PER_MONTH,
                                    "val": GAMES_PER_MONTH - TRAIN_PER_MONTH}:
                raise RuntimeError(f"Incorrect game split for {month}")
        for writer in writers.values():
            writer.finalize()
        manifest = {
            "format_version": 1, "status": "complete", "purpose": "stage4_300k",
            "encoder_version": "hwc14-white-v1", "action_version": "az73-rankflip-v1",
            "num_actions": 4672, "seed": seed,
            "split_ratios": {"train": 0.9, "val": 0.1, "test": 0.0},
            "split_unit": "accepted game within each month; seeded permutation of acceptance indices",
            "rng": {"ranking": "independent default_rng(seed) per month",
                    "sampling": "one default_rng(seed) across months",
                    "split": "separate default_rng(seed), sequential monthly permutations"},
            "max_samples_per_game": MAX_PLIES_PER_GAME,
            "pgn_draw_claim_policy": {"claim_draw": False},
            "fingerprint": "sha256(space-joined mainline UCI encoded as UTF-8)",
            "duplicate_handling": "across months; marked only after successful sample extraction",
            "source_metadata": {"sources_by_month": sources}, "by_month": stats,
            "num_games": {split: sum(row["num_games"][split] for row in stats.values())
                          for split in ("train", "val")},
            "num_samples": {split: writer.rows for split, writer in writers.items()},
            "arrays": {f"{split}_{key}.npy": {
                "shape": [writer.rows, *shape], "dtype": str(dtype),
            } for split, writer in writers.items() for key, (dtype, shape) in ARRAYS.items()},
            "game_index": "games.jsonl; exclusive sample_row_stop, zero-based ply_indices",
            "batch_samples": batch_samples,
            "elapsed_seconds": time.perf_counter() - started,
            "leakage_note": "Complete-game UCI duplicates are excluded; positions in distinct games may coincide.",
        }
        manifest["num_games"]["test"] = manifest["num_samples"]["test"] = 0
        _atomic_json(output / "manifest.json", manifest)
        _atomic_json(status_path, {"status": "complete", "by_month": stats})
        print(json.dumps({"num_games": manifest["num_games"],
                          "num_samples": manifest["num_samples"]}), flush=True)
        return manifest
    except BaseException as exc:
        try:
            _atomic_json(status_path, {"status": "failed", "error": str(exc), "by_month": stats})
        except OSError:
            pass
        raise
    finally:
        for writer in writers.values():
            writer.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pgn-jan", type=Path, required=True)
    parser.add_argument("--pgn-feb", type=Path, required=True)
    parser.add_argument("--pgn-mar", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-samples", type=int, default=2048)
    args = parser.parse_args(argv)
    build_dataset(dict(zip(MONTHS, (args.pgn_jan, args.pgn_feb, args.pgn_mar))),
                  args.output_dir, args.seed, args.batch_samples)


if __name__ == "__main__":
    main()
