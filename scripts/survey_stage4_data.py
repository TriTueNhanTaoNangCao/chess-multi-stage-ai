"""Streaming metadata survey and reservoir-sampled PGN validation for Stage 4."""

from __future__ import annotations

import argparse
import ctypes
import csv
import hashlib
import heapq
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

import chess
import chess.pgn


SEED = 42
RESERVOIR_SIZE = 2_000
KNOWN_RESULTS = {"1-0", "0-1", "1/2-1/2", "*"}
DECISIVE_OR_DRAW_RESULTS = {"1-0", "0-1", "1/2-1/2"}
TIME_CONTROL_RE = re.compile(r"^(?:\d+/\d+:)*\d+(?:\+\d+)?$")
DATE_RE = re.compile(r"^(\d{4})\.(\d{2})\.(\d{2})$")


class RunningMedian:
    def __init__(self) -> None:
        self.lower: list[int] = []
        self.upper: list[int] = []

    def add(self, value: int) -> None:
        if not self.lower or value <= -self.lower[0]:
            heapq.heappush(self.lower, -value)
        else:
            heapq.heappush(self.upper, value)
        if len(self.lower) > len(self.upper) + 1:
            heapq.heappush(self.upper, -heapq.heappop(self.lower))
        elif len(self.upper) > len(self.lower):
            heapq.heappush(self.lower, -heapq.heappop(self.upper))

    def value(self) -> float | None:
        if not self.lower:
            return None
        if len(self.lower) == len(self.upper):
            return (-self.lower[0] + self.upper[0]) / 2
        return float(-self.lower[0])


class EloStats:
    def __init__(self) -> None:
        self.valid = 0
        self.missing = 0
        self.invalid = 0
        self.minimum: int | None = None
        self.maximum: int | None = None
        self.median = RunningMedian()

    def add(self, raw: str | None) -> bool:
        if raw is None or not raw.strip():
            self.missing += 1
            return False
        try:
            value = int(raw)
        except ValueError:
            self.invalid += 1
            return False
        self.valid += 1
        self.minimum = value if self.minimum is None else min(self.minimum, value)
        self.maximum = value if self.maximum is None else max(self.maximum, value)
        self.median.add(value)
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "missing": self.missing,
            "invalid": self.invalid,
            "min": self.minimum,
            "median": self.median.value(),
            "max": self.maximum,
        }


def normalized_variant(raw: str | None) -> str:
    value = (raw or "Standard").strip().lower()
    return "standard" if value in {"standard", "chess"} else value or "standard"


def valid_date(raw: str | None) -> tuple[bool, str | None, str | None]:
    if raw is None or not raw.strip():
        return False, None, None
    match = DATE_RE.fullmatch(raw.strip())
    if not match:
        return False, None, None
    year, month, day = match.groups()
    if not (1 <= int(month) <= 12 and 1 <= int(day) <= 31):
        return False, None, None
    return True, year, f"{year}-{month}"


def percentile(values: list[int], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scan_headers(path: Path, reservoir_size: int, seed: int) -> tuple[dict[str, Any], list[tuple[int, int]]]:
    rng = random.Random(seed)
    reservoir: list[tuple[int, int]] = []
    players: set[str] = set()
    white_elo = EloStats()
    black_elo = EloStats()
    time_controls: Counter[str] = Counter()
    results: Counter[str] = Counter()
    variants: Counter[str] = Counter()
    normalized_variants: Counter[str] = Counter()
    date_years: Counter[str] = Counter()
    date_months: Counter[str] = Counter()
    total = 0
    tc_missing = 0
    tc_ununderstood = 0
    date_missing = 0
    date_invalid = 0
    metadata_counts = Counter()

    with path.open("r", encoding="utf-8", errors="replace") as stream:
        while True:
            offset = stream.tell()
            headers = chess.pgn.read_headers(stream)
            if headers is None:
                break
            total += 1
            item = (total, offset)
            if len(reservoir) < reservoir_size:
                reservoir.append(item)
            else:
                replacement = rng.randrange(total)
                if replacement < reservoir_size:
                    reservoir[replacement] = item

            for key in ("White", "Black"):
                player = headers.get(key, "").strip()
                if player:
                    players.add(player)

            white_ok = white_elo.add(headers.get("WhiteElo"))
            black_ok = black_elo.add(headers.get("BlackElo"))

            tc = headers.get("TimeControl", "").strip()
            if not tc:
                tc_missing += 1
                time_controls["<missing>"] += 1
                tc_ok = False
            else:
                time_controls[tc] += 1
                tc_ok = TIME_CONTROL_RE.fullmatch(tc) is not None
                if not tc_ok:
                    tc_ununderstood += 1

            result = headers.get("Result", "").strip() or "<missing>"
            results[result] += 1
            result_ok = result in DECISIVE_OR_DRAW_RESULTS

            variant_raw = headers.get("Variant", "").strip()
            variants[variant_raw or "<missing/default Standard>"] += 1
            variant = normalized_variant(variant_raw)
            normalized_variants[variant] += 1
            variant_ok = variant == "standard"

            date_raw = headers.get("Date", "").strip()
            date_ok, year, month = valid_date(date_raw)
            if not date_raw:
                date_missing += 1
            elif not date_ok:
                date_invalid += 1
            else:
                date_years[year] += 1
                date_months[month] += 1

            metadata_counts["both_elos_valid"] += white_ok and black_ok
            metadata_counts["time_control_understood"] += tc_ok
            metadata_counts["known_result"] += result_ok
            metadata_counts["valid_date"] += date_ok
            metadata_counts["standard_variant"] += variant_ok
            metadata_counts["strict_intersection"] += (
                white_ok and black_ok and tc_ok and result_ok and date_ok and variant_ok
            )

    return {
        "scope": "full_file_headers",
        "total_games": total,
        "unique_players": len(players),
        "white_elo": white_elo.to_dict(),
        "black_elo": black_elo.to_dict(),
        "time_control": {
            "distribution": dict(time_controls.most_common()),
            "missing": tc_missing,
            "missing_rate": tc_missing / total if total else 0,
            "ununderstood": tc_ununderstood,
            "ununderstood_rate": tc_ununderstood / total if total else 0,
        },
        "result_distribution": dict(results.most_common()),
        "date": {
            "missing": date_missing,
            "invalid_or_partial": date_invalid,
            "year_distribution": dict(sorted(date_years.items())),
            "month_distribution": dict(sorted(date_months.items())),
        },
        "variant_distribution": dict(variants.most_common()),
        "normalized_variant_distribution": dict(normalized_variants.most_common()),
        "metadata_eligibility": {
            **dict(metadata_counts),
            "strict_intersection_definition": (
                "standard variant + both numeric Elo values + understood TimeControl + "
                "known finished result + valid full date; diagnostic only, not a selected filter"
            ),
        },
    }, reservoir


def inspect_reservoir(path: Path, reservoir: list[tuple[int, int]]) -> dict[str, Any]:
    parse_error_games = 0
    parse_error_types: Counter[str] = Counter()
    invalid_root_games = 0
    illegal_move_games = 0
    standard_start_games = 0
    custom_start_games = 0
    ply_lengths: list[int] = []
    fingerprints: Counter[str] = Counter()
    players: Counter[str] = Counter()
    openings: Counter[str] = Counter()
    ecos: Counter[str] = Counter()

    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for _, offset in sorted(reservoir, key=lambda item: item[1]):
            stream.seek(offset)
            game = chess.pgn.read_game(stream)
            if game is None:
                parse_error_games += 1
                parse_error_types["missing_game_at_offset"] += 1
                continue
            if game.errors:
                parse_error_games += 1
                for error in game.errors:
                    parse_error_types[type(error).__name__] += 1

            for key in ("White", "Black"):
                value = game.headers.get(key, "").strip()
                if value:
                    players[value] += 1
            opening = game.headers.get("Opening", "").strip()
            eco = game.headers.get("ECO", "").strip()
            if opening:
                openings[opening] += 1
            if eco:
                ecos[eco] += 1

            try:
                board = game.board()
            except Exception as error:
                invalid_root_games += 1
                parse_error_types[type(error).__name__] += 1
                continue
            if not board.is_valid():
                invalid_root_games += 1
                continue
            if board.fen(en_passant="fen") == chess.STARTING_FEN:
                standard_start_games += 1
            else:
                custom_start_games += 1

            move_ucis: list[str] = []
            illegal = False
            try:
                for move in game.mainline_moves():
                    if move not in board.legal_moves:
                        illegal = True
                        break
                    move_ucis.append(move.uci())
                    board.push(move)
            except Exception as error:
                illegal = True
                parse_error_types[type(error).__name__] += 1
            if illegal:
                illegal_move_games += 1
                continue

            ply_lengths.append(len(move_ucis))
            root = game.board().root().fen(en_passant="fen")
            variant = normalized_variant(game.headers.get("Variant"))
            canonical = f"{variant}|{root}|{' '.join(move_ucis)}"
            fingerprints[hashlib.sha256(canonical.encode("utf-8")).hexdigest()] += 1

    duplicate_games = sum(count - 1 for count in fingerprints.values() if count > 1)
    duplicate_groups = sum(1 for count in fingerprints.values() if count > 1)
    return {
        "scope": "reservoir_sample",
        "seed": SEED,
        "requested_size": RESERVOIR_SIZE,
        "sample_size": len(reservoir),
        "parse_error_games": parse_error_games,
        "parse_error_types": dict(parse_error_types.most_common()),
        "initial_board": {
            "standard_start": standard_start_games,
            "custom_start": custom_start_games,
            "invalid_root": invalid_root_games,
        },
        "illegal_move_games": illegal_move_games,
        "ply_length": {
            "count": len(ply_lengths),
            "min": min(ply_lengths) if ply_lengths else None,
            "median": percentile(ply_lengths, 0.5),
            "p90": percentile(ply_lengths, 0.9),
            "max": max(ply_lengths) if ply_lengths else None,
        },
        "fingerprints": {
            "valid_games": sum(fingerprints.values()),
            "unique": len(fingerprints),
            "duplicate_games": duplicate_games,
            "duplicate_groups": duplicate_groups,
        },
        "top_players": players.most_common(20),
        "top_openings": openings.most_common(20),
        "top_eco": ecos.most_common(20),
        "opening_metadata_present_games": sum(openings.values()),
        "eco_metadata_present_games": sum(ecos.values()),
        "claim_draw_policy": "Claimable draws are not treated as mandatory termination during move validation.",
    }


def survey(path: Path, label: str) -> dict[str, Any]:
    full, reservoir = scan_headers(path, RESERVOIR_SIZE, SEED)
    sample = inspect_reservoir(path, reservoir)
    return {
        "label": label,
        "path": path.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "full_file": full,
        "sample": sample,
    }


def comparison_row(item: dict[str, Any]) -> dict[str, Any]:
    full = item["full_file"]
    sample = item["sample"]
    return {
        "dataset": item["label"],
        "total_games": full["total_games"],
        "unique_players": full["unique_players"],
        "white_elo_missing": full["white_elo"]["missing"],
        "white_elo_min": full["white_elo"]["min"],
        "white_elo_median": full["white_elo"]["median"],
        "white_elo_max": full["white_elo"]["max"],
        "black_elo_missing": full["black_elo"]["missing"],
        "black_elo_min": full["black_elo"]["min"],
        "black_elo_median": full["black_elo"]["median"],
        "black_elo_max": full["black_elo"]["max"],
        "time_control_missing": full["time_control"]["missing"],
        "time_control_ununderstood": full["time_control"]["ununderstood"],
        "strict_metadata_candidate": full["metadata_eligibility"]["strict_intersection"],
        "sample_size": sample["sample_size"],
        "sample_parse_error_games": sample["parse_error_games"],
        "sample_invalid_root_games": sample["initial_board"]["invalid_root"],
        "sample_illegal_move_games": sample["illegal_move_games"],
        "sample_ply_min": sample["ply_length"]["min"],
        "sample_ply_median": sample["ply_length"]["median"],
        "sample_ply_p90": sample["ply_length"]["p90"],
        "sample_ply_max": sample["ply_length"]["max"],
        "sample_duplicate_games": sample["fingerprints"]["duplicate_games"],
    }


def machine_memory() -> dict[str, int]:
    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("length", ctypes.c_ulong),
            ("memory_load", ctypes.c_ulong),
            ("total_physical", ctypes.c_ulonglong),
            ("available_physical", ctypes.c_ulonglong),
            ("total_page_file", ctypes.c_ulonglong),
            ("available_page_file", ctypes.c_ulonglong),
            ("total_virtual", ctypes.c_ulonglong),
            ("available_virtual", ctypes.c_ulonglong),
            ("available_extended_virtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("GlobalMemoryStatusEx failed")
    return {
        "total_physical_bytes": status.total_physical,
        "available_physical_bytes": status.available_physical,
    }


def array_memory_estimates() -> dict[str, Any]:
    components = {
        "states_float32_8x8x14": 8 * 8 * 14 * 4,
        "targets_int32": 4,
        "legal_masks_bool_4672": 4672,
        "game_ids_unicode_U64": 64 * 4,
        "ply_indices_int32": 4,
        "outcomes_float32": 4,
        "outcome_known_bool": 1,
    }
    final_per_sample = sum(components.values())
    source_array_per_sample = components["states_float32_8x8x14"] + components["legal_masks_bool_4672"]
    estimates = {}
    for count in (100_000, 500_000, 1_000_000):
        final_bytes = final_per_sample * count
        retained_source_array_bytes = source_array_per_sample * count
        estimates[str(count)] = {
            "final_dense_arrays_bytes": final_bytes,
            "final_dense_arrays_gib": final_bytes / (1024**3),
            "retained_per_sample_state_and_mask_bytes": retained_source_array_bytes,
            "retained_per_sample_state_and_mask_gib": retained_source_array_bytes / (1024**3),
            "peak_lower_bound_bytes": final_bytes + retained_source_array_bytes,
            "peak_lower_bound_gib": (final_bytes + retained_source_array_bytes) / (1024**3),
        }
    return {
        "build_dataset_reference": "training/dataset.py:318-344",
        "behavior": (
            "ChessSample objects remain grouped by split; np.stack copies state and legal_mask "
            "into final dense arrays, while np.array creates the remaining dense arrays."
        ),
        "final_dense_bytes_per_sample": final_per_sample,
        "final_components_bytes_per_sample": components,
        "retained_source_state_and_mask_bytes_per_sample": source_array_per_sample,
        "estimates": estimates,
        "unmeasured_overhead": [
            "ChessSample and Python scalar/string objects",
            "NumPy array object headers and allocator fragmentation",
            "temporary Python reference lists passed to np.stack/np.array",
            "possible astype and np.stack temporaries",
            "np.savez_compressed buffers",
        ],
    }


def inspect_short_games(path: Path, reservoir: list[tuple[int, int]]) -> dict[str, Any]:
    bins = Counter()
    shortest: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for game_index, offset in sorted(reservoir, key=lambda item: item[1]):
            stream.seek(offset)
            game = chess.pgn.read_game(stream)
            if game is None:
                continue
            plies = sum(1 for _ in game.mainline_moves())
            if plies < 10:
                bins["<10"] += 1
            elif plies < 20:
                bins["10-19"] += 1
            elif plies < 40:
                bins["20-39"] += 1
            else:
                bins[">=40"] += 1
            shortest.append({
                "game_index": game_index,
                "plies": plies,
                "result": game.headers.get("Result", ""),
                "headers": dict(game.headers),
            })
    shortest.sort(key=lambda item: (item["plies"], item["game_index"]))
    return {
        "scope": "reservoir_sample",
        "seed": SEED,
        "sample_size": len(reservoir),
        "ply_bins": {key: bins[key] for key in ("<10", "10-19", "20-39", ">=40")},
        "five_shortest": shortest[:5],
    }


def supplementary_survey(path: Path) -> dict[str, Any]:
    full, reservoir = scan_headers(path, RESERVOIR_SIZE, SEED)
    return {
        "source": {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        },
        "full_file": {
            "total_games": full["total_games"],
            "games_by_yyyy_mm": full["date"]["month_distribution"],
            "date_missing": full["date"]["missing"],
            "date_invalid_or_partial": full["date"]["invalid_or_partial"],
        },
        "sample": inspect_short_games(path, reservoir),
        "machine_memory_at_run": machine_memory(),
        "dataset_build_memory_estimate": array_memory_estimates(),
        "no_tensor_dataset_or_training": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--elite", type=Path, required=True)
    parser.add_argument("--carlsen", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--supplement-output", type=Path)
    args = parser.parse_args()

    if not args.elite.is_file():
        parser.error(f"PGN not found: {args.elite}")

    if args.supplement_output is not None:
        if args.supplement_output.exists():
            raise FileExistsError(f"Refusing to overwrite existing result: {args.supplement_output}")
        payload = supplementary_survey(args.elite)
        args.supplement_output.parent.mkdir(parents=True, exist_ok=True)
        with args.supplement_output.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if args.carlsen is None or args.output_dir is None:
        parser.error("--carlsen and --output-dir are required for the full comparison survey")
    if not args.carlsen.is_file():
        parser.error(f"PGN not found: {args.carlsen}")

    summary_path = args.output_dir / "survey_summary.json"
    comparison_path = args.output_dir / "comparison.csv"
    for path in (summary_path, comparison_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing result: {path}")

    elite = survey(args.elite, "lichess_elite_2023-02")
    carlsen = survey(args.carlsen, "carlsen")
    payload = {
        "method": {
            "header_scope": "All games, streamed with chess.pgn.read_headers().",
            "deep_scope": "Reservoir sample of at most 2,000 game offsets, seed 42; only sampled games are fully parsed.",
            "no_tensor_dataset_or_training": True,
        },
        "source": {
            "page_url": "https://database.nikonoel.fr/",
            "download_url": "https://database.nikonoel.fr/lichess_elite_2023-02.zip",
            "month": "2023-02",
            "published_compressed_size": "64 MB",
            "selection_criteria": "Standard games; 2500+ rated players vs 2300+ rated players; bullet excluded.",
            "source_page_license_notice": "No explicit license notice found on the Lichess Elite Database page.",
            "upstream_lichess_database_license": "Creative Commons CC0; research, commercial use, publication, modification and redistribution are permitted without asking permission.",
            "upstream_license_url": "https://database.lichess.org/",
        },
        "datasets": [elite, carlsen],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with summary_path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)

    rows = [comparison_row(elite), comparison_row(carlsen)]
    with comparison_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps({"summary": str(summary_path), "comparison": str(comparison_path), "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
