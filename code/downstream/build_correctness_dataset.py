"""Build candidate-plan correctness datasets and frozen-model feature caches."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from code.downstream.families import resolve_source_families
from code.downstream.features import FrozenTransitionFeatureExtractor, save_feature_matrix
from code.downstream.plan_utils import (
    discover_val_path,
    generate_labeled_candidates_for_problem,
    iter_problem_names,
    label_plan_internal,
    load_problem_context,
    read_jsonl,
    recover_gold_plan,
    summarize_labels,
    write_jsonl,
)
from code.downstream.config import DATA_DIR, DOMAINS, SOURCE_MODEL_DIR, SPLITS_EVAL, CORRECTNESS_DATASET_DIR


DEFAULT_SPLITS = ["train", *SPLITS_EVAL]
DEFAULT_SEEDS = [13, 23, 37]


def build_candidates(args) -> dict[str, dict]:
    """Recover positives, generate corruptions, and label all candidates."""
    rng = random.Random(args.candidate_seed)
    summaries: dict[str, dict] = {}
    candidates_dir = Path(args.output_dir) / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        split_rows: list[dict] = []
        for domain in args.domains:
            problem_names = iter_problem_names(
                data_dir=args.source_data_dir,
                domain=domain,
                split=split,
                max_problems=args.max_problems,
            )
            for problem in problem_names:
                context = load_problem_context(
                    data_dir=args.source_data_dir,
                    domain=domain,
                    split=split,
                    problem=problem,
                )
                gold_plan = recover_gold_plan(
                    data_dir=args.source_data_dir,
                    domain=domain,
                    split=split,
                    problem=problem,
                )
                validator = None
                if args.labeler == "internal":
                    validator = lambda plan, ctx=context: label_plan_internal(ctx, plan)
                rows = generate_labeled_candidates_for_problem(
                    context=context,
                    gold_plan=gold_plan,
                    val_path=args.val_path,
                    negative_ratio=args.negative_ratio,
                    rng=rng,
                    validator=validator,
                    max_attempts_per_negative=args.max_attempts_per_negative,
                )
                split_rows.extend(rows)

        path = candidates_dir / f"{split}.jsonl"
        write_jsonl(path, split_rows)
        summaries[split] = summarize_labels(split_rows)
        print(f"Wrote {path} | {summaries[split]}")

    return summaries


def build_features(args, families) -> dict[str, dict]:
    """Extract frozen transition-model features for all selected families."""
    feature_root = Path(args.output_dir) / "features"
    summaries: dict[str, dict] = {}

    for family in families:
        for seed in args.seeds:
            extractor = FrozenTransitionFeatureExtractor(
                run_root=args.run_root,
                source_data_dir=args.source_data_dir,
                family=family,
                seed=seed,
                device=args.device,
                xgb_n_jobs=args.xgb_n_jobs,
            )
            family_summary: dict[str, dict] = {}

            for split in args.splits:
                candidate_path = Path(args.output_dir) / "candidates" / f"{split}.jsonl"
                if not candidate_path.exists():
                    raise FileNotFoundError(
                        f"Missing candidate file {candidate_path}. Build candidates first."
                    )
                candidates = read_jsonl(candidate_path)
                out_path = feature_root / family.family_id / f"seed_{seed}" / f"{split}.npz"
                if out_path.exists() and not args.overwrite_features:
                    print(f"Skipping existing features: {out_path}")
                    family_summary[split] = {"skipped_existing": True}
                    continue

                rows = []
                for idx, candidate in enumerate(candidates, start=1):
                    rows.append(extractor.extract(candidate))
                    if args.progress_every and idx % args.progress_every == 0:
                        print(
                            f"  {family.family_id}/seed_{seed}/{split}: "
                            f"{idx}/{len(candidates)}"
                        )

                X = np.vstack(rows).astype(np.float32) if rows else np.zeros((0, 0), dtype=np.float32)
                save_feature_matrix(
                    path=out_path,
                    candidates=candidates,
                    features=X,
                    feature_names=extractor.feature_names,
                )
                family_summary[split] = {
                    "path": str(out_path),
                    "num_rows": int(X.shape[0]),
                    "num_features": int(X.shape[1]) if X.ndim == 2 else 0,
                }
                print(f"Wrote {out_path} | {family_summary[split]}")

            summaries[f"{family.family_id}/seed_{seed}"] = family_summary

    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build frozen-transition-model plan-validity data."
    )
    parser.add_argument("--run_root", default=str(SOURCE_MODEL_DIR))
    parser.add_argument("--source_data_dir", default=str(DATA_DIR))
    parser.add_argument(
        "--output_dir",
        default=str(CORRECTNESS_DATASET_DIR),
    )
    parser.add_argument("--domains", nargs="+", default=DOMAINS)
    parser.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument(
        "--source_families",
        nargs="+",
        default=["weighted_best"],
        help="Family ids or 'weighted_best'.",
    )
    parser.add_argument("--negative_ratio", type=int, default=4)
    parser.add_argument("--candidate_seed", type=int, default=2026)
    parser.add_argument("--max_attempts_per_negative", type=int, default=25)
    parser.add_argument("--max_problems", type=int, default=None)
    parser.add_argument("--val_path", default=None)
    parser.add_argument(
        "--labeler",
        choices=["val", "internal"],
        default="val",
        help="'internal' is for local smoke tests when VAL is unavailable.",
    )
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="cpu")
    parser.add_argument("--xgb_n_jobs", type=int, default=1)
    parser.add_argument("--progress_every", type=int, default=100)
    parser.add_argument("--skip_candidates", action="store_true")
    parser.add_argument("--skip_features", action="store_true")
    parser.add_argument("--overwrite_features", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    args.run_root = str(Path(args.run_root).resolve())
    args.source_data_dir = str(Path(args.source_data_dir).resolve())
    args.output_dir = str(Path(args.output_dir).resolve())
    args.val_path = discover_val_path(repo_root, args.val_path)
    if args.labeler == "val" and args.val_path is None and not args.skip_candidates:
        raise RuntimeError("VAL binary was not found. Provide --val_path.")

    families = resolve_source_families(args.source_families)
    manifest = {
        "run_root": args.run_root,
        "source_data_dir": args.source_data_dir,
        "output_dir": args.output_dir,
        "domains": args.domains,
        "splits": args.splits,
        "seeds": args.seeds,
        "source_families": [family.family_id for family in families],
        "negative_ratio": args.negative_ratio,
        "candidate_seed": args.candidate_seed,
        "val_path": args.val_path,
        "labeler": args.labeler,
    }
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    summaries: dict[str, dict] = {}
    if not args.skip_candidates:
        summaries["candidates"] = build_candidates(args)
    if not args.skip_features:
        summaries["features"] = build_features(args, families)

    manifest["summaries"] = summaries
    manifest_path = Path(args.output_dir) / "build_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
