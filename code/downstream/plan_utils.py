"""Plan recovery, corruption, rollout, and labeling utilities."""

from __future__ import annotations

import json
import os
import random
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

PREDICATE_REGEX = re.compile(r"\(([\w-]+(?: [\w-]+)*)\)")
WHITESPACE_REGEX = re.compile(r"\s+")


@dataclass(frozen=True)
class ProblemContext:
    """Parsed symbolic context for one planning problem."""

    domain: str
    split: str
    problem: str
    domain_path: Path
    problem_path: Path
    task: object
    operators: tuple[object, ...]
    operator_by_name: dict[str, object]
    objects: tuple[str, ...]
    goal_atoms: tuple[str, ...]


def canonical_action(action: str) -> str:
    """Normalize action strings for pyperplan and VAL interoperability."""
    text = WHITESPACE_REGEX.sub(" ", str(action).strip().lower())
    if not text:
        return text
    if not text.startswith("("):
        text = f"({text})"
    return text


def parse_trajectory_state(line: str) -> frozenset[str]:
    """Parse one `.traj` line into the atom-string format used by pyperplan."""
    return frozenset(f"({match.strip()})" for match in PREDICATE_REGEX.findall(line.strip()))


def load_problem_context(
    *,
    data_dir: str | Path,
    domain: str,
    split: str,
    problem: str,
) -> ProblemContext:
    """Parse and ground a PDDL problem."""
    from pyperplan.grounding import ground
    from pyperplan.pddl.parser import Parser

    data_root = Path(data_dir)
    domain_path = data_root / "pddl" / domain / "domain.pddl"
    problem_path = data_root / "pddl" / domain / split / f"{problem}.pddl"

    parser = Parser(str(domain_path), str(problem_path))
    parsed_domain = parser.parse_domain()
    parsed_problem = parser.parse_problem(parsed_domain)
    task = ground(parsed_problem)
    operators = tuple(sorted(task.operators, key=lambda op: op.name))
    operator_by_name = {canonical_action(op.name): op for op in operators}
    objects = tuple(_extract_objects(parsed_problem, parsed_domain))
    goal_atoms = tuple(str(goal) for goal in task.goals)

    return ProblemContext(
        domain=domain,
        split=split,
        problem=problem,
        domain_path=domain_path,
        problem_path=problem_path,
        task=task,
        operators=operators,
        operator_by_name=operator_by_name,
        objects=objects,
        goal_atoms=goal_atoms,
    )


def _extract_objects(parsed_problem, parsed_domain) -> list[str]:
    """Extract sorted object and constant names from pyperplan structures."""
    objects: set[str] = set()
    problem_objects = getattr(parsed_problem, "objects", {})
    if isinstance(problem_objects, dict):
        objects.update(str(name) for name in problem_objects.keys())
    else:
        for obj in problem_objects:
            objects.add(obj.name if hasattr(obj, "name") else str(obj))

    constants = getattr(parsed_domain, "constants", {})
    if isinstance(constants, dict):
        objects.update(str(name) for name in constants.keys())
    else:
        for obj in constants:
            objects.add(obj.name if hasattr(obj, "name") else str(obj))
    return sorted(objects)


def load_trajectory_states(traj_path: str | Path) -> list[frozenset[str]]:
    """Load a symbolic state trajectory from disk."""
    with open(traj_path, "r", encoding="utf-8") as f:
        return [parse_trajectory_state(line) for line in f if line.strip()]


def recover_plan_from_states(
    states: list[frozenset[str]],
    operators: tuple[object, ...],
) -> list[str]:
    """Recover the unique grounded action that connects each trajectory step."""
    recovered: list[str] = []
    for step_idx, (current, next_state) in enumerate(zip(states, states[1:])):
        matches = [
            op.name
            for op in operators
            if op.applicable(set(current)) and frozenset(op.apply(set(current))) == next_state
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"Could not recover a unique action at step {step_idx}: "
                f"found {len(matches)} matches"
            )
        recovered.append(canonical_action(matches[0]))
    return recovered


def recover_gold_plan(
    *,
    data_dir: str | Path,
    domain: str,
    split: str,
    problem: str,
) -> list[str]:
    """Recover a gold action plan from the existing state trajectory."""
    context = load_problem_context(
        data_dir=data_dir,
        domain=domain,
        split=split,
        problem=problem,
    )
    traj_path = Path(data_dir) / "states" / domain / split / f"{problem}.traj"
    states = load_trajectory_states(traj_path)
    return recover_plan_from_states(states, context.operators)


def rollout_plan(context: ProblemContext, plan: list[str]) -> list[frozenset[str]]:
    """
    Roll out a candidate plan into a state sequence.

    Inapplicable or unknown actions are represented as stalled no-op transitions.
    This keeps feature extraction fixed-length without passing explicit symbolic
    failure flags to the downstream classifier.
    """
    states = [frozenset(context.task.initial_state)]
    current = set(context.task.initial_state)

    for action in plan:
        op = context.operator_by_name.get(canonical_action(action))
        if op is not None and op.applicable(current):
            current = set(op.apply(current))
        states.append(frozenset(current))

    return states


def label_plan_internal(context: ProblemContext, plan: list[str]) -> tuple[bool, bool]:
    """
    Label a plan with pyperplan semantics.

    This is intended for smoke tests when VAL is unavailable. Paper-quality
    dataset builds should use VAL.
    """
    current = set(context.task.initial_state)
    executable = True
    for action in plan:
        op = context.operator_by_name.get(canonical_action(action))
        if op is None or not op.applicable(current):
            executable = False
            break
        current = set(op.apply(current))
    valid = executable and set(context.task.goals).issubset(current)
    return bool(valid), bool(executable)


def validate_candidate_plan(
    *,
    domain_path: str | Path,
    problem_path: str | Path,
    plan: list[str],
    val_path: str | Path,
) -> tuple[bool, bool]:
    """Label a candidate plan using VAL."""
    if not plan:
        return False, False

    val_bin = Path(val_path)
    if not val_bin.exists() or not os.access(val_bin, os.X_OK):
        raise FileNotFoundError(f"VAL binary not found or not executable: {val_path}")

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".plan") as tmp:
        for action in plan:
            tmp.write(f"{canonical_action(action)}\n")
        tmp_plan_path = Path(tmp.name).resolve()

    try:
        result = subprocess.run(
            [
                str(val_bin),
                "-v",
                str(Path(domain_path).resolve()),
                str(Path(problem_path).resolve()),
                str(tmp_plan_path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        output = result.stdout or ""
        if result.returncode != 0 and not output.strip():
            raise RuntimeError(
                f"VAL failed without output for {problem_path}; "
                f"exit code {result.returncode}. Check VAL runtime dependencies."
            )
        is_valid = "Plan valid" in output
        is_executable = is_valid or "Plan executed successfully" in output
        return is_valid, is_executable
    finally:
        if tmp_plan_path.exists():
            tmp_plan_path.unlink()


def corruption_types() -> list[str]:
    """Supported negative-plan corruption operators."""
    return ["truncate", "delete", "swap", "replace", "insert", "repeat"]


def corrupt_plan(
    gold_plan: list[str],
    all_actions: list[str],
    rng: random.Random,
    corruption_type: str,
) -> list[str]:
    """Create one corrupted action sequence."""
    plan = list(gold_plan)
    if not plan:
        return plan

    if corruption_type == "truncate":
        cut = rng.randint(0, max(0, len(plan) - 1))
        return plan[:cut]

    if corruption_type == "delete":
        idx = rng.randrange(len(plan))
        return plan[:idx] + plan[idx + 1 :]

    if corruption_type == "swap":
        if len(plan) == 1:
            return corrupt_plan(plan, all_actions, rng, "replace")
        idx = rng.randrange(len(plan) - 1)
        plan[idx], plan[idx + 1] = plan[idx + 1], plan[idx]
        return plan

    if corruption_type == "replace":
        idx = rng.randrange(len(plan))
        choices = [action for action in all_actions if action != plan[idx]]
        if not choices:
            return plan
        plan[idx] = rng.choice(choices)
        return plan

    if corruption_type == "insert":
        idx = rng.randrange(len(plan) + 1)
        return plan[:idx] + [rng.choice(all_actions)] + plan[idx:]

    if corruption_type == "repeat":
        start = rng.randrange(len(plan))
        end = rng.randint(start + 1, min(len(plan), start + 3))
        insert_at = rng.randrange(len(plan) + 1)
        segment = plan[start:end]
        return plan[:insert_at] + segment + plan[insert_at:]

    raise ValueError(f"Unknown corruption type: {corruption_type}")


def make_candidate_id(
    domain: str,
    split: str,
    problem: str,
    variant_index: int,
    corruption_type: str,
) -> str:
    """Build a stable candidate identifier."""
    return f"{domain}::{split}::{problem}::{variant_index:03d}::{corruption_type}"


def generate_labeled_candidates_for_problem(
    *,
    context: ProblemContext,
    gold_plan: list[str],
    val_path: str | Path | None = None,
    negative_ratio: int,
    rng: random.Random,
    validator: Validator | None = None,
    max_attempts_per_negative: int = 25,
) -> list[dict]:
    """Create one positive and several VAL-confirmed negative candidates."""
    candidates: list[dict] = []
    seen_plans = {tuple(gold_plan)}

    label = validator or _make_val_validator(context, val_path)
    is_valid, is_executable = label(gold_plan)
    candidates.append(
        {
            "candidate_id": make_candidate_id(
                context.domain,
                context.split,
                context.problem,
                0,
                "gold",
            ),
            "domain": context.domain,
            "split": context.split,
            "problem": context.problem,
            "corruption_type": "gold",
            "plan": gold_plan,
            "plan_len": len(gold_plan),
            "gold_plan_len": len(gold_plan),
            "label_valid": int(is_valid),
            "label_executable": int(is_executable),
        }
    )

    action_space = [canonical_action(op.name) for op in context.operators]
    type_cycle = corruption_types()
    variant_index = 1
    attempts = 0
    max_attempts = max_attempts_per_negative * max(1, negative_ratio)

    while variant_index <= negative_ratio and attempts < max_attempts:
        attempts += 1
        corruption_type = type_cycle[(attempts - 1) % len(type_cycle)]
        candidate_plan = corrupt_plan(gold_plan, action_space, rng, corruption_type)
        plan_key = tuple(candidate_plan)
        if plan_key in seen_plans:
            continue
        seen_plans.add(plan_key)

        is_valid, is_executable = label(candidate_plan)
        if is_valid:
            continue

        candidates.append(
            {
                "candidate_id": make_candidate_id(
                    context.domain,
                    context.split,
                    context.problem,
                    variant_index,
                    corruption_type,
                ),
                "domain": context.domain,
                "split": context.split,
                "problem": context.problem,
                "corruption_type": corruption_type,
                "plan": candidate_plan,
                "plan_len": len(candidate_plan),
                "gold_plan_len": len(gold_plan),
                "label_valid": 0,
                "label_executable": int(is_executable),
            }
        )
        variant_index += 1

    return candidates


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    """Write records as JSON lines."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    """Read records from JSON lines."""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def discover_val_path(repo_root: str | Path, user_val_path: str | None = None) -> str | None:
    """Resolve the VAL binary used for plan labels."""
    if user_val_path:
        return user_val_path
    root = Path(repo_root)
    candidates = [
        root / "VAL" / "build" / "bin" / "Validate.exe",
        root / "VAL" / "build" / "bin" / "Validate",
        root / "VAL" / "bin" / "Validate.exe",
        root / "VAL" / "bin" / "Validate",
        root / "VAL" / "VAL" / "build" / "bin" / "Validate.exe",
    ]
    for candidate in candidates:
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def iter_problem_names(
    *,
    data_dir: str | Path,
    domain: str,
    split: str,
    max_problems: int | None = None,
) -> list[str]:
    """Return problem names with both trajectory and PDDL files available."""
    state_dir = Path(data_dir) / "states" / domain / split
    pddl_dir = Path(data_dir) / "pddl" / domain / split
    if not state_dir.exists() or not pddl_dir.exists():
        return []

    names = []
    for traj_path in sorted(state_dir.glob("*.traj")):
        if (pddl_dir / f"{traj_path.stem}.pddl").exists():
            names.append(traj_path.stem)
    if max_problems is not None:
        names = names[:max_problems]
    return names


def summarize_labels(rows: list[dict]) -> dict[str, int]:
    """Small summary for CLI status messages and metadata."""
    total = len(rows)
    valid = sum(int(row["label_valid"]) for row in rows)
    executable = sum(int(row["label_executable"]) for row in rows)
    return {
        "total": total,
        "valid": valid,
        "invalid": total - valid,
        "executable": executable,
        "non_executable": total - executable,
    }


Validator = Callable[[list[str]], tuple[bool, bool]]


def _make_val_validator(context: ProblemContext, val_path: str | Path | None) -> Validator:
    if val_path is None:
        raise ValueError("val_path is required when no custom validator is provided.")

    def validate(plan: list[str]) -> tuple[bool, bool]:
        return validate_candidate_plan(
            domain_path=context.domain_path,
            problem_path=context.problem_path,
            plan=plan,
            val_path=val_path,
        )

    return validate
