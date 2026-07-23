from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import multiprocessing as mp
import os
import re
import sys
import time
import traceback
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_AGENT_MODULE = "creations.ifeval_solver_pipeline.ifeval_solver_pipeline"
DEFAULT_AGENT_OBJECT = "ifeval_solver_pipeline_app"
DEFAULT_DATASET_NAME = "google/IFEval"
DEFAULT_DATASET_SPLIT = "train"


@dataclass(frozen=True)
class IFEvalTask:
    key: int
    prompt: str
    instruction_id_list: list[str]
    kwargs: list[dict[str, Any]]
    dataset_index: int


@dataclass
class GenerationRecord:
    key: int
    dataset_index: int
    status: str
    elapsed_seconds: float
    answer_present: bool
    answer_characters: int
    internal_review_loops: int | None = None
    internal_passed_latest_review: bool | None = None
    error: str | None = None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path: Path, item: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(dict(item), ensure_ascii=False) + "\n")
        file.flush()
        os.fsync(file.fileno())


def write_jsonl(path: Path, items: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for item in items:
            file.write(json.dumps(dict(item), ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(
                    f"Expected a JSON object at {path}:{line_number}."
                )
            rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")


def installed_version(distribution_name: str) -> str | None:
    try:
        return importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def load_library_dependencies() -> tuple[Any, Any]:
    """Import the dataset loader and lm-eval IFEval scoring function."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The 'datasets' package is missing. Install the IFEval evaluator "
            "library with:\n"
            f"{sys.executable} -m pip install \"lm_eval[ifeval]\""
        ) from exc

    try:
        from lm_eval.tasks.ifeval.utils import process_results
    except ImportError as exc:
        raise RuntimeError(
            "The lm-eval IFEval implementation or one of its task-specific "
            "dependencies is missing. Install it with:\n"
            f"{sys.executable} -m pip install \"lm_eval[ifeval]\"\n"
            f"{sys.executable} -m nltk.downloader punkt punkt_tab"
        ) from exc

    return load_dataset, process_results


def load_ifeval_tasks(
    *,
    dataset_name: str,
    dataset_split: str,
    cache_dir: Path | None,
) -> list[IFEvalTask]:
    load_dataset, _ = load_library_dependencies()

    print(
        f"Loading IFEval dataset '{dataset_name}' "
        f"(split='{dataset_split}') through Hugging Face datasets..."
    )
    dataset = load_dataset(
        dataset_name,
        split=dataset_split,
        cache_dir=str(cache_dir.resolve()) if cache_dir else None,
    )

    tasks: list[IFEvalTask] = []
    seen_keys: set[int] = set()
    seen_prompts: set[str] = set()

    for dataset_index, row in enumerate(dataset):
        try:
            key = int(row["key"])
            prompt = str(row["prompt"])
            instruction_id_list = [
                str(value) for value in row["instruction_id_list"]
            ]
            kwargs = [dict(value) for value in row["kwargs"]]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Malformed IFEval row at dataset index {dataset_index}."
            ) from exc

        if len(instruction_id_list) != len(kwargs):
            raise ValueError(
                f"IFEval key {key} has {len(instruction_id_list)} "
                f"instruction IDs but {len(kwargs)} kwargs entries."
            )
        if key in seen_keys:
            raise ValueError(f"Duplicate IFEval key: {key}")
        if prompt in seen_prompts:
            raise ValueError(f"Duplicate IFEval prompt for key {key}.")

        seen_keys.add(key)
        seen_prompts.add(prompt)
        tasks.append(
            IFEvalTask(
                key=key,
                prompt=prompt,
                instruction_id_list=instruction_id_list,
                kwargs=kwargs,
                dataset_index=dataset_index,
            )
        )

    return tasks


def choose_tasks(
    tasks: Sequence[IFEvalTask],
    *,
    start_index: int,
    limit: int | None,
    selected_keys: set[int],
) -> list[IFEvalTask]:
    if selected_keys:
        tasks_by_key = {task.key: task for task in tasks}
        missing = selected_keys.difference(tasks_by_key)
        if missing:
            raise ValueError(f"Unknown IFEval keys: {sorted(missing)}")
        return [task for task in tasks if task.key in selected_keys]

    selected = list(tasks[start_index:])
    if limit is not None:
        selected = selected[:limit]
    return selected


def load_agent(module_name: str, object_name: str) -> Any:
    module = importlib.import_module(module_name)
    try:
        agent = getattr(module, object_name)
    except AttributeError as exc:
        raise AttributeError(
            f"Module '{module_name}' has no object named '{object_name}'."
        ) from exc

    if not hasattr(agent, "invoke"):
        raise TypeError(
            f"'{module_name}.{object_name}' does not expose invoke(...)."
        )
    return agent


def invoke_agent(
    agent: Any,
    task: IFEvalTask,
    *,
    recursion_limit: int,
    run_id: str,
) -> Mapping[str, Any]:
    # The agent receives only the public prompt. The hidden evaluator metadata
    # is used only after answer generation.
    state = {
        "messages": [],
        "ifeval_prompt": task.prompt,
        "mode": "generate",
        "review_loops": 0,
        "answer": "",
        "review_feedback": None,
        "passed_latest_review": None,
    }

    safe_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(task.key))
    config = {
        "recursion_limit": recursion_limit,
        "configurable": {
            "user_id": "ifeval_benchmark",
            "run_name": "ifeval_benchmark",
            "thread_id": f"ifeval:{run_id}:{safe_key}:{uuid.uuid4().hex}",
        },
    }

    response = agent.invoke(state, config=config)
    if not isinstance(response, Mapping):
        raise TypeError(
            f"Agent returned {type(response).__name__}; expected a mapping."
        )
    return response


def invoke_agent_worker(
    result_queue: Any,
    module_name: str,
    object_name: str,
    task: IFEvalTask,
    recursion_limit: int,
    run_id: str,
) -> None:
    """Run one agent invocation in an isolated child process.

    Only small, serializable fields are returned to the parent process. The
    process can be terminated safely by the parent when the task exceeds its
    time limit.
    """
    try:
        agent = load_agent(module_name, object_name)
        response = invoke_agent(
            agent,
            task,
            recursion_limit=recursion_limit,
            run_id=run_id,
        )
        result_queue.put(
            {
                "kind": "success",
                "answer": normalize_answer(response.get("answer")),
                "review_loops": optional_int(
                    response.get("review_loops")
                ),
                "passed_latest_review": optional_bool(
                    response.get("passed_latest_review")
                ),
            }
        )
    except BaseException as exc:
        result_queue.put(
            {
                "kind": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )


def terminate_process(process: Any, *, grace_seconds: float = 5.0) -> None:
    """Terminate a child process and force-kill it if necessary."""
    if not process.is_alive():
        process.join(timeout=0.1)
        return

    process.terminate()
    process.join(timeout=grace_seconds)

    if process.is_alive():
        kill = getattr(process, "kill", None)
        if callable(kill):
            kill()
        process.join(timeout=grace_seconds)


def normalize_answer(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    return None


def count_values(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def evaluate_response(
    task: IFEvalTask,
    response: str,
    process_results: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    doc = {
        "key": task.key,
        "prompt": task.prompt,
        "instruction_id_list": task.instruction_id_list,
        "kwargs": task.kwargs,
    }
    metrics = process_results(doc, [response])

    strict_list = [
        bool(value) for value in metrics["inst_level_strict_acc"]
    ]
    loose_list = [
        bool(value) for value in metrics["inst_level_loose_acc"]
    ]

    strict_row = {
        "key": task.key,
        "prompt": task.prompt,
        "response": response,
        "instruction_id_list": task.instruction_id_list,
        "follow_all_instructions": bool(
            metrics["prompt_level_strict_acc"]
        ),
        "follow_instruction_list": strict_list,
    }
    loose_row = {
        "key": task.key,
        "prompt": task.prompt,
        "response": response,
        "instruction_id_list": task.instruction_id_list,
        "follow_all_instructions": bool(
            metrics["prompt_level_loose_acc"]
        ),
        "follow_instruction_list": loose_list,
    }
    return strict_row, loose_row


def compute_accuracy_report(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    prompt_total = len(rows)
    prompt_correct = 0
    instruction_total = 0
    instruction_correct = 0

    tier0_total: defaultdict[str, int] = defaultdict(int)
    tier0_correct: defaultdict[str, int] = defaultdict(int)
    tier1_total: defaultdict[str, int] = defaultdict(int)
    tier1_correct: defaultdict[str, int] = defaultdict(int)

    for row_index, row in enumerate(rows):
        instruction_ids = [
            str(value) for value in row.get("instruction_id_list", [])
        ]
        follow_list = [
            bool(value) for value in row.get("follow_instruction_list", [])
        ]
        if len(instruction_ids) != len(follow_list):
            raise ValueError(
                "Evaluator output has mismatched instruction and result "
                f"counts at row {row_index}."
            )

        followed_all = bool(row.get("follow_all_instructions", False))
        prompt_correct += int(followed_all)
        instruction_total += len(follow_list)
        instruction_correct += sum(follow_list)

        for instruction_id, followed in zip(instruction_ids, follow_list):
            tier0 = instruction_id.split(":", 1)[0]
            tier0_total[tier0] += 1
            tier1_total[instruction_id] += 1
            if followed:
                tier0_correct[tier0] += 1
                tier1_correct[instruction_id] += 1

    prompt_accuracy = prompt_correct / prompt_total if prompt_total else 0.0
    instruction_accuracy = (
        instruction_correct / instruction_total if instruction_total else 0.0
    )

    return {
        "prompt_total": prompt_total,
        "prompt_correct": prompt_correct,
        "prompt_level_accuracy": prompt_accuracy,
        "instruction_total": instruction_total,
        "instruction_correct": instruction_correct,
        "instruction_level_accuracy": instruction_accuracy,
        "average_of_prompt_and_instruction_accuracy": (
            prompt_accuracy + instruction_accuracy
        ) / 2,
        "tier0_accuracy": {
            key: tier0_correct[key] / total
            for key, total in sorted(tier0_total.items())
        },
        "tier1_accuracy": {
            key: tier1_correct[key] / total
            for key, total in sorted(tier1_total.items())
        },
    }


def build_score_summary(
    *,
    strict_rows: Sequence[Mapping[str, Any]],
    loose_rows: Sequence[Mapping[str, Any]],
    dataset_task_count: int,
    evaluated_task_count: int,
) -> dict[str, Any]:
    strict = compute_accuracy_report(strict_rows)
    loose = compute_accuracy_report(loose_rows)
    final_score = strict["prompt_level_accuracy"]

    return {
        "created_at": utc_now_iso(),
        "benchmark": "IFEval",
        "dataset_task_count": dataset_task_count,
        "evaluated_task_count": evaluated_task_count,
        "full_dataset_evaluation": evaluated_task_count == dataset_task_count,
        "official_metrics": {
            "strict": strict,
            "loose": loose,
        },
        "primary_metric": "strict_prompt_level_accuracy",
        "final_score": final_score,
        "final_score_percent": final_score * 100,
        "final_score_definition": (
            "Strict prompt-level accuracy: the fraction of evaluated prompts "
            "for which every verifiable instruction passed. The runner also "
            "retains strict/loose prompt-level and instruction-level scores."
        ),
        "evaluator": {
            "implementation": "EleutherAI lm-evaluation-harness IFEval",
            "lm_eval_version": installed_version("lm_eval"),
            "datasets_version": installed_version("datasets"),
            "repository_clone_used": False,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a custom LangGraph agent on IFEval and score its responses "
            "locally with the lm-eval IFEval library implementation. No Git "
            "repository is cloned."
        )
    )
    parser.add_argument(
        "--agent-module",
        default=DEFAULT_AGENT_MODULE,
        help="Python module containing the compiled LangGraph app.",
    )
    parser.add_argument(
        "--agent-object",
        default=DEFAULT_AGENT_OBJECT,
        help="Object exposing invoke(state, config=...).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Result directory. The default is "
            "benchmark_results/ifeval/ifeval_<UTC timestamp>."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip prompts already present in responses.jsonl.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Zero-based first dataset index for a partial run.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum tasks to run. Omit for the complete dataset.",
    )
    parser.add_argument(
        "--task-key",
        action="append",
        type=int,
        default=[],
        help="Run one IFEval key. Repeat for multiple keys.",
    )
    parser.add_argument(
        "--recursion-limit",
        type=int,
        default=100,
        help="LangGraph recursion limit for each task invocation.",
    )
    parser.add_argument(
        "--max-parallel-tasks",
        type=int,
        default=2,
        help=(
            "Maximum number of IFEval prompts processed simultaneously. "
            "The default is 2."
        ),
    )
    parser.add_argument(
        "--task-timeout-seconds",
        type=float,
        default=300.0,
        help=(
            "Hard time limit for one agent invocation. Timed-out tasks are "
            "terminated and left missing so --resume retries them. "
            "The default is 300 seconds."
        ),
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop generation after the first agent exception.",
    )
    parser.add_argument(
        "--evaluation-mode",
        choices=("library", "none"),
        default="library",
        help="Score with lm-eval locally or only generate responses.",
    )
    parser.add_argument(
        "--dataset-name",
        default=DEFAULT_DATASET_NAME,
        help="Hugging Face dataset name.",
    )
    parser.add_argument(
        "--dataset-split",
        default=DEFAULT_DATASET_SPLIT,
        help="Dataset split to load.",
    )
    parser.add_argument(
        "--dataset-cache-dir",
        type=Path,
        default=None,
        help=(
            "Optional Hugging Face dataset cache directory. This stores only "
            "dataset files; no evaluator repository is cloned."
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.start_index < 0:
        raise ValueError("--start-index must be at least 0.")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be greater than 0.")
    if args.recursion_limit <= 0:
        raise ValueError("--recursion-limit must be greater than 0.")
    if args.max_parallel_tasks <= 0:
        raise ValueError("--max-parallel-tasks must be greater than 0.")
    if args.task_timeout_seconds <= 0:
        raise ValueError("--task-timeout-seconds must be greater than 0.")

    print("Evaluator backend: lm-eval library (no Git clone).")

    # Import and validate the evaluator before making any model calls.
    _, process_results = load_library_dependencies()

    all_tasks = load_ifeval_tasks(
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        cache_dir=args.dataset_cache_dir,
    )
    dataset_task_count = len(all_tasks)

    selected_tasks = choose_tasks(
        all_tasks,
        start_index=args.start_index,
        limit=args.limit,
        selected_keys=set(args.task_key),
    )
    if not selected_tasks:
        raise ValueError("No IFEval tasks were selected.")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or Path(
        "benchmark_results",
        "ifeval",
        f"ifeval_{timestamp}",
    )
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    responses_path = output_dir / "responses.jsonl"
    generation_log_path = output_dir / "generation_log.jsonl"
    selected_input_path = output_dir / "selected_input_data.jsonl"
    evaluation_input_path = output_dir / "evaluation_input_data.jsonl"
    strict_path = output_dir / "eval_results_strict.jsonl"
    loose_path = output_dir / "eval_results_loose.jsonl"
    generation_summary_path = output_dir / "generation_summary.json"
    score_summary_path = output_dir / "score_summary.json"
    run_config_path = output_dir / "run_config.json"

    if (
        not args.resume
        and responses_path.exists()
        and responses_path.stat().st_size > 0
    ):
        raise FileExistsError(
            f"{responses_path} already exists. Use --resume or choose "
            "another --output-dir."
        )

    write_jsonl(
        selected_input_path,
        (
            {
                "key": task.key,
                "prompt": task.prompt,
                "instruction_id_list": task.instruction_id_list,
                "kwargs": task.kwargs,
            }
            for task in selected_tasks
        ),
    )

    run_id = uuid.uuid4().hex
    run_config = {
        "created_at": utc_now_iso(),
        "run_id": run_id,
        "benchmark": "IFEval",
        "dataset_name": args.dataset_name,
        "dataset_split": args.dataset_split,
        "dataset_task_count": dataset_task_count,
        "selected_task_count": len(selected_tasks),
        "full_dataset_selection": len(selected_tasks) == dataset_task_count,
        "agent_module": args.agent_module,
        "agent_object": args.agent_object,
        "recursion_limit": args.recursion_limit,
        "max_parallel_tasks": args.max_parallel_tasks,
        "task_timeout_seconds": args.task_timeout_seconds,
        "start_index": args.start_index,
        "limit": args.limit,
        "selected_task_keys": args.task_key,
        "evaluation_mode": args.evaluation_mode,
        "evaluator_backend": "lm_eval.tasks.ifeval.utils.process_results",
        "evaluator_repository_downloaded": False,
        "methodology_note": (
            "The agent receives only the published prompt text. The "
            "instruction_id_list and kwargs remain hidden and are used only "
            "after generation by the local rule-based evaluator. The agent's "
            "internal passed_latest_review value is logged but is never used "
            "as the benchmark result."
        ),
    }
    write_json(run_config_path, run_config)

    existing_responses = read_jsonl(responses_path) if args.resume else []

    # Only non-empty answers count as completed. This deliberately ignores
    # empty rows left by older runner versions, so --resume retries them.
    response_by_prompt = {
        str(row.get("prompt")): str(row.get("response") or "")
        for row in existing_responses
        if row.get("prompt") is not None
        and str(row.get("response") or "").strip()
    }
    pending_tasks = [
        task for task in selected_tasks if task.prompt not in response_by_prompt
    ]

    print(
        f"Selected {len(selected_tasks)} task(s); "
        f"{len(pending_tasks)} remain after resume."
    )
    print(
        f"Parallel tasks: {args.max_parallel_tasks}; "
        f"hard timeout: {args.task_timeout_seconds:g}s per task."
    )

    started = time.perf_counter()
    generation_records: list[GenerationRecord] = []
    agent_errors = 0
    task_timeouts = 0
    empty_answers = 0

    # 'spawn' works consistently on Windows and keeps each invocation isolated.
    # A fresh process per task also lets us enforce a true hard timeout.
    process_context = mp.get_context("spawn")
    pending_iterator = iter(enumerate(pending_tasks, start=1))
    active: dict[int, dict[str, Any]] = {}
    no_more_tasks = False
    stop_requested = False

    def start_task(position: int, task: IFEvalTask) -> None:
        result_queue = process_context.Queue(maxsize=1)
        process = process_context.Process(
            target=invoke_agent_worker,
            args=(
                result_queue,
                args.agent_module,
                args.agent_object,
                task,
                args.recursion_limit,
                run_id,
            ),
            name=f"ifeval-{task.key}",
        )
        process.start()
        active[task.key] = {
            "position": position,
            "task": task,
            "process": process,
            "queue": result_queue,
            "started_at": time.perf_counter(),
        }
        print(
            f"[{position}/{len(pending_tasks)}] started key={task.key} "
            f"(dataset index {task.dataset_index}, pid={process.pid})"
        )

    def save_task_result(
        task: IFEvalTask,
        *,
        status: str,
        elapsed_seconds: float,
        answer: str = "",
        review_loops: int | None = None,
        passed_latest_review: bool | None = None,
        error: str | None = None,
    ) -> None:
        nonlocal agent_errors, task_timeouts, empty_answers

        answer = normalize_answer(answer)
        answer_present = bool(answer.strip())

        if status == "ok" and answer_present:
            # Successes alone enter responses.jsonl. A timeout, exception, or
            # empty answer remains missing and is therefore retried by --resume.
            append_jsonl(
                responses_path,
                {
                    "prompt": task.prompt,
                    "response": answer,
                },
            )
            response_by_prompt[task.prompt] = answer
        elif status == "timeout":
            task_timeouts += 1
        elif status == "empty_answer":
            empty_answers += 1
        elif status == "agent_error":
            agent_errors += 1

        record = GenerationRecord(
            key=task.key,
            dataset_index=task.dataset_index,
            status=status,
            elapsed_seconds=round(elapsed_seconds, 3),
            answer_present=answer_present,
            answer_characters=len(answer),
            internal_review_loops=review_loops,
            internal_passed_latest_review=passed_latest_review,
            error=error,
        )
        generation_records.append(record)
        append_jsonl(generation_log_path, asdict(record))

        print(
            f"    finished key={task.key}, status={status}, "
            f"chars={len(answer)}, elapsed={record.elapsed_seconds:.3f}s"
        )

    while active or not no_more_tasks:
        while (
            not stop_requested
            and not no_more_tasks
            and len(active) < args.max_parallel_tasks
        ):
            try:
                position, task = next(pending_iterator)
            except StopIteration:
                no_more_tasks = True
                break
            start_task(position, task)

        if not active:
            break

        made_progress = False
        now = time.perf_counter()

        for task_key, invocation in list(active.items()):
            task = invocation["task"]
            process = invocation["process"]
            result_queue = invocation["queue"]
            elapsed = now - invocation["started_at"]

            if process.is_alive() and elapsed < args.task_timeout_seconds:
                continue

            made_progress = True

            if process.is_alive():
                terminate_process(process)
                save_task_result(
                    task,
                    status="timeout",
                    elapsed_seconds=elapsed,
                    error=(
                        "TaskTimeoutError: agent invocation exceeded "
                        f"{args.task_timeout_seconds:g} seconds"
                    ),
                )
                failure_occurred = True
            else:
                process.join(timeout=0.1)
                try:
                    result = result_queue.get(timeout=1.0)
                except Empty:
                    result = {
                        "kind": "error",
                        "error": (
                            "WorkerProcessError: child process exited "
                            f"with code {process.exitcode} without returning a result"
                        ),
                    }

                if result.get("kind") == "success":
                    answer = normalize_answer(result.get("answer"))
                    status = "ok" if answer.strip() else "empty_answer"
                    save_task_result(
                        task,
                        status=status,
                        elapsed_seconds=elapsed,
                        answer=answer,
                        review_loops=optional_int(result.get("review_loops")),
                        passed_latest_review=optional_bool(
                            result.get("passed_latest_review")
                        ),
                    )
                    failure_occurred = status != "ok"
                else:
                    error = str(
                        result.get("error")
                        or "WorkerProcessError: unknown worker failure"
                    )
                    worker_traceback = result.get("traceback")
                    if worker_traceback:
                        print(worker_traceback, file=sys.stderr)
                    save_task_result(
                        task,
                        status="agent_error",
                        elapsed_seconds=elapsed,
                        error=error,
                    )
                    failure_occurred = True

            result_queue.close()
            result_queue.join_thread()
            del active[task_key]

            if failure_occurred and args.stop_on_error:
                stop_requested = True

        if stop_requested and active:
            for task_key, invocation in list(active.items()):
                task = invocation["task"]
                process = invocation["process"]
                result_queue = invocation["queue"]
                elapsed = time.perf_counter() - invocation["started_at"]
                terminate_process(process)
                save_task_result(
                    task,
                    status="agent_error",
                    elapsed_seconds=elapsed,
                    error=(
                        "CancelledError: cancelled because --stop-on-error "
                        "was triggered by another task"
                    ),
                )
                result_queue.close()
                result_queue.join_thread()
                del active[task_key]
            break

        if not made_progress:
            time.sleep(0.1)

    all_response_rows = read_jsonl(responses_path)
    all_log_rows = read_jsonl(generation_log_path)
    response_by_prompt = {
        str(row.get("prompt")): str(row.get("response") or "")
        for row in all_response_rows
        if row.get("prompt") is not None
        and str(row.get("response") or "").strip()
    }
    selected_completed_tasks = [
        task for task in selected_tasks if task.prompt in response_by_prompt
    ]

    generation_summary = {
        "finished_at": utc_now_iso(),
        "dataset_task_count": dataset_task_count,
        "selected_tasks": len(selected_tasks),
        "previously_completed_selected_tasks": (
            len(selected_tasks) - len(pending_tasks)
        ),
        "generated_this_run": len(generation_records),
        "completed_selected_tasks": len(selected_completed_tasks),
        "response_rows_in_file": len(all_response_rows),
        "usable_nonempty_responses": len(response_by_prompt),
        "full_dataset_complete": all(
            task.prompt in response_by_prompt for task in all_tasks
        ),
        "agent_errors_this_run": agent_errors,
        "timeouts_this_run": task_timeouts,
        "empty_answers_this_run": empty_answers,
        "retryable_missing_tasks_this_run": sum(
            1 for record in generation_records if record.status != "ok"
        ),
        "elapsed_seconds_this_run": round(time.perf_counter() - started, 3),
        "status_counts_all_logs": count_values(
            str(row.get("status", "unknown")) for row in all_log_rows
        ),
        "responses_path": str(responses_path),
        "evaluation_requested": args.evaluation_mode == "library",
        "evaluation_performed": False,
    }

    print(f"\nGeneration complete. Responses: {responses_path}")

    if args.evaluation_mode == "none":
        write_json(generation_summary_path, generation_summary)
        print(f"Generation summary: {generation_summary_path}")
        return 0

    if not selected_completed_tasks:
        generation_summary["evaluation_skipped_reason"] = (
            "No selected task has a saved response."
        )
        write_json(generation_summary_path, generation_summary)
        print("Evaluation skipped because no responses are available.")
        return 0

    write_jsonl(
        evaluation_input_path,
        (
            {
                "key": task.key,
                "prompt": task.prompt,
                "instruction_id_list": task.instruction_id_list,
                "kwargs": task.kwargs,
            }
            for task in selected_completed_tasks
        ),
    )

    strict_rows: list[dict[str, Any]] = []
    loose_rows: list[dict[str, Any]] = []
    for task in selected_completed_tasks:
        strict_row, loose_row = evaluate_response(
            task,
            response_by_prompt[task.prompt],
            process_results,
        )
        strict_rows.append(strict_row)
        loose_rows.append(loose_row)

    write_jsonl(strict_path, strict_rows)
    write_jsonl(loose_path, loose_rows)

    score_summary = build_score_summary(
        strict_rows=strict_rows,
        loose_rows=loose_rows,
        dataset_task_count=dataset_task_count,
        evaluated_task_count=len(selected_completed_tasks),
    )
    write_json(score_summary_path, score_summary)

    generation_summary["evaluation_performed"] = True
    generation_summary["evaluated_task_count"] = len(
        selected_completed_tasks
    )
    generation_summary["score_summary_path"] = str(score_summary_path)
    generation_summary["final_score"] = score_summary["final_score"]
    generation_summary["final_score_percent"] = score_summary[
        "final_score_percent"
    ]
    write_json(generation_summary_path, generation_summary)

    strict = score_summary["official_metrics"]["strict"]
    loose = score_summary["official_metrics"]["loose"]

    print("\nIFEval scores")
    print(
        "  Strict prompt-level:       "
        f"{strict['prompt_level_accuracy']:.4f}"
    )
    print(
        "  Strict instruction-level:  "
        f"{strict['instruction_level_accuracy']:.4f}"
    )
    print(
        "  Loose prompt-level:        "
        f"{loose['prompt_level_accuracy']:.4f}"
    )
    print(
        "  Loose instruction-level:   "
        f"{loose['instruction_level_accuracy']:.4f}"
    )
    print(
        "  Final score:                "
        f"{score_summary['final_score']:.4f} (strict prompt-level)"
    )
    print(f"\nScore summary: {score_summary_path}")
    print(f"Generation summary: {generation_summary_path}")

    if len(selected_completed_tasks) != dataset_task_count:
        print(
            "\nNote: this is a partial-run score. Run the complete dataset "
            "for the full IFEval benchmark result."
        )

    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())

    # Limit
    # python .\run_ifeval_benchmark.py `
    # --limit 3 `
    # --output-dir .\benchmark_results\ifeval\ifeval_test

    # Full
    # python .\run_ifeval_benchmark.py `
    # --output-dir .\benchmark_results\ifeval\ifeval_full `
    # --max-parallel-tasks 2 `
    # --task-timeout-seconds 300

    # Resume
    # python .\run_ifeval_benchmark.py `
    # --output-dir .\benchmark_results\ifeval\ifeval_full `
    # --resume

    # Specific
    # python .\run_ifeval_benchmark.py `
    # --task-key 1000 `
    # --task-key 1001 `
    # --output-dir .\benchmark_results\ifeval\specific_task