from __future__ import annotations

import argparse
import importlib
import json
import multiprocessing
import os
import queue
import re
import shutil
import string
import subprocess
import sys
import time
import traceback
import uuid
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import sleep
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_AGENT_MODULE = "creations.gaia_task_solver.gaia_task_solver"
DEFAULT_AGENT_OBJECT = "gaia_task_solver_app"
DEFAULT_DATASET_REPO = "gaia-benchmark/GAIA"
DEFAULT_DATASET_CONFIG = "2023_all"
DEFAULT_DOCKER_IMAGE = "gaia-python:3.11"

# Target agent:
# creations/gaia_task_solver/gaia_task_solver.py
# object: gaia_task_solver_app


@dataclass(frozen=True)
class GaiaTask:
    task_id: str
    task_index: int
    question: str
    level: int
    attachment_path: str | None
    ground_truth: str | None


@dataclass
class GenerationRecord:
    task_id: str
    task_index: int
    level: int
    status: str
    elapsed_seconds: float
    attachment_present: bool
    final_output_present: bool
    agent_status: str | None = None
    failure_reason: str | None = None
    total_tool_calls: int | None = None
    replans: int | None = None
    repeated_searches: int | None = None
    duplicate_actions: int | None = None
    candidate_answer_present: bool | None = None
    review_decision: str | None = None
    review_feedback: str | None = None
    score: bool | None = None
    error: str | None = None


@dataclass
class ActiveProcess:
    task: GaiaTask
    process: multiprocessing.Process
    result_queue: Any
    started_at: float


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path: Path, item: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(dict(item), ensure_ascii=False, default=json_fallback) + "\n")
        file.flush()
        os.fsync(file.fileno())


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
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc

            if not isinstance(value, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}.")

            rows.append(value)

    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2, default=json_fallback)
        file.write("\n")


def json_fallback(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)

    if isinstance(value, tuple):
        return list(value)

    if isinstance(value, set):
        return sorted(value)

    item_method = getattr(value, "item", None)

    if callable(item_method):
        return item_method()

    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _count_values(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}

    for value in values:
        counts[value] = counts.get(value, 0) + 1

    return counts


def load_agent(module_name: str, object_name: str) -> Any:
    module = importlib.import_module(module_name)

    try:
        agent = getattr(module, object_name)
    except AttributeError as exc:
        raise AttributeError(f"Module '{module_name}' has no object named '{object_name}'.") from exc

    if not hasattr(agent, "invoke"):
        raise TypeError(f"'{module_name}.{object_name}' does not expose invoke(...).")

    return agent


def normalize_number_str(number_str: str) -> float:
    """
    Match the official GAIA numeric normalization.

    Dollar signs, percent signs, and commas are removed before float parsing.
    """
    for character in ("$", "%", ","):
        number_str = number_str.replace(character, "")

    try:
        return float(number_str)
    except ValueError:
        return float("inf")


def split_string(
    value: str,
    character_list: Sequence[str] = (",", ";"),
) -> list[str]:
    pattern = f"[{''.join(character_list)}]"
    return re.split(pattern, value)


def is_float(value: Any) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def normalize_str(value: Any, remove_punct: bool = True) -> str:
    """
    Match the official GAIA string normalization.

    All whitespace is removed, text is lowercased, and punctuation is
    optionally removed.
    """
    no_spaces = re.sub(r"\s", "", str(value))

    if not remove_punct:
        return no_spaces.lower()

    translator = str.maketrans("", "", string.punctuation)
    return no_spaces.lower().translate(translator)


def question_scorer(model_answer: str, ground_truth: str) -> bool:
    """
    Score one answer using the official GAIA quasi-exact-match rules.
    """
    model_answer = str(model_answer)
    ground_truth = str(ground_truth)

    if is_float(ground_truth):
        normalized_answer = normalize_number_str(model_answer)
        return normalized_answer == float(ground_truth)

    if any(character in ground_truth for character in (",", ";")):
        ground_truth_elements = split_string(ground_truth)
        model_answer_elements = split_string(model_answer)

        if len(ground_truth_elements) != len(model_answer_elements):
            warnings.warn("Answer lists have different lengths; returning False.", UserWarning,)
            return False

        comparisons: list[bool] = []

        for model_element, truth_element in zip(
            model_answer_elements,
            ground_truth_elements,
        ):
            if is_float(truth_element):
                comparisons.append(normalize_number_str(model_element) == float(truth_element))
            else:
                comparisons.append(
                    normalize_str(
                        model_element,
                        remove_punct=False,
                    ) == normalize_str(
                        truth_element,
                        remove_punct=False,
                    )
                )

        return all(comparisons)

    return normalize_str(model_answer) == normalize_str(ground_truth)


def resolve_attachment_path(
    dataset_root: Path,
    record: Mapping[str, Any],
) -> str | None:
    """
    Resolve the attachment path from the current GAIA dataset format.

    The October 2025 format exposes file_path relative to the repository root.
    file_name is retained as a conservative fallback for older cached copies.
    """
    raw_path = record.get("file_path")

    if isinstance(raw_path, str) and raw_path.strip():
        candidate = Path(raw_path)

        if not candidate.is_absolute():
            candidate = dataset_root / candidate

        candidate = candidate.resolve()

        if candidate.exists() and candidate.is_file():
            return str(candidate)

        raise FileNotFoundError(
            f"Attachment declared by task {record.get('task_id')} "
            f"does not exist: {candidate}"
        )

    file_name = record.get("file_name")

    if not isinstance(file_name, str) or not file_name.strip():
        return None

    matches = list(dataset_root.rglob(file_name))

    if len(matches) == 1 and matches[0].is_file():
        return str(matches[0].resolve())

    if len(matches) > 1:
        raise ValueError(
            f"Attachment name '{file_name}' is ambiguous for task "
            f"{record.get('task_id')}."
        )

    raise FileNotFoundError(
        f"Could not locate attachment '{file_name}' for task "
        f"{record.get('task_id')}."
    )


def load_gaia_dataset(
    *,
    repo_id: str,
    config_name: str,
    split: str,
    token: str | None,
) -> tuple[Path, list[dict[str, Any]]]:
    """
    Download the gated GAIA repository and load one split.

    Access must already have been accepted on Hugging Face. A token may be
    provided through HF_TOKEN or HUGGINGFACEHUB_API_TOKEN.
    """
    try:
        from datasets import load_dataset
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "GAIA loading requires 'datasets' and 'huggingface_hub'. "
            "Install them with: pip install -U datasets huggingface_hub"
        ) from exc

    try:
        snapshot_path = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
        )
    except Exception as exc:
        raise RuntimeError(
            "Could not download the gated GAIA dataset. Confirm that you "
            "accepted the dataset conditions and that HF_TOKEN or "
            "HUGGINGFACEHUB_API_TOKEN is available."
        ) from exc

    dataset_root = Path(snapshot_path).resolve()

    try:
        dataset = load_dataset(str(dataset_root), config_name, split=split)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load GAIA config '{config_name}' split '{split}' "
            f"from {dataset_root}."
        ) from exc

    records = [dict(record) for record in dataset]

    return dataset_root, records


def parse_gaia_tasks(
    records: Sequence[Mapping[str, Any]],
    *,
    dataset_root: Path,
    split: str,
) -> list[GaiaTask]:
    tasks: list[GaiaTask] = []

    for index, record in enumerate(records):
        task_id = str(record.get("task_id") or "").strip()
        question = str(record.get("Question") or "").strip()

        if not task_id:
            raise ValueError(f"GAIA row {index} has no task_id.")

        if not question:
            raise ValueError(f"GAIA task {task_id} has no question.")

        try:
            level = int(record.get("Level"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"GAIA task {task_id} has an invalid Level.") from exc

        attachment_path = resolve_attachment_path(dataset_root, record)

        raw_ground_truth = record.get("Final answer")

        if split == "validation":
            if raw_ground_truth is None or str(raw_ground_truth).strip() == "":
                raise ValueError(f"Validation task {task_id} has no Final answer.")

            ground_truth: str | None = str(raw_ground_truth)
        else:
            ground_truth = None

        tasks.append(
            GaiaTask(
                task_id=task_id,
                task_index=index,
                question=question,
                level=level,
                attachment_path=attachment_path,
                ground_truth=ground_truth,
            )
        )

    return tasks


def choose_tasks(
    tasks: Sequence[GaiaTask],
    *,
    level: str,
    start_index: int,
    limit: int | None,
    selected_task_ids: set[str],
) -> list[GaiaTask]:
    selected = list(tasks)

    if level != "all":
        selected_level = int(level)
        selected = [
            task
            for task in selected
            if task.level == selected_level
        ]

    if selected_task_ids:
        available_ids = {
            task.task_id
            for task in selected
        }
        missing_ids = selected_task_ids.difference(available_ids)

        if missing_ids:
            raise ValueError(
                f"Unknown task IDs for the selected split/level: "
                f"{sorted(missing_ids)}"
            )

        selected = [
            task
            for task in selected
            if task.task_id in selected_task_ids
        ]
    else:
        selected = selected[start_index:]

        if limit is not None:
            selected = selected[:limit]

    return selected


def build_agent_state(task: GaiaTask) -> dict[str, Any]:
    """
    Build the initial state expected by gaia_task_solver_app.

    The benchmark passes only the question and optional attachment path.
    Reference answers and scorer results are never included.
    """
    return {
        "messages": [],
        "question": task.question,
        "attachments": (
            [task.attachment_path]
            if task.attachment_path
            else []
        ),
        "task_analysis": "",
        "candidate_answer": "",
        "answer_format": "",
        "steps_completed": [],
        "evidence": [],
        "calculations": [],
        "unresolved_issues": [],
        "review_decision": "",
        "review_feedback": None,
        "review_count": 0,
        "final_output": "",
    }


def invoke_agent(
    *,
    module_name: str,
    object_name: str,
    task: GaiaTask,
    recursion_limit: int,
    run_id: str,
) -> Mapping[str, Any]:
    agent = load_agent(
        module_name,
        object_name,
    )

    safe_task_id = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        task.task_id,
    )

    config = {
        "recursion_limit": recursion_limit,
        "configurable": {
            "user_id": "gaia_benchmark",
            "run_name": "gaia_benchmark",
            "thread_id": (
                f"gaia:{run_id}:{safe_task_id}:{uuid.uuid4().hex}"
            ),
        },
    }

    response = agent.invoke(
        build_agent_state(task),
        config=config,
    )

    if not isinstance(response, Mapping):
        raise TypeError(
            f"Agent returned {type(response).__name__}; expected a mapping."
        )

    return response


def build_worker_result(
    response: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Convert the current gaia_task_solver_app response into data that can
    safely cross the multiprocessing queue.
    """
    final_output = response.get("final_output")
    candidate_answer = response.get("candidate_answer")
    messages = response.get("messages")

    if not isinstance(messages, Sequence) or isinstance(
        messages,
        (str, bytes),
    ):
        messages = []

    total_tool_calls = 0

    for message in messages:
        tool_calls = getattr(message, "tool_calls", None)

        if not tool_calls:
            additional_kwargs = getattr(
                message,
                "additional_kwargs",
                {},
            )

            if isinstance(additional_kwargs, Mapping):
                tool_calls = additional_kwargs.get(
                    "tool_calls",
                    [],
                )

        if isinstance(tool_calls, Sequence) and not isinstance(
            tool_calls,
            (str, bytes),
        ):
            total_tool_calls += len(tool_calls)

    final_output_text = (
        str(final_output).strip()
        if final_output is not None
        else ""
    )

    review_decision = response.get("review_decision")
    review_feedback = response.get("review_feedback")

    return {
        "ok": True,
        "final_output": final_output_text,
        "agent_status": (
            "completed"
            if final_output_text
            else "failed"
        ),
        "failure_reason": (
            str(review_feedback)
            if review_feedback and not final_output_text
            else None
        ),
        "format_check_summary": (
            f"review_decision={review_decision}; "
            f"review_count={response.get('review_count', 0)}"
        ),
        "total_tool_calls": total_tool_calls,
        "replans": _optional_int(
            response.get("review_count")
        ),
        "repeated_searches": None,
        "duplicate_actions": None,
        "candidate_answer_present": bool(
            str(candidate_answer or "").strip()
        ),
        "review_decision": (
            str(review_decision)
            if review_decision is not None
            else None
        ),
        "review_feedback": (
            str(review_feedback)
            if review_feedback
            else None
        ),
    }


def agent_worker(
    payload: Mapping[str, Any],
    result_queue: Any,
) -> None:
    """
    Run one GAIA task in its own process.

    A separate process allows the parent runner to terminate invocations that
    exceed the per-task timeout.
    """
    try:
        task = GaiaTask(
            task_id=str(payload["task_id"]),
            task_index=int(payload["task_index"]),
            question=str(payload["question"]),
            level=int(payload["level"]),
            attachment_path=(
                str(payload["attachment_path"])
                if payload.get("attachment_path")
                else None
            ),
            ground_truth=None,
        )

        response = invoke_agent(
            module_name=str(payload["agent_module"]),
            object_name=str(payload["agent_object"]),
            task=task,
            recursion_limit=int(payload["recursion_limit"]),
            run_id=str(payload["run_id"]),
        )

        result_queue.put(
            build_worker_result(response)
        )

    except BaseException as exc:
        result_queue.put(
            {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )


def terminate_process(process: multiprocessing.Process) -> None:
    if not process.is_alive():
        process.join(timeout=0.5)
        return

    process.terminate()
    process.join(timeout=5)

    if process.is_alive():
        kill_method = getattr(process, "kill", None)

        if callable(kill_method):
            kill_method()
            process.join(timeout=2)


def start_task_process(
    *,
    context: Any,
    task: GaiaTask,
    agent_module: str,
    agent_object: str,
    recursion_limit: int,
    run_id: str,
) -> ActiveProcess:
    result_queue = context.Queue(maxsize=1)

    payload = {
        "task_id": task.task_id,
        "task_index": task.task_index,
        "question": task.question,
        "level": task.level,
        "attachment_path": task.attachment_path,
        "agent_module": agent_module,
        "agent_object": agent_object,
        "recursion_limit": recursion_limit,
        "run_id": run_id,
    }

    process = context.Process(
        target=agent_worker,
        args=(payload, result_queue),
        name=f"gaia-{task.task_index}",
    )

    process.start()

    return ActiveProcess(
        task=task,
        process=process,
        result_queue=result_queue,
        started_at=time.perf_counter(),
    )


def get_worker_result(active: ActiveProcess) -> dict[str, Any] | None:
    try:
        result = active.result_queue.get_nowait()
    except queue.Empty:
        return None

    if not isinstance(result, dict):
        return {
            "ok": False,
            "error": (
                f"Worker returned {type(result).__name__}; expected a dict."
            ),
        }

    return result


def close_worker(active: ActiveProcess) -> None:
    active.process.join(timeout=1)

    try:
        active.result_queue.close()
        active.result_queue.join_thread()
    except Exception:
        pass


def score_prediction(
    task: GaiaTask,
    prediction: str,
) -> bool | None:
    if task.ground_truth is None:
        return None

    return question_scorer(
        prediction,
        task.ground_truth,
    )


def process_completed_task(
    *,
    active: ActiveProcess,
    worker_result: Mapping[str, Any],
    answers_path: Path,
    log_path: Path,
    scored_path: Path,
) -> GenerationRecord:
    elapsed = round(
        time.perf_counter() - active.started_at,
        3,
    )

    if not worker_result.get("ok"):
        record = GenerationRecord(
            task_id=active.task.task_id,
            task_index=active.task.task_index,
            level=active.task.level,
            status="agent_error",
            elapsed_seconds=elapsed,
            attachment_present=bool(active.task.attachment_path),
            final_output_present=False,
            error=str(
                worker_result.get("error")
                or "Unknown worker error"
            ),
        )

        append_jsonl(
            log_path,
            asdict(record),
        )

        return record

    final_output = str(
        worker_result.get("final_output") or ""
    ).strip()

    agent_status = (
        str(worker_result.get("agent_status"))
        if worker_result.get("agent_status") is not None
        else None
    )
    failure_reason = (
        str(worker_result.get("failure_reason"))
        if worker_result.get("failure_reason")
        else None
    )

    if not final_output:
        status = (
            "agent_failed"
            if agent_status == "failed"
            else "missing_output"
        )

        record = GenerationRecord(
            task_id=active.task.task_id,
            task_index=active.task.task_index,
            level=active.task.level,
            status=status,
            elapsed_seconds=elapsed,
            attachment_present=bool(active.task.attachment_path),
            final_output_present=False,
            agent_status=agent_status,
            failure_reason=failure_reason,
            total_tool_calls=_optional_int(
                worker_result.get("total_tool_calls")
            ),
            replans=_optional_int(
                worker_result.get("replans")
            ),
            repeated_searches=_optional_int(
                worker_result.get("repeated_searches")
            ),
            duplicate_actions=_optional_int(
                worker_result.get("duplicate_actions")
            ),
            candidate_answer_present=bool(
                worker_result.get("candidate_answer_present")
            ),
            review_decision=(
                str(worker_result.get("review_decision"))
                if worker_result.get("review_decision") is not None
                else None
            ),
            review_feedback=(
                str(worker_result.get("review_feedback"))
                if worker_result.get("review_feedback")
                else None
            ),
            error=(
                failure_reason
                or "The agent returned no final_output."
            ),
        )

        append_jsonl(
            log_path,
            asdict(record),
        )

        return record

    score = score_prediction(
        active.task,
        final_output,
    )

    append_jsonl(
        answers_path,
        {
            "task_id": active.task.task_id,
            "model_answer": final_output,
        },
    )

    if score is not None:
        append_jsonl(
            scored_path,
            {
                "task_id": active.task.task_id,
                "model_answer": final_output,
                "score": score,
                "level": active.task.level,
            },
        )

    record = GenerationRecord(
        task_id=active.task.task_id,
        task_index=active.task.task_index,
        level=active.task.level,
        status="ok",
        elapsed_seconds=elapsed,
        attachment_present=bool(active.task.attachment_path),
        final_output_present=True,
        agent_status=agent_status,
        failure_reason=failure_reason,
        total_tool_calls=_optional_int(
            worker_result.get("total_tool_calls")
        ),
        replans=_optional_int(
            worker_result.get("replans")
        ),
        repeated_searches=_optional_int(
            worker_result.get("repeated_searches")
        ),
        duplicate_actions=_optional_int(
            worker_result.get("duplicate_actions")
        ),
        candidate_answer_present=bool(
            worker_result.get("candidate_answer_present")
        ),
        review_decision=(
            str(worker_result.get("review_decision"))
            if worker_result.get("review_decision") is not None
            else None
        ),
        review_feedback=(
            str(worker_result.get("review_feedback"))
            if worker_result.get("review_feedback")
            else None
        ),
        score=score,
    )

    append_jsonl(
        log_path,
        asdict(record),
    )

    return record


def process_timeout(
    *,
    active: ActiveProcess,
    timeout_seconds: int,
    log_path: Path,
) -> GenerationRecord:
    terminate_process(active.process)

    elapsed = round(
        time.perf_counter() - active.started_at,
        3,
    )

    record = GenerationRecord(
        task_id=active.task.task_id,
        task_index=active.task.task_index,
        level=active.task.level,
        status="timeout",
        elapsed_seconds=elapsed,
        attachment_present=bool(active.task.attachment_path),
        final_output_present=False,
        error=(
            f"Agent invocation exceeded {timeout_seconds} seconds "
            "and was terminated."
        ),
    )

    append_jsonl(
        log_path,
        asdict(record),
    )

    return record


def run_tasks(
    *,
    tasks: Sequence[GaiaTask],
    agent_module: str,
    agent_object: str,
    recursion_limit: int,
    workers: int,
    task_timeout_seconds: int,
    run_id: str,
    answers_path: Path,
    log_path: Path,
    scored_path: Path,
    stop_on_error: bool,
) -> list[GenerationRecord]:
    context = multiprocessing.get_context("spawn")
    pending = list(tasks)
    active_processes: dict[str, ActiveProcess] = {}
    records: list[GenerationRecord] = []
    completed_count = 0
    stop_requested = False

    while pending or active_processes:
        while (
            pending
            and len(active_processes) < workers
            and not stop_requested
        ):
            task = pending.pop(0)

            active = start_task_process(
                context=context,
                task=task,
                agent_module=agent_module,
                agent_object=agent_object,
                recursion_limit=recursion_limit,
                run_id=run_id,
            )

            active_processes[task.task_id] = active

            print(
                f"[START] task={task.task_id}, "
                f"index={task.task_index}, "
                f"level={task.level}, "
                f"pid={active.process.pid}"
            )

        if not active_processes:
            break

        made_progress = False

        for task_id, active in list(active_processes.items()):
            worker_result = get_worker_result(active)

            if worker_result is not None:
                active.process.join(timeout=1)

                record = process_completed_task(
                    active=active,
                    worker_result=worker_result,
                    answers_path=answers_path,
                    log_path=log_path,
                    scored_path=scored_path,
                )

                close_worker(active)
                del active_processes[task_id]
                records.append(record)
                completed_count += 1
                made_progress = True

                print(
                    f"[{completed_count}/{len(tasks)}] "
                    f"{task_id}: status={record.status}, "
                    f"elapsed={record.elapsed_seconds:.3f}s"
                    + (
                        f", score={record.score}"
                        if record.score is not None
                        else ""
                    )
                )

                if (
                    stop_on_error
                    and record.status != "ok"
                ):
                    stop_requested = True

                continue

            elapsed = time.perf_counter() - active.started_at

            if elapsed >= task_timeout_seconds:
                record = process_timeout(
                    active=active,
                    timeout_seconds=task_timeout_seconds,
                    log_path=log_path,
                )

                close_worker(active)
                del active_processes[task_id]
                records.append(record)
                completed_count += 1
                made_progress = True

                print(
                    f"[{completed_count}/{len(tasks)}] "
                    f"{task_id}: status=timeout, "
                    f"elapsed={record.elapsed_seconds:.3f}s"
                )

                if stop_on_error:
                    stop_requested = True

                continue

            if not active.process.is_alive():
                active.process.join(timeout=0.5)
                worker_result = get_worker_result(active)

                if worker_result is None:
                    worker_result = {
                        "ok": False,
                        "error": (
                            "Worker exited without returning a result. "
                            f"Exit code: {active.process.exitcode}."
                        ),
                    }

                record = process_completed_task(
                    active=active,
                    worker_result=worker_result,
                    answers_path=answers_path,
                    log_path=log_path,
                    scored_path=scored_path,
                )

                close_worker(active)
                del active_processes[task_id]
                records.append(record)
                completed_count += 1
                made_progress = True

                print(
                    f"[{completed_count}/{len(tasks)}] "
                    f"{task_id}: status={record.status}, "
                    f"elapsed={record.elapsed_seconds:.3f}s"
                )

                if (
                    stop_on_error
                    and record.status != "ok"
                ):
                    stop_requested = True

        if stop_requested:
            for active in active_processes.values():
                terminate_process(active.process)
                close_worker(active)

            active_processes.clear()
            break

        if not made_progress:
            sleep(0.1)

    return records


def validate_answer_file(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    answers: dict[str, str] = {}

    for row_number, row in enumerate(rows, start=1):
        task_id = str(row.get("task_id") or "").strip()
        model_answer = row.get("model_answer")

        if not task_id:
            raise ValueError(
                f"answers.jsonl row {row_number} has no task_id."
            )

        if task_id in answers:
            raise ValueError(
                f"answers.jsonl contains duplicate task_id: {task_id}"
            )

        if model_answer is None:
            raise ValueError(
                f"answers.jsonl row {row_number} has no model_answer."
            )

        answers[task_id] = str(model_answer)

    return answers


def score_answers(
    *,
    tasks: Sequence[GaiaTask],
    answers: Mapping[str, str],
) -> dict[str, Any]:
    level_counts = {
        1: {"correct": 0, "answered": 0, "selected": 0},
        2: {"correct": 0, "answered": 0, "selected": 0},
        3: {"correct": 0, "answered": 0, "selected": 0},
    }

    total_correct = 0
    total_answered = 0

    for task in tasks:
        level_counts[task.level]["selected"] += 1

        prediction = answers.get(task.task_id)

        if prediction is None:
            continue

        total_answered += 1
        level_counts[task.level]["answered"] += 1

        if task.ground_truth is None:
            continue

        if question_scorer(
            prediction,
            task.ground_truth,
        ):
            total_correct += 1
            level_counts[task.level]["correct"] += 1

    selected_count = len(tasks)

    return {
        "selected_tasks": selected_count,
        "answered_tasks": total_answered,
        "correct_answers": total_correct,
        "score_selected": (
            total_correct / selected_count
            if selected_count
            else 0.0
        ),
        "accuracy_answered": (
            total_correct / total_answered
            if total_answered
            else 0.0
        ),
        "by_level": {
            str(level): {
                **counts,
                "score_selected": (
                    counts["correct"] / counts["selected"]
                    if counts["selected"]
                    else None
                ),
                "accuracy_answered": (
                    counts["correct"] / counts["answered"]
                    if counts["answered"]
                    else None
                ),
            }
            for level, counts in level_counts.items()
        },
    }


def prepare_docker_runtime(
    *,
    image: str,
    startup_timeout_seconds: int = 180,
    pull_timeout_seconds: int = 900,
    retry_interval_seconds: int = 2,
) -> None:
    docker_executable = shutil.which("docker")

    if docker_executable is None:
        raise RuntimeError(
            "Docker is not installed or is not available on PATH."
        )

    print("Preparing Docker runtime...")

    if os.name == "nt":
        try:
            desktop_status = subprocess.run(
                [
                    docker_executable,
                    "desktop",
                    "status",
                    "--format",
                    "json",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=15,
            )

            status_output = (
                (desktop_status.stdout or "")
                + "\n"
                + (desktop_status.stderr or "")
            ).lower()

            if (
                desktop_status.returncode != 0
                or "running" not in status_output
            ):
                print("Starting Docker Desktop...")

                subprocess.run(
                    [
                        docker_executable,
                        "desktop",
                        "start",
                        "--detach",
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=30,
                )

        except (
            subprocess.TimeoutExpired,
            FileNotFoundError,
            OSError,
        ):
            pass

    attempts = max(
        1,
        startup_timeout_seconds // max(
            1,
            retry_interval_seconds,
        ),
    )
    last_error = ""

    for attempt in range(1, attempts + 1):
        try:
            docker_info = subprocess.run(
                [
                    docker_executable,
                    "info",
                    "--format",
                    "{{.ServerVersion}}",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=15,
            )

            if docker_info.returncode == 0:
                server_version = docker_info.stdout.strip()

                print(
                    "Docker daemon is ready"
                    + (
                        f" (server {server_version})."
                        if server_version
                        else "."
                    )
                )
                break

            last_error = (
                docker_info.stderr.strip()
                or docker_info.stdout.strip()
                or f"docker info exited with {docker_info.returncode}"
            )

        except subprocess.TimeoutExpired:
            last_error = "docker info timed out."

        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"

        if attempt < attempts:
            sleep(retry_interval_seconds)

    else:
        raise RuntimeError(
            "Docker did not become ready within "
            f"{startup_timeout_seconds} seconds.\n"
            f"Last error: {last_error}"
        )

    inspection = subprocess.run(
        [
            docker_executable,
            "image",
            "inspect",
            image,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )

    if inspection.returncode != 0:
        print(f"Pulling Docker image: {image}")

        try:
            pull = subprocess.run(
                [
                    docker_executable,
                    "pull",
                    image,
                ],
                timeout=pull_timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"Timed out while pulling Docker image '{image}' "
                f"after {pull_timeout_seconds} seconds."
            ) from exc

        if pull.returncode != 0:
            raise RuntimeError(
                f"Could not pull Docker image '{image}'. "
                f"Docker exited with code {pull.returncode}."
            )
    else:
        print(f"Docker image is available: {image}")

    print(
        f"Starting disposable Docker warm-up: {image}"
    )

    try:
        warmup = subprocess.run(
            [
                docker_executable,
                "run",
                "--rm",
                "--network",
                "none",
                "--read-only",
                "--tmpfs",
                "/tmp:rw,noexec,nosuid,size=16m",
                image,
                "python",
                "-c",
                "print('docker-runtime-ready')",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            "The Docker warm-up container did not finish within 60 seconds."
        ) from exc

    if warmup.returncode != 0:
        output = (
            warmup.stderr.strip()
            or warmup.stdout.strip()
            or "No error output was returned."
        )

        raise RuntimeError(
            f"The Docker warm-up container failed.\n{output}"
        )

    print("Docker runtime warm-up completed.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run gaia_task_solver_app on the gated GAIA benchmark, "
            "write leaderboard-compatible answers, and score the public "
            "validation split with the official GAIA normalization."
        )
    )

    parser.add_argument(
        "--agent-module",
        default=DEFAULT_AGENT_MODULE,
        help="Python module containing the compiled GAIA LangGraph app.",
    )
    parser.add_argument(
        "--agent-object",
        default=DEFAULT_AGENT_OBJECT,
        help="Object exposing invoke(state, config=...).",
    )
    parser.add_argument(
        "--dataset-repo",
        default=DEFAULT_DATASET_REPO,
        help="Hugging Face dataset repository.",
    )
    parser.add_argument(
        "--dataset-config",
        default=DEFAULT_DATASET_CONFIG,
        help="GAIA dataset config, normally 2023_all.",
    )
    parser.add_argument(
        "--split",
        choices=("validation", "test"),
        default="validation",
        help=(
            "Validation has public reference answers and is scored locally. "
            "Test produces a leaderboard submission without local scoring."
        ),
    )
    parser.add_argument(
        "--level",
        choices=("all", "1", "2", "3"),
        default="all",
        help="Run all levels or only one GAIA level.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Result directory. Default: "
            "benchmark_results/gaia_<split>_<UTC timestamp>."
        ),
    )
    continuation_group = parser.add_mutually_exclusive_group()
    continuation_group.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Skip task IDs already present in answers.jsonl. Failed and "
            "timed-out tasks are rerun in their normal dataset order."
        ),
    )
    continuation_group.add_argument(
        "--continue",
        dest="continue_run",
        action="store_true",
        help=(
            "Continue after the highest previously attempted task index, then "
            "retry earlier failed, timed-out, or unanswered tasks at the end."
        ),
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Zero-based first index after applying the level filter.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of tasks. Omit for all selected tasks.",
    )
    parser.add_argument(
        "--task-id",
        action="append",
        default=[],
        help="Run one task ID. Repeat for multiple tasks.",
    )
    parser.add_argument(
        "--recursion-limit",
        type=int,
        default=600,
        help="LangGraph recursion limit for each task.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Maximum simultaneous agent tasks. Default: 2.",
    )
    parser.add_argument(
        "--task-timeout",
        type=int,
        default=300,
        help=(
            "Hard timeout in seconds for one task. A timed-out process is "
            "terminated and remains pending for --resume. Default: 300."
        ),
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop scheduling tasks after the first error or timeout.",
    )
    parser.add_argument(
        "--skip-docker-warmup",
        action="store_true",
        help=(
            "Do not connect to Docker or pre-pull the restricted-Python image."
        ),
    )
    parser.add_argument(
        "--docker-image",
        default=os.getenv(
            "GAIA_DOCKER_IMAGE",
            DEFAULT_DOCKER_IMAGE,
        ),
        help="Docker image used by restricted_python_tool.",
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

    if args.workers <= 0:
        raise ValueError("--workers must be greater than 0.")

    if args.task_timeout <= 0:
        raise ValueError("--task-timeout must be greater than 0.")

    if not args.skip_docker_warmup:
        prepare_docker_runtime(
            image=args.docker_image,
            startup_timeout_seconds=180,
            pull_timeout_seconds=900,
            retry_interval_seconds=2,
        )

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )

    output_dir = args.output_dir or Path(
        "benchmark_results",
        f"gaia_task_solver_{args.split}_{timestamp}",
    )
    output_dir = output_dir.resolve()
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    answers_path = output_dir / "answers.jsonl"
    log_path = output_dir / "generation_log.jsonl"
    scored_path = output_dir / "scored_results.jsonl"
    summary_path = output_dir / "generation_summary.json"
    config_path = output_dir / "run_config.json"

    if (
        not args.resume
        and not args.continue_run
        and answers_path.exists()
        and answers_path.stat().st_size > 0
    ):
        raise FileExistsError(
            f"{answers_path} already exists. Use --resume, --continue, or another "
            "--output-dir."
        )

    token = (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )

    print(
        f"Loading GAIA {args.split} split into {output_dir}"
    )

    dataset_root, dataset_records = load_gaia_dataset(
        repo_id=args.dataset_repo,
        config_name=args.dataset_config,
        split=args.split,
        token=token,
    )

    all_tasks = parse_gaia_tasks(
        dataset_records,
        dataset_root=dataset_root,
        split=args.split,
    )

    selected_tasks = choose_tasks(
        all_tasks,
        level=args.level,
        start_index=args.start_index,
        limit=args.limit,
        selected_task_ids=set(args.task_id),
    )

    selected_task_ids = {
        task.task_id
        for task in selected_tasks
    }

    existing_rows = (
        read_jsonl(answers_path)
        if args.resume or args.continue_run
        else []
    )
    existing_answers = validate_answer_file(
        existing_rows
    )
    completed_task_ids = set(
        existing_answers
    )
    existing_log_rows = (
        read_jsonl(log_path)
        if args.continue_run
        else []
    )

    unknown_completed = completed_task_ids.difference(
        task.task_id
        for task in all_tasks
    )

    if unknown_completed:
        raise ValueError(
            f"answers.jsonl contains task IDs not found in the selected "
            f"GAIA split: {sorted(unknown_completed)}"
        )

    continue_from_index = None

    if args.continue_run:
        selected_task_id_set = {
            task.task_id
            for task in selected_tasks
        }

        attempted_indices = [
            int(row["task_index"])
            for row in existing_log_rows
            if (
                str(row.get("task_id") or "") in selected_task_id_set
                and _optional_int(row.get("task_index")) is not None
            )
        ]

        if attempted_indices:
            highest_attempted_index = max(attempted_indices)
            continue_from_index = highest_attempted_index + 1

            forward_tasks = [
                task
                for task in selected_tasks
                if (
                    task.task_index > highest_attempted_index
                    and task.task_id not in completed_task_ids
                )
            ]

            prior_unfinished_tasks = [
                task
                for task in selected_tasks
                if (
                    task.task_index <= highest_attempted_index
                    and task.task_id not in completed_task_ids
                )
            ]

            pending_tasks = forward_tasks + prior_unfinished_tasks

            print(
                f"Continuing after task index {highest_attempted_index}. "
                f"{len(prior_unfinished_tasks)} earlier unfinished task(s) "
                f"will run at the end."
            )

        else:
            pending_tasks = [
                task
                for task in selected_tasks
                if task.task_id not in completed_task_ids
            ]

            print(
                "No previous attempted tasks found; starting from the beginning."
            )

    else:
        pending_tasks = [
            task
            for task in selected_tasks
            if task.task_id not in completed_task_ids
        ]

    run_id = uuid.uuid4().hex

    run_config = {
        "created_at": utc_now_iso(),
        "run_id": run_id,
        "dataset": "GAIA",
        "dataset_repo": args.dataset_repo,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "level": args.level,
        "resume": args.resume,
        "continue": args.continue_run,
        "continue_from_index": continue_from_index,
        "dataset_task_count": len(all_tasks),
        "selected_task_count": len(selected_tasks),
        "pending_task_count": len(pending_tasks),
        "agent_module": args.agent_module,
        "agent_object": args.agent_object,
        "recursion_limit": args.recursion_limit,
        "workers": args.workers,
        "task_timeout_seconds": args.task_timeout,
        "start_index": args.start_index,
        "limit": args.limit,
        "selected_task_ids": args.task_id,
        "methodology_note": (
            "The agent receives only the GAIA question and an attachments list "
            "containing zero or one local file path. It never receives "
            "the reference answer, "
            "annotator metadata, or scorer output. Validation predictions "
            "are scored after the agent invocation with the official GAIA "
            "quasi-exact normalization. Reference answers are not written "
            "to output files."
        ),
    }

    write_json(
        config_path,
        run_config,
    )

    mode = (
        "continue" if args.continue_run
        else "resume" if args.resume
        else "selection"
    )

    print(
        f"Selected {len(selected_tasks)} task(s); "
        f"{len(pending_tasks)} remain after {mode}."
    )

    started = time.perf_counter()

    generated_records = run_tasks(
        tasks=pending_tasks,
        agent_module=args.agent_module,
        agent_object=args.agent_object,
        recursion_limit=args.recursion_limit,
        workers=args.workers,
        task_timeout_seconds=args.task_timeout,
        run_id=run_id,
        answers_path=answers_path,
        log_path=log_path,
        scored_path=scored_path,
        stop_on_error=args.stop_on_error,
    )

    all_answer_rows = read_jsonl(answers_path)
    all_answers = validate_answer_file(
        all_answer_rows
    )
    all_log_rows = read_jsonl(log_path)

    elapsed = round(
        time.perf_counter() - started,
        3,
    )

    selected_answer_ids = set(all_answers).intersection(
        selected_task_ids
    )

    summary: dict[str, Any] = {
        "finished_at": utc_now_iso(),
        "dataset": "GAIA",
        "split": args.split,
        "level": args.level,
        "dataset_task_count": len(all_tasks),
        "selected_tasks": len(selected_tasks),
        "previously_completed_tasks": len(
            completed_task_ids.intersection(
                selected_task_ids
            )
        ),
        "generated_this_run": len(generated_records),
        "answers_in_file": len(all_answer_rows),
        "selected_tasks_answered": len(
            selected_answer_ids
        ),
        "selected_tasks_complete": (
            selected_answer_ids == selected_task_ids
        ),
        "elapsed_seconds_this_run": elapsed,
        "status_counts_all_logs": _count_values(
            str(row.get("status", "unknown"))
            for row in all_log_rows
        ),
        "answers_path": str(answers_path),
        "generation_log_path": str(log_path),
        "run_config_path": str(config_path),
        "scored_results_path": (
            str(scored_path)
            if args.split == "validation"
            else None
        ),
    }

    if args.split == "validation":
        validation_scores = score_answers(
            tasks=selected_tasks,
            answers=all_answers,
        )
        summary["validation_scores"] = validation_scores

        print(
            "\nGAIA validation score:"
        )
        print(
            f"  Correct: {validation_scores['correct_answers']}/"
            f"{validation_scores['selected_tasks']}"
        )
        print(
            f"  Score on selected tasks: "
            f"{validation_scores['score_selected']:.4f}"
        )
        print(
            f"  Accuracy on answered tasks: "
            f"{validation_scores['accuracy_answered']:.4f}"
        )

        for level in ("1", "2", "3"):
            level_result = validation_scores["by_level"][level]

            if level_result["selected"] == 0:
                continue

            print(
                f"  Level {level}: "
                f"{level_result['correct']}/"
                f"{level_result['selected']} "
                f"({level_result['score_selected']:.4f})"
            )

    write_json(
        summary_path,
        summary,
    )

    print(
        f"\nAnswers: {answers_path}"
    )
    print(
        f"Generation log: {log_path}"
    )
    print(
        f"Generation summary: {summary_path}"
    )

    failed_this_run = any(
        record.status != "ok"
        for record in generated_records
    )

    return 1 if failed_this_run and args.stop_on_error else 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())


# Install:
# pip install -U datasets huggingface_hub

# Before the first run:
# 1. Accept the gated dataset conditions:
#    https://huggingface.co/datasets/gaia-benchmark/GAIA
# 2. Run `hf auth login

# Three-task validation smoke test:
# python .\run_gaia_task_solver.py `
#     --limit 3 `
#     --output-dir .\benchmark_results\gaia_task_solver\gaia_validation_test

# Level 1 validation:
# python .\run_gaia_task_solver.py `
#     --level 1 `
#     --output-dir .\benchmark_results\gaia_task_solver\gaia_level1_validation

# Full validation:
# python .\run_gaia_task_solver.py `
#     --output-dir .\benchmark_results\gaia_task_solver\gaia_validation_full `
#     --task-timeout 300

# Resume failed, missing, or timed-out tasks:
# python .\run_gaia_task_solver.py `
#     --output-dir .\benchmark_results\gaia_task_solver\gaia_validation_full `
#     --resume `
#     --task-timeout 300

# Generate the private-test leaderboard submission:
# python .\run_gaia_task_solver.py `
#     --split test `
#     --output-dir .\benchmark_results\gaia_task_solver\gaia_test_submission
