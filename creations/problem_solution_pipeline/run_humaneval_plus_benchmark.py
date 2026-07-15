from __future__ import annotations

import argparse
import ast
import doctest
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from time import sleep
import io
import tokenize


DEFAULT_AGENT_MODULE = (
    "creations.problem_solution_pipeline.problem_solution_pipeline"
)
DEFAULT_AGENT_OBJECT = "problem_solution_pipeline_app"
DEFAULT_DOCKER_IMAGE = "ganler/evalplus:latest"


@dataclass(frozen=True)
class ProblemSpec:
    task_id: str
    entry_point: str
    prompt: str
    function_signature: str
    docstring: str
    preamble: str


@dataclass
class GenerationRecord:
    task_id: str
    task_index: int
    status: str
    elapsed_seconds: float
    visible_test_count: int
    final_solution_present: bool
    syntax_valid: bool
    internal_total_passed: int | None = None
    internal_total_failed: int | None = None
    internal_repair_round: int | None = None
    internal_generation_attempt: int | None = None
    internal_valid_samples: int | None = None
    internal_invalid_samples: int | None = None
    error: str | None = None

def strip_python_inline_comment(source: str) -> str:
    """
    Remove Python comments from a code expression while preserving '#' symbols
    that occur inside string literals.

    Examples:
        'round(f([1, 2]), 2) # explanation'
        becomes:
        'round(f([1, 2]), 2)'

        '"value # text"'
        remains:
        '"value # text"'

    If tokenization fails because the expression is incomplete, a conservative
    character-based fallback is used.
    """
    if not source:
        return ""

    try:
        tokens = tokenize.generate_tokens(
            io.StringIO(source).readline
        )

        filtered_tokens = [
            token
            for token in tokens
            if token.type != tokenize.COMMENT
        ]

        cleaned = tokenize.untokenize(filtered_tokens).strip()
        return cleaned

    except (
        tokenize.TokenError,
        IndentationError,
        SyntaxError,
    ):
        in_single_quote = False
        in_double_quote = False
        escaped = False

        for index, character in enumerate(source):
            if escaped:
                escaped = False
                continue

            if character == "\\":
                escaped = True
                continue

            if character == "'" and not in_double_quote:
                in_single_quote = not in_single_quote
                continue

            if character == '"' and not in_single_quote:
                in_double_quote = not in_double_quote
                continue

            if (
                character == "#"
                and not in_single_quote
                and not in_double_quote
            ):
                return source[:index].rstrip()

        return source.strip()
    
def clean_doctest_comments(docstring: str) -> str:
    """
    Remove inline Python comments from doctest prompt lines in a docstring.

    Only code lines beginning with '>>>' or '...' are modified. Normal
    descriptive docstring text is preserved.

    Example:

        >>> round(find_zero([1, 2]), 2) # f(x) = 1 + 2x
        -0.5

    becomes:

        >>> round(find_zero([1, 2]), 2)
        -0.5
    """
    if not docstring:
        return ""

    cleaned_lines: list[str] = []

    for line in docstring.splitlines():
        stripped_line = line.lstrip()
        indentation = line[:len(line) - len(stripped_line)]

        if stripped_line.startswith(">>>"):
            code = stripped_line[3:].lstrip()
            code = strip_python_inline_comment(code)

            if code:
                cleaned_lines.append(
                    f"{indentation}>>> {code}"
                )
            else:
                cleaned_lines.append(
                    f"{indentation}>>>"
                )

        elif stripped_line.startswith("..."):
            code = stripped_line[3:].lstrip()
            code = strip_python_inline_comment(code)

            if code:
                cleaned_lines.append(
                    f"{indentation}... {code}"
                )
            else:
                cleaned_lines.append(
                    f"{indentation}..."
                )

        else:
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines)
    
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path: Path, item: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(item, ensure_ascii=False) + "\n")
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
                raise ValueError(
                    f"Invalid JSONL at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(
                    f"Expected a JSON object at {path}:{line_number}."
                )
            rows.append(value)
    return rows


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
            f"'{module_name}.{object_name}' does not expose an invoke(...) method."
        )
    return agent


def find_target_function(tree: ast.Module, entry_point: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == entry_point:
                if isinstance(node, ast.AsyncFunctionDef):
                    raise ValueError(
                        f"HumanEval task '{entry_point}' unexpectedly uses async def."
                    )
                return node
    raise ValueError(f"Could not find function '{entry_point}' in the prompt.")


def parse_problem(
    task_id: str,
    problem: Mapping[str, Any],
) -> ProblemSpec:
    """
    Parse one HumanEval problem into the fields required by the generated
    problem-solving agent.

    Inline comments are removed from doctest expressions before the docstring
    is stored and passed to the agent. Normal descriptive comments and prose
    outside doctest prompt lines are preserved.
    """
    prompt = str(problem["prompt"])
    entry_point = str(problem["entry_point"])

    try:
        tree = ast.parse(prompt)

    except SyntaxError as exc:
        raise ValueError(
            f"Prompt for {task_id} is not valid Python: {exc}"
        ) from exc

    function_node = find_target_function(
        tree,
        entry_point
    )

    lines = prompt.splitlines()

    if not function_node.body:
        raise ValueError(
            f"Function '{entry_point}' has no body in {task_id}."
        )

    first_body_line = function_node.body[0].lineno - 1

    function_signature = "\n".join(
        lines[
            function_node.lineno - 1:
            first_body_line
        ]
    ).rstrip()

    if not function_signature.endswith(":"):
        raise ValueError(
            f"Could not extract a complete function signature "
            f"for {task_id}."
        )

    original_docstring = (
        ast.get_docstring(
            function_node,
            clean=False
        )
        or ""
    )

    cleaned_docstring = clean_doctest_comments(
        original_docstring
    )

    preamble = "\n".join(
        lines[:function_node.lineno - 1]
    ).strip()

    return ProblemSpec(
        task_id=task_id,
        entry_point=entry_point,
        prompt=prompt,
        function_signature=function_signature,
        docstring=cleaned_docstring,
        preamble=preamble,
    )


def extract_visible_doctest_asserts(
    docstring: str,
    *,
    max_tests: int,
) -> list[str]:
    """
    Extract visible examples from a HumanEval docstring and convert them into
    executable assert statements.

    Supported formats:

    Standard doctest:

        >>> add(1, 2)
        3

    Inline equality:

        add(1, 2) == 3

    Inline equality with an explanatory calculation:

        double_the_difference([1, 3, 2, 0]) == 1 + 9 + 0 + 0 = 10

    The last literal value is treated as the expected result. Inline Python
    comments are removed before parsing.

    Malformed or unsupported examples are skipped instead of terminating the
    benchmark. Hidden HumanEval and HumanEval+ tests are never accessed.
    """
    if not docstring or max_tests <= 0:
        return []

    cleaned_docstring = clean_doctest_comments(docstring)

    tests: list[str] = []
    seen_tests: set[str] = set()

    def add_assertion(
        source: str,
        expected_text: str,
    ) -> None:
        """
        Add an assertion only when the source is a valid Python expression and
        the expected result is a Python literal.
        """
        if len(tests) >= max_tests:
            return

        source = strip_python_inline_comment(source).strip()
        expected_text = strip_python_inline_comment(expected_text).strip()

        if not source or not expected_text:
            return

        if "Traceback (most recent call last)" in expected_text:
            return

        try:
            ast.parse(source, mode="eval")
            expected_value = ast.literal_eval(expected_text)

        except (
            SyntaxError,
            ValueError,
            TypeError,
            MemoryError,
            RecursionError,
        ):
            return

        assertion = f"assert ({source}) == {expected_value!r}"

        if assertion not in seen_tests:
            seen_tests.add(assertion)
            tests.append(assertion)

    # ---------------------------------------------------------------
    # 1. Parse standard doctest examples.
    # ---------------------------------------------------------------
    try:
        parser = doctest.DocTestParser()
        examples = parser.get_examples(cleaned_docstring)

    except Exception:
        examples = []

    for example in examples:
        add_assertion(
            example.source,
            example.want,
        )

        if len(tests) >= max_tests:
            return tests

    # ---------------------------------------------------------------
    # 2. Conservative fallback for malformed >>> doctest examples.
    # ---------------------------------------------------------------
    lines = cleaned_docstring.splitlines()
    line_index = 0

    while (
        line_index < len(lines)
        and len(tests) < max_tests
    ):
        current_line = lines[line_index].lstrip()

        if not current_line.startswith(">>>"):
            line_index += 1
            continue

        source = strip_python_inline_comment(
            current_line[3:].strip()
        )

        if not source:
            line_index += 1
            continue

        expected_index = line_index + 1

        while (
            expected_index < len(lines)
            and not lines[expected_index].strip()
        ):
            expected_index += 1

        if expected_index >= len(lines):
            break

        expected_line = lines[expected_index].strip()

        if expected_line.startswith((">>>", "...")):
            line_index = expected_index
            continue

        add_assertion(
            source,
            expected_line,
        )

        line_index = expected_index + 1

    # ---------------------------------------------------------------
    # 3. Parse inline equality examples.
    # ---------------------------------------------------------------
    for original_line in lines:
        if len(tests) >= max_tests:
            break

        line = strip_python_inline_comment(
            original_line.strip()
        )

        if not line:
            continue

        if line.startswith((">>>", "...")):
            continue

        if "==" not in line:
            continue

        source_part, remainder = line.split("==", 1)

        source = source_part.strip()
        remainder = remainder.strip()

        if not source or not remainder:
            continue

        # Validate the left side as a Python expression. This prevents ordinary
        # descriptive prose containing "==" from becoming a test.
        try:
            ast.parse(source, mode="eval")
        except (SyntaxError, ValueError):
            continue

        expected_candidates: list[str] = []

        # First try the complete expression after ==.
        #
        # Example:
        #     function([1]) == 1
        expected_candidates.append(remainder)

        # Then try the value after the final single or repeated equals sign.
        #
        # Example:
        #     function([1, 3]) == 1 + 9 = 10
        #
        # The final candidate becomes "10".
        if "=" in remainder:
            final_part = remainder.rsplit("=", 1)[-1].strip()

            if final_part:
                expected_candidates.insert(0, final_part)

        for expected_text in expected_candidates:
            previous_count = len(tests)

            add_assertion(
                source,
                expected_text,
            )

            if len(tests) > previous_count:
                break

    return tests


def strip_markdown_fence(text: str) -> str:
    value = text.strip()
    fenced = re.fullmatch(
        r"```(?:python)?\s*\n(?P<code>.*)\n```",
        value,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if fenced:
        return fenced.group("code").strip()
    return value


def contains_entry_function(code: str, entry_point: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return bool(
            re.search(rf"^\s*def\s+{re.escape(entry_point)}\s*\(", code, re.MULTILINE)
        )

    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == entry_point
        for node in tree.body
    )


def build_self_contained_solution(spec: ProblemSpec, generated: str) -> str:
    generated = strip_markdown_fence(generated)

    if not generated:
        return make_failure_solution(spec, "The agent returned an empty solution.")

    if contains_entry_function(generated, spec.entry_point):
        if spec.preamble:
            return f"{spec.preamble}\n\n{generated.strip()}\n"
        return generated.strip() + "\n"

    # This fallback supports completion-style output even though the generated
    # agent is expected to return a complete function definition.
    return f"{spec.prompt.rstrip()}\n{generated.rstrip()}\n"


def make_failure_solution(spec: ProblemSpec, reason: str) -> str:
    safe_reason = reason.replace("\\", "\\\\").replace('"', '\\"')
    pieces: list[str] = []
    if spec.preamble:
        pieces.append(spec.preamble)
    pieces.append(spec.function_signature)
    pieces.append(f'    raise NotImplementedError("{safe_reason[:500]}")')
    return "\n\n".join(pieces[:1]) + (
        "\n\n" if spec.preamble else ""
    ) + "\n".join(pieces[1:] if spec.preamble else pieces) + "\n"


def syntax_is_valid(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except (SyntaxError, ValueError):
        return False


def invoke_agent(
    agent: Any,
    spec: ProblemSpec,
    visible_tests: Sequence[str],
    *,
    recursion_limit: int,
    run_id: str,
) -> Mapping[str, Any]:
    state = {
        "function_signature": spec.function_signature,
        "docstring": spec.docstring,
        "test_cases": list(visible_tests),
        "task_id": spec.task_id,
    }

    safe_task_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", spec.task_id)
    config = {
        "recursion_limit": recursion_limit,
        "configurable": {
            "user_id": "humaneval_plus_benchmark",
            "run_name": "humaneval_plus_benchmark",
            "thread_id": f"humaneval:{run_id}:{safe_task_id}:{uuid.uuid4().hex}",
        },
    }

    response = agent.invoke(state, config=config)
    if not isinstance(response, Mapping):
        raise TypeError(
            f"Agent returned {type(response).__name__}; expected a mapping."
        )
    return response


def task_sort_key(task_id: str) -> tuple[str, int | str]:
    prefix, separator, suffix = task_id.partition("/")
    if separator and suffix.isdigit():
        return prefix, int(suffix)
    return prefix, suffix


def choose_tasks(
    dataset: Mapping[str, Mapping[str, Any]],
    *,
    start_index: int,
    limit: int | None,
    selected_task_ids: set[str],
) -> list[tuple[str, Mapping[str, Any]]]:
    items = sorted(dataset.items(), key=lambda item: task_sort_key(item[0]))

    if selected_task_ids:
        missing = selected_task_ids.difference(dataset.keys())
        if missing:
            raise ValueError(f"Unknown task IDs: {sorted(missing)}")
        items = [item for item in items if item[0] in selected_task_ids]
    else:
        items = items[start_index:]
        if limit is not None:
            items = items[:limit]

    return items


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")


def stream_subprocess(command: Sequence[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("\nEvaluator command:")
    print(" ".join(command))
    print()

    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
            log_file.flush()
        return process.wait()


def run_evalplus_evaluation(
    *,
    samples_path: Path,
    output_dir: Path,
    mode: str,
    docker_image: str,
    parallel: int | None,
    base_only: bool,
    test_details: bool,
) -> int:
    common_args = [
        "--dataset",
        "humaneval",
        "--samples",
    ]

    extra_args = ["--i-just-wanna-run"]
    if parallel is not None:
        extra_args.extend(["--parallel", str(parallel)])
    if base_only:
        extra_args.append("--base-only")
    if test_details:
        extra_args.append("--test-details")

    if mode == "docker":
        if shutil.which("docker") is None:
            raise RuntimeError(
                "Docker is not available. Install Docker or use "
                "--evaluation-mode local."
            )

        mount_source = str(output_dir.resolve())
        command = [
            "docker",
            "run",
            "--rm",
            "--mount",
            f"type=bind,source={mount_source},target=/app",
            docker_image,
            "evalplus.evaluate",
            *common_args,
            f"/app/{samples_path.name}",
            *extra_args,
        ]
    elif mode == "local":
        command = [
            sys.executable,
            "-m",
            "evalplus.evaluate",
            *common_args,
            str(samples_path.resolve()),
            *extra_args,
        ]
    else:
        return 0

    return stream_subprocess(command, output_dir / "evaluation_output.txt")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate one final solution per HumanEval task with a custom "
            "LangGraph agent, then evaluate the samples with EvalPlus."
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
            "benchmark_results/humaneval_plus_<UTC timestamp>."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip task IDs already present in samples.jsonl.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Zero-based first task index for a partial run.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of tasks to generate. Omit for all 164 tasks.",
    )
    parser.add_argument(
        "--task-id",
        action="append",
        default=[],
        help="Run one specific task ID. Repeat the option for multiple tasks.",
    )
    parser.add_argument(
        "--visible-tests",
        choices=("doctest", "none"),
        default="doctest",
        help=(
            "Tests exposed to the agent. 'doctest' uses only examples already "
            "visible in the published prompt. Hidden benchmark tests are never "
            "passed to the agent."
        ),
    )
    parser.add_argument(
        "--max-visible-tests",
        type=int,
        default=20,
        help="Maximum visible doctest assertions given to the agent per task.",
    )
    parser.add_argument(
        "--recursion-limit",
        type=int,
        default=100,
        help="LangGraph recursion limit for each task invocation.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop generation after the first agent error.",
    )
    parser.add_argument(
        "--evaluation-mode",
        choices=("docker", "local", "none"),
        default="docker",
        help=(
            "How to run EvalPlus. Docker is safer because generated code is "
            "untrusted."
        ),
    )
    parser.add_argument(
        "--docker-image",
        default=DEFAULT_DOCKER_IMAGE,
        help="EvalPlus Docker image used in docker evaluation mode.",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=None,
        help="Worker count passed to evalplus.evaluate.",
    )
    parser.add_argument(
        "--base-only",
        action="store_true",
        help=(
            "Evaluate only original HumanEval tests. By default EvalPlus reports "
            "both HumanEval Base and HumanEval+ Base + Extra."
        ),
    )
    parser.add_argument(
        "--test-details",
        action="store_true",
        help="Ask EvalPlus to retain detailed test information.",
    )
    return parser

def main() -> int:
    args = build_parser().parse_args()

    if args.start_index < 0:
        raise ValueError("--start-index must be at least 0.")

    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be greater than 0.")

    if args.max_visible_tests < 0:
        raise ValueError("--max-visible-tests must be at least 0.")

    try:
        from evalplus.data import get_human_eval_plus
    except ImportError as exc:
        raise RuntimeError(
            "EvalPlus is not installed. Run: "
            "pip install --upgrade evalplus"
        ) from exc

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )

    output_dir = args.output_dir or Path(
        "benchmark_results",
        f"humaneval_plus_{timestamp}"
    )

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    samples_path = output_dir / "samples.jsonl"
    log_path = output_dir / "generation_log.jsonl"
    summary_path = output_dir / "generation_summary.json"
    config_path = output_dir / "run_config.json"

    if (
        not args.resume
        and samples_path.exists()
        and samples_path.stat().st_size > 0
    ):
        raise FileExistsError(
            f"{samples_path} already exists. "
            f"Use --resume or another --output-dir."
        )

    run_id = uuid.uuid4().hex

    print(
        f"Loading HumanEval+ dataset into {output_dir}"
    )

    dataset = get_human_eval_plus()
    dataset_task_count = len(dataset)

    tasks = choose_tasks(
        dataset,
        start_index=args.start_index,
        limit=args.limit,
        selected_task_ids=set(args.task_id),
    )

    selected_task_count = len(tasks)
    is_full_dataset_run = (
        selected_task_count == dataset_task_count
        and {task_id for task_id, _ in tasks} == set(dataset.keys())
    )

    run_config = {
        "created_at": utc_now_iso(),
        "run_id": run_id,
        "dataset": "HumanEval+",
        "dataset_task_count": dataset_task_count,
        "selected_task_count": selected_task_count,
        "full_dataset_run": is_full_dataset_run,
        "agent_module": args.agent_module,
        "agent_object": args.agent_object,
        "visible_tests": args.visible_tests,
        "max_visible_tests": args.max_visible_tests,
        "recursion_limit": args.recursion_limit,
        "start_index": args.start_index,
        "limit": args.limit,
        "selected_task_ids": args.task_id,
        "requested_evaluation_mode": args.evaluation_mode,
        "base_only": args.base_only,
        "methodology_note": (
            "The agent receives the published HumanEval prompt and "
            "optional doctest examples extracted from that prompt. "
            "It never receives base_input, plus_input, "
            "canonical_solution, or evaluator outputs."
        ),
    }

    write_json(config_path, run_config)

    existing_rows = (
        read_jsonl(samples_path)
        if args.resume
        else []
    )

    completed_task_ids = {
        str(row.get("task_id"))
        for row in existing_rows
        if row.get("task_id") is not None
    }

    pending_tasks = [
        item
        for item in tasks
        if item[0] not in completed_task_ids
    ]

    print(
        f"Selected {len(tasks)} task(s); "
        f"{len(pending_tasks)} remain after resume."
    )

    agent = load_agent(
        args.agent_module,
        args.agent_object
    )

    started = time.perf_counter()
    generation_records: list[GenerationRecord] = []
    agent_errors = 0

    sorted_dataset_ids = sorted(
        dataset.keys(),
        key=task_sort_key
    )

    dataset_indices = {
        task_id: index
        for index, task_id in enumerate(sorted_dataset_ids)
    }

    for position, (task_id, problem) in enumerate(
        pending_tasks,
        start=1
    ):
        absolute_index = dataset_indices[task_id]

        print(
            f"[{position}/{len(pending_tasks)}] "
            f"{task_id} "
            f"(dataset index {absolute_index})"
        )

        task_started = time.perf_counter()
        spec = parse_problem(task_id, problem)

        if args.visible_tests == "doctest":
            visible_tests = extract_visible_doctest_asserts(
                spec.docstring,
                max_tests=args.max_visible_tests,
            )
        else:
            visible_tests = []

        try:
            response = invoke_agent(
                agent,
                spec,
                visible_tests,
                recursion_limit=args.recursion_limit,
                run_id=run_id,
            )

            generated = str(
                response.get("final_solution") or ""
            )

            solution = build_self_contained_solution(
                spec,
                generated
            )

            status = "ok"
            error = None

        except Exception as exc:
            agent_errors += 1
            error = f"{type(exc).__name__}: {exc}"
            status = "agent_error"
            response = {}

            solution = make_failure_solution(
                spec,
                error
            )

            traceback.print_exc()

        syntax_valid = syntax_is_valid(solution)

        if not syntax_valid and status == "ok":
            status = "invalid_python"

        append_jsonl(
            samples_path,
            {
                "task_id": task_id,
                "solution": solution,
            },
        )

        record = GenerationRecord(
            task_id=task_id,
            task_index=absolute_index,
            status=status,
            elapsed_seconds=round(
                time.perf_counter() - task_started,
                3
            ),
            visible_test_count=len(visible_tests),
            final_solution_present=bool(
                response.get("final_solution")
            ),
            syntax_valid=syntax_valid,
            internal_total_passed=_optional_int(
                response.get("total_passed")
            ),
            internal_total_failed=_optional_int(
                response.get("total_failed")
            ),
            internal_repair_round=_optional_int(
                response.get("repair_round")
            ),
            internal_generation_attempt=_optional_int(
                response.get("generation_attempt")
            ),
            internal_valid_samples=_optional_len(
                response.get("valid_samples")
            ),
            internal_invalid_samples=_optional_len(
                response.get("invalid_samples")
            ),
            error=error,
        )

        generation_records.append(record)
        append_jsonl(log_path, asdict(record))

        print(
            f"    status={status}, "
            f"visible_tests={len(visible_tests)}, "
            f"elapsed={record.elapsed_seconds:.3f}s"
        )

        if error and args.stop_on_error:
            break

    all_log_rows = read_jsonl(log_path)

    elapsed_total = round(
        time.perf_counter() - started,
        3
    )

    all_sample_rows = read_jsonl(samples_path)

    generated_task_ids = {
        str(row.get("task_id"))
        for row in all_sample_rows
        if row.get("task_id") is not None
    }

    complete_full_dataset = (
        generated_task_ids == set(dataset.keys())
    )

    evaluation_skipped_reason: str | None = None

    if (
        args.evaluation_mode != "none"
        and not complete_full_dataset
    ):
        evaluation_skipped_reason = (
            "EvalPlus evaluation was skipped because samples.jsonl "
            f"contains {len(generated_task_ids)} of "
            f"{dataset_task_count} HumanEval tasks. The standard "
            "EvalPlus evaluator requires at least one solution for "
            "every task in the selected dataset."
        )

    summary = {
        "finished_at": utc_now_iso(),
        "dataset_task_count": dataset_task_count,
        "selected_tasks": len(tasks),
        "previously_completed_tasks": len(
            completed_task_ids.intersection(
                task_id
                for task_id, _ in tasks
            )
        ),
        "generated_this_run": len(generation_records),
        "samples_in_file": len(all_sample_rows),
        "unique_tasks_in_samples": len(generated_task_ids),
        "full_dataset_complete": complete_full_dataset,
        "agent_errors_this_run": agent_errors,
        "syntax_invalid_this_run": sum(
            1
            for record in generation_records
            if not record.syntax_valid
        ),
        "elapsed_seconds_this_run": elapsed_total,
        "status_counts_all_logs": _count_values(
            str(row.get("status", "unknown"))
            for row in all_log_rows
        ),
        "samples_path": str(samples_path),
        "evaluation_requested": (
            args.evaluation_mode != "none"
        ),
        "evaluation_performed": False,
        "evaluation_skipped_reason": (
            evaluation_skipped_reason
        ),
    }

    print(
        f"\nGeneration complete. Samples: {samples_path}"
    )

    if args.evaluation_mode == "none":
        write_json(summary_path, summary)

        print(
            f"Generation summary: {summary_path}"
        )

        return 0

    if not complete_full_dataset:
        write_json(summary_path, summary)

        print(
            f"Generation summary: {summary_path}"
        )

        print(
            "\nEvalPlus evaluation skipped."
        )

        print(
            evaluation_skipped_reason
        )

        print(
            "\nUse --evaluation-mode none for partial generation "
            "tests, or generate all 164 HumanEval tasks for an "
            "official HumanEval/HumanEval+ score."
        )

        return 0

    evaluation_code = run_evalplus_evaluation(
        samples_path=samples_path,
        output_dir=output_dir,
        mode=args.evaluation_mode,
        docker_image=args.docker_image,
        parallel=args.parallel,
        base_only=args.base_only,
        test_details=args.test_details,
    )

    summary["evaluation_performed"] = True
    summary["evaluation_exit_code"] = evaluation_code
    write_json(summary_path, summary)

    print(
        f"Generation summary: {summary_path}"
    )

    if evaluation_code != 0:
        print(
            f"EvalPlus exited with code {evaluation_code}. "
            f"See {output_dir / 'evaluation_output.txt'}",
            file=sys.stderr,
        )

    return evaluation_code


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_len(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return len(value)
    except TypeError:
        return None


def _count_values(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts

def prepare_docker_runtime(
    *,
    images: Sequence[str],
    warmup_image: str = "python:3.11-slim",
    startup_timeout_seconds: int = 180,
    pull_timeout_seconds: int = 900,
    retry_interval_seconds: int = 2,
) -> None:
    """
    Start Docker Desktop when possible, wait for the Docker daemon, ensure the
    required images are available, and start one disposable warm-up container.

    This separates Docker startup and image-download time from the short
    timeout used when generated code is tested.

    Args:
        images:
            Docker images that must exist before the benchmark starts.
        warmup_image:
            Image used to start a small disposable container.
        startup_timeout_seconds:
            Maximum time to wait for the Docker daemon.
        pull_timeout_seconds:
            Maximum time allowed for downloading each missing image.
        retry_interval_seconds:
            Delay between Docker daemon connection attempts.

    Raises:
        RuntimeError:
            If Docker is unavailable, the daemon does not become ready, an
            image cannot be pulled, or the warm-up container fails.
    """
    docker_executable = shutil.which("docker")

    if docker_executable is None:
        raise RuntimeError(
            "Docker is not installed or is not available on PATH."
        )

    print("Preparing Docker runtime...")

    # On Windows, try to start Docker Desktop automatically.
    # Older Docker Desktop versions may not support these commands, so this
    # step is best effort. The docker-info loop below remains authoritative.
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
            # Continue to docker info. This supports Docker Desktop versions
            # that do not provide the desktop subcommands.
            pass

    attempts = max(
        1,
        startup_timeout_seconds // max(1, retry_interval_seconds),
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

    unique_images = list(dict.fromkeys(
        image.strip()
        for image in images
        if image and image.strip()
    ))

    for image in unique_images:
        image_inspection = subprocess.run(
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

        if image_inspection.returncode == 0:
            print(f"Docker image is available: {image}")
            continue

        print(f"Pulling Docker image: {image}")

        try:
            image_pull = subprocess.run(
                [
                    docker_executable,
                    "pull",
                    image,
                ],
                stdout=None,
                stderr=None,
                text=True,
                timeout=pull_timeout_seconds,
            )

        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"Timed out while pulling Docker image '{image}' "
                f"after {pull_timeout_seconds} seconds."
            ) from exc

        if image_pull.returncode != 0:
            raise RuntimeError(
                f"Could not pull Docker image '{image}'. "
                f"Docker exited with code {image_pull.returncode}."
            )

    if warmup_image:
        print(
            f"Starting disposable Docker warm-up container: "
            f"{warmup_image}"
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
                    warmup_image,
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
                "The Docker warm-up container did not finish within "
                "60 seconds."
            ) from exc

        if warmup.returncode != 0:
            error_output = (
                warmup.stderr.strip()
                or warmup.stdout.strip()
                or "No error output was returned."
            )

            raise RuntimeError(
                "The Docker warm-up container failed.\n"
                f"{error_output}"
            )

        print("Docker runtime warm-up completed.")

if __name__ == "__main__":
    prepare_docker_runtime(
        images=(
            "python:3.11-slim",
            DEFAULT_DOCKER_IMAGE,
        ),
        warmup_image="python:3.11-slim",
        startup_timeout_seconds=180,
        pull_timeout_seconds=900,
        retry_interval_seconds=2,
    )
    raise SystemExit(main())

    # Test 3
    # python .\run_humaneval_plus_benchmark.py `
    # --limit 3 `
    # --output-dir .\benchmark_results\humaneval_plus_test `
    # --evaluation-mode none

    # Run All
    # python .\run_humaneval_plus_benchmark.py `
    # --output-dir .\benchmark_results\humaneval_plus_full `
    # --evaluation-mode docker

    # Resume
    # python .\run_humaneval_plus_benchmark.py `
    # --output-dir .\benchmark_results\humaneval_plus_full `
    # --resume `
    # --evaluation-mode docker