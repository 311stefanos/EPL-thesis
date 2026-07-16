from __future__ import annotations

import argparse
import ast
import gzip
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
from time import sleep
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_AGENT_MODULE = (
    "creations.problem_solution_pipeline.problem_solution_pipeline"
)
DEFAULT_AGENT_OBJECT = "problem_solution_pipeline_app"
DEFAULT_DOCKER_IMAGE = "ganler/evalplus:latest"
DEFAULT_SANDBOX_IMAGE = "python:3.11-slim"


@dataclass(frozen=True)
class ProblemSpec:
    task_id: str
    entry_point: str
    evalplus_prompt: str
    function_signature: str
    docstring: str
    preamble: str
    public_tests: tuple[str, ...]
    internal_tests: tuple[str, ...]
    test_imports: tuple[str, ...]


@dataclass
class GenerationRecord:
    task_id: str
    task_index: int
    status: str
    elapsed_seconds: float
    visible_test_count: int
    test_import_count: int
    final_solution_present: bool
    syntax_valid: bool
    internal_total_passed: int | None = None
    internal_total_failed: int | None = None
    internal_repair_round: int | None = None
    internal_generation_attempt: int | None = None
    internal_valid_samples: int | None = None
    internal_invalid_samples: int | None = None
    error: str | None = None


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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")


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

    module_file = getattr(module, "__file__", None)
    if module_file:
        print(f"Loaded agent module from: {module_file}")

    return agent


def task_numeric_id(task_id: str) -> str:
    value = str(task_id)
    return value.rsplit("/", 1)[-1]


def task_sort_key(task_id: str) -> tuple[str, int | str]:
    value = str(task_id)
    prefix, separator, suffix = value.partition("/")

    if separator and suffix.isdigit():
        return prefix.lower(), int(suffix)

    if value.isdigit():
        return "mbpp", int(value)

    return prefix.lower(), suffix


def find_target_function(
    tree: ast.Module,
    entry_point: str,
) -> ast.FunctionDef:
    """Find the public reference function matching the MBPP entry point."""
    for node in tree.body:
        if isinstance(node, ast.AsyncFunctionDef) and node.name == entry_point:
            raise ValueError(
                f"MBPP task '{entry_point}' unexpectedly uses async def."
            )

        if isinstance(node, ast.FunctionDef) and node.name == entry_point:
            return node

    raise ValueError(
        f"Could not find function '{entry_point}' in the public MBPP code."
    )


def build_signature_from_function_node(
    function_node: ast.FunctionDef,
) -> str:
    """
    Build a clean function signature from a public MBPP reference function.

    Type annotations are removed because the reference code may use annotation
    names that are not imported in the generated submission. Parameter names,
    defaults, positional-only markers, keyword-only markers, *args, and
    **kwargs are preserved.
    """
    import copy

    arguments = copy.deepcopy(function_node.args)

    all_arguments = [
        *arguments.posonlyargs,
        *arguments.args,
        *arguments.kwonlyargs,
    ]

    for argument in all_arguments:
        argument.annotation = None
        argument.type_comment = None

    if arguments.vararg is not None:
        arguments.vararg.annotation = None
        arguments.vararg.type_comment = None

    if arguments.kwarg is not None:
        arguments.kwarg.annotation = None
        arguments.kwarg.type_comment = None

    return f"def {function_node.name}({ast.unparse(arguments)}):"


def infer_signature_from_public_tests(
    entry_point: str,
    public_tests: Sequence[str],
) -> str | None:
    """
    Infer a conservative callable signature from public MBPP assertions.

    This is only a fallback when the public reference code cannot be parsed.
    Positional parameters receive generic names. Explicit keyword names are
    preserved so keyword-based public tests can still call the function.
    """
    matching_calls: list[ast.Call] = []

    for test in public_tests:
        try:
            tree = ast.parse(test)
        except (SyntaxError, ValueError):
            continue

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue

            if (
                isinstance(node.func, ast.Name)
                and node.func.id == entry_point
            ):
                matching_calls.append(node)

    if not matching_calls:
        return None

    # Use the call containing the most explicit arguments.
    call = max(
        matching_calls,
        key=lambda item: len(item.args) + len(item.keywords),
    )

    positional_names = [
        f"arg{index}"
        for index in range(1, len(call.args) + 1)
    ]

    keyword_names: list[str] = []
    for keyword in call.keywords:
        if keyword.arg and keyword.arg not in keyword_names:
            keyword_names.append(keyword.arg)

    parameters = positional_names + keyword_names

    if not parameters:
        return f"def {entry_point}():"

    return f"def {entry_point}({', '.join(parameters)}):"


def derive_mbpp_function_signature(
    *,
    entry_point: str,
    original_problem: Mapping[str, Any],
    public_tests: Sequence[str],
) -> str:
    """
    Derive the agent-facing signature without exposing a reference solution.

    The preferred source is the public sanitized-MBPP ``code`` field. Only
    the target function's signature is extracted; its implementation body is
    never included in the agent state. If parsing fails, the argument count is
    inferred from the public assertions.
    """
    public_code = str(original_problem.get("code") or "").strip()

    if public_code:
        try:
            tree = ast.parse(public_code)
            function_node = find_target_function(tree, entry_point)
            return build_signature_from_function_node(function_node)
        except (SyntaxError, ValueError, TypeError):
            pass

    inferred_signature = infer_signature_from_public_tests(
        entry_point,
        public_tests,
    )

    if inferred_signature is not None:
        return inferred_signature

    # Last-resort signature. It keeps the benchmark running but should be rare
    # for sanitized MBPP because public code and tests are normally available.
    return f"def {entry_point}(*args, **kwargs):"


def normalize_test_imports(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, str):
        candidates = [value]
    elif isinstance(value, Sequence):
        candidates = [str(item) for item in value]
    else:
        return []

    imports: list[str] = []

    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate:
            continue

        try:
            tree = ast.parse(candidate)
        except SyntaxError:
            continue

        if not tree.body:
            continue

        if all(isinstance(node, (ast.Import, ast.ImportFrom)) for node in tree.body):
            imports.append(candidate)

    return imports


def normalize_public_tests(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, str):
        candidates = [value]
    elif isinstance(value, Sequence):
        candidates = [str(item) for item in value]
    else:
        return []

    tests: list[str] = []

    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate:
            continue

        try:
            tree = ast.parse(candidate)
        except SyntaxError:
            continue

        if not tree.body:
            continue

        # Original MBPP test_list entries are normally one assert statement.
        # Keeping only asserts prevents accidental execution of arbitrary
        # auxiliary statements as visible tests.
        if all(isinstance(node, ast.Assert) for node in tree.body):
            tests.append(candidate)

    return tests


def add_imports_to_assert(test: str, imports: Sequence[str]) -> str:
    """
    Preserve a clean original MBPP assertion while making test_imports
    available inside the existing agent sandbox.

    The agent sandbox accepts strings that begin with ``assert``. Therefore,
    imports are executed inside the assertion before its condition is checked.
    The clean original tests are also included in the agent docstring.
    """
    if not imports:
        return test

    try:
        tree = ast.parse(test)
    except SyntaxError:
        return test

    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assert):
        return test

    assertion = tree.body[0]
    condition = ast.unparse(assertion.test)
    import_code = "\n".join(imports)

    wrapped = (
        f"assert (exec({import_code!r}, globals()) is None) "
        f"and ({condition})"
    )

    if assertion.msg is not None:
        wrapped += f", {ast.unparse(assertion.msg)}"

    return wrapped


def build_agent_docstring(
    *,
    description: str,
    fallback_docstring: str,
    public_tests: Sequence[str],
    test_imports: Sequence[str],
) -> str:
    description = description.strip() or fallback_docstring.strip()

    sections: list[str] = []

    if description:
        sections.append(description)

    if test_imports:
        sections.append(
            "Imports used by the original MBPP test environment:\n"
            + "\n".join(test_imports)
        )

    if public_tests:
        sections.append(
            "Original MBPP visible tests:\n"
            + "\n".join(public_tests)
        )

    return "\n\n".join(sections).strip()


def parse_mbpp_problem(
    task_id: str,
    plus_problem: Mapping[str, Any],
    original_problem: Mapping[str, Any] | None,
    *,
    visible_tests_mode: str,
    max_visible_tests: int,
) -> ProblemSpec:
    """
    Convert one EvalPlus MBPP+ record into the state expected by the custom
    problem-solving agent.

    MBPP+ ``prompt`` values are natural-language task descriptions, unlike
    HumanEval prompts, which contain Python function stubs. Therefore, the
    signature is extracted from the public sanitized-MBPP ``code`` field and
    only the signature is retained. The reference implementation body is never
    supplied to the agent.

    Generation input contains only:
      * the curated MBPP+ natural-language prompt;
      * a derived target function signature;
      * original public MBPP assertions when visible tests are enabled;
      * public test-environment imports.

    It never supplies canonical_solution, base_input, plus_input, contracts,
    hidden tests, or evaluator output to the agent.
    """
    original_problem = original_problem or {}

    entry_point = str(plus_problem.get("entry_point") or "").strip()
    if not entry_point:
        raise ValueError(f"MBPP+ task {task_id} has no entry_point.")

    plus_description = str(plus_problem.get("prompt") or "").strip()
    original_description = str(original_problem.get("prompt") or "").strip()
    description = plus_description or original_description

    test_imports = normalize_test_imports(
        original_problem.get("test_imports")
    )
    public_tests = normalize_public_tests(
        original_problem.get("test_list")
    )

    function_signature = derive_mbpp_function_signature(
        entry_point=entry_point,
        original_problem=original_problem,
        public_tests=public_tests,
    )

    if visible_tests_mode == "none":
        selected_public_tests: list[str] = []
    else:
        selected_public_tests = public_tests[:max_visible_tests]

    internal_tests = [
        add_imports_to_assert(test, test_imports)
        for test in selected_public_tests
    ]

    agent_docstring = build_agent_docstring(
        description=description,
        fallback_docstring="",
        public_tests=selected_public_tests,
        test_imports=test_imports,
    )

    return ProblemSpec(
        task_id=str(task_id),
        entry_point=entry_point,
        evalplus_prompt=description,
        function_signature=function_signature,
        docstring=agent_docstring,
        preamble="",
        public_tests=tuple(selected_public_tests),
        internal_tests=tuple(internal_tests),
        test_imports=tuple(test_imports),
    )


def strip_markdown_fence(text: str) -> str:
    value = text.strip()
    fenced = re.fullmatch(
        r"```(?:python|py)?\s*\n(?P<code>.*)\n```",
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
            re.search(
                rf"^\s*def\s+{re.escape(entry_point)}\s*\(",
                code,
                re.MULTILINE,
            )
        )

    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == entry_point
        for node in tree.body
    )


def build_self_contained_solution(spec: ProblemSpec, generated: str) -> str:
    """
    Convert the agent output into a standalone MBPP submission.

    MBPP prompts are natural language, so they must never be prepended to code.
    The normal path accepts a complete function definition. A conservative
    fallback also supports a function-body completion by indenting it beneath
    the derived signature.
    """
    generated = strip_markdown_fence(generated)

    if not generated:
        return make_failure_solution(
            spec,
            "The agent returned an empty solution.",
        )

    if contains_entry_function(generated, spec.entry_point):
        return generated.strip() + "\n"

    # Support body-only output without appending the natural-language prompt.
    indented_body = "\n".join(
        f"    {line}" if line.strip() else ""
        for line in generated.splitlines()
    )
    wrapped = f"{spec.function_signature}\n{indented_body}\n"

    if syntax_is_valid(wrapped):
        return wrapped

    return make_failure_solution(
        spec,
        "The agent output did not define the required entry-point function.",
    )


def make_failure_solution(spec: ProblemSpec, reason: str) -> str:
    safe_reason = reason.replace("\\", "\\\\").replace('"', '\\"')
    pieces: list[str] = []

    if spec.preamble:
        pieces.append(spec.preamble)

    pieces.append(
        f"{spec.function_signature}\n"
        f"    raise NotImplementedError(\"{safe_reason[:500]}\")"
    )

    return "\n\n".join(pieces) + "\n"


def syntax_is_valid(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except (SyntaxError, ValueError):
        return False


def invoke_agent(
    agent: Any,
    spec: ProblemSpec,
    *,
    recursion_limit: int,
    run_id: str,
) -> Mapping[str, Any]:
    state = {
        "function_signature": spec.function_signature,
        "docstring": spec.docstring,
        "test_cases": list(spec.internal_tests),
        "task_id": spec.task_id,
    }

    safe_task_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", spec.task_id)
    config = {
        "recursion_limit": recursion_limit,
        "configurable": {
            "user_id": "mbpp_plus_benchmark",
            "run_name": "mbpp_plus_benchmark",
            "thread_id": (
                f"mbpp:{run_id}:{safe_task_id}:{uuid.uuid4().hex}"
            ),
        },
    }

    response = agent.invoke(state, config=config)

    if not isinstance(response, Mapping):
        raise TypeError(
            f"Agent returned {type(response).__name__}; expected a mapping."
        )

    return response


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
        return [item for item in items if item[0] in selected_task_ids]

    items = items[start_index:]

    if limit is not None:
        items = items[:limit]

    return items


def stream_subprocess(
    command: Sequence[str],
    log_path: Path,
    *,
    env: Mapping[str, str] | None = None,
) -> int:
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
            env=dict(env) if env is not None else None,
        )

        assert process.stdout is not None

        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
            log_file.flush()

        return process.wait()


def json_fallback(value: Any) -> Any:
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    if isinstance(value, complex):
        return str(value)

    item_method = getattr(value, "item", None)
    if callable(item_method):
        return item_method()

    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def serialize_mbpp_problem_for_override(
    task_id: str,
    problem: Mapping[str, Any],
    mbpp_serialize_inputs: Any,
) -> dict[str, Any]:
    record = {
        key: value
        for key, value in problem.items()
        if not str(key).startswith("_")
    }

    record["task_id"] = str(task_id)

    for input_key in ("base_input", "plus_input"):
        if input_key in record:
            record[input_key] = mbpp_serialize_inputs(
                str(task_id),
                record[input_key],
            )

    return record


def write_mbpp_subset_override(
    path: Path,
    task_ids: Sequence[str],
    dataset: Mapping[str, Mapping[str, Any]],
    mbpp_serialize_inputs: Any,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    ordered_task_ids = sorted(task_ids, key=task_sort_key)

    with gzip.open(path, "wt", encoding="utf-8") as file:
        for task_id in ordered_task_ids:
            if task_id not in dataset:
                raise ValueError(
                    f"Cannot create MBPP override: unknown task {task_id}."
                )

            record = serialize_mbpp_problem_for_override(
                task_id,
                dataset[task_id],
                mbpp_serialize_inputs,
            )

            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    default=json_fallback,
                )
                + "\n"
            )


def run_evalplus_evaluation(
    *,
    samples_path: Path,
    output_dir: Path,
    mode: str,
    docker_image: str,
    parallel: int | None,
    base_only: bool,
    test_details: bool,
    subset_override_path: Path | None,
) -> int:
    common_args = [
        "--dataset",
        "mbpp",
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
        ]

        if subset_override_path is not None:
            command.extend(
                [
                    "-e",
                    f"MBPP_OVERRIDE_PATH=/app/{subset_override_path.name}",
                ]
            )

        command.extend(
            [
                docker_image,
                "evalplus.evaluate",
                *common_args,
                f"/app/{samples_path.name}",
                *extra_args,
            ]
        )

        return stream_subprocess(
            command,
            output_dir / "evaluation_output.txt",
        )

    if mode == "local":
        command = [
            sys.executable,
            "-m",
            "evalplus.evaluate",
            *common_args,
            str(samples_path.resolve()),
            *extra_args,
        ]

        environment = os.environ.copy()

        if subset_override_path is not None:
            environment["MBPP_OVERRIDE_PATH"] = str(
                subset_override_path.resolve()
            )

        return stream_subprocess(
            command,
            output_dir / "evaluation_output.txt",
            env=environment,
        )

    return 0


def prepare_docker_runtime(
    *,
    images: Sequence[str],
    warmup_image: str = DEFAULT_SANDBOX_IMAGE,
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

    unique_images = list(
        dict.fromkeys(
            image.strip()
            for image in images
            if image and image.strip()
        )
    )

    for image in unique_images:
        inspection = subprocess.run(
            [docker_executable, "image", "inspect", image],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )

        if inspection.returncode == 0:
            print(f"Docker image is available: {image}")
            continue

        print(f"Pulling Docker image: {image}")

        try:
            pull = subprocess.run(
                [docker_executable, "pull", image],
                timeout=pull_timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"Timed out while pulling Docker image '{image}' after "
                f"{pull_timeout_seconds} seconds."
            ) from exc

        if pull.returncode != 0:
            raise RuntimeError(
                f"Could not pull Docker image '{image}'. "
                f"Docker exited with code {pull.returncode}."
            )

    if warmup_image:
        print(f"Starting disposable Docker warm-up: {warmup_image}")

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
            "Generate one final solution per curated MBPP task with a custom "
            "LangGraph agent, then evaluate MBPP Base and MBPP+ with EvalPlus."
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
            "Result directory. Default: "
            "benchmark_results/mbpp_plus_<UTC timestamp>."
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
        help="Zero-based first index in the curated MBPP+ task set.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Maximum number of tasks. Omit for the complete current MBPP+ "
            "task set. Partial runs are evaluated through MBPP_OVERRIDE_PATH."
        ),
    )
    parser.add_argument(
        "--task-id",
        action="append",
        default=[],
        help=(
            "Run one specific task, for example Mbpp/2. Repeat for multiple "
            "tasks."
        ),
    )
    parser.add_argument(
        "--visible-tests",
        choices=("original", "none"),
        default="original",
        help=(
            "'original' gives the agent the public original MBPP test_list. "
            "Hidden EvalPlus inputs are never exposed."
        ),
    )
    parser.add_argument(
        "--max-visible-tests",
        type=int,
        default=20,
        help="Maximum original MBPP assertions supplied per task.",
    )
    parser.add_argument(
        "--recursion-limit",
        type=int,
        default=100,
        help="LangGraph recursion limit for each task.",
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
        help="Docker is recommended because generated code is untrusted.",
    )
    parser.add_argument(
        "--docker-image",
        default=DEFAULT_DOCKER_IMAGE,
        help="EvalPlus Docker image.",
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
            "Evaluate only the original MBPP tests for the curated task set. "
            "By default EvalPlus reports Base and Base + Extra (MBPP+)."
        ),
    )
    parser.add_argument(
        "--test-details",
        action="store_true",
        help="Retain detailed EvalPlus test information.",
    )
    parser.add_argument(
        "--skip-docker-warmup",
        action="store_true",
        help="Do not pre-connect to Docker or pre-pull images.",
    )

    return parser


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


def main() -> int:
    args = build_parser().parse_args()

    if args.start_index < 0:
        raise ValueError("--start-index must be at least 0.")

    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be greater than 0.")

    if args.max_visible_tests < 0:
        raise ValueError("--max-visible-tests must be at least 0.")

    try:
        # EvalPlus 0.3.1 exports get_mbpp_plus from evalplus.data, while
        # get_mbpp and mbpp_serialize_inputs remain in evalplus.data.mbpp.
        from evalplus.data import get_mbpp_plus
        from evalplus.data.mbpp import get_mbpp, mbpp_serialize_inputs
    except ImportError as exc:
        raise RuntimeError(
            "EvalPlus is installed, but the required MBPP data API could not "
            "be imported. Expected EvalPlus 0.3.1-compatible imports: "
            "'get_mbpp_plus' from evalplus.data and 'get_mbpp' plus "
            "'mbpp_serialize_inputs' from evalplus.data.mbpp."
        ) from exc

    needs_internal_docker = args.visible_tests == "original"
    needs_eval_docker = args.evaluation_mode == "docker"

    if (
        not args.skip_docker_warmup
        and (needs_internal_docker or needs_eval_docker)
    ):
        images = [DEFAULT_SANDBOX_IMAGE]
        if needs_eval_docker:
            images.append(args.docker_image)

        prepare_docker_runtime(
            images=images,
            warmup_image=DEFAULT_SANDBOX_IMAGE,
            startup_timeout_seconds=180,
            pull_timeout_seconds=900,
            retry_interval_seconds=2,
        )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or Path(
        "benchmark_results",
        f"mbpp_plus_{timestamp}",
    )
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    samples_path = output_dir / "samples.jsonl"
    log_path = output_dir / "generation_log.jsonl"
    summary_path = output_dir / "generation_summary.json"
    config_path = output_dir / "run_config.json"
    subset_override_path = output_dir / "mbpp_subset_override.jsonl.gz"

    if (
        not args.resume
        and samples_path.exists()
        and samples_path.stat().st_size > 0
    ):
        raise FileExistsError(
            f"{samples_path} already exists. Use --resume or another "
            "--output-dir."
        )

    print(f"Loading MBPP+ into {output_dir}")
    plus_dataset = get_mbpp_plus()
    original_mbpp = get_mbpp()

    dataset_task_count = len(plus_dataset)
    tasks = choose_tasks(
        plus_dataset,
        start_index=args.start_index,
        limit=args.limit,
        selected_task_ids=set(args.task_id),
    )

    selected_task_ids = {task_id for task_id, _ in tasks}
    full_dataset_selected = selected_task_ids == set(plus_dataset.keys())
    run_id = uuid.uuid4().hex

    run_config = {
        "created_at": utc_now_iso(),
        "run_id": run_id,
        "dataset": "MBPP / MBPP+",
        "evalplus_dataset_argument": "mbpp",
        "dataset_task_count": dataset_task_count,
        "selected_task_count": len(tasks),
        "full_dataset_selected": full_dataset_selected,
        "agent_module": args.agent_module,
        "agent_object": args.agent_object,
        "visible_tests": args.visible_tests,
        "max_visible_tests": args.max_visible_tests,
        "recursion_limit": args.recursion_limit,
        "start_index": args.start_index,
        "limit": args.limit,
        "selected_task_ids": args.task_id,
        "evaluation_mode": args.evaluation_mode,
        "base_only": args.base_only,
        "methodology_note": (
            "The agent receives the curated MBPP task description, EvalPlus's "
            "standardized function signature, and optionally the original "
            "public MBPP test_list. It never receives canonical_solution, "
            "base_input, plus_input, contracts, or evaluator outputs. EvalPlus "
            "reports Base for the original MBPP tests on the curated task set "
            "and Base + Extra for MBPP+."
        ),
    }
    write_json(config_path, run_config)

    existing_rows = read_jsonl(samples_path) if args.resume else []
    completed_task_ids = {
        str(row.get("task_id"))
        for row in existing_rows
        if row.get("task_id") is not None
    }

    pending_tasks = [
        item for item in tasks if item[0] not in completed_task_ids
    ]

    print(
        f"Selected {len(tasks)} task(s); "
        f"{len(pending_tasks)} remain after resume."
    )

    agent = load_agent(args.agent_module, args.agent_object)
    sorted_dataset_ids = sorted(plus_dataset.keys(), key=task_sort_key)
    dataset_indices = {
        task_id: index for index, task_id in enumerate(sorted_dataset_ids)
    }

    started = time.perf_counter()
    generation_records: list[GenerationRecord] = []
    agent_errors = 0

    for position, (task_id, plus_problem) in enumerate(
        pending_tasks,
        start=1,
    ):
        absolute_index = dataset_indices[task_id]
        print(
            f"[{position}/{len(pending_tasks)}] {task_id} "
            f"(curated index {absolute_index})"
        )

        task_started = time.perf_counter()
        original_problem = original_mbpp.get(task_numeric_id(task_id))

        try:
            spec = parse_mbpp_problem(
                task_id,
                plus_problem,
                original_problem,
                visible_tests_mode=args.visible_tests,
                max_visible_tests=args.max_visible_tests,
            )

            response = invoke_agent(
                agent,
                spec,
                recursion_limit=args.recursion_limit,
                run_id=run_id,
            )

            generated = str(response.get("final_solution") or "")
            solution = build_self_contained_solution(spec, generated)
            status = "ok"
            error = None

        except Exception as exc:
            agent_errors += 1
            error = f"{type(exc).__name__}: {exc}"
            status = "agent_error"
            response = {}
            traceback.print_exc()

            try:
                fallback_spec = parse_mbpp_problem(
                    task_id,
                    plus_problem,
                    original_problem,
                    visible_tests_mode="none",
                    max_visible_tests=0,
                )
                solution = make_failure_solution(fallback_spec, error)
                spec = fallback_spec
            except Exception:
                solution = (
                    "def mbpp_generation_failure(*args, **kwargs):\n"
                    f"    raise NotImplementedError({error!r})\n"
                )
                spec = ProblemSpec(
                    task_id=str(task_id),
                    entry_point=str(plus_problem.get("entry_point", "")),
                    evalplus_prompt=str(plus_problem.get("prompt", "")),
                    function_signature="",
                    docstring="",
                    preamble="",
                    public_tests=(),
                    internal_tests=(),
                    test_imports=(),
                )

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
                3,
            ),
            visible_test_count=len(spec.public_tests),
            test_import_count=len(spec.test_imports),
            final_solution_present=bool(response.get("final_solution")),
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
            f"visible_tests={record.visible_test_count}, "
            f"elapsed={record.elapsed_seconds:.3f}s"
        )

        if error and args.stop_on_error:
            break

    all_log_rows = read_jsonl(log_path)
    all_sample_rows = read_jsonl(samples_path)
    elapsed_total = round(time.perf_counter() - started, 3)

    sample_task_ids = [
        str(row.get("task_id"))
        for row in all_sample_rows
        if row.get("task_id") is not None
    ]
    unique_sample_task_ids = set(sample_task_ids)

    unknown_sample_ids = unique_sample_task_ids.difference(
        plus_dataset.keys()
    )
    if unknown_sample_ids:
        raise ValueError(
            f"samples.jsonl contains unknown MBPP task IDs: "
            f"{sorted(unknown_sample_ids)}"
        )

    full_dataset_complete = unique_sample_task_ids == set(
        plus_dataset.keys()
    )

    override_used = False

    if args.evaluation_mode != "none" and not full_dataset_complete:
        if not unique_sample_task_ids:
            raise RuntimeError("No generated MBPP samples are available to evaluate.")

        write_mbpp_subset_override(
            subset_override_path,
            sorted(unique_sample_task_ids, key=task_sort_key),
            plus_dataset,
            mbpp_serialize_inputs,
        )
        override_used = True

    summary = {
        "finished_at": utc_now_iso(),
        "dataset": "MBPP / MBPP+",
        "dataset_task_count": dataset_task_count,
        "selected_tasks": len(tasks),
        "previously_completed_tasks": len(
            completed_task_ids.intersection(selected_task_ids)
        ),
        "generated_this_run": len(generation_records),
        "samples_in_file": len(all_sample_rows),
        "unique_tasks_in_samples": len(unique_sample_task_ids),
        "full_dataset_complete": full_dataset_complete,
        "subset_override_used": override_used,
        "subset_override_path": (
            str(subset_override_path) if override_used else None
        ),
        "agent_errors_this_run": agent_errors,
        "syntax_invalid_this_run": sum(
            1 for record in generation_records if not record.syntax_valid
        ),
        "elapsed_seconds_this_run": elapsed_total,
        "status_counts_all_logs": _count_values(
            str(row.get("status", "unknown"))
            for row in all_log_rows
        ),
        "samples_path": str(samples_path),
        "evaluation_requested": args.evaluation_mode != "none",
        "evaluation_performed": False,
        "reported_scores": (
            ["Base"] if args.base_only else ["Base", "Base + Extra"]
        ),
    }

    print(f"\nGeneration complete. Samples: {samples_path}")

    if args.evaluation_mode == "none":
        write_json(summary_path, summary)
        print(f"Generation summary: {summary_path}")
        return 0

    evaluation_code = run_evalplus_evaluation(
        samples_path=samples_path,
        output_dir=output_dir,
        mode=args.evaluation_mode,
        docker_image=args.docker_image,
        parallel=args.parallel,
        base_only=args.base_only,
        test_details=args.test_details,
        subset_override_path=(
            subset_override_path if override_used else None
        ),
    )

    summary["evaluation_performed"] = True
    summary["evaluation_exit_code"] = evaluation_code
    write_json(summary_path, summary)

    print(f"Generation summary: {summary_path}")

    if evaluation_code != 0:
        print(
            f"EvalPlus exited with code {evaluation_code}. See "
            f"{output_dir / 'evaluation_output.txt'}",
            file=sys.stderr,
        )

    return evaluation_code


if __name__ == "__main__":
    raise SystemExit(main())


# Three-task smoke test, including subset evaluation:
# python .\run_mbpp_plus_benchmark.py `
#     --limit 3 `
#     --output-dir .\benchmark_results\mbpp\mbpp_plus_test `
#     --evaluation-mode docker

# One task:
# python .\run_mbpp_plus_benchmark.py `
#     --task-id Mbpp/2 `
#     --output-dir .\benchmark_results\mbpp\mbpp_task_2 `
#     --evaluation-mode docker

# Complete curated MBPP / MBPP+ run:
# python .\run_mbpp_plus_benchmark.py `
#     --output-dir .\benchmark_results\mbpp\mbpp_plus_full `
#     --evaluation-mode docker

# Resume:
# python .\run_mbpp_plus_benchmark.py `
#     --output-dir .\benchmark_results\mbpp\mbpp_plus_full `
#     --resume `
#     --evaluation-mode docker