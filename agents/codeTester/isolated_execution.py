from typing import Any, Dict, List, Optional
from pathlib import Path
import subprocess
import tempfile
import shutil
import uuid
import json
import ast
import os


def _normalise_import_statement(import_statement: str) -> str:
    '''Validates and normalises one Python import statement.'''
    try:
        parsed = ast.parse(import_statement.strip())
    except SyntaxError as exc:
        raise ValueError(f'Invalid import statement: {import_statement!r}') from exc

    if len(parsed.body) != 1:
        raise ValueError('Each import entry must contain exactly one import statement.')

    node = parsed.body[0]
    if not isinstance(node, (ast.Import, ast.ImportFrom)):
        raise ValueError(f'Expected an import statement, received: {import_statement!r}')

    return ast.unparse(node).strip()


def _extract_source_imports(source_code: str) -> List[str]:
    '''Extracts top-level imports from the original source file.'''
    try:
        parsed_source = ast.parse(source_code)
    except SyntaxError as exc:
        raise ValueError('The original source file could not be parsed while extracting imports.') from exc

    imports: List[str] = []
    for node in parsed_source.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(ast.unparse(node).strip())

    return imports


def _merge_imports(*, source_code: str, additional_imports: Optional[List[str]]) -> List[str]:
    '''Merges source imports and Coder imports without duplicates.'''
    source_imports = _extract_source_imports(source_code)
    all_imports = source_imports + list(additional_imports or [])
    future_imports: List[str] = []
    normal_imports: List[str] = []
    seen: set[str] = set()

    for import_statement in all_imports:
        if not import_statement:
            continue

        normalised = _normalise_import_statement(import_statement)
        if normalised in seen:
            continue

        seen.add(normalised)
        if normalised.startswith('from __future__ import '):
            future_imports.append(normalised)
        else:
            normal_imports.append(normalised)

    return future_imports + normal_imports


def _build_candidate_code(*, source_code: str, additional_imports: Optional[List[str]], implementation: str) -> str:
    '''Creates the code executed inside the isolated environment.'''
    merged_imports = _merge_imports(source_code= source_code, additional_imports= additional_imports)
    code_parts = [*merged_imports, implementation.strip()]
    return '\n\n'.join(part for part in code_parts if part)


def _failed_result(
    kwargs: Dict[str, Any],
    *,
    error_type: str,
    error_message: str,
    traceback_text: Optional[str] = None,
    execution_time_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    '''Creates one failed function-execution result.'''
    return {
        'kwargs': kwargs,
        'completed': False,
        'output': None,
        'output_type': None,
        'error_type': error_type,
        'error_message': error_message,
        'traceback': traceback_text,
        'execution_time_seconds': execution_time_seconds,
    }


runner_code = r'''
import asyncio
import inspect
import json
import math
import os
import time
import traceback


original_stdout_fd = os.dup(1)


def emit(payload):
    # Emit the final result through the original stdout file descriptor.
    data = json.dumps(payload, ensure_ascii= False)
    os.write(original_stdout_fd, data.encode('utf-8') + b'\n')


def json_safe(value):
    # Convert common Python values into JSON-compatible values.
    if value is None or isinstance(value, (bool, int, str)):
        return value

    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)

    if isinstance(value, bytes):
        return value.decode('utf-8', errors= 'replace')

    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]

    model_dump = getattr(value, 'model_dump', None)
    if callable(model_dump):
        try:
            return json_safe(model_dump())
        except Exception:
            pass

    return repr(value)


try:
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_fd, 1)
    os.dup2(devnull_fd, 2)

    with open('/sandbox/payload.json', 'r', encoding= 'utf-8') as file:
        payload = json.load(file)

    function_name = payload['function_name']
    candidate_code = payload['candidate_code']
    kwargs = payload['kwargs']
    namespace = {'__name__': '__submitted_solution__'}

    try:
        compiled_code = compile(candidate_code, '/sandbox/submitted_solution.py', 'exec')
        exec(compiled_code, namespace)
        target_function = namespace.get(function_name)

        if target_function is None:
            raise ValueError(f'Function {function_name!r} was not found after executing the implementation.')

        invoke_method = getattr(target_function, 'invoke', None)
        if not callable(target_function) and not callable(invoke_method):
            raise TypeError(f'Object {function_name!r} is not callable.')

    except BaseException as exc:
        emit({
            'kwargs': kwargs,
            'completed': False,
            'output': None,
            'output_type': None,
            'error_type': type(exc).__name__,
            'error_message': str(exc),
            'traceback': traceback.format_exc(),
            'execution_time_seconds': None,
        })
        raise SystemExit(0)

    started_at = time.perf_counter()

    try:
        invoke_method = getattr(target_function, 'invoke', None)

        if callable(invoke_method) and not inspect.isfunction(target_function):
            output = invoke_method(kwargs)
        else:
            output = target_function(**kwargs)

        if inspect.isawaitable(output):
            output = asyncio.run(output)

        elapsed = time.perf_counter() - started_at
        emit({
            'kwargs': kwargs,
            'completed': True,
            'output': json_safe(output),
            'output_type': type(output).__name__,
            'error_type': None,
            'error_message': None,
            'traceback': None,
            'execution_time_seconds': elapsed,
        })

    except BaseException as exc:
        elapsed = time.perf_counter() - started_at
        emit({
            'kwargs': kwargs,
            'completed': False,
            'output': None,
            'output_type': None,
            'error_type': type(exc).__name__,
            'error_message': str(exc),
            'traceback': traceback.format_exc(),
            'execution_time_seconds': elapsed,
        })

except BaseException as exc:
    emit({
        'kwargs': {},
        'completed': False,
        'output': None,
        'output_type': None,
        'error_type': type(exc).__name__,
        'error_message': str(exc),
        'traceback': traceback.format_exc(),
        'execution_time_seconds': None,
    })
'''


def _remove_container(container_name: str) -> None:
    '''Forcefully removes a Docker container without raising an exception.'''
    try:
        subprocess.run(
            ['docker', 'rm', '-f', container_name],
            stdout= subprocess.DEVNULL,
            stderr= subprocess.DEVNULL,
            timeout= 3,
            check= False,
        )
    except Exception:
        pass


def _run_single_input(
    *,
    function_name: str,
    candidate_code: str,
    kwargs: Dict[str, Any],
    utils_path: Path,
    agents_path: Path,
    timeout_seconds: int,
    memory_limit: str,
    cpus: str,
    pids_limit: int,
    docker_image: str,
) -> Dict[str, Any]:
    '''Executes one function input inside a separate Docker container.'''
    payload = {'function_name': function_name, 'candidate_code': candidate_code, 'kwargs': kwargs}

    try:
        payload_json = json.dumps(payload, ensure_ascii= False)
    except TypeError as exc:
        return _failed_result(kwargs, error_type= type(exc).__name__, error_message= f'Input is not JSON-serializable: {exc}')

    container_name = f'code-tester-{uuid.uuid4().hex}'

    with tempfile.TemporaryDirectory(prefix= 'code_tester_') as temp_dir:
        temp_path = Path(temp_dir)
        runner_path = temp_path / 'runner.py'
        payload_path = temp_path / 'payload.json'
        runner_path.write_text(runner_code, encoding= 'utf-8')
        payload_path.write_text(payload_json, encoding= 'utf-8')

        try:
            os.chmod(temp_path, 0o755)
            os.chmod(runner_path, 0o644)
            os.chmod(payload_path, 0o644)
        except OSError:
            pass

        resolved_utils_path = Path(utils_path).resolve()
        resolved_agents_path = Path(agents_path).resolve()

        docker_command = [
            'docker',
            'run',
            '--rm',
            '--name',
            container_name,
            '--init',
            '--network',
            'none',
            '--memory',
            memory_limit,
            '--memory-swap',
            memory_limit,
            '--cpus',
            cpus,
            '--pids-limit',
            str(pids_limit),
            '--cap-drop',
            'ALL',
            '--security-opt',
            'no-new-privileges:true',
            '--user',
            '65534:65534',
            '--read-only',
            '--tmpfs',
            '/tmp:rw,noexec,nosuid,size=64m',
            '--mount',
            f'type=bind,source={temp_path.resolve()},target=/sandbox,readonly',
            '--mount',
            f'type=bind,source={resolved_utils_path},target=/project/utils,readonly',
            '--mount',
            f'type=bind,source={resolved_agents_path},target=/project/agents,readonly',
            '--workdir',
            '/sandbox',
            '-e',
            'PYTHONDONTWRITEBYTECODE=1',
            '-e',
            'HOME=/tmp',
            '-e',
            'PYTHONPATH=/project',
            docker_image,
            'python',
            '/sandbox/runner.py',
        ]

        try:
            process = subprocess.Popen(docker_command, stdout= subprocess.PIPE, stderr= subprocess.PIPE, text= True)
            stdout, stderr = process.communicate(timeout= timeout_seconds)
        except subprocess.TimeoutExpired:
            _remove_container(container_name)
            return _failed_result(
                kwargs,
                error_type= 'TimeoutError',
                error_message= f'Execution exceeded {timeout_seconds} seconds.',
                execution_time_seconds= float(timeout_seconds),
            )
        except Exception as exc:
            _remove_container(container_name)
            return _failed_result(
                kwargs,
                error_type= type(exc).__name__,
                error_message= f'Failed to run the isolated environment: {exc}',
            )

        stdout = (stdout or '').strip()
        stderr = (stderr or '').strip()

        if not stdout:
            return _failed_result(
                kwargs,
                error_type= 'SandboxExecutionError',
                error_message= (
                    f'The sandbox returned no JSON result. Exit code: {process.returncode}.'
                    + (f' Stderr: {stderr[:2000]}' if stderr else '')
                ),
            )

        try:
            result = json.loads(stdout.splitlines()[-1])
        except json.JSONDecodeError as exc:
            return _failed_result(
                kwargs,
                error_type= type(exc).__name__,
                error_message= f'The sandbox returned invalid JSON. Stdout: {stdout[:2000]}. Stderr: {stderr[:2000]}.',
            )

        result['kwargs'] = kwargs
        return result


def run_in_isolated_env(
    *,
    function_name: str,
    source_code: str,
    implementation: str,
    imports: Optional[List[str]],
    function_inputs: List[Dict[str, Any]],
    utils_path: str,
    agents_path: str,
    timeout_seconds: int = 8,
    memory_limit: str = '256m',
    cpus: str = '1.0',
    pids_limit: int = 64,
    docker_image: str = 'thesis-code-tester',
) -> Dict[str, Any]:
    '''Executes the candidate function once for every kwargs input.'''
    if shutil.which('docker') is None:
        raise RuntimeError('Docker is not installed or is not available on PATH.')

    resolved_utils_path = Path(utils_path).resolve()
    resolved_agents_path = Path(agents_path).resolve()

    if not resolved_utils_path.is_dir():
        raise FileNotFoundError(f'Utils directory not found: {resolved_utils_path}')

    if not resolved_agents_path.is_dir():
        raise FileNotFoundError(f'Agents directory not found: {resolved_agents_path}')

    try:
        image_check = subprocess.run(
            ['docker', 'image', 'inspect', docker_image],
            stdout= subprocess.DEVNULL,
            stderr= subprocess.DEVNULL,
            check= False,
            timeout= 10,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f'Docker did not respond while checking the image: {docker_image}.') from exc
    except Exception as exc:
        raise RuntimeError(f'Failed to inspect Docker image {docker_image}: {exc}') from exc

    if image_check.returncode != 0:
        raise RuntimeError(f'Docker image not found: {docker_image}. Build the Code Tester image before running.')

    candidate_code = _build_candidate_code(
        source_code= source_code,
        additional_imports= imports,
        implementation= implementation,
    )

    results: List[Dict[str, Any]] = []

    for function_input in function_inputs:
        if not isinstance(function_input, dict):
            results.append(
                _failed_result(
                    {},
                    error_type= 'InvalidInputError',
                    error_message= 'Every generated function input must be a dictionary containing keyword arguments.',
                )
            )
            continue

        result = _run_single_input(
            function_name= function_name,
            candidate_code= candidate_code,
            kwargs= function_input,
            utils_path= resolved_utils_path,
            agents_path= resolved_agents_path,
            timeout_seconds= timeout_seconds,
            memory_limit= memory_limit,
            cpus= cpus,
            pids_limit= pids_limit,
            docker_image= docker_image,
        )
        results.append(result)

    completed_executions = sum(1 for result in results if result['completed'])
    failed_executions = len(results) - completed_executions

    return {
        'function_name': function_name,
        'total_inputs': len(function_inputs),
        'completed_executions': completed_executions,
        'failed_executions': failed_executions,
        'results': results,
    }