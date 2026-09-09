from typing import Any, Dict, List, Optional
from pathlib import Path
import subprocess
import tempfile
import hashlib
import shutil
import uuid
import json
import ast
import re
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

def _normalise_pip_packages(packages: List[str]) -> List[str]:
    '''Validates and normalises PyPI distribution names.'''
    package_pattern = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]*(?:\[[A-Za-z0-9,._-]+\])?$')

    normalised_packages: List[str] = []

    for package in packages:
        package = package.strip()

        if not package:
            continue

        if not package_pattern.fullmatch(package):
            raise ValueError(f'Invalid pip package name: {package!r}')

        if package not in normalised_packages:
            normalised_packages.append(package)

    return sorted(normalised_packages)


def _ensure_dependency_image(
    *,
    base_image: str,
    packages: List[str],
) -> str:
    '''Returns a cached Docker image containing the requested PyPI packages.'''
    packages = _normalise_pip_packages(packages)

    if not packages:
        return base_image

    dependency_string = f'{base_image}|{"|".join(packages)}'
    dependency_hash = hashlib.sha256(dependency_string.encode('utf-8')).hexdigest()[:12]
    dependency_image = f'thesis-code-tester:deps-{dependency_hash}'

    image_check = subprocess.run(
        ['docker', 'image', 'inspect', dependency_image],
        stdout= subprocess.DEVNULL,
        stderr= subprocess.DEVNULL,
        check= False,
        timeout= 10,
    )

    if image_check.returncode == 0:
        return dependency_image

    packages_string = ' '.join(packages)

    dockerfile = (
        '# syntax=docker/dockerfile:1.7\n'
        f'FROM {base_image}\n\n'
        'RUN --mount=type=cache,target=/root/.cache/pip '
        f'python -m pip install --disable-pip-version-check {packages_string}\n'
    )

    with tempfile.TemporaryDirectory(prefix= 'code_tester_image_') as temp_dir:
        temp_path = Path(temp_dir)
        dockerfile_path = temp_path / 'Dockerfile'
        dockerfile_path.write_text(dockerfile, encoding= 'utf-8')

        build = subprocess.run(
            [
                'docker',
                'build',
                '-t',
                dependency_image,
                str(temp_path),
            ],
            stdout= subprocess.PIPE,
            stderr= subprocess.PIPE,
            text= True,
            check= False,
        )

    if build.returncode != 0:
        raise RuntimeError(
            f'Failed to install required packages {packages}. '
            f'Docker build error: {build.stderr[-3000:]}'
        )

    return dependency_image

def _find_qualified_function_node(parsed_source: ast.Module, function_name: str) -> ast.AST:
    '''Finds one top-level function or qualified class method.'''
    parts: List[str] = function_name.split('.')
    body = parsed_source.body

    for class_name in parts[:-1]:
        matching_classes = [node for node in body if isinstance(node, ast.ClassDef) and node.name == class_name]

        if len(matching_classes) != 1:
            raise ValueError(f'Expected exactly one class named {class_name!r} while resolving {function_name!r}, but found {len(matching_classes)}.')

        body = matching_classes[0].body

    code_function_name: str = parts[-1]
    matching_functions = [node for node in body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == code_function_name]

    if len(matching_functions) != 1:
        raise ValueError(f'Expected exactly one function or method named {function_name!r}, but found {len(matching_functions)}.')

    return matching_functions[0]

def _replace_function_in_source(*, function_name: str, source_code: str, implementation: str) -> str:
    '''Replaces exactly one top-level function or qualified class method with the candidate implementation.'''
    try:
        parsed_source = ast.parse(source_code)
        parsed_implementation = ast.parse(implementation)
    except SyntaxError as exc:
        raise ValueError(f'Unable to parse the source or candidate implementation: {exc}') from exc

    target_function = _find_qualified_function_node(parsed_source, function_name)
    code_function_name: str = function_name.split('.')[-1]

    implementation_functions = [node for node in parsed_implementation.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]

    if len(implementation_functions) != 1:
        raise ValueError('The candidate implementation must contain exactly one function or method definition.')

    if implementation_functions[0].name != code_function_name:
        raise ValueError(f'The candidate implements {implementation_functions[0].name!r}, but the expected function or method is {function_name!r}.')

    source_lines = source_code.splitlines(keepends=True)
    decorator_lines = [decorator.lineno for decorator in target_function.decorator_list]
    start_line = min([target_function.lineno, *decorator_lines]) - 1
    end_line = target_function.end_lineno

    definition_line = source_lines[target_function.lineno - 1]
    indentation = definition_line[:len(definition_line) - len(definition_line.lstrip())]

    candidate_lines = implementation.strip('\n').splitlines()
    candidate = '\n'.join([f'{indentation}{line}' if line.strip() else line for line in candidate_lines]) + '\n'

    updated_source = ''.join(source_lines[:start_line]) + candidate + ''.join(source_lines[end_line:])

    try:
        ast.parse(updated_source)
    except SyntaxError as exc:
        raise ValueError(f'Replacing {function_name!r} would make the candidate module invalid: {exc}') from exc

    return updated_source

def _insert_additional_imports(
    *,
    source_code: str,
    additional_imports: Optional[List[str]],
) -> str:
    '''Adds new imports after the module docstring and future imports.'''
    if not additional_imports:
        return source_code

    parsed_source = ast.parse(source_code)

    existing_imports = {
        ast.unparse(node).strip()
        for node in parsed_source.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    }

    new_imports: List[str] = []
    for import_statement in additional_imports:
        if not import_statement:
            continue

        normalised = _normalise_import_statement(import_statement)
        if normalised not in existing_imports and normalised not in new_imports:
            new_imports.append(normalised)

    if not new_imports:
        return source_code

    insertion_line = 0
    body_index = 0

    if (
        parsed_source.body
        and isinstance(parsed_source.body[0], ast.Expr)
        and isinstance(parsed_source.body[0].value, ast.Constant)
        and isinstance(parsed_source.body[0].value.value, str)
    ):
        insertion_line = parsed_source.body[0].end_lineno
        body_index = 1

    while body_index < len(parsed_source.body):
        node = parsed_source.body[body_index]

        if (
            isinstance(node, ast.ImportFrom)
            and node.module == '__future__'
        ):
            insertion_line = node.end_lineno
            body_index += 1
            continue

        break

    source_lines = source_code.splitlines(keepends= True)
    import_block = '\n'.join(new_imports) + '\n\n'

    return (
        ''.join(source_lines[:insertion_line])
        + import_block
        + ''.join(source_lines[insertion_line:])
    )

def _find_tool_description_errors(source_code: str) -> List[str]:
    '''Returns module-level and class @tool functions that have neither a docstring nor an explicit description.'''
    parsed_source = ast.parse(source_code)
    invalid_tools: List[str] = []

    def inspect_body(body: List[ast.stmt], prefix: str = '') -> None:
        for node in body:
            if isinstance(node, ast.ClassDef):
                inspect_body(node.body, f'{prefix}{node.name}.')
                continue

            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue

            for decorator in node.decorator_list:
                decorator_target = decorator.func if isinstance(decorator, ast.Call) else decorator
                decorator_name = ast.unparse(decorator_target).strip()

                if decorator_name != 'tool' and not decorator_name.endswith('.tool'):
                    continue

                has_description = isinstance(decorator, ast.Call) and any(keyword.arg == 'description' for keyword in decorator.keywords)
                has_docstring = ast.get_docstring(node) is not None

                if not has_description and not has_docstring:
                    invalid_tools.append(f'{prefix}{node.name}')

    inspect_body(parsed_source.body)
    return invalid_tools

def _build_candidate_code(
    *,
    function_name: str,
    source_code: str,
    additional_imports: Optional[List[str]],
    implementation: str,
) -> str:
    '''Creates the complete modified source module for isolated execution.'''
    candidate_code = _replace_function_in_source(
        function_name= function_name,
        source_code= source_code,
        implementation= implementation,
    )

    return _insert_additional_imports(
        source_code= candidate_code,
        additional_imports= additional_imports,
    )


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
    data = json.dumps(payload, ensure_ascii=False)
    os.write(original_stdout_fd, data.encode('utf-8') + b'\n')


def json_safe(value):
    if value is None or isinstance(value, (bool, int, str)):
        return value

    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)

    if isinstance(value, bytes):
        return value.decode('utf-8', errors='replace')

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


def resolve_target(namespace, function_name, kwargs):
    parts = function_name.split('.')
    call_kwargs = dict(kwargs)

    if len(parts) == 1:
        target = namespace.get(function_name)

        if target is None:
            raise ValueError(f'Function {function_name!r} was not found after executing the implementation.')

        return target, call_kwargs

    owner = namespace.get(parts[0])

    if owner is None:
        raise ValueError(f'Class or object {parts[0]!r} was not found while resolving {function_name!r}.')

    for part in parts[1:-1]:
        owner = getattr(owner, part)

    method_name = parts[-1]

    try:
        descriptor = inspect.getattr_static(owner, method_name)
    except AttributeError as exc:
        raise ValueError(f'Method {function_name!r} was not found after executing the implementation.') from exc

    if isinstance(descriptor, staticmethod):
        return getattr(owner, method_name), call_kwargs

    if isinstance(descriptor, classmethod):
        return getattr(owner, method_name), call_kwargs

    instance_kwargs = call_kwargs.pop('__instance__', None)

    if not isinstance(instance_kwargs, dict):
        raise ValueError(f'Instance method {function_name!r} requires a JSON-compatible "__instance__" dictionary containing constructor arguments for {parts[-2]!r}.')

    try:
        instance = owner(**instance_kwargs)
    except Exception as exc:
        raise ValueError(f'Could not construct an instance for {function_name!r} using __instance__: {exc}') from exc

    return getattr(instance, method_name), call_kwargs


try:
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_fd, 1)
    os.dup2(devnull_fd, 2)

    with open('/sandbox/payload.json', 'r', encoding='utf-8') as file:
        payload = json.load(file)

    function_name = payload['function_name']
    candidate_code = payload['candidate_code']
    source_file = payload['source_file']
    kwargs = payload['kwargs']
    namespace = {'__name__': '__submitted_solution__', '__file__': source_file, '__package__': None}

    try:
        compiled_code = compile(candidate_code, '/sandbox/submitted_solution.py', 'exec')
        exec(compiled_code, namespace)
        target_function, call_kwargs = resolve_target(namespace, function_name, kwargs)
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

        if callable(invoke_method) and not inspect.isfunction(target_function) and not inspect.ismethod(target_function):
            output = invoke_method(call_kwargs)
        else:
            output = target_function(**call_kwargs)

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

except Exception as exc:
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
    source_file: str,
    kwargs: Dict[str, Any],
    utils_path: Path,
    agents_path: Path,
    creations_path: Path,
    timeout_seconds: int,
    memory_limit: str,
    cpus: str,
    pids_limit: int,
    docker_image: str,
) -> Dict[str, Any]:
    '''Executes one function input inside a separate Docker container.'''
    payload = {'function_name': function_name, 'candidate_code': candidate_code, 'source_file': source_file, 'kwargs': kwargs}

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
        resolved_creations_path = Path(creations_path).resolve()

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
            '--mount',
            f'type=bind,source={resolved_creations_path},target=/project/creations,readonly',
            '--workdir',
            '/sandbox',
            '-e',
            'PYTHONDONTWRITEBYTECODE=1',
            '-e',
            'HOME=/tmp',
            '-e',
            'PYTHONPATH=/project',
            '-e',
            'PROVIDER=OPENROUTER',
            '-e',
            'OPENROUTER_API_KEY=code-tester-dummy-key',
            '-e',
            'OPENROUTER_BASE_URL=https://openrouter.ai/api/v1',
            '-e',
            'MODEL_NAME=openrouter/auto:free',
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
    source_file_path: str,
    implementation: str,
    imports: Optional[List[str]],
    function_inputs: List[Dict[str, Any]],
    pip_packages: List[str],
    utils_path: str,
    agents_path: str,
    creations_path: str,
    timeout_seconds: int = 8,
    memory_limit: str = '256m',
    cpus: str = '1.0',
    pids_limit: int = 64,
    docker_image: str = 'thesis-code-tester:latest',
) -> Dict[str, Any]:
    '''Executes the candidate function once for every kwargs input.'''
    if shutil.which('docker') is None:
        raise RuntimeError('Docker is not installed or is not available on PATH.')

    resolved_source_file_path = Path(source_file_path).resolve()
    resolved_utils_path = Path(utils_path).resolve()
    resolved_agents_path = Path(agents_path).resolve()
    resolved_creations_path = Path(creations_path).resolve()

    if not resolved_source_file_path.is_file():
        raise FileNotFoundError(f'Source file not found: {resolved_source_file_path}')

    if not resolved_utils_path.is_dir():
        raise FileNotFoundError(f'Utils directory not found: {resolved_utils_path}')

    if not resolved_agents_path.is_dir():
        raise FileNotFoundError(f'Agents directory not found: {resolved_agents_path}')

    if not resolved_creations_path.is_dir():
        raise FileNotFoundError(f'Creations directory not found: {resolved_creations_path}')

    if resolved_source_file_path.is_relative_to(resolved_agents_path):
        relative_source_file = resolved_source_file_path.relative_to(resolved_agents_path)
        container_source_file = f'/project/agents/{relative_source_file.as_posix()}'

    elif resolved_source_file_path.is_relative_to(resolved_creations_path):
        relative_source_file = resolved_source_file_path.relative_to(resolved_creations_path)
        container_source_file = f'/project/creations/{relative_source_file.as_posix()}'

    elif resolved_source_file_path.is_relative_to(resolved_utils_path):
        relative_source_file = resolved_source_file_path.relative_to(resolved_utils_path)
        container_source_file = f'/project/utils/{relative_source_file.as_posix()}'

    else:
        raise ValueError('The source file must be located inside the agents, creations, or utils directory.')

    try:
        image_check = subprocess.run(['docker', 'image', 'inspect', docker_image], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False, timeout=10)

    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f'Docker did not respond while checking the image: {docker_image}.') from exc

    except Exception as exc:
        raise RuntimeError(f'Failed to inspect Docker image {docker_image}: {exc}') from exc

    if image_check.returncode != 0:
        raise RuntimeError(f'Docker image not found: {docker_image}. Build the Code Tester image before running.')

    candidate_code = _build_candidate_code(function_name=function_name, source_code=source_code, additional_imports=imports, implementation=implementation)

    execution_image = _ensure_dependency_image(base_image=docker_image, packages=pip_packages)

    invalid_tools: List[str] = _find_tool_description_errors(candidate_code)

    if invalid_tools:
        target_name = function_name
        target_leaf_name = function_name.split('.')[-1]
        target_is_invalid = target_name in invalid_tools or ('.' not in target_name and target_leaf_name in invalid_tools)

        if target_is_invalid:
            error_type = 'TargetToolDescriptionError'
            error_message = f'Target implementation defect: tool {function_name!r} has neither a function docstring nor an explicit @tool(description=...) value.'
        else:
            error_type = 'ModuleToolDescriptionError'
            error_message = f'The candidate module cannot be imported because unrelated tools have neither a function docstring nor an explicit @tool(description=...) value: {invalid_tools}.'

        results = [_failed_result(function_input, error_type=error_type, error_message=error_message) for function_input in function_inputs]
        return {'function_name': function_name, 'total_inputs': len(function_inputs), 'completed_executions': 0, 'failed_executions': len(results), 'results': results}

    results: List[Dict[str, Any]] = []

    for function_input in function_inputs:
        if not isinstance(function_input, dict):
            results.append(_failed_result({}, error_type='InvalidInputError', error_message='Every generated function input must be a dictionary containing keyword arguments.'))
            continue

        result = _run_single_input(function_name=function_name, candidate_code=candidate_code, source_file=container_source_file, kwargs=function_input, utils_path=resolved_utils_path, agents_path=resolved_agents_path, creations_path=resolved_creations_path, timeout_seconds=timeout_seconds, memory_limit=memory_limit, cpus=cpus, pids_limit=pids_limit, docker_image=execution_image)
        results.append(result)

    completed_executions = sum(1 for result in results if result['completed'])
    failed_executions = len(results) - completed_executions

    return {'function_name': function_name, 'total_inputs': len(function_inputs), 'completed_executions': completed_executions, 'failed_executions': failed_executions, 'results': results}