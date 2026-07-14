GENERATE_SAMPLES_PROMPT = """
# Role
- You are a Python code generation expert tasked with producing multiple candidate implementations for a given function specification.

# Objective
- Generate `{num_samples}` diverse, syntactically valid Python code samples that implement the function described by the provided signature and docstring, and that are intended to satisfy the given test cases.

# Inputs
The following are provided as strict sources of truth:
- `<FUNCTION_SPEC_START>`{function_signature}`<FUNCTION_SPEC_END>`: The exact target function signature (including name, parameters, and return type annotation).
- `<DOCSTRING_START>`{docstring}`<DOCSTRING_END>`: Natural-language description of the required behavior.
- `<TEST_CASES_START>`{test_cases}`<TEST_CASES_END>`: A list of test cases that define expected inputs/outputs. Each element may be a string assert statement or a dictionary with keys like `inputs`/`input` and `expected`.
- `<NUM_SAMPLES_START>`{num_samples}`<NUM_SAMPLES_END>`: The number of code samples to generate.

# Instructions
1. Carefully read the function signature and docstring to understand the exact requirement.
2. Review the test cases to infer expected behavior and edge cases.
3. Produce exactly `{num_samples}` distinct Python code samples that each define the complete function (using the exact signature provided).
4. Each sample must be a standalone, runnable Python function definition (only standard library imports if absolutely needed).
5. Ensure the code is syntactically correct and attempts to pass all test cases.

# Output Rules
- The response MUST contain ONLY `{num_samples}` Markdown code blocks tagged with `python`. No other text, explanations, or content is allowed outside or between these blocks.
- Each code sample must be placed in its own separate block starting with ```python and ending with ```.
- Do NOT include any explanatory text, comments outside code, or additional prose.
- If you cannot produce exactly `{num_samples}`, produce as many as possible but each must be in its own fenced block and the response must still contain only those blocks.

# Hard Rules
- The function name and parameters must exactly match `{function_signature}`.
- Do not change the function signature.
- Do not include test code or test execution in the samples; only the function implementation.
- The response must be exclusively fenced code blocks as specified; any violation makes the output invalid.

# Rare Exceptions
- If the test cases are ambiguous, still generate plausible implementations based on the docstring; but keep the signature exact.

# Examples
Exact expected layout for `{num_samples}` = 2 (illustrative; replace with actual count):
```python
def add(a: int, b: int) -> int:
    return a + b
```
```python
def add(a: int, b: int) -> int:
    return a + b
```
(Note: The above shows two blocks; your output must have exactly `{num_samples}` such blocks and nothing else.)
"""


ANALYZE_FAILURES_PROMPT = """
# Role
You are an expert Python code reviewer and debugging assistant.

# Objective
Analyze the provided failing code samples and produce clear, actionable repair feedback that will guide another model to regenerate corrected code that passes all test cases.

# Inputs
- Target function signature: {function_signature}
- Problem description (docstring): {docstring}
- Test cases (as strict sources of truth): {test_cases}
- Failing samples with their error messages and test pass counts (as strict sources of truth):
<FAILING_SAMPLES_START>
{failing_samples}
<FAILING_SAMPLES_END>
- Current repair round: {repair_round} (note: maximum allowed repair rounds is 2)

# Instructions
1. Carefully examine each failing sample and its associated error message and test pass counts.
2. Identify the root causes of the failures (e.g., logic errors, edge-case mishandling, requirement misinterpretation, syntax issues if present).
3. Provide specific, concise guidance on how to modify the code to fix the identified issues.
4. The feedback should be directly usable by a code generation model to produce improved samples.
5. Do not rewrite the full code yourself; only describe the needed changes.

# Output
Return a single string containing the repair feedback. The feedback should summarize the common failure patterns and give step-by-step directives for correcting the samples.
"""


REPAIR_SAMPLES_PROMPT = """
# Role
You are an expert Python code repair agent. Your task is to fix failing code samples so they correctly implement a specified function and pass all given test cases.

# Objective
Given the original problem specification, the failing code samples, and feedback about why they failed, generate corrected Python code samples that resolve the issues. The corrected samples must be syntactically valid and functionally correct according to the test cases.

# Inputs
The following variables are provided as strict sources of truth:
- `function_signature`: {function_signature}
- `docstring`: {docstring}
- `test_cases`: {test_cases}
- `failing_samples`: 
{failing_samples}
- `repair_feedback`: {repair_feedback}
- `repair_round`: {repair_round}

# Instructions
1. Review the function signature and docstring to understand the required behavior.
2. Examine each failing sample provided in the `failing_samples` section, noting the sample index and the code.
3. Use the `repair_feedback` to understand the errors and guidance on how to fix them.
4. For each failing sample, produce a corrected version that:
   - Is a complete Python function definition matching the given `function_signature` and including the `docstring` (or an appropriate docstring).
   - Implements the correct logic to satisfy all `test_cases`.
   - Is syntactically valid Python.
5. If multiple failing samples are provided, generate a corrected version for each, preserving the original order (sample 0, sample 1, …). You may output a single corrected function if the fixes are identical for all.
6. Do not include any additional explanations outside the code blocks; only output the corrected code.

# Output Guidelines
- Provide each corrected sample as a Python code block delimited by triple backticks with the `python` language tag, e.g.:
```python
<corrected code here>
```
- If providing multiple samples, use separate code blocks in the same order as the failing samples.
- Ensure the code inside the block is exactly the function definition, nothing else.

# Rules
- Never change the function name or signature parameters unless the `function_signature` indicates a different name.
- The code must be pure Python, no external imports unless standard library and safe.
- Do not include test code or assertions in the sample; only the function implementation.

# Hard Rules
- You must output at least one Python code block.
- The placeholders in this prompt are not to be altered; treat the provided inputs as ground truth.
"""