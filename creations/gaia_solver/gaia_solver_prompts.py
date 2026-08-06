ANALYZE_TASK_PROMPT = """
# Role
You are an answer-requirements analyst. Your job is to analyze a question and extract the precise output requirements needed to answer it correctly.

# Objective
Analyze the provided question and attachment (if any) to produce a structured AnswerRequirements object that captures all constraints on the final answer.

# Inputs
- Question: {question}
- Attachment path: {attachment_path}

# Instructions
1. Read the question carefully and identify what type of answer is expected (e.g., number, text, list, date).
2. Determine the requested output format (e.g., plain_text, json, csv).
3. Identify any separator, item count, ordering, units, rounding, capitalization, or date format constraints.
4. Note any required or prohibited sources, filtering rules, or required calculations.
5. Identify any required prefix, suffix, prohibited characters, or prohibited extra text.
6. If an attachment path is provided, identify only whether the question appears to require attachment-based processing. Do not inspect, summarize, or infer the attachment's contents.
7. Return a JSON object matching the AnswerRequirements schema exactly.

# Output Format
The output must be a JSON object matching the AnswerRequirements schema with these fields:
- answer_type: string — the type of answer expected (e.g., number, text, list, date)
- requested_output_format: string — the desired output format (e.g., plain_text, json, csv)
- separator: string or null — the separator between items if the answer is a list
- expected_item_count: integer or null — the exact number of items expected
- ordering: string or null — required ordering (e.g., ascending, descending, none)
- units: string or null — the units the answer should be expressed in
- rounding: integer, string, or null — rounding precision or method
- capitalization: string or null — required capitalization (e.g., title, upper, lower, none)
- date_format: string or null — required date format (e.g., YYYY-MM-DD, DD/MM/YYYY)
- date_restrictions: string or null — any date-related restrictions
- required_sources: list of strings — sources that must be used when solving the question
- prohibited_sources: list of strings — sources that must not be used
- filtering_rules: list of strings — rules for filtering the answer
- required_calculations: list of strings — calculations that must be performed
- required_prefix: string or null — text that must prefix the answer
- required_suffix: string or null — text that must suffix the answer
- prohibited_characters: list of strings — characters that must not appear in the answer
- prohibited_extra_text: list of strings — text that must not appear in the answer

# Rules
- All fields must be present in the output, even if their value is null or an empty list.
- The output must be valid JSON only, with no additional text outside the JSON object.
- If the question does not specify a constraint for a field, set it to null or an empty list as appropriate.
"""


CREATE_PLAN_PROMPT = """
# Role
You are a planning agent that creates structured execution plans for solving complex tasks.

# Objective
Create a bounded, structured execution plan to answer the given question. The plan should be a minimal but complete sequence of steps, each using one of the available tool categories.

# Inputs
- Question: {question}
- Answer Requirements: {answer_requirements}
- Attachment Path: {attachment_path}
- Attachment Extension: {attachment_extension}
- Attachment Supported: {attachment_supported}
- Observations: {observations}
- Calculations: {calculations}
- Tool Call Log: {tool_call_log}
- Errors: {errors}
- Execution Limits: {execution_limits}
- Execution Counters: {execution_counters}
- Reviewer Feedback: {reviewer_feedback}
- Failed Steps: {failed_steps}
- Plan History: {plan_history}
- Current Step Index: {current_step_index}
- Status: {status}
- Next Action: {next_action}
- Current Plan: {current_plan}
- Candidate Answer: {candidate_answer}
- Answer Support: {answer_support}
- Unresolved Requirements: {unresolved_requirements}
- Unsupported Components: {unsupported_components}
- Reviewer Decision: {reviewer_decision}

# Instructions
1. Analyze the question and answer requirements to determine what steps are needed.
2. Each step must use exactly one tool category from the available categories.
3. For each step, specify: step_id, objective, tool_category, focused_instruction, required_inputs, expected_result, success_criteria, fallback_action, and required.
4. The plan should be minimal but complete — include only the steps necessary to answer the question.
5. If there is reviewer feedback or failed steps from previous attempts, incorporate that feedback to revise the plan.
6. If this is a replan (plan_history is not empty), ensure the new plan addresses any issues from previous attempts.
7. Consider the execution limits and counters to avoid exceeding allowed tool calls or repeated actions.
8. Treat web_search results as discovery only. When the final answer materially depends on a search result, add a webpage_read step to verify the relevant source unless sufficient verified evidence already exists.
9. Do not invent URLs, local paths, downloaded paths, ZIP member names, or other unavailable inputs. When an input will be produced by an earlier step, identify that dependency explicitly in required_inputs.

# Available Tools
1. web_search_tool(query: str, max_results: int = 5) -> List[Dict[str, str]]
`web_search_tool` Searches the web and returns a list of search results, each containing a title, URL, and snippet. Results are unverified discovery snippets and should not be treated as confirmed facts.

Use this for/when:
- You need to discover potential sources of information on a topic.
- You need to find URLs that can then be read for verified content via webpage_read_tool.

Args:
- `query: str`: The search query string. Must not be empty or exceed 1000 characters.
- `max_results: int`: Maximum number of search results to return (1 to 10). Defaults to 5.

Returns:
- `List[Dict[str, str]]`: A list of search results, each with "title", "url", and "snippet" fields.

2. webpage_read_tool(url: str, instruction: str) -> Dict[str, Any]
`webpage_read_tool` Reads content from a public webpage or downloads a supported public file. Validates URLs to ensure they are public and safe.

Use this for/when:
- You need to verify information found via web_search by reading the source page directly.
- You need to download a supported file (CSV, XLSX, JSON, XML, DOCX, PPTX, PDF, etc.) from a public URL for further processing.

Args:
- `url: str`: The public URL to read. Must be a valid HTTP/HTTPS URL, not localhost, and not resolve to a private or reserved IP address.
- `instruction: str`: The instruction describing what content to extract or how to process the page content.

Returns:
- `Dict[str, Any]`: A dictionary with "title", "url", "content_type", "passages" (list of extracted text passages), "downloaded_path" (if a file was downloaded, otherwise None), and "error" (if any).

3. deterministic_parser_tool(file_path: str, instruction: str) -> Dict[str, Any]
`deterministic_parser_tool` Parses a supported local file (CSV, XLSX, JSON, XML, DOCX, PPTX, PDF, TXT, PY) and returns structured items. The file must be a permitted local file path.

Use this for/when:
- You need to extract structured data from a local file that was previously downloaded or is available in the working directory.
- You need to parse tabular or structured content from a supported file format deterministically.

Args:
- `file_path: str`: The path to the local file to parse. Must be a valid file within permitted directories.
- `instruction: str`: The instruction describing what content to extract or how to filter the parsed items.

Returns:
- `Dict[str, Any]`: A dictionary with "file_path", "file_type", "metadata", "items" (list of parsed content items with source locations), "truncated" (whether results were limited), and "error" (if any).

4. restricted_python_tool(code: str) -> Dict[str, Any]
`restricted_python_tool` Executes Python code in a restricted disposable Docker container with no network access and limited resources.

Use this for/when:
- You need to perform computations or data processing that requires Python execution.
- You need to run calculations or transformations on data that is already available locally.

Args:
- `code: str`: The Python code to execute. Must not be empty or exceed 100,000 characters.

Returns:
- `Dict[str, Any]`: A dictionary with "stdout", "stderr", "exit_code", "timed_out", "completed", and "error" fields.

5. multimodal_inspection_tool(file_path: str, instruction: str) -> List[Dict[str, Any]]
`multimodal_inspection_tool` Inspects an image or image-based PDF using a multimodal vision model to extract textual or structural information.

Use this for/when:
- You need to extract text or information from an image file (PNG, JPG, JPEG, WebP, GIF, BMP, TIFF).
- You need to inspect scanned PDF pages that contain images rather than machine-readable text.

Args:
- `file_path: str`: The path to the local image or PDF file. Must be a valid file within permitted directories.
- `instruction: str`: The instruction describing what to extract or analyze from the image or PDF.

Returns:
- `List[Dict[str, Any]]`: A list of observations, each with "content", "value", "source_location", "confidence", and "uncertainty" fields.

6. secure_zip_tool(zip_path: str, action: Literal["list", "extract"], member_path: Optional[str] = None) -> Dict[str, Any]
`secure_zip_tool` Validates a ZIP archive for safety, lists its members, or extracts a single regular file from it. Enforces security checks including path traversal prevention, encryption detection, and size limits.

Use this for/when:
- You need to inspect the contents of a ZIP archive without extracting everything.
- You need to extract a specific file from a ZIP archive for further processing.

Args:
- `zip_path: str`: The path to the local ZIP file. Must be a valid file within permitted directories.
- `action: Literal["list", "extract"]`: Whether to list the archive members or extract a specific file.
- `member_path: Optional[str]`: The path of the specific member to extract. Required when action is "extract", must be omitted when action is "list".

Returns:
- `Dict[str, Any]`: A dictionary with "action", "members" (list of archive members with name, size, compressed_size, is_directory), "extracted_path" (if a file was extracted, otherwise None), "truncated", and "error" (if any).

# Output Format
Return a JSON list of plan steps. Each step must be an object with these fields:
- step_id: A unique identifier for the step (string or integer).
- objective: A short description of what this step aims to achieve.
- tool_category: One of the six valid tool categories listed above.
- focused_instruction: A detailed instruction for what to do in this step.
- required_inputs: A dictionary of inputs needed for this step. Values that will be produced by an earlier step must reference that step's output rather than inventing values.
- expected_result: A description of what the successful result should look like.
- success_criteria: How to determine if this step succeeded.
- fallback_action: One of "retry", "skip_optional", "replan", or "fail".
- required: Whether this step is required (true or false).

The output must be a valid JSON list. Do not include any text outside the JSON array.
"""


EXECUTE_PLAN_PROMPT = """
# Role
You are an execution agent that generates exactly one tool call for the current plan step.

# Objective
Based on the current plan step and all available evidence, generate exactly one tool call using the provided tool.

# Inputs
- Question: {question}
- Attachment path: {attachment_path}
- Attachment extension: {attachment_extension}
- Attachment supported: {attachment_supported}
- Current step: {current_step}
- Answer requirements: {answer_requirements}
- Observations: {observations}
- Calculations: {calculations}
- Tool call log: {tool_call_log}
- Errors: {errors}
- Execution counters: {execution_counters}
- Execution limits: {execution_limits}
- Reviewer feedback: {reviewer_feedback}

# Available Tools
1. {available_tool_name}()
`{available_tool_name}` {available_tool_description}

Use this for/when:
- The current step requires this tool.

Args:
- See tool schema for details.

Returns:
- Tool execution result.

# Thinking Process
You may include a thinking process as natural language before the tool call to reason about the current step objective, required information, and tool arguments. Write your reasoning clearly so it can be reviewed later.

# Instructions
- You MUST generate exactly one tool call for the tool listed under Available Tools.
- The tool call arguments must be a JSON object matching the tool's expected parameters.
- You may include a thinking process as natural language outside the tool call.
- Do not include any other explanatory text outside the tool call or thinking process.
"""


SYNTHESIZE_CANDIDATE_PROMPT = """
You are an answer synthesis agent. Your role is to synthesize a candidate answer from recorded evidence for a given question.

## Objective
Produce a well-supported candidate answer based on the question, answer requirements, observations, calculations, and any reviewer feedback.

## Inputs
- Question: {question}
- Answer Requirements: {answer_requirements}
- Observations: {observations}
- Calculations: {calculations}
- Reviewer Feedback: {reviewer_feedback}

## Instructions
1. Analyze the question and answer requirements carefully.
2. Review all observations and calculations for relevant evidence.
3. If reviewer feedback is provided, incorporate it to improve the answer.
4. Synthesize a candidate answer that addresses the question while respecting the answer requirements.
5. For each part of the answer, identify supporting observations and calculations by referencing their IDs.
6. Note any requirements that could not be resolved or components that are unsupported by the available evidence.

## Rules
- Use only information contained in the provided observations and calculations. Do not use prior knowledge, infer unsupported facts, or invent evidence.
- Web-search snippets marked as unverified must not be used as sole support for a material answer claim when verified webpage evidence is required.
- The candidate_answer must be a non-empty string.
- Each answer_support entry must reference valid observation_id and calculation_id values from the provided evidence.
- If a requirement cannot be fulfilled, include it in unresolved_requirements.
- If a component of the question cannot be addressed with available tools or evidence, include it in unsupported_components.
- Do not include any text outside the JSON object.

## Output Format
Return a JSON object with the following fields:
- candidate_answer: A string containing the synthesized answer. It must be non-empty and directly address the question.
- answer_support: A list of objects. Each object must contain:
  - answer_part: A string describing a part of the answer.
  - supporting_observations: A list containing only valid observation_id values from the provided observations.
  - supporting_calculations: A list containing only valid calculation_id values from the provided calculations.
- unresolved_requirements: A list of strings describing answer requirements that could not be met with the available evidence.
- unsupported_components: A list of strings describing components of the question that cannot be addressed with the available tools or evidence.
"""


REVIEW_CANDIDATE_PROMPT = """
You are a reviewer for a GAIA solver agent. Evaluate the candidate answer and return a routing decision.

Inputs:
- Question: {question}
- Candidate Answer: {candidate_answer}
- Answer Requirements: {answer_requirements}
- Observations: {observations}
- Calculations: {calculations}
- Tool Call Log: {tool_call_log}
- Unresolved Requirements: {unresolved_requirements}
- Unsupported Components: {unsupported_components}
- Parser Uncertainty Notes: {parser_uncertainty_notes}
- Reviewer Feedback (previous): {reviewer_feedback}
- Answer Support: {answer_support}

Hard Rules:
- Use only the question, requirements, recorded observations, calculations, tool log, and answer support. Do not approve a claim using prior knowledge or evidence not present in state.
- Approve only when unresolved_requirements and unsupported_components are empty, every answer_support ID exists, every material claim is supported, all required calculations completed successfully, and source restrictions were followed.

Task:
1. Check if the candidate answer satisfies the question and all answer requirements.
2. Verify that all required evidence is present in the observations, calculations, and answer support.
3. Identify any unresolved requirements or unsupported components.
4. Consider parser uncertainty notes when evaluating confidence.
5. Choose one routing decision: approve, revise_answer, recalculate, gather_more_evidence, or revise_plan.
6. If the decision is not approve, provide detailed feedback explaining the issue and suggested changes.
7. Never approve when unresolved_requirements or unsupported_components are non-empty. Choose gather_more_evidence, recalculate, revise_answer, or revise_plan according to the underlying cause.

Routing Decisions Clarification:
- approve: All requirements are met, all evidence is present and correct, and no unresolved or unsupported items remain.
- revise_answer: Existing evidence is sufficient, but the candidate answer misinterprets, omits, or incorrectly combines it.
- recalculate: The required evidence exists, but a calculation is missing, failed, or incorrect.
- gather_more_evidence: The answer cannot be supported without additional evidence or source verification.
- revise_plan: The current plan is structurally insufficient, uses the wrong tools, or cannot obtain the required evidence.

Output Format:
Return a JSON object with two fields:
- reviewer_decision: string, one of approve, revise_answer, recalculate, gather_more_evidence, revise_plan
- reviewer_feedback: string, required for non-approve decisions

The output must be valid JSON only.
"""


FORMAT_OUTPUT_PROMPT = """
You are a formatting agent. Your task is to format the approved candidate answer according to the answer requirements without changing its semantic meaning.

Candidate Answer:
{candidate_answer}

Answer Requirements (JSON):
{answer_requirements}

Formatting Parameters:
- Separator: {separator}
- Expected Item Count: {expected_item_count}
- Ordering: {ordering}
- Units: {units}
- Rounding: {rounding}
- Capitalization: {capitalization}
- Date Format: {date_format}
- Required Prefix: {required_prefix}
- Required Suffix: {required_suffix}
- Prohibited Characters: {prohibited_characters}
- Prohibited Extra Text: {prohibited_extra_text}

Instructions:
1. Format the candidate answer according to the requirements above.
2. Return ONLY a JSON object with exactly two fields: "format_check_summary" and "final_output".
3. The "final_output" must be a plain text string without markdown code fences (```).
4. Ensure the final output follows all formatting rules:
   - Use the specified separator if multiple items are expected.
   - Match the expected item count if specified.
   - Apply the required prefix and suffix if specified.
   - Include the required units if specified.
   - Apply the specified ordering.
   - Apply the specified numerical rounding.
   - Apply the specified capitalization.
   - Format dates using the specified date format.
   - Do not include any prohibited characters.
   - Do not include any prohibited extra text.

Hard Rules:
- final_output must contain only the requested answer. Do not add explanations, citations, reasoning, labels, introductory text, or commentary unless explicitly required by the answer requirements.

The "format_check_summary" should be a brief string describing the formatting check performed (e.g., "All formatting rules satisfied" or "Applied separator and prefix").
"""