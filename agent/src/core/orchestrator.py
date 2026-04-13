# Copyright (c) 2025 MiroMind
# This source code is licensed under the MIT License.

import asyncio
import gc
import json
import logging
import os
import re
from json_repair import repair_json
import time
import uuid
from collections import defaultdict
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

from miroflow_tools.manager import ToolManager
from omegaconf import DictConfig

from ..config.settings import expose_sub_agents_as_tools
from ..io.input_handler import process_input
from ..io.output_formatter import OutputFormatter
from ..llm.factory import ClientFactory
from ..logging.task_logger import (
    TaskLog,
    get_utc_plus_8_time,
)
from ..utils.parsing_utils import extract_llm_response_text
from ..utils.prompt_utils import (
    generate_agent_specific_system_prompt,
    generate_agent_summarize_prompt,
    generate_agent_progress_prompt,
    mcp_tags,
    refusal_keywords,
)
from ..utils.wrapper_utils import ErrorBox, ResponseBox

logger = logging.getLogger(__name__)


def _list_tools(sub_agent_tool_managers: Dict[str, ToolManager]):
    # Use a dictionary to store the cached result
    cache = None

    async def wrapped():
        nonlocal cache
        if cache is None:
            # Only fetch tool definitions if not already cached
            result = {
                name: await tool_manager.get_all_tool_definitions()
                for name, tool_manager in sub_agent_tool_managers.items()
            }
            cache = result
        return cache

    return wrapped


class Orchestrator:
    def __init__(
        self,
        main_agent_tool_manager: ToolManager,
        sub_agent_tool_managers: Dict[str, ToolManager],
        llm_client: ClientFactory,
        output_formatter: OutputFormatter,
        cfg: DictConfig,
        task_log: Optional["TaskLog"] = None,
        stream_queue: Optional[Any] = None,
        tool_definitions: Optional[List[Dict[str, Any]]] = None,
        sub_agent_tool_definitions: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    ):
        self.main_agent_tool_manager = main_agent_tool_manager
        self.sub_agent_tool_managers = sub_agent_tool_managers
        self.llm_client = llm_client
        self.output_formatter = output_formatter
        self.cfg = cfg
        self.task_log = task_log
        self.stream_queue = stream_queue
        self.tool_definitions = tool_definitions
        self.sub_agent_tool_definitions = sub_agent_tool_definitions
        # call this once, then use cache value
        self._list_sub_agent_tools = None
        if sub_agent_tool_managers:
            self._list_sub_agent_tools = _list_tools(sub_agent_tool_managers)

        # Pass task_log to llm_client
        if self.llm_client and task_log:
            self.llm_client.task_log = task_log

        # Track boxed answers extracted during main loop turns
        self.intermediate_boxed_answers = []
        # Record used subtask / q / Query
        self.used_queries = {}

        # Retry loop protection limits
        self.MAX_CONSECUTIVE_ROLLBACKS = 10
        self.MAX_FINAL_ANSWER_RETRIES = 3 if cfg.agent.keep_tool_result == -1 else 1

    def _get_result_dir(self) -> Optional[str]:
        """Get the result directory path: ./results/{task_id}/"""

        if not self.task_log or not self.task_log.task_id:
            return None

        # Use relative path ./results/{task_id}/
        result_dir = os.path.join("results", self.task_log.task_id)
        os.makedirs(result_dir, exist_ok=True)

        return result_dir

    def _initialize_result_directories(self) -> Optional[str]:
        """Initialize result directories at the start of workflow.

        Creates:
        - ./results/{task_id}/ (main result directory)
        - ./results/{task_id}/charts/ (for charts)
        - ./results/{task_id}/images/ (for images)

        Returns:
            Result directory path if successful, None otherwise
        """
        # Early return if no result directory
        self.result_dir = self._get_result_dir()
        if not self.result_dir:
            self.task_log.log_step(
                "error",
                "Report Workflow | Initialize Directories",
                "Failed to initialize directories: task_id is missing",
            )
            return None

        try:
            # Define subdirectories
            subdirs = {
                "charts": os.path.join(self.result_dir, "charts"),
                "images": os.path.join(self.result_dir, "images"),
            }

            # Create all directories in one pass
            for subdir_name, subdir_path in subdirs.items():
                os.makedirs(subdir_path, exist_ok=True)

            # Get absolute paths for logging
            abs_result_dir = os.path.abspath(self.result_dir)
            abs_subdirs = {
                name: os.path.abspath(path) for name, path in subdirs.items()
            }

            # Log success with formatted output
            log_message = (
                f"Initialized result directories:\n"
                f"  Result dir: {abs_result_dir}\n"
                + "\n".join(
                    f"  {name.capitalize()} dir: {path}"
                    for name, path in abs_subdirs.items()
                )
            )

            self.task_log.log_step(
                "info",
                "Report Workflow | Initialize Directories",
                log_message,
            )

            return self.result_dir

        except Exception as e:
            # Unified exception handling for all errors
            error_type = type(e).__name__
            self.task_log.log_step(
                "error",
                "Report Workflow | Initialize Directories",
                f"Failed to initialize directories ({error_type}): {e}",
            )
            return None

    def _save_intermediate_output(
        self, phase_name: str, content: Any, file_extension: str = "md"
    ):
        """Save intermediate output to ./results/{task_id}/ folder.

        Args:
            phase_name: Phase identifier (e.g., "01_outline", "02_draft_sections", "03_polished_report")
            content: Content to save (string for markdown, dict for JSON)
            file_extension: File extension ("md" or "json")
        """
        if not self.result_dir:
            self.task_log.log_step(
                "error",
                f"Report Workflow | Save Intermediate",
                "Result directory not initialized, cannot save intermediate output.",
            )
            return

        try:
            # Generate filename: {phase_name}.{extension}
            filename = f"{phase_name}.{file_extension}"
            filepath = os.path.join(self.result_dir, filename)
            absolute_filepath = os.path.abspath(filepath)

            # Write content based on file type
            if file_extension == "json":
                # For JSON, content should be a dict
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(content, f, ensure_ascii=False, indent=2)
            else:
                # For markdown/text, content should be a string
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(content if isinstance(content, str) else str(content))

            self.task_log.log_step(
                "info",
                f"Report Workflow | Save Intermediate",
                f"Saved intermediate output for {phase_name} to {absolute_filepath}",
            )
        except Exception as e:
            self.task_log.log_step(
                "error",
                f"Report Workflow | Save Intermediate",
                f"Failed to save intermediate output for {phase_name}: {e}",
            )

    async def _stream_update(self, event_type: str, data: dict):
        """Send streaming update in new SSE protocol format"""
        if self.stream_queue:
            try:
                stream_message = {
                    "event": event_type,
                    "data": data,
                }
                await self.stream_queue.put(stream_message)
            except Exception as e:
                logger.warning(f"Failed to send stream update: {e}")

    async def _stream_start_workflow(self, user_input: str) -> str:
        """Send start_of_workflow event"""
        workflow_id = str(uuid.uuid4())
        await self._stream_update(
            "start_of_workflow",
            {
                "workflow_id": workflow_id,
                "input": [
                    {
                        "role": "user",
                        "content": user_input,
                    }
                ],
            },
        )
        return workflow_id

    async def _stream_end_workflow(self, workflow_id: str):
        """Send end_of_workflow event"""
        await self._stream_update(
            "end_of_workflow",
            {
                "workflow_id": workflow_id,
            },
        )

    async def _stream_show_error(self, error: str):
        """Send show_error event"""
        await self._stream_tool_call("show_error", {"error": error})
        if self.stream_queue:
            try:
                await self.stream_queue.put(None)
            except Exception as e:
                logger.warning(f"Failed to send show_error: {e}")

    async def _stream_start_agent(self, agent_name: str, display_name: str = None):
        """Send start_of_agent event"""
        agent_id = str(uuid.uuid4())
        await self._stream_update(
            "start_of_agent",
            {
                "agent_name": agent_name,
                "display_name": display_name,
                "agent_id": agent_id,
            },
        )
        return agent_id

    async def _stream_end_agent(self, agent_name: str, agent_id: str):
        """Send end_of_agent event"""
        await self._stream_update(
            "end_of_agent",
            {
                "agent_name": agent_name,
                "agent_id": agent_id,
            },
        )

    async def _stream_start_llm(self, agent_name: str, display_name: str = None):
        """Send start_of_llm event"""
        await self._stream_update(
            "start_of_llm",
            {
                "agent_name": agent_name,
                "display_name": display_name,
            },
        )

    async def _stream_end_llm(self, agent_name: str):
        """Send end_of_llm event"""
        await self._stream_update(
            "end_of_llm",
            {
                "agent_name": agent_name,
            },
        )

    async def _stream_message(self, message_id: str, delta_content: str):
        """Send message event"""
        await self._stream_update(
            "message",
            {
                "message_id": message_id,
                "delta": {
                    "content": delta_content,
                },
            },
        )

    async def _stream_tool_call(
        self,
        tool_name: str,
        payload: dict,
        streaming: bool = False,
        tool_call_id: str = None,
    ) -> str:
        """Send tool_call event"""
        if not tool_call_id:
            tool_call_id = str(uuid.uuid4())

        if streaming:
            for key, value in payload.items():
                await self._stream_update(
                    "tool_call",
                    {
                        "tool_call_id": tool_call_id,
                        "tool_name": tool_name,
                        "delta_input": {key: value},
                    },
                )
        else:
            # Send complete tool call
            await self._stream_update(
                "tool_call",
                {
                    "tool_call_id": tool_call_id,
                    "tool_name": tool_name,
                    "tool_input": payload,
                },
            )

        return tool_call_id

    def get_scrape_result(self, result: str) -> str:
        """
        Process scrape result and truncate if too long to support more conversation turns.
        """
        SCRAPE_MAX_LENGTH = 20000
        try:
            scrape_result_dict = json.loads(result)
            text = scrape_result_dict.get("text")
            if text and len(text) > SCRAPE_MAX_LENGTH:
                text = text[:SCRAPE_MAX_LENGTH]
            return json.dumps({"text": text}, ensure_ascii=False)
        except json.JSONDecodeError:
            if isinstance(result, str) and len(result) > SCRAPE_MAX_LENGTH:
                result = result[:SCRAPE_MAX_LENGTH]
            return result

    def post_process_tool_call_result(self, tool_name, tool_call_result: dict):
        """Process tool call results"""
        # Only in demo mode: truncate scrape results to 20,000 chars
        # to support more conversation turns. Skipped in perf tests to avoid loss.
        if os.environ.get("DEMO_MODE") == "1":
            if "result" in tool_call_result and tool_name in [
                "scrape",
                "scrape_website",
            ]:
                tool_call_result["result"] = self.get_scrape_result(
                    tool_call_result["result"]
                )
        return tool_call_result

    def _fix_tool_call_arguments(self, tool_name: str, arguments: dict) -> dict:
        """
        Fix common parameter name mistakes made by LLM.
        """
        # Create a copy to avoid modifying the original
        fixed_args = arguments.copy()
        # Fix scrape_and_extract_info parameter names
        if tool_name == "scrape_and_extract_info":
            # Map common mistakes to the correct parameter name
            mistake_names = [
                "description",
                "introduction",
            ]
            if "info_to_extract" not in fixed_args:
                for mistake_name in mistake_names:
                    if mistake_name in fixed_args:
                        fixed_args["info_to_extract"] = fixed_args.pop(mistake_name)
                        break

        return fixed_args

    def _get_query_str_from_tool_call(
        self, tool_name: str, arguments: dict
    ) -> Optional[str]:
        """
        Extracts the query string from tool call arguments based on tool_name.
        Supports search_and_browse, google_search, sougou_search, scrape_website, and scrape_and_extract_info.
        """
        if tool_name == "search_and_browse":
            return tool_name + "_" + arguments.get("subtask", "")
        elif tool_name == "google_search":
            return tool_name + "_" + arguments.get("q", "")
        elif tool_name == "sougou_search":
            return tool_name + "_" + arguments.get("Query", "")
        elif tool_name == "scrape_website":
            return tool_name + "_" + arguments.get("url", "")
        elif tool_name == "scrape_and_extract_info":
            return (
                tool_name
                + "_"
                + arguments.get("url", "")
                + "_"
                + arguments.get("info_to_extract", "")
            )
        return None

    async def _handle_llm_call(
        self,
        system_prompt,
        message_history,
        tool_definitions,
        step_id: int,
        purpose: str = "",
        agent_type: str = "main",
    ) -> Tuple[Optional[str], bool, Optional[Any], List[Dict[str, Any]]]:
        """Unified LLM call and logging processing
        Returns:
            Tuple[Optional[str], bool, Optional[Any], List[Dict[str, Any]]]:
                (response_text, should_break, tool_calls_info, message_history)
        """
        original_message_history = message_history
        try:
            response, message_history = await self.llm_client.create_message(
                system_prompt=system_prompt,
                message_history=message_history,
                tool_definitions=tool_definitions,
                keep_tool_result=self.cfg.agent.keep_tool_result,
                stream=self.cfg.llm.stream,
                step_id=step_id,
                task_log=self.task_log,
                agent_type=agent_type,
            )
            if ErrorBox.is_error_box(response):
                await self._stream_show_error(str(response))
                response = None
            if ResponseBox.is_response_box(response):
                if response.has_extra_info():
                    extra_info = response.get_extra_info()
                    if extra_info.get("warning_msg"):
                        await self._stream_show_error(
                            extra_info.get("warning_msg", "Empty warning message")
                        )

                response = response.get_response()
            # Check if response is None (indicating an error occurred)
            if response is None:
                self.task_log.log_step(
                    "error",
                    f"{purpose} | LLM Call Failed",
                    f"{purpose} failed - no response received",
                )
                return "", False, None, original_message_history

            # Use client's response processing method
            assistant_response_text, should_break, message_history = (
                self.llm_client.process_llm_response(
                    response, message_history, agent_type
                )
            )

            # Use client's tool call information extraction method
            tool_calls_info = self.llm_client.extract_tool_calls_info(
                response, assistant_response_text
            )

            self.task_log.log_step(
                "info",
                f"{purpose} | LLM Call",
                "completed successfully",
            )
            return (
                assistant_response_text,
                should_break,
                tool_calls_info,
                message_history,
            )

        except Exception as e:
            self.task_log.log_step(
                "error",
                f"{purpose} | LLM Call ERROR",
                f"{purpose} error: {str(e)}",
            )
            # Return empty response with should_break=False, need to retry
            return "", False, None, original_message_history

    def _detect_language(self, text: str) -> str:
        """Detect if text is primarily Chinese or English.

        Args:
            text: Text to analyze

        Returns:
            "zh" for Chinese, "en" for English
        """
        if not text:
            return "en"

        # Count Chinese characters
        chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
        # Count English letters
        english_chars = len(re.findall(r"[a-zA-Z]", text))

        # # If more than 30% Chinese characters, consider it Chinese
        # total_chars = len(text)
        # if total_chars > 0 and chinese_chars / total_chars > 0.3:
        #     return "zh"

        # # If has Chinese but also significant English, check ratio
        # if chinese_chars > 0 and english_chars > 0:
        #     return "zh" if chinese_chars > english_chars * 0.5 else "en"

        return "zh" if chinese_chars > 0 else "en"

    async def run_sub_agent(
        self,
        sub_agent_name: str,
        task_description: str,
    ):
        """Run sub agent"""
        # task_description += "\n\nPlease provide the answer and detailed supporting information of the subtask given to you."
        self.task_log.log_step(
            "info",
            f"{sub_agent_name} | Task Description",
            f"Subtask: {task_description}",
        )

        # Stream sub-agent start
        display_name = sub_agent_name.replace("agent-", "")
        sub_agent_id = await self._stream_start_agent(display_name)
        await self._stream_start_llm(display_name)

        # Start new sub-agent session
        self.task_log.start_sub_agent_session(sub_agent_name, task_description)

        # Simplified initial user content (no file attachments)
        initial_user_content = task_description
        message_history = [{"role": "user", "content": initial_user_content}]

        # Get sub-agent tool definitions
        if not self.sub_agent_tool_definitions:
            tool_definitions = await self._list_sub_agent_tools()
            tool_definitions = tool_definitions.get(sub_agent_name, {})
        else:
            tool_definitions = self.sub_agent_tool_definitions[sub_agent_name]

        if not tool_definitions:
            self.task_log.log_step(
                "warning",
                f"{sub_agent_name} | No Tools",
                "No tool definitions available.",
            )

        # Generate sub-agent system prompt
        system_prompt = self.llm_client.generate_agent_system_prompt(
            date=date.today(),
            mcp_servers=tool_definitions,
        ) + generate_agent_specific_system_prompt(
            agent_type=sub_agent_name, language=self.task_lang
        )

        # Limit sub-agent turns
        if self.cfg.agent.sub_agents:
            max_turns = self.cfg.agent.sub_agents[sub_agent_name].max_turns
        else:
            max_turns = 0
        turn_count = 0
        total_attempts = 0
        max_attempts = max_turns + 200
        consecutive_rollbacks = 0

        skip_summary = False
        tool_call_format_error = False

        while turn_count < max_turns and total_attempts < max_attempts:
            turn_count += 1
            total_attempts += 1
            if consecutive_rollbacks >= self.MAX_CONSECUTIVE_ROLLBACKS:
                self.task_log.log_step(
                    "error",
                    f"{sub_agent_name} | Too Many Rollbacks",
                    f"Reached {consecutive_rollbacks} consecutive rollbacks (limit: {self.MAX_CONSECUTIVE_ROLLBACKS}), breaking loop. Total attempts: {total_attempts}/{max_attempts}",
                )
                break

            self.task_log.save()

            # Reset 'last_call_tokens'
            self.llm_client.last_call_tokens = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
            }

            # Use unified LLM call processing
            (
                assistant_response_text,
                should_break,
                tool_calls,
                message_history,
            ) = await self._handle_llm_call(
                system_prompt
                + generate_agent_progress_prompt(
                    turn_count, max_turns, tool_call_format_error
                ),
                message_history,
                tool_definitions,
                turn_count,
                f"{sub_agent_name} | Turn: {turn_count}",
                agent_type=sub_agent_name,
            )

            tool_call_format_error = False

            if should_break:
                self.task_log.log_step(
                    "info",
                    f"{sub_agent_name} | Turn: {turn_count} | LLM Call",
                    "should break is True, breaking the loop",
                )
                break

            # Process LLM response
            elif assistant_response_text:
                text_response = extract_llm_response_text(assistant_response_text)
                if text_response:
                    await self._stream_tool_call("show_text", {"text": text_response})

            else:
                # LLM call failed, end current turn
                turn_count -= 1
                self.task_log.log_step(
                    "info",
                    f"{sub_agent_name} | Turn: {turn_count} | LLM Call",
                    "LLM call failed",
                )
                await asyncio.sleep(5)
                continue

            # Use tool calls parsed from LLM response
            if not tool_calls:
                if any(mcp_tag in assistant_response_text for mcp_tag in mcp_tags):
                    # If we haven't reached rollback limit, rollback and retry
                    if consecutive_rollbacks < self.MAX_CONSECUTIVE_ROLLBACKS - 1:
                        turn_count -= 1
                        consecutive_rollbacks += 1
                        if message_history[-1]["role"] == "assistant":
                            message_history.pop()
                        self.task_log.log_step(
                            "warning",
                            f"{sub_agent_name} | Turn: {turn_count} | Rollback",
                            f"Tool call format incorrect - found MCP tags in response. Consecutive rollbacks: {consecutive_rollbacks}/{self.MAX_CONSECUTIVE_ROLLBACKS}, Total attempts: {total_attempts}/{max_attempts}",
                        )
                        tool_call_format_error = True
                        continue
                    else:
                        # Reached rollback limit, allow the loop to end naturally
                        self.task_log.log_step(
                            "warning",
                            f"{sub_agent_name} | Turn: {turn_count} | End After Max Rollbacks",
                            f"Ending sub-agent loop after {consecutive_rollbacks} consecutive MCP format errors",
                        )
                        break
                elif any(
                    keyword in assistant_response_text for keyword in refusal_keywords
                ):
                    # If we haven't reached rollback limit, rollback and retry
                    if consecutive_rollbacks < self.MAX_CONSECUTIVE_ROLLBACKS - 1:
                        turn_count -= 1
                        consecutive_rollbacks += 1
                        matched_keywords = [
                            kw
                            for kw in refusal_keywords
                            if kw in assistant_response_text
                        ]
                        if message_history[-1]["role"] == "assistant":
                            message_history.pop()
                        self.task_log.log_step(
                            "warning",
                            f"{sub_agent_name} | Turn: {turn_count} | Rollback",
                            f"LLM refused to answer - found refusal keywords: {matched_keywords}. Consecutive rollbacks: {consecutive_rollbacks}/{self.MAX_CONSECUTIVE_ROLLBACKS}, Total attempts: {total_attempts}/{max_attempts}",
                        )
                        continue
                    else:
                        # Reached rollback limit, allow the loop to end naturally
                        matched_keywords = [
                            kw
                            for kw in refusal_keywords
                            if kw in assistant_response_text
                        ]
                        self.task_log.log_step(
                            "warning",
                            f"{sub_agent_name} | Turn: {turn_count} | End After Max Rollbacks",
                            f"Ending sub-agent loop after {consecutive_rollbacks} consecutive refusals with keywords: {matched_keywords}",
                        )
                        break
                else:
                    self.task_log.log_step(
                        "info",
                        f"{sub_agent_name} | Turn: {turn_count} | LLM Call",
                        f"No tool calls found in {sub_agent_name}, ending on turn {turn_count}",
                    )

                    pattern = r"```(\w+)?\s*(.*)```"
                    match = re.search(pattern, assistant_response_text, re.DOTALL)

                    if match:
                        # Found code block, use it as final answer
                        final_answer_text = match.group(2).strip()
                        skip_summary = True

                        self.task_log.log_step(
                            "info",
                            f"{sub_agent_name} | Skip Summary",
                            f"Found code block in last turn output, skipping summary phase",
                        )
                    break

            # Execute tool calls
            tool_calls_data = []
            all_tool_results_content_with_id = []
            should_rollback_turn = False

            for call in tool_calls:
                server_name = call["server_name"]
                tool_name = call["tool_name"]
                arguments = call["arguments"]
                call_id = call["id"]

                # Fix common parameter name mistakes
                arguments = self._fix_tool_call_arguments(tool_name, arguments)

                self.task_log.log_step(
                    "info",
                    f"{sub_agent_name} | Turn: {turn_count} | Tool Call",
                    f"Executing {tool_name} on {server_name}",
                )

                call_start_time = time.time()
                try:
                    # Check for duplicate query before sending stream events
                    query_str = self._get_query_str_from_tool_call(tool_name, arguments)
                    if query_str:
                        cache_name = sub_agent_id + "_" + tool_name
                        self.used_queries.setdefault(cache_name, defaultdict(int))
                        count = self.used_queries[cache_name][query_str]
                        if count > 0:
                            # If we haven't reached rollback limit, rollback and retry
                            if (
                                consecutive_rollbacks
                                < self.MAX_CONSECUTIVE_ROLLBACKS - 1
                            ):
                                message_history.pop()
                                turn_count -= 1
                                consecutive_rollbacks += 1
                                should_rollback_turn = True
                                self.task_log.log_step(
                                    "warning",
                                    f"{sub_agent_name} | Turn: {turn_count} | Rollback",
                                    f"Duplicate query detected - tool: {tool_name}, query: '{query_str}', previous count: {count}. Consecutive rollbacks: {consecutive_rollbacks}/{self.MAX_CONSECUTIVE_ROLLBACKS}, Total attempts: {total_attempts}/{max_attempts}",
                                )
                                break  # Exit inner for loop, then continue outer while loop
                            else:
                                # Reached rollback limit, allow execution but add feedback
                                self.task_log.log_step(
                                    "warning",
                                    f"{sub_agent_name} | Turn: {turn_count} | Allow Duplicate",
                                    f"Allowing duplicate query after {consecutive_rollbacks} rollbacks - tool: {tool_name}, query: '{query_str}', previous count: {count}",
                                )

                    # Send stream event only after duplicate check
                    tool_call_id = await self._stream_tool_call(tool_name, arguments)

                    # Execute tool call
                    tool_result = await self.sub_agent_tool_managers[
                        sub_agent_name
                    ].execute_tool_call(server_name, tool_name, arguments)

                    # Update query count if successful
                    if query_str and "error" not in tool_result:
                        self.used_queries[cache_name][query_str] += 1

                    # Only in demo mode: truncate scrape results to 20,000 chars
                    tool_result = self.post_process_tool_call_result(
                        tool_name, tool_result
                    )
                    result = (
                        tool_result.get("result")
                        if tool_result.get("result")
                        else tool_result.get("error")
                    )

                    # Check for "Unknown tool:" error and rollback
                    if str(result).startswith("Unknown tool:"):
                        # If we haven't reached rollback limit, rollback and retry
                        if consecutive_rollbacks < self.MAX_CONSECUTIVE_ROLLBACKS - 1:
                            message_history.pop()
                            turn_count -= 1
                            consecutive_rollbacks += 1
                            should_rollback_turn = True
                            self.task_log.log_step(
                                "warning",
                                f"{sub_agent_name} | Turn: {turn_count} | Rollback",
                                f"Unknown tool error - tool: {tool_name}, error: '{str(result)[:200]}'. Consecutive rollbacks: {consecutive_rollbacks}/{self.MAX_CONSECUTIVE_ROLLBACKS}, Total attempts: {total_attempts}/{max_attempts}",
                            )
                            break  # Exit inner for loop, then continue outer while loop
                        else:
                            # Reached rollback limit, allow error to be sent to LLM as feedback
                            self.task_log.log_step(
                                "warning",
                                f"{sub_agent_name} | Turn: {turn_count} | Allow Error Feedback",
                                f"Allowing unknown tool error to be sent to LLM after {consecutive_rollbacks} rollbacks - tool: {tool_name}, error: '{str(result)[:200]}'",
                            )

                    await self._stream_tool_call(
                        tool_name, {"result": result}, tool_call_id=tool_call_id
                    )
                    call_end_time = time.time()
                    call_duration_ms = int((call_end_time - call_start_time) * 1000)

                    self.task_log.log_step(
                        "info",
                        f"{sub_agent_name} | Turn: {turn_count} | Tool Call",
                        f"Tool {tool_name} completed in {call_duration_ms}ms",
                    )

                    tool_calls_data.append(
                        {
                            "server_name": server_name,
                            "tool_name": tool_name,
                            "arguments": arguments,
                            "result": tool_result,
                            "duration_ms": call_duration_ms,
                            "call_time": get_utc_plus_8_time(),
                        }
                    )

                except Exception as e:
                    call_end_time = time.time()
                    call_duration_ms = int((call_end_time - call_start_time) * 1000)

                    tool_calls_data.append(
                        {
                            "server_name": server_name,
                            "tool_name": tool_name,
                            "arguments": arguments,
                            "error": str(e),
                            "duration_ms": call_duration_ms,
                            "call_time": get_utc_plus_8_time(),
                        }
                    )
                    tool_result = {
                        "error": f"Tool call failed: {str(e)}",
                        "server_name": server_name,
                        "tool_name": tool_name,
                    }
                    self.task_log.log_step(
                        "error",
                        f"{sub_agent_name} | Turn: {turn_count} | Tool Call",
                        f"Tool {tool_name} failed to execute: {str(e)}",
                    )

                tool_result_for_llm = self.output_formatter.format_tool_result_for_user(
                    tool_result
                )

                all_tool_results_content_with_id.append((call_id, tool_result_for_llm))

            # Check if we need to rollback and retry the turn
            if should_rollback_turn:
                continue  # Continue outer while loop

            # Reset consecutive rollbacks on successful execution
            if consecutive_rollbacks > 0:
                self.task_log.log_step(
                    "info",
                    f"{sub_agent_name} | Turn: {turn_count} | Recovery",
                    f"Successfully recovered after {consecutive_rollbacks} consecutive rollbacks",
                )
            consecutive_rollbacks = 0

            # Record tool calls to current sub-agent turn
            message_history = self.llm_client.update_message_history(
                message_history, all_tool_results_content_with_id
            )

            # Generate summary_prompt to check token limits
            temp_summary_prompt = generate_agent_summarize_prompt(
                task_description,
                agent_type=sub_agent_name,
                language=self.task_lang,
            )

            pass_length_check, message_history = self.llm_client.ensure_summary_context(
                message_history, temp_summary_prompt
            )

            # Check if current context will exceed limits, if so automatically rollback messages and trigger summary
            if not pass_length_check:
                # Context exceeded limits, set turn_count to trigger summary
                turn_count = max_turns
                self.task_log.log_step(
                    "info",
                    f"{sub_agent_name} | Turn: {turn_count} | Context Limit Reached",
                    "Context limit reached, triggering summary",
                )
                break

        # Record browsing agent loop end
        if turn_count >= max_turns:
            self.task_log.log_step(
                "info",
                f"{sub_agent_name} | Max Turns Reached / Context Limit Reached",
                f"Reached maximum turns ({max_turns}) or context limit reached",
            )
        else:
            self.task_log.log_step(
                "info",
                f"{sub_agent_name} | Main Loop Completed",
                f"Main loop completed after {turn_count} turns",
            )

        if not skip_summary:
            # Final summary
            self.task_log.log_step(
                "info",
                f"{sub_agent_name} | Final Summary",
                f"Generating {sub_agent_name} final summary",
            )

            # Generate sub agent summary prompt
            summary_prompt = generate_agent_summarize_prompt(
                task_description,
                agent_type=sub_agent_name,
                language=self.task_lang,
            )

            if message_history[-1]["role"] == "user":
                message_history.pop()
            message_history.append({"role": "user", "content": summary_prompt})

            await self._stream_tool_call(
                "Partial Summary", {}, tool_call_id=str(uuid.uuid4())
            )

            # Use unified LLM call processing to generate final summary
            (
                final_answer_text,
                should_break,
                tool_calls_info,
                message_history,
            ) = await self._handle_llm_call(
                system_prompt,
                message_history,
                tool_definitions,
                turn_count + 1,
                f"{sub_agent_name} | Final summary",
                agent_type=sub_agent_name,
            )

            if final_answer_text:
                self.task_log.log_step(
                    "info",
                    f"{sub_agent_name} | Final Summary",
                    "Final summary generated successfully",
                )
            else:
                final_answer_text = (
                    f"No final summary generated by sub agent {sub_agent_name}."
                )
                self.task_log.log_step(
                    "error",
                    f"{sub_agent_name} | Final Summary",
                    "Unable to generate final summary",
                )

        self.task_log.sub_agent_message_history_sessions[
            self.task_log.current_sub_agent_session_id
        ] = {"system_prompt": system_prompt, "message_history": message_history}

        self.task_log.save()
        self.task_log.end_sub_agent_session(sub_agent_name)

        # Remove thinking content in tool response
        # For the return result of a sub-agent, the content within the `<think>` tags is unnecessary in any case.
        final_answer_text = final_answer_text.split("<think>")[-1].strip()
        final_answer_text = final_answer_text.split("</think>")[-1].strip()

        # Stream sub-agent end
        await self._stream_end_llm(display_name)
        await self._stream_end_agent(display_name, sub_agent_id)

        # Return final answer instead of conversation log, so main agent can use it directly
        return final_answer_text

    async def run_main_agent(
        self, task_description, task_file_name=None, task_id="default_task"
    ):
        """Execute the main end-to-end task"""
        workflow_id = await self._stream_start_workflow(task_description)

        self.task_log.log_step("info", "Main Agent", f"Start task with id: {task_id}")
        self.task_log.log_step(
            "info", "Main Agent", f"Task description: {task_description}"
        )
        if task_file_name:
            self.task_log.log_step(
                "info", "Main Agent", f"Associated file: {task_file_name}"
            )

        # Process input
        initial_user_content, processed_task_desc = process_input(
            task_description, task_file_name
        )
        message_history = [{"role": "user", "content": initial_user_content}]

        # Record initial user input
        user_input = processed_task_desc
        if task_file_name:
            user_input += f"\n[Attached file: {task_file_name}]"

        # Get tool definitions
        if not self.tool_definitions:
            tool_definitions = (
                await self.main_agent_tool_manager.get_all_tool_definitions()
            )
            if self.cfg.agent.sub_agents is not None:
                tool_definitions += expose_sub_agents_as_tools(
                    self.cfg.agent.sub_agents
                )
        else:
            tool_definitions = self.tool_definitions
        if not tool_definitions:
            self.task_log.log_step(
                "warning",
                "Main Agent | Tool Definitions",
                "Warning: No tool definitions found. LLM cannot use any tools.",
            )

        # Generate system prompt
        system_prompt = self.llm_client.generate_agent_system_prompt(
            date=date.today(),
            mcp_servers=tool_definitions,
        ) + generate_agent_specific_system_prompt(agent_type="main")
        system_prompt = system_prompt.strip()

        # Main loop: LLM <-> Tools
        max_turns = self.cfg.agent.main_agent.max_turns
        turn_count = 0
        total_attempts = 0
        max_attempts = max_turns + 200
        consecutive_rollbacks = 0

        self.current_agent_id = await self._stream_start_agent("main")
        await self._stream_start_llm("main")
        while turn_count < max_turns and total_attempts < max_attempts:
            turn_count += 1
            total_attempts += 1
            if consecutive_rollbacks >= self.MAX_CONSECUTIVE_ROLLBACKS:
                self.task_log.log_step(
                    "error",
                    "Main Agent | Too Many Rollbacks",
                    f"Reached {consecutive_rollbacks} consecutive rollbacks (limit: {self.MAX_CONSECUTIVE_ROLLBACKS}), breaking loop. Total attempts: {total_attempts}/{max_attempts}",
                )
                break

            self.task_log.save()

            # Use unified LLM call processing
            (
                assistant_response_text,
                should_break,
                tool_calls,
                message_history,
            ) = await self._handle_llm_call(
                system_prompt,
                message_history,
                tool_definitions,
                turn_count,
                f"Main agent | Turn: {turn_count}",
                agent_type="main",
            )

            # Process LLM response
            if assistant_response_text:
                text_response = extract_llm_response_text(assistant_response_text)
                if text_response:
                    await self._stream_tool_call("show_text", {"text": text_response})

                # Try to extract boxed content from this turn's response
                boxed_content = self.output_formatter._extract_boxed_content(
                    assistant_response_text
                )
                if boxed_content:
                    self.intermediate_boxed_answers.append(boxed_content)

                if should_break:
                    self.task_log.log_step(
                        "info",
                        f"Main Agent | Turn: {turn_count} | LLM Call",
                        "should break is True, breaking the loop",
                    )
                    break
            else:
                self.task_log.log_step(
                    "info",
                    f"Main Agent | Turn: {turn_count} | LLM Call",
                    "No valid response from LLM, retrying",
                )
                await asyncio.sleep(5)
                continue

            if not tool_calls:
                if any(mcp_tag in assistant_response_text for mcp_tag in mcp_tags):
                    # If we haven't reached rollback limit, rollback and retry
                    if consecutive_rollbacks < self.MAX_CONSECUTIVE_ROLLBACKS - 1:
                        turn_count -= 1
                        consecutive_rollbacks += 1
                        if message_history[-1]["role"] == "assistant":
                            message_history.pop()
                        self.task_log.log_step(
                            "warning",
                            f"Main Agent | Turn: {turn_count} | Rollback",
                            f"Tool call format incorrect - found MCP tags in response. Consecutive rollbacks: {consecutive_rollbacks}/{self.MAX_CONSECUTIVE_ROLLBACKS}, Total attempts: {total_attempts}/{max_attempts}",
                        )
                        continue
                    else:
                        # Reached rollback limit, allow the loop to end naturally
                        self.task_log.log_step(
                            "warning",
                            f"Main Agent | Turn: {turn_count} | End After Max Rollbacks",
                            f"Ending main agent loop after {consecutive_rollbacks} consecutive MCP format errors",
                        )
                        break
                elif any(
                    keyword in assistant_response_text for keyword in refusal_keywords
                ):
                    # If we haven't reached rollback limit, rollback and retry
                    if consecutive_rollbacks < self.MAX_CONSECUTIVE_ROLLBACKS - 1:
                        turn_count -= 1
                        consecutive_rollbacks += 1
                        matched_keywords = [
                            kw
                            for kw in refusal_keywords
                            if kw in assistant_response_text
                        ]
                        if message_history[-1]["role"] == "assistant":
                            message_history.pop()
                        self.task_log.log_step(
                            "warning",
                            f"Main Agent | Turn: {turn_count} | Rollback",
                            f"LLM refused to answer - found refusal keywords: {matched_keywords}. Consecutive rollbacks: {consecutive_rollbacks}/{self.MAX_CONSECUTIVE_ROLLBACKS}, Total attempts: {total_attempts}/{max_attempts}",
                        )
                        continue
                    else:
                        # Reached rollback limit, allow the loop to end naturally
                        matched_keywords = [
                            kw
                            for kw in refusal_keywords
                            if kw in assistant_response_text
                        ]
                        self.task_log.log_step(
                            "warning",
                            f"Main Agent | Turn: {turn_count} | End After Max Rollbacks",
                            f"Ending main agent loop after {consecutive_rollbacks} consecutive refusals with keywords: {matched_keywords}",
                        )
                        break
                else:
                    self.task_log.log_step(
                        "info",
                        f"Main Agent | Turn: {turn_count} | LLM Call",
                        "LLM did not request tool usage, ending process.",
                    )
                    break

            # Execute tool calls (execute in order)
            tool_calls_data = []
            all_tool_results_content_with_id = []
            should_rollback_turn = False
            main_agent_last_call_tokens = self.llm_client.last_call_tokens

            for call in tool_calls:
                server_name = call["server_name"]
                tool_name = call["tool_name"]
                arguments = call["arguments"]
                call_id = call["id"]

                # Fix common parameter name mistakes
                arguments = self._fix_tool_call_arguments(tool_name, arguments)

                call_start_time = time.time()
                try:
                    if server_name.startswith("agent-") and self.cfg.agent.sub_agents:
                        # Check for duplicate query before sending stream events
                        query_str = self._get_query_str_from_tool_call(
                            tool_name, arguments
                        )
                        if query_str:
                            cache_name = "main_" + tool_name
                            self.used_queries.setdefault(cache_name, defaultdict(int))
                            count = self.used_queries[cache_name][query_str]
                            if count > 0:
                                # If we haven't reached rollback limit, rollback and retry
                                if (
                                    consecutive_rollbacks
                                    < self.MAX_CONSECUTIVE_ROLLBACKS - 1
                                ):
                                    message_history.pop()
                                    turn_count -= 1
                                    consecutive_rollbacks += 1
                                    should_rollback_turn = True
                                    self.task_log.log_step(
                                        "warning",
                                        f"Main Agent | Turn: {turn_count} | Rollback",
                                        f"Duplicate sub-agent query detected - agent: {server_name}, query: '{query_str}', previous count: {count}. Consecutive rollbacks: {consecutive_rollbacks}/{self.MAX_CONSECUTIVE_ROLLBACKS}, Total attempts: {total_attempts}/{max_attempts}",
                                    )
                                    break  # Exit inner for loop, then continue outer while loop
                                else:
                                    # Reached rollback limit, allow execution but add feedback
                                    self.task_log.log_step(
                                        "warning",
                                        f"Main Agent | Turn: {turn_count} | Allow Duplicate",
                                        f"Allowing duplicate sub-agent query after {consecutive_rollbacks} rollbacks - agent: {server_name}, query: '{query_str}', previous count: {count}",
                                    )

                        # Send stream events only after duplicate check
                        await self._stream_end_llm("main")
                        await self._stream_end_agent("main", self.current_agent_id)

                        # Execute sub-agent
                        sub_agent_result = await self.run_sub_agent(
                            server_name,
                            arguments["subtask"],
                        )

                        # Update query count if successful
                        if query_str:
                            self.used_queries[cache_name][query_str] += 1

                        tool_result = {
                            "server_name": server_name,
                            "tool_name": tool_name,
                            "result": sub_agent_result,
                        }
                        self.current_agent_id = await self._stream_start_agent(
                            "main", display_name="Summarizing"
                        )
                        await self._stream_start_llm("main", display_name="Summarizing")
                    else:
                        # Check for duplicate query before sending stream events
                        query_str = self._get_query_str_from_tool_call(
                            tool_name, arguments
                        )
                        if query_str:
                            cache_name = "main_" + tool_name
                            self.used_queries.setdefault(cache_name, defaultdict(int))
                            count = self.used_queries[cache_name][query_str]
                            if count > 0:
                                # If we haven't reached rollback limit, rollback and retry
                                if (
                                    consecutive_rollbacks
                                    < self.MAX_CONSECUTIVE_ROLLBACKS - 1
                                ):
                                    message_history.pop()
                                    turn_count -= 1
                                    consecutive_rollbacks += 1
                                    should_rollback_turn = True
                                    self.task_log.log_step(
                                        "warning",
                                        f"Main Agent | Turn: {turn_count} | Rollback",
                                        f"Duplicate tool query detected - tool: {tool_name}, query: '{query_str}', previous count: {count}. Consecutive rollbacks: {consecutive_rollbacks}/{self.MAX_CONSECUTIVE_ROLLBACKS}, Total attempts: {total_attempts}/{max_attempts}",
                                    )
                                    break  # Exit inner for loop, then continue outer while loop
                                else:
                                    # Reached rollback limit, allow execution but add feedback
                                    self.task_log.log_step(
                                        "warning",
                                        f"Main Agent | Turn: {turn_count} | Allow Duplicate",
                                        f"Allowing duplicate query after {consecutive_rollbacks} rollbacks - tool: {tool_name}, query: '{query_str}', previous count: {count}",
                                    )

                        # Send stream event only after duplicate check
                        tool_call_id = await self._stream_tool_call(
                            tool_name, arguments
                        )

                        # Execute tool call
                        tool_result = (
                            await self.main_agent_tool_manager.execute_tool_call(
                                server_name=server_name,
                                tool_name=tool_name,
                                arguments=arguments,
                            )
                        )

                        # Update query count if successful
                        if query_str and "error" not in tool_result:
                            self.used_queries[cache_name][query_str] += 1
                        # Only in demo mode: truncate scrape results to 20,000 chars
                        tool_result = self.post_process_tool_call_result(
                            tool_name, tool_result
                        )
                        result = (
                            tool_result.get("result")
                            if tool_result.get("result")
                            else tool_result.get("error")
                        )

                        # Check for "Unknown tool:" error and rollback
                        if str(result).startswith("Unknown tool:"):
                            # If we haven't reached rollback limit, rollback and retry
                            if (
                                consecutive_rollbacks
                                < self.MAX_CONSECUTIVE_ROLLBACKS - 1
                            ):
                                message_history.pop()
                                turn_count -= 1
                                consecutive_rollbacks += 1
                                should_rollback_turn = True
                                self.task_log.log_step(
                                    "warning",
                                    f"Main Agent | Turn: {turn_count} | Rollback",
                                    f"Unknown tool error - tool: {tool_name}, error: '{str(result)[:200]}'. Consecutive rollbacks: {consecutive_rollbacks}/{self.MAX_CONSECUTIVE_ROLLBACKS}, Total attempts: {total_attempts}/{max_attempts}",
                                )
                                break  # Exit inner for loop, then continue outer while loop
                            else:
                                # Reached rollback limit, allow error to be sent to LLM as feedback
                                self.task_log.log_step(
                                    "warning",
                                    f"Main Agent | Turn: {turn_count} | Allow Error Feedback",
                                    f"Allowing unknown tool error to be sent to LLM after {consecutive_rollbacks} rollbacks - tool: {tool_name}, error: '{str(result)[:200]}'",
                                )

                        await self._stream_tool_call(
                            tool_name, {"result": result}, tool_call_id=tool_call_id
                        )

                    call_end_time = time.time()
                    call_duration_ms = int((call_end_time - call_start_time) * 1000)

                    tool_calls_data.append(
                        {
                            "server_name": server_name,
                            "tool_name": tool_name,
                            "arguments": arguments,
                            "result": tool_result,
                            "duration_ms": call_duration_ms,
                            "call_time": get_utc_plus_8_time(),
                        }
                    )
                    self.task_log.log_step(
                        "info",
                        f"Main Agent | Turn: {turn_count} | Tool Call",
                        f"Tool {tool_name} completed in {call_duration_ms}ms",
                    )

                except Exception as e:
                    call_end_time = time.time()
                    call_duration_ms = int((call_end_time - call_start_time) * 1000)

                    tool_calls_data.append(
                        {
                            "server_name": server_name,
                            "tool_name": tool_name,
                            "arguments": arguments,
                            "error": str(e),
                            "duration_ms": call_duration_ms,
                            "call_time": get_utc_plus_8_time(),
                        }
                    )
                    tool_result = {
                        "server_name": server_name,
                        "tool_name": tool_name,
                        "error": str(e),
                    }
                    self.task_log.log_step(
                        "error",
                        f"Main Agent | Turn: {turn_count} | Tool Call",
                        f"Tool {tool_name} failed to execute: {str(e)}",
                    )

                # Format results to feedback to LLM (more concise)
                tool_result_for_llm = self.output_formatter.format_tool_result_for_user(
                    tool_result
                )

                all_tool_results_content_with_id.append((call_id, tool_result_for_llm))

            # Check if we need to rollback and retry the turn
            if should_rollback_turn:
                continue  # Continue outer while loop

            # Reset consecutive rollbacks on successful execution
            if consecutive_rollbacks > 0:
                self.task_log.log_step(
                    "info",
                    f"Main Agent | Turn: {turn_count} | Recovery",
                    f"Successfully recovered after {consecutive_rollbacks} consecutive rollbacks",
                )
            consecutive_rollbacks = 0

            # Update 'last_call_tokens'
            self.llm_client.last_call_tokens = main_agent_last_call_tokens

            # Update message history with tool calls data (llm client specific)
            message_history = self.llm_client.update_message_history(
                message_history, all_tool_results_content_with_id
            )

            self.task_log.main_agent_message_history = {
                "system_prompt": system_prompt,
                "message_history": message_history,
            }
            self.task_log.save()

            # Assess current context length, determine if we need to trigger summary
            temp_summary_prompt = generate_agent_summarize_prompt(
                task_description,
                agent_type="main",
            )

            pass_length_check, message_history = self.llm_client.ensure_summary_context(
                message_history, temp_summary_prompt
            )

            # Check if current context will exceed limits, if so automatically rollback messages and trigger summary
            if not pass_length_check:
                turn_count = max_turns
                self.task_log.log_step(
                    "warning",
                    f"Main Agent | Turn: {turn_count} | Context Limit Reached",
                    "Context limit reached, triggering summary",
                )
                break

        await self._stream_end_llm("main")
        await self._stream_end_agent("main", self.current_agent_id)

        # Record main loop end
        if turn_count >= max_turns:
            self.task_log.log_step(
                "warning",
                "Main Agent | Max Turns Reached / Context Limit Reached",
                f"Reached maximum turns ({max_turns}) or context limit reached",
            )
        else:
            self.task_log.log_step(
                "info",
                "Main Agent | Main Loop Completed",
                f"Main loop completed after {turn_count} turns",
            )

        # Final summary
        self.task_log.log_step(
            "info", "Main Agent | Final Summary", "Generating final summary"
        )

        self.current_agent_id = await self._stream_start_agent("Final Summary")
        await self._stream_start_llm("Final Summary")

        # Generate summary prompt (generate only once)
        summary_prompt = generate_agent_summarize_prompt(
            task_description,
            agent_type="main",
        )

        if message_history[-1]["role"] == "user":
            message_history.pop(-1)
        message_history.append({"role": "user", "content": summary_prompt})

        # Retry mechanism for generating boxed answer
        final_answer_text = None
        final_boxed_answer = None
        final_summary = ""
        usage_log = ""

        for retry_idx in range(self.MAX_FINAL_ANSWER_RETRIES):
            # Use unified LLM call processing
            (
                final_answer_text,
                should_break,
                tool_calls_info,
                message_history,
            ) = await self._handle_llm_call(
                system_prompt,
                message_history,
                tool_definitions,
                turn_count + 1 + retry_idx,
                f"Main agent | Final Summary (attempt {retry_idx + 1}/{self.MAX_FINAL_ANSWER_RETRIES})",
                agent_type="main",
            )

            if final_answer_text:
                # Try to extract boxed answer
                final_summary, final_boxed_answer, usage_log = (
                    self.output_formatter.format_final_summary_and_log(
                        final_answer_text, self.llm_client
                    )
                )

                # Check if we got a valid boxed answer
                if (
                    final_boxed_answer
                    != "No \\boxed{} content found in the final answer."
                ):
                    self.task_log.log_step(
                        "info",
                        "Main Agent | Final Answer",
                        f"Boxed answer found on attempt {retry_idx + 1}",
                    )
                    break
                else:
                    self.task_log.log_step(
                        "warning",
                        "Main Agent | Final Answer",
                        f"No boxed answer on attempt {retry_idx + 1}, retrying...",
                    )
                    # Remove the failed assistant response before retry
                    if retry_idx < self.MAX_FINAL_ANSWER_RETRIES - 1:
                        if (
                            message_history
                            and message_history[-1]["role"] == "assistant"
                        ):
                            message_history.pop()
            else:
                self.task_log.log_step(
                    "warning",
                    "Main Agent | Final Answer",
                    f"Failed to generate answer on attempt {retry_idx + 1}",
                )
                # Remove the failed assistant response before retry
                if retry_idx < self.MAX_FINAL_ANSWER_RETRIES - 1:
                    if message_history and message_history[-1]["role"] == "assistant":
                        message_history.pop()

        self.task_log.main_agent_message_history = {
            "system_prompt": system_prompt,
            "message_history": message_history,
        }
        self.task_log.save()

        # Final validation and fallback
        if not final_answer_text:
            final_answer_text = "No final answer generated."
            final_summary = final_answer_text
            final_boxed_answer = "No \\boxed{} content found in the final answer."
            self.task_log.log_step(
                "error",
                "Main Agent | Final Answer",
                "Unable to generate final answer after all retries",
            )
        else:
            self.task_log.log_step(
                "info",
                "Main Agent | Final Answer",
                f"Final answer content:\n\n{final_answer_text}",
            )

        # Fallback to intermediate answer if still no boxed answer
        if (
            final_boxed_answer == "No \\boxed{} content found in the final answer."
            and self.intermediate_boxed_answers
        ):
            final_boxed_answer = self.intermediate_boxed_answers[-1]
            self.task_log.log_step(
                "info",
                "Main Agent | Final Answer",
                f"Using intermediate boxed answer as fallback: {final_boxed_answer}",
            )

        await self._stream_tool_call("show_text", {"text": final_boxed_answer})
        await self._stream_end_llm("Final Summary")
        await self._stream_end_agent("Final Summary", self.current_agent_id)
        await self._stream_end_workflow(workflow_id)

        self.task_log.log_step(
            "info", "Main Agent | Usage Calculation", f"Usage log: {usage_log}"
        )

        self.task_log.log_step(
            "info",
            "Main Agent | Final boxed answer",
            f"Final boxed answer:\n\n{final_boxed_answer}",
        )

        self.task_log.log_step(
            "info",
            "Main Agent | Task Completed",
            f"Main agent task {task_id} completed successfully",
        )
        gc.collect()
        return final_summary, final_boxed_answer

    async def run_report_workflow(self, task_description: str) -> str:
        """High-level multi-stage report workflow with charts support.

        Stages:
        1) Plan: generate structured outline (JSON-like) with image information and visual elements planning, optionally with research
        2) Section writing: generate each section's content based on outline with chart/image generation
        3) Polish: global refinement + formatting to final markdown report with proper figure handling
        """

        # Initialize result directories at the start
        self._initialize_result_directories()
        self.task_lang = self._detect_language(task_description)

        workflow_start_time = time.time()

        outline = await self.run_plan_phase(task_description)
        self._save_intermediate_output("01_outline", outline, "json")

        outline_new = await self.run_visual_phase(outline)
        self._save_intermediate_output("02_outline_with_visuals", outline_new, "json")

        report = await self.run_write_phase(outline_new)
        self._save_intermediate_output("03_report", report, "json")

        final_report = await self.run_polish_phase(report)
        self._save_intermediate_output("04_final_report", final_report, "md")

        workflow_elapsed = time.time() - workflow_start_time

        self.task_log.log_step(
            "info",
            "Report Workflow | Completed",
            f"Report workflow completed in {workflow_elapsed:.2f} seconds",
        )

        return final_report

    async def run_plan_phase(self, task_description: str) -> Dict[str, Any]:
        """Execute plan phase using agent-planner to generate structured report outline."""

        plan_phase_start_time = time.time()
        self.task_log.log_step(
            "info",
            "Report Workflow | Plan",
            "Starting plan phase to generate report outline",
        )

        # Check if plan_agent is available
        if "agent-planner" not in self.sub_agent_tool_managers:
            self.task_log.log_step(
                "warning",
                "Report Workflow | Plan",
                "agent-planner not available, using fallback outline",
            )
            return self._get_fallback_outline(task_description)

        try:
            # Call plan_agent using existing run_sub_agent
            plan_result = await self.run_sub_agent("agent-planner", task_description)

            # print("Plan agent result:", plan_result)

            if not plan_result or not isinstance(plan_result, str):
                raise ValueError("agent-planner returned empty or invalid result")

            # Parse result to extract outline
            outline = self._parse_json_result(plan_result)

            # Validate outline
            if (
                not outline
                or not isinstance(outline, dict)
                or "sections" not in outline
            ):
                raise ValueError("Invalid outline structure from agent-planner")

            # Ensure has_image fields
            for section in outline.get("sections", []):
                if "has_image" not in section:
                    section["has_image"] = (
                        bool(section.get("visual_elements"))
                        and len(section.get("visual_elements", [])) > 0
                    )

            # Log success
            try:
                outline_str = json.dumps(outline, ensure_ascii=False, indent=2)
            except Exception:
                outline_str = str(outline)
            if len(outline_str) > 2000:
                outline_str = outline_str[:2000] + "... [truncated]"

            self.task_log.log_step(
                "info",
                "Report Workflow | Plan | Outline Generated",
                f"Successfully generated outline:\n{outline_str}",
            )

            plan_phase_elapsed = time.time() - plan_phase_start_time

            self.task_log.log_step(
                "info",
                "Report Workflow | Plan | Summary",
                f"Plan phase completed in {plan_phase_elapsed:.2f} seconds",
            )

            return outline

        except Exception as e:
            self.task_log.log_step(
                "error",
                "Report Workflow | Plan",
                f"agent-planner failed: {e}. Using fallback outline.",
            )
            return self._get_fallback_outline(task_description)

    def _parse_json_result(self, result: str) -> Dict[str, Any]:
        """Parse agent-planner result using json-repair library."""

        # Remove BOM and invisible characters
        text = result.strip("\ufeff\u200b\u200c\u200d")

        # Use json-repair to fix and parse
        try:
            repaired = repair_json(text)
            output = json.loads(repaired)

            if not isinstance(output, dict):
                raise ValueError(f"Invalid type: {type(output)}")

            return output
        except Exception as e:
            raise ValueError(f"Could not parse JSON: {e}")

    def _get_fallback_outline(self, task_description: str) -> Dict[str, Any]:
        """Generate basic fallback outline."""

        if self.task_lang == "zh":
            return {
                "title": task_description[:80],
                "sections": [
                    {
                        "id": "1",
                        "title": "引言",
                        "summary": "介绍主题背景、研究目标和报告范围。",
                        "word_count": "~200字",
                        "has_image": False,
                        "visual_elements": [],
                    },
                    {
                        "id": "2",
                        "title": "主要内容",
                        "summary": "基于研究的详细分析和论述。",
                        "word_count": "~2000字",
                        "has_image": False,
                        "visual_elements": [],
                    },
                    {
                        "id": "3",
                        "title": "结论",
                        "summary": "总结主要发现和建议。",
                        "word_count": "~300字",
                        "has_image": False,
                        "visual_elements": [],
                    },
                ],
            }
        else:
            return {
                "title": task_description[:80],
                "sections": [
                    {
                        "id": "1",
                        "title": "Introduction",
                        "summary": "Introduce the topic background, objectives, and scope.",
                        "word_count": "~200 words",
                        "has_image": False,
                        "visual_elements": [],
                    },
                    {
                        "id": "2",
                        "title": "Main Content",
                        "summary": "Detailed analysis and discussion based on research.",
                        "word_count": "~2000 words",
                        "has_image": False,
                        "visual_elements": [],
                    },
                    {
                        "id": "3",
                        "title": "Conclusion",
                        "summary": "Summary of findings and recommendations.",
                        "word_count": "~300 words",
                        "has_image": False,
                        "visual_elements": [],
                    },
                ],
            }

    async def run_visual_phase(
        self,
        outline: dict,
    ) -> dict:
        """Generate charts and search images for all sections in the outline.

        Args:
            outline: Report outline with sections containing visual_elements

        Returns:
            Updated outline with chart and image information added to visual_elements
        """
        visual_phase_start_time = time.time()
        self.task_log.log_step(
            "info",
            "Report Workflow | Visual",
            "Starting visual phase to generate charts and search images",
        )

        # Validate outline structure
        if not outline or not isinstance(outline, dict):
            self.task_log.log_step(
                "error",
                "Report Workflow | Visual",
                "Invalid outline structure, skipping visual phase",
            )
            return outline

        sections = outline.get("sections", [])
        if not sections:
            self.task_log.log_step(
                "warning",
                "Report Workflow | Visual",
                "No sections found in outline, skipping visual phase",
            )
            return outline

        # Check if visual agents are available
        if "agent-chart-generator" not in self.sub_agent_tool_managers:
            self.task_log.log_step(
                "warning",
                "Report Workflow | Visual",
                "agent-chart-generator not available, skipping visual phase",
            )
            return outline

        report_title = outline.get("title", "")
        total_charts_requested = 0
        total_charts_generated = 0
        total_charts_failed = 0
        total_images_requested = 0
        total_images_found = 0
        total_images_failed = 0

        # Process each section
        for section_idx, section in enumerate(sections):
            section_id = section.get("id", str(section_idx + 1))
            section_title = section.get("title", f"Section {section_id}")
            section_summary = section.get("summary", "")
            visual_elements = section.get("visual_elements", [])
            research_notes = section.get("research_notes", [])

            if not visual_elements:
                self.task_log.log_step(
                    "info",
                    f"Report Workflow | Visual | Section {section_id}",
                    f"No visual elements in section '{section_title}', skipping",
                )
                continue

            self.task_log.log_step(
                "info",
                f"Report Workflow | Visual | Section {section_id}",
                f"Processing {len(visual_elements)} visual element(s) for section '{section_title}'",
            )

            # Process each visual element in the section
            updated_visual_elements = []

            for elem_idx, visual_element in enumerate(visual_elements):
                if not isinstance(visual_element, str):
                    self.task_log.log_step(
                        "warning",
                        f"Report Workflow | Visual | Section {section_id}",
                        f"Invalid visual element format at index {elem_idx}, skipping",
                    )
                    continue

                # Check if it's an image search request
                if visual_element.startswith("tool-image-search:"):
                    total_images_requested += 1
                    visual_element = visual_element.replace(
                        "tool-image-search:", ""
                    ).strip()

                    self.task_log.log_step(
                        "info",
                        f"Report Workflow | Visual | Section {section_id} | Image {elem_idx + 1}",
                        f"Searching image: {visual_element[:100]}",
                    )

                    try:
                        # Construct task description for image searcher
                        task_description = self._build_image_search_task_description(
                            report_title=report_title,
                            section_id=section_id,
                            section_title=section_title,
                            section_summary=section_summary,
                            visual_element=visual_element,
                        )

                        # Call image searcher agent
                        image_start_time = time.time()
                        image_result = await self.run_sub_agent(
                            "agent-image-searcher", task_description
                        )
                        image_elapsed = time.time() - image_start_time

                        if not image_result or not isinstance(image_result, str):
                            raise ValueError(
                                "Image searcher returned empty or invalid result"
                            )

                        # Parse image result
                        image_data = self._parse_json_result(image_result)

                        # Validate image result
                        if not image_data or not isinstance(image_data, dict):
                            raise ValueError("Invalid image data structure")

                        status = image_data.get("status", "error")

                        if status == "success":
                            # Success: update visual element with image information
                            image_title = image_data.get("title", visual_element)
                            image_url = image_data.get("url", "")
                            image_description = image_data.get("description", "")
                            image_source = image_data.get("source", {})

                            # Create updated visual element entry
                            updated_element = {
                                "type": "image",
                                "title": image_title,
                                "url": image_url,
                                "description": image_description,
                                "source": image_source,
                            }

                            updated_visual_elements.append(updated_element)
                            total_images_found += 1

                            self.task_log.log_step(
                                "info",
                                f"Report Workflow | Visual | Section {section_id} | Image {elem_idx + 1}",
                                f"Image found successfully in {image_elapsed:.2f}s: {image_title}",
                            )

                        elif status == "error":
                            # Error: log and skip this element
                            error_message = image_data.get(
                                "error_message", "No error message"
                            )

                            total_images_failed += 1

                            self.task_log.log_step(
                                "warning",
                                f"Report Workflow | Visual | Section {section_id} | Image {elem_idx + 1}",
                                f"Image search failed: {error_message[:200]}",
                            )

                            # Don't add failed image to updated_visual_elements (remove it)

                        else:
                            raise ValueError(f"Unknown status: {status}")

                    except Exception as e:
                        total_images_failed += 1
                        self.task_log.log_step(
                            "error",
                            f"Report Workflow | Visual | Section {section_id} | Image {elem_idx + 1}",
                            f"Image search exception: {str(e)}",
                        )
                        # Don't add failed image to updated_visual_elements

                    continue

                # Check if it's a chart generation request
                if not visual_element.startswith("tool-python:"):
                    self.task_log.log_step(
                        "warning",
                        f"Report Workflow | Visual | Section {section_id}",
                        f"Unknown visual element type: {visual_element[:100]}, skipping",
                    )
                    # updated_visual_elements.append(visual_element)
                    continue

                # Process chart generation
                total_charts_requested += 1
                visual_element = visual_element.replace("tool-python:", "").strip()

                self.task_log.log_step(
                    "info",
                    f"Report Workflow | Visual | Section {section_id} | Chart {elem_idx + 1}",
                    f"Generating chart: {visual_element[:100]}",
                )

                try:
                    # Construct task description for chart generator
                    task_description = self._build_chart_task_description(
                        report_title=report_title,
                        section_id=section_id,
                        section_title=section_title,
                        section_summary=section_summary,
                        visual_element=visual_element,
                        research_notes=research_notes,
                    )

                    # Call chart generator agent
                    chart_start_time = time.time()
                    chart_result = await self.run_sub_agent(
                        "agent-chart-generator", task_description
                    )
                    chart_elapsed = time.time() - chart_start_time

                    if not chart_result or not isinstance(chart_result, str):
                        raise ValueError(
                            "Chart generator returned empty or invalid result"
                        )

                    # Parse chart result
                    chart_data = self._parse_json_result(chart_result)

                    # Validate chart result
                    if not chart_data or not isinstance(chart_data, dict):
                        raise ValueError("Invalid chart data structure")

                    status = chart_data.get("status", "error")

                    if status == "success":
                        # Success: update visual element with chart information
                        chart_title = chart_data.get("title", visual_element)
                        chart_path = chart_data.get("path", "")
                        chart_description = chart_data.get("description", "")
                        chart_data_sources = chart_data.get("data_sources", [])
                        chart_research_notes = chart_data.get("research_notes", [])

                        # Validate chart file exists
                        full_chart_path = os.path.join(self.result_dir, chart_path)

                        if not os.path.exists(full_chart_path):
                            # Chart file doesn't exist, log error and skip
                            total_charts_failed += 1
                            self.task_log.log_step(
                                "warning",
                                f"Report Workflow | Visual | Section {section_id} | Chart {elem_idx + 1}",
                                f"Chart file not found at path: {chart_path}. Skipping this chart.",
                            )
                            continue  # Skip adding this chart to updated_visual_elements

                        # Create updated visual element entry
                        updated_element = {
                            "type": "chart",
                            "title": chart_title,
                            "path": chart_path,
                            "description": chart_description,
                            "data_sources": chart_data_sources,
                            "research_notes": chart_research_notes,
                        }

                        updated_visual_elements.append(updated_element)
                        total_charts_generated += 1

                        self.task_log.log_step(
                            "info",
                            f"Report Workflow | Visual | Section {section_id} | Chart {elem_idx + 1}",
                            f"Chart generated successfully in {chart_elapsed:.2f}s: {chart_title}",
                        )

                    elif status == "error":
                        # Error: log and skip this element
                        error_message = chart_data.get(
                            "error_message", "No error message"
                        )

                        total_charts_failed += 1

                        self.task_log.log_step(
                            "warning",
                            f"Report Workflow | Visual | Section {section_id} | Chart {elem_idx + 1}",
                            f"Chart generation failed: {error_message[:200]}",
                        )

                        # Don't add failed chart to updated_visual_elements (remove it)

                    else:
                        raise ValueError(f"Unknown status: {status}")

                except Exception as e:
                    total_charts_failed += 1
                    self.task_log.log_step(
                        "error",
                        f"Report Workflow | Visual | Section {section_id} | Chart {elem_idx + 1}",
                        f"Chart generation exception: {str(e)}",
                    )
                    # Don't add failed chart to updated_visual_elements

            # Update section with processed visual elements
            section["visual_elements"] = updated_visual_elements
            section["has_image"] = len(updated_visual_elements) > 0

            self.task_log.log_step(
                "info",
                f"Report Workflow | Visual | Section {section_id}",
                f"Section '{section_title}' completed: "
                f"{len(updated_visual_elements)} visual element(s) retained",
            )

        visual_phase_elapsed = time.time() - visual_phase_start_time

        self.task_log.log_step(
            "info",
            "Report Workflow | Visual | Summary",
            f"Visual phase completed in {visual_phase_elapsed:.2f}s\n"
            f"Charts requested: {total_charts_requested}\n"
            f"Charts generated: {total_charts_generated}\n"
            f"Charts failed: {total_charts_failed}\n"
            f"Images requested: {total_images_requested}\n"
            f"Images found: {total_images_found}\n"
            f"Images failed: {total_images_failed}",
        )

        return outline

    def _build_chart_task_description(
        self,
        report_title: str,
        section_id: str,
        section_title: str,
        section_summary: str,
        visual_element: str,
        research_notes: list,
    ) -> str:
        """Build task description for chart generator agent."""

        if self.task_lang == "zh":
            return f"""# 图表生成任务

## 报告信息
- 报告标题：{report_title}

## 章节信息
- 章节ID：{section_id}
- 章节标题：{section_title}
- 章节摘要：{section_summary}

## 图表要求
{visual_element}

## 研究笔记
{json.dumps(research_notes, ensure_ascii=False, indent=2) if research_notes else "暂无"}

## 任务说明
请基于以上信息，执行以下步骤：
1. **数据收集阶段**：
   - 仔细阅读研究笔记，判断是否包含足够的数据
   - 如果数据不足，使用 google_search 和 scrape_website 工具进行补充搜索
   - 确保收集到真实、可验证、可溯源的数据
   
2. **图表生成阶段**：
   - 使用 create_sandbox 创建Python执行环境
   - 使用 run_python_code 执行matplotlib代码生成图表
   - 使用 download_file_from_sandbox_to_local 下载图表到本地

3. **输出要求**：
   - 如果成功：返回包含 status="success"、title、path、description、data_sources、research_notes 的JSON
   - 如果失败：返回包含 status="error"、error_message 的JSON
   - 不要编造数据或虚构来源
"""
        else:
            return f"""# Chart Generation Task

## Report Information
- Report Title: {report_title}

## Section Information
- Section ID: {section_id}
- Section Title: {section_title}
- Section Summary: {section_summary}

## Chart Requirements
{visual_element}

## Research Notes
{json.dumps(research_notes, ensure_ascii=False, indent=2) if research_notes else "None"}

## Task Instructions
Based on the above information and research notes, please:

1. **Data Collection Phase**:
   - Carefully review research notes to determine if sufficient data is available
   - If data is insufficient, use google_search and scrape_website tools for additional research
   - Ensure collected data is real, verifiable, and traceable

2. **Chart Generation Phase**:
   - Use create_sandbox to create Python execution environment
   - Use run_python_code to execute matplotlib code and generate chart
   - Use download_file_from_sandbox_to_local to download chart to local storage

3. **Output Requirements**:
   - If successful: return JSON with status="success", title, path, description, data_sources, research_notes
   - If failed: return JSON with status="error", error_message
   - Do not fabricate data or invent sources
"""

    def _build_image_search_task_description(
        self,
        report_title: str,
        section_id: str,
        section_title: str,
        section_summary: str,
        visual_element: str,
    ) -> str:
        """Build task description for image searcher agent."""

        if self.task_lang == "zh":
            return f"""# 图片搜索任务

## 报告信息
- 报告标题：{report_title}

## 章节信息
- 章节ID：{section_id}
- 章节标题：{section_title}
- 章节摘要：{section_summary}

## 图片要求
{visual_element}

## 任务说明
请基于以上信息，执行以下步骤：

1. **图片搜索阶段**：
   - 理解图片需求，明确需要什么类型的图片
   - 使用 google_image_search 工具搜索相关图片
   - 从搜索结果中选择最相关的候选图片（通常3-5张）
   - 使用 visual_question_answering 工具验证候选图片的内容和质量
   - 基于验证结果，选择最符合需求的图片

2. **输出要求**：
   - 如果找到合适的图片：返回包含 status="success"、title、url、description、source 的JSON
   - 如果没有找到：返回包含 status="error"、error_message 的JSON
   - 不要编造图片URL或来源信息
"""
        else:
            return f"""# Image Search Task

## Report Information
- Report Title: {report_title}

## Section Information
- Section ID: {section_id}
- Section Title: {section_title}
- Section Summary: {section_summary}

## Image Requirements
{visual_element}

## Task Instructions
Based on the above information, please:

1. **Image Search Phase**:
   - Understand the image requirements and clarify what type of image is needed
   - Use google_image_search tool to search for relevant images
   - Select the most relevant candidate images from search results (typically 3-5 images)
   - Use visual_question_answering tool to verify the content and quality of candidate images
   - Based on verification results, select the image that best meets the requirements

2. **Output Requirements**:
   - If suitable image found: return JSON with status="success", title, url, description, source
   - If no suitable image found: return JSON with status="error", error_message
   - Do not fabricate image URLs or source information
"""

    async def run_write_phase(
        self,
        outline: dict,
    ) -> dict:
        """Write all sections based on outline with visual elements.

        Args:
            outline: Report outline with sections and visual elements

        Returns:
            Dictionary containing all section contents and references
        """
        write_phase_start_time = time.time()
        self.task_log.log_step(
            "info",
            "Report Workflow | Write",
            "Starting write phase to generate section contents",
        )

        # Validate outline structure
        if not outline or not isinstance(outline, dict):
            self.task_log.log_step(
                "error",
                "Report Workflow | Write",
                "Invalid outline structure, aborting write phase",
            )
            return {"sections": [], "references": []}

        sections = outline.get("sections", [])
        if not sections:
            self.task_log.log_step(
                "warning",
                "Report Workflow | Write",
                "No sections found in outline, aborting write phase",
            )
            return {"sections": [], "references": []}

        report_title = outline.get("title", "")

        # Check if writer agent is available
        if "agent-writer" not in self.sub_agent_tool_managers:
            self.task_log.log_step(
                "error",
                "Report Workflow | Write",
                "agent-writer not available, aborting write phase",
            )
            return {"sections": [], "references": []}

        # Initialize tracking variables
        all_sections = []
        previous_sections_summary = []

        # Process each section
        for section_idx, section in enumerate(sections):
            section_id = section.get("id", str(section_idx + 1))
            section_title = section.get("title", f"Section {section_id}")
            section_summary = section.get("summary", "")
            word_count = section.get(
                "word_count", "~500字" if self.task_lang == "zh" else "~500 words"
            )
            visual_elements = section.get("visual_elements", [])
            research_notes = section.get("research_notes", [])

            self.task_log.log_step(
                "info",
                f"Report Workflow | Write | Section {section_id}",
                f"Writing section '{section_title}'",
            )

            try:
                # Build task description for writer agent
                task_description = self._build_writer_task_description(
                    report_title=report_title,
                    section_id=section_id,
                    section_title=section_title,
                    section_summary=section_summary,
                    word_count=word_count,
                    visual_elements=visual_elements,
                    research_notes=research_notes,
                    previous_sections_summary=previous_sections_summary,
                )

                # Call writer agent
                section_start_time = time.time()
                writer_result = await self.run_sub_agent(
                    "agent-writer", task_description
                )
                section_elapsed = time.time() - section_start_time

                if not writer_result or not isinstance(writer_result, str):
                    raise ValueError("Writer agent returned empty or invalid result")

                # Clean markdown result (remove code block wrappers if present)
                section_content = self._clean_markdown_result(writer_result)

                # Extract section content and references
                content_without_refs, section_references = (
                    self._extract_references_from_section(section_content)
                )

                # Extract subsections
                subsections = self._extract_subsections(content_without_refs)

                # Add to previous sections summary
                previous_sections_summary.append(
                    {
                        "id": section_id,
                        "title": section_title,
                        "summary": section_summary,
                        "subsections": subsections,
                    }
                )

                # Store section content and references
                all_sections.append(
                    {
                        "id": section_id,
                        "title": section_title,
                        "content": content_without_refs,
                        "references": section_references,
                    }
                )

                self.task_log.log_step(
                    "info",
                    f"Report Workflow | Write | Section {section_id}",
                    f"Section '{section_title}' completed in {section_elapsed:.2f}s.",
                )

            except Exception as e:
                self.task_log.log_step(
                    "error",
                    f"Report Workflow | Write | Section {section_id}",
                    f"Failed to write section '{section_title}': {str(e)}",
                )
                # Add placeholder for failed section
                all_sections.append(
                    {
                        "id": section_id,
                        "title": section_title,
                        "content": f"## {section_title}\n\n[Section content generation failed]\n\n",
                        "references": [],
                    }
                )

        write_phase_elapsed = time.time() - write_phase_start_time

        self.task_log.log_step(
            "info",
            "Report Workflow | Write | Summary",
            f"Write phase completed in {write_phase_elapsed:.2f}s",
        )

        return {
            "title": report_title,
            "sections": all_sections,
        }

    def _build_writer_task_description(
        self,
        report_title: str,
        section_id: str,
        section_title: str,
        section_summary: str,
        word_count: str,
        visual_elements: list,
        research_notes: list,
        previous_sections_summary: list,
    ) -> str:
        """Build task description for writer agent."""

        if self.task_lang == "zh":
            return f"""# 章节撰写任务

## 报告信息
- 报告标题：{report_title}

## 前序章节概要
{json.dumps(previous_sections_summary, ensure_ascii=False, indent=2) if previous_sections_summary else '无'}

## 当前章节信息
- 章节ID：{section_id}
- 章节标题：{section_title}
- 章节摘要：{section_summary}
- 目标字数：{word_count}

## 视觉元素
{json.dumps(visual_elements, ensure_ascii=False, indent=2) if visual_elements else '无'}

## 研究笔记
{json.dumps(research_notes, ensure_ascii=False, indent=2) if research_notes else '暂无'}

## 任务说明
请基于以上信息，执行以下步骤：

1. **信息收集阶段**：
   - 仔细阅读提供的研究笔记，判断是否包含足够的信息
   - 如果信息不足，使用 google_search 和 scrape_website 工具进行补充搜索
   - 确保收集到准确、可靠、可验证的信息

2. **内容撰写阶段**：
   - 基于收集的信息撰写当前章节的完整内容
   - 必须包含章节标题（`## {section_title}`）
   - 严格遵守目标字数要求
   - 在合适位置插入所有提供的视觉元素
   - 使用 `<a href="#refN">[N]</a>` 格式引用来源
   - 确保视觉元素和引用编号连续

3. **输出要求**：
   - 使用 ```markdown 代码块包裹输出内容
   - 所有参考文献必须写在该章节的最后（`## 参考文献`）
   - 不要编造数据或虚构来源
"""
        else:
            return f"""# Section Writing Task

## Report Information
- Report Title: {report_title}

## Previous Sections Summary
{json.dumps(previous_sections_summary, ensure_ascii=False, indent=2) if previous_sections_summary else 'None'}

## Current Section Information
- Section ID: {section_id}
- Section Title: {section_title}
- Section Summary: {section_summary}
- Target Word Count: {word_count}

## Visual Elements
{json.dumps(visual_elements, ensure_ascii=False, indent=2) if visual_elements else 'None'}

## Research Notes
{json.dumps(research_notes, ensure_ascii=False, indent=2) if research_notes else 'None'}

## Task Instructions
Based on the above information, please:

1. **Information Collection Phase** (if needed):
   - Carefully review research notes to determine if sufficient information is available
   - If information is insufficient, use google_search and scrape_website tools for additional research
   - Ensure collected information is accurate, reliable, and verifiable

2. **Content Writing Phase**:
   - Write complete content for the current section based on collected information
   - Must include section title (`## {section_title}`)
   - Strictly follow the target word count
   - Insert all provided visual elements at appropriate positions
   - Use `<a href="#refN">[N]</a>` format for citations
   - Ensure visual element and reference numbers are continuous

3. **Output Requirements**:
   - Use ```markdown code block to wrap the output content
   - All references must be written at the end of this section (`## References`)
   - Do not fabricate data or invent sources
"""

    def _clean_markdown_result(self, result: str) -> str:
        """Clean markdown result by removing code block wrappers if present."""
        result = result.strip()

        # Remove markdown code block wrappers
        if result.startswith("```markdown") or result.startswith("```md"):
            # Find the first newline after the opening ```
            first_newline = result.find("\n")
            if first_newline != -1:
                result = result[first_newline + 1 :]

        if result.startswith("```"):
            # Find the first newline after the opening ```
            first_newline = result.find("\n")
            if first_newline != -1:
                result = result[first_newline + 1 :]

        # Remove closing ```
        if result.endswith("```"):
            result = result[:-3].rstrip()

        return result.strip()

    def _extract_references_from_section(
        self, section_content: str
    ) -> Tuple[str, List[Dict[str, str]]]:
        """Extract references from section content and return content without references.

        Returns:
            Tuple of (content_without_references, list_of_references)
        """
        references = []

        # Find the references section
        if self.task_lang == "zh":
            ref_pattern = r"\n## 参考文献\s*\n(.*?)(?=\n##|\Z)"
        else:
            ref_pattern = r"\n## References\s*\n(.*?)(?=\n##|\Z)"

        match = re.search(ref_pattern, section_content, re.DOTALL | re.IGNORECASE)

        if match:
            ref_section = match.group(1).strip()
            # Remove references section from content
            content_without_refs = section_content[: match.start()].rstrip()

            # Parse individual references
            # Format: <a id="refN"></a> [N] Description URL
            ref_lines = ref_section.split("\n")
            for line in ref_lines:
                line = line.strip()
                if not line:
                    continue

                # Try to extract reference number、content and URL
                ref_match = re.match(
                    r'<a\s+id="ref(\d+)"></a>\s*\[(\d+)\]\s*(.*?)\s*(http\S+)?$', line
                )

                if ref_match:
                    ref_id = ref_match.group(1)
                    ref_number = ref_match.group(2)
                    ref_content = ref_match.group(3).strip()
                    ref_url = ref_match.group(4).strip() if ref_match.group(4) else ""

                    references.append(
                        {
                            "id": ref_id,
                            "number": ref_number,
                            "content": ref_content,
                            "url": ref_url,
                        }
                    )

            return content_without_refs, references
        else:
            # No references section found
            return section_content, []

    def _extract_subsections(self, section_content: str) -> Dict[str, Any]:
        """Extract subsections from section content."""

        # Extract subsections (### and #### headings with hierarchy)
        subsection_pattern = r"^(#{3,4})\s+(.+)$"
        matches = re.findall(subsection_pattern, section_content, re.MULTILINE)

        # Format subsections with hierarchy indication
        formatted_subsections = []
        for level_markers, text in matches:
            level = len(level_markers)  # 3 for ###, 4 for ####
            if level == 3:
                # Level 1 subsection (###)
                formatted_subsections.append(text.strip())
            elif level == 4:
                # Level 2 subsection (####) - indent with "  - "
                formatted_subsections.append(f"  - {text.strip()}")

        return formatted_subsections

    async def run_polish_phase(
        self,
        report: dict,
    ) -> str:
        """Polish phase: generate final markdown report with deduplicated references and validated figures.

        Args:
            report: Write phase output with sections and references

        Returns:
            Final markdown report as string
        """
        polish_phase_start_time = time.time()
        self.task_log.log_step(
            "info",
            "Report Workflow | Polish",
            "Starting polish phase to generate final markdown report",
        )

        # Extract report title
        report_title = report.get("title", "Research Report")

        # Extract sections
        sections = report.get("sections", [])

        # Remove uncited references from each section
        sections = self._remove_uncited_references(sections)

        # Deduplicate and renumber references
        deduplicated_refs, sections = self._deduplicate_and_renumber_references(
            sections
        )

        # Fix figure numbers
        sections = self._fix_section_figures(sections)

        # Generate final markdown report
        final_markdown = self._generate_final_markdown(
            report_title, sections, deduplicated_refs
        )

        polish_phase_elapsed = time.time() - polish_phase_start_time

        self.task_log.log_step(
            "info",
            "Report Workflow | Polish | Summary",
            f"Polish phase completed in {polish_phase_elapsed:.2f}s",
        )

        return final_markdown

    def _normalize_ref_content(self, content: str) -> str:
        """Normalize reference content for deduplication comparison.

        Examples:
            "36氪 (2026). AI存储量价齐飞，国产芯片迎来历史性机遇."
                -> "ai存储量价齐飞，国产芯片迎来历史性机遇."
        """

        # Step 1: Remove author and date prefix
        prefix_pattern = r"^[^(]+\s*\([^)]+\)\.?\s*"
        normalized = re.sub(prefix_pattern, "", content)

        # Step 2: Convert to lowercase
        normalized = normalized.lower()

        # Step 3: Remove all whitespace
        normalized = re.sub(r"[\s\u3000]+", "", normalized)

        return normalized

    def _deduplicate_and_renumber_references(
        self, sections: List[Dict]
    ) -> Tuple[List[Dict[str, str]], List[Dict]]:
        """Deduplicate references and renumber globally.

        Args:
            sections: List of sections, each containing content and references

        Returns:
            Tuple of (deduplicated_references, updated_sections)
        """
        seen_urls = {}
        seen_contents = {}  # Track normalized content for deduplication
        deduplicated = []
        updated_sections = []

        for section in sections:
            content = section.get("content", "")
            references = section.get("references", [])

            # Build ref mapping (old id -> new id)
            ref_mapping = {}

            for ref in references:
                url = ref.get("url", "")
                old_id = ref.get("id", "")
                ref_content = ref.get("content", "")

                # Normalize content for comparison
                normalized_content = self._normalize_ref_content(ref_content)

                # Check for duplicates by URL or normalized content
                if url and url in seen_urls:
                    # Duplicate by URL
                    ref_mapping[old_id] = seen_urls[url]
                elif normalized_content and normalized_content in seen_contents:
                    # Duplicate by content
                    ref_mapping[old_id] = seen_contents[normalized_content]
                else:
                    # New unique reference
                    new_id = str(len(deduplicated) + 1)
                    deduplicated.append(
                        {
                            "id": new_id,
                            "number": new_id,
                            "content": ref_content,
                            "url": url,
                        }
                    )
                    if url:
                        seen_urls[url] = new_id
                    if normalized_content:
                        seen_contents[normalized_content] = new_id
                    ref_mapping[old_id] = new_id

            # Update citations in this section's content
            def replace_citation(match):
                old_id = match.group(1)
                new_id = ref_mapping.get(old_id, old_id)
                return f'<a href="#ref{new_id}">[{new_id}]</a>'

            content = re.sub(
                r'<a\s+href="#ref(\d+)">\[(\d+)\]</a>',
                replace_citation,
                content,
            )

            # Remove consecutive duplicate citations
            content = self._remove_consecutive_duplicate_citations(content)

            # Save updated section
            updated_section = section.copy()
            updated_section["content"] = content
            updated_sections.append(updated_section)

        self.task_log.log_step(
            "info",
            "Report Workflow | Polish | References",
            f"Deduplicated to {len(deduplicated)} unique references",
        )

        return deduplicated, updated_sections

    def _remove_consecutive_duplicate_citations(self, content: str) -> str:
        """Remove duplicate citations within consecutive citation groups.

        Identifies all consecutive citation groups (citations separated only by spaces or adjacent),
        removes duplicate citation ids within each group, keeps only the first occurrence of
        each id, and normalizes spacing to a single space between all retained citations.

        Examples:
            Input: [1][2][1] -> Output: [1] [2]
            Input: [1] [2] [1] -> Output: [1] [2]
            Input: [1] text [2][2] -> Output: [1] text [2]
            Input: [1][2][1][3][2] -> Output: [1] [2] [3]

        Args:
            content: Text content containing citation markers

        Returns:
            Deduplicated content with duplicate citations removed within consecutive groups
            and citations separated by single spaces
        """
        # Regex pattern: matches consecutive citation groups
        # Requires at least 2 citations, which can be adjacent or space-separated
        # Example matches: [1][2], [1] [2], [1][2] [3]
        group_pattern = (
            r'<a\s+href="#ref\d+">\[\d+\]</a>(?:\s*<a\s+href="#ref\d+">\[\d+\]</a>)+'
        )

        def process_group(match):
            """Process a single citation group: deduplicate and join with spaces."""
            group_text = match.group(0)

            # Regex pattern to extract citation ids
            citation_pattern = r'<a\s+href="#ref(\d+)">\[(\d+)\]</a>'

            seen_refs = set()  # Track citation ids already seen
            result = []  # List of citations to keep

            # Iterate through all citations in the group
            for cite_match in re.finditer(citation_pattern, group_text):
                ref_id = cite_match.group(1)  # Citation id
                full_citation = cite_match.group(0)  # Full citation HTML

                if ref_id not in seen_refs:
                    # First occurrence, keep it
                    seen_refs.add(ref_id)
                    result.append(full_citation)
                # Duplicate occurrence, skip it

            # Join all retained citations with single spaces
            return " ".join(result)

        # Process all citation groups in the content
        return re.sub(group_pattern, process_group, content)

    def _remove_uncited_references(self, sections: List[Dict]) -> List[Dict]:
        """Remove uncited references from each section.

        Args:
            sections: List of sections, each containing content and references

        Returns:
            Updated sections list with uncited references removed
        """
        updated_sections = []
        total_removed = 0  # Track total removed references

        for section in sections:
            content = section.get("content", "")
            references = section.get("references", [])

            if not references:
                updated_sections.append(section)
                continue

            # Find ref ids actually cited in content
            cited_ids = set()
            citation_pattern = r'<a\s+href="#ref(\d+)">\[(\d+)\]</a>'
            for match in re.finditer(citation_pattern, content):
                cited_ids.add(match.group(1))

            # Filter out unreferenced refs
            filtered_refs = []
            for ref in references:
                ref_id = ref.get("id", "")

                if ref_id in cited_ids:
                    filtered_refs.append(ref)
                else:
                    total_removed += 1

            # Save processed section with filtered references
            updated_section = section.copy()
            updated_section["references"] = filtered_refs
            updated_sections.append(updated_section)

        self.task_log.log_step(
            "info",
            "Report Workflow | Polish | References",
            f"Removed {total_removed} uncited references",
        )

        return updated_sections

    def _fix_section_figures(self, sections: List[Dict]) -> List[Dict]:
        """Validate and fix figure numbers across all sections to ensure they are sequential and unique.

        Args:
            sections: List of section dictionaries

        Returns:
            Updated sections with corrected figure numbers
        """
        updated_sections = []
        expected_figure_num = 1

        for section in sections:
            content = section.get("content", "")
            section_id = section.get("id", "")

            # Fix figure numbers in this section
            fixed_content, actual_figures = self._fix_single_section_figures(
                content, expected_figure_num
            )

            # Update section with fixed content
            updated_section = section.copy()
            updated_section["content"] = fixed_content
            updated_sections.append(updated_section)

            # Update expected figure number for next section
            expected_figure_num += actual_figures

        # Log results
        total_figures = expected_figure_num - 1

        self.task_log.log_step(
            "info",
            "Report Workflow | Polish | Figures",
            f"Validated and fixed {total_figures} figures",
        )

        return updated_sections

    def _fix_single_section_figures(
        self, content: str, start_figure_num: int
    ) -> tuple[str, int]:
        """Fix figure numbers in a single section.

        Args:
            content: Section content
            start_figure_num: Starting figure number for this section

        Returns:
            Tuple of (updated_content, actual_figure_count)
        """
        # Find all figure components in this section
        if self.task_lang == "zh":
            figure_pattern = r'<figure>.*?<figcaption\s+id="(fig\d+)">(图\s*\d+)(.*?)</figcaption>.*?</figure>'
        else:
            figure_pattern = r'<figure>.*?<figcaption\s+id="(fig\d+)">(Figure\s*\d+)(.*?)</figcaption>.*?</figure>'

        figures = []
        for match in re.finditer(figure_pattern, content, re.DOTALL):
            figures.append(
                {
                    "start": match.start(),
                    "end": match.end(),
                    "old_id": match.group(1),
                    "old_label": match.group(2),
                    "caption": match.group(3),
                    "full_match": match.group(0),
                }
            )

        if not figures:
            return content, 0

        # Sort by position and assign new numbers
        figures.sort(key=lambda x: x["start"])
        id_mapping = {}

        for idx, fig in enumerate(figures):
            new_num = start_figure_num + idx
            new_id = f"fig{new_num}"
            if self.task_lang == "zh":
                new_label = f"图 {new_num}"
            else:
                new_label = f"Figure {new_num}"

            id_mapping[fig["old_id"]] = new_id
            fig["new_id"] = new_id
            fig["new_label"] = new_label

        # Replace figure components (from back to front)
        result = content
        for fig in reversed(figures):
            new_figure = (
                fig["full_match"]
                .replace(f'id="{fig["old_id"]}"', f'id="{fig["new_id"]}"')
                .replace(fig["old_label"], fig["new_label"])
            )
            result = result[: fig["start"]] + new_figure + result[fig["end"] :]

        # Replace figure references in text
        if self.task_lang == "zh":
            ref_pattern = r'<a\s+href="#(fig\d+)">图\s*\d+</a>'
        else:
            ref_pattern = r'<a\s+href="#(fig\d+)">Figure\s*\d+</a>'

        def replace_ref(match):
            old_id = match.group(1)
            new_id = id_mapping.get(old_id, old_id)
            fig_num = new_id.replace("fig", "")
            if self.task_lang == "zh":
                return f'<a href="#{new_id}">图 {fig_num}</a>'
            else:
                return f'<a href="#{new_id}">Figure {fig_num}</a>'

        result = re.sub(ref_pattern, replace_ref, result)

        return result, len(figures)

    def _generate_final_markdown(
        self,
        title: str,
        sections: List[Dict],
        references: List[Dict[str, str]],
    ) -> str:
        """Generate final markdown report with title, sections, and references.

        Args:
            title: Report title
            sections: List of section dictionaries with content
            references: List of deduplicated reference dictionaries

        Returns:
            Complete markdown report as string
        """
        markdown_parts = []

        # Add title
        markdown_parts.append(f"# {title}\n\n")

        # Add sections
        for section in sections:
            content = section.get("content", "")
            markdown_parts.append(content)
            markdown_parts.append("\n\n")

        # Add references section
        if references:
            if self.task_lang == "zh":
                markdown_parts.append("## 参考文献\n\n")
            else:
                markdown_parts.append("## References\n\n")

            for ref in references:
                ref_id = ref.get("id", "")
                ref_number = ref.get("number", "")
                ref_content = ref.get("content", "")
                ref_url = ref.get("url", "")
                markdown_parts.append(
                    f'<a id="ref{ref_id}"></a> [{ref_number}] {ref_content} {ref_url}\n\n'
                )

        return "".join(markdown_parts)
