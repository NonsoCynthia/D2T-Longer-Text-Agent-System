# agents/agents_modules/task.py

__author__ = "chinonsocynthiaosuji"

"""
Author: Chinonso Cynthia Osuji
Date: 10/07/2025
Description:
    Unified worker that can perform content ordering, text structuring,
    and surface realization using a single prompt.
"""

from typing import Any, Dict, List, Text, Union
import json

from langchain_classic.agents import AgentExecutor, create_json_chat_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from agents.utilities.utils import ExecutionState, AgentStepOutput
from agents.agent_prompts import WORKER_SYSTEM_PROMPT, WORKER_HUMAN_PROMPT
from agents.llm_model import UnifiedModel, model_name
from agents.agent_prompts import (
    UNIFIED_WORKER_PROMPT_EN,
    UNIFIED_WORKER_PROMPT_GA,
)
from agents.utilities.agent_utils import apply_variable_substitution, _handle_parsing_errors


class UnifiedTaskWorker:
    @classmethod
    def init(
        cls,
        provider: str = "ollama",
        language: str = "en",
    ) -> AgentExecutor:
        """
        Initialise a unified worker as a JSON agent:
        - system message = unified worker spec + generic worker instructions
        - human message = WORKER_HUMAN_PROMPT (which will receive {input})
        """
        params = model_name.get(provider.lower(), {}).copy()
        params["temperature"] = 0.0
        model = UnifiedModel(provider=provider, **params).raw_model()

        # Choose unified prompt for language
        base_prompt = (
            UNIFIED_WORKER_PROMPT_GA
            if language.lower() == "ga"
            else UNIFIED_WORKER_PROMPT_EN
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "AGENT DESCRIPTION:\n"
                        f"{base_prompt}\n\n"
                        "EXECUTION INSTRUCTION:\n"
                        f"{WORKER_SYSTEM_PROMPT}"
                    ),
                ),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                ("human", WORKER_HUMAN_PROMPT),
            ]
        ).partial(output_format="text")

        tools: List[Any] = []

        return AgentExecutor(
            agent=create_json_chat_agent(model, tools, prompt),
            tools=tools,
            verbose=True,
            max_iterations=max(4, 4 * len(tools)),
            handle_parsing_errors=_handle_parsing_errors,
            return_result_steps=True,
        )

    @classmethod
    def execute(cls, agent: AgentExecutor, language: str = "en"):
        """
        Return a LangGraph node function that:
        - Builds a stage specific input for the unified worker
        - Calls the JSON agent
        - Logs the step into history_of_steps
        - Hands control back to the guardrail
        """

        def latest_output_for(
            history: List[AgentStepOutput],
            agent_name: str,
        ) -> Text:
            target = agent_name.strip().lower()
            for step in reversed(history):
                if getattr(step, "agent_name", "").strip().lower() == target:
                    return str(step.agent_output)
            return ""

        def normalise_feedback(raw: Any) -> Text:
            if not raw:
                return ""
            if isinstance(raw, (dict, list)):
                try:
                    return json.dumps(raw, ensure_ascii=False, indent=2)
                except Exception:
                    return str(raw)
            return str(raw)

        def build_task_input(state: ExecutionState):
            """
            Build the text that will be given to the unified worker.

            Returns (task_name, payload_for_model).
            task_name is one of 'content ordering', 'text structuring', 'surface realization'.
            """
            task = state.get("next_agent", "").strip().lower()
            orch_instruction = state.get("next_agent_payload", "").strip()
            history = state.get("history_of_steps", []) or []

            if task == "content ordering":
                data_input = state.get("data_input", "")
                payload = (
                    "Task: content ordering\n"
                    f"INSTRUCTION: {orch_instruction}\n"
                    f"DATA: {data_input}"
                ).strip()
                return task, payload

            if task == "text structuring":
                ordering_output = latest_output_for(history, "content ordering")
                payload = (
                    "Task: text structuring\n"
                    f"INSTRUCTION: {orch_instruction}\n"
                    f"ORDERING OUTPUT: {ordering_output}"
                ).strip()
                return task, payload

            if task == "surface realization":
                structuring_output = latest_output_for(history, "text structuring")
                prev_sr_output = latest_output_for(history, "surface realization")

                raw_feedback = (
                    state.get("review")
                    or state.get("guardrail_feedback")
                    or state.get("guardrail_review")
                    or ""
                )
                feedback_str = normalise_feedback(raw_feedback)

                prev_block = f"\nPREV OUTPUT:\n{prev_sr_output}" if prev_sr_output else ""
                feedback_block = (
                    f"\nGUARDRAIL FEEDBACK:\n{feedback_str}" if feedback_str else ""
                )

                payload = (
                    "Task: surface realization\n"
                    f"INSTRUCTION: {orch_instruction}\n"
                    f"TEXT INPUT:\n{structuring_output}"
                    f"{prev_block}"
                    f"{feedback_block}"
                ).strip()
                return task, payload

            # Fallback
            payload = f"Task: unknown\nINSTRUCTION: {orch_instruction}"
            return task or "unknown", payload

        def run(state: ExecutionState):
            idx = state.get("iteration_count", 0)
            history = state.get("history_of_steps", []) or []

            task_name, worker_input = build_task_input(state)

            out = agent.invoke({"input": worker_input})

            # Extract the textual answer from JSON agent
            raw_output = out.get("output", out)
            if isinstance(raw_output, dict) and "action_input" in raw_output:
                text = raw_output["action_input"]
            else:
                if isinstance(raw_output, str):
                    text = raw_output
                elif isinstance(out, dict) and "action_input" in out:
                    text = out["action_input"]
                else:
                    text = str(raw_output)

            history.append(
                AgentStepOutput(
                    agent_name=task_name,
                    agent_input=worker_input,
                    agent_output=text,
                    rationale=text,
                    tool_steps=out.get("result_steps", []) if isinstance(out, dict) else [],
                )
            )

            worker_attempts: Dict[str, int] = state.get("worker_attempts", {}) or {}
            if task_name:
                worker_attempts[task_name] = worker_attempts.get(task_name, 0) + 1

            return {
                "next_agent": "guardrail",
                "history_of_steps": history,
                "iteration_count": idx + 1,
                "worker_attempts": worker_attempts,
                "last_worker": task_name,
            }

        return run
