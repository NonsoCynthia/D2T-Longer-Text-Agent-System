__author__='chinonsocynthiaosuji'

"""
Author: Chinonso Cynthia Osuji
Date: 10/07/2025
Description:
    Orchestrate the task execution by directing agents to perform specific tasks
"""

import re
from typing import Dict, List, Text, Any, Union, Optional, Set
from langchain_classic.agents import AgentExecutor
from agents.utilities.utils import ExecutionState, AgentStepOutput
from agents.llm_model import UnifiedModel, model_name
from agents.utilities.agent_utils import summarize_agent_steps
from agents.agent_prompts import (
    ORCHESTRATOR_PROMPT,
    ORCHESTRATOR_INPUT,
)

class TaskOrchestrator:
    @classmethod
    def init(cls, provider: str = "ollama") -> AgentExecutor:
        conf = model_name.get(provider.lower(), {}).copy()
        conf["temperature"] = 0.0
        return UnifiedModel(provider=provider, **conf).model_(ORCHESTRATOR_PROMPT)

    @classmethod
    def execute(cls, executor: AgentExecutor):
        def run(state: ExecutionState):
            idx = state.get("iteration_count", 0)
            limit = state.get("max_iteration", 50)
            history = state.get("history_of_steps", []) 

            # New: worker attempts info
            worker_attempts: Dict[str, int] = state.get("worker_attempts", {}) or {}
            raw_limit = state.get("max_worker_attempts", 3)

            # Allow max_worker_attempts to be either an int or a dict of per worker limits
            if isinstance(raw_limit, dict):
                per_worker_limits: Dict[str, int] = raw_limit
                try:
                    default_limit = max(per_worker_limits.values())  # just for display
                except ValueError:
                    default_limit = 3
            else:
                per_worker_limits = {}
                try:
                    default_limit = int(raw_limit)
                except Exception:
                    default_limit = 3

            if worker_attempts:
                attempts_lines = []
                for name, count in worker_attempts.items():
                    limit_for_this = per_worker_limits.get(name, default_limit)
                    attempts_lines.append(
                        f"- {name}: {count} attempt(s) out of {limit_for_this} allowed"
                    )
                attempts_block = "WORKER ATTEMPTS:\n" + "\n".join(attempts_lines)
            else:
                attempts_block = "WORKER ATTEMPTS:\n- no worker has run yet"


            prompt = state.get("user_prompt", "")
            summary = "\n\n".join(summarize_agent_steps(history)[-2:])  # last 2 steps
            feedback = state.get("review", "")

            payload = ORCHESTRATOR_INPUT.format(
                input=prompt,
                result_steps=f"\nRESULT STEPS:\n{summary}" if summary else "",
                feedback=f"\nFEEDBACK:\n{feedback}" if feedback else "",
                attempts=f"\n{attempts_block}",
            ).replace("\n\n\n", "\n")

            output = executor.invoke({"input": payload}).content.strip()
            
            try:
                output_lower = output.lower()
                if any(keyword in output_lower for keyword in ["instructions:", "instruction:"]):
                    rationale, role, role_input, instruction = re.findall(
                        r"Thought:\s*(.*?)\s*Worker:\s*(.*?)\s*Worker Input:\s*(.*?)\s*Instructions?:\s*(.*)",
                        output,
                        re.DOTALL,
                    )[0]
                else:
                    rationale, role, role_input = re.findall(
                        r"Thought:\s*(.*?)\s*Worker:\s*(.*?)\s*Worker Input:\s*(.*)",
                        output,
                        re.DOTALL,
                    )[0]
                    instruction = None
            except Exception:
                rationale, role, role_input, instruction = "parse error", "finish", output, None

            role = role.lower().strip("'\"").replace("_", " ")

            # Enforce minimum pipeline progression
            expected_order = ["content ordering", "text structuring", "surface realization"]
            expected_set: Set[str] = set(expected_order)

            done: Set[str] = {
                step.agent_name.strip().lower()
                for step in history or []
                if getattr(step, "agent_name", None)
                and step.agent_name.strip().lower() in expected_set
            }

            # If the orchestrator tries to FINISH early, force the next missing stage
            if role in ("finish", "finalizer"):
                if not expected_set.issubset(done):
                    # choose the next missing stage in the canonical order
                    if "content ordering" not in done:
                        role = "content ordering"
                    elif "text structuring" not in done:
                        role = "text structuring"
                    elif "surface realization" not in done:
                        role = "surface realization"

            history.append(
                AgentStepOutput(
                    agent_name="orchestrator",
                    agent_input=payload,
                    agent_output=f"{role}(input='{role_input}', instruction='{instruction}')",
                    rationale=f"{rationale}\nInstruction:\n{instruction}",
                )
            )

            if idx >= limit:
                return {
                    "next_agent": "finish",
                    "final_response": "Stopped due to limit reached.",
                    "next_agent_payload": "Limit reached.",
                    "history_of_steps": history,
                    "iteration_count": idx + 1,
                    "max_iteration": limit,
                }

            return {
                "next_agent": role,
                "final_response": role_input,
                "next_agent_payload": (
                    f"{role_input}\nAdditional Instruction: {instruction}"
                    if instruction
                    else role_input
                ),
                "history_of_steps": history,
                "iteration_count": idx + 1,
                "max_iteration": limit,
            }

        return run