__author__ = "chinonsocynthiaosuji"

"""
Author: Chinonso Cynthia Osuji
Date: 10/07/2025
Description:
    Worker agent that executes tasks based on the orchestrator's instructions.
"""

from typing import Dict, List, Text, Any, Union
import json
import re

from langchain_classic.agents import AgentExecutor, create_json_chat_agent
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_core.exceptions import OutputParserException

from langgraph.errors import GraphRecursionError

from agents.utilities.utils import ExecutionState, AgentStepOutput
from agents.llm_model import UnifiedModel, model_name
from agents.agent_prompts import WORKER_SYSTEM_PROMPT, WORKER_HUMAN_PROMPT
from agents.utilities.agent_utils import apply_variable_substitution


class TaskWorker:
    @classmethod
    def init(
        cls,
        description: Text,
        tools: List[Any],
        context: Union[Text, Dict[str, Any]],
        provider: str = "ollama",
    ) -> AgentExecutor:
        params = model_name.get(provider.lower())
        model = UnifiedModel(provider=provider, **params).raw_model()

        agent_description = (
            apply_variable_substitution(description, context) if description else ""
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "AGENT DESCRIPTION:\n"
                        f"{agent_description}\n\n"
                        "EXECUTION INSTRUCTION:\n"
                        f"{WORKER_SYSTEM_PROMPT}"
                    ),
                ),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                ("human", WORKER_HUMAN_PROMPT),
            ]
        ).partial(output_format="text")

        # custom parsing error handler
        def _handle_parsing_errors(e: OutputParserException) -> str:
            """
            Make the JSON agent tolerant of extra text around the JSON.
            """
            raw = getattr(e, "llm_output", "") or ""
            if not raw.strip():
                return json.dumps(
                    {"action": "Final Answer", "action_input": "PARSING_ERROR"}
                )

            # Cleanup raw output
            cleaned = raw.replace("```json", "```").replace("```", "")
            cleaned = re.sub(r"^\s*Action\s*:\s*", "", cleaned, flags=re.IGNORECASE)

            # Extract content starting from the first JSON bracket
            idxs = [i for i in [cleaned.find("{"), cleaned.find("[")] if i != -1]
            if idxs:
                cleaned = cleaned[min(idxs) :]

            # Drop comment lines to prevent parse errors
            cleaned_no_comments = "\n".join(
                line for line in cleaned.splitlines()
                if not line.lstrip().startswith("//")
            ).strip()

            try:
                parsed_inner = json.loads(cleaned_no_comments)
            except Exception:
                # Fallback: treat whole output as the final answer string
                return json.dumps(
                    {
                        "action": "Final Answer",
                        "action_input": raw.strip(),
                    }
                )

            # If inner JSON is already an action object, use it
            if isinstance(parsed_inner, dict) and "action" in parsed_inner and "action_input" in parsed_inner:
                return json.dumps(parsed_inner)

            # Otherwise wrap the content
            return json.dumps(
                {
                    "action": "Final Answer",
                    "action_input": parsed_inner,
                }
            )

        return AgentExecutor(
            agent=create_json_chat_agent(model, tools, prompt),
            tools=tools,
            verbose=True,
            max_iterations=max(4, 4 * len(tools)),
            handle_parsing_errors=_handle_parsing_errors,
            return_result_steps=True,
        )

    @classmethod
    def execute(cls, agent: AgentExecutor, role: str):
        role = role.strip().lower()

        def latest_output_for(history: List[AgentStepOutput], agent_name: str) -> Text:
            target = agent_name.strip().lower()
            for step in reversed(history):
                if step.agent_name.strip().lower() == target:
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

        def build_worker_input(state: ExecutionState) -> Text:
            orch_instruction = state.get("next_agent_payload", "").strip()
            history = state.get("history_of_steps", []) or []

            if role == "content ordering":
                data_input = state.get("data_input", "")
                return (
                    f"Worker: content ordering\n"
                    f"INSTRUCTION: {orch_instruction}\n"
                    f"DATA: {data_input}"
                ).strip()

            if role == "text structuring":
                ordering_output = latest_output_for(history, "content ordering")
                return (
                    f"Worker: text structuring\n"
                    f"INSTRUCTION: {orch_instruction}\n"
                    f"ORDERING OUTPUT: {ordering_output}"
                ).strip()

            if role == "surface realization":
                structuring_output = latest_output_for(history, "text structuring")
                previous_sr_output = latest_output_for(history, "surface realization")

                raw_feedback = (
                    state.get("review")
                    or state.get("guardrail_feedback")
                    or state.get("guardrail_review")
                    or ""
                )
                guardrail_feedback_str = normalise_feedback(raw_feedback)

                previous_block = (
                    f"\nPREV OUTPUT: {previous_sr_output}" if previous_sr_output else ""
                )
                
                feedback_block = (
                    f"\nGUARDRAIL FEEDBACK: {guardrail_feedback_str}"
                    if guardrail_feedback_str else ""
                )

                # OPTIMIZED REVISION INSTRUCTIONS (Cost-effective & fixes underscores)
                revision_instructions = """
*** REVISION RULES ***
1. REMOVE UNDERSCORES: Convert "Berlin_Germany" -> "Berlin German". Only keep underscores in strict technical codes (e.g., ISO_3166).
2. ACCURACY: Express every input triple. Do not omit facts.
3. NO HALLUCINATION: Do not add promotional words ("major", "famous") or infer roles not in the data.
4. CORRECTION: If guardrail reports "additions", remove them. If "omissions", add them using exact values (minus underscores).
5. FORMAT: Return the full revised text only.
"""

                return (
                    f"Worker: surface realization\n"
                    f"INSTRUCTION: {orch_instruction}\n"
                    f"TEXT INPUT: {structuring_output}"
                    f"{previous_block}"
                    f"{feedback_block}"
                    # f"\n{revision_instructions}"
                ).strip()

            return orch_instruction

        def run(state: ExecutionState):
            idx = state.get("iteration_count", 0)
            inputs = build_worker_input(state)
            history = state.get("history_of_steps", []) or []

            try:
                out = agent.invoke({"input": inputs})
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

                tools = out.get("result_steps", []) if isinstance(out, dict) else []
            except GraphRecursionError:
                text, tools = "Too many iterations. Try splitting task.", []

            history.append(
                AgentStepOutput(
                    agent_name=role,
                    agent_input=inputs,
                    agent_output=text,
                    rationale=text,
                    tool_steps=tools,
                )
            )

            return {
                "next_agent": "guardrail",
                "history_of_steps": history,
                "iteration_count": idx + 1,
            }

        return run