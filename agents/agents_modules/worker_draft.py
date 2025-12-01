__author__ = "chinonsocynthiaosuji"

"""
Author: Chinonso Cynthia Osuji
Date: 10/07/2025
Description:
    Worker agent that executes tasks based on the orchestrator's instructions.
"""

from typing import Dict, List, Text, Any, Union, Optional
import json

from langchain_classic.agents import AgentExecutor, create_json_chat_agent
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
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

        return AgentExecutor(
            agent=create_json_chat_agent(model, tools, prompt),
            tools=tools,
            verbose=True,
            max_iterations=max(4, 4 * len(tools)),
            handle_parsing_errors=True,
            return_result_steps=True,
        )

    @classmethod
    def execute(cls, agent: AgentExecutor, role: str):
        role = role.strip().lower()

        def build_worker_input(state: ExecutionState) -> Text:
            """
            Compose the worker input based on the role and the current state.

            - content ordering: orchestrator instruction + raw data_input
            - text structuring: orchestrator instruction + latest content ordering output
            - surface realization: orchestrator instruction
                                   + latest text structuring output
                                   + previous surface realization output (if any)
                                   + guardrail feedback (if any)
            """

            # Instruction from the orchestrator for this step
            orch_instruction = state.get("next_agent_payload", "").strip()

            # All previous steps
            history = state.get("history_of_steps", []) or []

            # Helper to fetch the latest output for a given agent name
            def latest_output_for(agent_name: str) -> Text:
                target = agent_name.strip().lower()
                for step in reversed(history):
                    if step.agent_name.strip().lower() == target:
                        return str(step.agent_output)
                return ""

            if role == "content ordering":
                data_input = state.get("data_input", "")
                return (
                    "Worker: content ordering\n\n"
                    "Worker Input:\n"
                    "ORCHESTRATOR INSTRUCTION:\n"
                    f"{orch_instruction}\n\n"
                    "DATA INPUT:\n"
                    f"{data_input}"
                ).strip()

            if role == "text structuring":
                ordering_output = latest_output_for("content ordering")
                return (
                    "Worker: text structuring\n\n"
                    "Worker Input:\n"
                    "ORCHESTRATOR INSTRUCTION:\n"
                    f"{orch_instruction}\n\n"
                    "ORDERING OUTPUT:\n"
                    f"{ordering_output}"
                ).strip()

            if role == "surface realization":
                structuring_output = latest_output_for("text structuring")
                previous_sr_output = latest_output_for("surface realization")

                # Guardrail feedback can be stored under "review" (or similar) in the state.
                # We accept dict, list, or string and normalize to a JSON string for clarity.
                raw_feedback = (
                    state.get("review")
                    or state.get("guardrail_feedback")
                    or state.get("guardrail_review")
                    or ""
                )

                if isinstance(raw_feedback, (dict, list)):
                    guardrail_feedback_str = json.dumps(
                        raw_feedback, ensure_ascii=False, indent=2
                    )
                else:
                    guardrail_feedback_str = str(raw_feedback).strip()

                previous_block = (
                    "\n\nPREVIOUS SURFACE REALIZATION OUTPUT:\n"
                    f"{previous_sr_output}"
                    if previous_sr_output
                    else ""
                )

                feedback_block = (
                    "\n\nGUARDRAIL FEEDBACK (JSON):\n"
                    f"{guardrail_feedback_str}"
                    if guardrail_feedback_str
                    else ""
                )

                return (
                    "Worker: surface realization\n\n"
                    "Worker Input:\n"
                    "ORCHESTRATOR INSTRUCTION:\n"
                    f"{orch_instruction}\n\n"
                    "TEXT STRUCTURING OUTPUT:\n"
                    f"{structuring_output}"
                    f"{previous_block}"
                    f"{feedback_block}"
                ).strip()

            # Fallback for any other roles if you add more in future
            return orch_instruction

        def run(state: ExecutionState):
            idx = state.get("iteration_count", 0)
            inputs = build_worker_input(state)
            history = state.get("history_of_steps", []) or []

            try:
                out = agent.invoke({"input": inputs})
                text = (
                    out.get("output")
                    or out.get("action_input")
                    or getattr(out, "content", str(out))
                )
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
    

# #####################################################################

__author__ = "chinonsocynthiaosuji"

"""
Author: Chinonso Cynthia Osuji
Date: 10/07/2025
Description:
    Worker agent that executes tasks based on the orchestrator's instructions.
"""

from typing import Dict, List, Text, Any, Union
import json

from langchain_classic.agents import AgentExecutor, create_json_chat_agent
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
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

        return AgentExecutor(
            agent=create_json_chat_agent(model, tools, prompt),
            tools=tools,
            verbose=True,
            max_iterations=max(4, 4 * len(tools)),
            handle_parsing_errors=True,
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
            """Turn review/guardrail feedback into a clean JSON string if possible."""
            if not raw:
                return ""
            if isinstance(raw, (dict, list)):
                try:
                    return json.dumps(raw, ensure_ascii=False, indent=2)
                except Exception:
                    return str(raw)
            return str(raw)

        def build_worker_input(state: ExecutionState) -> Text:
            """
            Compose the worker input based on the role and the current state.

            - content ordering: orchestrator instruction + raw data_input
            - text structuring: orchestrator instruction + latest content ordering output
            - surface realization:
                - orchestrator instruction
                - latest text structuring output
                - previous surface realization output (if any)
                - guardrail feedback, with very explicit revision instructions
            """

            orch_instruction = state.get("next_agent_payload", "").strip()
            history = state.get("history_of_steps", []) or []

            if role == "content ordering":
                data_input = state.get("data_input", "")
                return (
                    "Worker: content ordering\n\n"
                    "Worker Input:\n"
                    "ORCHESTRATOR INSTRUCTION:\n"
                    f"{orch_instruction}\n\n"
                    "DATA INPUT:\n"
                    f"{data_input}"
                ).strip()

            if role == "text structuring":
                ordering_output = latest_output_for(history, "content ordering")
                return (
                    "Worker: text structuring\n\n"
                    "Worker Input:\n"
                    "ORCHESTRATOR INSTRUCTION:\n"
                    f"{orch_instruction}\n\n"
                    "ORDERING OUTPUT:\n"
                    f"{ordering_output}"
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
                    "\n\nPREVIOUS SURFACE REALIZATION OUTPUT:\n"
                    f"{previous_sr_output}"
                    if previous_sr_output
                    else ""
                )

                feedback_block = (
                    "\n\nGUARDRAIL FEEDBACK (JSON OR TEXT):\n"
                    f"{guardrail_feedback_str}"
                    if guardrail_feedback_str
                    else "\n\nGUARDRAIL FEEDBACK: (none provided)"
                )

                # Strong instructions to stop over abstract additions and to obey feedback
                revision_instructions = """
REVISION INSTRUCTIONS:

You are revising the surface realization output based on the guardrail feedback.

1. Your goal is to produce a version that the guardrail will accept.
2. You MUST:
   - Ensure every triple from the structured input is expressed somewhere in the text.
   - For sensitive string attributes such as mottos, names, types, codes, and numeric values,
     copy the value EXACTLY as given in the triples, including parentheses, quotes, signs,
     and numbers. Do NOT paraphrase these fields.
   - If the guardrail lists triples under "omissions" with status "partially_expressed"
     or "missing", explicitly state those triples in the revised text using the exact
     subject, relation, and object strings.

3. You MUST NOT:
   - Add high level evaluative or promotional phrases that are not directly in the triples.
     Avoid words like "major", "important", "key", "vibrant", "hub", "center",
     "significant", "famous", "historic" unless those exact words appear in the data.
   - Infer roles that are not literally present in the triples, such as "represents",
     "is known as", "is a transportation hub", "is an economic center", unless you can
     quote a triple that literally encodes that relation.
   - Introduce any new facts that are not supported by the triples.

4. If the guardrail feedback lists "additions", REMOVE or REWRITE the corresponding text
   so that it is purely factual and directly grounded in the triples.

5. It is acceptable if the resulting text is slightly repetitive or less natural, as long
   as it is grammatical and fully faithful to the triples without unsupported additions.

Produce the FULL revised text, not a diff, and do NOT mention these instructions or the word "guardrail" in your answer.
"""

                return (
                    "Worker: surface realization\n\n"
                    "Worker Input:\n"
                    "ORCHESTRATOR INSTRUCTION:\n"
                    f"{orch_instruction}\n\n"
                    "TEXT STRUCTURING OUTPUT:\n"
                    f"{structuring_output}"
                    f"{previous_block}"
                    f"{feedback_block}"
                    f"\n\n{revision_instructions.strip()}\n"
                ).strip()

            # Fallback for future roles
            return orch_instruction

        def run(state: ExecutionState):
            idx = state.get("iteration_count", 0)
            inputs = build_worker_input(state)
            history = state.get("history_of_steps", []) or []

            try:
                out = agent.invoke({"input": inputs})
                # For a JSON agent, the final "Final Answer" content often ends up here:
                text = (
                    out.get("output")
                    or out.get("action_input")
                    or getattr(out, "content", str(out))
                )
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

