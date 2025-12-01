__author__='chinonsocynthiaosuji'

"""
Author: Chinonso Cynthia Osuji
Date: 10/07/2025
Description:
    Refine, polish and return the final output
"""

from langchain_classic.agents import AgentExecutor
from agents.utilities.utils import ExecutionState, AgentStepOutput
from agents.llm_model import UnifiedModel, model_name
from agents.agent_prompts import FINALIZER_PROMPT, FINALIZER_INPUT


class TaskFinalizer:
    @classmethod
    def init(cls, provider: str = "ollama") -> AgentExecutor:
        """
        Initialise the finalizer with a conservative configuration
        so it only performs light post processing.
        """
        cfg = model_name.get(provider.lower(), {})
        cfg = {**cfg, "temperature": 0.0}
        return UnifiedModel(provider=provider, **cfg).model_(FINALIZER_PROMPT)

    @classmethod
    def compile(cls, executor: AgentExecutor):
        def run(state: ExecutionState):
            history = state.get("history_of_steps", [])

            # Find the last surface realization step
            sr_steps = [
                step
                for step in history
                if getattr(step, "agent_name", "").strip().lower() == "surface realization"
            ]

            if sr_steps:
                candidate = sr_steps[-1].agent_output
            else:
                candidate = "NO SURFACE REALIZATION OUTPUT AVAILABLE."

            final_input = FINALIZER_INPUT.format(
                surface_realization_output=candidate
            )

            if state.get("response") == "incomplete":
                reply = (
                    "Final Answer: The system reached the maximum number of iterations "
                    "and no stable final output could be produced."
                )
            else:
                raw = executor.invoke({"input": final_input})
                content = getattr(raw, "content", raw)
                reply = str(content).strip()

                # Normalise in case the model omits the prefix
                if not reply.lower().startswith("final answer:"):
                    reply = "Final Answer: " + reply

            history.append(
                AgentStepOutput(
                    agent_name="finalizer",
                    agent_input=final_input,
                    agent_output=reply,
                )
            )

            return {
                "final_response": reply,
                "history_of_steps": history,
            }

        return run

