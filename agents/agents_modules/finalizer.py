__author__='chinonsocynthiaosuji'

"""
Author: Chinonso Cynthia Osuji
Date: 10/07/2025
Description:
    Refine, polish and return the final output
"""

from langchain.agents import AgentExecutor
from agents.utilities.utils import ExecutionState, AgentStepOutput
from agents.llm_model import UnifiedModel, model_name
from agents.agent_prompts import FINALIZER_PROMPT, FINALIZER_INPUT
from agents.utilities.agent_utils import summarize_agent_steps

class TaskFinalizer:
    @classmethod
    def init(cls, provider: str = "ollama") -> AgentExecutor:
        cfg = model_name.get(provider.lower())
        return UnifiedModel(provider=provider, **cfg).model_(FINALIZER_PROMPT)

    @classmethod
    def compile(cls, executor: AgentExecutor):
        def run(state: ExecutionState):
            # prompt = state.get('user_prompt', '') # User input
            history = state.get("history_of_steps", [])
            # step_log = "\n\n".join(summarize_agent_steps(history)[-2:]) or "No result steps." # Last 2 agent interactions formated
            filtered_steps = [t for t in history if getattr(t, "agent_name", "").lower() not in ["orchestrator", "guardrail"]][-2:]
            step_log = "\n\n".join(summarize_agent_steps(filtered_steps)) or "No result steps."

            final_input = FINALIZER_INPUT.format(
                                                #  input=prompt, 
                                                 result_steps=step_log
                                                 )

            if state.get("response") == "incomplete":
                reply = "I couldn't provide an answer because the maximum number of iterations was reached. Please try breaking the instruction into smaller questions by looking at the intermediate steps."
            else:
                reply = executor.invoke({"input": final_input}).content.replace("Final Answer:", "").strip()
                
            # print(f"\n\nFINALIZER INPUT: {final_input}")
            # print(f"\n\nFINALIZER OUTPUT: {reply}")

            history.append(AgentStepOutput(
                                        agent_name="finalizer",
                                        agent_input=final_input,
                                        agent_output=reply
                                    ))

            return {"final_response": reply, 
                    "history_of_steps": history
                    }
        return run
