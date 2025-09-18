__author__='chinonsocynthiaosuji'

"""
Author: Chinonso Cynthia Osuji
Date: 10/07/2025
Description:
    Orchestrate the task execution by directing agents to perform specific tasks
"""


import re
from typing import Dict, List, Text, Any, Union, Optional
from langchain.agents import AgentExecutor
from agents.utilities.utils import ExecutionState, AgentStepOutput
from agents.llm_model import UnifiedModel, model_name
from agents.utilities.agent_utils import summarize_agent_steps
from agents.agent_prompts import (ORCHESTRATOR_PROMPT, 
                                  ORCHESTRATOR_INPUT,
                                #   CONTENT_SELECTION_PROMPT,     
                                  CONTENT_ORDERING_PROMPT, 
                                  TEXT_STRUCTURING_PROMPT, 
                                  SURFACE_REALIZATION_PROMPT)

class TaskOrchestrator:
    @classmethod
    def init(cls, provider: str = "ollama") -> AgentExecutor:
        conf = model_name.get(provider.lower(), {}).copy()
        conf["temperature"] = 0.0
        return UnifiedModel(provider=provider, **conf).model_(ORCHESTRATOR_PROMPT)
            # ORCHESTRATOR_PROMPT.format(
            #                         #    CS=CONTENT_SELECTION_PROMPT,
            #                            CO=CONTENT_ORDERING_PROMPT,
            #                            TS=TEXT_STRUCTURING_PROMPT,
            #                            SR=SURFACE_REALIZATION_PROMPT))

    @classmethod
    def execute(cls, executor: AgentExecutor):
        def run(state: ExecutionState):
            idx = state.get("iteration_count", 0)
            limit = state.get("max_iteration", 50)
            history = state.get("history_of_steps", []) 

            prompt = state.get('user_prompt', '') # User input
            summary = "\n\n".join(summarize_agent_steps(history)[-2:]) # Last 2 agent interactions formated
            feedback = state.get('review', '') #Guardrail feedback

            #Input to the orch. agent
            payload = ORCHESTRATOR_INPUT.format(
                                        input=prompt,
                                        result_steps=f"\nRESULT STEPS: {summary}" if summary else "",
                                        feedback=f"\nFEEDBACK: {feedback}" if feedback else "",
                                        ).replace("\n\n\n", "\n")

            output = executor.invoke({"input": payload}).content.strip()
            
            # print(f"\n\nORCHESTRATOR INPUT: {payload}")
            # print(f"\n\nORCHESTRATOR OUTPUT: {output}")
            
            try:
                output_lower = output.lower()
                if any(keyword in output_lower for keyword in ["instructions:", "instruction:"]):
                    rationale, role, role_input, instruction = re.findall(r"Thought:\s*(.*?)\s*Worker:\s*(.*?)\s*Worker Input:\s*(.*?)\s*Instructions?:\s*(.*)", output, re.DOTALL)[0]
                else:
                    rationale, role, role_input = re.findall(r"Thought:\s*(.*?)\s*Worker:\s*(.*?)\s*Worker Input:\s*(.*)", output, re.DOTALL)[0]
                    instruction = None
            except Exception:
                rationale, role, role_input, instruction = "parse error", "finish", output, None

            role = role.lower().strip("'\"").replace("_", " ")

            history.append(AgentStepOutput(
                            agent_name="orchestrator",
                            agent_input=payload,
                            # agent_output=f"{role}(input='{role_input}')",
                            agent_output=f"{role}(input='{role_input}', instruction='{instruction}')",
                            rationale=f"{rationale} \nInstruction:\n{instruction}",
                        ))

            if idx >= limit:
                return {
                    "next_agent": "finish",
                    "final_response": "Stopped due to limit reached.",
                    "next_agent_payload": "Limit reached.",
                    "history_of_steps": history,
                    "iteration_count": idx + 1,
                    "max_iteration": limit
                }

            return {
                "next_agent": role,
                "final_response": role_input,
                "next_agent_payload": f"{role_input} \nAdditional Instruction: {instruction}" if instruction else role_input,
                "history_of_steps": history,
                "iteration_count": idx + 1,
                "max_iteration": limit
            }
        return run
