__author__='chinonsocynthiaosuji'

"""
Author: Chinonso Cynthia Osuji
Date: 10/07/2025
Description:
    Provide guardrail evaluation for worker outputs
"""

import re
import json
from langchain_classic.agents import AgentExecutor
from agents.utilities.utils import ExecutionState, AgentStepOutput
from agents.llm_model import UnifiedModel, model_name
from agents.agent_prompts import (
    GUARDRAIL_PROMPT,
    GUARDRAIL_INPUT,
    GUARDRAIL_PROMPT_CONTENT_ORDERING,
    GUARDRAIL_PROMPT_TEXT_STRUCTURING,
    GUARDRAIL_PROMPT_SURFACE_REALIZATION, # Imported the new unified prompt
)

class TaskGuardrail:
    provider = "openai"

    @classmethod
    def init(cls, provider: str = "ollama") -> AgentExecutor:
        cls.provider = provider
        conf = model_name.get(provider.lower(), {}).copy()
        conf["temperature"] = 0.0
        return UnifiedModel(provider=provider, **conf).model_(GUARDRAIL_PROMPT)

    @classmethod
    def evaluate(cls, agent: AgentExecutor):
        def run(state: ExecutionState):
            history = state.get("history_of_steps", [])
            idx = state.get("iteration_count", 0)
            max_iter = state.get("max_iteration", 50)
            user_input = state.get("user_prompt", "")
            data_input = state.get("data_input", "")

            # Identify agents
            orch = next((s for s in reversed(history) if s.agent_name == "orchestrator"), None)
            worker = next((s for s in reversed(history) if s.agent_name not in ["orchestrator", "guardrail"]), None)
            
            task, task_input, output, rationale = "", "", "", ""
            if orch:
                rationale = orch.rationale
                match = re.search(r"^(.*?)\(input=['\"](.*?)['\"]\)$", orch.agent_output.strip(), re.DOTALL)
                if match:
                    task, task_input = match.groups()
            if worker:
                output = worker.agent_output

            # Prepare base context
            base_context = f"""Orchestrator Thought: {rationale}\nWorker Input: {task_input}\nWorker Output: {output}"""
            prompt = GUARDRAIL_INPUT.format(input=base_context)
            final_verdict = ""
            
            # --- SURFACE REALIZATION (OPTIMIZED: 1 CALL) ---
            if task == "surface realization":
                conf = model_name.get(cls.provider)
                unified_guard = UnifiedModel(cls.provider, **conf).model_(GUARDRAIL_PROMPT_SURFACE_REALIZATION)

                # Prepare Triples + Text context
                triples_text = data_input
                if isinstance(data_input, list):
                    triples_text = "\n".join(str(t) for t in data_input)

                unified_context = f"""INPUT TRIPLES:\n{triples_text}\n\nGENERATED TEXT:\n{output}"""
                unified_prompt = GUARDRAIL_INPUT.format(input=unified_context)

                # Single LLM Call
                raw_response = unified_guard.invoke({"input": unified_prompt}).content.strip()
                
                # Cleanup potential markdown wrapper from LLM (e.g., ```json ... ```)
                clean_json = raw_response.replace("FEEDBACK:", "").strip()
                if "```" in clean_json:
                    clean_json = clean_json.split("```")[1].replace("json", "").strip()

                try:
                    eval_data = json.loads(clean_json)
                    
                    overall_status = eval_data.get("overall_verdict", "FAIL").upper()
                    ling_score = eval_data.get("linguistic_score", "FAIL")
                    ling_feed = eval_data.get("linguistic_feedback", "")
                    omissions = eval_data.get("omissions", [])
                    additions = eval_data.get("additions", [])

                    # Construct readable feedback for the Orchestrator
                    review_parts = [
                        "=== GUARDRAIL REVIEW (Unified) ===",
                        f"[Linguistic Quality]: {ling_score} - {ling_feed}",
                        f"[Factuality]: {eval_data.get('factuality_verdict', 'FAIL')}",
                    ]
                    if omissions:
                        review_parts.append(f"  - Omissions: {omissions}")
                    if additions:
                        review_parts.append(f"  - Additions/Hallucinations: {additions}")
                    
                    review_parts.append(f"OVERALL: {overall_status}")
                    
                    final_verdict = "\n".join(review_parts)

                except json.JSONDecodeError:
                    # Fallback if JSON fails
                    print(f"JSON Parse Error. Raw: {raw_response}")
                    final_verdict = f"GUARDRAIL ERROR: Could not parse evaluation.\nRaw output: {raw_response}\nOVERALL: FAIL"

            # --- CONTENT ORDERING ---
            elif task == "content ordering":
                conf = model_name.get(cls.provider)
                ordering_guard = UnifiedModel(cls.provider, **conf).model_(GUARDRAIL_PROMPT_CONTENT_ORDERING)
                result = ordering_guard.invoke({"input": prompt}).content.strip()
                final_verdict = result.split("FEEDBACK:")[-1].strip()

            # --- TEXT STRUCTURING ---
            elif task == "text structuring":
                conf = model_name.get(cls.provider)
                structuring_guard = UnifiedModel(cls.provider, **conf).model_(GUARDRAIL_PROMPT_TEXT_STRUCTURING)
                result = structuring_guard.invoke({"input": prompt}).content.strip()
                final_verdict = result.split("FEEDBACK:")[-1].strip()

            # --- DEFAULT ---
            else:
                response = agent.invoke({"input": prompt}).content.strip()
                final_verdict = response.split("FEEDBACK:")[-1].strip()
            
            print(f"\n\nGUARDRAIL OUTPUT: {final_verdict}")

            history.append(
                AgentStepOutput(
                    agent_name="guardrail",
                    agent_input=prompt,
                    agent_output=final_verdict,
                    rationale="Evaluation of worker output."
                )
            )

            # Check for "OVERALL: CORRECT" or just "CORRECT"
            overall_match = re.search(r"OVERALL:\s*(.+)$", final_verdict, re.MULTILINE)
            overall_status = overall_match.group(1).strip().upper() if overall_match else final_verdict.strip().upper()
            
            # Allow pass if status is CORRECT or if it's a simple "CORRECT" string
            done = "CORRECT" in overall_status

            return {
                "next_agent": "finalizer" if done else "orchestrator",
                "response": "done" if done else None,
                "history_of_steps": history,
                "iteration_count": idx + 1,
                "max_iteration": max_iter,
                "next_agent_payload": user_input,
                "review": final_verdict,
            }

        return run
