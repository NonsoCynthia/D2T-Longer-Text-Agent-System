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
                        # GUARDRAIL_PROMPT_SURFACE_REALIZATION,
                        GUARDRAIL_PROMPT_ADDITIONS_OMISSIONS,
                        GUARDRAIL_PROMPT_FLUENCY_GRAMMAR,
                        GUARDRAIL_PROMPT_FAITHFUL_ADEQUACY,
                        GUARDRAIL_PROMPT_COHERENT_NATURAL,
                    )


class TaskGuardrail:
    provider = "openai"  # default

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
            data_input = state.get("data_input", "")  # <-- triples live here

            # Find the most recent orchestrator step and worker step
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

            # Common context used by the generic guardrail and the fluency, faithfulness, coherence checks
            base_context = f"""Orchestrator Thought: {rationale}

Worker Input: {task_input}

Worker Output: {output}"""

            prompt = GUARDRAIL_INPUT.format(input=base_context)

            final_verdict = ""

            # According to the task, supply the guardrail prompt
            if task == "surface realization":
                conf = model_name.get(cls.provider)
                fluency_guard = UnifiedModel(cls.provider, **conf).model_(GUARDRAIL_PROMPT_FLUENCY_GRAMMAR)
                faithful_guard = UnifiedModel(cls.provider, **conf).model_(GUARDRAIL_PROMPT_FAITHFUL_ADEQUACY)
                coherence_guard = UnifiedModel(cls.provider, **conf).model_(GUARDRAIL_PROMPT_COHERENT_NATURAL)
                add_omission_guard = UnifiedModel(cls.provider, **conf).model_(GUARDRAIL_PROMPT_ADDITIONS_OMISSIONS)

                # Use the orchestrator, worker context for style, coherence style checks
                fluency_result = fluency_guard.invoke({"input": prompt}).content.strip().split("FEEDBACK:")[-1].strip()
                faith_result = faithful_guard.invoke({"input": prompt}).content.strip().split("FEEDBACK:")[-1].strip()
                coherence_result = coherence_guard.invoke({"input": prompt}).content.strip().split("FEEDBACK:")[-1].strip()

                # For additions and omissions, we MUST give the raw triples and the generated text
                triples_text = data_input
                # If data_input is a list of triples, make it readable
                if isinstance(data_input, list):
                    triples_text = "\n".join(str(t) for t in data_input)

                ao_context = f"""INPUT TRIPLES:\n{triples_text}\n\nGENERATED TEXT:\n{output}"""

                ao_prompt = GUARDRAIL_INPUT.format(input=ao_context)

                add_omission_result = add_omission_guard.invoke({"input": ao_prompt}).content.strip().split("FEEDBACK:")[-1].strip()

                # Extract verdict from the additions and omissions JSON
                add_omission_verdict = ""
                try:
                    ao_obj = json.loads(add_omission_result)
                    add_omission_verdict = ao_obj.get("verdict", "").strip().upper()
                except Exception:
                    # Fallback, in case the model did not return valid JSON
                    add_omission_verdict = add_omission_result.strip().upper()

                flu_ok = fluency_result.strip().upper() == "CORRECT"
                faith_ok = faith_result.strip().upper() == "CORRECT"
                coh_ok = coherence_result.strip().upper() == "CORRECT"
                ao_ok = add_omission_verdict in {"PASS", "CORRECT", ""}

                all_ok = flu_ok and faith_ok and coh_ok and ao_ok
                overall_status = "CORRECT" if all_ok else f"Rerun {task} with feedback"

                review_message = (
                    "=== GUARDRAIL REVIEW (surface realization) ===\n"
                    f"[Fluency & Grammar]: {fluency_result}\n"
                    f"[Faithfulness & Adequacy]: {faith_result}\n"
                    f"[Coherence & Naturalness]: {coherence_result}\n"
                    f"[Additions & Omissions]: {add_omission_result}\n"
                    f"OVERALL: {overall_status}"
                )
                final_verdict = review_message

            elif task == "content ordering":
                conf = model_name.get(cls.provider)
                ordering_guard = UnifiedModel(cls.provider, **conf).model_(GUARDRAIL_PROMPT_CONTENT_ORDERING)
                result = ordering_guard.invoke({"input": prompt}).content.strip()
                final_verdict = result.split("FEEDBACK:")[-1].strip()

            elif task == "text structuring":
                conf = model_name.get(cls.provider)
                structuring_guard = UnifiedModel(cls.provider, **conf).model_(GUARDRAIL_PROMPT_TEXT_STRUCTURING)
                result = structuring_guard.invoke({"input": prompt}).content.strip()
                final_verdict = result.split("FEEDBACK:")[-1].strip()

            else:
                response = agent.invoke({"input": prompt}).content.strip()
                final_verdict = response.split("FEEDBACK:")[-1].strip()
            
            print(f"\n\nGUARDRAIL OUTPUT: {final_verdict}")

            history.append(AgentStepOutput(
                agent_name="guardrail",
                agent_input=prompt,
                agent_output=final_verdict,
                rationale="Evaluation of worker output."
            ))

            # Prefer the OVERALL line if present
            overall_match = re.search(r"OVERALL:\s*(.+)$", final_verdict, re.MULTILINE)
            overall_status = overall_match.group(1).strip().upper() if overall_match else final_verdict.strip().upper()
            done = overall_status == "CORRECT"

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
