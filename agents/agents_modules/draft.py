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

#################################################################

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
                        # GUARDRAIL_PROMPT_ADDITIONS_OMISSIONS,  # no longer needed, kept here for reference but commented
                        GUARDRAIL_PROMPT_OMISSIONS,
                        GUARDRAIL_PROMPT_ADDITIONS,
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
            base_context = f"""Orchestrator Thought: {rationale}\nWorker Input: {task_input}\nWorker Output: {output}"""

            prompt = GUARDRAIL_INPUT.format(input=base_context)

            final_verdict = ""

            # According to the task, supply the guardrail prompt
            if task == "surface realization":
                conf = model_name.get(cls.provider)
                fluency_guard = UnifiedModel(cls.provider, **conf).model_(GUARDRAIL_PROMPT_FLUENCY_GRAMMAR)
                faithful_guard = UnifiedModel(cls.provider, **conf).model_(GUARDRAIL_PROMPT_FAITHFUL_ADEQUACY)
                coherence_guard = UnifiedModel(cls.provider, **conf).model_(GUARDRAIL_PROMPT_COHERENT_NATURAL)

                # add_omission_guard = UnifiedModel(cls.provider, **conf).model_(GUARDRAIL_PROMPT_ADDITIONS_OMISSIONS)
                # New, separate guards for omissions and additions
                omission_guard = UnifiedModel(cls.provider, **conf).model_(GUARDRAIL_PROMPT_OMISSIONS)
                addition_guard = UnifiedModel(cls.provider, **conf).model_(GUARDRAIL_PROMPT_ADDITIONS)

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

                # add_omission_result = add_omission_guard.invoke({"input": ao_prompt}).content.strip().split("FEEDBACK:")[-1].strip()
                
                omissions_raw = omission_guard.invoke({"input": ao_prompt}).content.strip().split("FEEDBACK:")[-1].strip()
                additions_raw = addition_guard.invoke({"input": ao_prompt}).content.strip().split("FEEDBACK:")[-1].strip()
                
                # Extract verdict from the additions and omissions JSON
                add_omission_verdict = ""
                omissions_verdict = ""
                additions_verdict = ""
                # try:
                #     ao_obj = json.loads(add_omission_result)
                #     add_omission_verdict = ao_obj.get("verdict", "").strip().upper()
                # except Exception:
                #     # Fallback, in case the model did not return valid JSON
                #     add_omission_verdict = add_omission_result.strip().upper()

                try:
                    om_obj = json.loads(omissions_raw)
                    omissions_verdict = om_obj.get("verdict", "").strip().upper()
                except Exception:
                    omissions_verdict = omissions_raw.strip().upper()

                try:
                    ad_obj = json.loads(additions_raw)
                    additions_verdict = ad_obj.get("verdict", "").strip().upper()
                except Exception:
                    additions_verdict = additions_raw.strip().upper()

                flu_ok = fluency_result.strip().upper() == "CORRECT"
                faith_ok = faith_result.strip().upper() == "CORRECT"
                coh_ok = coherence_result.strip().upper() == "CORRECT"
                # ao_ok = add_omission_verdict in {"PASS", "CORRECT", ""}
                om_ok = omissions_verdict in {"PASS", "CORRECT", ""}
                ad_ok = additions_verdict in {"PASS", "CORRECT", ""}

                ao_ok = om_ok and ad_ok  # Both omissions and additions must be okay
                all_ok = flu_ok and faith_ok and coh_ok and ao_ok
                overall_status = "CORRECT" if all_ok else f"Rerun {task} with feedback"

                review_message = (
                    "=== GUARDRAIL REVIEW (surface realization) ===\n"
                    f"[Fluency & Grammar]: {fluency_result}\n"
                    f"[Faithfulness & Adequacy]: {faith_result}\n"
                    f"[Coherence & Naturalness]: {coherence_result}\n"
                    # f"[Additions & Omissions]: {add_omission_result}\n"
                    f"[Omissions]: {omissions_raw}\n"
                    f"[Additions]: {additions_raw}\n"
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

            history.append(
                AgentStepOutput(
                    agent_name="guardrail",
                    agent_input=prompt,
                    agent_output=final_verdict,
                    rationale="Evaluation of worker output."
                )
            )

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

######################################################
GUARDRAIL_PROMPT_OMISSIONS = """
You are a factuality guardrail for a data to text system. Your job is to find only clear omissions.

You are given:
1. A list of triples (subject, relation, object).
2. A generated text that should describe information supported by these triples.

Closed world assumption:
- The triples are the only facts you know.
- Ignore real world knowledge.
- You only judge whether any input triples are missing from the text.

Definitions:
- Supported content: the meaning of a triple appears in the text, even if paraphrased, reordered, translated, or with minor wording changes.
- Omission: an input triple whose core meaning does not appear anywhere in the text.

Rules:
- If the meaning of a triple is expressed even loosely, do not mark it as omitted.
- Reversed relation direction is allowed. If (A, relation, B) is expressed as B having that relation to A, treat it as supported.
- Triples where the main entity appears as object still count.

Verdict:
- "PASS" if there are no clearly important omitted triples.
- "FAIL" otherwise.

Output:
Return a strict JSON object with no extra text:
```json
{{
  "verdict": "PASS" or "FAIL",
  "omissions": [
    {{
      "triple": "(subject, relation, object)",
      "status": "omitted",
      "evidence": "Short explanation of what is missing."
    }}
  ],
  "additions": []
}}
```
Constraints:
- Always return "additions" as an empty list.
- Use only status "omitted".
- Never report a triple as omitted if its meaning is even partially expressed.
- Do not judge additions here. That is done by a separate guardrail.
"""

GUARDRAIL_PROMPT_ADDITIONS = """
You are a factuality guardrail for a data to text system. Your job is to find only clear additions or hallucinations.

You are given:
1. A list of triples (subject, relation, object).
2. A generated text that should describe information supported by these triples.

Closed world assumption:
- The triples are the only facts you know.
- Ignore real world knowledge.
- You only judge whether any factual claims in the text cannot be grounded in the triples.

Definitions:
- Supported content: the meaning of a claim can be mapped to one or more triples, allowing paraphrase, reordering, translation and minor wording changes.
- Addition or hallucination: a factual claim that cannot be matched to any triple.

Rules for additions:
- Mark as addition if the text introduces:
  - a new entity not in any triple, or
  - a new relation between entities not in any triple, or
  - a new number, date, rank or count not supported by the triples.
- Do not mark as additions harmless wording choices that clearly reflect some triple.
- Reversed relation direction is allowed. If (A, relation, B) is expressed as B having that relation to A, treat it as supported.

Verdict:
- "PASS" if there are no clear hallucinated additions.
- "FAIL" otherwise.

Output:
Return a strict JSON object with no extra text:
```json
{{
  "verdict": "PASS" or "FAIL",
  "omissions": [],
  "additions": [
    {{
      "text_span": "Exact quote or short excerpt from the generated text",
      "reason": "Why this cannot be grounded in the input triples."
    }}
  ]
}}
```
Constraints:
- Always return "omissions" as an empty list.
- Do not try to detect omitted triples. That is done by a separate guardrail.
- Do not treat a sentence as an addition only because it reverses a relation between two entities that appear together in a triple.
"""


GUARDRAIL_PROMPT_ADDITIONS_OMISSIONS = """You are a factuality guardrail for a data to text system. Your job is to compare a generated description with a list of input triples and report clear omissions and clear additions.

You will be given:
1. A list of triples in the form (subject, relation, object).
2. A generated text that should describe only information supported by these triples.

CLOSED WORLD ASSUMPTION.
- Treat the triples as the only facts you know.
- Ignore real world knowledge and do not assume extra facts.
- When you talk about omissions, you may only reference triples that literally appear in the input.

Key principle.
This is not summarisation. The ideal text would cover all triples, but minor gaps are acceptable. Your focus is:
- Detecting clear hallucinations.
- Detecting triples whose meaning is completely missing from the text.
If a triple is mentioned even loosely, it should be treated as covered and not reported as an omission.

Definitions.
- Supported content. A statement whose main factual meaning can be directly mapped to one or more input triples, allowing:
  - light paraphrase,
  - changes in word order,
  - formatting differences,
  - small wording variations such as spelling changes, function words, or synonyms.
  Example: the triple object "Capital den largest city for Romania" is considered supported if the text says "the capital and largest city of Romania".
- Omission. An input triple whose core meaning does not appear at all in the text.
- Addition or hallucination. A specific factual claim in the text that cannot be grounded in any input triple.

Omissions.
- A triple is omitted only if its central meaning is not present anywhere in the text.
- If the meaning of a triple is expressed approximately, paraphrased, translated, or slightly underspecified, treat it as supported and do not list it as an omission.
- Triples where the main entity appears as object, such as (OtherEntity, relation, MainEntity), still count and should not be ignored.

Additions.
- A statement is an addition if it introduces, including but not limited to:
  - a new entity not present in any triple, or
  - a new relation between entities that is not present in any triple, or
  - a new numeric value, ranking, date, or count that is not supported by the triples.
- Do not mark as additions harmless wording choices or paraphrases that clearly reflect some triple.

Direction of relations.
- It does not matter if the direction of a relation is reversed in the text.
- If the triple is (A, relation, B), and the text says something that clearly corresponds to B having the same relation to A, treat this as supported, not as an omission and not as an addition.
- Concretely, do not penalise sentences that flip subject and object, as long as the same pair of entities and the same qualitative relation are preserved.

Your job.
1. Read all input triples. These are the only facts you may use.
2. Read the generated text carefully.
3. For each triple, internally decide whether it is expressed or omitted, but only report triples that are clearly omitted.
4. Scan the text for clear factual claims that cannot be mapped to any triple, allowing harmless wording differences, paraphrases, and reversed relation direction. These are "additions".
5. Decide an overall verdict:
   - "PASS" if:
     - there are no clear hallucinated additions, and
     - there are no clearly important omitted triples.
   - Otherwise, "FAIL".

Output format.
Return a strict JSON object with no extra commentary:
```json
{{
  "verdict": "PASS" or "FAIL",
  "omissions": [
    {{
      "triple": "(subject, relation, object)",
      "status": "omitted",
      "evidence": "Short explanation of what is missing."
    }}
  ],
  "additions": [
    {{
      "text_span": "Exact quote or short excerpt from the generated text",
      "reason": "Why this cannot be grounded in the input triples."
    }}
  ]
}}

Constraints.
- Never use any status other than "omitted".
- Never report a triple as omitted if its meaning is even loosely or partially expressed in the text.
- Never treat a sentence as an addition solely because it reverses the direction of a relation between two entities that appear together in a triple.
"""


GUARDRAIL_PROMPT_FLUENCY_GRAMMAR = """You are a guardrail focused on evaluating the **fluency** and **grammatical correctness** of a generated text in a data-to-text generation pipeline. You will receive a complete paragraph level or sentence level generated text for evaluation.

*** Definitions ***
- **Fluency** refers to how smoothly and naturally the output reads. A fluent sentence has appropriate word choice, sentence rhythm, and no awkward or choppy phrasing.
- **Grammaticality** refers to the correctness of language according to standard grammar rules, including subject-verb agreement, tense consistency, punctuation, and syntactic structure.

*** Task ***
Determine whether the generated output is readable, well-formed, and free of grammatical issues.

*** Evaluation Criteria ***
- **Fluency**: Sentences should read naturally and avoid awkward constructions or unnatural collocations.
- **Grammaticality**: The text must be grammatically correct according to formal written English norms.
- Penalize the text if there are repetitions such as in facts.

*** Output Format ***
- If both criteria are met: respond with 'CORRECT'
- If either is violated: return a concise one-sentence specific explanation.

FEEDBACK:
"""


GUARDRAIL_PROMPT_FAITHFUL_ADEQUACY = """You are a guardrail focused on evaluating **faithfulness** to the input data and the **adequacy** of the output content in a data-to-text generation task. You will receive a complete paragraph level or sentence level generated text for evaluation.

*** Definitions ***
- **Faithfulness** means that the output must remain factually accurate and reflect only the information present in the input. No fabricated, altered, or hallucinated information is allowed.
- **Adequacy** means that the output must include all the critical and salient facts from the input data. It should not omit important content necessary for understanding the data.

*** Task ***
Verify that the output is strictly derived from the input and comprehensively conveys its key information.

*** Evaluation Criteria ***
- **Faithfulness**: Every statement in the output must be traceable to the input data.
- **Adequacy**: All major data points should be present; the text should not skip or ignore essential facts.

*** Output Format ***
- If both criteria are satisfied: respond with 'CORRECT'
- If either is violated: return a concise one-sentence specific explanation.

FEEDBACK:
"""


GUARDRAIL_PROMPT_COHERENT_NATURAL = """You are a guardrail evaluating whether the generated text is **coherent** and **natural** in a data-to-text generation task. You will receive a complete paragraph level or sentence level generated text for evaluation.

*** Definitions ***
- **Coherence** refers to how well the ideas and facts in the text are organized and connected. A coherent output has a logical structure and clear flow, even when multiple data points are presented.
- **Naturalness** refers to whether the output sounds like it was written by a human. It should avoid stilted, robotic, or overly templated language.

*** Task ***
Assess whether the text presents the information in a clear, logically connected manner and reads as if authored by a human.

*** Evaluation Criteria ***
- **Coherence**: Sentences should connect well; transitions between ideas must make sense.
- **Naturalness**: The phrasing should resemble that of human writing, not mechanical output.

*** Output Format ***
- If both criteria are met: respond with 'CORRECT'
- If either is violated: return a concise one-sentence specific explanation.

FEEDBACK:
"""

