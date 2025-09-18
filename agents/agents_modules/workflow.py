__author__='chinonsocynthiaosuji'

"""
Author: Chinonso Cynthia Osuji
Date: 10/07/2025
Description:
    Define the workflow for the data-to-text agent system
"""

from typing import List, Dict, Union, Any, Literal
from langgraph.graph import START, END, StateGraph
from agents.utilities.utils import ExecutionState
from agents.agents_modules.orchestrator import TaskOrchestrator
from agents.agents_modules.worker import TaskWorker
from agents.agents_modules.guardrail import TaskGuardrail
from agents.agents_modules.finalizer import TaskFinalizer
from agents.agent_prompts import (CONTENT_ORDERING_PROMPT, 
                                  TEXT_STRUCTURING_PROMPT, 
                                  SURFACE_REALIZATION_PROMPT)


WORKER_ROLES = {
    "content ordering": CONTENT_ORDERING_PROMPT,
    "text structuring": TEXT_STRUCTURING_PROMPT,
    "surface realization": SURFACE_REALIZATION_PROMPT,
}

def add_workers_(worker_prompts: Dict[str, str], graph: StateGraph, tools: List[Any], user_prompt: Union[str, Dict[str, Any]], provider: str) -> List[str]:
    added = []
    for name, prompt in worker_prompts.items():
        model = TaskWorker.init(description=prompt, tools=tools, context=user_prompt, provider=provider)
        graph.add_node(name, TaskWorker.execute(model, role=name))
        added.append(name)
    return added

def add_workers(worker_prompts: Dict[str, str], graph: StateGraph, tools: List[Any], provider: str) -> List[str]:
    added = []
    for name, prompt in worker_prompts.items():
        def node_fn(state, prompt=prompt, name=name):
            user_prompt = state.get("user_prompt", "")
            model = TaskWorker.init(description=prompt, tools=tools, context=user_prompt, provider=provider)
            return TaskWorker.execute(model, role=name)(state)
        graph.add_node(name, node_fn)
        added.append(name)
    return added

def guardrail_routing(state: ExecutionState) -> Literal["orchestrator", "finalizer"]:
    expected = {"content ordering", "text structuring", "surface realization"}
    done = {step.agent_name.strip().lower() for step in state.get("history_of_steps", []) if getattr(step, 'agent_name', None) and step.agent_name.strip().lower() in expected}
    review = state.get("review", "").strip().lower()

    # Move to finalizer only if all required steps are done and output is marked correct
    if expected.issubset(done) and "correct" in review:
        return "finalizer"
    # If surface realization needs to be rerun, route back to orchestrator
    if "rerun surface realization with feedback" in review:
        return "orchestrator"
    # Default: continue orchestration
    return "orchestrator"
    # return "finalizer" if expected.issubset(done) and "correct" in review else "orchestrator"

def build_agent_workflow(provider: str = "ollama") -> StateGraph:
    flow = StateGraph(ExecutionState)
    workers = list(WORKER_ROLES.keys())

    flow.add_edge(START, "orchestrator")
    flow.set_entry_point("orchestrator")

    # Orchestrator
    flow.add_node("orchestrator", TaskOrchestrator.execute(TaskOrchestrator.init(provider)))

    # Workers
    tools = []
    user_prompt = ""
    add_workers_(WORKER_ROLES, flow, tools, user_prompt, provider) #remove user prompt
    # add_workers(WORKER_ROLES, flow, tools, provider) #Add user prompt

    # guardrail & Finalizer
    flow.add_node("guardrail", TaskGuardrail.evaluate(TaskGuardrail.init(provider)))
    flow.add_node("finalizer", TaskFinalizer.compile(TaskFinalizer.init(provider)))

    # Routing
    routes = {name: name for name in workers}
    routes.update({"finish": "finalizer"})
    flow.add_conditional_edges("orchestrator", lambda state: state["next_agent"], routes)
    for w in workers:
        flow.add_edge(w, "guardrail")
    flow.add_conditional_edges("guardrail", guardrail_routing) #Original
    # flow.add_edge("guardrail", "orchestrator") # Always to orchestrator
    flow.add_edge("finalizer", END)

    return flow.compile()

# display(Image(process_flow.get_graph(xray=True).draw_mermaid_png()))
