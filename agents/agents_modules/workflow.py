__author__ = "chinonsocynthiaosuji"

"""
Author: Chinonso Cynthia Osuji
Date: 01/10/2025
Description:
    Define the workflow for the data-to-text agent system with ablations.
"""

from typing import List, Dict, Union, Any, Literal
from langgraph.graph import START, END, StateGraph
from agents.utilities.utils import ExecutionState
from agents.agents_modules.orchestrator import TaskOrchestrator
from agents.agents_modules.worker import TaskWorker
from agents.agents_modules.guardrail import TaskGuardrail
from agents.agents_modules.finalizer import TaskFinalizer
from agents.agent_prompts import (
    CONTENT_ORDERING_PROMPT,
    TEXT_STRUCTURING_PROMPT,
    SURFACE_REALIZATION_PROMPT_EN,
    SURFACE_REALIZATION_PROMPT_GA,
)

LanguageCode = Literal["en", "ga"]

# Default surface realization prompt used by legacy code
SURFACE_REALIZATION_PROMPT = SURFACE_REALIZATION_PROMPT_EN

# Worker role maps per language
WORKER_ROLES_EN: Dict[str, str] = {
    "content ordering": CONTENT_ORDERING_PROMPT,
    "text structuring": TEXT_STRUCTURING_PROMPT,
    "surface realization": SURFACE_REALIZATION_PROMPT_EN,
}

WORKER_ROLES_GA: Dict[str, str] = {
    "content ordering": CONTENT_ORDERING_PROMPT,  # still in English for now
    "text structuring": TEXT_STRUCTURING_PROMPT,  # still in English for now
    "surface realization": SURFACE_REALIZATION_PROMPT_GA,
}

# Backwards compatible default
WORKER_ROLES = WORKER_ROLES_EN


def get_worker_roles(language: LanguageCode) -> Dict[str, str]:
    if language == "ga":
        return WORKER_ROLES_GA
    return WORKER_ROLES_EN


def get_surface_prompt(language: LanguageCode) -> str:
    if language == "ga":
        return SURFACE_REALIZATION_PROMPT_GA
    return SURFACE_REALIZATION_PROMPT_EN


def add_workers_(worker_prompts: Dict[str, str], graph: StateGraph, tools: List[Any], user_prompt: Union[str, Dict[str, Any]], provider: str,) -> List[str]:
    added: List[str] = []
    for name, prompt in worker_prompts.items():
        model = TaskWorker.init(
            description=prompt,
            tools=tools,
            context=user_prompt,
            provider=provider,
        )
        graph.add_node(name, TaskWorker.execute(model, role=name))
        added.append(name)
    return added


def add_workers(worker_prompts: Dict[str, str], graph: StateGraph, tools: List[Any], provider: str,) -> List[str]: 
    added: List[str] = []
    for name, prompt in worker_prompts.items():

        def node_fn(state, prompt=prompt, name=name):
            user_prompt = state.get("user_prompt", "")
            model = TaskWorker.init(
                description=prompt,
                tools=tools,
                context=user_prompt,
                provider=provider,
            )
            return TaskWorker.execute(model, role=name)(state)

        graph.add_node(name, node_fn)
        added.append(name)
    return added


def guardrail_routing(state: ExecutionState) -> Literal["orchestrator", "finalizer"]:
    expected = {"content ordering", "text structuring", "surface realization"}
    done = {
        step.agent_name.strip().lower()
        for step in state.get("history_of_steps", [])
        if getattr(step, "agent_name", None)
        and step.agent_name.strip().lower() in expected
    }
    review = state.get("review", "").strip().lower()

    # 1. Explicit rerun instructions always win
    if "rerun surface realization with feedback" in review:
        return "orchestrator"

    # 2. Finalize only if we are sure it is correct
    if expected.issubset(done) and "incorrect" not in review and "correct" in review:
        return "finalizer"

    # 3. Otherwise keep orchestrating
    return "orchestrator"

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


def build_agent_workflow_single_module(provider: str = "ollama", language: LanguageCode = "en",):
    """
    Ablation 1.
    Use a single generic worker module for content ordering, text structuring
    and surface realization instead of three specialized worker prompts.

    Architecture:
      orchestrator + 3 worker roles (CO, TS, SR) + guardrail + finalizer
    But all three worker roles share the same description prompt, so they are
    effectively the same module.
    """
    flow = StateGraph(ExecutionState)

    worker_roles = get_worker_roles(language)
    workers = list(worker_roles.keys())

    # Orchestrator entry
    flow.add_edge(START, "orchestrator")
    flow.set_entry_point("orchestrator")
    flow.add_node(
        "orchestrator",
        TaskOrchestrator.execute(TaskOrchestrator.init(provider)),
    )

    # Workers.
    tools: List[Any] = []
    user_prompt: Union[str, Dict[str, Any]] = ""
    single_prompt = get_surface_prompt(language)

    unified_worker_prompts: Dict[str, str] = {
        name: single_prompt for name in worker_roles.keys()
    }
    add_workers_(unified_worker_prompts, flow, tools, user_prompt, provider)

    # Guardrail and Finalizer stay as in the default
    flow.add_node("guardrail", TaskGuardrail.evaluate(TaskGuardrail.init(provider)))
    flow.add_node("finalizer", TaskFinalizer.compile(TaskFinalizer.init(provider)))

    # Routing identical to default
    routes = {name: name for name in workers}
    routes.update({"finish": "finalizer"})
    flow.add_conditional_edges("orchestrator", lambda state: state["next_agent"], routes)

    for w in workers:
        flow.add_edge(w, "guardrail")

    flow.add_conditional_edges("guardrail", guardrail_routing)
    flow.add_edge("finalizer", END)

    return flow.compile()


def build_agent_workflow_no_guardrail(provider: str = "ollama", language: LanguageCode = "en",):
    """
    Ablation 2.
    Remove the Guardrail LLM module.
    Workers send their output directly back to the orchestrator, and the
    orchestrator decides when to finish and hand off to the finalizer.
    """
    flow = StateGraph(ExecutionState)

    worker_roles = get_worker_roles(language)
    workers = list(worker_roles.keys())

    # Orchestrator entry
    flow.add_edge(START, "orchestrator")
    flow.set_entry_point("orchestrator")
    flow.add_node(
        "orchestrator",
        TaskOrchestrator.execute(TaskOrchestrator.init(provider)),
    )

    # Workers: same as default, including separate CO, TS, SR roles
    tools: List[Any] = []
    user_prompt: Union[str, Dict[str, Any]] = ""
    add_workers_(worker_roles, flow, tools, user_prompt, provider)

    # No Guardrail LLM. Replace with a simple passthrough node so that
    # we do not have to change TaskWorker logic which returns next_agent="guardrail".
    def noop_guardrail(state: ExecutionState) -> ExecutionState:
        state["review"] = "Guardrail disabled for this ablation."
        return state

    flow.add_node("guardrail", noop_guardrail)

    # Finalizer remains the same
    flow.add_node("finalizer", TaskFinalizer.compile(TaskFinalizer.init(provider)))

    # Routing.
    routes = {name: name for name in workers}
    routes.update({"finish": "finalizer"})
    flow.add_conditional_edges("orchestrator", lambda state: state["next_agent"], routes)

    # Workers go to the noop guardrail, which always goes straight back to orchestrator.
    for w in workers:
        flow.add_edge(w, "guardrail")

    flow.add_edge("guardrail", "orchestrator")
    flow.add_edge("finalizer", END)

    return flow.compile()


def build_agent_workflow_no_finalizer(provider: str = "ollama", language: LanguageCode = "en",):
    """
    Ablation 3.
    Remove the Finalizer LLM module.
    The system stops once the orchestrator signals completion.
    The last worker output is treated as the final output.
    """
    flow = StateGraph(ExecutionState)

    worker_roles = get_worker_roles(language)
    workers = list(worker_roles.keys())

    # Orchestrator entry
    flow.add_edge(START, "orchestrator")
    flow.set_entry_point("orchestrator")
    flow.add_node(
        "orchestrator",
        TaskOrchestrator.execute(TaskOrchestrator.init(provider)),
    )

    # Workers unchanged
    tools: List[Any] = []
    user_prompt: Union[str, Dict[str, Any]] = ""
    add_workers_(worker_roles, flow, tools, user_prompt, provider)

    # Guardrail unchanged
    flow.add_node("guardrail", TaskGuardrail.evaluate(TaskGuardrail.init(provider)))

    # No Finalizer LLM. Use a simple identity node that just terminates the graph.
    def noop_finalizer(state: ExecutionState) -> ExecutionState:
        return state

    flow.add_node("finalizer", noop_finalizer)

    # Routing largely matches default.
    routes = {name: name for name in workers}
    routes.update({"finish": "finalizer"})
    flow.add_conditional_edges("orchestrator", lambda state: state["next_agent"], routes)

    for w in workers:
        flow.add_edge(w, "guardrail")

    flow.add_conditional_edges("guardrail", guardrail_routing)

    flow.add_edge("finalizer", END)

    return flow.compile()


def build_agent_workflow_no_orchestrator(provider: str = "ollama", language: LanguageCode = "en",):
    """
    Ablation 4.
    Remove the Orchestrator.
    Run content ordering, text structuring and surface realization as a
    fixed workflow: CO -> TS -> SR, with guardrail and finalizer still present.

    The workers no longer depend on state["next_agent"] for routing.
    The execution order is purely given by the graph structure.
    """
    flow = StateGraph(ExecutionState)

    worker_roles = get_worker_roles(language)

    # Fixed worker order
    ordered_roles: List[str] = ["content ordering", "text structuring", "surface realization",]

    # Entry point is now the first worker, not the orchestrator
    first_role = ordered_roles[0]
    flow.add_edge(START, first_role)
    flow.set_entry_point(first_role)

    # Workers with their usual prompts
    tools: List[Any] = []
    user_prompt: Union[str, Dict[str, Any]] = ""
    add_workers_(worker_roles, flow, tools, user_prompt, provider)

    # Guardrail instances, one per stage
    guardrail_agent = TaskGuardrail.init(provider)

    flow.add_node("guardrail_co", TaskGuardrail.evaluate(guardrail_agent))
    flow.add_node("guardrail_ts", TaskGuardrail.evaluate(guardrail_agent))
    flow.add_node("guardrail_sr", TaskGuardrail.evaluate(guardrail_agent))

    # Finalizer as in the default configuration
    flow.add_node("finalizer", TaskFinalizer.compile(TaskFinalizer.init(provider)))

    # Deterministic workflow:
    # START -> CO -> guardrail_co -> TS -> guardrail_ts -> SR -> guardrail_sr -> finalizer -> END
    flow.add_edge("content ordering", "guardrail_co")
    flow.add_edge("guardrail_co", "text structuring")

    flow.add_edge("text structuring", "guardrail_ts")
    flow.add_edge("guardrail_ts", "surface realization")

    flow.add_edge("surface realization", "guardrail_sr")
    flow.add_edge("guardrail_sr", "finalizer")

    flow.add_edge("finalizer", END)

    return flow.compile()


# display(Image(process_flow.get_graph(xray=True).draw_mermaid_png()))
