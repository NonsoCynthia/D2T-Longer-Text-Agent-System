import os
import json
from typing import List, Literal, Optional, Text, Union, Dict, Any

from IPython.display import Image, display
# Remove this (it is causing the ImportError):
# from langchain_core.runnables.graph import RunnableGraph
from langchain_core.runnables.graph_mermaid import MermaidDrawMethod

from agents.llm_model import UnifiedModel

from load_data import (
    extract_modified_triplesets_from_file,
    save_result_to_json,
)
from agents.agent_prompts import (
    END_TO_END_GENERATION_PROMPT_EN,
    END_TO_END_GENERATION_PROMPT_GA,
    input_prompt,
)
from agents.agents_modules.workflow import (
    build_agent_workflow,
    build_agent_workflow_single_module,
    build_agent_workflow_no_guardrail,
    build_agent_workflow_no_finalizer,
    build_agent_workflow_no_orchestrator,
)

Language = Literal["en", "ga"]
WorkflowName = Literal[
    "default",
    "single_module",
    "no_guardrail",
    "no_finalizer",
    "no_orchestrator",
]


class D2TAgentExperimentRunner:
    """
    Utility class to run data to text generation experiments with the
    multi agent system and its ablation architectures.

    Typical usage:
    - Instantiate the runner
    - Pick a sample id (1 based)
    - Pick a workflow
    - Call run_sample(...)
    """

    def __init__(
        self,
        provider: str = "openai",
        language: Language = "en",
        dataset_path: str = "data/D2T-1-FA_same3to6_min50_max500_sample.xml",
        max_iteration: int = 100,
        output_dir: str = "results",
    ) -> None:
        self.provider = provider
        self.language = language
        self.dataset_path = dataset_path
        self.max_iteration = max_iteration
        self.output_dir = output_dir

        os.makedirs(self.output_dir, exist_ok=True)

        self.triplesets: List[Any] = self._load_triplesets(self.dataset_path)
        # Use Any here instead of RunnableGraph
        self.architectures: Dict[WorkflowName, Any] = self._build_workflows()

    @staticmethod
    def _load_triplesets(path: str) -> List[Any]:
        triplesets = extract_modified_triplesets_from_file(path)
        print(f"Extracted {len(triplesets)} modified triplesets from {path}.")
        # triple_length = [{i: len(j)} for i, j in enumerate(triplesets, start=1)]
        # print(f"Triple lengths: {triple_length}")
        return triplesets

    def _build_workflows(self) -> Dict[WorkflowName, Any]:
        """
        Build and cache all workflow variants for the given provider and language.
        """
        print(f"Building workflows with provider={self.provider}, language={self.language}.")

        workflows: Dict[WorkflowName, Any] = {}

        # Default architecture
        workflows["default"] = build_agent_workflow(provider=self.provider)

        # Ablations
        workflows["single_module"] = build_agent_workflow_single_module(
            provider=self.provider,
            language=self.language,
        )
        workflows["no_guardrail"] = build_agent_workflow_no_guardrail(
            provider=self.provider,
            language=self.language,
        )
        workflows["no_finalizer"] = build_agent_workflow_no_finalizer(
            provider=self.provider,
            language=self.language,
        )
        workflows["no_orchestrator"] = build_agent_workflow_no_orchestrator(
            provider=self.provider,
            language=self.language,
        )

        print("Workflows built:", list(workflows.keys()))
        return workflows

    @property
    def triple_lengths(self) -> List[Dict[int, int]]:
        """
        Returns a list of dicts like [{1: len(tripleset1)}, {2: len(tripleset2)}, ...]
        so you can inspect how many triples each sample has.
        """
        return [{i: len(j)} for i, j in enumerate(self.triplesets, start=1)]

    @property
    def num_samples(self) -> int:
        return len(self.triplesets)

    def build_query(
        self,
        data: Any,
        dataset_name: str = "webnlg",
        custom_prefix: Optional[str] = None,
    ) -> str:
        """
        Build the natural language query that seeds the orchestrator.

        Uses the global input_prompt template:

            Dataset: {dataset_name}
            Data: {data}
        """
        template = custom_prefix if custom_prefix is not None else input_prompt

        query = template.format(
            dataset_name=dataset_name,
            data=data,
        )
        return query

    def build_initial_state(self, data: Any, query: str) -> Dict[str, Any]:
        """
        Construct the initial ExecutionState for LangGraph.
        """
        initial_state: Dict[str, Any] = {
            "data_input": data,
            "user_prompt": query,
            "raw_data": data,
            "history_of_steps": [],
            "final_response": "",
            "next_agent": "",
            "next_agent_payload": "",
            "current_step": 0,
            "iteration_count": 0,
            "max_iteration": self.max_iteration,
        }
        return initial_state

    def get_workflow(self, workflow: WorkflowName = "default") -> Any:
        if workflow not in self.architectures:
            raise ValueError(
                f"Unknown workflow '{workflow}'. "
                f"Available: {list(self.architectures.keys())}"
            )
        return self.architectures[workflow]

    def run_sample(
        self,
        sample_id: int,
        workflow: WorkflowName = "default",
        dataset_name: str = "webnlg",
        save: bool = True,
        save_prefix: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run a chosen workflow on a single sample.

        Parameters
        ----------
        sample_id:
            1 based sample id (sample 1, sample 2, etc).
        workflow:
            Which architecture to use, for example "default" or "no_orchestrator".
        dataset_name:
            Used only in the query text for documentation.
        save:
            Whether to save the full state as JSON.
        save_prefix:
            Optional prefix for the JSON filename. If None, uses workflow name.

        Returns
        -------
        state:
            Final state returned by the LangGraph workflow.
        """
        if sample_id < 1 or sample_id > len(self.triplesets):
            raise IndexError(
                f"sample_id {sample_id} is out of range. "
                f"Valid range is 1 to {len(self.triplesets)}."
            )

        index = sample_id - 1
        data = self.triplesets[index]
        print(f"Running workflow='{workflow}' on sample_id={sample_id} (index={index}).")

        query = self.build_query(data=data, dataset_name=dataset_name)
        initial_state = self.build_initial_state(data=data, query=query)

        graph = self.get_workflow(workflow)

        state = graph.invoke(
            initial_state,
            config={"recursion_limit": initial_state["max_iteration"]},
        )

        generated_text= state.get("final_response", "")

        if save:
            if save_prefix is None:
                save_prefix = workflow
            filename = f"{save_prefix}_sample{sample_id}.json"
            save_path = os.path.join(self.output_dir, filename)
            save_result_to_json(state, filename=save_path)
            print(f"Saved state to {save_path}.")

        return state, generated_text

    def show_workflow_graph(
        self,
        workflow: WorkflowName = "default",
        xray: bool = True,
    ) -> None:
        """
        Display a Mermaid diagram of the selected workflow in Jupyter.
        """
        graph = self.get_workflow(workflow)
        png_bytes = graph.get_graph(xray=xray).draw_mermaid_png(
            draw_method=MermaidDrawMethod.API
        )
        display(Image(png_bytes))




# if __name__ == "__main__":

# runner = D2TAgentExperimentRunner(
#     provider="openai",
#     language="en",
#     dataset_path="data/D2T-1-FA_same3to6_min50_max500_sample.xml",
#     output_dir="results_debug",
# )

# # Inspect how many samples and triple sizes
# print("Num samples:", runner.num_samples)
# print("Triple lengths:", runner.triple_lengths)

# # Run sample 1 with default architecture
# state1 = runner.run_sample(sample_id=1, workflow="default")

# # Run sample 2 with no_orchestrator ablation
# state2 = runner.run_sample(sample_id=2, workflow="no_orchestrator")

# # If you ever want to run everything, you can loop yourself
# for sid in range(1, runner.num_samples + 1):
#     runner.run_sample(sample_id=sid, workflow="default", save_prefix="full_default")

# # Run sample 1 with default architecture
# state1 = runner.run_sample(sample_id=1, workflow="default")

# # Run sample 2 with no_orchestrator ablation
# state2 = runner.run_sample(sample_id=2, workflow="no_orchestrator")

# # If you ever want to run everything, you can loop yourself
# for sid in range(1, runner.num_samples + 1):
#     runner.run_sample(sample_id=sid, workflow="default", save_prefix="full_default")