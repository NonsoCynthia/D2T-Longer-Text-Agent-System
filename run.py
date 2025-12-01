#!/usr/bin/env python
import argparse
import json
import os
from typing import Any, Dict, Optional

from main import D2TAgentExperimentRunner
from agents.llm_model import model_name


def run_all_samples(
    runner: D2TAgentExperimentRunner,
    workflow: str,
    run_mode: str,
    provider: str,
    temperature: float,
    start_id: int,
    end_id: Optional[int],
) -> None:
    data, num_samples, _ = runner.inspect_data

    if end_id is None or end_id > num_samples:
        end_id = num_samples

    if start_id < 1 or start_id > num_samples:
        raise ValueError(f"start-id {start_id} is out of range. Valid range is 1 to {num_samples}.")

    print(f"Total samples: {num_samples}")
    print(f"Running samples {start_id} to {end_id} inclusive.")
    print(f"Run mode: {run_mode}. Workflow: {workflow}. Provider: {provider}.")

    for sample_id in range(start_id, end_id + 1):
        print(f"\n=== Sample {sample_id} ===")

        if run_mode in ("e2e", "both"):
            print("Running end to end generation...")
            e2e_result = runner.run_end_to_end(
                sample_id=sample_id,
                provider=provider,
                temperature=temperature,
            )

            txt_path = os.path.join(
                runner.output_dir,
                f"e2e_{runner.language}_sample{sample_id}.txt",
            )
            meta_path = os.path.join(
                runner.output_dir,
                f"e2e_{runner.language}_sample{sample_id}.json",
            )

            os.makedirs(runner.output_dir, exist_ok=True)

            with open(txt_path, "w", encoding="utf-8") as f_txt:
                f_txt.write(e2e_result["generated_text"])

            meta: Dict[str, Any] = {
                "sample_id": sample_id,
                "language": runner.language,
                "provider": provider,
                "query": e2e_result["query"],
                "data": e2e_result["data"],
            }
            with open(meta_path, "w", encoding="utf-8") as f_meta:
                json.dump(meta, f_meta, ensure_ascii=False, indent=2)

            print(f"Saved end to end outputs to {txt_path} and {meta_path}.")

        if run_mode in ("pipeline", "both"):
            print(f"Running multi agent pipeline with workflow '{workflow}'...")
            state = runner.run_sample(
                sample_id=sample_id,
                workflow=workflow,
                save=True,
                save_prefix=f"{workflow}",
            )
            final_text = state.get("final_response", "")
            print(f"Pipeline final_response length: {len(final_text)} characters.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run data to text experiments over a full dataset."
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="openai",
        help="Model provider. Example: openai, ollama, hf, aixplain.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        choices=["en", "ga"],
        help="Language code. en or ga.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="data/D2T-1-FA_same3to6_min50_max500_sample.xml",
        help="Path to the XML dataset file with modified triplesets.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results_cli",
        help="Directory where outputs will be saved.",
    )
    parser.add_argument(
        "--workflow",
        type=str,
        default="default",
        choices=["default", "single_module", "no_guardrail", "no_finalizer", "no_orchestrator"],
        help="Which multi agent workflow to run for the pipeline mode.",
    )
    parser.add_argument(
        "--run-mode",
        type=str,
        default="both",
        choices=["e2e", "pipeline", "both"],
        help="e2e for single model, pipeline for multi agent, both for both.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for the end to end model.",
    )
    parser.add_argument(
        "--max-iteration",
        type=int,
        default=100,
        help="Maximum LangGraph recursion limit for the pipeline.",
    )
    parser.add_argument(
        "--start-id",
        type=int,
        default=1,
        help="First sample id to process. 1 based index.",
    )
    parser.add_argument(
        "--end-id",
        type=int,
        default=None,
        help="Last sample id to process. If omitted, runs until the last sample.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    provider = args.provider
    language = args.language
    dataset_path = args.dataset_path
    output_dir = args.output_dir
    workflow = args.workflow
    run_mode = args.run_mode
    temperature = args.temperature
    max_iteration = args.max_iteration
    start_id = args.start_id
    end_id = args.end_id

    conf = model_name.get(provider.lower(), {}).copy()
    conf["temperature"] = temperature

    print(f"Initialising runner with provider={provider}, language={language}.")
    runner = D2TAgentExperimentRunner(
        provider=provider,
        language=language,  # type: ignore[arg-type]
        dataset_path=dataset_path,
        max_iteration=max_iteration,
        output_dir=output_dir,
    )

    run_all_samples(
        runner=runner,
        workflow=workflow,
        run_mode=run_mode,
        provider=provider,
        temperature=temperature,
        start_id=start_id,
        end_id=end_id,
    )


if __name__ == "__main__":
    main()
