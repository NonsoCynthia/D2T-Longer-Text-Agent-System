#!/usr/bin/env python
import argparse
import json
import os
import time
from typing import Callable, Dict, List, Tuple, Union

from main import D2TAgentExperimentRunner

# Default configuration
DEFAULT_DATASET_PATH = "data/D2T-1-FA_same3to6_min8_max199.xml"
DEFAULT_MODEL_PROVIDER = "openai"
DEFAULT_RESULTS_DIR = "results"

# Outputs that should trigger a rerun
INVALID_OUTPUTS = {
    "",
    "NO SURFACE REALIZATION OUTPUT AVAILABLE.",
}

# Safety cap so we do not loop forever
DEFAULT_MAX_RERUNS = 3


def clean_text(text: str) -> str:
    """
    Normalise model output.
    - Strip leading and trailing whitespace
    - Remove a 'final answer:' prefix if present
    """
    if not text:
        return ""
    text = text.strip()
    prefix = "final answer:"
    lower = text.lower()
    if lower.startswith(prefix):
        text = text[len(prefix):].lstrip()
    return text


def is_invalid(text: str) -> bool:
    """
    Decide whether an output should be considered invalid.
    """
    if text is None:
        return True
    stripped = text.strip()
    if stripped in INVALID_OUTPUTS:
        return True
    return False


def run_with_retries(
    run_once: Callable[[], str],
    label: str,
    sample_id: int,
    lang: str,
    max_retries: int,
) -> Tuple[str, int, float]:
    """
    Run a generation function repeatedly until we get a valid output
    or hit the retry limit.

    Returns:
        (final_output, attempts_used, total_time_seconds)
    """
    attempt = 0
    last_output = ""
    start_time = time.perf_counter()

    while attempt < max_retries:
        attempt += 1
        print(f"[{label}][{lang}] sample_id={sample_id} attempt {attempt}/{max_retries}...")
        try:
            raw = run_once()
            cleaned = clean_text(raw)
            last_output = cleaned
            if not is_invalid(cleaned):
                elapsed = time.perf_counter() - start_time
                print(
                    f"[{label}][{lang}] sample_id={sample_id} "
                    f"succeeded on attempt {attempt} in {elapsed:.2f}s"
                )
                return cleaned, attempt, elapsed
            else:
                print(
                    f"[{label}][{lang}] sample_id={sample_id} produced invalid output: "
                    f"{repr(cleaned[:80])}..."
                )
        except Exception as e:
            print(f"[{label}][{lang}] sample_id={sample_id} error on attempt {attempt}: {e}")
            last_output = "ERROR"

    elapsed = time.perf_counter() - start_time
    print(
        f"[{label}][{lang}] sample_id={sample_id} failed "
        f"after {max_retries} attempts in {elapsed:.2f}s."
    )
    return last_output, attempt, elapsed


def load_existing_results(path: str) -> Dict[int, Dict]:
    """
    Load existing JSON result file and index by zero based 'index'.
    If the file does not exist, return an empty dict.
    """
    if not os.path.exists(path):
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[WARN] Could not read existing file {path}: {e}")
        return {}

    by_index: Dict[int, Dict] = {}
    for item in data:
        idx = item.get("index")
        if isinstance(idx, int):
            by_index[idx] = item
    return by_index


def save_results_incremental(path: str, by_index: Dict[int, Dict]) -> None:
    """
    Save the results dict as a sorted list after each update.
    """
    results_list: List[Dict] = [
        by_index[i] for i in sorted(by_index.keys())
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results_list, f, indent=4, ensure_ascii=False)


def run_system_for_language(
    lang: str,
    system: str,
    provider: str,
    dataset_path: str,
    data: List,
    limit: int,
    results_dir: str,
    max_retries: int,
) -> None:
    """
    Run exactly one system (default, unified_worker, or e2e)
    for one language, with retries, skipping already valid examples.

    Agent step files are saved by D2TAgentExperimentRunner in:
        {system}_{lang}/...

    Aggregated final outputs are saved in:
        results/{system}_{lang}.json
    """
    # Folder for agent steps (LangGraph states etc)
    state_dir = f"{system}_{lang}"

    # Runner for this (system, language) pair
    runner = D2TAgentExperimentRunner(
        provider=provider,
        language=lang,
        dataset_path=dataset_path,
        output_dir=state_dir,
    )

    os.makedirs(results_dir, exist_ok=True)
    result_filename = f"{system}_{lang}.json"
    json_path = os.path.join(results_dir, result_filename)

    # Load any existing aggregated results so we can skip valid ones
    existing = load_existing_results(json_path)
    print(f"[{lang}][{system}] Existing entries: {len(existing)} from {json_path}")

    # Decide label used in logs
    label = system.upper()

    for idx in range(limit):
        sample_id = idx + 1
        triples = data[idx]

        # Skip if we already have a valid output
        prev = existing.get(idx)
        if prev is not None and not is_invalid(prev.get("output", "")):
            print(
                f"[{label}][{lang}] index={idx} sample_id={sample_id} "
                f"already has valid output. skipping."
            )
            continue

        print(
            f"\n[{label}][{lang}] Processing sample index={idx} "
            f"(sample_id={sample_id}) / {limit - 1}..."
        )

        # Define single run function according to system
        if system == "default":
            def run_once() -> str:
                state = runner.run_sample(
                    sample_id=sample_id,
                    workflow="default",
                    save=True,
                )
                return state.get("final_response", "")

        elif system == "unified_worker":
            def run_once() -> str:
                state = runner.run_sample(
                    sample_id=sample_id,
                    workflow="unified_worker",
                    save=True,
                )
                return state.get("final_response", "")

        elif system == "e2e":
            def run_once() -> str:
                e2e = runner.run_end_to_end(
                    sample_id=sample_id,
                    provider=provider,
                    temperature=0.0,
                )
                return e2e.get("generated_text", "")
        else:
            raise ValueError(f"Unknown system: {system}")

        out, attempts, elapsed = run_with_retries(
            run_once=run_once,
            label=label,
            sample_id=sample_id,
            lang=lang,
            max_retries=max_retries,
        )

        row = {
            "index": idx,           # zero based
            "sample_id": sample_id, # one based for trace back
            "triples": triples,
            "output": out,
            "attempts": attempts,
            "time": elapsed,
        }
        existing[idx] = row
        save_results_incremental(json_path, existing)

    print(f"[{lang}][{system}] Completed. Final file at {json_path}")


def normalise_to_list(x: Union[str, List[str]]) -> List[str]:
    """
    Allow systems and languages to be passed as a single string or a list.
    """
    if isinstance(x, str):
        return [x]
    return list(x)


def run_for_languages_and_systems(
    languages: Union[List[str], str],
    systems: Union[List[str], str],
    dataset_path: str,
    provider: str,
    results_dir: str,
    max_samples_to_run: int,
    max_retries: int,
) -> None:
    languages_list = normalise_to_list(languages)
    systems_list = normalise_to_list(systems)

    for lang in languages_list:
        print(f"\n{'=' * 40}")
        print(f"Starting experiments for language: {lang.upper()}")
        print(f"{'=' * 40}")

        # Temporary runner just to load dataset and stats
        tmp_runner = D2TAgentExperimentRunner(
            provider=provider,
            language=lang,
            dataset_path=dataset_path,
            output_dir="_tmp_ignore",
        )
        data, num_samples, triple_lengths = tmp_runner.inspect_data
        print(f"[{lang}] Total samples available: {num_samples}")
        print(f"[{lang}] Triple lengths (first few): {triple_lengths[:5]}")

        if max_samples_to_run is not None and max_samples_to_run < num_samples:
            limit = max_samples_to_run
        else:
            limit = num_samples

        for system in systems_list:
            run_system_for_language(
                lang=lang,
                system=system,
                provider=provider,
                dataset_path=dataset_path,
                data=data,
                limit=limit,
                results_dir=results_dir,
                max_retries=max_retries,
            )

    print("\nAll requested experiments completed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run D2T experiments for selected architectures and languages."
    )

    parser.add_argument(
        "--languages",
        "-l",
        nargs="+",
        choices=["en", "ga"],
        default=["en", "ga"],
        help="Languages to run. choose from 'en', 'ga'. default is both.",
    )

    parser.add_argument(
        "--systems",
        "-s",
        nargs="+",
        choices=["default", "unified_worker", "e2e"],
        default=["default", "unified_worker", "e2e"],
        help="Systems to run: default, unified_worker, e2e.",
    )

    parser.add_argument(
        "--dataset",
        "-d",
        default=DEFAULT_DATASET_PATH,
        help=f"Path to dataset XML. default: {DEFAULT_DATASET_PATH}",
    )

    parser.add_argument(
        "--provider",
        "-p",
        default=DEFAULT_MODEL_PROVIDER,
        help=f"LLM provider. default: {DEFAULT_MODEL_PROVIDER}",
    )

    parser.add_argument(
        "--results-dir",
        "-r",
        default=DEFAULT_RESULTS_DIR,
        help=f"Directory where aggregated result JSONs are stored. default: {DEFAULT_RESULTS_DIR}",
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional limit on number of samples to run.",
    )

    parser.add_argument(
        "--max-reruns",
        type=int,
        default=DEFAULT_MAX_RERUNS,
        help=f"Maximum reruns per sample if output is invalid. default: {DEFAULT_MAX_RERUNS}",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    run_for_languages_and_systems(
        languages=args.languages,
        systems=args.systems,
        dataset_path=args.dataset,
        provider=args.provider,
        results_dir=args.results_dir,
        max_samples_to_run=args.max_samples,
        max_retries=args.max_reruns,
    )


if __name__ == "__main__":
    main()
