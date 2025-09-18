__author__='chinonsocynthiaosuji'

"""
Author: Chinonso Cynthia Osuji
Date: 10/07/2025
Description:
    This loads datasets for the Data-to-text-Agent project. 

    The Dataset loader extended to use the *grouped* WebNLG reference file.

    It adds a new dataset key `webnlg_grouped` that reads the JSON produced by your
    `fa_test_grouped.json` utility.  `extract_example()` is amended to expose the
    full `references` list so the evaluator can exploit every variant.
"""

from pathlib import Path
from typing import Dict, List, Callable

from datasets import load_dataset

# ---------------------------------------------------------------------------
# Constants – adjust path if you move the grouped file
# ---------------------------------------------------------------------------
_GROUPED_WEBNLG_PATH = Path("/home/chinonso/PHD_PROJECTS/Data-to-text-Agent/results/fa_test_grouped.json")

# ---------------------------------------------------------------------------
# Loader map
# ---------------------------------------------------------------------------

def get_dataset_loader() -> Dict[str, Callable]:
    """Return a mapping from dataset name → lazy loader function."""
    return {
        "rotowire": lambda: load_dataset("mrm8488/rotowire-sbnation", trust_remote_code=True),
        "turku_hockey": lambda: load_dataset("GEM/turku_hockey_data2text", trust_remote_code=True),
        "totto": lambda: load_dataset("GEM/totto", trust_remote_code=True),
        "sportsett_basketball": lambda: load_dataset("GEM/sportsett_basketball", trust_remote_code=True),
        "webnlg_hf": lambda: load_dataset("GEM/web_nlg", "en", trust_remote_code=True),
        "webnlg": lambda: load_dataset("json", data_files={"test": str(_GROUPED_WEBNLG_PATH)}, trust_remote_code=True),
        "conversational_weather": lambda: load_dataset("GEM/conversational_weather", trust_remote_code=True),
        "dart": lambda: load_dataset("GEM/dart", trust_remote_code=True),
        "mlb": lambda: load_dataset("GEM/mlb_data_to_text", trust_remote_code=True),
    }


def load_dataset_by_name(name: str):
    loaders = get_dataset_loader()
    if name not in loaders:
        raise ValueError(f"Dataset '{name}' is not supported. Choose from: {list(loaders.keys())}")
    print(f"Loading dataset: {name}")
    return loaders[name]()


# ---------------------------------------------------------------------------
# Normalise examples so evaluator can use them uniformly
# ---------------------------------------------------------------------------

def extract_example(dataset_name: str, example: Dict, index: int = None) -> Dict:
    """Return a dict with keys: input, target, references (if available)."""

    if dataset_name == "rotowire":
        return {
            "input": example.get("box_score", ""),
            "target": " ".join(example.get("summary", [])),
        }

    elif dataset_name in ["totto", "sportsett_basketball", "mlb"]:
        return {
            "input": example.get("linearized_input", ""),
            "target": example.get("target", ""),
            "references": example.get("references", []),
        }

    # ---------------------------------------------------------------------
    # WebNLG original GEM version (single reference per example)
    # ---------------------------------------------------------------------

    elif dataset_name == "webnlg_hf":
        # Use GEM's input/target, use our grouped references
        return {
            "input": example.get("input", ""),
            "target": example.get("target", ""),
            "references": example.get("references", ""),
        }

    # ---------------------------------------------------------------------
    # NEW: our grouped WebNLG file with multiple references
    # ---------------------------------------------------------------------
    elif dataset_name == "webnlg":
        return {
            # Original triples list → linear string like GEM baseline
            "input": ", ".join(example.get("triples", [])),
            "target": example.get("references", [])[0],
            "references": example.get("references", []),
        }

    elif dataset_name == "conversational_weather":
        return {
            "input": example.get("tree_str_mr", example.get("user_query", "")),
            "target": example.get("target", ""),
            "references": example.get("references", []),
        }

    elif dataset_name == "dart":
        return {
            "input": str(example.get("tripleset", "")),
            "target": example.get("target", ""),
            "references": example.get("references", []),
        }

    elif dataset_name == "turku_hockey":
        return {
            "input": example.get("input", ""),
            "target": example.get("target", ""),
        }

    # Fallback for any other dataset — no multi refs assumed
    return {
        "input": example.get("input", ""),
        "target": example.get("target", ""),
        "references": example.get("references", []),
    }


# ---------------------------------------------------------------------------
# Quick sanity test (optional)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ds_name = "webnlg_hf"
    data = load_dataset_by_name(ds_name)
    sample = extract_example(ds_name, data["test"][0])
    print(sample)
