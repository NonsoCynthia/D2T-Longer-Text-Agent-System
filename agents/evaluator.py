__author__='chinonsocynthiaosuji'

"""
Author: Chinonso Cynthia Osuji
Date: 10/07/2025
Description:
    Evaluate the generated text using various metrics.
""" 

import nltk
import torch
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import SmoothingFunction
smoother = SmoothingFunction()
from nltk.translate.meteor_score import single_meteor_score
from rouge_score import rouge_scorer
import sacrebleu
import pyter
from comet import download_model, load_from_checkpoint
from bleurt_pytorch import BleurtForSequenceClassification, BleurtTokenizer
import evaluate

def ensure_list(x):
    return x if isinstance(x, list) else [x]

class BatchEvaluator:
    """Load metric models once and reuse them for every example."""

    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        comet_ckpt = download_model("Unbabel/wmt22-comet-da")
        self.comet = load_from_checkpoint(comet_ckpt).to(self.device)
        self.comet.eval()
        self.bleurt_tok = BleurtTokenizer.from_pretrained("lucadiliello/BLEURT-20-D12")
        self.bleurt = (
            BleurtForSequenceClassification.from_pretrained("lucadiliello/BLEURT-20-D12")
            .to(self.device)
        )
        self.bleurt.eval()
        self.bertscore = evaluate.load("bertscore")

    def score(self, references, prediction, sources=None) -> dict:
        """Return a dict with all metrics normalized to the 0‒1 interval. Multi-ref for BLEU, TER, chrF++."""
        references = ensure_list(references)
        sources = ensure_list(sources) if sources is not None else references

        # BLEU (supports multiple references)
        bleu = sacrebleu.sentence_bleu(
            prediction,
            references,
            smooth_method="exp",
            tokenize="intl",
            lowercase=False,
        ).score / 100.0

        # METEOR (single reference, already 0‑1)
        meteor = single_meteor_score(
            word_tokenize(references[0]), word_tokenize(prediction)
        )

        # ROUGE (first reference, F‑measure already 0‑1)
        rouge_scores = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        ).score(references[0], prediction)
        rouge_f1 = sum(r.fmeasure for r in rouge_scores.values()) / 3

        # COMET (uses first reference/source)
        inp = [{"src": sources[0], "mt": prediction, "ref": references[0]}]
        comet_score = self.comet.predict(inp, gpus=1 if self.device == "cuda" else 0)[0][0]

        # BLEURT (first reference)
        with torch.no_grad():
            bleurt_inputs = self.bleurt_tok(
                references[0],
                prediction,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)
            bleurt_score = self.bleurt(**bleurt_inputs).logits.flatten().item()

        # BERTScore (multi-ref aware)
        bert_f1 = self.bertscore.compute(
            predictions=[prediction], references=[references], lang="en"
        )["f1"][0]

        # TER (minimum TER over all references)
        ter_scores = [
            pyter.ter(prediction.split(), ref.split())
            for ref in references if ref.strip()
        ]
        ter_score = min(ter_scores) if ter_scores else 1.0

        # chrF++ (using sacrebleu, supports multi-ref)
        chrfpp = sacrebleu.metrics.CHRF(word_order=2, char_order=6, beta=2)
        chrf_score = chrfpp.sentence_score(prediction, references).score / 100.0

        return {
            "BLEU": bleu,
            "METEOR": meteor,
            "ROUGE-F1": rouge_f1,
            "COMET": comet_score,
            "BLEURT": bleurt_score,
            "BERTScore-F1": bert_f1,
            "TER": ter_score,
            "chrF++": chrf_score,
        }

def evaluate_single(references, prediction, sources=None):
    _default_evaluator = BatchEvaluator()
    return _default_evaluator.score(references, prediction, sources)

if __name__ == "__main__":
    references = [
        "Gauff, just 15, shocks 5-time champ Venus, 39, at Wimbledon",
        "15-year-old Gauff beats 5-time Wimbledon champion Venus Williams"
    ]
    prediction = "15-year-old Gauff beats 5-time Wimbledon champ Venus"
    source = ["Cori Gauff, 15, defeated Venus Williams at Wimbledon."]

    evaluator = BatchEvaluator()
    scores = evaluator.score(references, prediction, source)

    for metric, value in scores.items():
        print(f"{metric}: {value:.4f}")
