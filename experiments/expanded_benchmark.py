"""Expanded, hand-authored alignment stress test; not a full JEDEC benchmark.

Run from this repository root. Optional --deps/--cache enable the locally cached
FastEmbed model. No model service or document upload is used.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

from compare_baselines import block, build_cases, select, similarity
from evaluate_encoder import groups
from real_jedec_scope import OLD, NEW


def extra_cases():
    """Manual challenge cases. All technical statements below are invented."""
    rows = [
        ("refresh_paraphrase", "paraphrase", "The device shall complete refresh before entering sleep.", "Sleep entry requires refresh to be complete.", "Sleep"),
        ("cke_paraphrase", "paraphrase", "CKE shall remain LOW for two clock cycles.", "Keep CKE at a low level for two cycles of the clock.", "CKE"),
        ("voltage_paraphrase", "paraphrase", "VDDQ must be stable before initialization starts.", "Initialization may begin only after VDDQ has settled.", "Power"),
        ("mode_register_paraphrase", "paraphrase", "The controller shall write MR1 prior to issuing READ.", "A READ command requires a preceding MR1 write by the controller.", "MRW"),
        ("numeric_increase", "number", "The device shall wait at least 5 ns after RESET.", "The device shall wait at least 8 ns after RESET.", "Reset"),
        ("numeric_decrease", "number", "The maximum burst length is 16 beats.", "The maximum burst length is 8 beats.", "Burst"),
        ("range_lower", "number", "The supported density range is 4 Gb through 16 Gb.", "The supported density range is 2 Gb through 16 Gb.", "Density"),
        ("range_upper", "number", "The supported density range is 4 Gb through 16 Gb.", "The supported density range is 4 Gb through 32 Gb.", "Density"),
        ("not_added", "polarity", "The controller shall issue PRECHARGE before REFRESH.", "The controller shall not issue PRECHARGE before REFRESH.", "Refresh"),
        ("not_removed", "polarity", "The device shall not enter sleep during calibration.", "The device shall enter sleep during calibration.", "Sleep"),
        ("only_added", "condition", "The command is valid in Mode_A.", "The command is valid only in Mode_A.", "Mode"),
        ("unless_added", "condition", "The device shall ignore READ during refresh.", "The device shall ignore READ during refresh unless override is enabled.", "Refresh"),
        ("shall_to_should", "modality", "The controller shall check parity.", "The controller should check parity.", "Parity"),
        ("shall_to_may", "modality", "The controller shall retry after timeout.", "The controller may retry after timeout.", "Retry"),
        ("unit_1000ps", "equivalent", "The minimum delay is 1 ns.", "The minimum delay is 1000 ps.", "Timing"),
        ("symbol_alias", "equivalent", "The delay tRCD shall be at least 10 ns.", "The minimum tRCD delay is 10 ns.", "Timing"),
        ("sentence_reorder", "equivalent", "RESET shall finish before training begins.", "Training begins after RESET finishes.", "Training"),
        ("reference_renumber", "reference", "The requirement is listed in Table 7.", "The requirement is listed in Table 9.", "Timing tables"),
        ("reference_changed", "reference", "The requirement is defined in Table 7.", "The requirement is defined in Table 8.", "Timing tables"),
        ("object_swap", "object", "The controller shall hold CA LOW during sleep.", "The device shall hold CA LOW during sleep.", "Sleep"),
        ("subject_swap", "object", "MR1 controls the read latency.", "MR2 controls the read latency.", "Mode registers"),
        ("moved_section", "move", "The device shall support a burst length of 16.", "The device shall support a burst length of 16.", "5.3 Burst operation"),
        ("format_only", "format", "The controller shall wait 10 ns after RESET.", "The controller shall\n wait 10 ns after RESET.", "Reset"),
        ("list_to_sentence", "paraphrase", "Entry conditions: RESET complete; VDDQ stable.", "Enter only after RESET completes and VDDQ is stable.", "Entry"),
        ("zero_distractor", "deletion", "The device shall support Mode_Q.", None, "Mode_Q"),
    ]
    cases = []
    for name, category, old_text, new_text, title in rows:
        old_title = "4.2 Burst operation" if name == "moved_section" else title
        new = [block("other", "The device shall remain in standby.", "Standby")]
        if new_text is not None:
            new.append(block("target", new_text, title))
        cases.append({"name": "extra_" + name, "category": category,
                      "old": block("old", old_text, old_title), "new": new,
                      "gold_ids": ["target"] if new_text is not None else []})
    # Same wording in the wrong section is a deliberate false anchor.
    cases.append({"name": "extra_duplicate_template", "category": "duplicate",
                  "old": block("old", "Wait until calibration completes.", "Calibration"),
                  "new": [block("other", "Wait until calibration completes.", "Initialization"),
                          block("target", "Wait for calibration to finish.", "Calibration")],
                  "gold_ids": ["target"]})
    cases.append({"name": "extra_split_three", "category": "split",
                  "old": block("old", "VDDQ shall be stable. RESET shall be complete. Training shall be enabled.", "Entry"),
                  "new": [block("a", "VDDQ shall be stable.", "Entry"),
                          block("b", "RESET shall be complete.", "Entry"),
                          block("c", "Training shall be enabled.", "Entry")],
                  "gold_ids": ["a", "b", "c"]})
    cases.append({"name": "extra_split_two", "category": "split",
                  "old": block("old", "MR1 shall be written before READ. MR2 shall be written before WRITE.", "Mode registers"),
                  "new": [block("a", "MR1 shall be written before READ.", "Mode registers"),
                          block("b", "MR2 shall be written before WRITE.", "Mode registers")],
                  "gold_ids": ["a", "b"]})
    for case in cases:
        ids = [item["id"] for item in case["new"]]
        assert len(ids) == len(set(ids)) and set(case["gold_ids"]) <= set(ids)
    return cases


def all_cases():
    cases = [{**case, "category": "original_synthetic"} for case in build_cases()]
    cases += extra_cases()
    for old in OLD:
        cases.append({"name": "real_" + old["id"], "category": "real_scope",
                      "old": {**old, "title": "1 Scope"},
                      "new": [{**new, "title": "1 Scope"} for new in NEW],
                      "gold_ids": [old["id"].replace("D-", "E-")]})
    return cases


def rank(case, vectors=None, np=None):
    candidates = []
    for group in groups(case):
        text = " ".join(item["text"] for item in group)
        lex = similarity(case["old"]["text"], text)
        title = sum(similarity(case["old"]["title"], item["title"]) for item in group) / len(group)
        row = {"ids": [item["id"] for item in group],
               "lexical_score": round(.7 * lex + .3 * title - .02 * (len(group) - 1), 5)}
        if vectors is not None:
            a, b = vectors[case["old"]["text"]], vectors[text]
            row["encoder_cosine"] = round(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))), 5)
        candidates.append(row)
    return candidates


def run(use_encoder=False, deps=None, cache=None):
    cases = all_cases()
    vectors = np = None
    if use_encoder:
        sys.path.insert(0, str(deps.resolve()))
        import numpy as np
        from fastembed import TextEmbedding
        texts = list(dict.fromkeys(text for case in cases for text in
                   [case["old"]["text"], *(" ".join(x["text"] for x in g) for g in groups(case))]))
        model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5", cache_dir=str(cache.resolve()), providers=["CPUExecutionProvider"])
        vectors = dict(zip(texts, model.embed(texts)))
    details = []
    for case in cases:
        candidates = rank(case, vectors, np)
        lex = sorted(candidates, key=lambda x: x["lexical_score"], reverse=True)
        row = {"case": case["name"], "category": case["category"], "gold_ids": case["gold_ids"],
               "lexical_top1": lex[0], "lexical_selected": select(case, "lexical_context_groups")}
        if use_encoder:
            enc = sorted(candidates, key=lambda x: x["encoder_cosine"], reverse=True)
            row["encoder_top1"] = enc[0]
            row["encoder_top2_ids"] = [x["ids"] for x in enc[:2]]
        details.append(row)
    counts = defaultdict(lambda: {"cases": 0, "matched": 0, "lexical_top1_correct": 0,
                                  "lexical_accepted_correct": 0, "lexical_false_match_on_empty": 0,
                                  "encoder_top1_correct": 0, "encoder_top2_contains_gold": 0})
    for row in details:
        stats = counts[row["category"]]
        stats["cases"] += 1
        gold = set(row["gold_ids"])
        if gold:
            stats["matched"] += 1
            stats["lexical_top1_correct"] += set(row["lexical_top1"]["ids"]) == gold
            stats["lexical_accepted_correct"] += set(row["lexical_selected"]) == gold
            if use_encoder:
                stats["encoder_top1_correct"] += set(row["encoder_top1"]["ids"]) == gold
                stats["encoder_top2_contains_gold"] += any(set(x) == gold for x in row["encoder_top2_ids"])
        else:
            stats["lexical_false_match_on_empty"] += bool(row["lexical_selected"])
    return {"scope": "15 original synthetic + 28 new synthetic + 2 correlated real Scope pairs; hand-authored tiny candidate pools",
            "full_document": False, "encoder_run": use_encoder, "case_count": len(details),
            "by_category": dict(counts), "details": details}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--deps", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if bool(args.deps) != bool(args.cache):
        parser.error("--deps and --cache must be supplied together")
    report = run(bool(args.deps), args.deps, args.cache)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "details"}, ensure_ascii=False, indent=2))
