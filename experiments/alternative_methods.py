"""Additional candidate-ranking baselines on the fixed 45-case diagnostic.

All implementations are local, dependency-free except optional FastEmbed for
the encoder and its RRF fusion. This is not a full-document diff engine.
"""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re
import sys

from compare_baselines import select, similarity, tokens
from evaluate_encoder import groups
from expanded_benchmark import all_cases


def words(text):
    return [item.lower() for item in tokens(text) if re.search(r"[A-Za-z0-9]", item)]


def chars(text):
    text = re.sub(r"\s+", " ", text.lower()).strip()
    return Counter(text[i:i+n] for n in (3, 4, 5) for i in range(max(0, len(text)-n+1)))


def candidate_items(cases):
    return [[{"ids": [b["id"] for b in group],
              "text": " ".join(b["text"] for b in group),
              "title": " ".join(b["title"] for b in group)}
             for group in groups(case)] for case in cases]


def idf_table(documents):
    counts = Counter(term for doc in documents for term in set(doc))
    n = len(documents)
    return {term: math.log((n+1)/(df+1)) + 1 for term, df in counts.items()}


def tfidf_cosine(query, candidate, idf):
    def weighted(terms):
        return {term: (1 + math.log(tf)) * idf.get(term, 1.0)
                for term, tf in terms.items()}
    a, b = weighted(query), weighted(candidate)
    numerator = sum(value*b.get(term, 0) for term, value in a.items())
    denom = math.sqrt(sum(x*x for x in a.values()) * sum(x*x for x in b.values()))
    return numerator/denom if denom else 0.0


def bm25(query, candidate, doc_freq, n, avg_len, k1=1.2, b=0.75):
    terms = Counter(candidate)
    length = len(candidate)
    result = 0.0
    for term in set(query):
        freq = terms[term]
        if not freq:
            continue
        df = doc_freq.get(term, 0)
        idf = math.log(1 + (n-df+0.5)/(df+0.5))
        result += idf * freq*(k1+1)/(freq+k1*(1-b+b*length/avg_len))
    return result


def weighted_jaccard(query, candidate, word_idf):
    left, right = set(query), set(candidate)
    union = left | right
    if not union:
        return 0.0
    return sum(word_idf.get(term, 1.0) for term in left & right) / sum(
        word_idf.get(term, 1.0) for term in union)


def reciprocal_rank_fusion(rankings, k=60):
    return {index: sum(1/(k + ranking.index(index) + 1) for ranking in rankings)
            for index in rankings[0]}


def rank_indices(scores):
    return sorted(range(len(scores)), key=lambda i: (-scores[i], i))


def calculate(cases, use_encoder=False, deps=None, cache=None):
    pools = candidate_items(cases)
    corpus = [candidate["text"] for pool in pools for candidate in pool]
    word_docs = [words(text) for text in corpus]
    char_docs = [chars(text) for text in corpus]
    word_df = Counter(term for doc in word_docs for term in set(doc))
    word_idf = idf_table(word_docs)
    char_idf = idf_table(char_docs)
    avg_len = sum(map(len, word_docs)) / len(word_docs)
    vectors = np = None
    if use_encoder:
        sys.path.insert(0, str(deps.resolve()))
        import numpy as np
        from fastembed import TextEmbedding
        texts = list(dict.fromkeys(corpus + [case["old"]["text"] for case in cases]))
        model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5",
                              cache_dir=str(cache.resolve()), providers=["CPUExecutionProvider"])
        vectors = dict(zip(texts, model.embed(texts)))

    results = []
    for case, pool in zip(cases, pools):
        old = case["old"]
        wq, cq = words(old["text"]), chars(old["text"])
        scores = defaultdict(list)
        for item in pool:
            scores["lexical"].append(similarity(old["text"], item["text"]))
            title = similarity(old["title"], item["title"])
            scores["lexical_title_group"].append(.7*scores["lexical"][-1]+.3*title-.02*(len(item["ids"])-1))
            scores["bm25"].append(bm25(wq, words(item["text"]), word_df, len(word_docs), avg_len))
            scores["weighted_jaccard"].append(weighted_jaccard(wq, words(item["text"]), word_idf))
            scores["char_tfidf"].append(tfidf_cosine(cq, chars(item["text"]), char_idf))
            if use_encoder:
                a, b = vectors[old["text"]], vectors[item["text"]]
                scores["encoder"].append(float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))))
        rankings = {method: rank_indices(values) for method, values in scores.items()}
        base = [rankings[name] for name in ("lexical_title_group", "bm25", "char_tfidf")]
        fused = reciprocal_rank_fusion(base)
        rankings["rrf_three"] = rank_indices([fused[i] for i in range(len(pool))])
        if use_encoder:
            fused = reciprocal_rank_fusion(base + [rankings["encoder"]])
            rankings["rrf_with_encoder"] = rank_indices([fused[i] for i in range(len(pool))])
        predictions = {method: pool[order[0]]["ids"] for method, order in rankings.items()}
        top2 = {method: [pool[i]["ids"] for i in order[:2]] for method, order in rankings.items()}
        results.append({"case": case["name"], "category": case["category"],
                        "gold_ids": case["gold_ids"],
                        "lexical_accepted": select(case, "lexical_context_groups"),
                        "predictions_top1": predictions, "predictions_top2": top2,
                        "scores": {method: [{"ids": item["ids"], "score": round(values[i], 5)}
                                            for i, item in enumerate(pool)] for method, values in scores.items()}})
    return results


def summarize(rows, use_encoder):
    methods = list(rows[0]["predictions_top1"])
    positive = [row for row in rows if row["gold_ids"]]
    by_method = {}
    for method in methods:
        by_method[method] = {
            "top1_exact": sum(set(row["predictions_top1"][method]) == set(row["gold_ids"])
                              for row in positive),
            "top2_contains_gold": sum(any(set(x) == set(row["gold_ids"])
                                       for x in row["predictions_top2"][method])
                                      for row in positive),
            "per_category": {category: {"correct": sum(set(row["predictions_top1"][method]) == set(row["gold_ids"])
                                                     for row in positive if row["category"] == category),
                                        "total": sum(row["category"] == category for row in positive)}
                             for category in sorted({row["category"] for row in positive})}}
    return {"case_count": len(rows), "positive_alignment_cases": len(positive),
            "empty_gold_cases": len(rows)-len(positive), "encoder_run": use_encoder,
            "evaluation_unit": "top-1 and top-2 candidate-group ranking in hand-authored pools; no automatic deletion decision",
            "methods": by_method, "details": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--deps", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if bool(args.deps) != bool(args.cache):
        parser.error("--deps and --cache must be supplied together")
    rows = calculate(all_cases(), bool(args.deps), args.deps, args.cache)
    report = summarize(rows, bool(args.deps))
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k != "details"}, ensure_ascii=False, indent=2))
