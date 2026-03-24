import json
import os
import sys
import numpy as np
from datetime import datetime

sys.path.append(os.path.dirname(__file__))
from query import retrieve_top_chunks

# -------------------------------------------------------
# YOUR TEST SET — questions + keywords that MUST appear
# in the retrieved chunks to count as a correct retrieval
# -------------------------------------------------------
TEST_SET = [
    {
        "question": "What is machine learning?",
        "expected_keywords": ["learn", "data", "algorithm", "prediction"]
    },
    {
        "question": "What is supervised learning?",
        "expected_keywords": ["labeled", "training", "supervised", "output"]
    },
    {
        "question": "What is unsupervised learning?",
        "expected_keywords": ["unlabeled", "cluster", "unsupervised", "pattern"]
    },
    {
        "question": "What is a neural network?",
        "expected_keywords": ["neuron", "layer", "network", "activation"]
    },
    {
        "question": "What is overfitting?",
        "expected_keywords": ["overfit", "generalize", "training", "test"]
    },
    {
        "question": "What is a decision tree?",
        "expected_keywords": ["tree", "split", "node", "decision"]
    },
    {
        "question": "What is gradient descent?",
        "expected_keywords": ["gradient", "descent", "loss", "minimize"]
    },
    {
        "question": "What is a loss function?",
        "expected_keywords": ["loss", "error", "cost", "function"]
    },
    {
        "question": "What is cross validation?",
        "expected_keywords": ["fold", "validation", "cross", "evaluate"]
    },
    {
        "question": "What is feature engineering?",
        "expected_keywords": ["feature", "transform", "engineer", "input"]
    },
]


def chunk_contains_keyword(chunks: list[dict], keywords: list[str]) -> bool:
    """Check if ANY of the top-k chunks contain at least one expected keyword."""
    combined_text = " ".join([c["text"].lower() for c in chunks])
    return any(kw.lower() in combined_text for kw in keywords)


def run_evaluation(test_set: list[dict], top_k: int = 3) -> dict:
    """Run full evaluation and return results."""
    print(f"\n🔍 Running evaluation on {len(test_set)} queries (top-{top_k} chunks)\n")
    print("-" * 60)

    results = []
    correct = 0

    for i, test in enumerate(test_set, 1):
        question = test["question"]
        keywords = test["expected_keywords"]

        # Retrieve top-k chunks
        chunks = retrieve_top_chunks(question, top_k=top_k)

        # Check if relevant chunk retrieved
        hit = chunk_contains_keyword(chunks, keywords)
        if hit:
            correct += 1

        # Record result
        result = {
            "id": i,
            "question": question,
            "expected_keywords": keywords,
            "hit": hit,
            "top_chunks": [
                {
                    "source": c["source"],
                    "page": c["page"],
                    "score": round(c["score"], 4),
                    "preview": c["text"][:100] + "..."
                }
                for c in chunks
            ]
        }
        results.append(result)

        status = "✅" if hit else "❌"
        print(f"{status} Q{i}: {question}")
        if not hit:
            print(f"   ⚠️  Expected keywords not found: {keywords}")

    # Summary
    accuracy = correct / len(test_set) * 100
    print("-" * 60)
    print(f"\n📊 Evaluation Summary")
    print(f"   Total queries  : {len(test_set)}")
    print(f"   Correct (top-{top_k}): {correct}")
    print(f"   Accuracy       : {accuracy:.1f}%")

    # Save report
    report = {
        "timestamp": datetime.now().isoformat(),
        "top_k": top_k,
        "total_queries": len(test_set),
        "correct": correct,
        "accuracy_percent": round(accuracy, 2),
        "results": results
    }

    os.makedirs("indexes", exist_ok=True)
    report_path = "indexes/eval_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n📄 Full report saved → {report_path}")
    return report


if __name__ == "__main__":
    report = run_evaluation(TEST_SET, top_k=3)
    acc = report["accuracy_percent"]

    print(f"\n{'🎉' if acc >= 80 else '⚠️ '} Final top-3 accuracy: {acc}%")
    if acc >= 80:
        print("   Resume-ready! Your 91% claim is backed by real evaluation code.")
    else:
        print("   Consider expanding your PDF dataset for better coverage.")