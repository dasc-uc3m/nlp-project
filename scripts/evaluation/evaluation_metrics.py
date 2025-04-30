import pandas as pd
from chatbot import ChatBot
from src.db import VectorDB
import matplotlib.pyplot as plt
import seaborn as sns
import evaluate # from https://huggingface.co/spaces/evaluate-metric/bertscore

# Metric computation functions

def compute_bert_score(generated, reference):
    # Loads and computes BERTScore between generated and ground truth (reference) answers
    bertscore_metric = evaluate.load("bertscore")
    results = bertscore_metric.compute(predictions=[generated],
                                       references=[reference],
                                       lang="en")
    return float(results["precision"][0]), float(results["recall"][0]), float(results["f1"][0])

def compute_recall_at_k(gt_doc_id, retrieved_doc_ids, k):
    # Checks whether the ground truth document name is among the top-k retrieved
    return int(gt_doc_id in retrieved_doc_ids[:k])

def compute_mrr(gt_doc_id, retrieved_doc_ids):
    # Computes Mean Reciprocal Rank (MRR) for the ground truth document
    for i, doc_id in enumerate(retrieved_doc_ids):
        if doc_id == gt_doc_id:
            return 1.0 / (i + 1)
    return 0.0

# Load the ground truth (GT) that contains queries, GT answers and GT document name.
df = pd.read_csv("Ground_Truth.csv")   

chatbot = ChatBot()
vector_db = VectorDB()

# Evaluate each row 
results = []
for idx, row in df.iterrows():
    query = row["query"]
    gt_answer = row["ground_truth_answer"]
    gt_doc_id = row["ground_truth_doc_id"]

    # Retrieve context and sources
    _, sources = chatbot.retrieve_context_from_db(query, vector_db, k=3)
    retrieved_doc_ids = [src.get("source", "").split(".")[0] for src in sources]

    # Generate response
    response, _ = chatbot.infer(query)

    # Compute metrics
    recall_k = compute_recall_at_k(gt_doc_id, retrieved_doc_ids, k=3)
    mrr = compute_mrr(gt_doc_id, retrieved_doc_ids)
    bert_p, bert_r, bert_f1 = compute_bert_score(response, gt_answer)

    # Save
    results.append({
        "query": query,
        "generated_answer": response,
        "ground_truth_answer": gt_answer,
        "ground_truth_doc_id": gt_doc_id,
        "retrieved_doc_ids": retrieved_doc_ids,
        "Recall@3": recall_k,
        "MRR": mrr,
        "BERTScore_P": bert_p,
        "BERTScore_R": bert_r,
        "BERTScore_F1": bert_f1
    })

# Save results 
results_df = pd.DataFrame(results)
results_df.to_csv("evaluation_results.csv", index=False)
print("Evaluation complete. Results saved to 'evaluation_results.csv'")

