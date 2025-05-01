import pandas as pd
import os
import sys
sys.path.append(".")
import time
from src.chatbot import ChatBot
from src.db import VectorDB
import matplotlib.pyplot as plt
import seaborn as sns
import evaluate # from https://huggingface.co/spaces/evaluate-metric/bertscore

from transformers import AutoTokenizer

# Metric computation functions

def compute_bert_score(generated, reference, bertscore_metric):
    # Loads and computes BERTScore between generated and ground truth (reference) answers
    
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
df = pd.read_csv("scripts/evaluation/Ground_Truth.csv", encoding = "utf-8", delimiter=";")   

chatbot = ChatBot()
vector_db = VectorDB()
bertscore_metric = evaluate.load("bertscore")

tokenizer = AutoTokenizer.from_pretrained(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    token=os.getenv("HUGGINGFACE_TOKEN", None)
)

output_path = "scripts/evaluation/deepseek/"
just_time_measurement = False

# Evaluate each row 
results = []
tks_results = []
for idx, row in df.iterrows():
    print(f"Processing query nº {idx+1}/{len(df)}...")
    query = row["query"]
    gt_answer = row["ground_truth_answer"]
    gt_doc_id = row["ground_truth_doc_id"]

    # Retrieve context and sources
    _, sources = chatbot.retrieve_context_from_db(query, vector_db, k=3)
    retrieved_doc_ids = [src.get("source", "").split(".")[0] for src in sources]

    # Generate response
    start = time.time()
    response, _ = chatbot.infer(query)
    response_time = time.time() - start
    # We tokenize manually the response
    tokens = tokenizer.tokenize(response, return_tensors="pt")
    tok_per_sec = len(tokens) / response_time
    
    # Reset memory
    chatbot.memory.reset_memory()

    if not just_time_measurement:
        # Compute metrics
        recall_k = compute_recall_at_k(gt_doc_id, retrieved_doc_ids, k=3)
        mrr = compute_mrr(gt_doc_id, retrieved_doc_ids)
        bert_p, bert_r, bert_f1 = compute_bert_score(response, gt_answer, bertscore_metric=bertscore_metric)

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
        tks_results.append({
            "Tokens_per_second": tok_per_sec
        })
    else:
        tks_results.append({
            "Tokens_per_second": tok_per_sec
        })

    print(f"Succesfully processed query.")
    if idx == 15:
        break

# Save results 
if len(results) > 0:
    results_df = pd.DataFrame(results)
tks_results_df = pd.DataFrame(tks_results)

os.makedirs(output_path, exist_ok=True)
if len(results) > 0:
    results_df.to_csv(output_path+"evaluation_results.csv", index=False)
tks_results_df.to_csv(output_path+"tks_evaluation_results.csv", index=False)
print("Evaluation complete. Results saved to 'evaluation_results.csv'")

