import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import json
import warnings
import traceback
import torch
import joblib
from sentence_transformers import SentenceTransformer, util

warnings.filterwarnings("ignore")

def normalize_title(title: str) -> str:
    return title.lower().strip()

def clean_job_title(title: str) -> str:
    return title.replace("-", " ").strip()

def load_model_data(path="/home/sopanha/Class/I3/Semister_2/PDS/Project/backend/src/ensemble_model_data.joblib"):
    try:
        return joblib.load(path)
    except Exception as e:
        raise RuntimeError(f"Failed to load model data: {str(e)}")

def load_models():
    try:
        model1 = SentenceTransformer("intfloat/e5-large-v2")
        model2 = SentenceTransformer("BAAI/bge-large-en")
        return model1, model2
    except Exception as e:
        raise RuntimeError(f"Model loading failed: {str(e)}")

def predict_ensemble(user_skills, model1, model2, job_titles, job_emb_1, job_emb_2, best_params):
    input_1 = f"query: {user_skills}"
    input_2 = user_skills

    user_emb1 = model1.encode(input_1, convert_to_tensor=True)
    user_emb2 = model2.encode(input_2, convert_to_tensor=True)

    scores1 = util.pytorch_cos_sim(user_emb1, job_emb_1)[0]
    scores2 = util.pytorch_cos_sim(user_emb2, job_emb_2)[0]

    w1 = best_params.get("w1", 0.5)
    w2 = best_params.get("w2", 0.5)
    top_k = best_params.get("top_k", 3)

    combined_scores = w1 * scores1 + w2 * scores2
    top_indices = torch.topk(combined_scores, k=top_k).indices

    seen = set()
    results = []
    for idx in top_indices:
        title = job_titles[idx]
        norm = normalize_title(clean_job_title(title))
        if norm not in seen:
            seen.add(norm)
            results.append({
                "title": title,
                "score": round(combined_scores[idx].item(), 4)
            })
        if len(results) == top_k:
            break
    return results

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No input provided"}))
        sys.exit(1)

    user_input = sys.argv[1]

    try:
        data = load_model_data()
        model1, model2 = load_models()

        job_emb_1 = data["job_embeddings_1"]
        job_emb_2 = data["job_embeddings_2"]

        if not torch.is_tensor(job_emb_1):
            job_emb_1 = torch.tensor(job_emb_1)
        if not torch.is_tensor(job_emb_2):
            job_emb_2 = torch.tensor(job_emb_2)

        predictions = predict_ensemble(
            user_input,
            model1,
            model2,
            data["job_titles"],
            job_emb_1,
            job_emb_2,
            data["best_params"]
        )
        print(json.dumps(predictions))
        sys.exit(0)

    except Exception as e:
        error_details = {
            "error": "Unhandled Python Exception",
            "message": str(e),
            "traceback": traceback.format_exc()
        }
        print(json.dumps(error_details), file=sys.stderr)  # 🔥 send to stderr
        sys.exit(1)


if __name__ == "__main__":
    main()
