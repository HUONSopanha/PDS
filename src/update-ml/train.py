import re
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
import torch
from sentence_transformers import SentenceTransformer, util
from itertools import product
import joblib

# === Helpers ===

def normalize_title(title: str) -> str:
    title = title.lower()
    title = re.sub(r'[^\w\s]', '', title)
    title = re.sub(r'\b(for|in|with|using|on|at|of)\b.*$', '', title)
    title = re.sub(r'\s+', ' ', title).strip()
    if title.endswith('s'):
        title = title[:-1]
    return title

def clean_job_title(title: str) -> str:
    title = re.sub(r'\b\d+\+?\s*(years|yrs|year)\b', '', title, flags=re.IGNORECASE)
    title = re.sub(r'\([^)]*\)', '', title)
    title = re.split(r'[|\-–]', title)[0]
    return title.strip()

# === Load Data ===

df = pd.read_csv('src/update-ml/titles.csv')
df.dropna(subset=['Title', 'Skills'], inplace=True)
df.drop_duplicates(subset=['Title', 'Skills'], inplace=True)
df['Title'] = df['Title'].astype(str)
df['Skills'] = df['Skills'].astype(str)

# Normalize titles for stratification
df['norm_title'] = df['Title'].apply(lambda x: normalize_title(clean_job_title(x)))

# === Split dataset ===
train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df['norm_title'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# === Oversample train ===
# Apply RandomOverSampler only on training data
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(train_df[['Skills']], train_df['norm_title'])

# Convert the resampled features and labels back to a DataFrame
train_df = pd.DataFrame({
    'Skills': X_resampled['Skills'].values,  # Extract 'Skills' column values as 1D array
    'norm_title': y_resampled
})

print(f"Train size: {len(train_df)} | Val size: {len(val_df)} | Test size: {len(test_df)}")

train_df = train_df[['Skills', 'norm_title']].copy()
val_df = val_df[['Skills', 'norm_title']].copy()
test_df = test_df[['Skills', 'norm_title']].copy()

def check_overlap(df1, df2):
    set1 = set(zip(df1['Skills'], df1['norm_title']))
    set2 = set(zip(df2['Skills'], df2['norm_title']))
    return len(set1 & set2)

print("Train-Val Overlap:", check_overlap(train_df, val_df))
print("Train-Test Overlap:", check_overlap(train_df, test_df))
print("Val-Test Overlap:", check_overlap(val_df, test_df))

# === Load Sentence-Transformer Models ===

model1 = SentenceTransformer("intfloat/e5-large-v2")  # E5
model2 = SentenceTransformer("BAAI/bge-large-en")    # BGE

# === Prepare job titles and embeddings ===

# Use unique normalized job titles from train_df for final training
train_titles = sorted(set(train_df['norm_title'].tolist()))
job_inputs_1 = [f"passage: {t}" for t in train_titles]
job_inputs_2 = train_titles

job_embeddings_1 = model1.encode(job_inputs_1, convert_to_tensor=True, batch_size=32)
job_embeddings_2 = model2.encode(job_inputs_2, convert_to_tensor=True, batch_size=32)

# === Prediction Function ===

def predict_job_titles_ensemble(user_skills, job_titles, job_emb_1, job_emb_2,
                                model1, model2, top_k=3, w1=0.5, w2=0.5):
    input_1 = f"query: {user_skills}"
    input_2 = user_skills

    user_emb1 = model1.encode(input_1, convert_to_tensor=True)
    user_emb2 = model2.encode(input_2, convert_to_tensor=True)

    scores1 = util.pytorch_cos_sim(user_emb1, job_emb_1)[0]
    scores2 = util.pytorch_cos_sim(user_emb2, job_emb_2)[0]

    combined_scores = w1 * scores1 + w2 * scores2
    top_indices = torch.topk(combined_scores, k=top_k).indices

    seen = set()
    results = []
    for idx in top_indices:
        title = job_titles[idx]
        if title not in seen:
            seen.add(title)
            results.append((title, combined_scores[idx].item()))
        if len(results) == top_k:
            break

    return results

# === Evaluation Function ===

def evaluate_ensemble_accuracy(eval_df, job_titles, job_emb_1, job_emb_2,
                               model1, model2, top_k=1, w1=0.5, w2=0.5):
    correct = 0
    total = len(eval_df)

    for _, row in eval_df.iterrows():
        user_skills = row['Skills']
        true_title = row['norm_title']

        preds = predict_job_titles_ensemble(user_skills, job_titles, job_emb_1, job_emb_2,
                                            model1, model2, top_k, w1, w2)
        pred_titles = [p[0] for p in preds]

        if true_title in pred_titles:
            correct += 1

    accuracy = correct / total if total > 0 else 0
    return accuracy

# === Hyperparameter Tuning ===

def tune_ensemble(val_df, job_titles, job_emb_1, job_emb_2, model1, model2, save_path='best_params.json'):
    best_score = 0
    best_params = {}

    print("🔧 Tuning ensemble weights and top_k...")
    for top_k, w1 in product([3], [0.2, 0.4, 0.6, 0.8]):
        w2 = 1.0 - w1
        acc = evaluate_ensemble_accuracy(val_df, job_titles, job_emb_1, job_emb_2,
                                         model1, model2, top_k=top_k, w1=w1, w2=w2)
        print(f"top_k={top_k}, w1={w1:.1f}, w2={w2:.1f} -> Accuracy={acc:.4f}")
        if acc > best_score:
            best_score = acc
            best_params = {'top_k': top_k, 'w1': w1, 'w2': w2}

    print(f"✅ Best Config: top_k={best_params['top_k']} | w1={best_params['w1']:.1f} | w2={best_params['w2']:.1f} | Accuracy={best_score:.4f}")

    with open(save_path, 'w') as f:
        json.dump(best_params, f)

    return best_params

# === Load best params helper ===

def load_best_params(path='best_params.json'):
    with open(path, 'r') as f:
        return json.load(f)

# === Final Evaluation ===

def final_eval(test_df, best_params):
    return evaluate_ensemble_accuracy(
        eval_df=test_df,
        job_titles=train_titles,
        job_emb_1=job_embeddings_1,
        job_emb_2=job_embeddings_2,
        model1=model1,
        model2=model2,
        top_k=best_params['top_k'],
        w1=best_params['w1'],
        w2=best_params['w2']
    )

# === Main Run ===

if __name__ == "__main__":
    best_params = tune_ensemble(val_df, train_titles, job_embeddings_1, job_embeddings_2, model1, model2)
    print("\n📊 Final Evaluation on Test Set")
    accuracy = final_eval(test_df, best_params)
    print(f"Test Accuracy: {accuracy:.4f}")

    # Save model data
    job_embeddings_1_cpu = job_embeddings_1.to('cpu')
    job_embeddings_2_cpu = job_embeddings_2.to('cpu')

    joblib.dump({
        "job_titles": train_titles,
        "job_embeddings_1": job_embeddings_1_cpu,
        "job_embeddings_2": job_embeddings_2_cpu,
        "best_params": best_params
    }, "/home/sopanha/Class/I3/Semister_2/PDS/Project/backend/src/ensemble_model_data.joblib")

    print("✅ Saved model to ensemble_model_data.joblib")
