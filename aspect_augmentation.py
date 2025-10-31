import argparse
import pandas as pd
import numpy as np
import json
from scipy.sparse import csr_matrix
from dotenv import load_dotenv
from tqdm import tqdm
import openai
import os
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.dataset import Dataset
from model.TPC import TPC
from sentence_transformers import SentenceTransformer

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Aspect Augmentation Configuration")
    parser.add_argument('--dataset_name', type=str, default='amazon', help='Name of the dataset')
    parser.add_argument('--mode', type=str, choices=['train', 'val', 'test'], help='Mode of operation: train, val or test')
    parser.add_argument('--num_cluster', type=int, default=50, help='Number of clusters for grouping users/items')
    return parser.parse_args()

def clustering(dataset: Dataset, num_cluster, col, emb_model):
    """Create a sparse matrix of interactions between users/items and aspects, and group users/items based on their interactions."""

    # Explode the list of aspect_ids to multiple rows
    df_exploded = dataset.trn_csv.explode('aspect_id')
    
    # Now count interactions per (uid, aspect_id)
    interaction_counts = df_exploded.groupby([col, 'aspect_id']).size().reset_index(name='count')
    
    # Map uids to zero-based indices
    uid_mapping = {uid: idx for idx, uid in enumerate(interaction_counts[col].unique())}
    idx_to_uid = {uid_mapping[uid]: uid for uid in uid_mapping}
    
    # Prepare lists for the sparse matrix
    uids, aspects, weights = [], [], []
    
    for _, row in interaction_counts.iterrows():
        uids.append(uid_mapping[row[col]])
        aspects.append(row['aspect_id'])
        weights.append(row['count'])
    
    # Number of users and aspects
    n_users = len(uid_mapping)
    n_aspects = df_exploded['aspect_id'].max() + 1  # assuming aspect ids are zero-based integers
    
    # Create csr_matrix with counts as weights
    csr = csr_matrix((weights, (uids, aspects)), shape=(n_users, n_aspects))

    uid_to_user_profile = dataset.trn_csv.drop_duplicates(col).set_index(col)['user_summary'].to_dict()
    uid_to_user_profile = {uid_mapping[idx]: uid_to_user_profile[idx] for idx in uid_to_user_profile}

    sorted_uids = sorted(uid_to_user_profile.keys())
    sorted_user_profiles = [uid_to_user_profile[uid] for uid in sorted_uids]
    embs = emb_model.encode(sorted_user_profiles)

    labels = TPC(csr, embs, alpha=0.5, dim=64, k=num_cluster, gamma=1, tf=5, tg=20)
    counts = np.bincount(labels)
    for number, count in enumerate(counts):
        print(f"Number {number} appears {count} times")

    group_dict = {num: [idx_to_uid[idx] for idx in np.where(labels == num)[0].tolist()] for num in np.unique(labels)}
    
    idx_to_group = {idx: group for group in group_dict for idx in group_dict[group]}
    return group_dict, idx_to_group

def data_augment(row_idx, aspect_name, csv, dataset: Dataset, group_user, user_to_group, group_item, item_to_group):
    """Augment data by extracting aspects from other relevant user-item pairs."""

    if len(aspect_name) == 0: return {}
    row = csv.iloc[row_idx]
    uid = row['uid']
    iid = row['iid']
    group_uid = user_to_group[uid]
    group_iid = item_to_group[iid]
    user_same_group = group_user[group_uid]
    item_same_group = group_item[group_iid]
    trn = dataset.trn_csv
    filtered_trn = trn[
        (trn['uid'].isin(user_same_group)) &
        (trn['iid'].isin(item_same_group))
    ]

    detail_aspects = {}

    for _, row in filtered_trn.iterrows():
        asps = json.loads(row['aspect'])
        list_asps = {k: v for k, v in asps.items() if k in aspect_name and type(v) == str}
        for key, value in list_asps.items():
            detail_aspects.setdefault(key, set()).add(value)
    return str({key: list(value) for key, value in detail_aspects.items()})

def process_augment_row(i, row, prompt, trn_csv):
    aspect_name = row['aspect_pred']
    aspect_name = ast.literal_eval(aspect_name) if isinstance(aspect_name, str) else aspect_name

    if not aspect_name:
        return i, ""

    augmented_details = data_augment(i, aspect_name, trn_csv)

    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": augmented_details}
            ],
            temperature=0,
            max_tokens=256
        )
        answer = response.choices[0].message.content.strip()
        return i, answer
    except Exception as e:
        print(f"ERROR at row {i}: {e}")
        return i, ""

if __name__ == "__main__":
    
    # Load environment variables
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    args = parse_args()

    dataset = Dataset(dataset_root="data", dataset_name=args.dataset_name)
    trn_csv = pd.read_csv(f"data/{args.dataset_name}/aspect_prediction_train.csv")

    # Classify users and items into groups based on and their interactions with aspects and profile embeddings 
    emb_model = SentenceTransformer('all-MiniLM-L6-v2')
    group_user, user_to_group = clustering(dataset, num_cluster=args.num_cluster, col='uid', emb_model=emb_model)
    group_item, item_to_group = clustering(dataset, num_cluster=args.num_cluster, col='iid', emb_model=emb_model)

    if args.mode == 'train':
        csv = trn_csv
    elif args.mode == 'val':
        csv = pd.read_csv(f"data/{args.dataset_name}/aspect_prediction_val.csv")
    elif args.mode == 'test':
        csv = pd.read_csv(f"data/{args.dataset_name}/aspect_prediction_test.csv")
    else:
        raise ValueError("Invalid mode. Choose either 'train', 'val' or 'test'.")
    
    with open(f'prompt/{args.dataset_name}/aspect_augmentation.txt', 'r', encoding='utf-8') as f:
        prompt = f.read()

    augment_summary = [""] * len(csv)

    with ThreadPoolExecutor(max_workers=64) as executor:
        futures = [
            executor.submit(process_augment_row, i, row, prompt, trn_csv)
            for i, row in csv.iterrows()
        ]

        for future in tqdm(as_completed(futures), total=len(futures)):
            i, result = future.result()
            augment_summary[i] = result

    csv['augment_summary'] = augment_summary
    csv.to_csv(f'data/{args.dataset_name}/augment_{args.mode}.csv', index=False)