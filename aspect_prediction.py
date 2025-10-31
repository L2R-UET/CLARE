from tqdm import tqdm
import argparse
import json
import ast
import torch
import pickle
import openai
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import torch.nn.functional as F
from model.HHGNN import HHGNN
from utils.dataset import Dataset


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Aspect Prediction Configuration")
    parser.add_argument('--dataset_name', type=str, default='amazon', help='Name of the dataset')
    parser.add_argument('--num_epochs', type=int, default=2000, help='Number of training epochs')
    parser.add_argument('--top_k', type=int, default=10, help='Number of top aspects to predict')
    parser.add_argument('--mode', type=str, choices=['train', 'val', 'test'], help='Mode of operation: train, val or test')
    return parser.parse_args()

def top_k_aspect(row_idx, k, csv, node_emb, aspect_emb, dataset: Dataset):
    row = csv.iloc[row_idx]
    if row['uid'] not in dataset.user_to_index or row['iid'] not in dataset.item_to_index:
        return []
    uid = dataset.user_to_index[row['uid']] + dataset.user_offset
    iid = dataset.item_to_index[row['iid']] + dataset.item_offset
    user_emb = node_emb[uid]
    item_emb = node_emb[iid]
    ui_emb = (user_emb + item_emb).unsqueeze(0).expand(aspect_emb.size(0), -1)
    scores = F.cosine_similarity(ui_emb, aspect_emb, dim=1)
    top_k = torch.topk(scores, k).indices
    aspect_name = [dataset.index_to_aspect[idx.item()] for idx in top_k]
    return aspect_name

def extract_details(row_idx, aspect_name, csv, dataset: Dataset):
    if len(aspect_name) == 0: return {}
    row = csv.iloc[row_idx]
    trn = dataset.trn_csv
    trn_uid = trn[trn['uid'] == row['uid'] & trn['iid'] != row['iid']]
    trn_iid = trn[trn['iid'] == row['iid'] & trn['uid'] != row['uid']]

    detail_aspects = {}

    for df in [trn_uid, trn_iid]:
        for _, row in df.iterrows():
            asps = json.loads(row['aspect'])
            list_asps = {k: v for k, v in asps.items() if k in aspect_name and type(v) == str}
            for key, value in list_asps.items():
                detail_aspects.setdefault(key, set()).add(value)
    return str({key: list(value) for key, value in detail_aspects.items()})

def process_row(i, row, prompt, dataset):
    aspect_name = row['aspect_pred']
    aspect_name = ast.literal_eval(aspect_name) if isinstance(aspect_name, str) else aspect_name

    if len(aspect_name) == 0:
        return i, ""

    aspect_details = extract_details(i, aspect_name, new_csv, dataset)

    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": aspect_details}
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

    dataset = Dataset(dataset_root='data', dataset_name=args.dataset_name)

    node_emb_path = f'data/{args.dataset_name}/node_emb.pkl'
    if os.path.exists(node_emb_path):
        with open(node_emb_path, 'rb') as f:
            node_emb = pickle.load(f)
        node_emb = node_emb.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    else:
        model = HHGNN(args, dataset=dataset, nfeat=64, nhid=64, out_dim=64, out_nhead=3, nhead=3, node_input_dim=[64])
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Training the HHGNN model for aspect prediction
        all_loss = []
        pbar = tqdm(range(args.num_epochs))
        for epoch in pbar:
            model.train()
            loss = model()
            all_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward(retain_graph=False)
            optimizer.step()

            pbar.set_postfix(loss=loss.item())

        # Save the embeddings
        model.eval()
        with torch.no_grad():
            node_emb = model.evaluate()
            with open(f'data/{args.dataset_name}/node_emb.pkl', 'wb') as f:
                pickle.dump(node_emb.cpu(), f)

    
    aspect_emb = node_emb[:dataset.n_aspect]
    if args.mode == 'train':
        csv = dataset.trn_csv
    elif args.mode == 'test':
        csv = dataset.tst_csv
    elif args.mode == 'val':
        csv = dataset.val_csv
    else:
        raise ValueError("Mode must be either 'train', 'val', or 'test'.")
    
    # Infer the top-k aspects for each user-item pair
    aspect_pred = []
    for i in tqdm(range(len(csv))):
        aspect_name = top_k_aspect(i, args.top_k, csv, node_emb, aspect_emb, dataset)
        aspect_pred.append(aspect_name)

    new_csv = csv.copy(deep=True)
    new_csv['aspect_pred'] = aspect_pred

    # Summarize for next step reasoning
    with open(f'prompt/{args.dataset_name}/aspect_prediction.txt', 'r', encoding='utf-8') as f:
        prompt = f.read()

    detail_summary = [""] * len(new_csv)

    with ThreadPoolExecutor(max_workers=64) as executor:
        futures = [
            executor.submit(process_row, i, row, prompt, dataset)
            for i, row in new_csv.iterrows()
        ]

        for future in tqdm(as_completed(futures), total=len(futures)):
            i, result = future.result()
            detail_summary[i] = result

    new_csv['detail_summary'] = detail_summary

    new_csv.to_csv(f'data/{args.dataset_name}/aspect_prediction_{args.mode}.csv', index=False)