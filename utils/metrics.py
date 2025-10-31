import torch
import torch.nn as nn
import traceback
from transformers import BartTokenizer, BartForConditionalGeneration
from typing import List
import numpy as np
import pickle as pkl
import evaluate
import numpy as np
import json
from tqdm import tqdm
from openai import OpenAI
import concurrent.futures

class BARTScorer:
    def __init__(self, device='cuda:0', max_length=1024, checkpoint='facebook/bart-large-cnn'):
        # Set up model
        self.device = device
        self.max_length = max_length
        self.tokenizer = BartTokenizer.from_pretrained(checkpoint)
        self.model = BartForConditionalGeneration.from_pretrained(checkpoint)
        self.model.eval()
        self.model.to(device)

        # Set up loss
        self.loss_fct = nn.NLLLoss(reduction='none', ignore_index=self.model.config.pad_token_id)
        self.lsm = nn.LogSoftmax(dim=1)

    def load(self, path=None):
        """ Load model from paraphrase finetuning """
        if path is None:
            path = 'models/bart.pth'
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def score(self, srcs, tgts, batch_size=4):
        """ Score a batch of examples """
        score_list = []
        for i in range(0, len(srcs), batch_size):
            src_list = srcs[i: i + batch_size]
            tgt_list = tgts[i: i + batch_size]
            try:
                with torch.no_grad():
                    encoded_src = self.tokenizer(
                        src_list,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    )
                    encoded_tgt = self.tokenizer(
                        tgt_list,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    )
                    src_tokens = encoded_src['input_ids'].to(self.device)
                    src_mask = encoded_src['attention_mask'].to(self.device)

                    tgt_tokens = encoded_tgt['input_ids'].to(self.device)
                    tgt_mask = encoded_tgt['attention_mask']
                    tgt_len = tgt_mask.sum(dim=1).to(self.device)

                    output = self.model(
                        input_ids=src_tokens,
                        attention_mask=src_mask,
                        labels=tgt_tokens
                    )
                    logits = output.logits.view(-1, self.model.config.vocab_size)
                    loss = self.loss_fct(self.lsm(logits), tgt_tokens.view(-1))
                    loss = loss.view(tgt_tokens.shape[0], -1)
                    loss = loss.sum(dim=1) / tgt_len
                    curr_score_list = [-x.item() for x in loss]
                    score_list += curr_score_list

            except RuntimeError:
                traceback.print_exc()
                print(f'source: {src_list}')
                print(f'target: {tgt_list}')
                exit(0)
        return score_list

    def multi_ref_score(self, srcs, tgts: List[List[str]], agg="mean", batch_size=4):
        # Assert we have the same number of references
        ref_nums = [len(x) for x in tgts]
        if len(set(ref_nums)) > 1:
            raise Exception("You have different number of references per test sample.")

        ref_num = len(tgts[0])
        score_matrix = []
        for i in range(ref_num):
            curr_tgts = [x[i] for x in tgts]
            scores = self.score(srcs, curr_tgts, batch_size)
            score_matrix.append(scores)
        if agg == "mean":
            score_list = np.mean(score_matrix, axis=0)
        elif agg == "max":
            score_list = np.max(score_matrix, axis=0)
        else:
            raise NotImplementedError
        return list(score_list)

    def test(self, batch_size=3):
        """ Test """
        src_list = [
            'This is a very good idea. Although simple, but very insightful.',
            'Can I take a look?',
            'Do not trust him, he is a liar.'
        ]

        tgt_list = [
            "That's stupid.",
            "What's the problem?",
            'He is trustworthy.'
        ]

        print(self.score(src_list, tgt_list, batch_size))

class MetricScore:
    def __init__(self, dataset, api_key):
        if api_key != None:
            self.client = OpenAI(api_key=api_key)
        else:
            self.client = None

        with open("gpt_score_prompt.txt", "r") as f:
            self.gpt_score_prompt = f.read()
        self.dataset = dataset
        self.pred_input_path = f"data/{dataset}/tst_pred.pkl"
        self.ref_input_path = f"data/{dataset}/tst_ref.pkl"

        with open(self.pred_input_path, "rb") as f:
            self.data = pkl.load(f)
        with open(self.ref_input_path, "rb") as f:
            self.ref_data = pkl.load(f)

    def two_seq_same(self, sa, sb):
        if len(sa) != len(sb):
            return False
        for wa, wb in zip(sa, sb):
            if wa != wb:
                return False
        return True

    def unique_sentence_percent(self, sequence_batch):
        unique_seq = []
        for seq in sequence_batch:
            # seq is a list of words
            count = 0
            for uni_seq in unique_seq:
                if self.two_seq_same(seq, uni_seq):
                    count += 1
                    break
            if count == 0:
                unique_seq.append(seq)

        return len(unique_seq) / len(sequence_batch), len(unique_seq)

    def get_gpt_response(self, prompt):
        completion = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": self.gpt_score_prompt},
                {"role": "user", "content": prompt},
            ],
            model="gpt-3.5-turbo",
        )
        response = completion.choices[0].message.content
        return float(response)

    def GPT_score(self, predictions, references):
        prompts = []
        for i in range(len(predictions)):
            prompt = {
                "prediction": predictions[i],
                "reference": references[i],
            }
            prompts.append(json.dumps(prompt))

        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=64) as executor:
            futures = [executor.submit(self.get_gpt_response, prompt) for prompt in prompts]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(prompts), desc="Processing GPT responses"):
                results.append(future.result())

        return np.mean(results), np.std(results)

    def BERT_score(self, predictions, references):
        bertscore = evaluate.load("bertscore")
        results = bertscore.compute(
            predictions=predictions,
            references=references,
            lang="en",
            rescale_with_baseline=True,
        )
        precision = results["precision"]
        recall = results["recall"]
        f1 = results["f1"]
        return (
            np.mean(precision),
            np.mean(recall),
            np.mean(f1),
            np.std(precision),
            np.std(recall),
            np.std(f1),
            precision,
            recall,
            f1
        )

    def BART_score(self, predictions, references):
        bart_scorer = BARTScorer(device='cuda:0', checkpoint='facebook/bart-large-cnn')
        scores = []
        for i in tqdm(range(0, len(predictions), 4), desc="Computing BART scores"):
            batch_pred = predictions[i:i+4]
            batch_ref = references[i:i+4]
            batch_scores = bart_scorer.score(batch_pred, batch_ref, batch_size=4)
            scores.extend(batch_scores)
        return np.mean(scores), np.std(scores), scores

    def BLEURT_score(self, predictions, references):
        bleurt = evaluate.load("bleurt", module_type="metric")
        results = bleurt.compute(predictions=predictions, references=references)
        scores = results['scores']
        return np.mean(scores), np.std(scores), scores

    def get_score(self):
        scores = {}
        (
            bert_precison,
            bert_recall,
            bert_f1,
            bert_precison_std,
            bert_recall_std,
            bert_f1_std,
            precision_list,
            recall_list,
            f1_list
        ) = self.BERT_score(self.data, self.ref_data)

        gpt_score, gpt_std = self.GPT_score(self.data, self.ref_data)
        bart_score, bart_score_std, bart_list = self.BART_score(self.data, self.ref_data)
        bleurt_score, bleurt_score_std, bleurt_list = self.BLEURT_score(self.data, self.ref_data)
        
        tokens_predict = [s.split() for s in self.data]
        usr, _ = self.unique_sentence_percent(tokens_predict)

        scores["gpt_score"] = gpt_score
        scores["bert_precision"] = bert_precison
        scores["bert_recall"] = bert_recall
        scores["bert_f1"] = bert_f1
        scores["usr"] = usr

        scores["gpt_std"] = gpt_std
        scores["bert_precision_std"] = bert_precison_std
        scores["bert_recall_std"] = bert_recall_std
        scores["bert_f1_std"] = bert_f1_std

        scores["bart_score"] = bart_score
        scores["bart_score_std"] = bart_score_std
        scores["bleurt_score"] = bleurt_score
        scores["bleurt_score_std"] = bleurt_score_std
        
        list_scores = {
            "bert_precision_list": precision_list,
            "bert_recall_list": recall_list,
            "bert_f1_list": f1_list,
            "bart_list": bart_list,
            "bleurt_list": bleurt_list
        }
        with open(f"{self.dataset}_tst_score_list.pkl", "wb") as file:
            pkl.dump(list_scores, file)
            
        return scores

    def print_score(self):
        scores = self.get_score()
        print(f"Dataset: {self.dataset}")
        # print(f"ratio: {args.ratio}")
        print("Explanability Evaluation Metrics:")
        print(f"gpt_score: {scores['gpt_score']:.4f}")
        print(f"bert_precision: {scores['bert_precision']:.4f}")
        print(f"bert_recall: {scores['bert_recall']:.4f}")
        print(f"bert_f1: {scores['bert_f1']:.4f}")
        print(f"bart_score: {scores['bart_score']:.4f}")
        print(f"bleurt_score: {scores['bleurt_score']:.4f}")
        print(f"usr: {scores['usr']:.4f}")
        print("-"*30)
        print("Standard Deviation:")
        print(f"gpt_std: {scores['gpt_std']:.4f}")
        print(f"bert_precision_std: {scores['bert_precision_std']:.4f}")
        print(f"bert_recall_std: {scores['bert_recall_std']:.4f}")
        print(f"bert_f1_std: {scores['bert_f1_std']:.4f}")
        print(f"bart_score_std: {scores['bart_score_std']:.4f}")
        print(f"bleurt_score_std: {scores['bleurt_score_std']:.4f}")

    def report_score(self):
        scores = self.get_score()
        with open(f"score_report_{self.dataset}.txt", "w", encoding="utf-8") as f:
            f.write(f"Dataset: {self.dataset}\n")
            f.write("Explainability Evaluation Metrics:\n")
            
            metrics = [
                'gpt_score', 'bert_precision', 'bert_recall',
                'bert_f1', 'bart_score', 'bleurt_score', 'usr'
            ]
            for key in metrics:
                f.write(f"{key}: {scores[key]:.4f}\n")
            
            f.write("-" * 30 + "\n")
            f.write("Standard Deviation:\n")
            
            for key in metrics:
                std_key = f"{key}_std"
                if std_key in scores:
                    f.write(f"{std_key}: {scores[std_key]:.4f}\n")