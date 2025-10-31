from tqdm import tqdm
from unsloth import FastLanguageModel
import pickle as pkl
import json
import pandas as pd
import argparse

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Inference Configuration")
    parser.add_argument('--dataset_name', type=str, default='amazon', help='Name of the dataset')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    dataset_name = args.dataset_name

    test_data = pd.read_csv(f'data/{dataset_name}/augment_test.csv')

    max_seq_length = 4096
    model_name = f"pretrained/{dataset_name}"
        
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        dtype=None,
    )

    FastLanguageModel.for_inference(model)

    instruction = json.load(open('prompt/llm_finetune_eval_prompt.json', 'r'))
    system_prompt = instruction[dataset_name]

    data_prompt = """{}

    ### Input:
    User profile: {}
    Item profile: {}
    Supporting information from given user/item: {}
    Supporting information from relevant user-item pairs: {}

    ### Response:
    {}"""

    EOS_TOKEN = tokenizer.eos_token
    
    batch_size = args.batch_size

    predictions, references = [], []
    for i in tqdm(range(0, len(test_data), batch_size)):
        end = i + batch_size if i + batch_size <= len(test_data) else len(test_data)
        batch = test_data[i:i + batch_size]
        gt = list(batch['explanation'])
        user_summaries = list(batch['user_summary'])
        item_summaries = list(batch['item_summary'])
        details = list(batch['detail_summary'])
        augments = list(batch['augment_summary'])
        inputs = [
            data_prompt.format(
                system_prompt,
                user_summary,
                item_summary,
                detail if detail != "" else "No information.",
                augment if augment != "" else "No information.",
                "",
            ) for user_summary, item_summary, detail, augment in zip(user_summaries, item_summaries, details, augments)
        ]
        
        
        inputs = tokenizer(inputs, return_tensors = "pt", padding = True).to("cuda")
        outputs = model.generate(**inputs, max_new_tokens = 256, do_sample=False, temperature=0.0)
        
        output_strs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        output_strs = [answer.split("### Response:\n")[-1] for answer in output_strs]
        predictions.extend(output_strs)
        references.extend(gt)
    
    with open(f"data/{dataset_name}/tst_pred.pkl", "wb") as file:
        pkl.dump(predictions, file)
    
    with open(f"data/{dataset_name}/tst_ref.pkl", "wb") as file:
        pkl.dump(references, file)