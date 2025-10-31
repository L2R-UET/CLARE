import os
import argparse
from dotenv import load_dotenv
from huggingface_hub import login
import transformers
import torch
import pickle as pkl
from tqdm import tqdm
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import openai
warnings.filterwarnings("ignore")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Aspect Extraction Configuration")
    parser.add_argument('--dataset_name', type=str, default='amazon', help='Name of the dataset')
    return parser.parse_args()

def chatgpt_extract_aspect_row(i, data, instruction, model):
    explanation = data.iloc[i]['explanation']
    try:
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": explanation}
            ],
            max_tokens=256
        )
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        answer = f"ERROR: {e}"
    return i, answer

def chatgpt_extract_aspects(instruction, data, model="gpt-4o-mini-2024-07-18", max_workers=64):
    aspect_list = [None] * len(data)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(chatgpt_extract_aspect_row, i, data, instruction, model)
            for i in range(len(data))
        ]

        for future in tqdm(as_completed(futures), total=len(futures)):
            i, result = future.result()
            aspect_list[i] = result

    return aspect_list

def hf_extract_aspects(instruction, data, model_id="meta-llama/Meta-Llama-3.1-8B-Instruct"):
    login(token=os.getenv("HF_TOKEN"))
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

    aspect_list = []

    for i in tqdm(range(len(data))):
        explanation = data.iloc[i]['explanation']

        messages=[
            {"role": "system", "content": instruction},
            {"role": "user", "content": explanation}
        ],
        
        outputs = pipeline(
            messages,
            max_new_tokens=512,
            do_sample=False,
            temperature=0.0,
            pad_token_id=128001
        )
        aspect = outputs[0]["generated_text"][-1]["content"].split("Aspect:")[-1].strip()
        aspect_list.append(aspect)

    return aspect_list


def main(args):

    dataset_name = args.dataset_name

    path = f'data/{dataset_name}/trn.pkl'
    with open(path, 'rb') as f:
        data = pkl.load(f)

    with open(f'prompt/{dataset_name}/extract_aspect.txt', 'r', encoding='utf-8') as f:
        instruction = f.read()
    
    openai_api_key = os.getenv("OPENAI_API_KEY") | None

    if openai_api_key != None:
        openai.api_key = openai_api_key
        aspect_list = chatgpt_extract_aspects(instruction, data, model="gpt-4o-mini-2024-07-18")
    else:
        aspect_list = hf_extract_aspects(instruction, data, model_id="meta-llama/Meta-Llama-3.1-8B-Instruct")
    

    data['aspect'] = aspect_list
    data.to_csv(f"data/{dataset_name}/{dataset_name}_aspect.csv", index=False)

if __name__ == "__main__":
    load_dotenv()
    args = parse_args()
    main(args)