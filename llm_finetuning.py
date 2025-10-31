import os
import argparse
from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import Dataset
import wandb
import json

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from dotenv import load_dotenv


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="LLM Fine-tuning Configuration")
    parser.add_argument('--dataset_name', type=str, default='amazon', help='Name of the dataset')
    parser.add_argument('--num_epochs', type=int, default=2, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size per device for training/validation')
    return parser.parse_args()

def formatting_prompt(examples):
    user_profiles = examples["user_summary"]
    item_profiles = examples["item_summary"]
    details = examples["detail_summary"]
    augments = examples["augment_summary"]
    outputs = examples["explanation"]
    texts = []
    for user_profile, item_profile, detail, augment, output in zip(user_profiles, item_profiles, details, augments, outputs):
        text = data_prompt.format(system_prompt, user_profile, item_profile, detail, augment, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }

if __name__ == "__main__":

    args = parse_args()
    load_dotenv()
    wandb.login(key=os.getenv("WANDB_KEY_LOGIN"))
    wandb.init(
        project='XRec_LLM', 
        job_type="training", 
        anonymous="allow"
    )

    dataset_name = args.dataset_name

    train_df = pd.read_csv(f'data/{dataset_name}/augment_train.csv')
    val_df = pd.read_csv(f'data/{dataset_name}/augment_val.csv')

    max_seq_length = 4096
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
        
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        dtype=None,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = True,
        random_state = 3407,
        use_rslora = False,
        loftq_config = None,
    )
    print(model.print_trainable_parameters())

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

    train_data = Dataset.from_pandas(train_df)
    train_data = train_data.map(formatting_prompt, batched=True)

    val_data = Dataset.from_pandas(val_df)
    val_data = val_data.map(formatting_prompt, batched=True)

    training_args = TrainingArguments(
        learning_rate=1e-3,
        lr_scheduler_type="cosine",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        num_train_epochs=args.num_epochs,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        warmup_steps=10,
        output_dir="output",
        seed=0,
        disable_tqdm=False,
        report_to="wandb",
        ddp_find_unused_parameters=False,
        eval_strategy="steps",
        eval_steps=20,
        save_strategy="epoch",
    ),

    trainer=SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_data,
        eval_dataset=val_data,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=4,
        packing=False,
        args=training_args,
    )

    os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    trainer.train()
    wandb.finish()

    finetuned_model_name = f'pretrained/{dataset_name}'
    model.save_pretrained(finetuned_model_name)
    tokenizer.save_pretrained(finetuned_model_name)