import os
import mlflow
import dagshub
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model
import torch

token = os.getenv("DAGSHUB_API_TOKEN")
dagshub.auth.add_app_token(token)
dagshub.init(
    repo_owner="naomatheus",
    repo_name="210-section-5-YOELO",
    mlflow=True
)
mlflow.set_experiment("finetuning-instruct")


def fine_tune(
    model_name,
    dataset_name,
    output_dir="./outputs",
    num_train_epochs=1,
    learning_rate=2e-4,
    batch_size=8,
    grad_accum=2,
    max_seq_length=768,
    fp16=False,
):
    with mlflow.start_run():
        mlflow.log_params({
            "model_name": model_name,
            "dataset_name": dataset_name,
            "epochs": num_train_epochs,
            "lr": learning_rate,
            "batch_size": batch_size,
            "grad_accum": grad_accum,
            "max_seq_length": max_seq_length,
        })

        # Load dataset from Hugging Face
        print(f"Loading dataset from Hugging Face: {dataset_name}")
        dataset = load_dataset(dataset_name, split="train")
        print(f"Loaded {len(dataset)} samples.")

        # Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Tokenize - your data already has complete [INST] format
        def tokenize(example):
            text = example.get("text", "")
            return tokenizer(text, truncation=True, max_length=max_seq_length, padding="max_length")

        tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)

        # Load model with fp16 for memory efficiency
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        # LoRA setup
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # Trainer setup
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            learning_rate=learning_rate,
            fp16=False,
            bf16=True,
            save_total_limit=2,
            logging_steps=50,
            report_to=["mlflow"],
            save_steps=500,
            optim="adamw_torch",
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
        )

        trainer = Trainer(
            model=model,
            train_dataset=tokenized,
            args=training_args,
            data_collator=data_collator,
        )

        trainer.train()
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

        mlflow.log_artifacts(output_dir)
        print("Fine-tuning complete.")


if __name__ == "__main__":
    model = os.getenv("MODEL_NAME", "codellama/CodeLlama-7b-Instruct-hf")
    dataset_name = os.getenv("HF_DATASET", "tahleestone/k8s_instruct_dataset")
    fine_tune(model_name=model, dataset_name=dataset_name)
    exit(0)