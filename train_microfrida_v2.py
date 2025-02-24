import os
import re
import torch
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments, T5EncoderModel, AutoTokenizer
from datasets import load_dataset, Dataset
from tqdm import tqdm

# Custom Distillation Trainer that computes KL divergence + MSE losses.
class DistillationTrainer(Trainer):
    def __init__(self, teacher_model, temperature=2.0, alpha=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model.to(self.args.device)
        self.teacher_model.eval()
        self.temperature = temperature  # Temperature scaling for soft targets.
        self.alpha = alpha  # Weight: loss = alpha * MSE + (1 - alpha) * KL loss.

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs_student = model(**inputs)
        with torch.no_grad():
            outputs_teacher = self.teacher_model(**inputs)

        kl_loss = F.kl_div(
            F.log_softmax(outputs_student.last_hidden_state / self.temperature, dim=-1),
            F.softmax(outputs_teacher.last_hidden_state / self.temperature, dim=-1),
            reduction="batchmean",
        ) * (self.temperature ** 2)

        mse_loss = F.mse_loss(outputs_student.last_hidden_state, outputs_teacher.last_hidden_state)
        loss = self.alpha * mse_loss + (1 - self.alpha) * kl_loss
        return (loss, outputs_student) if return_outputs else loss

def flatten_dataset(ds):
    new_examples = []
    for example in tqdm(ds, desc="Flattening examples"):
        paragraphs = re.split(r"\n\s*\n", example["text"])
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        # Prepend the prefix to each paragraph.
        new_examples.extend([{"text": "categorize_topic: " + p} for p in paragraphs])
    return Dataset.from_list(new_examples)

def main():
    # Load teacher and student models.
    teacher_model = T5EncoderModel.from_pretrained("ai-forever/FRIDA")
    student_model = T5EncoderModel.from_pretrained("./student_FRIDA")
    tokenizer = AutoTokenizer.from_pretrained("ai-forever/FRIDA")
    
    # ---- Load only the first four parquet files of the taiga_stripped_proza dataset ----
    data_files = {
        "train": [
            "data/train-00000-of-00083-5a836a36820bbc21.parquet",
            "data/train-00001-of-00083-6a059492052de562.parquet",
            "data/train-00002-of-00083-6ab99ef2eda1556f.parquet",
            "data/train-00003-of-00083-fc34df8e6a0b97a4.parquet",
        ]
    }
    
    # Use ignore_verifications=True to bypass split metadata check.
    ds = load_dataset("cointegrated/taiga_stripped_proza", data_files=data_files, ignore_verifications=True)
    ds = ds["train"]
    
    # ---- Flatten dataset: split each record's text into paragraphs and add the prefix. ----
    print("Splitting records into paragraphs ...")
    ds = flatten_dataset(ds)
    
    print("Total samples after flattening:", len(ds))
    
    # Limit to 20K training samples.
    if len(ds) > 20000:
        ds = ds.select(range(20000))
    print("Total training samples (limited to 20K):", len(ds))
    
    # ---- Tokenize the dataset. ----
    def preprocess(examples):
        return tokenizer(examples["text"], max_length=128, padding="max_length", truncation=True)
    
    print("Tokenizing training samples...")
    ds = ds.map(preprocess, batched=True, desc="Tokenizing dataset")
    ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
    
    # ---- Set up distillation training ----
    training_args = TrainingArguments(
        output_dir="./distilled_model",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        evaluation_strategy="steps",
        eval_steps=500,
        logging_steps=16,
        save_steps=100,
        save_total_limit=2,
    )
    
    distil_trainer = DistillationTrainer(
        teacher_model=teacher_model,
        model=student_model,
        args=training_args,
        train_dataset=ds,
        eval_dataset=ds,  # For simplicity, using the same split for evaluation.
        tokenizer=tokenizer,
        temperature=2.0,
        alpha=0.5,
    )
    
    # Start training. Trainer will display progress via tqdm.
    distil_trainer.train()
    student_model.save_pretrained("./distilled_student")
    print("Distillation completed and student model saved.")

if __name__ == "__main__":
    main()
