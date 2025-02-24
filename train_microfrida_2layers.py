import os
import re
import torch
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments, T5EncoderModel, AutoTokenizer
from datasets import load_dataset, Dataset
from tqdm import tqdm

# Optionally suppress TorchDynamo errors and fall back to eager mode.
import torch._dynamo
torch._dynamo.config.suppress_errors = True

# Custom Distillation Trainer that computes KL divergence + MSE losses.
class DistillationTrainer(Trainer):
    def __init__(self, teacher_model, temperature=2.0, alpha=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Do not move teacher_model here. Instead, ensure that in compute_loss both teacher and student operate on the same device.
        self.teacher_model = teacher_model  
        self.temperature = temperature   # Temperature scaling.
        self.alpha = alpha               # Weighting: loss = alpha * MSE + (1 - alpha) * KL.

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Compute student outputs.
        outputs_student = model(**inputs)
        # Determine the device where the student outputs were computed.
        device = outputs_student.last_hidden_state.device
        # Move the teacher inputs to the same device.
        teacher_inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            # When using DataParallel, the teacher_model should already be wrapped.
            outputs_teacher = self.teacher_model(**teacher_inputs)
        # Compute KL divergence loss with temperature scaling.
        kl_loss = F.kl_div(
            F.log_softmax(outputs_student.last_hidden_state / self.temperature, dim=-1),
            F.softmax(outputs_teacher.last_hidden_state / self.temperature, dim=-1),
            reduction="batchmean",
        ) * (self.temperature ** 2)
        # Compute Mean Squared Error loss.
        mse_loss = F.mse_loss(outputs_student.last_hidden_state, outputs_teacher.last_hidden_state)
        # Combine both losses.
        loss = self.alpha * mse_loss + (1 - self.alpha) * kl_loss
        return (loss, outputs_student) if return_outputs else loss

def flatten_dataset(ds):
    new_examples = []
    for example in tqdm(ds, desc="Flattening examples"):
        # Split the input text into paragraphs.
        paragraphs = re.split(r"\n\s*\n", example["text"])
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        # Prepend the prefix to each paragraph.
        new_examples.extend([{"text": "categorize_topic: " + p} for p in paragraphs])
    return Dataset.from_list(new_examples)

def main():
    # Load teacher and student models.
    teacher_model = T5EncoderModel.from_pretrained("ai-forever/FRIDA")
    teacher_model.eval()  # Set teacher model to evaluation mode.
    student_model = T5EncoderModel.from_pretrained("./student_FRIDA")
    tokenizer = AutoTokenizer.from_pretrained("ai-forever/FRIDA")
    
    # If multiple GPUs are available, wrap the teacher model with DataParallel.
    if torch.cuda.device_count() > 1:
        teacher_model = torch.nn.DataParallel(teacher_model)
    
    # ----- Enable gradient checkpointing on the student model -----
    student_model.config.gradient_checkpointing = True
    if hasattr(student_model, "_set_gradient_checkpointing"):
        student_model._set_gradient_checkpointing(True)
    
    # Optionally compile the student model with torch.compile when using a single GPU.
    if hasattr(torch, "compile") and torch.cuda.device_count() == 1:
        student_model = torch.compile(student_model)
    
    # ---- Load only the first few parquet files of the dataset ----
    data_files = {
        "train": [
            "data/train-00000-of-00083-5a836a36820bbc21.parquet",
            "data/train-00001-of-00083-6a059492052de562.parquet",
            "data/train-00002-of-00083-6ab99ef2eda1556f.parquet",
            "data/train-00003-of-00083-fc34df8e6a0b97a4.parquet",
        ]
    }
    ds = load_dataset("cointegrated/taiga_stripped_proza", data_files=data_files, ignore_verifications=True)
    ds = ds["train"]

    # ---- Flatten dataset: split text into paragraphs and add a prefix.
    print("Splitting records into paragraphs ...")
    ds = flatten_dataset(ds)
    print("Total samples after flattening:", len(ds))
    # Limit dataset sample size if needed.
    if len(ds) > 50000:
        ds = ds.select(range(50000))
    print("Total training samples (limited):", len(ds))
    
    # ---- Tokenize the dataset ----
    def preprocess(examples):
        return tokenizer(examples["text"], max_length=128, padding="max_length", truncation=True)
    
    print("Tokenizing training samples...")
    ds = ds.map(preprocess, batched=True, desc="Tokenizing dataset")
    ds.set_format(type="torch", columns=["input_ids", "attention_mask"])

    # ---- Set up TrainingArguments ----
    training_args = TrainingArguments(
        output_dir="./distilled_model_v2",
        num_train_epochs=1,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        evaluation_strategy="no",
        logging_steps=10,
        save_steps=100,
        dataloader_num_workers=8,
        warmup_ratio=0.1,
        report_to=["tensorboard"],
        remove_unused_columns=False
    )
    
    # Instantiate the custom distillation trainer.
    distil_trainer = DistillationTrainer(
        teacher_model=teacher_model,
        model=student_model,
        args=training_args,
        train_dataset=ds,
        eval_dataset=ds,  # For simplicity, using the same dataset for evaluation.
        tokenizer=tokenizer,
        temperature=2.0,
        alpha=0.5,
    )
    
    # Start training. The Trainer will automatically handle multi-GPU data distribution.
    distil_trainer.train()
    student_model.save_pretrained("./distilled_student_3layers")
    print("Distillation completed and student model saved.")

if __name__ == "__main__":
    main()
