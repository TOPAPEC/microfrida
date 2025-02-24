import os
import re
import torch
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments, T5EncoderModel, AutoTokenizer
from datasets import load_dataset, Dataset
from tqdm import tqdm
import torch._dynamo
torch._dynamo.config.suppress_errors = True
from torch.distributed.elastic.multiprocessing.errors import record

class DistillationTrainer(Trainer):
    def __init__(self, teacher_model, temperature=2.0, alpha=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.temperature = temperature
        self.alpha = alpha

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs_student = model(**inputs)
        device = outputs_student.last_hidden_state.device
        teacher_inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**teacher_inputs)
        teacher_hidden = teacher_outputs["last_hidden_state"]
        student_logits = outputs_student.last_hidden_state / self.temperature
        teacher_logits = teacher_hidden / self.temperature
        kl_loss = F.kl_div(F.log_softmax(student_logits, dim=-1), F.softmax(teacher_logits, dim=-1), reduction="batchmean") * (self.temperature ** 2)
        mse_loss = F.mse_loss(outputs_student.last_hidden_state, teacher_hidden)
        loss = self.alpha * mse_loss + (1 - self.alpha) * kl_loss
        return (loss, outputs_student) if return_outputs else loss

def flatten_dataset(ds):
    new_examples = []
    for example in tqdm(ds, desc="Flattening examples"):
        paragraphs = re.split(r"\n\s*\n", example["text"])
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        new_examples.extend([{"text": "categorize_topic: " + p} for p in paragraphs])
    return Dataset.from_list(new_examples)

@record
def main():
    local_rank = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else -1
    device = torch.device(f"cuda:{local_rank}") if local_rank != -1 else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    teacher_model = T5EncoderModel.from_pretrained("ai-forever/FRIDA").eval().to(device)
    student_model = T5EncoderModel.from_pretrained("./student_FRIDA").to(device)
    tokenizer = AutoTokenizer.from_pretrained("ai-forever/FRIDA")
    student_model.config.gradient_checkpointing = True
    if hasattr(student_model, "_set_gradient_checkpointing"):
        student_model._set_gradient_checkpointing(True)
    if hasattr(torch, "compile") and torch.cuda.device_count() == 1:
        student_model = torch.compile(student_model)
    data_files = {
        "train": [
            "data/train-00000-of-00083-5a836a36820bbc21.parquet",
            "data/train-00001-of-00083-6a059492052de562.parquet",
            "data/train-00002-of-00083-6ab99ef2eda1556f.parquet",
            "data/train-00003-of-00083-fc34df8e6a0b97a4.parquet"
        ]
    }
    ds = load_dataset("cointegrated/taiga_stripped_proza", data_files=data_files, ignore_verifications=True)["train"]
    print("Splitting records into paragraphs ...")
    ds = flatten_dataset(ds)
    print("Total samples after flattening:", len(ds))
    if len(ds) > 50000:
        ds = ds.select(range(50000))
    print("Total training samples (limited):", len(ds))
    def preprocess(examples):
        return tokenizer(examples["text"], max_length=128, padding="max_length", truncation=True)
    print("Tokenizing training samples...")
    ds = ds.map(preprocess, batched=True, desc="Tokenizing dataset")
    ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
    training_args = TrainingArguments(
        output_dir="./distilled_model_v2",
        num_train_epochs=1,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        evaluation_strategy="no",
        logging_steps=10,
        save_steps=100,
        dataloader_num_workers=0,
        warmup_ratio=0.1,
        report_to=["tensorboard"],
        remove_unused_columns=False,
        # strategy="ddp"
    )
    distil_trainer = DistillationTrainer(
        teacher_model=teacher_model,
        model=student_model,
        args=training_args,
        train_dataset=ds,
        eval_dataset=ds,
        tokenizer=tokenizer,
        temperature=2.0,
        alpha=0.5,
    )
    distil_trainer.train()
    student_model.save_pretrained("./distilled_student_3layers")
    print("Distillation completed and student model saved.")

if __name__ == "__main__":
    main()
