import torch
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments, T5EncoderModel, AutoTokenizer
from datasets import load_dataset

class DistillationTrainer(Trainer):
    def __init__(self, teacher_model, temperature=2.0, alpha=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model.to(self.args.device)
        self.teacher_model.eval()
        self.temperature = temperature  # temperature for KL divergence
        self.alpha = alpha              # weight for combining losses

    # Note: The compute_loss signature now accepts num_items_in_batch to avoid errors.
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Forward pass for the student
        outputs_student = model(**inputs)
        with torch.no_grad():
            outputs_teacher = self.teacher_model(**inputs)
        
        # Compute KL divergence loss between student and teacher logits (or last hidden states)
        # Use log_softmax for student and softmax for teacher because KLDivLoss
        # expects log probabilities as its input.
        kl_loss = F.kl_div(
            F.log_softmax(outputs_student.last_hidden_state / self.temperature, dim=-1),
            F.softmax(outputs_teacher.last_hidden_state / self.temperature, dim=-1),
            reduction="batchmean",
        ) * (self.temperature ** 2)
        
        # Optionally, also compute MSE loss on the hidden state representations
        mse_loss = F.mse_loss(outputs_student.last_hidden_state, outputs_teacher.last_hidden_state)
        
        # Combine the two losses using the weighting factor alpha:
        loss = self.alpha * mse_loss + (1 - self.alpha) * kl_loss
        
        return (loss, outputs_student) if return_outputs else loss

def main():
    # Load teacher and student models. The student has been pruned and saved previously.
    teacher_model = T5EncoderModel.from_pretrained("ai-forever/FRIDA")
    student_model = T5EncoderModel.from_pretrained("./student_FRIDA")
    tokenizer = AutoTokenizer.from_pretrained("ai-forever/FRIDA")
    
    # For demonstration, we use a Russian SuperGLUE task – here, for example, the RCB task.
    # Note: Russian SuperGLUE is a benchmark with a modest amount of examples and may be
    # insufficient by itself for large-scale distillation. In practice, you might want to
    # supplement it with additional data.
    dataset = load_dataset("RussianNLP/russian_super_glue", "rcb")
    
    def preprocess(examples):
        # This simple preprocessing concatenates the 'premise' and 'hypothesis' fields.
        texts = [p + " " + h for p, h in zip(examples["premise"], examples["hypothesis"])]
        return tokenizer(texts, max_length=128, padding="max_length", truncation=True)
    
    tokenized = dataset.map(preprocess, batched=True)
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])
    
    training_args = TrainingArguments(
        output_dir="./distilled_model",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        evaluation_strategy="steps",
        eval_steps=500,
        logging_steps=100,
        save_steps=500,
        save_total_limit=2,
    )
    
    distil_trainer = DistillationTrainer(
        teacher_model=teacher_model,
        model=student_model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        temperature=2.0,
        alpha=0.5,
    )
    
    distil_trainer.train()
    student_model.save_pretrained("./distilled_student")
    tokenizer.save_pretrained("./distilled_student")

if __name__ == "__main__":
    main()
