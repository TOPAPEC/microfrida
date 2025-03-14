import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
import lightning.fabric as fabric
import os
import time
import gc
from datetime import timedelta
from tqdm import tqdm
from student import FridaDistillationModel
import wandb
import random
from dotenv import load_dotenv

load_dotenv()
random.seed(42)
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import mteb
from mteb import MTEB
import numpy as np

frida_prompts = {
            "Classification": "categorize: ",
            "MultilabelClassification": "categorize: ",
            "Clustering": "categorize_topic: ",
            "PairClassification": "paraphrase: ",
            "Reranking": "paraphrase: ",
            "Reranking-query": "search_query: ",
            "Reranking-passage": "search_document: ",
            "STS": "paraphrase: ",
            "Summarization": "categorize: ",
            "query": "search_query: ",
            "passage": "search_document: ",
            "CEDRClassification": "categorize_sentiment: ",
            "GeoreviewClassification": "categorize_sentiment: ",
            "HeadlineClassification": "categorize_topic: ",
            "InappropriatenessClassification": "categorize_topic: ",
            "KinopoiskClassification": "categorize_sentiment: ",
            "MassiveIntentClassification": "paraphrase: ",
            "MassiveScenarioClassification": "paraphrase: ",
            "RuReviewsClassification": "categorize_sentiment: ",
            "RuSciBenchGRNTIClassification": "categorize_topic: ",
            "RuSciBenchOECDClassification": "categorize_topic: ",
            "SensitiveTopicsClassification": "categorize_topic: ",
            "TERRa": "categorize_entailment: ",
            "RiaNewsRetrieval": "categorize: ",
            None: "categorize: "
        }
# -------------------------------------------
# TeacherEmbedder: Uses model.get_embedding()
# -------------------------------------------
class TeacherEmbedder:
    def __init__(self, model, tokenizer, max_length=512, prompt="categorize: "):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt = prompt

    def pool(self, hidden_state, mask, pooling_method="cls"):
        if pooling_method == "mean":
            s = torch.sum(hidden_state * mask.unsqueeze(-1).float(), dim=1)
            d = mask.sum(dim=1, keepdim=True).float()
            return s / d
        elif pooling_method == "cls":
            return hidden_state[:, 0]

    def encode(self, texts, batch_size=128, task_type=None, **kwargs):
        """
        Encodes a list of sentences by first prepending a prompt (default "categorize: "),
        tokenizing (max_length=512, padding/truncation enabled), and then processing through
        the model's teacher pathway (using get_embedding). The output is CLS-pooled and L2-normalized.
        """
        all_embeddings = []
        prompt = frida_prompts[task_type]
        for i in tqdm(range(0, len(texts), batch_size), desc="Teacher encoding"):
            # Prepend prompt to each sentence
            batch_texts = [f"{prompt}{sent}" for sent in texts[i:i + batch_size]]
            encoded = self.tokenizer(
                batch_texts,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            encoded = {k: v.to(self.model.device) for k, v in encoded.items()}
            with torch.no_grad():
                # Get teacher embeddings using model.get_embedding()
                teacher_outputs = self.model.teacher_model(encoded["input_ids"], encoded["attention_mask"])
                # Apply CLS pooling (using the first token)
                embeddings = self.pool(teacher_outputs.last_hidden_state, encoded.get("attention_mask"), pooling_method="cls")
                embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu())
        return torch.cat(all_embeddings, dim=0)


# -------------------------------------------
# StudentEmbedder: Uses model forward pass
# -------------------------------------------
class StudentEmbedder:
    def __init__(self, model, tokenizer, max_length=512, prompt="categorize: "):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt = prompt

    def pool(self, hidden_state, mask, pooling_method="cls"):
        if pooling_method == "mean":
            s = torch.sum(hidden_state * mask.unsqueeze(-1).float(), dim=1)
            d = mask.sum(dim=1, keepdim=True).float()
            return s / d
        elif pooling_method == "cls":
            return hidden_state[:, 0]

    def encode(self, texts, batch_size=128, task_type=None, **kwargs):
        """
        Encodes a list of sentences by first prepending a prompt (default "categorize: "),
        tokenizing with max_length=512, and then processing through the student pathway
        (using the model’s forward pass with output_hidden_states enabled). The output is
        CLS-pooled and L2-normalized.
        """
        all_embeddings = []
        prompt = frida_prompts[task_type]
        for i in tqdm(range(0, len(texts), batch_size), desc="Student encoding"):
            # Prepend prompt to each sentence
            batch_texts = [f"{prompt}{sent}" for sent in texts[i:i + batch_size]]
            encoded = self.tokenizer(
                batch_texts,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            encoded = {k: v.to(self.model.device) for k, v in encoded.items()}
            with torch.no_grad():
                # Forward pass through the model (student pathway)
                student_outputs = self.model(
                    encoded["input_ids"],
                    encoded["attention_mask"]
                )
                # Use CLS pooling on the last hidden state
                embeddings = self.pool(student_outputs, encoded["attention_mask"], pooling_method="cls")
                embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu())
        return torch.cat(all_embeddings, dim=0)


# -------------------------------------------
# evaluate_rumteb: Evaluate on ruMTEB benchmark
# -------------------------------------------
def evaluate_rumteb(model, tokenizer, limit=50):
    """
    Loads the ruMTEB (Russian part of MTEB) lightweight subset and evaluates both teacher
    and student representations. Each pathway is wrapped in its corresponding embedder
    (which applies prompt addition, tokenization, CLS pooling, and normalization), and then
    the MTEB evaluation object (filtered to Russian tasks via task_langs=["ru"]) is run on each.
    A side-by-side summary of key evaluation scores is printed.
    """
    # Instantiate embedding wrappers for teacher and student with the updated encode() methods.
    teacher_embedder = TeacherEmbedder(model, tokenizer, max_length=512)
    student_embedder = StudentEmbedder(model, tokenizer, max_length=512)

    # Create the MTEB evaluation object for Russian tasks, limiting examples for a lightweight run.
    evaluation = MTEB(tasks=["GeoreviewClusteringP2P"])
    
    print("Evaluating teacher embeddings on ruMTEB lightweight subset...")
    teacher_results = evaluation.run(teacher_embedder, output_folder="results/teacher_rumteb")
    
    print("Evaluating student embeddings on ruMTEB lightweight subset...")
    student_results = evaluation.run(student_embedder, output_folder="results/student_rumteb")
    
    # Print a side-by-side comparison of evaluation metrics for each task.
    print("\nComparison of teacher vs. student on ruMTEB tasks:")
    header = f"{'Task':30} | {'Teacher Score':15} | {'Student Score':15}"
    divider = "-" * len(header)
    print(header)
    print(divider)
    teacher_score = teacher_results[0].scores["test"][0].get("main_score")
    student_score = student_results[0].scores["test"][0].get("main_score")
    print(f"{'':30} | {str(teacher_score):15} | {str(student_score):15}")
    
    return teacher_score, student_score


torch.set_float32_matmul_precision('medium')

def find_optimal_batch_size(model, tokenizer, starting_batch_size=4, max_batch_size=32, device=None):
    """Find largest batch size that fits in memory"""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = starting_batch_size
    
    # Generate sample text
    sample_text = ["Пример текста для обучения модели"] * batch_size
    
    while batch_size <= max_batch_size:
        try:
            # Try processing a batch
            tokens = tokenizer(
                sample_text, 
                return_tensors="pt", 
                padding="max_length",
                max_length=256,
                truncation=True
            ).to(device)
            
            # Forward and backward pass
            with torch.cuda.amp.autocast():
                loss = model.compute_distillation_loss(
                    tokens["input_ids"], 
                    tokens["attention_mask"]
                )
            
            # If successful, try a larger batch
            print(f"Batch size {batch_size} fits in memory")
            batch_size *= 2
            sample_text = sample_text * 2
            
            # Clean up
            del tokens, loss
            torch.cuda.empty_cache()
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                # Revert to the last successful batch size
                batch_size = batch_size // 2
                print(f"Selected optimal batch size: {batch_size}")
                torch.cuda.empty_cache()
                return batch_size
            else:
                raise e
    
    return max_batch_size

def train_with_fabric():
    # Initialize Fabric
    fab = fabric.Fabric(accelerator="auto", devices="auto", precision="16-mixed")
    fab.launch()

    if fab.global_rank == 0:  # Only initialize wandb on the main process
        wandb.init(
            project="frida-distillation",
            name=f"distil-frida-{time.strftime('%Y%m%d-%H%M%S')}",
            config={
                "hidden_size": 368,
                "num_heads": 8,
                "num_layers": 3,
                "batch_size": 128,
                "learning_rate": 5e-4,
                "weight_decay": 0.01,
                "max_steps": 100000,
                "warmup_steps": 5000,
            }
        )
    
    # Initialize model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("ai-forever/FRIDA")
    model = FridaDistillationModel(hidden_size=368, num_heads=8, num_layers=3)
    
    # Setup model with Fabric
    model = fab.setup_module(model)
    
    # Find optimal batch size
    # with fab.init_module():
    #     batch_size = find_optimal_batch_size(model, tokenizer)
    batch_size = 128
    
    # Collate function
    def collate_fn(batch):
        try:
            prefixes = [
                "search_query: ",
                "search_document: ",
                "paraphrase: ",
                "categorize: ",
                "categorize_sentiment: ",
                "categorize_topic: ",
                "categorize_entailment: "
            ]
            
            # Extract texts and truncate very long ones
            raw_texts = [item['text'][:10000] for item in batch]
            
            # Prepend random prefixes to each text
            texts = []
            for text in raw_texts:
                # Select a random prefix
                prefix = random.choice(prefixes)
                # Prepend prefix to text
                texts.append(prefix + text)
            
            tokenizer_output = tokenizer(
                texts,
                padding='max_length',
                truncation=True,
                max_length=256,
                return_tensors='pt'
            )
            
            return {
                'input_ids': tokenizer_output['input_ids'],
                'attention_mask': tokenizer_output['attention_mask']
            }
        except Exception as e:
            fab.print(f"Error in collate_fn: {str(e)}")
            return None  # Return None to be filtered out later
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=5e-5,
        weight_decay=0.01
    )
    
    max_steps = 20000
    warmup_steps = 2000
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps
    )
    
    optimizer = fab.setup_optimizers(optimizer)
    
    # Load dataset in streaming mode
    fab.print("Loading dataset...")
    dataset = load_dataset("cointegrated/taiga_stripped_proza", split="train", streaming=True)
    
    # Remove unnecessary columns and shuffle
    dataset = dataset.remove_columns(['file'])
    dataset = dataset.shuffle(buffer_size=10000, seed=42)
    
    # Setup dataloader
    num_workers = min(8, os.cpu_count() or 2)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        prefetch_factor=2,
        persistent_workers=True
    )
    
    # Training loop variables
    global_step = 0
    best_loss = float('inf')
    os.makedirs("checkpoints", exist_ok=True)
    
    # Progress tracking
    pbar = tqdm(total=max_steps, desc="Training")
    
    model.train()
    fab.print(f"Starting training with batch size {batch_size}")
    
    try:
        # Infinite dataset iterator
        data_iter = iter(dataloader)
        running_loss = 0.0
        log_interval = 100
        wandb_interval = 10
        eval_interval = 200
        save_interval = 5000
        start_time = time.time()
        plateau_window_size = 20
        plateau_threshold = 0.001
        loss_history = []
        grad_norm_history = []
        is_in_plateau = False
        
        while global_step < max_steps:
            try:
                # Get next batch
                try:
                    batch = next(data_iter)
                except StopIteration:
                    # Refresh iterator if needed
                    data_iter = iter(dataloader)
                    batch = next(data_iter)
                
                # Skip None batches (from failed collate)
                if batch is None:
                    continue
                
                # Forward pass
                loss = model.compute_distillation_loss(
                    batch['input_ids'], 
                    batch['attention_mask']
                )
                
                # Skip bad losses
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                
                # Backward pass
                fab.backward(loss)
                
                # Gradient clipping and optimization
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.zero_grad()
                
                # Update metrics
                loss_value = loss.item()
                running_loss += loss_value
                loss_history.append(loss_value)
                grad_norm_history.append(grad_norm.item())

                if global_step % eval_interval == 0:
                    teacher_res, student_res = evaluate_rumteb(model, tokenizer)
                    wandb.log({
                        "metrics/teacher_score": teacher_res,
                        "metrics/student_score": student_res,
                    }, step=global_step)

                global_step += 1
                pbar.update(1)

                
                if global_step % wandb_interval == 0 and fab.global_rank == 0:  # Only log on the main process
                    elapsed = time.time() - start_time
                    recent_losses = loss_history[-plateau_window_size:]
                    loss_mean = sum(recent_losses) / len(recent_losses)
                    loss_std = (sum((x - loss_mean)**2 for x in recent_losses) / len(recent_losses))**0.5
                    if len(loss_history) > plateau_window_size:
                        # Keep only recent history
                        if len(loss_history) > plateau_window_size * 2:
                            loss_history = loss_history[-plateau_window_size*2:]
                            grad_norm_history = grad_norm_history[-plateau_window_size*2:]
                
                        # Calculate metrics
                        prev_losses = loss_history[-plateau_window_size*2:-plateau_window_size]
                        
                        
                        prev_loss_mean = sum(prev_losses) / len(prev_losses)
                        relative_improvement = (prev_loss_mean - loss_mean) / prev_loss_mean
                        
                        # First derivative approximation (rate of change)
                        loss_derivative = (prev_loss_mean - loss_mean) / plateau_window_size
                        
                        # Detect plateau
                        was_in_plateau = is_in_plateau
                        is_in_plateau = relative_improvement < plateau_threshold
                        
                        # Calculate rate of convergence metrics
                        if not is_in_plateau:
                            convergence_rate = -loss_derivative / loss_mean
                            time_to_convergence_estimate = loss_mean / abs(loss_derivative) if loss_derivative != 0 else float('inf')
                        else:
                            convergence_rate = 0
                            time_to_convergence_estimate = float('inf')
                        wandb.log({
                            "train/loss": loss_value,
                            "train/avg_loss": loss_mean,
                            "train/loss_std": loss_std,
                            "train/learning_rate": scheduler.get_last_lr()[0],
                            "train/step": global_step,
                            "train/steps_per_second": wandb_interval/elapsed,
                            "train/gradient_norm": grad_norm.item(),
                            
                            # Convergence metrics
                            "convergence/relative_improvement": relative_improvement,
                            "convergence/loss_derivative": loss_derivative,
                            "convergence/is_plateau": 1 if is_in_plateau else 0,
                            "convergence/rate": convergence_rate,
                            "convergence/time_to_convergence_est": min(time_to_convergence_estimate, 100000)
                        }, step=global_step)
                    else:
                        wandb.log({
                            "train/loss": loss_value,
                            "train/avg_loss": loss_mean,
                            "train/loss_std": loss_std,
                            "train/learning_rate": scheduler.get_last_lr()[0],
                            "train/step": global_step,
                            "train/steps_per_second": log_interval/elapsed,
                            "train/gradient_norm": grad_norm.item(),
                        }, step=global_step)

                    start_time = time.time()
                # Logging
                if global_step % log_interval == 0:
                    avg_loss = running_loss / log_interval
                    
                    fab.print(
                        f"Step: {global_step} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"LR: {scheduler.get_last_lr()[0]:.6f}"
                    )


                    
                    # Save best model
                    if avg_loss < best_loss:
                        best_loss = avg_loss
                        fab.save("checkpoints/frida_best_model.pt", {
                            "model": model.state_dict(),
                            "step": global_step,
                            "loss": best_loss
                        })
                    
                    running_loss = 0.0
                
                # Checkpoint saving
                if global_step % save_interval == 0:
                    fab.save(f"checkpoints/frida_step_{global_step}.pt", {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "step": global_step
                    })
                    
                    # Clean memory
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            except Exception as e:
                fab.print(f"Error in training step: {str(e)}")
                import traceback
                print("=" * 80)
                print("TRAINING ERROR:")
                print("-" * 80)
                traceback.print_exc()
                print("=" * 80)

                # Continue with next batch
                continue
            
            if global_step >= max_steps:
                break
    
    except KeyboardInterrupt:
        fab.print("Training interrupted")
    
    finally:
        pbar.close()
        
        # Final save
        fab.save("checkpoints/frida_final.pt", {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": global_step
        })
        
        total_time = time.time() - start_time
        fab.print(f"Training completed: {global_step} steps in {timedelta(seconds=int(total_time))}")
        fab.print(f"Best loss: {best_loss:.4f}")

if __name__ == "__main__":
    train_with_fabric()
