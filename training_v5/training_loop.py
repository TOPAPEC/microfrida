import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import lightning.fabric as fabric
import os
import time
import gc
from datetime import timedelta
from tqdm.auto import tqdm
from student import FridaDistillationModel

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
    
    # Initialize model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("ai-forever/FRIDA")
    model = FridaDistillationModel(hidden_size=368, num_heads=8, num_layers=3)
    
    # Setup model with Fabric
    model = fab.setup_module(model)
    
    # Find optimal batch size
    with fab.init_module():
        batch_size = find_optimal_batch_size(model, tokenizer)
    
    # Collate function
    def collate_fn(batch):
        try:
            texts = [item['text'][:10000] for item in batch]  # Truncate very long texts
            
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
        lr=3e-5,
        weight_decay=0.01
    )
    
    max_steps = 100000
    warmup_steps = 5000
    
    scheduler = get_linear_schedule_with_warmup(
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
    num_workers = min(4, os.cpu_count() or 2)
    
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
        save_interval = 5000
        start_time = time.time()
        
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
                optimizer.zero_grad()
                
                # Update metrics
                loss_value = loss.item()
                running_loss += loss_value
                global_step += 1
                pbar.update(1)
                
                # Logging
                if global_step % log_interval == 0:
                    avg_loss = running_loss / log_interval
                    elapsed = time.time() - start_time
                    
                    fab.print(
                        f"Step: {global_step} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"Speed: {log_interval/elapsed:.1f} it/s | "
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
                    start_time = time.time()
                
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
