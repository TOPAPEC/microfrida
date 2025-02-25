#!/usr/bin/env python3
import os
import matplotlib.pyplot as plt
from transformers import T5EncoderModel, AutoTokenizer
from model_pipeline import build_student_model, load_training_dataset, run_distillation, evaluate_model

def main():
    teacher_model_id = "ai-forever/FRIDA"
    teacher_model = T5EncoderModel.from_pretrained(teacher_model_id)
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_id)
    train_dataset = load_training_dataset(teacher_tokenizer, limit=10000)
    # Define different pruning configurations as tuples (config_name, new_num_heads, blocks_to_keep_indices)
    configs = [
        ("config_A", 12, [0, 23]),
        ("config_B", 12, [0, 10, 20, 23]),
        ("config_C", 12, [0, 5, 10, 15, 20, 23]),
        ("config_D", 12, [0, 1, 20, 21, 22, 23])
    ]
    loss_curves = {}
    eval_results = {}
    output_base = "pipeline_runs"
    os.makedirs(output_base, exist_ok=True)
    for config_name, new_num_heads, blocks_to_keep in configs:
        print(f"Running configuration: {config_name}")
        student_model = build_student_model(teacher_model, new_num_heads=new_num_heads, blocks_to_keep_indices=blocks_to_keep)
        student_dir = os.path.join(output_base, config_name, "student_pre_pruned")
        os.makedirs(student_dir, exist_ok=True)
        student_model.save_pretrained(student_dir)
        output_dir = os.path.join(output_base, config_name, "distillation")
        os.makedirs(output_dir, exist_ok=True)
        loss_history, distilled_model_dir = run_distillation(teacher_model, student_model, teacher_tokenizer, train_dataset, output_dir)
        loss_curves[config_name] = loss_history
        eval_dir = os.path.join(output_base, config_name, "evaluation")
        os.makedirs(eval_dir, exist_ok=True)
        results, stats = evaluate_model(distilled_model_dir, eval_output_folder=eval_dir)
        eval_results[config_name] = results
        print(f"Configuration {config_name} evaluation results:")
        print(results)
    plt.figure(figsize=(10, 6))
    for config_name, losses in loss_curves.items():
        plt.plot(losses, label=config_name)
    plt.xlabel("Logging Step")
    plt.ylabel("Loss")
    plt.title("Training Loss Curves for Different Pruning Configurations")
    plt.legend()
    plt.savefig(os.path.join(output_base, "loss_curves.png"))
    plt.close()
    print("Loss curves plotted and saved.")
    print("Summary Evaluation Results:")
    for config_name, results in eval_results.items():
        print(f"\nConfiguration: {config_name}")
        print(results)

if __name__ == "__main__":
    main()
