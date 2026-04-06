import re
import time
import json
import tiktoken
import torch
from torch.utils.data import DataLoader
from functools import partial
from gpt_instruction import InstructionDataset, custom_collate_fn, format_input, load_file
from gpt_model import GPTModel
from gpt_train import train_model, calc_loss_loader, plot_losses
import sys
import os

# Model configuration dictionary
MODEL_CONFIGS = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}


def find_saved_models():
    """Find all saved fine-tuned models in current directory."""
    model_files = [f for f in os.listdir(".") if f.endswith("-sft-standalone.pth")]
    return sorted(model_files)


def display_model_selection(model_files):
    """Display available models for continued training."""
    print("\n" + "="*70)
    print("🔄 CONTINUE TRAINING - Select a Model")
    print("="*70)
    print("\n📋 Available Fine-Tuned Models:")
    print("-"*70)
    
    for i, mf in enumerate(model_files, 1):
        # Extract model info from filename
        model_name = mf.replace("-sft-standalone.pth", "").replace("-", " ").title()
        config_file = mf.replace(".pth", "_config.json")
        
        # Try to get config info
        config_info = ""
        if os.path.exists(config_file):
            try:
                with open(config_file, "r") as f:
                    config = json.load(f)
                config_info = f"({config.get('n_layers', '?')} layers, {config.get('emb_dim', '?')} dim)"
            except:
                pass
        
        print(f"\n  [{i}] {mf}")
        print(f"      {config_info}")
        
        # Show file size
        try:
            size_mb = os.path.getsize(mf) / (1024 * 1024)
            print(f"      • File size: {size_mb:.2f} MB")
        except:
            pass
    
    print("\n" + "-"*70)
    print("💡 Tips:")
    print("   • Select the model you want to continue training")
    print("   • Make sure the _config.json file exists alongside .pth file")
    print("   • Press 'q' anytime to quit")
    print("="*70 + "\n")


def load_model_for_continuation(model_path, config_path, device):
    """Load a fine-tuned model for continued training."""
    print(f"📦 Loading model config from: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    print(f"🤖 Model Configuration:")
    print(f"   • Vocabulary: {config.get('vocab_size', 'N/A')}")
    print(f"   • Context length: {config.get('context_length', 'N/A')}")
    print(f"   • Embedding dim: {config.get('emb_dim', 'N/A')}")
    print(f"   • Layers: {config.get('n_layers', 'N/A')}")
    print(f"   • Attention heads: {config.get('n_heads', 'N/A')}")
    
    # Initialize model
    model = GPTModel(config)
    
    # Load fine-tuned weights
    print(f"📥 Loading weights from: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.train()  # Set to training mode for continued fine-tuning
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model loaded successfully! ({total_params:,} parameters)")
    
    return model, config


def get_training_parameters():
    """Get continued training parameters from user."""
    print("\n" + "="*70)
    print("⚙️  CONTINUED TRAINING PARAMETERS")
    print("="*70)
        
    # New data option
    print("\n📁 Data Options:")
    print("   [1] Use existing instruction-data.json")
    print("   [2] Use new/updated data file")
    
    while True:
        data_choice = input("\nSelect data option (1-2, default=1): ").strip()
        if data_choice in ["", "1"]:
            data_file = "instruction-data.json"
            break
        elif data_choice == "2":
            data_file = input("Enter new data file path: ").strip()
            if not os.path.exists(data_file):
                print(f"⚠️  File not found: {data_file}")
                print("   Using default instruction-data.json instead")
                data_file = "instruction-data.json"
            break
        else:
            print("⚠️  Invalid choice")
    
    params = {
        "num_epochs": 2,
        "learning_rate": 0.00005,
        "batch_size": 8,
        "context_length": 1024,
        "eval_freq": 5,
        "data_file": data_file
    }
    
    print("\n" + "-"*70)
    print("✅ Training Parameters Summary:")
    for key, value in params.items():
        print(f"   • {key}: {value}")
    print("-"*70 + "\n")
    
    return params


def main():
    # ========== STEP 1: Find Saved Models ==========
    print("🔍 Searching for fine-tuned models...\n")
    
    model_files = find_saved_models()
    
    if not model_files:
        print("✗ No fine-tuned models found (*.pth files ending with '-sft-standalone.pth')")
        print("💡 Please run main.py first to train and save a model.\n")
        sys.exit(1)
    
    # Display available models
    display_model_selection(model_files)
    
    # Get user selection
    while True:
        choice = input(f"Select a model (1-{len(model_files)}) or 'q' to quit: ").strip().lower()
        
        if choice == 'q':
            print("👋 Continued training cancelled. Goodbye!")
            sys.exit(0)
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(model_files):
                selected_file = model_files[idx]
                break
            else:
                print("✗ Invalid selection. Please try again.")
        except ValueError:
            print("✗ Invalid input. Please enter a number or 'q'.")
    
    print(f"\n✅ Selected: {selected_file}\n")
    
    # Derive config file path
    config_file = selected_file.replace(".pth", "_config.json")
    if not os.path.exists(config_file):
        print(f"✗ Config file not found: {config_file}")
        print("💡 Please ensure the _config.json file exists alongside the .pth file.\n")
        sys.exit(1)
    
    # ========== STEP 2: Setup Device ==========
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Device: {device}")
    if device.type == "cpu":
        print("⚠️  Training on CPU will be slow. Consider using a GPU for faster results.\n")
    print(50*"-")
    
    # ========== STEP 3: Load Model ==========
    print("📦 Loading fine-tuned model for continued training...")
    try:
        model, config = load_model_for_continuation(selected_file, config_file, device)
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        sys.exit(1)
    
    # ========== STEP 4: Get Training Parameters ==========
    train_params = get_training_parameters()
    
    # ========== STEP 5: Load and Prepare Data ==========
    print(f"📥 Loading instruction data from: {train_params['data_file']}...")
    data = load_file(train_params["data_file"])
    print(f"✓ Loaded {len(data)} instruction examples\n")

    # Split data
    train_portion = int(len(data) * 0.85)
    test_portion = int(len(data) * 0.1)

    train_data = data[:train_portion]
    test_data = data[train_portion:train_portion + test_portion]
    val_data = data[train_portion + test_portion:]
    print(f"📊 Data split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}\n")

    # ========== STEP 6: Setup Tokenizer and DataLoaders ==========
    tokenizer = tiktoken.get_encoding("gpt2")
    
    customized_collate_fn = partial(
        custom_collate_fn, 
        device=device, 
        allowed_max_length=train_params["context_length"]
    )
    num_workers = 0
    batch_size = train_params["batch_size"]
    torch.manual_seed(123)

    print("📦 Creating datasets and data loaders...")
    train_dataset = InstructionDataset(train_data, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers
    )

    val_dataset = InstructionDataset(val_data, tokenizer)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )
    print("✓ Data loaders ready\n")

    # ========== STEP 7: Initial Loss Evaluation ==========
    print("📈 Evaluating initial losses (before continued training)...")
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)

    print(f"   Training loss:   {train_loss:.3f}")
    print(f"   Validation loss: {val_loss:.3f}")
    print(50*"-")

    # ========== STEP 8: Continued Fine-Tuning ==========
    print("🚀 Starting continued fine-tuning...\n")
    start_time = time.time()
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=train_params["learning_rate"], 
        weight_decay=0.1
    )
    num_epochs = train_params["num_epochs"]
    
    torch.manual_seed(123)
    train_losses, val_losses, tokens_seen = train_model(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, 
        eval_freq=train_params["eval_freq"], 
        eval_iter=5,
        start_context=format_input(val_data[0]), 
        tokenizer=tokenizer
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"\n✅ Continued training completed in {execution_time_minutes:.2f} minutes.")

    # ========== STEP 9: Plot and Save ==========
    print("📊 Generating loss plot...")
    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_name = selected_file.replace(".pth", "-continued-loss-plot.pdf")
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses, save_path=plot_name)
    print(50*"-")

    # Save the continued fine-tuned model with new name
    base_name = selected_file.replace("-sft-standalone.pth", "")
    new_file_name = f"{base_name}-sft-continued.pth"
    
    # Ensure unique filename
    counter = 1
    while os.path.exists(new_file_name):
        new_file_name = f"{base_name}-sft-continued-v{counter}.pth"
        counter += 1
    
    torch.save(model.state_dict(), new_file_name)
    print(f"💾 Model saved as: {new_file_name}")
    
    # Save config (same config, but update file)
    config_file_new = new_file_name.replace(".pth", "_config.json")
    with open(config_file_new, "w") as f:
        json.dump(config, f, indent=2)
    print(f"⚙️  Config saved as: {config_file_new}")
    
    # Save training metadata
    metadata = {
        "original_model": selected_file,
        "continued_epochs": num_epochs,
        "learning_rate": train_params["learning_rate"],
        "batch_size": batch_size,
        "context_length": train_params["context_length"],
        "training_time_minutes": execution_time_minutes,
        "final_train_loss": train_losses[-1] if train_losses else None,
        "final_val_loss": val_losses[-1] if val_losses else None
    }
    metadata_file = new_file_name.replace(".pth", "_metadata.json")
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"📝 Training metadata saved as: {metadata_file}")
    
    print("\n" + "🎉"*35)
    print(f"Continued fine-tuning complete!")
    print(f"Your model has been trained for {num_epochs} additional epochs.")
    print("🎉"*35 + "\n")
    
    return model, tokenizer, config


if __name__ == "__main__":
    main()