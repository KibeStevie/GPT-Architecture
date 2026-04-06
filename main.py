import re
import time
import tiktoken
import torch
from torch.utils.data import DataLoader
from functools import partial
from gpt_instruction import InstructionDataset, custom_collate_fn, format_input, load_file
from gpt_model import GPTModel
from gpt_download import download_and_load_gpt2
from gpt_train import train_model, calc_loss_loader, plot_losses, load_weights_into_gpt
import sys

# Model configuration dictionary
MODEL_CONFIGS = {
    "1": {
        "name": "gpt2-small (124M)",
        "size": "124M",
        "emb_dim": 768,
        "n_layers": 12,
        "n_heads": 12,
        "estimated_ram": "~2 GB",
        "estimated_disk": "~500 MB"
    },
    "2": {
        "name": "gpt2-medium (355M)",
        "size": "355M",
        "emb_dim": 1024,
        "n_layers": 24,
        "n_heads": 16,
        "estimated_ram": "~4 GB",
        "estimated_disk": "~1.4 GB"
    },
    "3": {
        "name": "gpt2-large (774M)",
        "size": "774M",
        "emb_dim": 1280,
        "n_layers": 36,
        "n_heads": 20,
        "estimated_ram": "~8 GB",
        "estimated_disk": "~3.0 GB"
    },
    "4": {
        "name": "gpt2-xl (1558M)",
        "size": "1558M",
        "emb_dim": 1600,
        "n_layers": 48,
        "n_heads": 25,
        "estimated_ram": "~16 GB",
        "estimated_disk": "~6.2 GB"
    },
}


def display_model_menu():
    """Display the interactive model selection menu."""
    print("\n" + "="*70)
    print("🎯 GPT-2 Instruction Fine-Tuning")
    print("="*70)
    print("\n📋 Available Models:")
    print("-"*70)
    
    for key, config in MODEL_CONFIGS.items():
        print(f"\n  [{key}] {config['name']}")
        print(f"      • Embedding dim: {config['emb_dim']}")
        print(f"      • Transformer layers: {config['n_layers']}")
        print(f"      • Attention heads: {config['n_heads']}")
        print(f"      • Estimated RAM: {config['estimated_ram']}")
        print(f"      • Estimated Disk: {config['estimated_disk']}")
    
    print("\n" + "-"*70)
    print("💡 Tips:")
    print("   • Start with [1] gpt2-small for faster experimentation")
    print("   • Larger models perform better but need more resources")
    print("   • Press 'q' anytime to quit")
    print("="*70 + "\n")


def get_model_choice():
    """Get and validate user model selection."""
    while True:
        choice = input("Select a model (1-4) or 'q' to quit: ").strip().lower()
        
        if choice == 'q':
            return None
        
        if choice in MODEL_CONFIGS:
            selected = MODEL_CONFIGS[choice]
            print(f"\n✅ Selected: {selected['name']}")
            print(f"⚠️  Ensure you have at least {selected['estimated_ram']} RAM available.\n")
            return selected
        else:
            print("✗ Invalid choice. Please enter 1, 2, 3, 4, or 'q'.\n")


def confirm_large_model(model_config):
    """Show warning for large models and get confirmation."""
    if model_config["size"] in ("774M", "1558M"):
        print("\n" + "⚠️"*35)
        print(f"WARNING: {model_config['name']} is a large model!")
        print(f"  • May require {model_config['estimated_ram']} RAM or more")
        print(f"  • Download size: ~{model_config['estimated_disk']}")
        print(f"  • Training will be slower and memory-intensive")
        print("⚠️"*35 + "\n")
        
        confirm = input("Continue with this model? (y/n): ").strip().lower()
        if confirm != 'y':
            print("🔄 Returning to model selection...\n")
            return False
    return True


def main():
    # ========== STEP 1: Model Selection ==========
    display_model_menu()
    selected_model = get_model_choice()
    
    if selected_model is None:
        print("👋 Fine-tuning cancelled. Goodbye!")
        sys.exit(0)
    
    # Confirm if large model selected
    if not confirm_large_model(selected_model):
        display_model_menu()
        selected_model = get_model_choice()
        if selected_model is None:
            print("👋 Fine-tuning cancelled. Goodbye!")
            sys.exit(0)
    
    model_name = selected_model["name"]
    model_size = selected_model["size"]
    
    # ========== STEP 2: Load and Prepare Data ==========
    print(f"📥 Loading instruction data...")
    file_path = "instruction-data.json"
    data = load_file(file_path)
    print(f"✓ Loaded {len(data)} instruction examples\n")

    # Split data
    train_portion = int(len(data) * 0.85)
    test_portion = int(len(data) * 0.1)

    train_data = data[:train_portion]
    test_data = data[train_portion:train_portion + test_portion]
    val_data = data[train_portion + test_portion:]
    print(f"📊 Data split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}\n")

    # ========== STEP 3: Setup Tokenizer and Device ==========
    tokenizer = tiktoken.get_encoding("gpt2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Device: {device}")
    if device.type == "cpu":
        print("⚠️  Training on CPU will be slow. Consider using a GPU for faster results.\n")
    print(50*"-")

    # ========== STEP 4: Create DataLoaders ==========
    customized_collate_fn = partial(custom_collate_fn, device=device, allowed_max_length=1024)
    num_workers = 0
    batch_size = 8
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

    # ========== STEP 5: Build and Load Model ==========
    print(f"🤖 Initializing {model_name}...")
    
    BASE_CONFIG = {
        "vocab_size": 50257,
        "context_length": 1024,
        "drop_rate": 0.0,
        "qkv_bias": True,
        "emb_dim": selected_model["emb_dim"],
        "n_layers": selected_model["n_layers"],
        "n_heads": selected_model["n_heads"],
    }

    # Download and load pre-trained weights
    print(f"📥 Downloading/loading pre-trained weights for {model_size}...")
    print("   (Files are cached locally after first download)\n")
    
    settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2_models")

    # Initialize model and load weights
    model = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(model, params)
    model.eval()
    model.to(device)
    print(f"✓ Loaded {model_name} with {sum(p.numel() for p in model.parameters()):,} parameters\n")

    # ========== STEP 6: Initial Loss Evaluation ==========
    print("📈 Evaluating initial losses (before fine-tuning)...")
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)

    print(f"   Training loss:   {train_loss:.3f}")
    print(f"   Validation loss: {val_loss:.3f}")
    print(50*"-")

    # ========== STEP 7: Fine-Tuning ==========
    print("🚀 Starting fine-tuning...\n")
    start_time = time.time()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)
    num_epochs = 2
    
    torch.manual_seed(123)
    train_losses, val_losses, tokens_seen = train_model(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=5, eval_iter=5,
        start_context=format_input(val_data[0]), tokenizer=tokenizer
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"\n✅ Training completed in {execution_time_minutes:.2f} minutes.")

    # ========== STEP 8: Plot and Save ==========
    print("📊 Generating loss plot...")
    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
    print(50*"-")

    # Save the fine-tuned model
    file_name = f"{re.sub(r'[ ()]', '', model_name)}-sft-standalone.pth"
    torch.save(model.state_dict(), file_name)
    print(f"💾 Model saved as: {file_name}")
    
    # Save config for later loading
    config_file = file_name.replace(".pth", "_config.json")
    import json
    with open(config_file, "w") as f:
        json.dump(BASE_CONFIG, f, indent=2)
    print(f"⚙️  Config saved as: {config_file}")
    
    print("\n" + "🎉"*35)
    print(f"Fine-tuning complete! Your model is ready for inference.")
    print("🎉"*35 + "\n")
    
    return model, tokenizer, BASE_CONFIG


if __name__ == "__main__":
    main()