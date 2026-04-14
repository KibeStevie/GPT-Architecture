import time
import json
import tiktoken
import torch
from torch.utils.data import DataLoader
from functools import partial
from gpt_instruction import InstructionDataset, custom_collate_fn, format_input, load_file
from gpt_model import GPTModel
from gpt_train import train_model, calc_loss_loader, plot_losses, load_weights_into_gpt
from gpt_download import download_and_load_gpt2
import sys
import os

# Model configuration dictionary
MODEL_CONFIGS = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

# Default training parameters
DEFAULT_PARAMS = {
    "num_epochs": 2,
    "learning_rate": 0.00005,
    "batch_size": 8,
    "context_length": 1024,
    "eval_freq": 5,
}


def find_saved_models():
    """Find all saved fine-tuned models in current directory."""
    model_files = [f for f in os.listdir(".") if f.endswith("-sft-standalone.pth") or f.endswith("-sft-continued.pth")]
    return sorted(model_files)


def find_json_files():
    """Find all JSON files in current directory (excluding config files)."""
    json_files = [f for f in os.listdir(".") if f.endswith(".json") and not f.endswith("_config.json") and not f.endswith("_metadata.json")]
    return sorted(json_files)


def display_model_source_selection():
    """Display options for model source selection."""
    print("\n" + "="*70)
    print("🎯 MODEL SOURCE SELECTION")
    print("="*70)
    print("\n📌 Choose where to load the model from:")
    print("-"*70)
    print("\n  [1] Continue Training on Fine-Tuned Model")
    print("      • Load previously saved model (*.pth files)")
    print("      • Continue from where you left off")
    print("      • Recommended for iterative improvements")
    print("\n  [2] Start Fresh with Pretrained GPT-2 Model")
    print("      • Download fresh GPT-2 weights from OpenAI")
    print("      • Choose from 124M, 355M, 774M, or 1558M")
    print("      • Recommended for new training experiments")
    print("\n" + "-"*70)
    print("💡 Tips:")
    print("   • Option [1] requires existing .pth files in current directory")
    print("   • Option [2] will download weights (cached after first download)")
    print("   • Press 'q' anytime to quit")
    print("="*70 + "\n")


def display_finetuned_models(model_files):
    """Display available fine-tuned models for selection."""
    print("\n" + "="*70)
    print("📋 AVAILABLE FINE-TUNED MODELS")
    print("="*70)
    print("\n📁 Found in Current Directory:")
    print("-"*70)
    
    if not model_files:
        print("   ✗ No fine-tuned models found!")
        return
    
    for i, mf in enumerate(model_files, 1):
        # Extract model info from filename
        model_name = mf.replace("-sft-standalone.pth", "").replace("-sft-continued.pth", "").replace("-", " ").title()
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
    print("="*70 + "\n")


def display_pretrained_models():
    """Display available pretrained GPT-2 models."""
    print("\n" + "="*70)
    print("📋 AVAILABLE PRETRAINED GPT-2 MODELS")
    print("="*70)
    print("\n🤖 Choose Model Size:")
    print("-"*70)
    
    for i, (name, config) in enumerate(MODEL_CONFIGS.items(), 1):
        size = name.split("(")[-1].rstrip(")")
        emb_dim = config["emb_dim"]
        n_layers = config["n_layers"]
        
        # Estimate RAM requirements
        if "124M" in size:
            ram = "~2 GB"
        elif "355M" in size:
            ram = "~4 GB"
        elif "774M" in size:
            ram = "~8 GB"
        else:
            ram = "~16 GB"
        
        print(f"\n  [{i}] {name}")
        print(f"      • Embedding dim: {emb_dim}")
        print(f"      • Transformer layers: {n_layers}")
        print(f"      • Attention heads: {config['n_heads']}")
        print(f"      • Estimated RAM: {ram}")
    
    print("\n" + "-"*70)
    print("💡 Tips:")
    print("   • Start with [1] gpt2-small for faster experimentation")
    print("   • Larger models perform better but need more resources")
    print("   • Weights are cached locally after first download")
    print("="*70 + "\n")


def display_json_selection(json_files):
    """Display available JSON data files for training."""
    print("\n" + "="*70)
    print("📁 AVAILABLE TRAINING DATA FILES")
    print("="*70)
    print("\n📋 Found JSON Files in Current Directory:")
    print("-"*70)
    
    if not json_files:
        print("   ✗ No JSON files found!")
        return
    
    for i, jf in enumerate(json_files, 1):
        # Show file size
        try:
            size_kb = os.path.getsize(jf) / 1024
            if size_kb > 1024:
                size_display = f"{size_kb/1024:.2f} MB"
            else:
                size_display = f"{size_kb:.2f} KB"
        except:
            size_display = "Unknown"
        
        # Try to count entries in JSON file
        entry_count = "?"
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    entry_count = len(data)
        except:
            pass
        
        print(f"\n  [{i}] {jf}")
        print(f"      • Size: {size_display}")
        print(f"      • Entries: {entry_count}")
    
    print("\n" + "-"*70)
    print("="*70 + "\n")


def display_default_params():
    """Display the default training parameters."""
    print("\n" + "="*70)
    print("⚙️  DEFAULT TRAINING PARAMETERS")
    print("="*70)
    print(f"\n   • Epochs:          {DEFAULT_PARAMS['num_epochs']}")
    print(f"   • Learning Rate:   {DEFAULT_PARAMS['learning_rate']}")
    print(f"   • Batch Size:      {DEFAULT_PARAMS['batch_size']}")
    print(f"   • Context Length:  {DEFAULT_PARAMS['context_length']}")
    print(f"   • Eval Frequency:  {DEFAULT_PARAMS['eval_freq']}")
    print("\n" + "-"*70)
    print("💡 These settings work well for most continued training scenarios.")
    print("="*70 + "\n")


def load_finetuned_model(model_path, config_path, device):
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
    print(f"✅ Fine-tuned model loaded successfully! ({total_params:,} parameters)")
    
    return model, config, "finetuned"


def load_pretrained_model(model_choice, device):
    """Load a fresh pretrained GPT-2 model from OpenAI."""
    model_names = list(MODEL_CONFIGS.keys())
    selected_name = model_names[model_choice - 1]
    model_size = selected_name.split("(")[-1].rstrip(")")
    
    print(f"\n🤖 Selected: {selected_name}")
    print(f"📥 Downloading/loading pretrained weights for {model_size}...")
    print("   (Files are cached locally after first download)\n")
    
    # Download and load pretrained weights
    settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2_models")
    
    # Build config
    BASE_CONFIG = {
        "vocab_size": 50257,
        "context_length": 1024,
        "drop_rate": 0.0,
        "qkv_bias": True,
        "emb_dim": MODEL_CONFIGS[selected_name]["emb_dim"],
        "n_layers": MODEL_CONFIGS[selected_name]["n_layers"],
        "n_heads": MODEL_CONFIGS[selected_name]["n_heads"],
    }
    
    # Initialize model and load weights
    model = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(model, params)
    model.to(device)
    model.train()  # Set to training mode
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Pretrained model loaded successfully! ({total_params:,} parameters)")
    
    return model, BASE_CONFIG, "pretrained"


def get_training_parameters(json_files):
    """Get continued training parameters from user."""
    print("\n" + "="*70)
    print("⚙️  TRAINING PARAMETERS CONFIGURATION")
    print("="*70)
    
    # Ask user if they want default or custom parameters
    print("\n📌 Parameter Configuration Options:")
    print("   [1] Use Default Parameters (Recommended for beginners)")
    print("   [2] Enter Custom Parameters (Advanced users)")
    
    while True:
        param_choice = input("\nSelect option (1-2, default=1): ").strip()
        
        if param_choice in ["", "1"]:
            # Use default parameters
            display_default_params()
            confirm = input("Use these default settings? (Y/n, default=Y): ").strip().lower()
            if confirm in ["", "y", "yes", "Y"]:
                print("\n✅ Using DEFAULT parameters")
                params = DEFAULT_PARAMS.copy()
                break
            else:
                print("🔄 Returning to parameter selection...\n")
                continue
        elif param_choice == "2":
            # Enter custom parameters
            print("\n📝 Enter Custom Training Parameters:")
            print("-"*70)
            
            # Number of additional epochs
            while True:
                try:
                    epochs = input("\n📅 Number of additional epochs (default=2): ").strip()
                    num_epochs = int(epochs) if epochs else 2
                    if num_epochs > 0:
                        break
                    print("⚠️  Please enter a positive number")
                except ValueError:
                    print("⚠️  Invalid input. Please enter a number")
            
            # Learning rate
            while True:
                try:
                    lr = input("📈 Learning rate (default=0.00005): ").strip()
                    learning_rate = float(lr) if lr else 0.00005
                    if learning_rate > 0:
                        break
                    print("⚠️  Please enter a positive number")
                except ValueError:
                    print("⚠️  Invalid input. Please enter a number")
            
            # Batch size
            while True:
                try:
                    bs = input("📦 Batch size (default=8): ").strip()
                    batch_size = int(bs) if bs else 8
                    if batch_size > 0:
                        break
                    print("⚠️  Please enter a positive number")
                except ValueError:
                    print("⚠️  Invalid input. Please enter a number")
            
            # Context length
            while True:
                try:
                    cl = input("📝 Context length (default=1024): ").strip()
                    context_length = int(cl) if cl else 1024
                    if context_length > 0:
                        break
                    print("⚠️  Please enter a positive number")
                except ValueError:
                    print("⚠️  Invalid input. Please enter a number")
            
            # Evaluation frequency
            while True:
                try:
                    ef = input("📊 Evaluation frequency (default=5): ").strip()
                    eval_freq = int(ef) if ef else 5
                    if eval_freq > 0:
                        break
                    print("⚠️  Please enter a positive number")
                except ValueError:
                    print("⚠️  Invalid input. Please enter a number")
            
            params = {
                "num_epochs": num_epochs,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "context_length": context_length,
                "eval_freq": eval_freq,
            }
            
            print("\n✅ Using CUSTOM parameters")
            break
        else:
            print("⚠️  Invalid choice. Please enter 1 or 2.")
    
    # Data file selection - Display available JSON files
    print("\n📁 Data File Selection:")
    
    if json_files:
        display_json_selection(json_files)
        
        while True:
            data_choice = input(f"Select a data file (1-{len(json_files)}, default=1): ").strip()
            
            if data_choice == "":
                data_choice = "1"
            
            try:
                idx = int(data_choice) - 1
                if 0 <= idx < len(json_files):
                    data_file = json_files[idx]
                    print(f"\n✅ Selected: {data_file}")
                    break
                else:
                    print("⚠️  Invalid selection. Please try again.")
            except ValueError:
                print("⚠️  Invalid input. Please enter a number.")
    else:
        print("⚠️  No JSON files found in current directory.")
        print("   Please add a training data file (e.g., instruction-data.json)")
        data_file = input("   Or enter custom file path: ").strip()
        if not data_file:
            data_file = "instruction-data.json"
    
    params["data_file"] = data_file
    
    print("\n" + "="*70)
    print("✅ TRAINING PARAMETERS SUMMARY")
    print("="*70)
    for key, value in params.items():
        print(f"   • {key}: {value}")
    print("="*70 + "\n")
    
    return params


def main():
    # ========== STEP 1: Select Model Source ==========
    display_model_source_selection()
    
    while True:
        source_choice = input("Select model source (1-2) or 'q' to quit: ").strip().lower()
        
        if source_choice == 'q':
            print("👋 Training cancelled. Goodbye!")
            sys.exit(0)
        elif source_choice in ["", "1"]:
            model_source = "finetuned"
            break
        elif source_choice == "2":
            model_source = "pretrained"
            break
        else:
            print("⚠️  Invalid choice. Please enter 1, 2, or 'q'.")
    
    # ========== STEP 2: Load Model ==========
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🔧 Device: {device}")
    if device.type == "cpu":
        print("⚠️  Training on CPU will be slow. Consider using a GPU for faster results.\n")
    print(50*"-")
    
    if model_source == "finetuned":
        # Find and select fine-tuned model
        print("🔍 Searching for fine-tuned models...\n")
        model_files = find_saved_models()
        
        if not model_files:
            print("✗ No fine-tuned models found (*.pth files ending with '-sft-standalone.pth' or '-sft-continued.pth')")
            print("💡 Please run main.py first to train and save a model, or choose Option [2] for pretrained model.\n")
            sys.exit(1)
        
        display_finetuned_models(model_files)
        
        # Get user selection
        while True:
            choice = input(f"Select a model (1-{len(model_files)}) or 'q' to quit: ").strip().lower()
            
            if choice == 'q':
                print("👋 Training cancelled. Goodbye!")
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
        
        # Load fine-tuned model
        print("📦 Loading fine-tuned model...")
        try:
            model, config, model_type = load_finetuned_model(selected_file, config_file, device)
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    else:  # pretrained
        display_pretrained_models()
        
        # Get user selection
        while True:
            choice = input(f"Select a model (1-4) or 'q' to quit: ").strip().lower()
            
            if choice == 'q':
                print("👋 Training cancelled. Goodbye!")
                sys.exit(0)
            
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(MODEL_CONFIGS):
                    break
                else:
                    print("✗ Invalid selection. Please try again.")
            except ValueError:
                print("✗ Invalid input. Please enter a number or 'q'.")
        
        # Load pretrained model
        print("📦 Loading pretrained GPT-2 model...")
        try:
            model, config, model_type = load_pretrained_model(idx + 1, device)
            selected_file = None  # No saved file for pretrained
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    # ========== STEP 3: Find Available JSON Files ==========
    print("\n🔍 Searching for JSON training data files...\n")
    json_files = find_json_files()
    
    # ========== STEP 4: Get Training Parameters ==========
    train_params = get_training_parameters(json_files)
    
    # ========== STEP 5: Load and Prepare Data ==========
    print(f"📥 Loading instruction data from: {train_params['data_file']}...")
    
    if not os.path.exists(train_params["data_file"]):
        print(f"✗ File not found: {train_params['data_file']}")
        sys.exit(1)
    
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
    print("📈 Evaluating initial losses (before training)...")
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=5, show_progress=True)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=5, show_progress=True)

    print(f"   Training loss:   {train_loss:.3f}")
    print(f"   Validation loss: {val_loss:.3f}")
    print(50*"-")

    # ========== STEP 8: Fine-Tuning ==========
    print("🚀 Starting fine-tuning...\n")
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
    print(f"\n✅ Training completed in {execution_time_minutes:.2f} minutes.")

    # ========== STEP 9: Plot and Save ==========
    print("📊 Generating loss plot...")
    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))    
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
    print(50*"-")

    # Save the fine-tuned model with new name
    if model_type == "finetuned":
        base_name = selected_file.replace("-sft-standalone.pth", "").replace("-sft-continued.pth", "")
        new_file_name = f"{base_name}-sft-continued.pth"
        
        # Ensure unique filename
        counter = 1
        while os.path.exists(new_file_name):
            new_file_name = f"{base_name}-sft-continued-v{counter}.pth"
            counter += 1
    else:
        model_size = config["emb_dim"]
        new_file_name = f"gpt2-{model_size}-sft-standalone.pth"
        
        # Ensure unique filename
        counter = 1
        while os.path.exists(new_file_name):
            new_file_name = f"gpt2-{model_size}-sft-standalone-v{counter}.pth"
            counter += 1
    
    torch.save(model.state_dict(), new_file_name)
    print(f"💾 Model saved as: {new_file_name}")
        
    print("\n" + "🎉"*35)
    print(f"Fine-tuning complete!")
    print(f"Your model has been trained for {num_epochs} epochs.")
    print("🎉"*35 + "\n")
    
    return model, tokenizer, config

if __name__ == "__main__":
    main()