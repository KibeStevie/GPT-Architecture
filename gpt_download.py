import os
import sys
import requests
import json
import numpy as np
import tensorflow as tf
from tqdm import tqdm

# Model configuration dictionary (display names → internal size codes)
MODEL_CONFIGS = {
    "1": {"name": "gpt2-small (124M)", "size": "124M", "emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "2": {"name": "gpt2-medium (355M)", "size": "355M", "emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "3": {"name": "gpt2-large (774M)", "size": "774M", "emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "4": {"name": "gpt2-xl (1558M)", "size": "1558M", "emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

def download_and_load_gpt2(model_size, models_dir):
    """Download and load GPT-2 weights from OpenAI's official checkpoints."""
    # Validate model size
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"Model size not in {allowed_sizes}")

    # Define paths
    model_dir = os.path.join(models_dir, model_size)
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
    
    filenames = [
        "checkpoint", "encoder.json", "hparams.json",
        "model.ckpt.data-00000-of-00001", "model.ckpt.index",
        "model.ckpt.meta", "vocab.bpe"
    ]

    # Download files
    os.makedirs(model_dir, exist_ok=True)
    for filename in filenames:
        file_url = os.path.join(base_url, model_size, filename)
        file_path = os.path.join(model_dir, filename)
        download_file(file_url, file_path)

    # Load settings and params
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    settings = json.load(open(os.path.join(model_dir, "hparams.json"), "r", encoding="utf-8"))
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)

    return settings, params


def download_file(url, destination):
    """Download a file with progress bar and caching."""
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()

        file_size = int(response.headers.get("Content-Length", 0))

        # Check if file exists and has same size (cache check)
        if os.path.exists(destination):
            file_size_local = os.path.getsize(destination)
            if file_size and file_size == file_size_local:
                print(f"✓ File already exists and is up-to-date: {destination}")
                return

        block_size = 1024  # 1 KB
        desc = os.path.basename(url)

        with tqdm(total=file_size, unit="iB", unit_scale=True, desc=desc) as progress_bar:
            with open(destination, "wb") as file:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        file.write(chunk)
                        progress_bar.update(len(chunk))
        print(f"✓ Downloaded: {destination}")

    except requests.exceptions.RequestException as e:
        print(f"✗ Failed to download from URL: {url}")
        print(f"  Error: {e}")
        print("  Check your internet connection or file availability.")
    except Exception as e:
        print(f"✗ An unexpected error occurred: {e}")


def load_gpt2_params_from_tf_ckpt(ckpt_path, settings):
    """Parse TensorFlow checkpoint into a Python dictionary for PyTorch loading."""
    # Initialize parameters dictionary with empty blocks for each layer
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}

    # Iterate over each variable in the checkpoint
    for name, _ in tf.train.list_variables(ckpt_path):
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))

        variable_name_parts = name.split("/")[1:]  # Skip 'model/' prefix

        target_dict = params
        if variable_name_parts[0].startswith("h"):
            layer_number = int(variable_name_parts[0][1:])
            target_dict = params["blocks"][layer_number]

        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})

        last_key = variable_name_parts[-1]
        target_dict[last_key] = variable_array

    return params


def display_model_menu():
    """Display the interactive model selection menu."""
    print("\n" + "="*60)
    print("🤖 GPT-2 Model Downloader")
    print("="*60)
    print("\nAvailable models:")
    print("-"*60)
    
    for key, config in MODEL_CONFIGS.items():
        print(f"  {key}. {config['name']}")
        print(f"     • Embedding dim: {config['emb_dim']}")
        print(f"     • Layers: {config['n_layers']}")
        print(f"     • Attention heads: {config['n_heads']}")
        print()
    
    print("-"*60)
    print("Enter a number (1-4) to download, or 'q' to quit.")
    print("="*60 + "\n")


def get_user_choice():
    """Get and validate user input."""
    while True:
        choice = input("Your choice: ").strip().lower()
        
        if choice == 'q':
            return None
        
        if choice in MODEL_CONFIGS:
            return choice
        
        print("✗ Invalid input. Please enter 1, 2, 3, 4, or 'q' to quit.\n")


def main():
    """Main entry point for the script."""
    # Display menu and get user choice
    display_model_menu()
    choice = get_user_choice()
    
    if choice is None:
        print("👋 Download cancelled. Goodbye!")
        sys.exit(0)
    
    # Get selected model info
    selected = MODEL_CONFIGS[choice]
    model_name = selected["name"]
    model_size = selected["size"]
    
    print(f"\n📥 You selected: {model_name}")
    print(f"📁 Models will be saved to: ./gpt2_models/{model_size}/")
    print("\nStarting download... (This may take several minutes)\n")
    
    # Set download directory
    models_dir = "gpt2_models"
    
    try:
        # Download and load the model
        settings, params = download_and_load_gpt2(model_size, models_dir)
        
        print("\n" + "✓"*60)
        print(f"✅ Successfully downloaded: {model_name}")
        print(f"📊 Model configuration:")
        print(f"   • Vocabulary size: {settings.get('n_vocab', 'N/A')}")
        print(f"   • Context length: {settings.get('n_ctx', 'N/A')}")
        print(f"   • Embedding dim: {settings.get('n_embd', 'N/A')}")
        print(f"   • Layers: {settings.get('n_layer', 'N/A')}")
        print(f"   • Attention heads: {settings.get('n_head', 'N/A')}")
        print(f"📁 Saved to: {os.path.join(models_dir, model_size)}")
        print("✓"*60 + "\n")
        
        # Optional: Return or save for use in other scripts
        return settings, params
        
    except Exception as e:
        print(f"\n✗ Error during download/loading: {e}")
        print("Please check your internet connection and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()