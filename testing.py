import torch
import json
import tiktoken
import sys
import os
from gpt_model import GPTModel
from gpt_instruction import format_input
from gpt_train import generate, text_to_token_ids, token_ids_to_text


def load_finetuned_model(model_path, config, device):
    """Load a fine-tuned GPT model from saved weights and config dict."""
    print(f"📦 Initializing model with config:")
    print(f"   • Vocabulary: {config.get('vocab_size', 'N/A')}")
    print(f"   • Context length: {config.get('context_length', 'N/A')}")
    print(f"   • Embedding dim: {config.get('emb_dim', 'N/A')}")
    print(f"   • Layers: {config.get('n_layers', 'N/A')}")
    print(f"   • Attention heads: {config.get('n_heads', 'N/A')}")
    
    model = GPTModel(config)
    
    print(f"📥 Loading fine-tuned weights from: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model loaded successfully! ({total_params:,} parameters)")
    
    return model


def get_user_instruction():
    """Prompt user for an instruction and return formatted input."""
    print("\n" + "-"*70)
    print("💬 Enter your instruction below (or type 'quit' / 'q' to exit):")
    print("-"*70)
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() in ['quit', 'q', 'exit']:
            return None
        
        if not user_input:
            print("⚠️  Please enter a non-empty instruction.")
            continue
        
        entry = {"instruction": user_input, "input": ""}
        formatted = format_input(entry)
        full_prompt = formatted + "\n\n### Response:\n"
        
        return full_prompt


def generate_response(model, tokenizer, prompt, device, config, 
                      max_new_tokens=256, temperature=0.3, top_k=25):
    """Generate a response from the model given a formatted prompt."""
    input_ids = text_to_token_ids(prompt, tokenizer)
    input_ids = input_ids.to(device)
    
    with torch.no_grad():
        output_ids = generate(
            model=model,
            idx=input_ids,
            max_new_tokens=max_new_tokens,
            context_size=config["context_length"],
            temperature=temperature,
            top_k=top_k,
            eos_id=50256
        )
    
    generated_text = token_ids_to_text(output_ids, tokenizer)
    
    if generated_text.startswith(prompt):
        response = generated_text[len(prompt):]
    else:
        if "### Response:" in generated_text:
            response = generated_text.split("### Response:")[-1]
        else:
            response = generated_text
    
    response = response.strip().lstrip("### Response:").strip()
    return response


def display_welcome(model_name, device):
    """Display welcome message and model info."""
    print("\n" + "✨"*35)
    print(f"🎯 Fine-Tuned GPT-2 Inference Interface")
    print("✨"*35)
    print(f"\n📋 Model: {model_name}")
    print(f"🔧 Device: {device.type.upper()} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"\n💡 Tips:")
    print("   • Be specific in your instructions for better results")
    print("   • Try: 'Explain quantum computing in simple terms'")
    print("   • Try: 'Write a Python function to reverse a string'")
    print("   • Type 'quit' or 'q' anytime to exit")
    print("\n" + "-"*70 + "\n")


def main():
    print("🔍 Searching for fine-tuned models...\n")
    
    model_files = [f for f in os.listdir(".") if f.endswith("-sft-standalone.pth")]
    
    if not model_files:
        print("✗ No fine-tuned models found (*.pth files ending with '-sft-standalone.pth')")
        print("💡 Please run main.py first to train and save a model.\n")
        sys.exit(1)
    
    print("✅ Found saved model(s):")
    for i, mf in enumerate(model_files, 1):
        print(f"   {i}. {mf}")
    print()
    
    if len(model_files) == 1:
        selected_file = model_files[0]
        print(f"🎯 Auto-selected: {selected_file}\n")
    else:
        choice = input(f"Select a model (1-{len(model_files)}): ").strip()
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(model_files):
                selected_file = model_files[idx]
            else:
                print("✗ Invalid selection. Using first model.")
                selected_file = model_files[0]
        except ValueError:
            print("✗ Invalid input. Using first model.")
            selected_file = model_files[0]
    
    config_file = selected_file.replace(".pth", "_config.json")
    
    # Load or infer config
    if os.path.exists(config_file):
        print(f"📦 Loading model config from: {config_file}")
        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)
    else:
        print(f"⚠️  Config file not found: {config_file}")
        print("💡 Inferring default GPT-2 configuration from model name...")
        
        # ✅ FIXED: qkv_bias set to True, drop_rate to 0.0 to match training config
        default_configs = {
            "768": {"vocab_size": 50257, "context_length": 1024, "emb_dim": 768, "n_layers": 12, "n_heads": 12, "drop_rate": 0.0, "qkv_bias": True},
            "1024": {"vocab_size": 50257, "context_length": 1024, "emb_dim": 1024, "n_layers": 24, "n_heads": 16, "drop_rate": 0.0, "qkv_bias": True},
            "1280": {"vocab_size": 50257, "context_length": 1024, "emb_dim": 1280, "n_layers": 36, "n_heads": 20, "drop_rate": 0.0, "qkv_bias": True},
            "1600": {"vocab_size": 50257, "context_length": 1024, "emb_dim": 1600, "n_layers": 48, "n_heads": 25, "drop_rate": 0.0, "qkv_bias": True}
        }
        
        config = None
        for size, cfg in default_configs.items():
            if size in selected_file:
                config = cfg
                break
                
        if config is None:
            print("❓ Could not infer model size. Defaulting to GPT-2 Small (768).")
            config = default_configs["768"]
            
        print(f"✅ Using inferred config: emb_dim={config['emb_dim']}, n_layers={config['n_layers']}, n_heads={config['n_heads']}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        model = load_finetuned_model(selected_file, config, device)
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print("💡 Make sure the model weights match the inferred architecture.\n")
        sys.exit(1)
    
    print("🔤 Loading tokenizer...")
    tokenizer = tiktoken.get_encoding("gpt2")
    print("✅ Tokenizer ready.\n")
    
    model_name = selected_file.replace("-sft-standalone.pth", "").replace("-", " ").title()
    
    display_welcome(model_name, device)
    print("🚀 Ready! Ask me anything...\n")
    
    while True:
        prompt = get_user_instruction()
        if prompt is None:
            print("\n👋 Goodbye! Thanks for testing.\n")
            break
        
        print("\n🤖 Thinking...", end="", flush=True)
        
        try:
            response = generate_response(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                device=device,
                config=config,
                max_new_tokens=256,
                temperature=0.3,
                top_k=25
            )
            print("\r" + " "*20 + "\r", end="")
            print(f"\nAssistant: {response}\n")
        except torch.cuda.OutOfMemoryError:
            print("\r" + " "*20 + "\r", end="")
            print("\n✗ Error: GPU out of memory. Try reducing max_new_tokens or using a smaller model.\n")
        except Exception as e:
            print("\r" + " "*20 + "\r", end="")
            print(f"\n✗ Error during generation: {e}\n")
            print("💡 Try simplifying your instruction or check model compatibility.\n")


if __name__ == "__main__":
    main()