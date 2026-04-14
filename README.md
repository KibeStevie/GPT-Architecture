<h1>🤖 GPT-2 Instruction Fine-Tuning Project</h1>
<p>A complete, modular system for fine-tuning GPT-2 models on custom instruction datasets</p>

<h3>📋 Table of Contents</h3>
<ul>
    <li><a href="#features">✨ Features</a></li>
    <li><a href="#installation">🛠️ Installation</a></li>
    <li><a href="#structure">📁 Project Structure</a></li>
    <li><a href="#quickstart">🚀 Quick Start</a></li>
    <li><a href="#usage">📖 Usage Guide</a></li>
    <li><a href="#configuration">⚙️ Configuration Options</a></li>
    <li><a href="#cv-usecase">📄 CV/Resume Use Case</a></li>
    <li><a href="#troubleshooting">🔧 Troubleshooting</a></li>
    <li><a href="#license">📝 License & Credits</a></li>
</ul>

<h2 id="features">✨ Features</h2>
🎯 Interactive Model Selection</br>
Choose from 4 GPT-2 sizes (124M to 1558M) </br>
📊 Progress Bars</br>
Real-time training & evaluation with tqdm </br>
🔄 Continue Training</br>
Resume training on saved models </br>
💾 Auto-Save</br>
Models, configs, and metadata saved automatically </br>
🧪 Test Mode</br>
Quick validation with small model </br>
📁 JSON Data Selection</br>
Browse available training data files

<h2 id="installation">🛠️ Installation</h2>

<h3>Requirements</h3>
<ul>
    <li>Python 3.8+</li>
    <li>PyTorch 2.0+</li>
    <li>CUDA (optional, for GPU acceleration)</li>
</ul>

<h3>Install Dependencies</h3>
<pre><code>pip install torch tiktoken matplotlib tqdm requests tensorflow numpy</code></pre>

<h3>Optional: Create Virtual Environment</h3>
<pre><code>python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
pip install -r requirements.txt</code></pre>

<h2 id="structure">📁 Project Structure</h2>
<div class="file-structure">

<pre>project/
├── main.py                      # Initial training script
├── test.py                      # Inference/testing script
├── continue_training.py         # Resume training on saved models
├── gpt.py                       # GPT model architecture
├── gpt_generate.py              # Text generation utilities
├── gpt_instruction_finetune.py  # Instruction fine-tuning utilities
├── gpt_model.py                 # Model class (alias for gpt.py)
├── gpt_train.py                 # Training functions
├── gpt_download.py              # Model weight downloading
├── gpt_instruction.py           # Dataset & collation functions
├── instruction-data.json        # Training dataset
├── *.pth                        # Saved model weights
├── *_config.json                # Model configurations
├── *_metadata.json              # Training metadata
├── loss-plot*.pdf               # Loss visualization plots
└── README.md                    # This file</pre>

<h2 id="quickstart">🚀 Quick Start</h2>

<h3>1️⃣ Train a Model</h3>
<pre><code>python main.py</code></pre>
<div class="info">
    <strong>What happens:</strong>
    <ul>
        <li>Select GPT-2 model size (124M, 355M, 774M, or 1558M)</li>
        <li>Download pretrained weights (cached after first download)</li>
        <li>Load instruction dataset</li>
        <li>Fine-tune for 2 epochs</li>
        <li>Save model + config files</li>
    </ul>
</div>

<h3>2️⃣ Test the Model</h3>
<pre><code>python test.py</code></pre>
<div class="info">
    <strong>What happens:</strong>
    <ul>
        <li>Auto-detects saved models</li>
        <li>Interactive terminal interface</li>
        <li>Enter instructions, get responses</li>
    </ul>
</div>

<h3>3️⃣ Continue Training</h3>
<pre><code>python continue_training.py</code></pre>
<div>
    <strong>What happens:</strong>
    <ul>
        <li>Select previously saved model</li>
        <li>Choose training data from available JSON files</li>
        <li>Use default or custom parameters</li>
        <li>Continue fine-tuning from where you left off</li>
    </ul>
</div>

<h2 id="usage">📖 Usage Guide</h2>

<h3>Training (main.py)</h3>
        <pre><code>
        
### Interactive mode (recommended)
python main.py
Model selection menu will appear:
[1] gpt2-small (124M) - ~2 GB RAM
[2] gpt2-medium (355M) - ~4 GB RAM
[3] gpt2-large (774M) - ~8 GB RAM
[4] gpt2-xl (1558M) - ~16 GB RAM</code></pre>

<h3>Testing (test.py)</h3>
<pre><code>

### Interactive inference

python test.py
You: Explain quantum computing in simple terms
Assistant: Quantum computing uses quantum mechanics principles...</code></pre>

<h3>Continued Training (continue_training.py)</h3>
        <pre><code>
        
### Resume training with default parameters
python continue_training.py
Select option [1] for default params
Resume training with custom parameters

python continue_training.py
Select option [2] for custom params
Enter: epochs, learning rate, batch size, etc.</code></pre>

<h2 id="configuration">⚙️ Configuration Options</h2>

<h3>Default Training Parameters</h3>
<table>
    <thead>
        <tr>
            <th>Parameter</th>
            <th>Default Value</th>
            <th>Description</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><code>num_epochs</code></td>
            <td>2</td>
            <td>Number of training epochs</td>
        </tr>
        <tr>
            <td><code>learning_rate</code></td>
            <td>0.00005</td>
            <td>AdamW learning rate</td>
        </tr>
        <tr>
            <td><code>batch_size</code></td>
            <td>8</td>
            <td>Samples per batch</td>
        </tr>
        <tr>
            <td><code>context_length</code></td>
            <td>1024</td>
            <td>Maximum sequence length</td>
        </tr>
        <tr>
            <td><code>eval_freq</code></td>
            <td>5</td>
            <td>Evaluation every N steps</td>
        </tr>
        <tr>
            <td><code>weight_decay</code></td>
            <td>0.1</td>
            <td>L2 regularization</td>
        </tr>
    </tbody>
</table>

<h3>Model Configurations</h3>
<table>
    <thead>
        <tr>
            <th>Model</th>
            <th>Parameters</th>
            <th>VRAM Required</th>
            <th>Best For</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>gpt2-small</td>
            <td>124M</td>
            <td>2 GB</td>
            <td>Testing, experimentation</td>
        </tr>
        <tr>
            <td>gpt2-medium</td>
            <td>355M</td>
            <td>4 GB</td>
            <td>General purpose</td>
        </tr>
        <tr>
            <td>gpt2-large</td>
            <td>774M</td>
            <td>8 GB</td>
            <td>Better quality outputs</td>
        </tr>
        <tr>
            <td>gpt2-xl</td>
            <td>1558M</td>
            <td>16 GB</td>
            <td>Production quality</td>
        </tr>
    </tbody>
</table>

<h2 id="cv-usecase">📄 CV/Resume Use Case</h2>

<p>This project can be adapted for CV/resume analysis tasks:</p>

<h3>Example Training Data Format</h3>
        <pre><code>[

{
"instruction": "Extract all technical skills from this CV.",
"input": "John Doe, Software Engineer, 5 years exp, Python, AWS...",
"output": "['Python', 'AWS', 'Docker', 'Kubernetes']"
},
{
"instruction": "Score this CV for a Senior Developer role (1-10).",
"input": "John Doe, Software Engineer, 5 years exp...",
"output": "Score: 7/10. Strong tech stack, lacks leadership experience."
}
]</code></pre>

<h3>Recommended Settings for CV Data</h3>
<table>
    <thead>
        <tr>
            <th>Setting</th>
            <th>Value</th>
            <th>Reason</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><code>context_length</code></td>
            <td>2048</td>
            <td>CVs are longer than typical prompts</td>
        </tr>
        <tr>
            <td><code>learning_rate</code></td>
            <td>0.00001</td>
            <td>Lower to preserve general knowledge</td>
        </tr>
        <tr>
            <td><code>num_epochs</code></td>
            <td>3-5</td>
            <td>More epochs for specialized tasks</td>
        </tr>
        <tr>
            <td><code>batch_size</code></td>
            <td>4</td>
            <td>Reduce if OOM with long contexts</td>
        </tr>
    </tbody>
</table>

<h2 id="troubleshooting">🔧 Troubleshooting</h2>

<h3>Common Issues</h3>
<table>
    <thead>
        <tr>
            <th>Issue</th>
            <th>Solution</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><strong>CUDA Out of Memory</strong></td>
            <td>Reduce <code>batch_size</code> or <code>context_length</code></td>
        </tr>
        <tr>
            <td><strong>Training too slow</strong></td>
            <td>Use GPU, reduce model size, or enable mixed precision</td>
        </tr>
        <tr>
            <td><strong>Model not loading</strong></td>
            <td>Ensure <code>_config.json</code> exists alongside <code>.pth</code> file</td>
        </tr>
        <tr>
            <td><strong>No JSON files found</strong></td>
            <td>Place training data in root directory</td>
        </tr>
        <tr>
            <td><strong>Poor output quality</strong></td>
            <td>Increase training epochs or add more data</td>
        </tr>
        <tr>
            <td><strong>Loss not decreasing</strong></td>
            <td>Lower learning rate or check data format</td>
        </tr>
    </tbody>
</table>

<div class="warning">
    <strong>⚠️ CPU Training Warning</strong>
    <p>Training on CPU is <strong>significantly slower</strong>. Expected times:</p>
    <ul>
        <li><strong>124M:</strong> ~2-3 hours (CPU) vs ~10-15 minutes (GPU)</li>
        <li><strong>355M:</strong> ~6-8 hours (CPU) vs ~30-45 minutes (GPU)</li>
        <li><strong>774M:</strong> ~12-15 hours (CPU) vs ~1-2 hours (GPU)</li>
    </ul>
</div>

<h3>Output Files</h3>
<p>After training, you'll get:</p>
<table>
    <thead>
        <tr>
            <th>File</th>
            <th>Description</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><code>*-sft-standalone.pth</code></td>
            <td>Fine-tuned model weights</td>
        </tr>
        <tr>
            <td><code>*-sft-continued.pth</code></td>
            <td>Continued training weights</td>
        </tr>
        <tr>
            <td><code>*_config.json</code></td>
            <td>Model architecture config</td>
        </tr>
        <tr>
            <td><code>*_metadata.json</code></td>
            <td>Training history & parameters</td>
        </tr>
        <tr>
            <td><code>loss-plot*.pdf</code></td>
            <td>Training/validation loss visualization</td>
        </tr>
    </tbody>
</table>

<h2 id="license">📝 License & Credits</h2>

<div>
    <strong>📚 References</strong>
    <ul>
        <li><strong>Book:</strong> <a href="https://www.manning.com/books/build-a-large-language-model-from-scratch">Build a Large Language Model From Scratch</a> by Sebastian Raschka</li>
        <li><strong>Code:</strong> <a href="https://github.com/rasbt/LLMs-from-scratch">GitHub Repository</a></li>
        <li><strong>License:</strong> Apache License 2.0</li>
        <li><strong>GPT-2:</strong> <a href="https://openai.com/research/better-language-models">OpenAI GPT-2</a></li>
    </ul>
</div>

<footer>
    <p><strong>🎉 Happy Fine-Tuning! 🚀</strong></p>
</footer>
</div>
