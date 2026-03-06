# 🎌 Anime Game Character Designer

This repository provides a professional pipeline for fine-tuning **Stable Diffusion 1.5** using **LoRA (Low-Rank Adaptation)** to create high-quality anime game characters. It includes a training script, a Gradio-based web interface, and a Jupyter notebook for experimentation.

## 🚀 Features

- **Fine-Tuning with LoRA**: Efficiently adapt Stable Diffusion to specific anime styles.
- **Automated Dataset Loading**: Streams anime face datasets directly from Hugging Face.
- **Professional Python Structure**: Refactored from notebook to modular Python scripts.
- **Interactive Web UI**: Generate characters using a user-friendly Gradio interface.
- **Mixed Precision Training**: Supports `fp16` for faster training and lower VRAM usage.

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/SeifEldenOsama/AI_Game_Desginer.git
   cd AI_Game_Desginer
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Set your Hugging Face token:
   ```bash
   export HF_TOKEN="YOUR_API_KEY"
   ```

## 📖 Usage

### Running the Web Interface
To start the Gradio UI for character generation:
```bash
python main.py --mode app
```

### Training the Model
To start the fine-tuning process:
```bash
python main.py --mode train
```

### Using the Notebook
For interactive experimentation, you can use the provided Jupyter notebook:
`Anime_Game_Character_Designer.ipynb`

## 🎨 Generation Parameters

- **Trigger Word**: `anime character, sks style`
- **Prompting**: Combine the trigger word with your character description (e.g., `anime character, sks style, warrior with golden armor`).
- **Negative Prompt**: Use to avoid common artifacts (e.g., `blurry, low quality, deformed`).

## 📂 Project Structure

- `main.py`: Entry point for the application.
- `src/`: Directory containing the core logic.
  - `train.py`: Logic for the LoRA fine-tuning process.
  - `app.py`: Gradio web interface for inference.
  - `model.py`: Dataset and model loading utilities.
  - `config.py`: Configuration parameters for training and inference.
- `requirements.txt`: List of required Python packages.

## 📜 License

This project is licensed under the MIT License.
