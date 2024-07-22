# BERT-Fine-Tuning-with-LoRA
---
This repository provides a step-by-step approach to fine-tune the BERT (**B**idirectional **E**ncoder **R**epresentations from **T**ransformers) model. Not only this, but also using **Lo**w **R**ank **A**daptation (*LoRA*) technique for better finetuning and size reduction.

## Usage
---
To work with this project, follow these steps:

1. **Clone the repository:**
    ```bash
    git clone https://github.com/sulaiman-shamasna/BERT-Fine-Tuning-with-LoRA.git
    cd BERT-Fine-Tuning-with-LoRA/
    ```

2. **Set up Python environment:**
    - Ensure you have **Python 3.10.X** or higher installed.
    - Create and activate a virtual environment:
      - For Windows (using Git Bash):
        ```bash
        source env/Scripts/activate
        ```
      - For Linux and macOS:
        ```bash
        source env/bin/activate
        ```

3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Setup CUDA:**
    ```bash
    pip install torchvision==0.13.1+cu116 -f https://download.pytorch.org/whl/torch_stable.html
    ```