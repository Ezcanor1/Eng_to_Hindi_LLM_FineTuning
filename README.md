# English-to-Hindi Translation using Fine-Tuned GPT-2 on Custom Dataset

This project fine-tunes the GPT-2 language model on a custom bilingual dataset (English-Hindi) to perform **instruction-based machine translation** from English to Hindi.

---

## 📌 Project Overview

- Fine-tunes **`openai-community/gpt2`** using the Hugging Face Transformers library.
- Uses a custom dataset `Dataset_English_Hindi.csv` with 50,000 English-Hindi sentence pairs.
- Implements token-level training with custom collate and padding for GPT-2 compatibility.
- Generates high-quality translations in an instruction-following format.

---

## 📂 Dataset

Dataset: `Dataset_English_Hindi.csv`  
Each row contains:

- `English`: English sentence
- `Hindi`: Corresponding Hindi translation

The dataset is split:
- **80% for training**
- **20% for testing**

---

## 📦 Model

Model: `GPT2LMHeadModel` (OpenAI GPT-2 small)  
Tokenizer: `GPT2Tokenizer`

Training objective:
```text
### Instruction:
Translate the given English sentence to Hindi.

### Input:
He is going to school.

### Response:
वह स्कूल जा रहा है।
