from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Загрузка модели и токенизатора
model_name = "meta-llama/Llama-2-7b-chat-hf"  # Пример модели
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token="YOUR_HF_TOKEN")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",  # Автоматическое распределение по GPU/CPU
    load_in_8bit=True,  # Quantization для снижения нагрузки
    use_auth_token="YOUR_HF_TOKEN"
)