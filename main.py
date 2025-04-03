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

def generate_response(prompt):
    # Токенизация входного текста
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    
    # Генерация ответа
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=150,         # Максимальная длина ответа
            temperature=0.7,        # Температура для контроля случайности
            top_p=0.9,              # Top-p sampling
            do_sample=True          # Использование семплирования
        )
    
    # Декодирование ответа
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response