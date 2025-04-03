from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Путь к локальной папке с моделью
model_path = "/path/to/local/model"

# Определяем устройство
device = "cuda" if torch.cuda.is_available() else "cpu"

# Загрузка токенизатора и модели локально
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto" if torch.cuda.is_available() else None,  # Автоматическое распределение только для GPU
    load_in_8bit=torch.cuda.is_available(),  # Quantization только для GPU
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32  # Тип данных
)

# # Загрузка модели и токенизатора через Hugging Face:
# model_name = "mistralai/Mistral-7B-Instruct-v0.1"
# tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token="YOUR_HF_TOKEN")
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     device_map="auto",
#     load_in_8bit=True, 
#     use_auth_token="YOUR_HF_TOKEN"
# )

#/////////<---------------->//////////#

# Функция для генерации ответов
def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)  # Перемещаем входные данные на устройство
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=150,         # Максимальная длина ответа
            temperature=0.7,        # Температура для контроля случайности
            top_p=0.9,              # Top-p sampling
            do_sample=True          # Использование семплирования
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

#/////////<---------------->//////////#

# Создание веб-интерфейса с Gradio
import gradio as gr

def chat_interface(user_input):
    bot_response = generate_response(user_input)
    return bot_response

with gr.Blocks() as demo:
    gr.Markdown("# Chatbot based on Mistral-7B")
    with gr.Row():
        user_input = gr.Textbox(label="Your question", placeholder="Enter your query here...")
        submit_button = gr.Button("Send")
    bot_output = gr.Textbox(label="Bot's response", interactive=False)

    submit_button.click(chat_interface, inputs=user_input, outputs=bot_output)

demo.launch(server_name="localhost", server_port=7860)