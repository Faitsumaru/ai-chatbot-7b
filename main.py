from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import gradio as gr

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


# Функция для обработки запросов через интерфейс
def chat_interface(user_input):
    bot_response = generate_response(user_input)
    return bot_response

# Создание интерфейса
with gr.Blocks() as demo:
    gr.Markdown("# Чат-бот на базе Llama2-7B")
    with gr.Row():
        user_input = gr.Textbox(label="Ваш вопрос", placeholder="Введите ваш запрос здесь...")
        submit_button = gr.Button("Отправить")
    bot_output = gr.Textbox(label="Ответ бота", interactive=False)

    # Привязка функции к кнопке
    submit_button.click(chat_interface, inputs=user_input, outputs=bot_output)

# Запуск интерфейса
demo.launch(server_name="0.0.0.0", server_port=7860)