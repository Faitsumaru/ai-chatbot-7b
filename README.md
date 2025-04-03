# **_Chatbot with Mistral-7B / Llama2-7B Models_**

## About
This project demonstrates a chatbot powered by two state-of-the-art large language models: **Mistral-7B** / **Llama2-7B**. Both models are optimized for generating human-like responses and can be deployed locally or in the cloud. 

The chatbot is built using the **Hugging Face Transformers** library, **Gradio** for the web interface, and **bitsandbytes** for 8-bit quantization to reduce memory usage. This project showcases how to load pre-trained models, generate responses, and create a user-friendly interface for interaction.

### Key Features
* **Models**: 
  - **Mistral-7B**: A modern language model with 7 billion parameters, optimized for instruction-following tasks.
  - **Llama2-7B**: A powerful open-source model from Meta, designed for conversational AI.
* **Quantization**: Support for 8-bit quantization to reduce GPU memory requirements.
* **Offline Mode**: Ability to run the chatbot offline by downloading model files locally.
* **Online Mode**: Option to load models directly from Hugging Face Hub (requires internet access and authentication).
* **Web Interface**: A simple and intuitive Gradio-based UI for seamless user interaction.
* **Optimized Performance**: Configurations for both GPU (CUDA) and CPU environments.

> Version: Apr 2025, created by Gleb 'Faitsuma' Kiryakov

---


## Project Structure

### Code Overview
1. **Model Loading**:
   * The project supports two models: **Mistral-7B** and **Llama2-7B**.
   * Models can be loaded either from the Hugging Face Hub (online mode) or from local files (offline mode).
   * The code dynamically detects whether a GPU is available and adjusts the device (`cuda` or `cpu`) accordingly.

2. **Tokenization**:
   * The tokenizer processes user input into tokens that the model can understand.
   * Special tokens (e.g., `<bos>`, `<eos>`) are handled automatically.

3. **Response Generation**:
   * The model generates responses using advanced techniques like temperature scaling, top-p sampling, and max token length control.
   * Responses are decoded back into human-readable text.

4. **Web Interface**:
   * A Gradio-based web interface allows users to interact with the chatbot via a browser.
   * The interface includes a text input box for user queries and a text output box for bot responses.

5. **Device Management**:
   * For GPU setups, 8-bit quantization is supported to reduce memory usage.
   * For CPU-only setups, 8-bit quantization is disabled, and the model runs in full precision.

---