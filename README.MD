## Prerequisites

Before you can run this system, ensure the following:

1. **Ollama Installation**: You must have Ollama installed on your local machine.
   - To install Ollama, follow the [installation instructions](https://ollama.com/download) for your platform.

2. **Model Setup**: You need to have a model uploaded to Hugging Face that supports Ollama or one of the pre-supported models by Ollama.

   If you need to upload a model to Hugging Face, follow these steps:
   - Fine-tune and upload your model to Hugging Face as you normally would.
   - Example: Mistral-7B is a popular model that can be used for autocomplete.

## Steps to Use Ollama for Autocomplete

### 1. Install Ollama (If Not Already Installed)

Follow the official Ollama installation guide to set it up on your local machine. The installation link is available on the [Ollama website](https://ollama.com/download).

### 2. Run the Model Locally Using Ollama

Once you have Ollama installed and your model is ready, you can run the model locally to generate autocomplete suggestions.

Once you have Ollama installed and your model is ready, you can run the model locally using the Ollama API to generate autocomplete suggestions.

### **PORT**
- **URL**: http://localhost:11434

### **CODE**
```bash
def get_ollama_suggestions(user_input, needed=5):
    system_prompt = f"""
    You are an autocomplete system. Your task is to predict the next word(s) based on the user's input. 
    Given the input: '{user_input}', provide up to {needed} next word predictions in JSON format, 
    excluding the input. Example output should be: ["word1", "word2", "word3", ..., "wordN"]
    """
```

```bash
    result = subprocess.run(
        ['ollama', 'run', ollama_model, '--prompt', system_prompt],
        capture_output=True,
        text=True
    )
```
