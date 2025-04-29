!pip install unsloth

from unsloth import FastLanguageModel

# Load quantized 4-bit model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-2-7b-bnb-4bit",  # << THIS is already quantized
    max_seq_length = 512,
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    lora_alpha = 32,
    lora_dropout = 0.05,
    bias = "none",
    use_gradient_checkpointing = True,
)

from datasets import load_dataset, Dataset
# --- Load text dataset ---
raw_dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")


# --- function to create input-target pairs ---
def generate_input_target_pairs(text):
    words = text.strip().split()
    pairs = []
    for i in range(1, len(words)):
        input_text = " ".join(words[:i])
        target_word = words[i]
        pairs.append((input_text, target_word))
    return pairs

def preprocess_and_tokenize(batch):
    input_ids = []
    labels = []

    for text in batch["text"]:
        pairs = generate_input_target_pairs(text)
        for input_text, target_word in pairs:
            tokenized = tokenizer(
                input_text,
                text_target=target_word,
                truncation=True,
            )
            input_ids.append(tokenized["input_ids"])
            labels.append(tokenized["labels"])

    return {"input_ids": input_ids, "labels": labels}

tokenized_dataset = raw_dataset.map(
    preprocess_and_tokenize,
    batched=True,
    batch_size=1000,
    remove_columns=["text"]
)

FastLanguageModel.fit(
    model = model,
    tokenizer = tokenizer,
    dataset = tokenized_dataset,
    batch_size = 32,
    epochs = 3,
    lr = 2e-4,
)

prompt = input("enter text here: ")
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
input_ids = input_ids.to(model.device) 
outputs = model.generate(input_ids, max_new_tokens=10)
predicted_token = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(predicted_token)
