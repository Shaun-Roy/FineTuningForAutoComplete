!pip install unsloth
!pip install bitsandbyes

from unsloth import FastLanguageModel
import torch
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/mistral-7b-v0.3-bnb-4bit",      # New Mistral v3 2x faster!
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/llama-3-8b-bnb-4bit",           # Llama-3 15 trillion tokens model 2x faster!
    "unsloth/llama-3-8b-Instruct-bnb-4bit",
    "unsloth/llama-3-70b-bnb-4bit",
    "unsloth/Phi-3-mini-4k-instruct",        # Phi-3 2x faster!
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/mistral-7b-bnb-4bit",
    "unsloth/gemma-7b-bnb-4bit",             # Gemma 2.2x faster!
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/mistral-7b-v0.3", # "unsloth/mistral-7b" for 16bit loading
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 128, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",

                      "embed_tokens", "lm_head",], # Add for continual pretraining
    lora_alpha = 32,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = True,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

from datasets import load_dataset
dataset = load_dataset("wikitext","wikitext-103-raw-v1", split = "train[:2500]")
EOS_TOKEN = tokenizer.eos_token
def formatting_prompts_func(examples):
    return { "text" : [example + EOS_TOKEN for example in examples["text"]] }
dataset = dataset.map(formatting_prompts_func, batched = True,)

from datasets import load_dataset
from transformers import AutoTokenizer

# Load the dataset
dataset = load_dataset("GregSamek/TinyNews", split="train[:2500]")

# Load the tokenizer (Mistral 7B in this case)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

# Get the EOS token from the tokenizer (if needed, we can remove it later)
EOS_TOKEN = tokenizer.eos_token

# Prepare lists to hold the input-text and target-word
input_texts = []
target_words = []

# Loop over the dataset and process each example
for text in dataset["text"]:
    # Tokenize the text without special tokens like <eos> or <pad>
    tokens = tokenizer.tokenize(text, add_special_tokens=False)

    # Create input-target pairs for next-word prediction
    for i in range(len(tokens) - 1):
        input_text = tokens[i]  # Current token as input
        target_word = tokens[i + 1]  # Next token as target word
        input_texts.append(input_text)
        target_words.append(target_word)

# Create a new dataset from the processed input-text and target-word
processed_dataset = {
    "input_text": input_texts,
    "target_word": target_words
}

# Convert the processed dataset into a Hugging Face Dataset format
from datasets import Dataset
final_dataset = Dataset.from_dict(processed_dataset)

# Verify the first few examples
print(final_dataset[:5])


from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from unsloth import UnslothTrainer, UnslothTrainingArguments

tokenizer.pad_token = tokenizer.eos_token

trainer = UnslothTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 8,

    args = UnslothTrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 8,

        warmup_ratio = 0.1,
        num_train_epochs = 2,

        learning_rate = 5e-5,
        embedding_learning_rate = 5e-6,

        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.00,
        lr_scheduler_type = "cosine",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # Use this for WandB etc
    ),
)

trainer_stats = trainer.train()

from transformers import TextIteratorStreamer
from threading import Thread

# Initialize the streamer for real-time token generation
text_streamer = TextIteratorStreamer(tokenizer)

# Input sentence/phrase for which we want to predict the next word
input_sentence = input("Type something: ")

# Tokenize the input sentence
inputs = tokenizer(input_sentence, return_tensors="pt").to("cuda")

# Define generation config to get only 1 next token
generation_kwargs = dict(
    inputs,
    streamer=text_streamer,
    max_new_tokens=1,  # Only predict the next word
    use_cache=True,
)

# Generate next word in a separate thread
thread = Thread(target=model.generate, kwargs=generation_kwargs)
thread.start()

# Print the predicted next word
for new_text in text_streamer:
    print("Next word prediction:", new_text)
    break  # Only the first new token is needed


model.push_to_hub_gguf (
    "Mistral-7b-q4_k_m-autocomplete-GGUF",
    tokenizer,
    quantization_method = "q4_k_m",
    token=os.environ.get("HF_TOKEN") # Replace with your actual HF token or retrieval method
)
