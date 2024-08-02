from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, AutoModelForCausalLM, Trainer, TrainingArguments
from dotenv import load_dotenv
import os

load_dotenv()

dataset = load_dataset('json', data_files='data.json', split='train')

model_name = os.getenv("MODEL_NAME")

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

def tokenize_function(examples):
    text = " ".join(str(value) for value in examples.values())
    return tokenizer(text, padding='max_length', truncation=True, max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

model = AutoModelForCausalLM.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))

training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir='./logs',
    logging_steps=500,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets,
)

trainer.train()