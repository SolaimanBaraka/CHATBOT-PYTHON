from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling

model_name = "openai-community/gpt2"
data_file = "preguntas_respuestas.jsonl"

dataset = load_dataset("json", data_files=data_file, split="train")

def filtrar_lineas_extras(ej):
    return ej["pregunta"] != "Pregunta" and ej["respuesta"] != "Respuesta"

dataset = dataset.filter(filtrar_lineas_extras)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)

def tokenize_function(examples):
    formatted_examples = [
        f"Pregunta: {p}\nRespuesta: {r}<|EOT|>"
        for p, r in zip(examples["pregunta"], examples["respuesta"])
    ]
    return tokenizer(
        formatted_examples,
        padding="max_length",
        truncation=True,
        max_length=64
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)

train_dataset = tokenized_dataset

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="modelo_preguntas_respuestas",
    num_train_epochs=10,
    per_device_train_batch_size=1,
    logging_steps=20,
    save_strategy="epoch",
    evaluation_strategy="no",
    learning_rate=2e-5,
    weight_decay=0.01,
    fp16=True,
    save_total_limit=1
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()

