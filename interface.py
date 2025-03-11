import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

qa_dict = {}
data_file = "preguntas_respuestas.jsonl"
with open(data_file, "r", encoding="utf-8") as f:
    for line in f:
        try:
            entry = json.loads(line)
            if "pregunta" in entry and "respuesta" in entry:
                pregunta = entry["pregunta"].strip()
                respuesta = entry["respuesta"].strip()
                if pregunta:
                    qa_dict[pregunta] = respuesta
        except json.JSONDecodeError:
            print(f"Línea ignorada por error de formato: {line}")

model_path = "modelo_preguntas_respuestas/checkpoint-2070"
base_model_name = "openai-community/gpt2"

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    local_files_only=True,
    device_map="auto",
    torch_dtype="auto"
)
model.eval()

def lookup_answer(question):
    return qa_dict.get(question.strip(), None)

def generate_answer(question, max_length=50):
    known_answer = lookup_answer(question)
    if known_answer is not None:
        return known_answer

    formatted_prompt = f"Pregunta: {question}\nRespuesta:"
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        do_sample=True,
        temperature=0.7,
        top_p=1.0,
        pad_token_id=tokenizer.eos_token_id
    )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = generated_text.split("Respuesta:")[-1].strip()
    return answer

if __name__ == "__main__":
    print("Escribe tu pregunta")
    while True:
        question = input("\nPregunta al UbuntuBot: ")
        if question.lower() in ["salir", "exit"]:
            print("Saliendo...")
            break
        answer = generate_answer(question)
        print("\nRespuesta:\n")
        print(answer)

