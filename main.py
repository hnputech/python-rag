from fastapi import FastAPI
import torch
from transformers import pipeline

print ("in 1")
def ask_llm(chat_history):
    print("in the llm")
    prompt = pipe.tokenizer.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=True)
    print("in the promt")
    outputs = pipe(prompt, max_new_tokens=4096, do_sample=False)
    print("output ")
    #print(outputs[0]["generated_text"])
    return outputs

def extract_trailing_string(text):
    print("in the taril starting ")
    # Split the text by "[/INST]" and get the last part
    parts = text.split("[/INST]")
    print ("parts ")
    trailing_string = parts[-1].strip()  # Remove any leading/trailing spaces
    print ("trail parts ")
    return trailing_string
print("in 2")
pipe = pipeline("text-generation", model="Equall/Saul-Instruct-v1", torch_dtype=torch.bfloat16, device_map="auto")
chat_history = []
print("=======chat_history",chat_history)

app = FastAPI()
print("in3")
@app.get("/chat")
async def root(chat):
        # chat ="how are you"
        print ("chart",chat)
        print("in 4")
        print("=======chat_history",chat_history)
        # We use the tokenizer’s chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
        chat_history.append({"role":"user", "content": chat})
        print ("append")
        gen_output = ask_llm(chat_history)
        print("output")
        llm_responce = extract_trailing_string(gen_output[0]["generated_text"])
        print("reponse")
        # Add model response to chat history
        chat_history.append({"role":"assistant", "content": llm_responce})
        
        print(f'\n### SauLM-7B:\n{llm_responce}')
        return llm_responce
