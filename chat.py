from transformers import AutoModelForCausalLM, AutoTokenizer

import torch
#torch_dtype=torch.float16

# model = AutoModelForCausalLM.from_pretrained("AdaptLLM/law-chat")
# tokenizer = AutoTokenizer.from_pretrained("AdaptLLM/law-chat")

model = AutoModelForCausalLM.from_pretrained("Equall/Saul-Instruct-v1")
tokenizer = AutoTokenizer.from_pretrained("Equall/Saul-Instruct-v1")

model = model.half()  # Use half precision (torch.float16)
model = model.to(torch.device("cuda"))  # Move the model to GPU

# Initialize chat history
chat_history = []

def ask_llm(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    outputs = model.generate(input_ids=inputs, max_length=4096)[0]

    answer_start = int(inputs.shape[-1])
    pred = tokenizer.decode(outputs[answer_start:], skip_special_tokens=True)
    
    return pred

our_system_prompt = "\nYou are a legal chatbot who is a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n" # Please do NOT change this

print("\nHi! I'm AdaptLLM, a law chatbot to assist you with your legal questions today.")
while True:
        user_input = input("\nPlease enter a question (or type 'exit' to quit): ")
        if user_input.lower() == "exit":
            print("Exiting the loop.")
            break
        else:
            chat_history.append(user_input)
            # Concatenate chat history for context
            prompt = f"<s>[INST] <<SYS>>{our_system_prompt}<</SYS>>\n\n{''.join(chat_history)} [/INST]"
            
            pred = ask_llm(prompt)
            # Add model response to chat history
            chat_history.append(pred)
            
            print(f'\n### AdaptLLM:\n{pred}')
            #print(f"You entered: {user_input}")
            

