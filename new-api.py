# """ Example handler file. """

# import runpod
# import torch
# from transformers import pipeline

# # If your handler runs inference on a model, load the model here.
# # You will want models to be loaded into memory before starting serverless.

# print ("in 1")

# pipe = pipeline(task="text-generation", model="./src", torch_dtype=torch.bfloat16, eos_token_id=2 ,pad_token_id=50256 ,max_length=4096, device_map="auto" ,offload_buffers=True )

# chat_history = []

# def ask_llm(chat_history):
#     print("in the llm")
#     prompt = pipe.tokenizer.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=True)
#     print("in the promt")
#     outputs = pipe(prompt, max_new_tokens=4096, do_sample=False)
#     print("output ")
#     #print(outputs[0]["generated_text"])
#     return outputs

# def extract_trailing_string(text):
#     print("in the taril starting ")
#     # Split the text by "[/INST]" and get the last part
#     parts = text.split("[/INST]")
#     print ("parts ")
#     trailing_string = parts[-1].strip()  # Remove any leading/trailing spaces
#     print ("trail parts ")
#     return trailing_string
# print("in 2")
# # pipe = pipeline("text-generation", model="Equall/Saul-Instruct-v1", torch_dtype=torch.bfloat16, device_map="auto")
# # pipe = pipeline(task="text-generation", model="./src", tokenizer=original_tokenizer)

# # ///
# # pipe = pipeline("text-generation", model="../src", torch_dtype=torch.bfloat16, device_map="auto", eos_token_id=2 )
# # ////

# # pipe = pipeline("text-generation", model="../src", torch_dtype=torch.bfloat16, device_map="auto", eos_token_id=2,disk_offload=True )



# # this
# # pipe = pipeline(task="text-generation", model="./src", torch_dtype=torch.bfloat16, eos_token_id=2 ,pad_token_id=50256 ,max_length=4096, device_map="auto" ,offload_buffers=True )

# # pipe = pipeline(task="text-generation", model=".cache/huggingface/hub/models--Equall--Saul-Instruct-v1/snapshots/2133ba7923533934e78f73848045299dd74f08d2/", eos_token_id=2 ,torch_dtype=torch.bfloat16, device_map="auto")

# # pipe = pipeline(task="text-generation", model="Equall/Saul-Instruct-v1", eos_token_id=2 ,torch_dtype=torch.bfloat16, device_map="auto")





# print("=======chat_history",chat_history)



# def handler(job):
#         ptint("in handler ")
#         chat=job['input']
#         # chat ="how are you"
#         print ("chart",chat["chat"])
#         print("in 4")
#         print("=======chat_history",chat_history)
#         # We use the tokenizer’s chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
#         chat_history.append({"role":"user", "content": chat["chat"]})
#         print ("append")
#         gen_output = ask_llm(chat_history)
#         print("output")
#         llm_responce = extract_trailing_string(gen_output[0]["generated_text"])
#         print("reponse")
#         # Add model response to chat history
#         chat_history.append({"role":"assistant", "content": llm_responce})
        
#         print(f'\n### SauLM-7B:\n{llm_responce}')
#         return llm_responce


# # runpod.serverless.start({"handler": handler})
# runpod.serverless.start({"handler": handler,"host":'0.0.0.0',"port":9000})






""" Example handler file. """

import runpod
import torch
from transformers import pipeline

pipe = pipeline(task="text-generation", model="./src", torch_dtype=torch.bfloat16, eos_token_id=2 ,pad_token_id=50256 ,max_length=4096, device_map="auto" ,offload_buffers=True )

chat_history = []

def ask_llm(chat_history):
    prompt = pipe.tokenizer.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=4096, do_sample=False)
    return outputs

def extract_trailing_string(text):
    # Split the text by "[/INST]" and get the last part
    parts = text.split("[/INST]")
    trailing_string = parts[-1].strip()  # Remove any leading/trailing spaces
    return trailing_string
def handler(job):
        chat=job['input']
        print ("chart",chat["chat"])
        # We use the tokenizer’s chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
        chat_history.append({"role":"user", "content": chat["chat"]})
        gen_output = ask_llm(chat_history)
        llm_responce = extract_trailing_string(gen_output[0]["generated_text"])
        # Add model response to chat history
        chat_history.append({"role":"assistant", "content": llm_responce})
        
        print(f'\n### SauLM-7B:\n{llm_responce}')
        return llm_responce


runpod.serverless.start({"handler": handler})
