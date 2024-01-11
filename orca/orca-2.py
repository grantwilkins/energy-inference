import torch
import transformers

if torch.cuda.is_available():
    torch.set_default_device("cuda")
else:
    torch.set_default_device("cpu")

torch.set_default_device("mps")
    
model = transformers.AutoModelForCausalLM.from_pretrained("microsoft/Orca-2-7b", device_map='auto', offload_folder="offload", offload_state_dict=True)

# https://github.com/huggingface/transformers/issues/27132
# please use the slow tokenizer since fast and slow tokenizer produces different tokens
tokenizer = transformers.AutoTokenizer.from_pretrained(
        "microsoft/Orca-2-7b",
        use_fast=False,
    )

system_message = "You are Orca, an AI language model created by Microsoft. You are a cautious assistant. You carefully follow instructions. You are helpful and harmless and you follow ethical guidelines and promote positive behavior."
user_message = "How can you determine if a restaurant is popular among locals or mainly attracts tourists, and why might this information be useful?"

prompt = f"<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant"

inputs = tokenizer(prompt, return_tensors='pt')
output_ids = model.generate(inputs["input_ids"],)
answer = tokenizer.batch_decode(output_ids)[0]

print(answer)

# This example continues showing how to add a second turn message by the user to the conversation
second_turn_user_message = "Give me a list of the key points of your first answer."

# we set add_special_tokens=False because we dont want to automatically add a bos_token between messages
second_turn_message_in_markup = f"\n<|im_start|>user\n{second_turn_user_message}<|im_end|>\n<|im_start|>assistant"
second_turn_tokens = tokenizer(second_turn_message_in_markup, return_tensors='pt', add_special_tokens=False)
second_turn_input = torch.cat([output_ids, second_turn_tokens['input_ids']], dim=1)

output_ids_2 = model.generate(second_turn_input,)
second_turn_answer = tokenizer.batch_decode(output_ids_2)[0]

print(second_turn_answer)

