from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from pyJoules.energy_meter import EnergyContext
from pyJoules.device.nvidia_device import NvidiaGPUDomain
def run_inference():
	model_name = "mistralai/Mistral-7B-v0.1"
	model = AutoModelForCausalLM.from_pretrained(model_name)
	tokenizer = AutoTokenizer.from_pretrained(model_name)
	inputs = tokenizer("The sky is blue because", return_tensors="pt")
	model.to("cuda")
	inputs = inputs.to("cuda")
	with torch.no_grad():  # Disable gradient calculations for inference
		outputs = model(**inputs)
	token_ids = outputs[0].tolist()
	print(outputs)
	# Decode the list of token ids to text
	predicted_text = tokenizer.decode(token_ids, skip_special_tokens=True)

with EnergyContext(domains=[NvidiaGPUDomain(0)]):
	run_inference()
