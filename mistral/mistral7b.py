from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from pyJoules.energy_meter import EnergyContext
from pyJoules.device.nvidia_device import NvidiaGPUDomain
from pyJoules.handler.csv_handler import CSVHandler

def run_inference():
	model_name = "mistralai/Mistral-7B-v0.1"
	tokenizer = AutoTokenizer.from_pretrained(model_name)
	model = AutoModelForCausalLM.from_pretrained(
        	model_name,
        	torch_dtype=torch.bfloat16,
        	device_map="auto",
        	trust_remote_code=True)	
	pipe = pipeline(
    		"text-generation", 
    		model=model, 
    		tokenizer = tokenizer, 
    		torch_dtype=torch.bfloat16, 
    		device_map="auto")
	prompt = "As a data scientist, can you explain the concept of regularization in machine learning?"
	sequences = pipe(
    		prompt,
    		do_sample=True,
    		max_new_tokens=100, 
    		temperature=0.7, 
    		top_k=50, 
    		top_p=0.95,
    		num_return_sequences=1,)
	print(sequences[0]['generated_text'])

csv_handle = CSVHandler("results.csv")
with EnergyContext(handler=csv_handle, domains=[NvidiaGPUDomain([0,1])]):
	run_inference()

csv_handle.save_data()
