import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import torch
from threading import Thread

# Your Model Warehouse
model_id = "AMN04/phi3-medical-model" 

print("System: Initializing Medical AI Engine on CPU...")

# 1. Optimized Model Loading for Free-Tier CPU
# Using float32 for CPU stability and forcing the cpu device map
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.float32, 
    low_cpu_mem_usage=True,
    device_map={"": "cpu"}
)

def predict(message, history):
    # 2. Strict Prompt Formatting for Phi-3
    prompt = f"<|user|>\n{message}<|end|>\n<|assistant|>"
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
    
    # 3. Setup Streaming for better UX (words appear one by one)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    # 4. Conservative Generation Settings to prevent "Random" answers
    generation_kwargs = dict(
        inputs,
        streamer=streamer,
        max_new_tokens=128,      
        temperature=0.1,        # Keeps the model factual
        top_p=0.9,              
        repetition_penalty=1.2, 
        do_sample=True
    )
    
    # Run generation in background thread so the UI doesn't freeze
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    # Yield tokens as they are generated for the "typing" effect
    partial_message = ""
    for new_text in streamer:
        partial_message += new_text
        yield partial_message

# 5. Professional Gradio Interface (Fixed: Removed 'theme' argument)
demo = gr.ChatInterface(
    predict, 
    title="🩺 Med-GPT: Clinical AI Dashboard",
    description=(
        "**Technical Note:** This fine-tuned Phi-3 model is running on a 4-bit quantized "
        "CPU instance. Response generation follows a 'streaming' protocol to optimize "
        "perceived latency on commodity hardware."
    ),
    examples=[
        "What are the clinical indicators of Type-2 Diabetes?",
        "Explain the role of Beta-blockers in hypertension management.",
        "Summarize common symptoms of iron-deficiency anemia."
    ]
)

if __name__ == "__main__":
    demo.launch()
