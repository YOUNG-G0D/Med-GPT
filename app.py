import os
import sys
import time
import torch
import eventlet
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextStreamer
from threading import Thread

# --- 1. Configuration for the Quantized Model ---
quantized_model_path = "/media/bisagn/data/FineTune/quantized_phi3_medical_model_4bit_FINAL" 

# --- 2. Load the Quantized Model and Tokenizer ---
try:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading 4-bit quantized model from '{quantized_model_path}'...")
    model = AutoModelForCausalLM.from_pretrained(
        quantized_model_path,
        quantization_config=quantization_config,
        trust_remote_code=True,
        device_map="auto",
        attn_implementation="eager"
    )
    
    print(f"Loading tokenizer from '{quantized_model_path}'...")
    tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("✅✅✅ 4-bit Quantized Model loaded successfully! Ready for deployment! ✅✅✅")

except Exception as e:
    print(f"❌ FATAL ERROR: Could not load the AI model. Reason: {e}")
    print("   (Hint: Make sure 'bitsandbytes' and 'accelerate' are installed in your environment: `pip install bitsandbytes accelerate`)")
    sys.exit(1)

# --- 3. Flask and SocketIO Setup ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'a-super-secret-key-that-you-should-change'
socketio = SocketIO(app, async_mode='eventlet')

# --- 4. Custom Streamer for Real-time Metrics ---
class SocketIOStreamer(TextStreamer):
    def __init__(self, tokenizer, sid):
        # Set skip_prompt=True so the template itself isn't streamed to the user
        super().__init__(tokenizer, skip_prompt=True, decode_only=False)
        self.sid = sid
        self.token_count = 0
        self.start_time = 0

    def on_finalized_text(self, text: str, stream_end: bool = False):
        if self.token_count == 0:
            self.start_time = time.time()
            socketio.emit('stream_start', room=self.sid)

        self.token_count += 1
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        tokens_per_sec = self.token_count / elapsed_time if elapsed_time > 0 else 0
        
        socketio.emit('stream_token', {
            'token': text,
            'token_count': self.token_count,
            'tokens_per_sec': round(tokens_per_sec, 2)
        }, room=self.sid)
        
        if stream_end:
            socketio.emit('stream_end', {'total_time': round(elapsed_time, 2)}, room=self.sid)
        
        # Force eventlet to yield control and send the buffered message immediately.
        socketio.sleep(0)

# --- 5. Flask Routes and WebSocket Event Handlers ---
@app.route('/')
def index():
    return render_template('index.html')

def generation_task(prompt, streamer):
    """The actual model generation, run in a background thread."""
    try:
        # --- START OF THE CHANGE ---
        # Format the raw prompt into the official Phi-3 Instruct format.
        formatted_prompt = f"<|user|>\n{prompt}<|end|>\n<|assistant|>"
        
        # Tokenize the formatted prompt
        inputs = tokenizer(formatted_prompt, return_tensors='pt').to(model.device)
        # --- END OF THE CHANGE ---
        
        _ = model.generate(
            **inputs, 
            max_new_tokens=512, 
            pad_token_id=tokenizer.eos_token_id,
            streamer=streamer,
            use_cache=False
        )
    except Exception as e:
        print(f"--- ERROR IN GENERATION THREAD ---")
        print(f"Error: {e}")
        socketio.emit('generation_error', {'error': str(e)}, room=streamer.sid)

@socketio.on('generate_text')
def handle_generate_text(data):
    from flask import request
    prompt = data['prompt'] # This is the raw prompt from the user
    sid = request.sid
    streamer = SocketIOStreamer(tokenizer, sid)
    
    # We pass the raw prompt to the task, which will then format it
    socketio.start_background_task(generation_task, prompt, streamer)

# --- 6. Run the Application ---
if __name__ == '__main__':
    print("\n" + "="*60)
    print("Starting Flask-SocketIO server.")
    print(f"Web App is running at: http://127.0.0.1:5000 (Local)")
    print("="*60 + "\n")
    socketio.run(app, host='0.0.0.0', port=5000)
