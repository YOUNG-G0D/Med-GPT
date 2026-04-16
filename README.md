# 🩺 Med-GPT: Clinical AI Dashboard (Phi-3 Fine-Tuned)

[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-blue)](https://huggingface.co/spaces/AMN04/Med-GPT)
[![Model on Hub](https://img.shields.io/badge/%F0%9F%A4%97%20Model-AMN04%2Fphi3--medical--model-orange)](https://huggingface.co/AMN04/phi3-medical-model)

This project features a **fine-tuned Phi-3-mini** model specialized for clinical inquiry and medical data analysis. The model has been optimized via **4-bit quantization** and is deployed as a live, streaming web application.

> **🚀 Live Demo:** [Click here to interact with Med-GPT on Hugging Face](https://huggingface.co/spaces/AMN04/Med-GPT)

---

## 🛠️ Technical Architecture

To overcome hardware limitations and deployment constraints, this project utilizes a hybrid cloud architecture:

* **Model Weights:** Hosted on Hugging Face Hub (`AMN04/phi3-medical-model`).
* **Optimization:** **4-bit Quantization (bitsandbytes)** was employed to reduce the memory footprint from ~14GB to **2.3GB**, enabling inference on standard CPU environments.
* **Inference Engine:** **Gradio SDK** with **Threaded Streaming (`TextIteratorStreamer`)**. This ensures words appear in real-time as they are processed, significantly improving user experience (UX) on CPU-bound instances.
* **Fine-Tuning:** Specialized on medical datasets for clinical accuracy and professional medical tone.

---

## 📂 Project Structure

* `app.py`: Production-grade Gradio application with threaded streaming logic.
* `requirements.txt`: Minimal dependency list for cloud-based inference.
* `quantization_config`: Integrated 4-bit loading protocol.

---

## 📂 How to Run Locally

If you wish to run the inference locally on your own machine (Ubuntu/Windows):

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUNG-G0D/Med-GPT.git](https://github.com/YOUNG-G0D/Med-GPT.git)
   cd Med-GPT
---

## 📂 How to Run Locally

If you wish to run the inference locally on your own machine (Ubuntu/Windows):

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUNG-G0D/Med-GPT.git](https://github.com/YOUNG-G0D/Med-GPT.git)
   cd Med-GPT
