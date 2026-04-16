# Live Performance Dashboard for Fine-Tuned Phi-3 Medical Model

![Demo GIF](demo.gif)

This project is a high-performance web dashboard for interacting with a 4-bit quantized Phi-3 model, fine-tuned on a custom medical dataset. It provides real-time token generation and detailed performance metrics.

---

## ⚠️ Model Download Required

This repository contains the application code only. The fine-tuned model (2.2 GB) is hosted separately due to GitHub's file size limits.

**You must download the model files and place them in this folder to run the project.**

---

## 🛠️ Tech Stack

* **Model:** Phi-3-mini (Fine-Tuned & 4-bit Quantized)
* **ML Libraries:** PyTorch, Hugging Face `transformers`, `bitsandbytes`, `accelerate`
* **Web Framework:** Streamlit
* **Deployment:** GitHub (Code) + Google Drive (Model)

---

## 📂 How to Run Locally

1.  **Clone the code repository:**
    ```bash
    git clone [https://github.com/YOUNG-G0D/phi3-medical-dashboard.git](https://github.com/YOUNG-G0D/phi3-medical-dashboard.git)
    cd phi3-medical-dashboard
    ```

2.  **Download the Model (2.2 GB):**
    * **[Click here to download the `model.zip` file](https://your-google-drive-share-link-goes-here)**
    * Unzip the file. You will have a folder named `quantized_phi3_medical_model_4bit_FINAL`.
    * **Go inside** that unzipped folder. **Copy all the files** from inside it (like `config.json`, `model.safetensors`, etc.).
    * **Paste all of those files** into the `phi3-medical-dashboard` directory you cloned, right next to the `app.py` file.

3.  **Create the environment and install dependencies:**
    ```bash
    # (Recommended) Create a conda environment
    conda create --name model_demo python=3.10
    conda activate model_demo
    
    # Install dependencies
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py