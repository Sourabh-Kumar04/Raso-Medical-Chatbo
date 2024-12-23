# End-to-end-Medical-Chatbot-using-Llama2

## Steps to run the project

### Clone the repository
```bash
git clone https://github.com/Sourabh-Kumar04/End-to-end-Medical-Chatbot.git
```

### Create virtual environment and activate that environment
```bash
conda create -n mchatbot python=3.8 -y
```
```bash
conda activate mchatbot
```
```bash
pip install -r requirements.txt
```

### Create a '.env' file in the root directory and your Pinecone credentials as follows

```ini
PINECONE_API_KEY = ""
PINECONE_API_ENV =""
```

### Download the quantize modelforn the link provided in model folder and keep the model in the model directory

#### Downlaod the Model 
llama-2-7b-chat.ggmlv3.q4_0.bin

#### From the following link:
https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main

```bash
# run the following command
python store_index.py
```

```bash 
# Finally run the following command
python app.py
```

```bash
open up localhost: 
```

## Techstack Used:
- Python
- Langchain
- Flask
- Pinecone

