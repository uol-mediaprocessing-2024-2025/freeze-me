# Frontend

## Setup
First download and install Node.js (https://nodejs.org/en/download/).
After installing open the project directory and run:
```sh
npm install
```

### Compile and Run
```sh
npm run dev
```

# Backend

## Creating environment
- conda env create -f environment.yml
- conda activate chatbot

## Downloading model
Download the LLaMA GGUF model weights from the HuggingFace [https://huggingface.co/openai/clip-vit-base-patch32/tree/main] and store the file in the assest/model folder.

## Running code
uvicorn main:app --port 8804 --reload
