# WebApp Base Project

This project is a simple web application consisting of a **frontend** (using Vue 3 and Vuetify) and a **backend** (using FastAPI).  
The frontend lets you upload images and apply a blur effect to them.

## Prerequisites

- **Node.js** (for the frontend) - [Download here](https://nodejs.org/en/download/)
- **Python 3.10+** (for the backend) - [Download here](https://www.python.org/downloads/)
- **Conda** (for environment management) - [Install here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)

---

## Setup Instructions

### 1. Backend Setup

1. **Navigate to the backend folder:**
   ```sh
   cd backend
   ```

2. **Set up the Python environment:**
   If you're using Conda, create and activate the environment:
   ```sh
   conda create --name simple-webapp python=3.10
   conda activate simple-webapp
   ```

3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
4. **Install Sam2**
   > Setup Sam2 and the correct pytorch-cuda version (if used with gpu)
   ```sh
   conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
   ```
   > Download a checkpoint and config from and add them to the config- and checkpoints
   > folder inside the src folder.
   ```
   https://github.com/facebookresearch/sam2?tab=readme-ov-file#sam-21-checkpoints
   ```
   
5. **Setup GPU Usage**
   > Ensure that both CUDA (12.4) and cuDNN (9.7.0) are installed and the PATH variables are set.

   > Replace the cv2-package in your conda environments <br> site-packages (<Path-To-Conda>/envs/simple-webapp/Lib/site-packages)
   > <br> with the cv2-package in other/cv2.zip <br>
   > <br>
   > The zip contains a custom build OpenCV-Build with CUDA support. It is build for CUDA 12.4 and cuDNN 9.7.0. 
   > Other (newer) versions might work as well.
   

6. **Run the backend server:**
   ```sh
   cd src
   python main.py
   ```

   Your backend should now be running at `http://localhost:8000`.

---

### 2. Frontend Setup

1. **Navigate to the frontend folder:**
   ```sh
   cd frontend
   ```

2. **Install the dependencies:**
   ```sh
   npm install
   ```

3. **Start the development server:**
   ```sh
   npm run dev
   ```

   Your frontend should now be running at `http://localhost:[PORT]`.

---

## Project Structure

```
webapp-base-project/
├─ backend/
│  ├─ requirements.txt            # Backend dependencies
│  └─ src/
│     ├─ image_processing.py       # Image processing logic (e.g., blur effect)
│     └─ main.py                   # FastAPI entry point
├─ frontend/
│  ├─ index.html                   # HTML entry point for the frontend
│  ├─ package.json                 # Frontend dependencies and project details
│  ├─ src/
│  │  ├─ App.vue                   # Root Vue component
│  │  ├─ assets/                   # Static assets like CSS, images
│  │  ├─ components/               # Vue components
│  │  │  ├─ GalleryComponent.vue   # Component for displaying image gallery
│  │  │  └─ MainComponent.vue      # Component for image upload and blur functionality
│  │  ├─ main.js                   # Frontend entry point
│  │  ├─ router/
│  │  │  └─ index.js               # Vue Router configuration
│  │  ├─ store.js                  # Global store for state management
│  │  └─ views/                    # Vue views (pages)
│  └─ vite.config.js               # Vite configuration for building the frontend
└─ README.md                       
```

## Useful links
- **Vue3 Introduction:** https://vuejs.org/guide/introduction.html
- **Vuetify Documentation:** https://vuetifyjs.com/en/components/buttons/#usage
- **FastAPI Examples:** https://fastapi.tiangolo.com/tutorial/first-steps/