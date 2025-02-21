# Freeze Me!
Freeze Me! is a project, in which we want to create dynamic, motion-illustrative images from video clips by isolating moving objects and visualizing their motion.


## Prerequisites

- **Node.js** (for the frontend) - [Download here](https://nodejs.org/en/download/)
- **Python 3.10+** (for the backend) - [Download here](https://www.python.org/downloads/)
- **Conda** (for environment management) - [Install here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)

Recommended, but optional:

- **CUDA 12.4** (for GPU Usage) - [Download here](https://developer.nvidia.com/cuda-12-4-0-download-archive)
- **cudNN 9.7.0** (for GPU Usage) - [Install here](https://developer.nvidia.com/cudnn-downloads)


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
   > folder inside the backend folder.
   ```
   https://github.com/facebookresearch/sam2?tab=readme-ov-file#sam-21-checkpoints
   ```
   
5. **Setup GPU Usage**
   > Ensure that both CUDA (12.4) and cuDNN (9.7.0) are installed and the PATH variables are set.
   
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

## Useful links
- **Vue3 Introduction:** https://vuejs.org/guide/introduction.html
- **Vuetify Documentation:** https://vuetifyjs.com/en/components/buttons/#usage
- **FastAPI Examples:** https://fastapi.tiangolo.com/tutorial/first-steps/