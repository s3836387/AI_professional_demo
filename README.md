# Video Inference with Local LLM

This repository provides a demonstration of real-time video analysis using a local Large Language Model (LLM) hosted via Docker. The system allows users to define a Region of Interest (ROI) for targeted model inference.

---

## Setup and Execution

Follow the steps below to run the demo on your local machine.

### 1. Launch the LLM Server
Ensure your Docker engine is running and start the container hosting the LLM. 

### 2. Configure the Script
Open `infer_vid.py` in your text editor and update the following configuration variables:

* **OLLAMA_URL**: Set this to your local Docker endpoint (e.g., `http://localhost:11434`).
* **MODEL_NAME**: Input the exact name of the model you are utilizing.
* **VIDEO_PATH**: Provide the full path to your source video file.

### 3. Run the Inference Script
Execute the script from your terminal:

```bash
python infer_vid.py
```

### 4. Select Region of Interest (ROI)
1. When the window opens, click and drag to select the area of the video you want the model to analyze.
2. Once the selection is made, the video processing will start automatically.

