# 📹Video Conversion from SD to HD using Diffusion Model

This project implements an AI-powered pipeline to upscale low-resolution **SD videos** to high-definition **HD** using **Python**, **PyTorch**, and advanced **Diffusion Models**. It processes videos frame-by-frame, intelligently enhancing resolution while maintaining temporal consistency to prevent flickering.

***

### 📖 Project Overview

Upscaling legacy video content is a major challenge, as traditional algorithms often produce blurry or artificial-looking results. This project demonstrates a **deep learning-driven approach** to video super-resolution.

* A video is deconstructed into individual frames for processing.
* A **Denoising Diffusion Model** generates highly realistic, detailed HD versions of each frame.
* A temporal consistency mechanism ensures smooth transitions between frames, eliminating flicker.
* The final upscaled frames are reassembled into a high-quality HD video file.

The pipeline is designed to be **modular, efficient, and capable of producing state-of-the-art results**.

***

### 🚀 Features

* **End-to-end Pipeline:** Raw SD video → upscaled HD video.
* **High-Fidelity Generation:** Uses Diffusion Models to create sharp, realistic details.
* **Temporal Consistency:** Sliding-window logic prevents motion artifacts and flickering between frames.
* **GPU-Optimized:** Batch processing achieves a **2x speedup** over naive frame-by-frame methods.

***

### 🛠️ Technologies Used

* Python 3.x
* PyTorch
* OpenCV-Python
* Pillow (PIL)
* NumPy

***

### 📂 Project Structure
```
|── README.md              # Project Documentation
├── upscale.py             # Core script for video upscaling
├── requirements.txt       # Required Python packages
├── models/                # Folder for trained model weights (.pth)
├── input_videos/          # Folder for your input SD videos
└── output_videos/         # Folder for the final HD output videos
```

***

### ⚡ Getting Started

1.  **Clone the repository**
    ```sh
    git clone https://github.com/JoshDilipkumarPatel/Video-Conversion-from-SD-to-HD-using-Diffusion-Model.git
    cd Video-Conversion-from-SD-to-HD-using-Diffusion-Model
    ```

2.  **Install dependencies**
    ```sh
    pip install -r requirements.txt
    ```
    The `requirements.txt` file should contain:
    ```
    torch
    torchvision
    torchaudio
    opencv-python-headless
    numpy
    Pillow
    ```

3.  **Download the Pre-trained Model**
    Place your trained model file (e.g., `model.pth`) into the `models/` directory.

4.  **Add Your Video**
    Place the SD video you want to upscale (e.g., `my_video_sd.mp4`) into the `input_videos/` folder.

***

### ▶️ How to Run

Execute the main script from the terminal, specifying the input video and model.

```sh
python upscale.py --input_video input_videos/my_video_sd.mp4 --model_path models/model.pth
```

The final HD video will be saved in the `output_videos/` directory.

-----

### 📊 Example Results

The script will output progress to the console and produce a high-quality video file. Visual quality is the key metric.

-----

*Left: Original SD Frame. Right: Upscaled HD Frame.*

  * **Quantitative:** Achieved a **2x speedup** in frame processing.
  * **Qualitative:** Significant reduction in pixelation and artifacting, with sharp, coherent details.

-----

### ✨ Future Improvements

  * **Hyperparameter Tuning:** Use Optuna or Ray Tune to optimize the Diffusion Model's performance.
  * **GUI Interface:** Build a simple front-end with Gradio or Streamlit for easy use.
  * **Model Comparison:** Benchmark against other super-resolution models like ESRGAN or SwinIR.
  * **Real-time Processing:** Explore model quantization and optimization for live video streams.

-----
