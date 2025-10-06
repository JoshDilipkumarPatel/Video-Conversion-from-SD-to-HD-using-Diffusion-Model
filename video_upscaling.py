import os
import cv2
import torch
from tqdm import tqdm
import numpy as np
from diffusers import StableDiffusionUpscalePipeline
from PIL import Image
from google.colab import drive

def check_video_path(video_path):
    if not os.path.isfile(video_path):
        if video_path.startswith("/content/drive/") and not os.path.isdir("/content/drive"):
            raise RuntimeError(
                f"Google Drive not mounted. Please run:\n"
                "from google.colab import drive\n"
                "drive.mount('/content/drive')"
            )
        else:
            raise RuntimeError(f"Input video not found: {video_path}")

def extract_frames(video_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")

    idx = 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(out_dir, f"frame_{idx:06d}.png"), frame)
        idx += 1

    cap.release()
    if idx == 0:
        raise RuntimeError(f"No frames extracted. Check video file: {video_path}")

    print(f"[INFO] Extracted {idx} frames at {fps:.2f} FPS, size=({w}x{h})")
    return idx, fps, (w, h)

def load_upscaler(model_name_or_path="stabilityai/stable-diffusion-x4-upscaler"):
    if torch.cuda.is_available():
        pipe = StableDiffusionUpscalePipeline.from_pretrained(
            model_name_or_path, torch_dtype=torch.float16
        )
        pipe = pipe.to("cuda")
    else:
        pipe = StableDiffusionUpscalePipeline.from_pretrained(
            model_name_or_path, torch_dtype=torch.float32
        )
        pipe = pipe.to("cpu")
    pipe.enable_attention_slicing()
    return pipe

def read_img(path):
    bgr = cv2.imread(path)
    if bgr is None:
        raise RuntimeError(f"Cannot read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb

def optical_flow_warp(prev_frame, flow):
    h, w = flow.shape[:2]
    flow_map = np.zeros_like(flow, dtype=np.float32)
    flow_map[...,0] = np.arange(w)
    flow_map[...,1] = np.arange(h)[:,None]
    remap = (flow_map + flow).astype(np.float32)
    map_x = remap[...,0]
    map_y = remap[...,1]
    warped = cv2.remap(prev_frame, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return warped

def main(args):
    check_video_path(args.input)
    print(f"[INFO] Processing video: {args.input}")

    frames_dir = args.frames_dir
    os.makedirs(frames_dir, exist_ok=True)

    if not args.skip_extract:
        nframes, fps, (w,h) = extract_frames(args.input, frames_dir)
    else:
        files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".png")])
        nframes = len(files)
        fps = args.fps or 30
        if nframes == 0:
            raise RuntimeError(f"No frames found in {frames_dir}. Cannot continue.")

    pipe = load_upscaler(args.upscaler_model)

    prev_enh = None
    prev_gray = None
    out_frames = []

    files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".png")])
    if len(files) == 0:
        raise RuntimeError(f"No frames to process in {frames_dir}.")

    print(f"[INFO] Found {len(files)} frames to process")

    for i, fname in enumerate(tqdm(files)):
        path = os.path.join(frames_dir, fname)
        frame_rgb = read_img(path)
        frame_pil = Image.fromarray(frame_rgb)
        up = pipe(prompt="", image=frame_pil, num_inference_steps=20, guidance_scale=4.0).images[0]
        up_np = np.array(up)

        if prev_enh is not None:
            cur_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev_gray, cur_gray,
                                                None, 0.5, 3, 15, 3, 5, 1.2, 0)
            scale = up_np.shape[1] / frame_rgb.shape[1]
            flow_hr = cv2.resize(flow, (up_np.shape[1], up_np.shape[0])) * scale
            warped_prev = optical_flow_warp(prev_enh, flow_hr)
            alpha = 0.6
            blended = (alpha * warped_prev + (1-alpha) * up_np).astype(np.uint8)
            out = blended
            prev_gray = cur_gray
        else:
            out = up_np
            prev_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)

        prev_enh = out
        out_frames.append(out)

    if not out_frames:
        raise RuntimeError("No output frames generated!")

    h_out, w_out = out_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (w_out, h_out))
    for fr in out_frames:
        writer.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"Saved upscaled video: {args.output}")

if __name__ == "__main__":
    class Args: pass
    args = Args()
    args.input = "/content/drive/MyDrive/harshit.mp4"
    args.output = "/content/drive/MyDrive/output_hd.mp4"
    args.frames_dir = "frames"
    args.upscaler_model = "stabilityai/stable-diffusion-x4-upscaler"
    args.skip_extract = False
    args.fps = None

    main(args)
