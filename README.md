# truck_diagnostics_arma
use yolov11 object tracking to detect truck parts and diagnose its status

![alt text](image.png)


YOLOv11 guide
https://docs.ultralytics.com/models/yolo11/

Arabic OCR for plate number
https://huggingface.co/NAMAA-Space/Qari-OCR-v0.3-VL-2B-Instruct
pip install transformers qwen_vl_utils accelerate>=0.26.0 PEFT -U
pip install -U bitsandbytes

Previouse OCR for plate number
https://huggingface.co/zai-org/GLM-OCR
pip install git+https://github.com/huggingface/transformers.git

FFMPEG installation guide
install ffmpeg
https://www.gyan.dev/ffmpeg/builds/
system variable path
or
winget install "FFmpeg (Essentials Build)"