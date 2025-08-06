from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from tqdm import tqdm

# --- Load Models ---

yolo_model_id = "yolov8s.pt"
yolo_model = YOLO(yolo_model_id)
captioning_model_id = "Salesforce/blip-image-captioning-base"
blip_processor = BlipProcessor.from_pretrained(captioning_model_id, use_fast = True)
blip_model = BlipForConditionalGeneration.from_pretrained(captioning_model_id, torch_dtype=torch.float16, device_map="auto")