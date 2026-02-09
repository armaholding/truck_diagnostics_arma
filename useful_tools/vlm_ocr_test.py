# # sample arabic characters: أ ب د ه و ط ي ك ل م ن س ع ف ص ق ر ت ح ج ش ض ظ غ

# # --- Using deepseek-ai/DeepSeek-OCR model ---
# # --- Load HF token FIRST (before ANY imports) ---
# import os
# from dotenv import load_dotenv

# load_dotenv()  # Load .env file from current directory
# hf_token = os.getenv("HF_TOKEN")

# if hf_token:
#     os.environ["HF_TOKEN"] = hf_token  # Set for Hugging Face Hub auth
#     print(f"✓ HF token loaded from .env")
# else:
#     print("⚠️ WARNING: HF_TOKEN not found in .env - downloads may be slow/rate-limited")
#     print("   Create .env file with: HF_TOKEN=your_hf_token_here")

# # --- CRITICAL PATCHES: MUST RUN BEFORE transformers import ---
# try:
#     # Patch 1: Missing LlamaFlashAttention2
#     from transformers.models.llama.modeling_llama import LlamaAttention
#     import transformers.models.llama.modeling_llama as llama_mod
#     if not hasattr(llama_mod, 'LlamaFlashAttention2'):
#         llama_mod.LlamaFlashAttention2 = LlamaAttention
#         print("✓ Patched LlamaFlashAttention2 -> LlamaAttention")
# except Exception as e:
#     print(f"⚠️ Patch 1 warning: {e}")

# try:
#     # Patch 2: Missing is_torch_fx_available
#     import transformers.utils.import_utils as import_utils_mod
#     if not hasattr(import_utils_mod, 'is_torch_fx_available'):
#         # Fallback: torch.fx is available in modern PyTorch
#         import torch
#         import_utils_mod.is_torch_fx_available = lambda: hasattr(torch, 'fx')
#         print("✓ Patched is_torch_fx_available")
# except Exception as e:
#     print(f"⚠️ Patch 2 warning: {e}")


# # Load test image
# image_file = os.path.join('cropped_parts', '20260204_140705_frame462_plate_number_0.89_3.jpg')
# if not os.path.exists(image_file):
#     print(f"ERROR: Image not found: {image_file}")
#     exit(1)

# img = cv2.imread(image_file)
# if img is None:
#     print("ERROR: OpenCV failed to load image")
#     exit(1)
# print(f"✓ Loaded plate crop: {img.shape[1]}x{img.shape[0]} pixels")

# # --- Normal imports after patches ---
# from transformers import AutoModel, AutoTokenizer
# import torch
# import cv2
# from huggingface_hub import hf_hub_download
# import time

# print("Loading DeepSeek-OCR model...")
# model = AutoModel.from_pretrained(
#     'deepseek-ai/DeepSeek-OCR',
#     trust_remote_code=True,
#     use_safetensors=True,
#     token=hf_token if hf_token else None
# ).eval().cuda()
# print("Loading tokenizer...")
# max_retries = 3
# for attempt in range(max_retries):
#     try:
#         tokenizer = AutoTokenizer.from_pretrained(
#             'deepseek-ai/DeepSeek-OCR',
#             trust_remote_code=True,
#             token=hf_token if hf_token else None,
#             timeout=30.0
#         )
#         print("✓ Tokenizer loaded")
#         break
#     except Exception as e:
#         print(f"⚠️ Tokenizer load attempt {attempt+1}/{max_retries} failed: {type(e).__name__}")
#         if attempt == max_retries - 1:
#             print("   → Falling back to offline mode...")
#             tokenizer = AutoTokenizer.from_pretrained(
#                 'deepseek-ai/DeepSeek-OCR',
#                 trust_remote_code=True,
#                 token=hf_token if hf_token else None,
#                 local_files_only=True
#             )
#             print("✓ Tokenizer loaded (offline mode)")
#         else:
#             time.sleep(2 ** attempt)
# print(f"✓ Model loaded on GPU: {torch.cuda.get_device_name(0)}")

# # Run OCR
# print("\nRunning OCR inference...")
# result = model.infer(
#     tokenizer,
#     prompt="<image>\nExtract license plate text only.",
#     image_file=image_file,
#     base_size=1024,
#     image_size=640,
#     crop_mode=True,
#     save_results=False,
#     output_path="."
# )

# # Show result
# raw_text = str(result).strip()
# print(raw_text)

# # Extract plate number
# import re
# matches = re.findall(r'[A-Z0-9]{4,10}', raw_text.upper())
# if matches:
#     plate = max(matches, key=len)
#     conf = min(1.0, len(plate) / 8.0)
#     print(f"\n✅ SUCCESS: Plate '{plate}' (confidence: {conf:.2f})")
# else:
#     print("\n⚠️ WARNING: No plate text detected")
#     print(f"   Raw output: '{raw_text}'")


# --- Using zai-org/GLM-OCR model ---
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
import os

image_file = os.path.join('cropped_parts', '20260204_140705_frame462_plate_number_0.89_3.jpg')
if not os.path.exists(image_file):
    raise FileNotFoundError(f"Image file not found: {image_file}")

MODEL_PATH = "zai-org/GLM-OCR"
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "url": image_file
            },
            {
                "type": "text",
                "text": "Extract the plate number, it has numbers and single arabic character. Output raw sequence with NO spaces, pipes, or extra texts."
                ""
            }
        ],
    }
]
processor = AutoProcessor.from_pretrained(MODEL_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForImageTextToText.from_pretrained(
    pretrained_model_name_or_path=MODEL_PATH,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    # device_map="auto",
)
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device)
inputs.pop("token_type_ids", None)
generated_ids = model.generate(
    **inputs,
    max_new_tokens=100,          # Constrained for plate numbers (was 8192)
    do_sample=False,             # Deterministic output
    pad_token_id=processor.tokenizer.pad_token_id
)
output_text = processor.decode(
    generated_ids[0][inputs["input_ids"].shape[1]:],
    skip_special_tokens=True  # Removes artifacts like  (was False)
).strip()
print(output_text)
final_output = output_text.replace(' ', '').replace('|', '').replace('\n', '').replace('\r', '').strip()
print(final_output)


# # --- Using Qwen/Qwen3-VL-2B-Instruct model ---
# from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
# import torch
# import os

# img_path = os.path.join('cropped_parts', '20260204_140705_frame462_plate_number_0.89_3.jpg')
# if not os.path.exists(img_path):
#     raise FileNotFoundError(f"Image file not found: {img_path}")

# device = "cuda" if torch.cuda.is_available() else "cpu"
# model_path = "Qwen/Qwen3-VL-2B-Instruct"

# model = Qwen3VLForConditionalGeneration.from_pretrained(
#     model_path,
#     torch_dtype=torch.float16,
# ).to(device)
# processor = AutoProcessor.from_pretrained(model_path)

# messages = [
#     {
#         "role": "user",
#         "content": [
#             {
#                 "type": "image",
#                 "image": img_path
#             },
#             {
#                 "type": "text", 
#                 "text": "Extract the plate number, it has numbers and single arabic character. Output raw sequence with NO spaces, pipes, or extra texts."
#             }
#         ],
#     }
# ]

# inputs = processor.apply_chat_template(
#     messages,
#     tokenize=True,
#     add_generation_prompt=True,
#     return_dict=True,
#     return_tensors="pt"
# ).to(model.device)

# generated_ids = model.generate(
#     **inputs,
#     max_new_tokens=100,          # Constrained for plate numbers (was 8192)
#     do_sample=False,             # Deterministic output
#     # pad_token_id=processor.tokenizer.pad_token_id
# )
# result = processor.decode(
#     generated_ids[0][inputs["input_ids"].shape[1]:],
#     skip_special_tokens=True  # Removes artifacts like  (was False)
# ).strip()
# print(result)
# print(f"RAW OUTPUT (hex): {[hex(ord(c)) for c in result[:20]]}")
# print(f"RAW OUTPUT (repr): {repr(result)}")
# print(f"STRIPPED OUTPUT: {result}")
# print(f"Contains Arabic? {any(0x0600 <= ord(c) <= 0x06FF for c in result)}")
# final_output = result.replace(' ', '').replace('|', '').replace('\n', '').replace('\r', '').strip()
# print(final_output)