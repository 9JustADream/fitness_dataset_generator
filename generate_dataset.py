import os
import json
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from transformers import AutoModelForImageSegmentation
from torchvision import transforms
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
import albumentations as A


# КОНФИГУРАЦИЯ (можно менять)
NUM_IMAGES = 10                # количество генерируемых изображений
PROB_REAL_REF = 0.1            # вероятность использовать реальный референс (вместо синтетической генерации)
PROB_MULTI_OBJECT = 0.35       # вероятность того, что на изображении будет 2 объекта
BASE_REF_FOLDER = "refs"       # папка с референсами (подпапки по названиям объектов)
OUTPUT_IMAGES = "generated"    # папка для сохранения изображений
JSON_PATH = "dataset.json"     # файл с разметкой

# Параметры аугментации погоды (вероятности)
AUG_PROBS = {
    'fog': 0.1,
    'rain': 0.1,
    'brightness': 0.15
}

# Доступные объекты (идентификаторы должны совпадать с именами подпапок в refs)
ALL_OBJECTS = [
    "pullup_bar_high",
    "monkey_bars",
    "monkey_bars_wave",
    "rings",
    "parallel_bars",
    "pushup_bars",
    "dip_bench",
    "wall_bars_vertical",
    "situp_bench",
    "situp_bench_incline",
    "plyo_box"
]

# Шаблоны промптов для каждого объекта (базовое описание)
OBJECT_TEMPLATES = {
    "pullup_bar_high": "high adult narrow slender horizontal bar, vertical proportions",
    "monkey_bars": "straight overhead ladder consisting of two parallel horizontal beams with evenly spaced rungs between them",
    "monkey_bars_wave": "wave snake shaped curved overhead ladder consisting of two parallel horizontal bars with wave shaped rungs between them",
    "rings": "workout similar pair parralel rings stand",
    "parallel_bars": "sport parallel horizontal bars pair",
    "pushup_bars": "very low push-up bars, horisontal proportions",
    "dip_bench": "low dip station with parallel bars in a row, metal frame, horisontal proportions",
    "wall_bars_vertical": "vertical narrow wall bars, climbing, vertical proportions",
    "situp_bench": "sit-up bench, ab bench, horisontal proportions",
    "situp_bench_incline": "incline sit-up bench, decline ab bench, horisontal proportions",
    "plyo_box": "plyo box, jump box, wooden plyometric box"
}

# Цвета и ракурсы для рандомизации
COLORS = ["black", "red", "matte blue", "matte green", "matte yellow", "orange", "purple",
          "green", "yellow", "blue", "grey", "brown", "silver", "matte black", "matte red", "camouflage"]
VIEWS = ["isometric view", "frontal view", "side view", "three-quarter view", "low angle"]

# Локации для фона
LOCATIONS = [
    "park with a lawn and trees",
    "urban fitness court with rubber flooring",
    "quiet courtyard with benches and pavement",
    "sports ground with artificial grass",
    "playground with soft tiles and fences",
    "city park with gravel paths and bushes",
    "schoolyard with concrete pavement",
    "open field with grass and wildflowers",
    "forest clearing with moss and plants"
]
TIMES = [
    "early morning, soft light, dew on grass",
    "bright midday sun, clear sky",
    "late afternoon, golden hour, long shadows",
    "evening, dusk, streetlights starting to glow",
    "night, illuminated by nearby lamps, dark sky"
]
CROWDS = [
    "no people",
    "one person jogging in the distance",
    "a few people walking by, blurred"
]


# Инициализация моделей
device = "cuda" if torch.cuda.is_available() else "cpu"

pipe_bg = StableDiffusionPipeline.from_pretrained(
    "SG161222/Realistic_Vision_V5.1_noVAE",
    torch_dtype=torch.float16,
    safety_checker=None
).to(device)
pipe_bg.scheduler = DDIMScheduler.from_config(pipe_bg.scheduler.config)

pipe = StableDiffusionPipeline.from_pretrained(
    "SG161222/Realistic_Vision_V5.1_noVAE",
    torch_dtype=torch.float16,
    safety_checker=None
).to(device)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

pipe.load_ip_adapter(
    "h94/IP-Adapter",
    subfolder="models",
    weight_name="ip-adapter-plus_sd15.bin"
)
pipe.set_ip_adapter_scale(0.8)

birefnet = AutoModelForImageSegmentation.from_pretrained(
    'zhengpeng7/BiRefNet',
    trust_remote_code=True
)
torch.set_float32_matmul_precision('high')
birefnet.to(device)
birefnet.eval()
birefnet.half()


def generate_random_background_prompt():
    loc = random.choice(LOCATIONS)
    t = random.choice(TIMES)
    crowd = random.choice(CROWDS)
    return f"{loc}, {t}, {crowd}, photorealistic"

def generate_object_prompt(object_id):
    base = OBJECT_TEMPLATES.get(object_id, object_id)
    color = random.choice(COLORS)
    view = random.choice(VIEWS)
    return f"A minimalistic outdoor {color} color, {base}, {view}, isolated on a white background, no people, plain background, 3d render style, clean sharp focus"

def extract_object(birefnet, image, image_size=(1024, 1024)):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0).to(device).half()
    with torch.no_grad():
        preds = birefnet(input_tensor)[-1].sigmoid().cpu()
    pred = preds[0].squeeze(0)
    mask = transforms.ToPILImage()(pred).resize(image.size, Image.LANCZOS)
    image_with_alpha = image.convert('RGBA')
    image_with_alpha.putalpha(mask)
    return image_with_alpha, mask

def crop_to_object(image_with_alpha):
    alpha = image_with_alpha.getchannel('A')
    bbox = alpha.getbbox()
    if bbox:
        return image_with_alpha.crop(bbox), bbox
    else:
        return image_with_alpha, (0, 0, image_with_alpha.width, image_with_alpha.height)

def generate_object_with_references(reference_folder, prompt, num_refs=3):
    ref_images = []
    for f in os.listdir(reference_folder):
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = Image.open(os.path.join(reference_folder, f)).convert("RGB")
            ref_images.append(img)
    if not ref_images:
        raise ValueError("Нет референсных изображений")
    selected = random.sample(ref_images, min(num_refs, len(ref_images)))
    images = pipe(
        prompt=prompt,
        ip_adapter_image=[selected],
        num_inference_steps=50,
        generator=torch.manual_seed(random.randint(0, 1000000)),
        negative_prompt="low quality, blurry, distorted, ugly, extra bars, clutter"
    ).images
    return images[0]

def extract_reference_object(ref_folder, object_id, aug_prob=0.5):
    obj_folder = os.path.join(ref_folder, object_id)
    files = [f for f in os.listdir(obj_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not files:
        raise ValueError(f"Нет референсов в {obj_folder}")
    img_path = os.path.join(obj_folder, random.choice(files))
    img = Image.open(img_path).convert("RGB")
    obj_cut, _ = extract_object(birefnet, img)
    obj_cut_cropped, _ = crop_to_object(obj_cut)
    if random.random() < aug_prob:
        if random.random() < 0.5:
            obj_cut_cropped = ImageOps.mirror(obj_cut_cropped)
        if random.random() < 0.3:
            angle = random.uniform(-10, 10)
            obj_cut_cropped = obj_cut_cropped.rotate(angle, expand=True, fillcolor=(0,0,0,0))
        if random.random() < 0.4:
            enhancer = ImageEnhance.Brightness(obj_cut_cropped)
            obj_cut_cropped = enhancer.enhance(random.uniform(0.7, 1.3))
    return obj_cut_cropped

def apply_weather_augmentation(image, probs):
    img_np = np.array(image)
    transforms_list = []
    if random.random() < probs.get('fog', 0.1):
        transforms_list.append(A.RandomFog(fog_coef_range=(0.3, 0.5), alpha_coef=0.1, p=1.0))
    if random.random() < probs.get('rain', 0.1):
        transforms_list.append(A.RandomRain(
            rain_type='drizzle',
            slant_range=(-10, 10),
            drop_length_range=(10, 30),
            drop_width_range=(1, 3),
            brightness_coefficient=0.7,
            p=1.0
        ))
    if random.random() < probs.get('brightness', 0.15):
        transforms_list.append(A.RandomBrightnessContrast(
            brightness_limit=(-0.3, 0.3),
            contrast_limit=(-0.15, 0.15),
            p=1.0
        ))
    if not transforms_list:
        return image
    transform = A.Compose(transforms_list)
    augmented = transform(image=img_np)['image']
    return Image.fromarray(augmented)

def place_objects(bg, objects_info, max_attempts=10):
    bg_w, bg_h = bg.size
    placed = []
    for obj_img, obj_id in objects_info:
        base_scale = random.uniform(0.45, 0.58) if len(objects_info) == 1 else random.uniform(0.35, 0.45)
        new_w = int(bg_w * base_scale)
        new_h = int(obj_img.height * new_w / obj_img.width)
        obj_resized = obj_img.resize((new_w, new_h), Image.LANCZOS)
        best_attempt = None
        best_overlap = float('inf')
        for _ in range(max_attempts):
            x = random.randint(0, bg_w - new_w)
            y = bg_h - new_h - random.randint(0, 10)
            y = max(0, y)
            overlap = 0
            for p in placed:
                px, py, pw, ph = p['bbox']
                ix1 = max(x, px)
                iy1 = max(y, py)
                ix2 = min(x+new_w, px+pw)
                iy2 = min(y+new_h, py+ph)
                if ix2 > ix1 and iy2 > iy1:
                    overlap += (ix2-ix1)*(iy2-iy1)
            if overlap < best_overlap:
                best_overlap = overlap
                best_attempt = (x, y)
                if overlap == 0:
                    break
        x, y = best_attempt
        placed.append({
            'image': obj_resized,
            'object_id': obj_id,
            'bbox': [x, y, x+new_w, y+new_h]
        })
    return placed

def generate_multi_composite(background_prompt, object_list, ref_folder, image_id, aug_probs=None, use_real_prob=0.0):
    bg = pipe_bg(background_prompt, num_inference_steps=50).images[0]
    objects_info = []
    for obj_id in object_list:
        if random.random() < use_real_prob:
            obj_img = extract_reference_object(ref_folder, obj_id)
            objects_info.append((obj_img, obj_id))
        else:
            obj_prompt = generate_object_prompt(obj_id)
            num_refs = random.randint(3, 5)
            obj_raw = generate_object_with_references(os.path.join(ref_folder, obj_id), obj_prompt, num_refs=num_refs)
            obj_cut, _ = extract_object(birefnet, obj_raw)
            obj_cut_cropped, _ = crop_to_object(obj_cut)
            objects_info.append((obj_cut_cropped, obj_id))

    placed = place_objects(bg, objects_info)
    for item in placed:
        bg.paste(item['image'], (item['bbox'][0], item['bbox'][1]), item['image'])

    if aug_probs:
        adjusted_probs = aug_probs.copy()
        if len(object_list) > 1:
            for key in ['fog', 'rain', 'brightness']:
                if key in adjusted_probs:
                    adjusted_probs[key] = adjusted_probs[key] * 0.5
        bg = apply_weather_augmentation(bg, adjusted_probs)

    filename = f"composite_multi_{image_id:04d}.png"
    bg.save(os.path.join(OUTPUT_IMAGES, filename))
    annotations = {item['object_id']: item['bbox'] for item in placed}
    return filename, annotations


def main():
    os.makedirs(OUTPUT_IMAGES, exist_ok=True)
    if os.path.exists(JSON_PATH):
        with open(JSON_PATH, 'r') as f:
            data = json.load(f)
    else:
        data = {"images": []}

    for i in range(1, NUM_IMAGES + 1):
        bg_prompt = generate_random_background_prompt()
        if random.random() < PROB_MULTI_OBJECT:
            num_objs = 2
        else:
            num_objs = 1
        selected_objects = [random.choice(ALL_OBJECTS) for _ in range(num_objs)]

        fname, boxes = generate_multi_composite(
            bg_prompt,
            selected_objects,
            BASE_REF_FOLDER,
            image_id=i,
            aug_probs=AUG_PROBS,
            use_real_prob=PROB_REAL_REF
        )
        data["images"].append({
            "file_name": fname,
            "objects": boxes
        })
        with open(JSON_PATH, "w") as f:
            json.dump(data, f, indent=2)
        print(f"✅ {fname} сохранён, объекты: {list(boxes.keys())}")

if __name__ == "__main__":
    main()