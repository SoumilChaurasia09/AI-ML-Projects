import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw, ImageFilter

class SyntheticFaceGenerator:
    """
    Generates synthetic face images for training detection and recognition models.
    """
    def __init__(self, img_size=128, crop_size=64):
        self.img_size = img_size
        self.crop_size = crop_size
        
    def _generate_identity_features(self, identity_id):
        # Seed generator based on identity_id to keep features consistent for the same person
        rng = np.random.default_rng(identity_id)
        
        # Skin tone (RGB)
        skin_r = rng.integers(180, 255)
        skin_g = rng.integers(130, 210)
        skin_b = rng.integers(100, 180)
        skin_color = (skin_r, skin_g, skin_b)
        
        # Eye color (RGB)
        eye_color = (int(rng.integers(20, 100)), int(rng.integers(40, 150)), int(rng.integers(20, 100)))
        
        # Hair color (RGB)
        hair_color = (int(rng.integers(10, 80)), int(rng.integers(10, 60)), int(rng.integers(10, 50)))
        
        # Facial proportion variations
        face_aspect_ratio = rng.uniform(0.85, 1.15)  # width / height ratio
        eye_spacing = rng.uniform(0.22, 0.28)        # relative to face width
        eye_height = rng.uniform(0.35, 0.45)         # relative to face height (from top)
        nose_length = rng.uniform(0.15, 0.25)        # relative to face height
        mouth_width = rng.uniform(0.30, 0.45)        # relative to face width
        mouth_height = rng.uniform(0.70, 0.80)       # relative to face height (from top)
        
        return {
            "skin_color": skin_color,
            "eye_color": eye_color,
            "hair_color": hair_color,
            "face_aspect_ratio": face_aspect_ratio,
            "eye_spacing": eye_spacing,
            "eye_height": eye_height,
            "nose_length": nose_length,
            "mouth_width": mouth_width,
            "mouth_height": mouth_height
        }

    def generate_image(self, identity_id=None, has_face=True):
        """
        Generates an image of size (img_size, img_size).
        If has_face is True, draws a face with characteristics determined by identity_id.
        If identity_id is None and has_face is True, picks a random identity.
        """
        # Create solid clean dark background
        img = Image.new("RGB", (self.img_size, self.img_size), (30, 30, 40))
        draw = ImageDraw.Draw(img)

        # Determine identity
        if identity_id is None:
            identity_id = random.randint(1, 1000)
            
        features = self._generate_identity_features(identity_id)
        skin_color = features["skin_color"]

        if not has_face:
            # Return background-only image with zero bounding box
            return img, (0.0, 0.0, 0.0, 0.0), None
        
        # Add random variations to the identity (pose, scale, illumination) for this specific rendering
        face_h = random.randint(35, 75)
        face_w = int(face_h * features["face_aspect_ratio"])
        # Ensure w/h are within bounds
        face_w = max(25, min(face_w, 80))
        
        # Position face randomly such that it stays within image borders mostly
        x_min = random.randint(5, self.img_size - face_w - 5)
        y_min = random.randint(5, self.img_size - face_h - 5)
        x_max = x_min + face_w
        y_max = y_min + face_h
        
        cx = (x_min + x_max) // 2
        cy = (y_min + y_max) // 2
        
        # 1. Draw Neck
        neck_w = int(face_w * 0.4)
        draw.rectangle([cx - neck_w//2, cy + face_h//4, cx + neck_w//2, y_max + 10], fill=skin_color)

        # 2. Draw Hair (back layer)
        hair_color = features["hair_color"]
        hair_w = int(face_w * 1.05)
        hair_h = int(face_h * 0.95)
        draw.ellipse([cx - hair_w//2, cy - hair_h//2 - 5, cx + hair_w//2, cy + hair_h//2], fill=hair_color)

        # 3. Draw Face Oval (skin)
        draw.ellipse([x_min, y_min, x_max, y_max], fill=skin_color, outline=(int(skin_color[0]*0.85), int(skin_color[1]*0.85), int(skin_color[2]*0.85)))
        
        # 4. Draw Hair (front bangs layer)
        draw.chord([x_min, y_min, x_max, y_min + int(face_h * 0.35)], start=180, end=360, fill=hair_color)

        # 5. Draw Eyes
        eye_y = y_min + int(face_h * features["eye_height"])
        eye_spacing_px = int(face_w * features["eye_spacing"])
        eye_r = max(3, int(face_w * 0.08))
        
        # Left eye
        left_eye_x = cx - eye_spacing_px
        draw.ellipse([left_eye_x - eye_r, eye_y - eye_r, left_eye_x + eye_r, eye_y + eye_r], fill=(255, 255, 255))
        draw.ellipse([left_eye_x - eye_r//2, eye_y - eye_r//2, left_eye_x + eye_r//2, eye_y + eye_r//2], fill=features["eye_color"])
        
        # Right eye
        right_eye_x = cx + eye_spacing_px
        draw.ellipse([right_eye_x - eye_r, eye_y - eye_r, right_eye_x + eye_r, eye_y + eye_r], fill=(255, 255, 255))
        draw.ellipse([right_eye_x - eye_r//2, eye_y - eye_r//2, right_eye_x + eye_r//2, eye_y + eye_r//2], fill=features["eye_color"])

        # 6. Draw Nose
        nose_start_y = eye_y
        nose_len_px = int(face_h * features["nose_length"])
        nose_end_y = nose_start_y + nose_len_px
        nose_color = (int(skin_color[0]*0.8), int(skin_color[1]*0.8), int(skin_color[2]*0.8))
        draw.line([(cx, nose_start_y), (cx, nose_end_y)], fill=nose_color, width=max(1, face_w // 25))
        
        # 7. Draw Mouth (happy curved arc)
        mouth_y = y_min + int(face_h * features["mouth_height"])
        mouth_w_px = int(face_w * features["mouth_width"])
        mouth_color = (190, 80, 80) # warm coral red
        
        # Draw a curved arc for smile instead of straight line
        draw.arc([cx - mouth_w_px//2, mouth_y - mouth_w_px//4, cx + mouth_w_px//2, mouth_y + mouth_w_px//4], start=0, end=180, fill=mouth_color, width=max(1, face_w // 18))

        # Bounding box as normalized [x_min, y_min, x_max, y_max] normalized:
        bbox = (x_min / self.img_size, y_min / self.img_size, x_max / self.img_size, y_max / self.img_size)
        
        # Crop the face with a slight margin
        margin = int(face_h * 0.1)
        crop_x_min = max(0, x_min - margin)
        crop_y_min = max(0, y_min - margin)
        crop_x_max = min(self.img_size, x_max + margin)
        crop_y_max = min(self.img_size, y_max + margin)
        
        face_crop = img.crop((crop_x_min, crop_y_min, crop_x_max, crop_y_max))
        face_crop = face_crop.resize((self.crop_size, self.crop_size), Image.Resampling.BILINEAR)

        return img, bbox, face_crop


class FaceDetectionDataset(Dataset):
    """
    Dataset for Face Detection (bounding box regression & binary classification).
    """
    def __init__(self, num_samples=1000, has_face_ratio=0.8, img_size=128):
        self.generator = SyntheticFaceGenerator(img_size=img_size)
        self.num_samples = num_samples
        self.has_face_ratio = has_face_ratio
        
        # Pre-generate or lazily generate? Let's pre-generate to speed up training epochs
        self.images = []
        self.labels = [] # List of tuples: (is_face, [x_min, y_min, x_max, y_max])
        
        for i in range(num_samples):
            has_face = random.random() < has_face_ratio
            identity_id = random.randint(1, 100) if has_face else None
            img, bbox, _ = self.generator.generate_image(identity_id=identity_id, has_face=has_face)
            
            # Convert image to float tensor [3, H, W] in [0, 1]
            img_arr = np.array(img).astype(np.float32) / 255.0
            img_tensor = torch.tensor(img_arr).permute(2, 0, 1)
            
            self.images.append(img_tensor)
            self.labels.append((1.0 if has_face else 0.0, torch.tensor(bbox, dtype=torch.float32)))
            
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        img = self.images[idx]
        is_face, bbox = self.labels[idx]
        return img, is_face, bbox


class FaceRecognitionTripletDataset(Dataset):
    """
    Dataset for Face Recognition (Triplet training: Anchor, Positive, Negative).
    """
    def __init__(self, num_identities=20, samples_per_identity=15, crop_size=64):
        self.generator = SyntheticFaceGenerator(crop_size=crop_size)
        self.num_identities = num_identities
        self.samples_per_identity = samples_per_identity
        
        # Generate face crops for all identities
        # Dict: identity_id -> list of tensors [3, crop_size, crop_size]
        self.face_crops = {}
        for id_val in range(1, num_identities + 1):
            self.face_crops[id_val] = []
            for _ in range(samples_per_identity):
                _, _, face_crop = self.generator.generate_image(identity_id=id_val, has_face=True)
                
                crop_arr = np.array(face_crop).astype(np.float32) / 255.0
                crop_tensor = torch.tensor(crop_arr).permute(2, 0, 1)
                self.face_crops[id_val].append(crop_tensor)
                
    def __len__(self):
        # We define epoch size as total number of triplets we want to draw
        return self.num_identities * self.samples_per_identity * 3
        
    def __getitem__(self, idx):
        # Select anchor identity
        anchor_id = random.randint(1, self.num_identities)
        
        # Select two different samples from anchor identity
        a_idx, p_idx = random.sample(range(self.samples_per_identity), 2)
        anchor_img = self.face_crops[anchor_id][a_idx]
        positive_img = self.face_crops[anchor_id][p_idx]
        
        # Select negative identity
        negative_id = anchor_id
        while negative_id == anchor_id:
            negative_id = random.randint(1, self.num_identities)
            
        # Select random sample from negative identity
        n_idx = random.randint(0, self.samples_per_identity - 1)
        negative_img = self.face_crops[negative_id][n_idx]
        
        return anchor_img, positive_img, negative_img
