import torch
from dataset import SyntheticFaceGenerator, FaceDetectionDataset, FaceRecognitionTripletDataset
from model import FaceDetectorCNN, FaceEmbeddingCNN

def test_synthetic_generator():
    print("Testing SyntheticFaceGenerator...")
    gen = SyntheticFaceGenerator(img_size=128, crop_size=64)
    img, bbox, face_crop = gen.generate_image(identity_id=1, has_face=True)
    
    assert img.size == (128, 128), f"Expected img size (128,128), got {img.size}"
    assert len(bbox) == 4, f"Expected 4 bounding box values, got {len(bbox)}"
    assert face_crop.size == (64, 64), f"Expected face crop size (64,64), got {face_crop.size}"
    print("OK: SyntheticFaceGenerator OK")

def test_detection_dataset():
    print("Testing FaceDetectionDataset...")
    dataset = FaceDetectionDataset(num_samples=10, has_face_ratio=0.5)
    assert len(dataset) == 10
    
    img, is_face, bbox = dataset[0]
    assert img.shape == (3, 128, 128), f"Expected image shape [3, 128, 128], got {img.shape}"
    assert isinstance(is_face, float), f"Expected is_face float, got {type(is_face)}"
    assert bbox.shape == (4,), f"Expected bbox shape [4], got {bbox.shape}"
    print("OK: FaceDetectionDataset OK")

def test_recognition_dataset():
    print("Testing FaceRecognitionTripletDataset...")
    dataset = FaceRecognitionTripletDataset(num_identities=5, samples_per_identity=4)
    anchor, positive, negative = dataset[0]
    
    assert anchor.shape == (3, 64, 64), f"Expected anchor shape [3, 64, 64], got {anchor.shape}"
    assert positive.shape == (3, 64, 64), f"Expected positive shape [3, 64, 64], got {positive.shape}"
    assert negative.shape == (3, 64, 64), f"Expected negative shape [3, 64, 64], got {negative.shape}"
    print("OK: FaceRecognitionTripletDataset OK")

def test_detector_cnn():
    print("Testing FaceDetectorCNN forward/backward pass...")
    model = FaceDetectorCNN()
    x = torch.randn(2, 3, 128, 128)
    cls_logits, bbox = model(x)
    
    assert cls_logits.shape == (2, 1), f"Expected cls_logits shape [2, 1], got {cls_logits.shape}"
    assert bbox.shape == (2, 4), f"Expected bbox shape [2, 4], got {bbox.shape}"
    
    # Test backward pass
    loss = cls_logits.sum() + bbox.sum()
    loss.backward()
    print("OK: FaceDetectorCNN forward/backward OK")

def test_embedding_cnn():
    print("Testing FaceEmbeddingCNN forward/backward pass...")
    model = FaceEmbeddingCNN()
    x = torch.randn(2, 3, 64, 64)
    embeds = model(x)
    
    assert embeds.shape == (2, 128), f"Expected embeds shape [2, 128], got {embeds.shape}"
    # Check L2 normalization
    norms = torch.norm(embeds, p=2, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms)), "Embeddings are not L2-normalized"
    
    # Test backward pass
    loss = embeds.sum()
    loss.backward()
    print("OK: FaceEmbeddingCNN forward/backward OK")

def run_tests():
    try:
        test_synthetic_generator()
        test_detection_dataset()
        test_recognition_dataset()
        test_detector_cnn()
        test_embedding_cnn()
        print("\nAll pipeline components verified successfully!")
    except Exception as e:
        print(f"\nTest failed: {e}")
        raise e

if __name__ == "__main__":
    run_tests()
