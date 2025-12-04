import sys
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Disclaimer: This was generated using Claude Sonnet 4.5, for easier demo purposes

MODEL_PATH = MODEL_PATH = Path(__file__).parent / "best_coin_detector_from_training.pt"
COIN_VALUES = {
    0: 1, 
    1: 5, 
    2: 5, 
    3: 10, 
    4: 20
}

COIN_NAMES = {
    0: '1 peso',
    1: '5 peso circle',
    2: '5 peso nonagon',
    3: '10 peso',
    4: '20 peso'
}

def detect_coins_in_image(image_path, save_result=True):
    if not MODEL_PATH.exists():
        print(f"Error: Model not found at {MODEL_PATH}")
        return None, None
    
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"Error: Image not found at {image_path}")
        return None, None
    
    model = YOLO(MODEL_PATH)
    results = model.predict(source=image_path, conf=0.5, save=save_result, verbose=False)

    detections = []
    total_value = 0
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            
            coin_value = COIN_VALUES[class_id]
            coin_name = COIN_NAMES[class_id]
            total_value += coin_value
            
            detections.append({
                'name': coin_name,
                'value': coin_value,
                'confidence': confidence
            })
    
    num_coins = len(detections)

    total_value = sum(COIN_VALUES[int(box.cls[0])] for r in results for box in r.boxes)
    num_coins = sum(len(r.boxes) for r in results)

    print(f"Total value detected: ₱{total_value} | Number of coins: {num_coins}")
    if detections:
        print("\n DETAILED BREAKDOWN:")
        print(f"{'#':<4} {'Coin Type':<20} {'Value':<8} {'Confidence':<12}")
        print("-" * 60)
        
        for i, det in enumerate(detections, 1):
            print(f"{i:<4} {det['name']:<20} ₱{det['value']:<7} {det['confidence']:.2%}")

def main():
    if len(sys.argv) < 2:
        print("Error: No image path provided!")
        print("\nUsage: python coin_detector_tester.py path/to/image.jpg")
        return
    
    image_path = sys.argv[1]
    detect_coins_in_image(image_path, save_result=True)

if __name__ == "__main__":
    main()