import torch
import cv2 as cv
import time
from ultralytics import YOLO

yolo_model = None
fallback_model = None
last_detection_time = 0
detection_interval = 0.5
current_prediction = None

HOUSEHOLD_CLASSES = [
    'air_freshner_dispenser', 'air_fryer', 'bluetooth_speaker', 'calculator',
    'computer_mouse', 'electric_kettle', 'electric_toothbrush', 'fitness_tracker',
    'hair_dryer', 'headphones', 'laptop', 'laptop_charger', 'mobile_phone',
    'power_bank', 'smartwatch', 'toaster', 'tv_remote_control', 'wireless_earbuds'
]

def initialize_model():
    global yolo_model, fallback_model
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        yolo_model = YOLO(r"C:\Users\garvk\OneDrive - Bhagwan Parshuram Institute of Technology\Desktop\advance_projects\candice\runs\detect\yolo11s_finetuned_final_model\weights\last.pt")
        fallback_model = YOLO(r"C:\Users\garvk\OneDrive - Bhagwan Parshuram Institute of Technology\Desktop\advance_projects\candice\runs\detect\yolo11s_finetuned_phase2\weights\last.pt")
        yolo_model.to(device)
        fallback_model.to(device)
        return True
    except Exception as e:
        print(f"Error loading YOLO models: {e}")
        return False

def classify_frame(frame):
    global last_detection_time, current_prediction
    try:
        current_time = time.time()
        if current_time - last_detection_time < detection_interval:
            return current_prediction

        device = "cuda" if torch.cuda.is_available() else "cpu"
        results = yolo_model(frame, imgsz=512, conf=0.5, device=device, verbose=False)
        
        if not results or len(results[0].boxes) == 0:
            results = fallback_model(frame, imgsz=512, conf=0.5, device=device, verbose=False)
            if not results or len(results[0].boxes) == 0:
                current_prediction = None
                return current_prediction

        boxes = results[0].boxes
        best_detection = None
        best_confidence = 0
        
        for i, box in enumerate(boxes):
            conf = float(box.conf[0])
            if conf > best_confidence:
                best_confidence = conf
                cls_id = int(box.cls[0])
                predicted_label = HOUSEHOLD_CLASSES[cls_id] if 0 <= cls_id < len(HOUSEHOLD_CLASSES) else f"Unknown_{cls_id}"
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                best_detection = {
                    "object_name": predicted_label,
                    "confidence": conf,
                    "detection_type": "yolo11s",
                    "bbox": (x1, y1, x2, y2)
                }
        
        if best_detection:
            current_prediction = best_detection
            last_detection_time = current_time
        else:
            current_prediction = None
            
        return current_prediction

    except Exception:
        return current_prediction

def draw_prediction(frame, prediction):
    if not prediction:
        cv.putText(frame, "Waiting for object detection...", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        return frame

    object_name = prediction['object_name']
    confidence = prediction['confidence']
    detection_type = prediction['detection_type']
    bbox = prediction.get('bbox')

    if bbox:
        x1, y1, x2, y2 = bbox
        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        label = f"{object_name} {confidence*100:.1f}%"
        label_size = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        
        cv.rectangle(frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), (0, 255, 0), -1)
        cv.putText(frame, label, (x1, y1 - 5), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    type_text = f"[{detection_type}]"
    cv.putText(frame, type_text, (10, frame.shape[0] - 40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    return frame

def main():
    if not initialize_model():
        print("Model initialization failed")
        return

    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not found")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv.flip(frame, 1)
            prediction = classify_frame(frame)
            frame = draw_prediction(frame, prediction)

            cv.putText(frame, "Press Q to quit", (10, frame.shape[0] - 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv.imshow('Candice Object Detection', frame)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv.destroyAllWindows()

if __name__ == "__main__":
    main()