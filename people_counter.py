import cv2
import json
from ultralytics import YOLO
from typing import List, Dict
import os
from tqdm import tqdm

# Define types for clarity
class EventType:
    ENTER = "ENTER"
    EXIT = "EXIT"

class DetectionEvent:
    def __init__(self, event_type: str, timestamp: float):
        self.event_type = event_type
        self.timestamp = str(timestamp)  # Convert timestamp to string

class PeopleCount:
    def __init__(self, people_count: int, timestamp: int):
        self.people_count = str(people_count)
        self.timestamp = str(timestamp)  # Convert timestamp to string

def load_model() -> YOLO:
    """
    Load the pre-trained YOLOv8x model for people detection.
    Returns the model object.
    """
    try:
        model = YOLO('yolov8n.pt')  # Make sure you have the correct YOLOv8n model file.
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def process_video(video_path: str, model: YOLO) -> Dict:
    """
    Process the video to count people entering/exiting the frame and log the count at each second.
    
    Args:
        video_path (str): Path to the input video file.
        model (YOLO): Pre-trained YOLO model for object detection.
    
    Returns:
        Dict: Data structure containing people count change events and total people count at each second.
    """
    # Open the video file
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file {video_path} not found.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Error opening video file.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    count_change_events: List[Dict] = []
    total_count: List[Dict] = []

    prev_people_count = 0
    last_timestamp = -1

    # Progress bar to indicate processing
    for frame_num in tqdm(range(total_frames), desc="Processing video"):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Predict people in the frame
        results = model(frame)

        # Access results and extract bounding boxes for people (class 0 for 'person')
        people_detections = [
            box for box in results[0].boxes if box.cls == 0
        ]
        people_count = len(people_detections)

        # Calculate current timestamp in seconds with two decimal places
        timestamp = round(frame_num / fps, 2)

        # Detect if the number of people has changed (enter or exit)
        if people_count != prev_people_count:
            if people_count > prev_people_count:
                count_change_events.append(DetectionEvent(EventType.ENTER, timestamp).__dict__)
            elif people_count < prev_people_count:
                count_change_events.append(DetectionEvent(EventType.EXIT, timestamp).__dict__)

        # Update total_count at the timestamp if it's a new second (rounded to integer)
        int_timestamp = int(timestamp)

        # Ensure we only log one entry per second
        if int_timestamp != last_timestamp:
            total_count.append(PeopleCount(people_count, int_timestamp).__dict__)
            last_timestamp = int_timestamp

        # Update previous people count for the next frame
        prev_people_count = people_count

    cap.release()

    # Final JSON structure
    return {
        "count_change_events": count_change_events,
        "total_count": total_count
    }


def save_results_to_json(data: Dict, output_file: str):
    """
    Save the results data to a JSON file.

    Args:
        data (Dict): The data to save in JSON format.
        output_file (str): The name of the output JSON file.
    """
    try:
        with open(output_file, 'w') as json_file:
            json.dump(data, json_file, indent=4)
    except Exception as e:
        print(f"Error saving JSON file: {e}")
        raise

def main():
    video_path = "user_activities.mp4"
    output_json = "activities.json"

    try:
        model = load_model()
        data = process_video(video_path, model)
        save_results_to_json(data, output_json)
        print(f"Processing complete. Results saved to {output_json}")
    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    main()