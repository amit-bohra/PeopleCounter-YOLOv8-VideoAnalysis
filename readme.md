# People Counter Video Processor

This Python script processes a video file, detects people using a pre-trained YOLOv8 model, and generates a JSON file that logs the number of people entering, exiting, and the total number of people in the frame at each second.

## Prerequisites

- Python 3.7+
- OpenCV
- Ultralytics YOLO library
- tqdm (for progress bar)

## Installation

1. **Create a virtual environment:**
    - Windows: 
        ```shell
        python -m venv people_counter_env
        ```
    - Mac/Linux:
        ```shell
        python3 -m venv people_counter_env
        ```
  
2. **Activate the virtual environment:**
    - Windows:
        ```shell
        .\people_counter_env\Scripts\activate
        ```
    - Mac/Linux:
        ```shell
        source people_counter_env/bin/activate
        ```

3. **Install the required packages:**
    ```shell
    pip install -r requirements.txt
    ```

4. **Download the YOLOv8 model weights**: 
    - You can download the weights (`yolov8n.pt`) from the [Ultralytics](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt).


## How to Run

1. Place your video file in the root directory and rename it `user_activities.mp4`.
2. Run the script:
    ```shell
    python people_counter.py
    ```
3. After processing, the JSON output will be saved to `activities.json`.

## Approach Overview

The approach involves initializing the video file and YOLO model to detect people, then processing each frame to count the number of detected individuals. The function calculates the timestamp for each frame, logs entry and exit events based on changes in count, and records the total number of people present per second, ensuring that there are no duplicate counts by only updating when the timestamp changes. Finally, the results are structured into a JSON format, providing a comprehensive overview of people activity in the video.


## Future Improvements

1. **Use a Better Model**: Consider upgrading to a more advanced model like YOLOv8x for improved accuracy in detecting people in various conditions.

2. **Adjust Prediction Configuration**: Fine-tune parameters like the confidence threshold (conf) and IoU threshold (iou) to optimize detection accuracy. Lowering the confidence threshold can increase detections but may lead to more false positives, while adjusting the IoU can help reduce duplicates in dense scenes, enhancing overall detection quality.

3. **Frame Resizing**: Implement frame resizing techniques to reduce the computational load, allowing for faster processing of videos without significantly sacrificing detection quality.

4. **Object Tracking**: Integrate object tracking algorithms (e.g., SORT, Deep SORT) to maintain the identity of detected individuals across frames, enhancing the accuracy of entry and exit logging.

5. **Asynchronous Programming**: Utilize asynchronous programming techniques to handle video processing tasks concurrently, which can significantly improve performance and responsiveness.

6. **Dockerization**: Convert the application into a Docker image to facilitate deployment on various environments, and configure it to run on GPUs for faster processing. This will enable parallel processing of multiple video files.

7. **Web Interface**: Develop a web-based interface that allows users to upload video files and receive the JSON output directly, making the tool more accessible to non-technical users.

8. **Real-Time Processing**: Explore the possibility of real-time people counting and tracking from live video streams, which could be valuable for applications like security and crowd monitoring.

9. **Distributed Processing**: Employ distributed processing frameworks (e.g., Dask, Ray) to handle large volumes of video data across multiple CPU cores or machines.



