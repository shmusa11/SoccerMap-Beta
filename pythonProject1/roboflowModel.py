import os

import cv2
from roboflow import Roboflow
import threading
import queue
import time

######################################
#TRYING TO USE MODEL LOCALLY STILL SLOW
######################################


# --- InferenceWorker Class (No changes needed here, it's correct) ---
class InferenceWorker(threading.Thread):
    # ... (same as before) ...
    def __init__(self, model, input_queue, output_queue):
        super().__init__()
        self.model = model
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.running = True

    def run(self):
        while self.running:
            try:
                original_frame = self.input_queue.get(timeout=1)
                resized_frame = cv2.resize(original_frame, (640, 640))
                results = self.model.predict(resized_frame, confidence=20, overlap=30).json()

                original_h, original_w, _ = original_frame.shape
                for prediction in results['predictions']:
                    x1 = prediction['x'] - prediction['width'] / 2
                    y1 = prediction['y'] - prediction['height'] / 2
                    x2 = prediction['x'] + prediction['width'] / 2
                    y2 = prediction['y'] + prediction['height'] / 2
                    x1_orig = int(x1 * (original_w / 640))
                    y1_orig = int(y1 * (original_h / 640))
                    x2_orig = int(x2 * (original_w / 640))
                    y2_orig = int(y2 * (original_h / 640))
                    label = prediction['class']
                    cv2.rectangle(original_frame, (x1_orig, y1_orig), (x2_orig, y2_orig), (0, 255, 0), 2)
                    cv2.putText(original_frame, label, (x1_orig, y1_orig - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 255, 0), 2)

                self.output_queue.put(original_frame)
            except queue.Empty:
                continue

    def stop(self):
        self.running = False


# --- Main Program ---
if __name__ == "__main__":
    # --- Roboflow Setup ---
    rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
    project = rf.workspace().project("soccer-player-ball-tracker-bechj")
    model = project.version(4).model

    # --- Queues ---
    raw_frame_queue = queue.Queue(maxsize=5)
    processed_frame_queue = queue.Queue(maxsize=5)

    # --- Start the Worker ---
    worker = InferenceWorker(model, raw_frame_queue, processed_frame_queue)
    worker.start()

    # --- Video Capture ---
    video_path = "local_clip.mp4"
    cap = cv2.VideoCapture(video_path)

    # =====================================================================
    #  KEY CHANGE: Keep track of the last good frame we processed
    # =====================================================================
    latest_processed_frame = None

    # --- Main GUI Loop ---
    while True:
        # 1. Read a new frame from the video file
        ret, frame = cap.read()
        if not ret:
            # If the video ends, wait for the last few frames to be processed
            while not processed_frame_queue.empty():
                cv2.imshow('Detection', processed_frame_queue.get())
                if cv2.waitKey(30) & 0xFF == ord('q'):  # Show remaining frames at ~30fps
                    break
            break

        # 2. Put the raw frame into the queue for the worker (non-blocking)
        try:
            raw_frame_queue.put_nowait(frame)
        except queue.Full:
            # If the worker is falling behind, we just drop the frame
            # This prevents the main loop from blocking and keeps the video source moving.
            pass

        # 3. Check for a newly processed frame from the worker
        try:
            # Update our display frame ONLY when a new one is ready
            latest_processed_frame = processed_frame_queue.get_nowait()
        except queue.Empty:
            # No new processed frame is ready, just continue
            pass

        # 4. Display the latest available processed frame
        if latest_processed_frame is not None:
            cv2.imshow('Detection', latest_processed_frame)

        # 5. This keeps the GUI responsive
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- Cleanup ---
    print("Stopping worker thread...")
    worker.stop()
    worker.join()
    cap.release()
    cv2.destroyAllWindows()