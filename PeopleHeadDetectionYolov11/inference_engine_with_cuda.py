import cv2
import torch
import numpy as np
import time
from absl import app, flags
from absl.flags import FLAGS
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO
1
# Define command line flags

# Change with ur own path
flags.DEFINE_string('weights', 'D:\A S2\Semester 2\Pemodelan dan Rekayasa Sistem\Pekan 17\PeopleHeadDetectionYolov11\_best.pt', 'Path to YOLOv11 Weights')
# flags.DEFINE_string('video', '0', 'Path to input video or webcam index (0)')



flags.DEFINE_float('conf', 0.50, 'confidence threshold')

def show_fps(frame, fps):
    x, y, w, h = 10, 10, 330, 45
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 0), -1)
    cv2.putText(frame, "FPS: " + str(fps), (20, 52), cv2.FONT_HERSHEY_PLAIN, 3.5, (0, 255, 0), 3)

def show_counter(frame, title, class_names, obj_count, x_init):
    overlay = frame.copy()
    y_init = 100
    gap = 30
    alpha = 0.5
    cv2.rectangle(overlay, (x_init - 10, y_init - 35), (x_init + 200, 265), (0, 255, 0), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.putText(frame, title, (x_init, y_init - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    for obj_id, count in obj_count.items():
        y_init += gap
        obj_name = class_names[obj_id]
        obj_count_str = "%.3i" % (count)
        cv2.putText(frame, obj_name, (x_init, y_init), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(frame, obj_count_str, (x_init + 135, y_init), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

def show_total(frame, title, entry, exit):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.8
    color = (0, 0, 0)
    thickness = 2
    position = (10, 150)
    cv2.putText(frame, f"{title}: {entry-exit}", position, font, fontScale, color, thickness)

def main(_argv):
    video_input = FLAGS.video
    if video_input.isdigit():
        video_input = int(video_input)
        cap = cv2.VideoCapture(video_input)
    else:
        cap = cv2.VideoCapture(video_input)

    if not cap.isOpened():
        print('Error: Unable to open video source.')
        return

    tracker = DeepSort(max_age=5)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    yolov8_weights = FLAGS.weights
    model = YOLO(yolov8_weights)

    # Solusi 2: Ganti langsung class_names tanpa flags
    class_names = ['person']  # Misal hanya ingin deteksi 'person'
    # Kalau punya file coco.names, bisa langsung tulis path absolut atau relatif di sini:
    # with open("coco.names", "r") as f:
    #     class_names = f.read().strip().split("\n")

    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(class_names), 3))

    entered_obj_ids = []
    exited_obj_ids = []

    obj_class_ids = [0]
    obj_entry_count = {0: 0}
    obj_exit_count = {0: 0}
    track_id_previous_positions = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Get frame dimensions
        height, width = frame.shape[:2]

        # Define entry and exit lines based on frame dimensions
        entry_line = int(0.5 * height)
        exit_line = int(0.6 * height)
        offset = int(0.02 * height)

        start_time = time.time()

        results = model(frame)[0]
        detect = []

        for det in results.boxes:
            bbox = det.xyxy[0].cpu().numpy().astype(int)
            confidence = det.conf[0].item()
            class_id = int(det.cls[0].item())

            if confidence < FLAGS.conf:
                continue

            x1, y1, x2, y2 = bbox
            detect.append([[x1, y1, x2 - x1, y2 - y1], confidence, class_id])

        tracks = tracker.update_tracks(detect, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            class_id = track.get_det_class()
            color = colors[class_id]
            B, G, R = map(int, color)
            text = f"{track_id} - {class_names[class_id]}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (B, G, R), 2)
            cv2.rectangle(frame, (x1 - 1, y1 - 20), (x1 + len(text) * 12, y1), (B, G, R), -1)
            cv2.putText(frame, text, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            center_y = int((y1 + y2) / 2)

            if track_id in track_id_previous_positions:
                prev_center_y = track_id_previous_positions[track_id]

                if prev_center_y < exit_line <= center_y:
                    if track_id not in entered_obj_ids and class_id in obj_class_ids:
                        obj_exit_count[class_id] += 1
                        entered_obj_ids.append(track_id)

                if prev_center_y > entry_line >= center_y:
                    if track_id not in exited_obj_ids and class_id in obj_class_ids:
                        obj_entry_count[class_id] += 1
                        exited_obj_ids.append(track_id)

            track_id_previous_positions[track_id] = center_y

        cv2.line(frame, (100, entry_line), (width-50, entry_line), (0, 255, 0), 2)
        cv2.line(frame, (0, exit_line), (width, exit_line), (0, 0, 255), 2)

        show_counter(frame, "Enter", class_names, obj_entry_count, 10)
        show_counter(frame, "Exit", class_names, obj_exit_count, width - 210)

        show_total(frame,"total",obj_entry_count[0],obj_exit_count[0])


        end_time = time.time()

        fps = 1 / (end_time - start_time)
        fps = float("{:.2f}".format(fps))
        show_fps(frame, fps)

        cv2.imshow('YOLOv11 Object tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
if __name__ == '__main__':
    app.run(main)
