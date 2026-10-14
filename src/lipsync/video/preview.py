import cv2
import time

def play_video(frames, fps=25.0):
    frame_interval = 1.0 / fps
    start_time = time.perf_counter()

    for index, frame in enumerate(frames):
        target_time = start_time + index * frame_interval

        while time.perf_counter() < target_time:
            time.sleep(0.001)

        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("GRID Video Preview", bgr)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()