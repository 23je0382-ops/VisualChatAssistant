import cv2
import os

# Output directory
output_dir = 'frames'
os.makedirs(output_dir, exist_ok=True)

# Load video
cap = cv2.VideoCapture(r'C:\Users\livea\OneDrive\Desktop\vuencode\sample_video.mp4')

# Get the original FPS of the video
fps = cap.get(cv2.CAP_PROP_FPS)

frame_count = 0
saved_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Save 1 frame per second
    if frame_count % int(fps) == 0:
        filename = f'{output_dir}/frame_{saved_count:04d}.jpg'
        cv2.imwrite(filename, frame)
        saved_count += 1

    frame_count += 1

cap.release()
cv2.destroyAllWindows()

