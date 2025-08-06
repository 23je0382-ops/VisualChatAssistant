# 🎥 Video Summarization with Motion-Aware Object Captioning

This project performs **intelligent video summarization** by combining object detection, caption generation, motion analysis, and recursive text summarization. The goal is to convert a video into a meaningful **textual summary** that captures **both static content (objects)** and **dynamic behavior (movement/actions)**.

---

## 🧠 Key Features

- 🧱 **Object Detection**: Uses YOLOv8 to detect objects in each frame.
- 🖼️ **Caption Generation**: Captions each detected object using BLIP (`Salesforce/blip-image-captioning-base`).
- 🏃 **Motion Tagging**: Labels each object as `moving` or `steady` across frames.
- 🔗 **Recursive Summarization**: Combines and compresses caption information recursively across frames to form a narrative.
- 📄 **Final Summary Output**: Saves a human-readable summary in `final_summary.txt`.

---

## 🗂️ Project Structure

```
.
├── frames/                     # Folder with extracted video frames (input)
├── summarize_captions.py      # Contains recursive summarization function
├── models.py                  # Loads YOLO and BLIP models
├── main.py                    # Main pipeline script
├── final_summary.txt          # Output file (textual summary)
└── README.md                  # This file
```

---

## 🔧 Requirements

- Python 3.8+
- PyTorch
- Transformers (`blip` model)
- `ultralytics` for YOLOv8
- OpenCV
- PIL
- tqdm

### Install Dependencies

```bash
pip install -r requirments.txt
```

---

## 🚀 How to Run

1. **Prepare Input Frames**  
   Extract frames from your video and save them as `.jpg` files in the `frames/` directory. You can use OpenCV:

   ```python
   import cv2
   import os

   cap = cv2.VideoCapture('input_video.mp4')
   os.makedirs('frames', exist_ok=True)
   count = 0
   while True:
       ret, frame = cap.read()
       if not ret:
           break
       cv2.imwrite(f'frames/frame_{count:04d}.jpg', frame)
       count += 1
   cap.release()
   ```

2. **Run the frontend Pipeline**

   ```bash
   python frontend.py
   ```

3. **View the Output**  
   Open `final_summary.txt` to see the final summarized narrative of the video.

---

## 🔄 Summary Logic

The recursive summarization works as follows:

```text
s1 = summarize(captions_1)
s2 = summarize(captions_2)
progressive_summary = summarize(s1 + s2)

s3 = summarize(captions_3)
progressive_summary = summarize(progressive_summary + s3)

... and so on.
```

Each new frame’s captions are merged with the previous summary and summarized again, producing a refined, concise narrative.

---

## 📌 Notes

- You can tune YOLO's object detection thresholds to ignore small/noisy objects.
- Motion tagging threshold is set to 10 pixels (you can change it).
- You may use other summarization models like `T5`, `BART`, etc. in `summarize_captions.py`.

---

## 📄 Example Output

```
Frame 0 Summary:
A car is parked (steady), A man is walking (moving)

Frame 1 Summary:
A man walks across the street (moving)

Final Summary:
A man walks past a parked car and crosses the street.
```

---

## 📚 References

- [BLIP Model - Salesforce](https://huggingface.co/Salesforce/blip-image-captioning-base)
- [YOLOv8 - Ultralytics](https://github.com/ultralytics/ultralytics)
- [Video Summarization Literature](https://arxiv.org/abs/1906.12174)

---

## 📬 Contact

For questions or collaborations, please open an issue or reach out!
