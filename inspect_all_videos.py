import cv2
import os
import glob
import json

video_files = glob.glob('data/raw/*.mp4')
info = []

os.makedirs('data/samples', exist_ok=True)

for vf in video_files:
    cap = cv2.VideoCapture(vf)
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dur = count / fps if fps > 0 else 0
    fname = os.path.basename(vf)
    
    # Save a frame at 30% duration
    mid_idx = int(count * 0.35)
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_idx)
    ret, frame = cap.read()
    sample_img = f"data/samples/{os.path.splitext(fname)[0][:25].strip().replace(' ', '_')}.jpg"
    if ret:
        cv2.imwrite(sample_img, frame)

    cap.release()
    item = {
        "filename": fname,
        "resolution": f"{w}x{h}",
        "width": w,
        "height": h,
        "fps": round(fps, 2),
        "frame_count": count,
        "duration_sec": round(dur, 2),
        "sample_image": sample_img
    }
    info.append(item)
    print(f"{fname}: {w}x{h} @ {fps:.1f} fps, {dur:.1f}s ({count} frames)")

with open('data/video_metadata.json', 'w') as f:
    json.dump(info, f, indent=2)
