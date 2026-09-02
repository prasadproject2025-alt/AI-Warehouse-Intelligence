import os
import glob
from backend.database.db import init_db, DatabaseManager
from video.processor import VideoProcessor

def process_all():
    init_db()
    processor = VideoProcessor(conf_threshold=0.25)
    
    videos = glob.glob("data/raw/*.mp4")
    print(f"Found {len(videos)} raw videos to process.")
    
    # Check which videos are already processed
    existing = {v["filename"] for v in DatabaseManager.get_all_videos()}
    
    for vpath in sorted(videos):
        fname = os.path.basename(vpath)
        if fname in existing:
            print(f"Skipping already processed: {fname}")
            continue
            
        print(f"\n==================================================")
        print(f"Processing: {fname}")
        print(f"==================================================")
        
        vid_id = f"vid_{abs(hash(fname)) % 1000000}"
        res = processor.process_video(
            video_path=vpath,
            video_id=vid_id,
            generate_annotated_video=True,
            frame_skip=2
        )
        print(f"Completed {fname}: {res['incidents_count']} incidents detected.")

if __name__ == "__main__":
    process_all()
