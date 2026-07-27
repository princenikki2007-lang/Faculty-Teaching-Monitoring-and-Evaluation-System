import cv2
import numpy as np
import os

class VisionProcessor:
    def __init__(self, output_dir="data/extracted_boards"):
        # Directory where extracted board images will be saved
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Sample 1 frame per second (assuming 30 FPS video)
        self.sample_rate = 30

    def _enhance_board_image(self, frame):
        """
        Applies adaptive thresholding and contrast enhancement 
        to isolate dark ink/chalk text on a bright background.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Remove shadows/background gradients using adaptive morphological closing
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
        background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        
        # Divide image by background to normalize lighting unevenness
        normalized = cv2.divide(gray, background, scale=255)
        
        # Apply Adaptive Thresholding to make writing crisp and high-contrast
        enhanced = cv2.adaptiveThreshold(
            normalized,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            21,
            15
        )
        return enhanced

    def analyze_video(self, video_path):
        """
        Processes video frames to extract clean whiteboard/blackboard snapshots.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {
                "error": f"Could not open video file at {video_path}",
                "presence_percentage": 0.0,
                "frames_analyzed": 0,
                "extracted_boards_count": 0,
                "presence_summary": "Video opening failed"
            }

        frame_count = 0
        analyzed_frames = 0
        saved_snapshots = []
        prev_snapshot = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % self.sample_rate != 0:
                continue

            analyzed_frames += 1
            
            # 1. Enhance the frame content
            enhanced_frame = self._enhance_board_image(frame)

            # 2. Compute visual difference compared to previous saved snapshot
            # (Prevents saving duplicate frames when the instructor isn't writing anything new)
            if prev_snapshot is None:
                is_different = True
            else:
                diff = cv2.absdiff(prev_snapshot, enhanced_frame)
                non_zero_ratio = np.count_nonzero(diff) / diff.size
                is_different = non_zero_ratio > 0.05  # Save if >5% of pixels changed

            # 3. Save unique board frames
            if is_different:
                snapshot_filename = f"board_frame_{frame_count}.png"
                save_path = os.path.join(self.output_dir, snapshot_filename)
                cv2.imwrite(save_path, enhanced_frame)
                
                saved_snapshots.append(save_path)
                prev_snapshot = enhanced_frame

        cap.release()

        count = len(saved_snapshots)
        summary_msg = f"Successfully extracted {count} clean board content snapshots."

        return {
            "presence_percentage": 100.0,  # Keeps pipeline compatibility with report_generator
            "frames_analyzed": analyzed_frames,
            "extracted_boards_count": count,
            "saved_paths": saved_snapshots,
            "presence_summary": summary_msg
        }