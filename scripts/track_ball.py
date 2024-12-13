import cv2
import numpy as np
import os
from pathlib import Path
import argparse
import matplotlib.pyplot as plt

from matprop3d.models.grounding_dino import GroundingDINO
from matprop3d.utils.utils_plot import plot_bounding_boxes

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Track Ball")

    parser.add_argument(
        "--video-file", 
        type=str, 
        help="Path to the input video file."
    )
    
    args = parser.parse_args()
    
    device = 'cuda' 
    text_prompt = 'tennis ball.'
    ws = Path(os.getenv('MATPROP3DWS'))
    out_dir = ws/'ball_track'
    out_dir.mkdir(exist_ok=True, parents=True)
    
    cap = cv2.VideoCapture(args.video_file)
    gdino = GroundingDINO(device, text_prompt, box_th=0.3, text_th=0.3)

    if not cap.isOpened():
        print("Error: Cannot open video file.")
        exit()


    frame_count = 0

    all_traj = []
    while True:
        # Read a frame from the video
        ret, frame = cap.read()
        if not ret:
            break
    
        bb_dino = gdino.detect(frame)
        all_traj.append(bb_dino[0])
        img_det = plot_bounding_boxes(frame.copy(), bb_dino)

        cv2.imwrite(str(out_dir/f"{frame_count:03d}.png"), img_det)
        print(frame_count)
        frame_count += 1
    
    all_traj = np.array(all_traj)
    np.savetxt(str(ws/'real_ball_traj.txt'), all_traj) 
    
    # all_traj = np.loadtxt(str(ws/'real_ball_traj.txt'))
    heights = 1000-all_traj[:,-1]
    
    max_height, min_height = heights.max(), heights.min()
    first_bounce_height = np.max(heights[50:70])
    rebounce_factor = first_bounce_height/max_height 
    
    np.savetxt(ws/'rebounce_factor.txt', np.array([rebounce_factor]))
     
    grads = np.gradient(heights)
    
    plt.figure(figsize=(8,5))
    plt.title('Real Ball Trajectory')
    plt.xlabel('Frames')
    plt.ylabel('Height')
    plt.plot(heights, label='Height')
    plt.plot(np.gradient(heights), label='Height Gradient')
    plt.legend()
    
    plt.savefig(ws/'real_ball_traj.png')