import os
import cv2
import pandas as pd
import numpy as np

def on_mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        param['clicked'] = True
        param['x'] = x

def display_videos(video_path_1, video_path_2):
    cap1 = cv2.VideoCapture(video_path_1)
    cap2 = cv2.VideoCapture(video_path_2)

    clicked = False
    mouse_params = {'clicked': False, 'x': -1}

    cv2.namedWindow('Side-by-side Videos', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Side-by-side Videos', on_mouse_click, mouse_params)

    while not clicked:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        # Loop the videos
        if not ret1:
            cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret1, frame1 = cap1.read()
        if not ret2:
            cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret2, frame2 = cap2.read()

        combined_frame = cv2.hconcat([frame1, frame2])
        cv2.imshow('Side-by-side Videos', combined_frame)
        cv2.setWindowTitle('Side-by-side Videos', 'Click on a video to label it')
        cv2.setWindowProperty('Side-by-side Videos', cv2.WND_PROP_TOPMOST, 1)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        clicked = mouse_params['clicked']

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

    return mouse_params['x']

def label_videos(input_folder, csv_path):
    df = pd.read_csv(csv_path)

    for idx, row in df.iterrows():
        if not np.isnan(row['label_1']):
            continue

        video_name_1, video_url_1, video_name_2, video_url_2 = row[['video_name_1', 'video_url_1', 'video_name_2', 'video_url_2']]
        video_path_1 = os.path.join(input_folder, video_name_1)
        video_path_2 = os.path.join(input_folder, video_name_2)

        x_click_position = display_videos(video_path_1, video_path_2)
        if x_click_position == -1:
            break

        cap = cv2.VideoCapture(video_path_1)
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap.release()

        label = 1 if x_click_position <= video_width else 2
        df.loc[idx, 'label_1'] = 1 if label == 1 else 0
        df.loc[idx, 'label_2'] = 1 if label == 2 else 0

        df.to_csv(csv_path, index=False)

# Example usage
input_folder = 'data/videos_random_policy'
csv_path = 's3_output_6.csv'

label_videos(input_folder, csv_path)
