import cv2
import numpy as np
import argparse
from pathlib import Path
import shutil

def chroma_key(foreground, background, lower_green, upper_green):
    hsv = cv2.cvtColor(foreground, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_green, upper_green)
    mask_inv = cv2.bitwise_not(mask)
    fg = cv2.bitwise_and(foreground, foreground, mask=mask_inv)
    background = cv2.resize(background, (foreground.shape[1], foreground.shape[0]))
    bg = cv2.bitwise_and(background, background, mask=mask)
    result = cv2.add(fg, bg)
    return result


ap = argparse.ArgumentParser()
ap.add_argument("-f", "--foreground", required=True, help="path to foreground images", type=Path)
ap.add_argument("-b", "--background", required=True, help="path to background images", type=Path)
ap.add_argument("-l", "--label", required=True, help="path to labels", type=Path)
ap.add_argument("-o", "--output", required=True, help="path to output image", type=Path)
args = vars(ap.parse_args())

fg_path=args["foreground"]
bg_path=args["background"]
out_path=args["output"]
lbl_path=args["label"]

l_bg=list(bg_path.glob("*.jpg"))

for fg in fg_path.glob("*.jpg"):
    foreground = cv2.imread(fg)

    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])

    for bg in l_bg:
        print(f"{fg} - {bg}")
        background = cv2.imread(bg)
        result = chroma_key(foreground, background, lower_green, upper_green)
        cv2.imwrite(out_path/f"{fg.stem}_{bg.stem}.jpg", result)
        lbl_name=fg.with_suffix(".txt").name
        shutil.copy2(lbl_path/lbl_name, lbl_path/f"{fg.stem}_{bg.stem}.txt")


