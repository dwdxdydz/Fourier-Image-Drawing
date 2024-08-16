#!/usr/bin/python3
# coding: utf-8

import cv2
import numpy as np
import tkinter as tk

def get_screen_dimensions():
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()
    return screen_width, screen_height

def drawing():
    prev_x, prev_y = -1, -1
    final_image = None

    def draw_line(event, x, y, flags, params):
        nonlocal prev_x, prev_y, final_image

        if event == cv2.EVENT_LBUTTONDOWN:
            prev_x, prev_y = x, y

        elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
            # cv2.line(img, (prev_x, prev_y), (x, y), (255, 255, 255), 1)
            cv2.circle(img, (prev_x, prev_y), 1, (255, 255, 255), -1)
            prev_x, prev_y = x, y

        elif event == cv2.EVENT_LBUTTONUP:
            final_image = img.copy()
            return

    img = np.zeros((800, 800, 3), np.uint8)

    cv2.namedWindow('Canvas')

    screen_width, screen_height = get_screen_dimensions()

    window_width, window_height = img.shape[1], img.shape[0]
    window_x = (screen_width - window_width) // 2
    window_y = (screen_height - window_height) // 2

    cv2.moveWindow('Canvas', window_x, window_y)

    cv2.setMouseCallback('Canvas', draw_line)

    while True:
        cv2.imshow('Canvas', img)
        key = cv2.waitKey(1)
        if key == 27 or final_image is not None:
            break

    cv2.destroyAllWindows()
    cv2.imwrite('image.jpg', img)
    return final_image

# drawing()