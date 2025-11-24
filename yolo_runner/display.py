from __future__ import annotations

import cv2


def show_frame(window_name: str, frame) -> bool:
    cv2.imshow(window_name, frame)
    return cv2.waitKey(1) & 0xFF != ord("q")


def close_window(window_name: str) -> None:
    try:
        cv2.destroyWindow(window_name)
    except cv2.error:
        pass
