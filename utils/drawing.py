import cv2

def draw_text(frame, text: str, position, font_scale: float=0.5, color:tuple=(255,255,255), thickness: float=2):
    """_summary_

    Args:
        frame (_type_): _description_
        text (_type_): _description_
        position (_type_): _description_
        font_scale (float, optional): _description_. Defaults to 0.5.
        color (tuple, optional): _description_. Defaults to (255,255,255).
        thickness (int, optional): _description_. Defaults to 2.
    """
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), thickness+3, cv2.LINE_AA)
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)