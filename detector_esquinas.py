import cv2
import os
import numpy as np
from picamera2 import Picamera2

def shi_tomasi_corner_detection(image: np.array, maxCorners: int, qualityLevel: float, minDistance: int, corner_color: tuple, radius: int):
    '''
    image - Input image
    maxCorners - Maximum number of corners to return. 
                 If there are more corners than are found, the strongest of them is returned. 
                 maxCorners <= 0 implies that no limit on the maximum is set and all detected corners are returned
    qualityLevel - Parameter characterizing the minimal accepted quality of image corners. 
                   The parameter value is multiplied by the best corner quality measure, which is the minimal eigenvalue or the Harris function response. 
                   The corners with the quality measure less than the product are rejected. 
                   For example, if the best corner has the quality measure = 1500, and the qualityLevel=0.01 , then all the corners with the quality measure less than 15 are rejected
    minDistance - Minimum possible Euclidean distance between the returned corners
    corner_color - Desired color to highlight corners in the original image
    radius - Desired radius (pixels) of the circle
    '''
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=maxCorners, qualityLevel=qualityLevel, minDistance=minDistance)
    corners = np.intp(corners)
    print(f'Cantidad de esquinas detectadas: {len(corners)}')

    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(image, (x, y), radius, corner_color, -1)
    return image

def stream_video():
    picam = Picamera2()
    picam.preview_configuration.main.size = (1280, 720)
    picam.preview_configuration.main.format = "RGB888"
    picam.preview_configuration.align()
    picam.configure("preview")
    picam.start()

    photo_index = 0
    purple_color = (255, 0, 255) 

    while True:
        frame = picam.capture_array()
        cv2.imshow("Picamera Stream", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):  
            break
        elif key == ord('f'): 
            save_folder = "/home/pi/practica_vision/fotos"
            os.makedirs(save_folder, exist_ok=True)  
            save_path = os.path.join(save_folder, f"Foto{photo_index}.jpg")
            cv2.imwrite(save_path, frame)
            print(f"Foto guardada: {save_path}")
            
            tomasi_image = shi_tomasi_corner_detection(frame, maxCorners=7, qualityLevel=0.3, minDistance=100, corner_color=purple_color, radius=8)
            
            save_path_with_corners = os.path.join(save_folder, f"FotoConEsquinas{photo_index}.jpg")
            cv2.imwrite(save_path_with_corners, tomasi_image)
            print(f"Foto con esquinas guardada: {save_path_with_corners}")
            
            photo_index += 1

    cv2.destroyAllWindows()

if __name__ == "__main__":
    stream_video()
