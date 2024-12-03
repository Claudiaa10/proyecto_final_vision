import cv2
import os
from picamera2 import Picamera2

def stream_video():
    picam = Picamera2()
    picam.preview_configuration.main.size=(1280, 720)
    picam.preview_configuration.main.format="RGB888"
    picam.preview_configuration.align()
    picam.configure("preview")
    picam.start()
    i = 0import cv2
import os
from picamera2 import Picamera2

def stream_video():
    picam = Picamera2()
    picam.preview_configuration.main.size = (1280, 720)
    picam.preview_configuration.main.format = "RGB888"
    picam.preview_configuration.align()
    picam.configure("preview")
    picam.start()

    photo_index = 0

    while True:
        frame = picam.capture_array()
        cv2.imshow("Picamera Stream", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):  # Salir 
            break
        elif key == ord('f'):  # Capturar foto 
            save_folder = "/home/pi/practica_vision/fotos"
            os.makedirs(save_folder, exist_ok=True)  
            save_path = os.path.join(save_folder, f"Foto{photo_index}.jpg")
            cv2.imwrite(save_path, frame)
            print(f"Foto guardada: {save_path}")
            photo_index += 1

    cv2.destroyAllWindows()

if __name__ == "__main__":
    stream_video()

    j = 0

    while True:
        frame = picam.capture_array()
        cv2.imshow("picam", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if j == 6:
            save_folder = "/home/pi/practica_vision/fotos"
            os.makedirs(save_folder, exist_ok=True)
            save_path = os.path.join(save_folder, f"Foto{i}.jpg")
            cv2.imwrite(save_path, frame)
            i+=1
            j = 0
        j += 1
    cv2.destroyAllWindows()

if __name__ == "__main__":
    stream_video()