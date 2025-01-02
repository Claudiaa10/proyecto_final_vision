import cv2
import os
import numpy as np
from picamera2 import Picamera2


def create_masks():
    imgs_secuencia1 = []
    imgs_secuencia2 = []
    masks_secuencia1 = []
    masks_secuencia2 = []

    for i in range(1, 5):
        img = cv2.imread(f'emojis/secuencia_ge_{i}.jpg')
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        imgs_secuencia1.append(img_hsv)

    for i in range(1, 5):
        img = cv2.imread(f'emojis/secuencia_ne_{i}.jpg')
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        imgs_secuencia2.append(img_hsv)

    light_yellow = (10, 80, 80)
    dark_yellow = (30, 255, 255)

    for img in imgs_secuencia1:
        yellow_mask = cv2.inRange(img, light_yellow, dark_yellow)
        masks_secuencia1.append(yellow_mask)

    for img in imgs_secuencia2:
        yellow_mask = cv2.inRange(img, light_yellow, dark_yellow)
        masks_secuencia2.append(yellow_mask) 
    
    if not os.path.exists('masks'):
        os.makedirs('masks')

    for i, img in enumerate(masks_secuencia1):
        save_path = os.path.join("masks/", f"mask_secuencia_ge_{i+1}.jpg")  
        cv2.imwrite(save_path, img)

    for i, img in enumerate(masks_secuencia2):
        save_path = os.path.join("masks/", f"mask_secuencia_ne_{i+1}.jpg")  
        cv2.imwrite(save_path, img)   




def similarity_percentage(mask1, mask2):
    if mask1.shape != mask2.shape:
        mask2 = cv2.resize(mask2, (mask1.shape[1], mask1.shape[0]))
    similar_pixels = np.sum(mask1 == mask2)
    total_pixels = mask1.size
    similarity = (similar_pixels / total_pixels) * 100
    return similarity


def stream_and_compare():
    create_masks()
    light_yellow = (10, 80, 80)
    dark_yellow = (40, 255, 255)
 
    picam = Picamera2()
    picam.preview_configuration.main.size = (1280, 720)
    picam.preview_configuration.main.format = "RGB888"
    picam.preview_configuration.align()
    picam.configure("preview")
    picam.start()

    masks_ge = [cv2.imread(f'masks/mask_secuencia_ge_{i+1}.jpg', cv2.IMREAD_GRAYSCALE) for i in range(4)]
    masks_ne = [cv2.imread(f'masks/mask_secuencia_ne_{i+1}.jpg', cv2.IMREAD_GRAYSCALE) for i in range(4)]

    sequence_ge_counter = 0
    sequence_ne_counter = 0

    while True:
        frame = picam.capture_array()
        cv2.imshow("Picamera Stream", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'): 
            break
        elif key == ord('f'): 
            save_folder = "/home/pi/practica_vision/fotos"
            os.makedirs(save_folder, exist_ok=True)
            save_path = os.path.join(save_folder, "foto_actual.jpg")
            cv2.imwrite(save_path, frame)

            img = cv2.imread(f'fotos/foto_actual.jpg')
            frame_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)  # Convierte a HSV

            img_yellow_mask = cv2.inRange(frame_hsv, light_yellow, dark_yellow)  # Filtra amarillo
            save_path = os.path.join(save_folder, "foto_actual_segmentada.jpg")
            cv2.imwrite(save_path, img_yellow_mask)

            similarity_ge = similarity_percentage(masks_ge[sequence_ge_counter], img_yellow_mask)
            similarity_ne = similarity_percentage(masks_ne[sequence_ne_counter], img_yellow_mask)
            if similarity_ge > 70:  
                print(f"Foto coincide con máscara ge_{sequence_ge_counter}, similitud: {similarity_ge:.2f}%")
                sequence_ge_counter += 1
                sequence_ne_counter = 0  
            elif similarity_ne > 70:  
                print(f"Foto coincide con máscara ne_{sequence_ne_counter}, similitud: {similarity_ne:.2f}%")
                sequence_ne_counter += 1
                sequence_ge_counter = 0 
            else:
                print("Foto no coincide con ninguna máscara, similitud: {similarity_ge:.2f}%, reiniciando...")
                sequence_ge_counter = 0
                sequence_ne_counter = 0

            if sequence_ge_counter == 4:
                print("Secuencia gernica completada.")
                break
            elif sequence_ne_counter == 4:
                print("Secuencia noche estrellada completada.")
                break

            cv2.imshow("Frame HSV", frame_hsv)
            cv2.imshow("Segmented Mask", img_yellow_mask)


    cv2.destroyAllWindows()

if __name__ == "__main__":
    stream_and_compare()
