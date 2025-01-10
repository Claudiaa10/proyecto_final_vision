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
        img_hsv = cv2.GaussianBlur(img_hsv, (5, 5), 0)
        imgs_secuencia1.append(img_hsv)

    for i in range(1, 5):
        img = cv2.imread(f'emojis/secuencia_ne_{i}.jpg')
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_hsv = cv2.GaussianBlur(img_hsv, (5, 5), 0)
        imgs_secuencia2.append(img_hsv)

    light_yellow = (10, 80, 80)
    dark_yellow = (40, 255, 255)

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


def translate_and_compare(frame_mask, reference_mask, max_shift=10):
    best_similarity = 0
    best_shift = (0, 0)

    for dx in range(-max_shift, max_shift + 1):
        for dy in range(-max_shift, max_shift + 1):
            shifted_mask = np.roll(frame_mask, shift=(dy, dx), axis=(0, 1))
            similarity = similarity_percentage(reference_mask, shifted_mask)
            if similarity > best_similarity:
                best_similarity = similarity
                best_shift = (dx, dy)

    return best_shift, best_similarity


def similarity_percentage(mask1, mask2):
    if mask1.shape != mask2.shape:
        h1, w1 = mask1.shape
        h2, w2 = mask2.shape
        
        crop_h = (h2 - h1) // 2
        crop_w = (w2 - w1) // 2
        
        crop_h = max(crop_h, 0)
        crop_w = max(crop_w, 0)
        
        mask2 = mask2[crop_h:crop_h + h1, crop_w:crop_w + w1]

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
            frame_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  

            img_yellow_mask = cv2.inRange(frame_hsv, light_yellow, dark_yellow)  
            save_path = os.path.join(save_folder, "foto_actual_segmentada.jpg")
            cv2.imwrite(save_path, img_yellow_mask)


            best_shift_ge, similarity_ge = translate_and_compare(img_yellow_mask, masks_ge[sequence_ge_counter])
            best_shift_ne, similarity_ne = translate_and_compare(img_yellow_mask, masks_ne[sequence_ne_counter])

            print(f"Mejor similitud para ge_{sequence_ge_counter}: {similarity_ge:.2f}% con desplazamiento {best_shift_ge}")
            print(f"Mejor similitud para ne_{sequence_ne_counter}: {similarity_ne:.2f}% con desplazamiento {best_shift_ne}")

            if sequence_ge_counter >= sequence_ne_counter and similarity_ge > 80:  
                print(f"Foto coincide con máscara ge_{sequence_ge_counter}, similitud: {similarity_ge:.2f}%")
                sequence_ge_counter += 1

            elif sequence_ge_counter < sequence_ne_counter and similarity_ne > 80:  
                print(f"Foto coincide con máscara ne_{sequence_ne_counter}, similitud: {similarity_ne:.2f}%")
                print(f"Foto coincide con máscara ne_{sequence_ne_counter}")
                sequence_ne_counter += 1
            
            elif similarity_ge > 80:  
                print(f"Foto coincide con máscara ge_{sequence_ge_counter}, similitud: {similarity_ge:.2f}%")
                sequence_ge_counter += 1
                sequence_ne_counter = 0 
            
            elif similarity_ne > 80:  
                print(f"Foto coincide con máscara ne_{sequence_ne_counter}, similitud: {similarity_ne:.2f}%")
                sequence_ne_counter += 1
                sequence_ge_counter = 0 
            else:
                print(f"Foto no coincide con ninguna máscara, reiniciando...")
                sequence_ge_counter = 0
                sequence_ne_counter = 0

            if sequence_ge_counter == 4:
                print("Secuencia gernica completada.")
                guernica = cv2.imread('cuadros/guernica.jpg')
                cv2.imshow("Guernica", guernica)
                cv2.waitKey(0)
                break
            elif sequence_ne_counter == 4:
                print("Secuencia noche estrellada completada.")
                noche_estrellada = cv2.imread('cuadros/noche_estrellada.jpg')
                cv2.imshow("Noche Estrellada", noche_estrellada)
                cv2.waitKey(0)
                break

            cv2.imshow("Segmented Mask", img_yellow_mask)

    cv2.destroyAllWindows()


def stream_optical_flow():
    picam = Picamera2()
    picam.preview_configuration.main.size = (1280, 720)
    picam.preview_configuration.main.format = "RGB888"
    picam.preview_configuration.align()
    picam.configure("preview")
    picam.start()

    first_frame = picam.capture_array()
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(first_frame)

    maxCorners = 100
    qualityLevel = 0.5
    minDistance = 100
    blockSize = 7

    winSize = (15, 15)
    maxLevel = 2
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)

    lower_bound = np.array([112, 132, 129])
    upper_bound = np.array([115, 251, 176])

    hsv = cv2.cvtColor(first_frame, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv, lower_bound, upper_bound)

    p0 = cv2.goodFeaturesToTrack(prev_gray, mask=blue_mask, maxCorners=maxCorners,
                                  qualityLevel=qualityLevel, minDistance=minDistance, blockSize=blockSize)

    while True:
        frame = picam.capture_array()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        blue_mask = cv2.inRange(hsv, lower_bound, upper_bound)

        if p0 is not None:
            p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, p0, None,
                                                   winSize=winSize, maxLevel=maxLevel, criteria=criteria)

            if p1 is not None:
                good_new = p1[st == 1]
                good_old = p0[st == 1]

                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel().astype(int)
                    c, d = old.ravel().astype(int)
                    frame = cv2.circle(frame, (a, b), 5, (255, 0, 0), -1)
                    mask = cv2.line(mask, (a, b), (c, d), (255, 0, 0), 2)

                p0 = good_new.reshape(-1, 1, 2)
        else:
            p0 = cv2.goodFeaturesToTrack(frame_gray, mask=blue_mask, maxCorners=maxCorners,
                                         qualityLevel=qualityLevel, minDistance=minDistance, blockSize=blockSize)

        prev_gray = frame_gray.copy()

        output = cv2.add(frame, mask)
        cv2.imshow("Optical Flow", output)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):  # Salir
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    while True:
        print("\nMenú Principal")
        print("1. Buscar Contraseñas")
        print("2. Dibujar con Optical Flow")
        print("3. Salir")

        choice = input("Selecciona una opción: ")

        if choice == "1":
            print("Iniciando búsqueda de contraseñas...")
            stream_and_compare()
        elif choice == "2":
            print("Iniciando dibujo con Optical Flow...")
            stream_optical_flow()
        elif choice == "3":
            print("Saliendo del programa.")
            break
        else:
            print("Opción no válida. Por favor, intenta de nuevo.")
