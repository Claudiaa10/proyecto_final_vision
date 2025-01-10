import cv2
import os
import numpy as np
from picamera2 import Picamera2


def shi_tomasi_corner_detection(image: np.array, maxCorners: int, qualityLevel: float, minDistance: int, corner_color: tuple, radius: int):
    '''
    Detecta las esquinas usando el método Shi-Tomasi.
    '''
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=maxCorners, qualityLevel=qualityLevel, minDistance=minDistance)
    corners = np.intp(corners)
    print(f'Cantidad de esquinas detectadas: {len(corners)}')

    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(image, (x, y), radius, corner_color, -1)
    return image, len(corners)


def create_masks():
    light_yellow = (10, 80, 80)
    dark_yellow = (40, 255, 255)

    light_red = (0, 100, 100)
    dark_red = (10, 255, 255)

    light_blue = (90, 80, 80)
    dark_blue = (130, 255, 255)

    return light_yellow, dark_yellow, light_red, dark_red, light_blue, dark_blue


def identify_color(frame, light_yellow, dark_yellow, light_red, dark_red, light_blue, dark_blue):
    """
    Detecta el color predominante de las figuras en la imagen.
    """
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    yellow_mask = cv2.inRange(frame_hsv, light_yellow, dark_yellow)
    red_mask = cv2.inRange(frame_hsv, light_red, dark_red)
    blue_mask = cv2.inRange(frame_hsv, light_blue, dark_blue)

    yellow_area = np.sum(yellow_mask) 
    red_area = np.sum(red_mask)
    blue_area = np.sum(blue_mask)

    if yellow_area > red_area and yellow_area > blue_area:
        return "yellow"
    elif red_area > yellow_area and red_area > blue_area:
        return "red"
    else:
        return "blue"


def stream_and_compare():
    light_yellow, dark_yellow, light_red, dark_red, light_blue, dark_blue = create_masks()

    picam = Picamera2()
    picam.preview_configuration.main.size = (1280, 720)
    picam.preview_configuration.main.format = "RGB888"
    picam.preview_configuration.align()
    picam.configure("preview")
    picam.start()

    # Contraseñas
    secuencia_guernica = [("rojo", "triangulo"), ("amarillo", "cuadrado"), ("rojo", "hexagono"), ("amarillo", "triangulo")]
    secuencia_noche_estrellada = [("azul", "hexagono"), ("amarillo", "cuadrado"), ("azul", "triangulo"), ("azul", "cuadrado")]

    contador_secuencia = 0  
    while True:
        frame = picam.capture_array()
        cv2.imshow("Picamera Stream", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):  # Salir
            break
        elif key == ord('f'):  # Capturar imagen
            save_folder = "/home/pi/practica_vision/fotos"
            os.makedirs(save_folder, exist_ok=True)
            save_path = os.path.join(save_folder, "foto_actual.jpg")
            cv2.imwrite(save_path, frame)

            corner_image, num_corners = shi_tomasi_corner_detection(frame, maxCorners=6, qualityLevel=0.3, minDistance=100, corner_color=(255, 0, 255), radius=8)
            color_detectado = identify_color(frame, light_yellow, dark_yellow, light_red, dark_red, light_blue, dark_blue)

            if num_corners == 3:
                figura_detectada = "triangulo"
            elif num_corners == 4:
                figura_detectada = "cuadrado"
            elif num_corners == 5:
                figura_detectada = "pentágono"
            elif num_corners == 6:
                figura_detectada = "hexagono"
            else:
                figura_detectada = "desconocida"

            if contador_secuencia < len(secuencia_guernica) and (color_detectado, figura_detectada) == secuencia_guernica[contador_secuencia]:
                print(f"Correcto para Guernica: {color_detectado}, {figura_detectada}")
                contador_secuencia += 1
            elif contador_secuencia < len(secuencia_noche_estrellada) and (color_detectado, figura_detectada) == secuencia_noche_estrellada[contador_secuencia]:
                print(f"Correcto para La Noche Estrellada: {color_detectado}, {figura_detectada}")
                contador_secuencia += 1
            else:
                print(f"Secuencia incorrecta, reiniciando...")

            cv2.imshow("Corners Detected", corner_image)

            if contador_secuencia == 4:
                if contador_secuencia == len(secuencia_guernica):
                    print("¡Has desbloqueado Guernica!")
                elif contador_secuencia == len(secuencia_noche_estrellada):
                    print("¡Has desbloqueado La Noche Estrellada!")
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    stream_and_compare()
