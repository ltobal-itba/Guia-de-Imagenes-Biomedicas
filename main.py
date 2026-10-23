# Pipeline de análisis del reflejo pupilar fotomotor.
#
# Uso:
#   1. Indicar el video a procesar en config.py (variable VIDEO_PATH).
#   2. Desde la carpeta raíz del proyecto, ejecutar:  python main.py
#
# Salidas (en la carpeta outputs/):
#   - Curva temporal del área/diámetro pupilar frame a frame (CSV)
#   - Parámetros fisiológicos: latencia, amplitud, velocidades (CSV)
#   - Imagen de control para verificar la detección del ojo y la pupila
# Los resultados de cada video se guardan en su propia subcarpeta (outputs/<nombre del video>/)

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import VIDEO_PATH, OUTPUT_DIR
from src.video import preparar_video, leer_frames
from src.localizacion import inicializar, recortar_roi
from src.segmentacion import segmentar_pupila_video
from src.visualizacion import dibujar_control, guardar_controles


def main():
    # Cada video guarda sus resultados en su propia subcarpeta (por ejemplo outputs/A1/),
    # para que procesar un video no sobrescriba los resultados de otro
    carpeta_resultados = OUTPUT_DIR / VIDEO_PATH.stem
    carpeta_resultados.mkdir(parents=True, exist_ok=True)

    # 1) Video de entrada
    ruta_video, info = preparar_video(VIDEO_PATH, OUTPUT_DIR)
    fps = info["fps"]
    print(f"Video: {ruta_video.name}")
    print(f"  {info['ancho']}×{info['alto']} px | {fps:.1f} FPS | "
          f"{info['n_frames']} frames | {info['duracion_s']:.1f} s")

    # 2) Localización automática del ojo
    x, y, r0 = inicializar(ruta_video)

    # Tamaño de la ROI proporcional a la pupila: así no depende de la distancia
    # a la cámara ni de la resolución del video. Con 10 × 8 radios entra el
    # iris completo (en nuestro video da una ROI parecida a la elegida a mano)
    tam_roi = (int(10 * r0), int(8 * r0))
    print(f"Pupila encontrada en ({x:.0f}, {y:.0f}) px, radio ≈ {r0:.0f} px")
    print(f"  ROI de {tam_roi[0]}×{tam_roi[1]} px")

    # 3) Segmentación frame a frame, siguiendo a la pupila

    # Rango de áreas admisible para la pupila, relativo al radio inicial.
    # Es amplio (de 0,4 a 2,5 veces el radio) porque la pupila cambia mucho de tamaño:
    # en el video A3 casi duplica su diámetro al apagarse la luz
    area_min = np.pi * (0.4 * r0) ** 2
    area_max = np.pi * (2.5 * r0) ** 2
    centro = (x, y)
    filas = []

    # Frames que se guardan como imagen de control, repartidos a lo largo del video
    frames_control = set(np.linspace(0, info["n_frames"] - 1, 6).astype(int))
    controles = []

    for idx, frame in leer_frames(ruta_video):
        roi, (x0, y0) = recortar_roi(frame, centro, tam_roi)
        mascara = segmentar_pupila_video(roi)
        area = int(np.count_nonzero(mascara))

        # Centroide y circularidad de la máscara
        if area > 0:
            ys, xs = np.nonzero(mascara)
            cx, cy = xs.mean() + x0, ys.mean() + y0     # en coordenadas del frame completo
            contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            _, radio_env = cv2.minEnclosingCircle(max(contornos, key=cv2.contourArea))
            circularidad = area / (np.pi * radio_env ** 2)
        else:
            cx = cy = np.nan
            circularidad = 0.0

        # Un frame es válido si la máscara parece una pupila: tamaño razonable,
        # forma redonda y sin un salto grande respecto de la posición anterior.
        # Los frames no válidos (parpadeos, saturación al prender la luz, máscaras
        # derramadas hacia las pestañas) se corrigen en el paso siguiente
        valido = (area_min < area < area_max
                  and circularidad > 0.7
                  and np.hypot(cx - centro[0], cy - centro[1]) < r0)

        # Movemos la ROI hacia la pupila solo con frames válidos.
        # Esto está porque, si la ROI siguiera una máscara errónea, se terminaría
        # perdiendo el ojo. El 0.7 / 0.3 hace que se mueva de a poco y no "tiemble"
        if valido:
            centro = (0.7 * centro[0] + 0.3 * cx, 0.7 * centro[1] + 0.3 * cy)

        filas.append({
            "frame": idx,
            "tiempo_s": idx / fps,
            "area_px": area,
            "diametro_px": 2 * np.sqrt(area / np.pi),
            "centro_x": cx,
            "centro_y": cy,
            "circularidad": circularidad,
            "valido": valido,
        })

        # Guardamos algunos frames con la ROI y la pupila dibujadas, para control
        if idx in frames_control:
            controles.append((idx, idx / fps, dibujar_control(frame, mascara, (x0, y0), tam_roi)))

        if idx % 100 == 0:
            print(f"  frame {idx}/{info['n_frames']}")

    curva = pd.DataFrame(filas)

    # Resultado provisorio: curva sin limpiar (todavía sin parpadeos ni estímulo)
    curva.to_csv(carpeta_resultados / "curva_cruda.csv", index=False)

    n_invalidos = int((~curva["valido"]).sum())
    print(f"Frames no válidos: {n_invalidos} de {len(curva)}")

    invalidos = curva[~curva["valido"]]
    plt.figure(figsize=(12, 4))
    plt.plot(curva["tiempo_s"], curva["area_px"], lw=0.8, label="Área")
    plt.plot(invalidos["tiempo_s"], invalidos["area_px"], "r.", ms=3, label="Frames no válidos")
    plt.legend()
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Área pupilar (px)")
    plt.title("Curva cruda del área pupilar")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(carpeta_resultados / "curva_cruda.png", dpi=120)

    guardar_controles(controles, carpeta_resultados / "control_segmentacion.png")
    print(f"Resultados guardados en {carpeta_resultados}")

    # 4) Detección de parpadeos y limpieza        (pendiente)
    # 5) Detección del estímulo luminoso          (pendiente)
    # 6) Parámetros fisiológicos                  (pendiente)
    # 7) Guardado de resultados finales           (pendiente)


if __name__ == "__main__":
    main()