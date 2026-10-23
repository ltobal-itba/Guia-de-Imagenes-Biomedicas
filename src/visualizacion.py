# Gráficos de control del pipeline.
# Sirven para verificar a simple vista que el ojo y la pupila se están
# detectando bien, sobre todo al procesar un video nuevo.

import cv2
import numpy as np
import matplotlib.pyplot as plt


# Dibuja sobre un frame la ROI (rectángulo amarillo) y el contorno de la
# pupila segmentada (rojo). Devuelve un recorte alrededor de la ROI.
def dibujar_control(frame_rgb, mascara, esquina, tam_roi):
    x0, y0 = esquina
    ancho, alto = tam_roi
    vis = frame_rgb.copy()

    # Contorno de la pupila, pasado a coordenadas del frame completo
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for c in contornos:
        cv2.drawContours(vis, [c + np.array([x0, y0])], -1, (255, 0, 0), 2)

    # ROI
    cv2.rectangle(vis, (x0, y0), (x0 + ancho, y0 + alto), (255, 255, 0), 3)

    # Recortamos con margen alrededor de la ROI para que se vea el contexto
    margen = ancho // 2
    alto_f, ancho_f = vis.shape[:2]
    return vis[max(0, y0 - margen):min(alto_f, y0 + alto + margen),
               max(0, x0 - margen):min(ancho_f, x0 + ancho + margen)]


# Guarda una figura con varios frames de control, cada uno con su número y tiempo
def guardar_controles(controles, ruta_salida):
    n = len(controles)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    for ax, (idx, t, imagen) in zip(np.atleast_1d(axes), controles):
        ax.imshow(imagen)
        ax.set_title(f"Frame {idx} ({t:.1f} s)")
        ax.axis("off")
    plt.suptitle("Control: ROI (amarillo) y pupila segmentada (rojo)")
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=100)
    plt.close()