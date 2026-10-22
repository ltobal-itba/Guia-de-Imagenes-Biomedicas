# Segmentación de la pupila: watershed con marcadores robustos, cierre morfológico y envolvente convexa (procedimiento elegido en el Módulo 2).

import cv2
import numpy as np
from skimage.segmentation import watershed
from skimage.filters import sobel
from skimage.morphology import convex_hull_image


def region_mas_circular(binaria, area_min=200):
    # De todas las regiones blancas, devuelve solo la más parecida a un círculo.
    contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mejor, mejor_circ = None, -1
    for c in contornos:
        area = cv2.contourArea(c)
        if area < area_min:
            continue
        _, radio = cv2.minEnclosingCircle(c)
        circ = area / (np.pi * radio ** 2)
        if circ > mejor_circ:
            mejor, mejor_circ = c, circ
    salida = np.zeros_like(binaria)
    if mejor is not None:
        cv2.drawContours(salida, [mejor], -1, 1, thickness=-1)
    return salida


def segmentar_pupila(roi_rgb, tam_ee=21):
    # Devuelve la máscara final de la pupila (0/255) a partir de una ROI en RGB.
    gris = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY)
    suave = cv2.GaussianBlur(gris, (5, 5), 1.5)
    gradiente = sobel(suave.astype(float))

    # Marcadores
    oscuro = (suave < 40).astype(np.uint8)
    ee_chico = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    oscuro = cv2.morphologyEx(oscuro, cv2.MORPH_OPEN, ee_chico)

    marcadores = np.zeros(gris.shape, dtype=np.int32)
    marcadores[suave > 70] = 2 # fondo
    marcadores[region_mas_circular(oscuro) > 0] = 1 # pupila
    diferencia = gris.astype(int) - cv2.medianBlur(gris, 15).astype(int)
    marcadores[diferencia > 60] = 3 # reflejos

    # Watershed
    etiquetas = watershed(gradiente, marcadores)
    mascara = (etiquetas == 1).astype(np.uint8) * 255

    # Limpieza: cierre + envolvente convexa
    ee = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tam_ee, tam_ee))
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, ee)
    if mascara.any():
        mascara = (convex_hull_image(mascara > 0) * 255).astype(np.uint8)
    return mascara