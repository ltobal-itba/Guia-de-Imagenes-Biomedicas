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


# Versión de la segmentación para el pipeline de video.
# Es el mismo procedimiento del Módulo 2 (watershed con marcadores, cierre y
# envolvente convexa), con cambios para que funcione con videos distintos al nuestro.
# Los notebooks siguen usando segmentar_pupila, así sus resultados no cambian.
def segmentar_pupila_video(roi_rgb):
    gris = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY)

    # Reflejos: píxeles mucho más brillantes que su entorno.
    # Los "borramos" rellenándolos con los valores de alrededor (inpainting).
    # Esto está porque en ojos con varios reflejos sobre la pupila, cada reflejo
    # generaba un borde que frenaba la inundación del watershed, y la máscara
    # dejaba afuera un pedazo de la pupila (nos pasó con el video A3)
    diferencia = gris.astype(int) - cv2.medianBlur(gris, 15).astype(int)
    reflejos = (diferencia > 40).astype(np.uint8)
    reflejos = cv2.dilate(reflejos, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    gris = cv2.inpaint(gris, reflejos, 5, cv2.INPAINT_TELEA)

    suave = cv2.GaussianBlur(gris, (5, 5), 1.5)
    gradiente = sobel(suave.astype(float))

    # Umbrales relativos a la propia imagen, en lugar de valores fijos (40 y 70).
    # Esto está porque los valores fijos se ajustaron sobre nuestro frame 778 y
    # fallaban en videos más oscuros, donde el iris queda por debajo de 70.
    # oscuro = intensidad de la pupila (lo más oscuro de la ROI)
    # tipico = intensidad típica de la ROI (mediana: piel, esclerótica, iris)
    # Las fracciones 0.12 y 0.36 equivalen a los umbrales 40 y 70 en nuestro frame 778
    oscuro = np.percentile(suave, 1)
    tipico = np.median(suave)
    umbral_pupila = oscuro + 0.12 * (tipico - oscuro)
    umbral_fondo = oscuro + 0.36 * (tipico - oscuro)

    # Marcador de pupila: la región oscura más circular
    candidatas = (suave < umbral_pupila).astype(np.uint8)
    candidatas = cv2.morphologyEx(candidatas, cv2.MORPH_OPEN,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    marcador_pupila = region_mas_circular(candidatas)
    if not marcador_pupila.any():
        return np.zeros_like(gris)

    # Tamaño del marcador: lo usamos para escalar las demás operaciones,
    # así funcionan igual con pupilas chicas o grandes
    radio_marcador = np.sqrt(marcador_pupila.sum() / np.pi)
    ys, xs = np.nonzero(marcador_pupila)
    filas, columnas = np.mgrid[0:gris.shape[0], 0:gris.shape[1]]
    distancia_al_marcador = np.hypot(columnas - xs.mean(), filas - ys.mean())

    marcadores = np.zeros(gris.shape, dtype=np.int32)
    marcadores[suave > umbral_fondo] = 2
    # Todo lo que esté lejos del marcador también es fondo.
    # Esto está porque en frames sobreexpuestos la sombra del párpado sobre el
    # iris quedaba sin marcar, y la pupila se "derramaba" hacia arriba
    marcadores[distancia_al_marcador > 1.5 * radio_marcador] = 2
    marcadores[marcador_pupila > 0] = 1

    etiquetas = watershed(gradiente, marcadores)
    mascara = (etiquetas == 1).astype(np.uint8) * 255

    # Apertura del tamaño de la pupila, quedándonos con la parte que contiene al marcador.
    # Esto está para cortar pestañas que quedaban pegadas a la máscara
    tam = _impar(radio_marcador)
    abierta = cv2.morphologyEx(mascara, cv2.MORPH_OPEN,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tam, tam)))
    n, regiones = cv2.connectedComponents(abierta)
    if n > 1:
        superposicion = [((regiones == i) & (marcador_pupila > 0)).sum() for i in range(1, n)]
        if max(superposicion) > 0:
            mascara = (regiones == 1 + int(np.argmax(superposicion))).astype(np.uint8) * 255

    # Cierre y envolvente convexa, como en el Módulo 2.
    # El elemento estructurante es proporcional a la pupila (en nuestro frame 778
    # equivale a los 21 px elegidos en el Módulo 2)
    tam = _impar(0.6 * radio_marcador)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tam, tam)))
    if mascara.any():
        mascara = (convex_hull_image(mascara > 0) * 255).astype(np.uint8)
    return mascara


# Redondea al impar más cercano (los elementos estructurantes necesitan tamaño impar)
def _impar(n):
    n = int(round(n))
    return max(3, n + (n % 2 == 0))