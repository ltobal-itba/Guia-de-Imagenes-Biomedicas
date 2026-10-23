# Localización automática del ojo y seguimiento de la pupila a lo largo del video.
# Reemplaza la ROI elegida a mano en los notebooks: la pupila se busca sola
# en el frame completo y después se sigue frame a frame.

import cv2
import numpy as np


# Busca la pupila en el frame completo.
# Devuelve (x, y, radio) en píxeles del frame original, o None si no la encuentra.
# Criterio: entre las regiones oscuras y redondas, la pupila es la más oscura
# y está rodeada por una zona más clara (el iris).
def localizar_pupila(frame_rgb, escala=0.25, contraste_min=30, circ_min=0.6):
    # Trabajamos sobre una versión achicada: es más rápido y alcanza para ubicarla
    chico = cv2.resize(frame_rgb, None, fx=escala, fy=escala, interpolation=cv2.INTER_AREA)
    gris = cv2.GaussianBlur(cv2.cvtColor(chico, cv2.COLOR_RGB2GRAY), (5, 5), 0)
    alto, ancho = gris.shape

    # Tamaño admisible de la pupila, relativo al ancho de la imagen
    # (así funciona aunque el video tenga otra resolución o distancia a la cámara)
    r_min, r_max = 0.015 * ancho, 0.12 * ancho
    ee = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    mejor = None   # (x, y, radio, intensidad media interior)

    # Probamos varios umbrales en lugar de uno fijo, porque el brillo del video
    # cambia mucho (por ejemplo, antes y después de prender la luz)
    for umbral in range(20, 121, 10):
        oscuro = (gris < umbral).astype(np.uint8)

        # Esto está para que las pestañas pegadas a la pupila no formen una sola
        # región con ella (si no, la forma deja de ser redonda y se descarta)
        oscuro = cv2.morphologyEx(oscuro, cv2.MORPH_OPEN, ee)

        contornos, _ = cv2.findContours(oscuro, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contornos:
            area = cv2.contourArea(c)
            (cx, cy), r = cv2.minEnclosingCircle(c)

            # 1) Tamaño razonable.
            # El mínimo está para descartar rulos chicos del pelo, que son oscuros y redondeados
            if not (r_min <= r <= r_max):
                continue

            # 2) Que no toque el borde de la imagen.
            # Esto está porque el pelo y el fondo oscuro en el borde izquierdo del
            # video aparecían como regiones más oscuras que la pupila
            if cx - r <= 1 or cy - r <= 1 or cx + r >= ancho - 2 or cy + r >= alto - 2:
                continue

            # 3) Forma redonda (descarta cejas, pestañas y sombras alargadas)
            if area / (np.pi * r ** 2) < circ_min:
                continue

            # 4) Rodeada por algo más claro (el iris).
            # Comparamos el interior con un anillo alrededor de la región
            interior = np.zeros_like(gris)
            cv2.drawContours(interior, [c], -1, 1, thickness=-1)
            anillo = np.zeros_like(gris)
            cv2.circle(anillo, (int(cx), int(cy)), int(2 * r), 1, thickness=-1)
            anillo[interior > 0] = 0
            media_interior = gris[interior > 0].mean()
            if gris[anillo > 0].mean() - media_interior < contraste_min:
                continue

            # De las regiones que pasan los filtros, nos quedamos con la más oscura.
            # Esto está porque el agujero de la regla también es redondo y está
            # rodeado de algo más claro, pero nunca es tan oscuro como la pupila
            if mejor is None or media_interior < mejor[3]:
                mejor = (cx / escala, cy / escala, r / escala, media_interior)

    return None if mejor is None else mejor[:3]


# Ubica la pupila al principio del video.
# La busca en varios frames de los primeros segundos y toma la mediana:
# esto está porque en algunos frames la búsqueda falla (por ejemplo, cuando la
# cámara se satura al prender la luz), y así un frame malo no arruina el resultado.
# Devuelve (x, y, radio) en píxeles.
def inicializar(ruta_video, n_muestras=15, segundos=3):
    cap = cv2.VideoCapture(str(ruta_video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    detecciones = []
    for idx in np.linspace(0, int(segundos * fps), n_muestras).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        resultado = localizar_pupila(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if resultado is not None:
            detecciones.append(resultado)
    cap.release()

    if not detecciones:
        raise RuntimeError("No se pudo encontrar la pupila al inicio del video.")
    x, y, r = np.median(np.array(detecciones), axis=0)
    return x, y, r


# Recorta una ROI de tamaño tam = (ancho, alto) centrada en centro = (x, y).
# Si se sale de la imagen, la desplaza para que quede adentro.
# Devuelve la ROI y la posición de su esquina superior izquierda.
def recortar_roi(frame_rgb, centro, tam):
    ancho, alto = tam
    alto_f, ancho_f = frame_rgb.shape[:2]
    x0 = int(np.clip(centro[0] - ancho / 2, 0, ancho_f - ancho))
    y0 = int(np.clip(centro[1] - alto / 2, 0, alto_f - alto))
    return frame_rgb[y0:y0 + alto, x0:x0 + ancho], (x0, y0)