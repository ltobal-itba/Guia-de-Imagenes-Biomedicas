# Segmentación de la pupila pensada para procesar el video completo.
#
# El watershed del Módulo 2 usa umbrales fijos (40 y 70) que funcionan en el frame de trabajo,
# pero a lo largo del video la cámara reajusta la exposición tras cada cambio de luz: en los
# tramos oscuros el iris queda casi tan oscuro como la pupila y la segmentación se "derrama"
# hacia la sombra del párpado. Por eso acá se usa un método adaptativo en tres pasos:
#   1. Umbral adaptativo: nivel de la pupila y del iris estimados en cada frame.
#   2. Rayos desde el centro (método "starburst"): en cada dirección se busca dónde la
#      intensidad cruza el punto medio entre pupila e iris, es decir, el borde pupilar.
#   3. Ajuste robusto de un círculo a esos puntos, descartando los rayos que se escapan
#      hacia el párpado o las pestañas (valores atípicos).

import cv2
import numpy as np


def quitar_reflejos(gris, delta=35, tam=21):
    # Reemplaza los reflejos corneales (puntos mucho más claros que su entorno) por la
    # intensidad de los alrededores, para que no corten los rayos dentro de la pupila.
    fondo = cv2.medianBlur(gris, tam)
    reflejos = (gris.astype(int) - fondo.astype(int)) > delta
    reflejos = cv2.dilate(reflejos.astype(np.uint8), np.ones((5, 5), np.uint8))
    return cv2.inpaint(gris, reflejos, 5, cv2.INPAINT_TELEA), reflejos


def _ajustar_circulo(x, y):
    # Ajuste algebraico de un círculo por mínimos cuadrados (método de Kasa).
    A = np.column_stack([x, y, np.ones_like(x)])
    b = x ** 2 + y ** 2
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    xc, yc = c[0] / 2, c[1] / 2
    r = np.sqrt(max(c[2] + xc ** 2 + yc ** 2, 0))
    return xc, yc, r


def ajustar_circulo_robusto(x, y, iteraciones=5, k=2.5, usar=None):
    # Ajusta el círculo y descarta iterativamente los puntos con residuo grande (> k MAD).
    # `usar` permite arrancar desde un subconjunto de puntos (por ejemplo, los que coinciden
    # con el círculo del frame anterior).
    usar = np.ones(len(x), bool) if usar is None else usar.copy()
    for _ in range(iteraciones):
        if usar.sum() < 6:
            break
        xc, yc, r = _ajustar_circulo(x[usar], y[usar])
        residuo = np.abs(np.hypot(x - xc, y - yc) - r)
        mad = np.median(residuo[usar]) * 1.4826 + 0.5
        nuevo = residuo < k * mad
        if (nuevo == usar).all():
            break
        usar = nuevo
    xc, yc, r = _ajustar_circulo(x[usar], y[usar])
    return xc, yc, r, usar


def segmentar_pupila_video(roi_rgb, circulo_previo=None, n_rayos=48, sector_superior=150, fraccion_umbral=0.3):
    # Devuelve (máscara 0/255, info) para una ROI en RGB. info es None si no se encontró
    # una pupila confiable (por ejemplo, en un parpadeo).
    # circulo_previo = (xc, yc, r) de la pupila en el frame anterior, en coordenadas de esta ROI.
    # Se usa para buscar el centro de la pupila cerca de donde estaba (evita arrancar desde
    # una cripta oscura del iris o desde la sombra del párpado).
    gris = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY)
    gris, _ = quitar_reflejos(gris)
    suave = cv2.GaussianBlur(gris, (0, 0), 2).astype(float)
    H, W = suave.shape
    vacia = np.zeros((H, W), np.uint8)

    # 1) Núcleo oscuro: píxeles cercanos al mínimo de la ROI. Con la transformada de distancia
    #    se busca el centro del disco más grande que entra en la región oscura de la pupila.
    nivel_pupila = np.percentile(suave, 0.5)
    oscuro = (suave < nivel_pupila + 25).astype(np.uint8)
    oscuro = cv2.morphologyEx(oscuro, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    dist = cv2.distanceTransform(oscuro, cv2.DIST_L2, 5)
    yy, xx = np.mgrid[:H, :W]
    r0 = 0
    if circulo_previo is not None:
        # Primero se busca cerca de donde estaba la pupila en el frame anterior
        xp, yp, rp = circulo_previo
        cerca = np.hypot(xx - xp, yy - yp) < 0.6 * rp
        if cerca.any():
            yc0, xc0 = np.unravel_index(np.argmax(np.where(cerca, dist, 0)), dist.shape)
            r0 = dist[yc0, xc0]
    if r0 < 4:
        # Sin referencia previa: se analiza cada región oscura por separado. La pupila es una región
        # compacta, casi un disco: su área ocupa buena parte del círculo que la encierra (aunque un
        # reflejo le haga una muesca). La sombra del párpado y las pestañas forman bandas alargadas,
        # que ocupan una fracción muy chica de su círculo envolvente. Entre las regiones compactas
        # se elige la que tiene el disco inscripto más grande.
        n, etiquetas, stats, _ = cv2.connectedComponentsWithStats(oscuro)
        mejor_compacta, mejor_cualquiera = None, None
        for k in range(1, n):
            region = (etiquetas == k).astype(np.uint8)
            dist_k = np.where(region > 0, dist, 0)
            yk, xk = np.unravel_index(np.argmax(dist_k), dist_k.shape)
            rk = dist_k[yk, xk]
            if rk < 4:
                continue
            contornos, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            _, radio_envolvente = cv2.minEnclosingCircle(max(contornos, key=cv2.contourArea))
            compacidad = stats[k, cv2.CC_STAT_AREA] / (np.pi * radio_envolvente ** 2)
            if compacidad >= 0.4 and (mejor_compacta is None or rk > mejor_compacta[0]):
                mejor_compacta = (rk, yk, xk)
            if mejor_cualquiera is None or rk > mejor_cualquiera[0]:
                mejor_cualquiera = (rk, yk, xk)
        elegido = mejor_compacta or mejor_cualquiera
        if elegido is None:
            return vacia, None
        r0, yc0, xc0 = elegido

    # 2) Niveles de pupila e iris alrededor de ese centro
    d = np.hypot(xx - xc0, yy - yc0)
    nivel_pup = np.median(suave[d < 0.7 * r0])
    corona = suave[(d > 1.6 * r0) & (d < 2.6 * r0)]
    if corona.size == 0:
        return vacia, None
    nivel_iris = np.percentile(corona, 60)
    if nivel_iris - nivel_pup < 12:          # sin contraste: pupila tapada o desenfocada
        return vacia, None
    umbral = nivel_pup + fraccion_umbral * (nivel_iris - nivel_pup)

    # 3) Rayos: primer cruce del umbral desde el centro hacia afuera.
    #    No se trazan rayos hacia arriba (sector de `sector_superior` grados centrado en la vertical):
    #    ahí el borde de la pupila suele quedar tapado por el párpado superior, las pestañas o su
    #    sombra, y los rayos se escaparían. El círculo se ajusta con el resto del contorno.
    #    (En la imagen el eje y apunta hacia abajo, así que "arriba" es el ángulo 270°.)
    mitad = np.radians(sector_superior) / 2
    angulos = np.linspace(1.5 * np.pi + mitad, 1.5 * np.pi - mitad + 2 * np.pi, n_rayos) % (2 * np.pi)
    radios = np.arange(0, 4 * r0 + 10, 0.5)
    px, py = [], []
    for a in angulos:
        xs = xc0 + radios * np.cos(a)
        ys = yc0 + radios * np.sin(a)
        dentro = (xs >= 0) & (xs <= W - 1) & (ys >= 0) & (ys <= H - 1)
        if dentro.sum() < 5:
            continue
        perfil = cv2.remap(suave.astype(np.float32), xs[dentro].astype(np.float32).reshape(1, -1),
                           ys[dentro].astype(np.float32).reshape(1, -1), cv2.INTER_LINEAR).ravel()
        cruce = np.flatnonzero(perfil > umbral)
        if cruce.size == 0 or cruce[0] == 0:
            continue
        i = cruce[0]
        # Interpolación lineal entre las dos muestras para ubicar el borde con precisión subpíxel
        f = (umbral - perfil[i - 1]) / (perfil[i] - perfil[i - 1] + 1e-9)
        r_borde = radios[dentro][i - 1] + f * 0.5
        px.append(xc0 + r_borde * np.cos(a))
        py.append(yc0 + r_borde * np.sin(a))

    if len(px) < 0.5 * n_rayos:
        return vacia, None
    px, py = np.array(px), np.array(py)
    xc, yc, r, inliers = ajustar_circulo_robusto(px, py)
    fraccion = inliers.mean()
    if fraccion < 0.5 or r < 5 or r > 0.45 * H:
        return vacia, None

    mascara = vacia.copy()
    cv2.circle(mascara, (int(round(xc)), int(round(yc))), int(round(r)), 255, -1)
    info = {"cx_roi": xc, "cy_roi": yc, "radio_px": r, "fraccion_rayos": fraccion,
            "contraste": nivel_iris - nivel_pup, "puntos": (px, py, inliers)}
    return mascara, info
