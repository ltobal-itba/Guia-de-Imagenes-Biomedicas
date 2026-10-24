# Pipeline completo del reflejo fotomotor (Módulo 3.2 y entrega final):
# video -> localización del ojo -> segmentación frame a frame -> curva D(t)/A(t)
# -> limpieza de artefactos -> parámetros fisiológicos -> CSV.

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from skimage.feature import blob_log

from .pupila_video import segmentar_pupila_video


# Tamaño de la ROI relativo al ancho del frame (400 x 300 px en un video de 1080 px de ancho,
# igual que la ROI usada en los Módulos 1 a 3)
ROI_ANCHO_REL = 400 / 1080
ROI_ALTO_REL = 300 / 1080


# 1) Lectura del video

def leer_frames(video_path):
    # Devuelve (índice, tiempo en s, frame BGR) para cada frame del video.
    # El tiempo se toma de la marca temporal de cada frame, porque los videos de celular
    # suelen tener frame rate variable.
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"No se pudo abrir el video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    idx = 0
    t_anterior = -1.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        if t <= t_anterior:            # sin marca temporal válida: se usa el fps nominal
            t = idx / fps
        t_anterior = t
        yield idx, t, frame
        idx += 1
    cap.release()


# 2) Localización automática del ojo

def _candidatos_pupila(frame, escala=4):
    # Busca regiones oscuras y redondeadas (pupila o pupila + iris) con un detector de blobs (LoG).
    # Cada candidato recibe un puntaje: cuánto más clara es la corona que lo rodea que su centro,
    # tomando el peor de 8 sectores. Así solo puntúan alto las regiones rodeadas de algo claro por
    # todos lados (la pupila dentro del iris y la esclerótica), y no el pelo, el fondo o las sombras.
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gris = cv2.resize(gris, None, fx=1 / escala, fy=1 / escala, interpolation=cv2.INTER_AREA)
    gris = cv2.GaussianBlur(gris, (0, 0), 1.5).astype(float)
    H, W = gris.shape
    blobs = blob_log((255 - gris) / 255, min_sigma=3, max_sigma=14, num_sigma=12, threshold=0.05)

    yy, xx = np.mgrid[:H, :W]
    bordes = np.linspace(-np.pi, np.pi, 9)
    candidatos = []
    for by, bx, sigma in blobs:
        r = sigma * np.sqrt(2)
        if bx < 2 * r or by < 2 * r or bx > W - 2 * r or by > H - 2 * r:
            continue
        d = np.hypot(yy - by, xx - bx)
        disco = gris[d <= 0.7 * r]
        en_corona = (d > 1.3 * r) & (d < 2.5 * r)
        if disco.size < 5:
            continue
        corona = gris[en_corona]
        angulo = np.arctan2(yy - by, xx - bx)[en_corona]
        sectores = [np.median(corona[(angulo >= a) & (angulo < b)]) for a, b in zip(bordes[:-1], bordes[1:])]
        candidatos.append((min(sectores) - np.median(disco), bx * escala, by * escala))
    return candidatos


def localizar_ojo(video_path, n_muestras=9):
    # Centro aproximado del ojo (x, y) en píxeles del frame completo.
    # Se analizan varios frames repartidos en el video y se elige la posición donde los
    # candidatos se repiten con mayor puntaje (votación), para no depender de un solo frame.
    cap = cv2.VideoCapture(str(video_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ancho, votos = None, []
    for idx in np.linspace(0, max(n - 10, 0), n_muestras).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        ancho = frame.shape[1]
        mejores = sorted(_candidatos_pupila(frame), reverse=True)[:3]
        votos += [c for c in mejores if c[0] > 0]
    cap.release()
    if not votos:
        raise RuntimeError("No se encontró el ojo en el video")

    votos = np.array(votos)
    radio = 0.06 * ancho               # candidatos a menos de ~65 px se consideran el mismo punto
    soporte = [votos[np.hypot(votos[:, 1] - x, votos[:, 2] - y) < radio, 0].sum() for _, x, y in votos]
    _, x0, y0 = votos[int(np.argmax(soporte))]
    vecinos = votos[np.hypot(votos[:, 1] - x0, votos[:, 2] - y0) < radio]
    return float(np.median(vecinos[:, 1])), float(np.median(vecinos[:, 2]))


def recortar_roi(frame, cx, cy):
    # ROI centrada en (cx, cy), del mismo tamaño relativo que la de los Módulos 1 a 3.
    H, W = frame.shape[:2]
    w, h = int(ROI_ANCHO_REL * W), int(ROI_ALTO_REL * W)
    x0 = int(np.clip(cx - w / 2, 0, W - w))
    y0 = int(np.clip(cy - h / 2, 0, H - h))
    return frame[y0:y0 + h, x0:x0 + w], x0, y0


# 3) Medición frame a frame

def procesar_video(video_path, progreso=True, transformar=None):
    # Segmenta la pupila en todos los frames y devuelve una tabla con una fila por frame.
    # La ROI sigue al ojo: se recentra en la pupila cada vez que la medición es válida.
    # `transformar` (opcional) es una función que se aplica a cada frame antes de procesarlo;
    # se usa en el Módulo 4 para medir el efecto de la compresión JPEG sobre la curva.
    cx, cy = localizar_ojo(video_path)
    previo = None                     # (x, y, r) de la pupila en el último frame válido, en coordenadas del frame
    filas = []
    for idx, t, frame in leer_frames(video_path):
        if transformar is not None:
            frame = transformar(frame)
        roi_bgr, x0, y0 = recortar_roi(frame, cx, cy)
        circulo = (previo[0] - x0, previo[1] - y0, previo[2]) if previo else None
        _, info = segmentar_pupila_video(cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB), circulo_previo=circulo)

        fila = {"frame": idx, "tiempo_s": t,
                "brillo": float(cv2.cvtColor(frame[::4, ::4], cv2.COLOR_BGR2GRAY).mean())}
        if info is not None:
            r = info["radio_px"]
            fila.update({"diametro_px": 2 * r, "area_px": np.pi * r ** 2,
                         "ajuste_rayos": info["fraccion_rayos"], "contraste": info["contraste"],
                         "cx": x0 + info["cx_roi"], "cy": y0 + info["cy_roi"]})
            # Seguimiento suavizado: un error puntual no arrastra la ROI fuera del ojo
            cx, cy = 0.7 * cx + 0.3 * fila["cx"], 0.7 * cy + 0.3 * fila["cy"]
            previo = (fila["cx"], fila["cy"], r)
        filas.append(fila)
        if progreso and idx % 200 == 0:
            print(f"  frame {idx}")

    df = pd.DataFrame(filas).set_index("frame")
    for col in ["diametro_px", "area_px", "ajuste_rayos", "contraste", "cx", "cy"]:
        if col not in df:
            df[col] = np.nan
    return df


# 4) Artefactos: parpadeos, movimientos y errores de segmentación

def limpiar_curva(df, ventana_s=0.5, n_mad=3.0, ajuste_min=0.6, ventana_suavizado_s=0.25, salto_brillo_max=12):
    # 1. Inválidos directos: frames sin pupila detectada (parpadeo: no hay núcleo oscuro o no
    #    hay contraste), con un ajuste de círculo pobre (pocos rayos coinciden con el círculo)
    #    o tomados mientras la cámara reajusta la exposición (salto brusco del brillo del frame).
    # 2. Valores atípicos respecto de sus vecinos (filtro de Hampel sobre el diámetro):
    #    capturan los cierres parciales del párpado y los saltos por movimientos del ojo.
    # 3. Los frames descartados se reemplazan por interpolación lineal en el tiempo y la curva
    #    se suaviza con un filtro de Savitzky-Golay (conserva mejor los cambios rápidos que una media móvil).
    df = df.copy()
    t = df["tiempo_s"].to_numpy()
    dt = np.median(np.diff(t))
    d = df["diametro_px"].to_numpy(dtype=float)

    valido = np.isfinite(d) & (df["ajuste_rayos"].to_numpy(dtype=float) >= ajuste_min)

    # Frames de transición de la cámara: justo al prender o apagar la luz, la exposición
    # automática tarda unos frames en ajustarse y la imagen queda saturada u oscura.
    brillo = df["brillo"].to_numpy()
    salto_brillo = np.abs(np.diff(brillo, prepend=brillo[0]))
    transicion = salto_brillo > salto_brillo_max
    transicion |= np.roll(transicion, 1)           # también el frame siguiente
    transicion |= brillo < 0.6 * np.median(brillo)  # frames muy subexpuestos
    valido &= ~transicion

    k = max(int(round(ventana_s / dt / 2)), 2)
    serie = pd.Series(np.where(valido, d, np.nan))
    mediana = serie.rolling(2 * k + 1, center=True, min_periods=3).median()
    desvio = (serie - mediana).abs()
    mad = desvio.rolling(2 * k + 1, center=True, min_periods=3).median()
    escala = 1.4826 * mad.clip(lower=0.015 * np.nanmedian(d)) if valido.any() else mad
    valido &= ~(desvio > n_mad * escala).to_numpy()

    df["valido"] = valido.astype(int)
    if valido.sum() < 5:
        df["diametro_px_limpio"] = np.nan
        df["area_px_limpia"] = np.nan
        df["interpolado"] = 0
        return df

    
    # Interpolación entre mediciones válidas
    d_interp = np.interp(t, t[valido], d[valido])

    # Evitamos extrapolar antes de la primera medición válida
    # y después de la última
    indices_validos = np.flatnonzero(valido)

    d_interp[:indices_validos[0]] = np.nan
    d_interp[indices_validos[-1] + 1:] = np.nan

    # Las interrupciones largas se conservan sin estimación
    # Solo interpolamos huecos de hasta 0.5 segundos
    max_hueco_s = 0.5

    for inicio, fin in zip(indices_validos[:-1], indices_validos[1:]):
        if t[fin] - t[inicio] > max_hueco_s:
            d_interp[inicio + 1:fin] = np.nan

    # Suavizado por tramos, sin atravesar huecos largos
    d_suave = np.full(len(d_interp), np.nan)
    disponibles = np.isfinite(d_interp)

    cambios = np.diff(
        np.r_[False, disponibles, False].astype(int)
    )
    comienzos = np.flatnonzero(cambios == 1)
    finales = np.flatnonzero(cambios == -1)

    ventana_base = max(
        int(round(ventana_suavizado_s / dt)) | 1, 5
    )

    for inicio, fin in zip(comienzos, finales):
        tramo = d_interp[inicio:fin]

        ventana = min(
            ventana_base,
            len(tramo) if len(tramo) % 2 else len(tramo) - 1
        )

        if ventana >= 5:
            d_suave[inicio:fin] = savgol_filter(
                tramo, ventana, 2
            )
        else:
            d_suave[inicio:fin] = tramo

    # Guardamos resultados y señalamos valores interpolados
    df["diametro_px_limpio"] = d_suave
    df["area_px_limpia"] = np.pi * d_suave ** 2 / 4
    df["interpolado"] = (
        ~valido & np.isfinite(d_suave)
    ).astype(int)

    return df

# 5) Estímulo y parámetros fisiológicos

def detectar_estimulo(t, brillo, margen=3):
    # Instante del cambio de luz: el mayor salto del brillo medio del frame entre frames
    # consecutivos. Si el brillo sube es un registro A (oscuridad -> luz); si baja, un registro B.
    b = pd.Series(brillo).rolling(3, center=True, min_periods=1).median().to_numpy()
    salto = np.diff(b)
    salto[:margen] = 0
    salto[-margen:] = 0
    i = int(np.argmax(np.abs(salto)))
    return float(t[i + 1]), i + 1, ("A" if salto[i] > 0 else "B")



def calcular_parametros(df, tipo="auto", ventana_basal_s=1.0, ventana_respuesta_s=6.0):
    # Cálculo de parámetros fisiológicos considerando posibles
    # intervalos sin mediciones confiables (NaN).

    t = df["tiempo_s"].to_numpy(dtype=float)
    d = df["diametro_px_limpio"].to_numpy(dtype=float)
    d_original = df["diametro_px"].to_numpy(dtype=float)
    valido = df["valido"].to_numpy() == 1

    t_est, i_est, tipo_det = detectar_estimulo(
        t, df["brillo"].to_numpy()
    )
    tipo = tipo_det if tipo == "auto" else tipo
    signo = -1 if tipo == "A" else 1
    dt = np.median(np.diff(t))

    def redondear(x, n=2):
        return round(float(x), n) if np.isfinite(x) else np.nan

    # Valores predeterminados si no hay datos suficientes
    basal = extremo = amplitud = np.nan
    latencia = t_extremo = np.nan
    v_constr = v_redil = v_media = np.nan
    ruido = np.nan

    # 1. Diámetro basal: mediciones válidas antes del estímulo
    antes = (
        (t >= t_est - ventana_basal_s)
        & (t < t_est)
        & valido
        & np.isfinite(d_original)
    )

    if antes.sum() >= 3:
        basal = float(np.median(d_original[antes]))
        ruido = float(np.std(d_original[antes]))

    # 2. Buscar extremo en la ventana de respuesta
    fin = t_est + ventana_respuesta_s if tipo == "A" else t[-1]

    ventana = (
        (t >= t_est)
        & (t <= fin)
        & np.isfinite(d)
    )

    if np.isfinite(basal) and ventana.sum() >= 5:

        ancho_mediana = max(int(round(1.0 / dt)) | 1, 3)

        d_lento = pd.Series(d).rolling(
            ancho_mediana,
            center=True,
            min_periods=1
        ).median().to_numpy()

        # Solo considerar extremos con medición limpia disponible
        candidatos = np.flatnonzero(
            ventana & np.isfinite(d_lento)
        )

        if tipo == "A":
            i_ext = candidatos[np.argmin(d_lento[candidatos])]
        else:
            i_ext = candidatos[np.argmax(d_lento[candidatos])]

        extremo = float(d_lento[i_ext])
        t_extremo = float(t[i_ext])

        # Comprobar que la respuesta tenga la dirección esperada
        cambio = signo * (extremo - basal)

        if cambio > 0:
            amplitud = cambio

            # 3. Latencia
            umbral = max(3 * ruido, 0.10 * amplitud)
            supera = (
                (signo * (d - basal) > umbral)
                & np.isfinite(d)
                & (t >= t_est)
            )

            n_sostener = max(int(round(0.3 / dt)), 1)

            # Primer bloque consecutivo que supera el umbral
            contador = 0
            i_inicio = None

            for i in range(i_est, i_ext + 1):
                contador = contador + 1 if supera[i] else 0

                if contador >= n_sostener:
                    i_inicio = i - n_sostener + 1
                    break

            if i_inicio is not None:
                latencia = float(t[i_inicio] - t_est)

            # 4. Velocidades: derivadas solo en tramos continuos
            velocidad = np.full(len(d), np.nan)

            disponibles = np.isfinite(d)
            cambios = np.diff(
                np.r_[False, disponibles, False].astype(int)
            )

            comienzos = np.flatnonzero(cambios == 1)
            finales = np.flatnonzero(cambios == -1)

            for inicio, final in zip(comienzos, finales):
                if final - inicio >= 3:
                    velocidad[inicio:final] = np.gradient(
                        d[inicio:final],
                        t[inicio:final]
                    )

            if tipo == "A":
                tramo = velocidad[i_est:i_ext + 1]
                if np.isfinite(tramo).any():
                    v_constr = float(-np.nanmin(tramo))

                tramo = velocidad[i_ext:]
                if np.isfinite(tramo).any():
                    v_redil = float(np.nanmax(tramo))

            else:
                tramo = velocidad[i_est:i_ext + 1]
                if np.isfinite(tramo).any():
                    v_redil = float(np.nanmax(tramo))

            if i_inicio is not None and t[i_ext] > t[i_inicio]:
                v_media = amplitud / (t[i_ext] - t[i_inicio])

    # 5. Resultados
    area_basal = np.pi * basal**2 / 4
    area_extremo = np.pi * extremo**2 / 4

    return {
        "tipo_registro": tipo,
        "t_estimulo_s": redondear(t_est, 3),
        "latencia_s": redondear(latencia, 3),
        "t_extremo_s": redondear(t_extremo, 3),
        "diametro_basal_px": redondear(basal),
        "diametro_extremo_px": redondear(extremo),
        "amplitud_diametro_px": redondear(amplitud),
        "area_basal_px": redondear(area_basal, 1),
        "area_extremo_px": redondear(area_extremo, 1),
        "amplitud_area_px": redondear(
            abs(area_basal - area_extremo), 1
        ),
        "amplitud_relativa_pct": redondear(
            100 * amplitud / basal
        ) if np.isfinite(basal) and basal > 0 else np.nan,
        "vel_constriccion_max_px_s": redondear(v_constr),
        "vel_redilatacion_max_px_s": redondear(v_redil),
        "vel_media_respuesta_px_s": redondear(v_media),
        "frames_invalidos_pct": redondear(
            100 * (1 - df["valido"].mean()), 1
        ),
        "duracion_s": redondear(t[-1] - t[0]),
        "fps_medio": redondear(
            (len(t) - 1) / (t[-1] - t[0])
        ),
    }



def evaluar_calidad(df, parametros):
    """
    Evalúa por separado la calidad técnica y las
    advertencias relacionadas con la respuesta fisiológica.

    Estos controles son orientativos y no sustituyen
    la validación contra mediciones manuales.
    """

    t = df["tiempo_s"].to_numpy(dtype=float)
    d = df["diametro_px_limpio"].to_numpy(dtype=float)
    valido = df["valido"].to_numpy() == 1

    t_est = parametros["t_estimulo_s"]
    basal = parametros["diametro_basal_px"]
    latencia = parametros["latencia_s"]
    amplitud = parametros["amplitud_diametro_px"]

    # 1. Calidad técnica
    cobertura = float(np.isfinite(d).mean())
    porcentaje_invalidos = 100 * (1 - valido.mean())

    if cobertura < 0.7:
        calidad_tecnica = "insuficiente"
    elif porcentaje_invalidos > 10:
        calidad_tecnica = "revisar"
    else:
        calidad_tecnica = "cobertura_adecuada"

    # 2. Advertencias fisiológicas
    advertencias = []

    if not all(np.isfinite(x) for x in
               [t_est, basal, latencia, amplitud]):
        advertencias.append(
            "Parámetros fisiológicos no estimables"
        )
    else:
        if latencia > 1.5:
            advertencias.append(
                "Latencia elevada: revisar inicio de respuesta"
            )

        # Tendencia del diámetro antes del estímulo
        antes = (
            (t >= t_est - 1.0)
            & (t < t_est)
            & valido
            & np.isfinite(d)
        )

        if antes.sum() >= 5:
            pendiente = np.polyfit(
                t[antes], d[antes], 1
            )[0]

            if abs(pendiente) > 3:
                advertencias.append(
                    "Tendencia importante antes del estímulo"
                )

        if t_est < 0.5 or t_est > t[-1] - 0.5:
            advertencias.append(
                "Estímulo cercano al extremo del video"
            )

    if advertencias:
        estado_fisiologico = "revisar"
        motivo = "; ".join(advertencias)
    else:
        estado_fisiologico = "sin_alertas_automaticas"
        motivo = "Sin advertencias en los controles aplicados"

    return calidad_tecnica, estado_fisiologico, motivo


# 6) Todo junto

COLUMNAS_CURVA = [
    "tiempo_s",
    "area_px",
    "diametro_px",
    "valido",
    "interpolado",
    "area_px_limpia",
    "diametro_px_limpio",
    "brillo",
    "cx",
    "cy",
    "ajuste_rayos"
]


def analizar_video(video_path, carpeta_salida, tipo="auto", mm_por_px=None, progreso=True):
    # Corre el pipeline completo y guarda la curva y los parámetros en CSV, más un gráfico.
    video_path, carpeta_salida = Path(video_path), Path(carpeta_salida)
    carpeta_salida.mkdir(parents=True, exist_ok=True)
    nombre = video_path.stem

    if progreso:
        print(f"Procesando {video_path.name}")
    df = limpiar_curva(procesar_video(video_path, progreso=progreso))
    param = {"video": video_path.name, **calcular_parametros(df, tipo=tipo)}
    
    # Evaluación automática de calidad
    calidad, estado, motivo = evaluar_calidad(df, param)

    param["calidad_tecnica"] = calidad
    param["estado_fisiologico"] = estado
    param["motivo_revision"] = motivo



    columnas = list(COLUMNAS_CURVA)
    if mm_por_px:
        df["diametro_mm"] = df["diametro_px"] * mm_por_px
        df["diametro_mm_limpio"] = df["diametro_px_limpio"] * mm_por_px
        columnas += ["diametro_mm", "diametro_mm_limpio"]
        for k in ["diametro_basal_px", "diametro_extremo_px", "amplitud_diametro_px",
                  "vel_constriccion_max_px_s", "vel_redilatacion_max_px_s", "vel_media_respuesta_px_s"]:
            param[k.replace("px", "mm")] = round(param[k] * mm_por_px, 3) if np.isfinite(param[k]) else np.nan

    df[columnas].round(4).to_csv(carpeta_salida / f"{nombre}_curva.csv")
    pd.DataFrame([param]).to_csv(carpeta_salida / f"{nombre}_parametros.csv", index=False)
    graficar(df, param, carpeta_salida / f"{nombre}_curva.png")
    if progreso:
        print(f"  listo: {nombre}_curva.csv, {nombre}_parametros.csv")
    return df, param


def graficar(df, param, ruta=None, ax=None):
    import matplotlib.pyplot as plt

    propio = ax is None
    if propio:
        fig, ax = plt.subplots(figsize=(11, 4.5))
    t = df["tiempo_s"]
    malo = df["valido"] == 0
    ax.plot(t[~malo], df.loc[~malo, "diametro_px"], ".", ms=3, color="0.6", label="Medición por frame")
    ax.plot(t[malo], df.loc[malo, "diametro_px"], "x", ms=4, color="tab:red", label="Descartado (artefacto)")
    ax.plot(t, df["diametro_px_limpio"], "-", lw=2, color="tab:blue", label="Curva limpia")
    ax.axvline(param["t_estimulo_s"], color="orange", ls="--", label="Cambio de luz")
    ax.axvline(param["t_estimulo_s"] + param["latencia_s"], color="green", ls=":", label="Inicio de la respuesta")
    ax.axhline(param["diametro_basal_px"], color="0.3", lw=0.8, ls="--")
    ax.plot(param["t_extremo_s"], param["diametro_extremo_px"],
            "v" if param["tipo_registro"] == "A" else "^", color="black", ms=8)
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Diámetro pupilar (px)")
    ax.set_title(f"{param['video']} – registro {param['tipo_registro']}: latencia {param['latencia_s']:.2f} s, "
                 f"amplitud {param['amplitud_diametro_px']:.1f} px ({param['amplitud_relativa_pct']:.0f} %)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    if propio:
        fig.tight_layout()
        if ruta is not None:
            fig.savefig(ruta, dpi=110)
        plt.close(fig)


def graficar_comparacion(df, suavizado_s=0.7, estimulo=None):
    import matplotlib.pyplot as plt

    t = df["tiempo_s"].to_numpy()
    d = df["diametro_px_limpio"].to_numpy()

    # Suavizado adicional, solo para visualizar
    dt = np.median(np.diff(t))
    ventana = max(5, int(round(suavizado_s / dt)) | 1)
    max_ventana = len(d) if len(d) % 2 else len(d) - 1
    ventana = min(ventana, max_ventana)

    suave = (
        savgol_filter(d, ventana, polyorder=2)
        if ventana >= 5 else d.copy()
    )

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.canvas.manager.set_window_title("Análisis pupilar")

    ax.plot(
        t, df["diametro_px"],
        color="0.75", linewidth=1, label="Original",
    )
    ax.plot(
    t, d,
    color="tab:blue", linewidth=1.2,
    label="Limpia", zorder=3,
    )

    ax.plot(
        t, suave,
        color="darkorange", linewidth=2.5,
        alpha=0.4,
        label=f"Suavizada ({suavizado_s} s)",
        zorder=2,
    )

    malos = df["valido"] == 0
    ax.scatter(
        t[malos], df.loc[malos, "diametro_px"],
        color="red", marker="x", s=20,
        label="Inválidos", zorder=3,
    )

    ax.set(
        title="Diámetro pupilar a lo largo del tiempo",
        xlabel="Tiempo (s)",
        ylabel="Diámetro pupilar (px)",
    )
    if estimulo is not None:
        t_estimulo, _, tipo = estimulo

        ax.axvline(
            t_estimulo,
            color="purple",
            linestyle="--",
            label=f"Cambio de luz ({tipo})",
        )

    # Mostrar el brillo en una segunda ventana
    fig_brillo, ax_brillo = plt.subplots(figsize=(11, 3))
    fig_brillo.canvas.manager.set_window_title("Detección del estímulo")

    ax_brillo.plot(t, df["brillo"], color="goldenrod")

    if estimulo is not None:
        ax_brillo.axvline(
            t_estimulo,
            color="purple",
            linestyle="--",
            label=f"Cambio detectado: {t_estimulo:.2f} s",
        )
        ax_brillo.legend()

    ax_brillo.set(
        title="Brillo promedio del video",
        xlabel="Tiempo (s)",
        ylabel="Intensidad media",
    )
    ax_brillo.grid(alpha=0.3)
    fig_brillo.tight_layout()
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    plt.show()