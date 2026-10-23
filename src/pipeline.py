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
        return df

    d_interp = np.interp(t, t[valido], d[valido])
    ventana = max(int(round(ventana_suavizado_s / dt)) | 1, 5)
    d_suave = savgol_filter(d_interp, ventana, 2)
    df["diametro_px_limpio"] = d_suave
    df["area_px_limpia"] = np.pi * d_suave ** 2 / 4
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
    # Parámetros del reflejo a partir de la curva limpia de diámetro.
    #  - Basal: mediana del diámetro en el segundo previo al cambio de luz (frames válidos).
    #  - Extremo: mínimo (registro A, en los 6 s siguientes) o máximo (registro B, hasta el final).
    #  - Amplitud: |extremo - basal|, en diámetro, en área y en % del basal.
    #  - Latencia: tiempo desde el cambio de luz hasta que el diámetro se aparta del basal más
    #    que max(3 desvíos del basal, 10 % de la amplitud) y se mantiene así al menos 0,3 s.
    #  - Velocidades: derivada del diámetro suavizado. Constricción = máxima velocidad de
    #    achicamiento; redilatación = máxima velocidad de agrandamiento (en A, la posterior al mínimo).
    t = df["tiempo_s"].to_numpy()
    d = df["diametro_px_limpio"].to_numpy()
    d_val = np.where(df["valido"].to_numpy() == 1, df["diametro_px"].to_numpy(dtype=float), np.nan)
    t_est, i_est, tipo_det = detectar_estimulo(t, df["brillo"].to_numpy())
    tipo = tipo_det if tipo == "auto" else tipo
    signo = -1 if tipo == "A" else 1           # A: el diámetro baja; B: el diámetro sube

    antes = (t >= t_est - ventana_basal_s) & (t < t_est)
    if np.isfinite(d_val[antes]).sum() < 3:
        antes = t < t_est
    basal = float(np.nanmedian(d_val[antes])) if np.isfinite(d_val[antes]).any() else float(d[i_est])
    ruido = float(np.nanstd(d_val[antes])) if np.isfinite(d_val[antes]).sum() > 2 else 0.0

    # El extremo se busca sobre la curva filtrada con una mediana de 1 s, para que un error
    # puntual que haya sobrevivido a la limpieza no se tome como el mínimo o el máximo.
    # En A la constricción se completa en pocos segundos (ventana de 6 s); en B la
    # redilatación es lenta y se busca hasta el final del registro.
    dt = np.median(np.diff(t))
    d_lento = pd.Series(d).rolling(max(int(1 / dt) | 1, 3), center=True, min_periods=1).median().to_numpy()
    fin_ventana = t_est + ventana_respuesta_s if tipo == "A" else t[-1]
    despues = np.flatnonzero((t >= t_est) & (t <= fin_ventana))
    i_ext = int(despues[np.argmin(d_lento[despues])] if signo == -1 else despues[np.argmax(d_lento[despues])])
    extremo = float(d_lento[i_ext])
    amplitud = abs(extremo - basal)

    # La respuesta tiene que sostenerse al menos 0,3 s para contar como inicio (no un pico aislado)
    umbral = max(3 * ruido, 0.10 * amplitud)
    supera = (signo * (d - basal) > umbral) & (np.arange(len(t)) >= i_est)
    n_sostener = max(int(round(0.3 / dt)), 1)
    sostenido = np.convolve(supera.astype(int), np.ones(n_sostener, int), mode="full")[n_sostener - 1:] == n_sostener
    candidatos = np.flatnonzero(sostenido[:i_ext + 1])
    i_ini = int(candidatos[0]) if candidatos.size else i_est
    latencia = float(t[i_ini] - t_est)

    v = np.gradient(d, t)
    if tipo == "A":
        v_constr = float(-v[i_est:i_ext + 1].min()) if i_ext > i_est else np.nan
        v_redil = float(v[i_ext:].max()) if len(t) - i_ext > 3 else np.nan
    else:
        v_constr = np.nan                       # en B no hay constricción después del estímulo
        v_redil = float(v[i_est:i_ext + 1].max()) if i_ext > i_est else np.nan
    v_media = amplitud / (t[i_ext] - t[i_ini]) if t[i_ext] > t[i_ini] else np.nan

    def r(x, n=2):
        return round(float(x), n) if np.isfinite(x) else np.nan

    return {
        "tipo_registro": tipo,
        "t_estimulo_s": r(t_est, 3),
        "latencia_s": r(latencia, 3),
        "t_extremo_s": r(t[i_ext], 3),
        "diametro_basal_px": r(basal),
        "diametro_extremo_px": r(extremo),
        "amplitud_diametro_px": r(amplitud),
        "area_basal_px": r(np.pi * basal ** 2 / 4, 1),
        "area_extremo_px": r(np.pi * extremo ** 2 / 4, 1),
        "amplitud_area_px": r(abs(np.pi * (basal ** 2 - extremo ** 2) / 4), 1),
        "amplitud_relativa_pct": r(100 * amplitud / basal, 1),
        "vel_constriccion_max_px_s": r(v_constr),
        "vel_redilatacion_max_px_s": r(v_redil),
        "vel_media_respuesta_px_s": r(v_media),
        "frames_invalidos_pct": r(100 * (1 - df["valido"].mean()), 1),
        "duracion_s": r(t[-1] - t[0]),
        "fps_medio": r((len(t) - 1) / (t[-1] - t[0])),
    }


# 6) Todo junto

COLUMNAS_CURVA = ["tiempo_s", "area_px", "diametro_px", "valido", "area_px_limpia",
                  "diametro_px_limpio", "brillo", "cx", "cy", "ajuste_rayos"]


def analizar_video(video_path, carpeta_salida, tipo="auto", mm_por_px=None, progreso=True):
    # Corre el pipeline completo y guarda la curva y los parámetros en CSV, más un gráfico.
    video_path, carpeta_salida = Path(video_path), Path(carpeta_salida)
    carpeta_salida.mkdir(parents=True, exist_ok=True)
    nombre = video_path.stem

    if progreso:
        print(f"Procesando {video_path.name}")
    df = limpiar_curva(procesar_video(video_path, progreso=progreso))
    param = {"video": video_path.name, **calcular_parametros(df, tipo=tipo)}

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
