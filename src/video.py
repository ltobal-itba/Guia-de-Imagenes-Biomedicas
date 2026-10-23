# Lectura del video de entrada.
# Abre el video, obtiene su información básica (FPS, cantidad de frames,
# resolución) y, si OpenCV no puede leerlo, lo convierte automáticamente a H.264.

import subprocess
from pathlib import Path

import cv2


# Devuelve True si OpenCV logra abrir el video y leer al menos un frame
def _se_puede_leer(ruta):
    cap = cv2.VideoCapture(str(ruta))
    ok, _ = cap.read()
    cap.release()
    return ok


# Convierte el video a H.264 (.mp4) con ffmpeg y devuelve la ruta nueva.
# Esto está porque los iPhone graban por defecto en HEVC (H.265), que OpenCV
# no siempre puede leer en Windows: nos pasó con nuestros propios videos.
def _convertir_a_h264(ruta, carpeta_salida):
    import imageio_ffmpeg   # solo se importa si hace falta convertir

    carpeta_salida.mkdir(parents=True, exist_ok=True)
    salida = carpeta_salida / f"{ruta.stem}_h264.mp4"

    # Si ya se convirtió en una corrida anterior, se reutiliza
    if salida.exists() and _se_puede_leer(salida):
        return salida

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([
        ffmpeg, "-y", "-loglevel", "error", "-i", str(ruta),
        "-c:v", "libx264", "-crf", "17", "-preset", "fast",
        "-pix_fmt", "yuv420p", "-an",
        str(salida)
    ], check=True)
    return salida


# Verifica que el video exista y se pueda leer; si no se puede, lo convierte.
# Devuelve la ruta del video a usar y un diccionario con su información.
def preparar_video(ruta, carpeta_salida):
    ruta = Path(ruta)
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró el video: {ruta}")

    if not _se_puede_leer(ruta):
        print(f"OpenCV no puede leer {ruta.name}; convirtiendo a H.264...")
        ruta = _convertir_a_h264(ruta, carpeta_salida)
        if not _se_puede_leer(ruta):
            raise RuntimeError(f"No se pudo leer el video ni después de convertirlo: {ruta}")

    cap = cv2.VideoCapture(str(ruta))
    fps = cap.get(cv2.CAP_PROP_FPS)
    info = {
        "ruta": ruta,
        "fps": fps if fps > 0 else 30.0,   # algunos videos no informan los FPS
        "n_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "ancho": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "alto": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    info["duracion_s"] = info["n_frames"] / info["fps"]
    return ruta, info


# Recorre el video frame por frame, en orden.
# En cada paso devuelve (número de frame, frame en RGB).
# Leemos en orden en lugar de saltar con cap.set porque, con algunos videos
# de celular, saltar a un frame específico no funcionaba bien.
def leer_frames(ruta):
    cap = cv2.VideoCapture(str(ruta))
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield idx, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        idx += 1
    cap.release()