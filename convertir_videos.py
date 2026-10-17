# Iphone graba en un formato que no es compatible con OpenCV, así que 
# se puede correr este código primero para convertirlos a mp4.
   
import subprocess
import imageio_ffmpeg
from config import DATA_DIR

ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

for video in sorted(DATA_DIR.glob("*.MOV")):
    salida = video.with_suffix(".mp4")
    print(f"Convirtiendo {video.name} → {salida.name}")
    subprocess.run([
        ffmpeg, "-y", "-i", str(video),
        "-c:v", "libx264", "-crf", "17", "-preset", "slow",
        "-pix_fmt", "yuv420p", "-an",
        str(salida)
    ], check=True)

print("Listo")