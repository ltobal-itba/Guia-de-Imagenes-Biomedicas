from pathlib import Path

ROOT = Path(__file__).parent

# Carpeta donde están todos los videos (para analizar varios registros juntos)
DATA_DIR = ROOT / "data"

# Video a procesar (CAMBIAR ACÁ)
VIDEO_PATH = DATA_DIR / "A3.mp4"

# Carpeta donde el pipeline guarda los resultados (CSV, gráficos)
OUTPUT_DIR = ROOT / "outputs"