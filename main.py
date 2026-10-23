import argparse
from pathlib import Path

from config import DATA_DIR, CARPETA_SALIDA
from src.pipeline import analizar_video


def main():
    parser = argparse.ArgumentParser(
        description="Analiza el reflejo pupilar de uno o varios videos."
    )
    parser.add_argument(
        "entrada",
        nargs="?",
        type=Path,
        default=DATA_DIR,
        help="Ruta de un video o una carpeta. Por defecto: data/",
    )
    args = parser.parse_args()

    extensiones = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}

    if args.entrada.is_file():
        videos = [args.entrada]
    elif args.entrada.is_dir():
        videos = sorted(
            p for p in args.entrada.iterdir()
            if p.is_file() and p.suffix.lower() in extensiones
        )
    else:
        parser.error(f"No existe la ruta: {args.entrada}")

    if not videos:
        parser.error("No se encontraron videos en la carpeta.")

    for video in videos:
        try:
            # Una carpeta por archivo, para separar los resultados
            salida = CARPETA_SALIDA / video.name
            _, parametros = analizar_video(
                video,
                salida,
                tipo="auto",
            )

            print(
                f"{video.name}: "
                f"tipo {parametros['tipo_registro']}, "
                f"estímulo {parametros['t_estimulo_s']:.3f} s, "
                f"latencia {parametros['latencia_s']:.3f} s"
            )

        except Exception as error:
            print(f"ERROR en {video.name}: {error}")


if __name__ == "__main__":
    main()