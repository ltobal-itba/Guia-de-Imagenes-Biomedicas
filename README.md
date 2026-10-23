# Guía de Imágenes Biomédicas 
Pipeline de procesamiento de video para medir la respuesta pupilar ante un estímulo luminoso.

## Instalación
1. Clonar el repositorio.
2. Instalar dependencias:
   pip install opencv-python numpy matplotlib scikit-image ipykernel
3. Crear la carpeta `data/` en la raíz y colocar ahí los videos
   (no se suben a GitHub).

## Instalación

Se requiere Python 3.10 o superior.

Desde una terminal ubicada en la carpeta del proyecto, instalar las dependencias:

```bash
python -m pip install numpy pandas scipy opencv-python scikit-image matplotlib
```

## Ejecución

El programa permite analizar un video individual o todos los videos de una carpeta, sin modificar el código.

### Analizar los videos de la carpeta predeterminada

Colocar los videos dentro de `data/` y ejecutar:

```bash
python main.py
```

### Analizar un video específico

```bash
python main.py "data/A1.mp4"
```

### Analizar una carpeta diferente

```bash
python main.py "C:\Videos de la catedra"
```

Las rutas que contienen espacios deben escribirse entre comillas. La búsqueda en carpetas incluye archivos MP4, MOV, AVI, MKV y M4V, sin recorrer subcarpetas. Su lectura depende de los códecs disponibles en OpenCV.

### Consultar la ayuda

```bash
python main.py --help
```

## Procesamiento

Para cada video, el programa:

1. Localiza automáticamente el ojo.
2. Segmenta la pupila y actualiza la posición de la región de interés.
3. Obtiene el diámetro estimado y el área a partir de un ajuste circular.
4. Marca mediciones inválidas, interpola y suaviza la señal.
5. Estima el momento del cambio de iluminación a partir del brillo promedio.
6. Calcula los parámetros de la respuesta pupilar.
7. Guarda las curvas, los parámetros y un gráfico.

La clasificación automática considera:
- **Tipo A:** aumento de iluminación.
- **Tipo B:** disminución de iluminación.

## Resultados

Las salidas se guardan en `resultados/`, en una carpeta por video. Por ejemplo:

```text
resultados/
└── A1.mp4/
    ├── A1_curva.csv
    ├── A1_parametros.csv
    └── A1_curva.png
```

- **`*_curva.csv`:** mediciones por frame, tiempo, brillo, indicador de validez y señal limpia.
- **`*_parametros.csv`:** basal, extremo, amplitud, latencia, velocidades y estadísticas del registro.
- **`*_curva.png`:** gráfico de la respuesta con el estímulo estimado y las mediciones descartadas.

Los diámetros se expresan en píxeles, las áreas en píxeles cuadrados y los tiempos en segundos. Las medidas no representan milímetros sin una calibración espacial.

La ejecución guarda los gráficos sin abrir ventanas. Si ocurre un error en un video, se informa en la terminal y se continúa con los restantes.

## Limitaciones actuales

El programa está pensado para videos del reflejo fotomotor con el ojo claramente visible. Los resultados deben revisarse mediante los gráficos generados.

La detección del estímulo selecciona el mayor salto de brillo, por lo que puede confundirse ante movimientos, cambios de exposición o varios cambios de iluminación. La ausencia de un estímulo real todavía no se identifica de forma robusta.

La región de interés mantiene un tamaño proporcional al ancho del video. La conversión automática de videos incompatibles y la recuperación global del seguimiento todavía no están integradas.