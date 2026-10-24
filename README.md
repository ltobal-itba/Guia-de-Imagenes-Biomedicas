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

## Validación de la segmentación pupilar

### Metodología

Para evaluar la precisión del algoritmo de segmentación pupilar, se realizó una validación cuantitativa preliminar mediante la comparación de las mediciones automáticas con mediciones manuales realizadas en **Fiji (ImageJ)**.

Se seleccionó un fotograma representativo de cada uno de los seis videos analizados (A1, A2, B1, B2, B3 y B4).

En cada imagen se delimitó manualmente la región pupilar utilizando una selección elíptica y se registró su área en píxeles cuadrados.

A partir del área medida, se calculó el diámetro equivalente suponiendo una geometría circular:

$$
D = 2\sqrt{\frac{A}{\pi}}
$$

Donde:
- $D$: diámetro pupilar equivalente, en píxeles.
- $A$: área pupilar medida manualmente, en píxeles cuadrados.

Los diámetros manuales se compararon con los valores obtenidos por el algoritmo en los mismos fotogramas, utilizando la columna `diametro_px` de los archivos `*_curva.csv`, correspondiente a las mediciones anteriores al suavizado temporal.

### Resultados de la validación

| Video | Frame | Área manual (px²) | Diámetro manual (px) | Diámetro automático (px) | Error absoluto (px) | Error relativo (%) |
|-------|------:|------------------:|--------------------:|------------------------:|--------------------:|-------------------:|
| A1 | 742 | 3220 | 64,03 | 63,97 | 0,06 | 0,10 |
| A2 | 772 | 2299 | 54,10 | 47,52 | 6,58 | 12,16 |
| B1 | 506 | 4128 | 72,50 | 65,19 | 7,30 | 10,07 |
| B2 | 568 | 5417 | 83,05 | 80,48 | 2,57 | 3,09 |
| B3 | 651 | 6948 | 94,06 | 89,49 | 4,57 | 4,86 |
| B4 | 615 | 4724 | 77,55 | 78,30 | 0,75 | 0,97 |

Para cuantificar las diferencias entre ambos métodos, se calcularon las siguientes métricas:

**Error absoluto:**

$$
E_a = |D_{manual} - D_{automatico}|
$$

**Error relativo porcentual:**

$$
E_r = \frac{|D_{manual} - D_{automatico}|}{D_{manual}}\times100
$$

### Métricas globales

| Métrica | Resultado |
|---------|----------:|
| Error absoluto medio (MAE) | 3,64 px |
| Error porcentual absoluto medio (MAPE) | 5,21 % |
| Raíz del error cuadrático medio (RMSE) | 4,56 px |
| Sesgo medio (automático − manual) | −3,39 px |

### Interpretación de los resultados

Los resultados obtenidos muestran una concordancia razonable entre las mediciones manuales realizadas en Fiji y los diámetros estimados automáticamente por el algoritmo, con un **error porcentual absoluto medio del 5,21 %**.

Los videos A1 y B4 presentaron los menores errores relativos, inferiores al 1 %, mientras que A2 y B1 registraron las mayores discrepancias, con errores del 12,16 % y 10,07 %, respectivamente.

Se observó que **las imágenes con mayores errores correspondían a fotogramas que presentaban artefactos visuales**, los cuales dificultaban la identificación precisa del contorno pupilar. Estos artefactos afectaban la calidad de la imagen y complicaban tanto la delimitación manual como la segmentación automática.

Esta observación sugiere que la precisión del algoritmo está condicionada, en parte, por la calidad de las imágenes de entrada, especialmente cuando existen interferencias visuales que reducen la definición del borde pupilar.

Adicionalmente, en cinco de los seis fotogramas evaluados, el algoritmo estimó diámetros inferiores a los obtenidos manualmente. Esto se refleja en un sesgo medio de −3,39 px, lo que podría indicar una tendencia a subestimar ligeramente el tamaño de la pupila.

### Limitaciones de la validación

Esta evaluación constituye una **validación cuantitativa preliminar**, dado que se analizó únicamente un fotograma por video.

Por lo tanto, los resultados permiten evaluar la concordancia puntual entre las mediciones manuales y automáticas, pero no caracterizan completamente el desempeño del algoritmo durante toda la secuencia temporal.

Además, los videos utilizados para la validación también fueron empleados durante el desarrollo y ajuste del algoritmo, por lo que los resultados corresponden a una evaluación interna y no a una validación sobre datos independientes.

