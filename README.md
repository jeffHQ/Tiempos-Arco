# Simulador de Tiempos de Arco para Transporte Público

Este proyecto simula tiempos de viaje para una semana completa de líneas de transporte público usando un modelo de Machine Learning entrenado y lookup tables livianas optimizadas.

## Descripción

El script `simulacion_horarios.py` genera simulaciones completas de tiempos de viaje para una línea específica durante una semana completa, considerando:
- **Semana completa**: Simula los 7 días de una semana específica (Lunes a Domingo)
- **Tipos de día**: Hábil (Lunes-Viernes), Sábado y Domingo
- **Horarios**: Desde 4:00 AM hasta 23:59 PM con intervalo configurable (por defecto: 1 minuto)
- **Ambos sentidos**: Ida y vuelta de la línea
- **Lookup tables livianas**: Usa archivos Parquet optimizados en lugar de cargar el dataset histórico completo
- **Predicciones arco por arco**: Simula cada arco de la ruta usando el modelo de ML
- **Franjas horarias**: Clasifica los tiempos según franjas (PicoManana, ValleManana, Mediodia, ValleTarde, PicoTarde, Noche)

## Requisitos

- Python 3.7+
- Librerías: pandas, numpy, tqdm, pickle, pyarrow (para Parquet)
- Modelo entrenado: `modelos/mejor_modelo.pkl`
- Lookup tables: 
  - `modelos/arco_stats.parquet`
  - `modelos/arco_franja_stats.parquet`
  - `modelos/arco_dia_stats.parquet`
- Feriados: `modelos/feriados.pkl`
- Rutas: Archivos CSV en `rutas/` con formato `linea_{numero}_{sentido}.csv`
  - `sentido = 1` para ida
  - `sentido = 2` para vuelta

## Instalación

1. Clonar o descargar el repositorio
2. Instalar las dependencias:
   ```bash
   pip install pandas numpy tqdm pyarrow
   ```
3. Asegurarse de que los archivos de modelos y rutas estén en las carpetas correspondientes

## Uso

### Ejecución básica

```bash
python simulacion_horarios.py
```

El script simula automáticamente la línea configurada (por defecto línea 305) para la semana especificada.

### Parámetros configurables

Edita las constantes al inicio del script (`simulacion_horarios.py`) para ajustar:

```python
LINEA = 305                         # Número de línea a simular
ANIO = 2026                         # Año de la simulación
MES = 2                             # Mes (1-12)
NUM_SEMANA = 4                      # Número de semana del mes (1-5)
INTERVALO_MINUTOS = 1               # Intervalo entre salidas (minutos)
HORA_INICIO = 4                     # Hora de inicio (04:00)
HORA_FIN = 24                       # Hora de fin (incluye 00:00)
```

**Nota**: El script busca automáticamente los archivos de ruta en `rutas/` con el formato:
- `linea_{LINEA}_1.csv` (ida, sentido = 1)
- `linea_{LINEA}_2.csv` (vuelta, sentido = 2)

Cada archivo debe contener una columna `arco_id` con los identificadores de los arcos en orden.

## Outputs Generados

El script genera los siguientes archivos en la carpeta `resultados/`:

### Archivos Generales

#### 1. `resumen_viajes.csv`
Resumen de todos los viajes simulados (todos los días y horarios):

| Columna | Descripción |
|---------|-------------|
| `fecha` | Fecha del viaje (YYYY-MM-DD) |
| `dia` | Nombre del día (Lunes, Martes, etc.) |
| `tipo_dia` | Tipo de día (habil, sabado, domingo) |
| `hora_salida` | Hora de salida (HH:MM) |
| `hora_llegada` | Hora de llegada estimada (HH:MM) |
| `direccion` | Sentido del viaje (ida, vuelta) |
| `tiempo_total_seg` | Tiempo total del viaje en segundos |
| `tiempo_total_min` | Tiempo total del viaje en minutos |

#### 2. `detallado_viajes.csv`
Detalle arco por arco de todos los viajes simulados:

| Columna | Descripción |
|---------|-------------|
| `fecha` | Fecha del viaje |
| `dia` | Nombre del día |
| `tipo_dia` | Tipo de día |
| `hora_salida_viaje` | Hora de salida del viaje (HH:MM) |
| `direccion` | Sentido (ida, vuelta) |
| `arco_id` | Identificador del arco |
| `orden` | Orden del arco en la ruta |
| `hora_inicio` | Hora de inicio del arco (HH:MM) |
| `hora_llegada` | Hora de llegada del arco (HH:MM) |
| `tiempo_arco_seg` | Tiempo del arco en segundos |
| `tiempo_arco_min` | Tiempo del arco en minutos |

### Archivos por Tipo de Día

#### 3. `programacion_dia_habil.csv` y `detallado_dia_habil.csv`
Programación y detalle para días hábiles (Lunes-Viernes):
- **Programación**: Resumen de viajes usando **Percentil 99 (P99)** por franja horaria
- **Detallado**: Desglose arco por arco con tiempos P99 por franja
- Incluye columna `franja` (PicoManana, ValleManana, Mediodia, ValleTarde, PicoTarde, Noche)

#### 4. `programacion_sabado.csv` y `detallado_sabado.csv`
Programación y detalle para sábados:
- Predicciones directas del modelo (sin agregación P99)

#### 5. `programacion_domingo.csv` y `detallado_domingo.csv`
Programación y detalle para domingos:
- Predicciones directas del modelo (sin agregación P99)

#### 6. Archivos de Feriados (condicional)
Si la semana incluye feriados, se generan archivos adicionales:
- `programacion_feriado_{fecha}.csv`
- `detallado_feriado_{fecha}.csv`

### Notebook de Exploración

El proyecto incluye `exploracion_resultados.ipynb` para analizar y visualizar los resultados:
- Gráficos de tiempos por hora del día
- Comparación entre diferentes días
- Análisis por tipo de día (hábil, sábado, domingo)
- Estadísticas descriptivas

## Ejemplo de Uso

```bash
# 1. (Opcional) Editar parámetros en simulacion_horarios.py:
#    - LINEA = 305
#    - ANIO = 2026
#    - MES = 2
#    - NUM_SEMANA = 4

# 2. Ejecutar el script
python simulacion_horarios.py

# 3. Esperar a que complete (puede tardar varios minutos)
# El script mostrará una barra de progreso con el avance

# 4. Revisar resultados en la carpeta resultados/:
#    - resumen_viajes.csv
#    - detallado_viajes.csv
#    - programacion_dia_habil.csv
#    - detallado_dia_habil.csv
#    - programacion_sabado.csv
#    - detallado_sabado.csv
#    - programacion_domingo.csv
#    - detallado_domingo.csv

# 5. Explorar resultados con el notebook:
#    jupyter notebook exploracion_resultados.ipynb
```

## Logging

El script genera un archivo `simulacion.log` con información detallada del proceso:
- Carga de modelo y datos
- Validación de arcos
- Errores y advertencias
- Resumen final

## Notas Importantes

1. **Lookup Tables Livianas**: El script usa archivos Parquet optimizados en lugar de cargar el dataset histórico completo, lo que reduce significativamente el uso de memoria y mejora el rendimiento.

2. **Validación de arcos**: El script valida que todos los arcos de la ruta existan en las lookup tables antes de comenzar la simulación.

3. **Pre-computación**: Las features estáticas de cada arco se pre-computan una vez al inicio para optimizar el rendimiento.

4. **Tiempo de ejecución**: Depende del intervalo configurado y la cantidad de arcos. Para intervalo de 1 minuto (1200+ horarios) y ~40 arcos por sentido, puede tardar 30-60 minutos.

5. **Día Hábil con P99**: Los días hábiles usan Percentil 99 por franja horaria para generar programaciones más conservadoras y robustas.

6. **Fechas de simulación**: Las fechas se calculan automáticamente según la semana especificada del mes. Los feriados se detectan automáticamente y se generan archivos adicionales.

## Solución de Problemas

### Error: "Arco '{arco}' no encontrado en lookup tables"
- Verifica que los archivos de ruta existan: `rutas/linea_{LINEA}_1.csv` y `rutas/linea_{LINEA}_2.csv`
- Verifica que los `arco_id` en los CSVs coincidan exactamente con los de las lookup tables
- Revisa que los archivos Parquet de lookup tables estén actualizados

### Error: "Modelo no encontrado" o "Lookup table no encontrada"
- Verifica que existan los archivos:
  - `modelos/mejor_modelo.pkl`
  - `modelos/arco_stats.parquet`
  - `modelos/arco_franja_stats.parquet`
  - `modelos/arco_dia_stats.parquet`
  - `modelos/feriados.pkl`

### Error: "Archivo de ruta no encontrado"
- Verifica que existan los archivos en `rutas/`:
  - `linea_{LINEA}_1.csv` (ida, sentido = 1)
  - `linea_{LINEA}_2.csv` (vuelta, sentido = 2)
- Los archivos deben tener una columna `arco_id` con los identificadores en orden

### Predicciones anómalas
- El script registra warnings en el log si una predicción es negativa o muy alta
- Revisa el archivo `simulacion.log` para detalles
- Verifica que las lookup tables estén actualizadas con datos recientes

## Estructura del Proyecto

```
simulador_tiempos_arco/
├── simulacion_horarios.py       # Script principal de simulación
├── exploracion_resultados.ipynb # Notebook para analizar resultados
├── README.md                     # Este archivo
├── .gitignore                    # Archivos excluidos de Git
├── modelos/                      # Modelos y lookup tables
│   ├── mejor_modelo.pkl
│   ├── arco_stats.parquet
│   ├── arco_franja_stats.parquet
│   ├── arco_dia_stats.parquet
│   ├── feriados.pkl
│   └── features_list.txt
├── rutas/                        # Archivos CSV de rutas
│   ├── linea_301_1.csv
│   ├── linea_301_2.csv
│   ├── linea_303_1.csv
│   ├── linea_303_2.csv
│   ├── linea_305_1.csv
│   └── linea_305_2.csv
├── resultados/                   # Archivos CSV generados (no versionados)
│   ├── resumen_viajes.csv
│   ├── detallado_viajes.csv
│   ├── programacion_dia_habil.csv
│   ├── detallado_dia_habil.csv
│   ├── programacion_sabado.csv
│   ├── detallado_sabado.csv
│   ├── programacion_domingo.csv
│   └── detallado_domingo.csv
└── simulacion.log                # Log de ejecución (no versionado)
```

**Nota**: Los archivos en `modelos/` (`.pkl`, `.parquet`) y `resultados/` (`.csv`, `.xlsx`) están excluidos del control de versiones por su tamaño. Solo se versionan los archivos de código y configuración.

## Control de Versiones

Este proyecto utiliza Git para el control de versiones. Los siguientes tipos de archivos están excluidos del repositorio (ver `.gitignore`):
- Modelos entrenados (`.pkl`, `.parquet`)
- Resultados generados (`.csv`, `.xlsx`)
- Logs (`.log`)
- Archivos temporales y de sistema

## Franjas Horarias

El sistema clasifica los tiempos según las siguientes franjas horarias:
- **PicoManana**: 6:00 - 9:00
- **ValleManana**: 9:00 - 12:00
- **Mediodia**: 12:00 - 14:00
- **ValleTarde**: 14:00 - 17:00
- **PicoTarde**: 17:00 - 21:00
- **Noche**: 21:00 - 6:00

**Importante**: La franja se determina según la **hora de inicio** del arco, no la hora de llegada. Esto significa que un arco que comienza a las 20:54 se clasifica como "PicoTarde" aunque termine después de las 21:00.

## Contacto

Para preguntas o problemas, revisa el log de simulación (`simulacion.log`) o los mensajes de error en consola.

