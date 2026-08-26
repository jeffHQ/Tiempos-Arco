# Simulador de Horarios del Corredor Azul

Este proyecto arma, para una semana cualquiera, el horario completo de las 4 líneas troncales del Corredor Azul (305, 303, 336 y 301): a qué hora sale cada bus, cuánto tarda en cada tramo de su ruta, y a qué hora llega. Lo hace con un modelo de Machine Learning entrenado con datos históricos reales, así que no hace falta medir nada a mano para tener un horario razonable.

## Qué hace

1. Tú le indicas **qué semana** quieres simular (por ejemplo, "la semana 36 de 2026").
2. El modelo predice, tramo por tramo (llamamos "arco" a cada tramo entre dos paraderos), cuánto va a tardar el bus en recorrerlo — a esa hora, ese día de la semana, en esa línea.
3. Con esos tiempos arma un horario completo para los 7 días, separando **día hábil**, **sábado** y **domingo**, y ambos sentidos (**ida** y **vuelta**).
4. Te entrega ese horario en un Excel por línea, listo para revisar.

El modelo no es perfecto — es una predicción estadística, no una medición. Este proyecto incluye una forma de chequear qué tan bien predijo: si la semana que simulaste ya pasó y tienes el registro real de esa semana, un segundo script te arma gráficos comparando lo simulado contra lo que realmente ocurrió (ver [Cómo revisar qué tan acertada fue la predicción](#cómo-revisar-qué-tan-acertada-fue-la-predicción)).

## Antes de empezar

Necesitas tener **Python** instalado. Todo lo demás — el modelo ya entrenado, las rutas de cada línea, etc. — ya viene incluido en este repositorio; no tienes que conseguir ni configurar nada más.

Instala las librerías necesarias una sola vez:

```bash
pip install -r requirements.txt
```

## Cómo correrlo

Son dos pasos, siempre en este orden y desde la carpeta raíz del proyecto:

```bash
# 1. Simula la semana configurada para las 4 líneas
python scripts/simulacion_horarios.py

# 2. Genera un notebook con gráficos de los resultados
python scripts/generar_analisis.py
```

El primer paso es el que hace el trabajo pesado: simula cerca de 17,000 viajes por línea, así que tardará varios minutos en total para las 4 líneas. Vas a ver una barra de progreso y mensajes en pantalla mientras corre. El segundo paso es rápido — arma y ejecuta el notebook de gráficos en menos de un minuto.

Cuando termina el primer paso, encuentras los resultados en `resultados/semana_{N}/` (siendo `N` el número de semana que configuraste). Cuando termina el segundo, el notebook queda en `analisis/analisis_semana_{N}.ipynb`.

## Cómo leer los resultados

Adentro de `resultados/semana_{N}/` hay un Excel por línea, `simulacion_{LINEA}_semana_{N}.xlsx` (por ejemplo `simulacion_305_semana_36.xlsx`), y cada uno trae varias hojas:

- **`prog_dia_habil`**: el horario recomendado para un día de semana típico (lunes a viernes) — una fila por salida, con hora de salida y de llegada. Es la hoja que probablemente más uses: para armarla, el modelo no promedia los tiempos de un día de semana cualquiera, sino que toma el peor caso razonable (percentil 99) de cada tramo y hora, para que el horario resultante sea realista incluso en un mal día de tráfico.
- **`prog_sabado`** y **`prog_domingo`**: lo mismo, pero para sábado y domingo (acá sí se usa la predicción directa del modelo, sin ese ajuste al peor caso, porque el tráfico de fin de semana es más parejo).
- **`det_dia_habil`**, **`det_sabado`**, **`det_domingo`**: el mismo horario, pero desglosado tramo por tramo en vez de viaje completo — útil si necesitas el detalle de cuánto tarda cada arco específico.
- **`resumen_viajes`** y **`detallado_viajes`**: todos los viajes simulados de la semana (los 7 días, sin agrupar), a nivel de viaje completo y a nivel de tramo respectivamente. Sirven como la data cruda detrás de las hojas anteriores.

Si la semana que simulaste incluye un feriado, vas a ver hojas extra con el prefijo `prog_feriado_`.

### Cómo revisar qué tan acertada fue la predicción

El notebook `analisis/analisis_semana_{N}.ipynb` trae un gráfico por línea: el tiempo de viaje promedio por hora del día, comparando la curva **simulada** contra la curva **real** de esa misma semana (separado por ida y vuelta). Para que aparezca la curva real, necesitas poner el registro operativo real de esa semana en `analisis/Semana {N}.csv` antes de correr `generar_analisis.py` — si no lo tienes (por ejemplo, porque estás simulando una semana que todavía no pasó), el notebook se genera igual, solo que sin esa curva de comparación.

## Cómo cambiar qué semana simular

Al principio de `scripts/simulacion_horarios.py` hay un bloque de configuración. Las líneas que normalmente vas a querer tocar son:

```python
ANIO = 2026
NUM_SEMANA = 36     # numero de semana ISO del año (1 a 53)
```

`NUM_SEMANA` es el número de semana del año siguiendo el estándar ISO 8601 (la semana 1 es la que contiene el primer jueves de enero). Si no estás seguro de qué número le corresponde a una fecha, cualquier buscador te lo dice con "qué semana ISO es el [fecha]".

Un poco más abajo también puedes ajustar cada cuántos minutos sale un bus (`INTERVALO_MINUTOS`) y el rango horario que se simula (`HORA_INICIO`, `HORA_FIN`), aunque normalmente no hace falta tocarlos.

Después de cambiar la semana, vuelve a correr los dos pasos de [Cómo correrlo](#cómo-correrlo) — cada corrida sobreescribe los archivos de esa misma semana si ya existían, así que no hay problema en simular la misma semana varias veces.

> Hay otras opciones de configuración más técnicas (qué líneas simular, qué versión del modelo usar) que no cubrimos acá porque tocarlas requiere entender cómo está armado el modelo por dentro. Si necesitas cambiar eso, es tema para alguien del equipo técnico — el detalle está documentado en `CLAUDE.md`.

## Si algo falla

**"Arco '...' no encontrado en lookup tables" (o similar)**
El modelo no tiene datos históricos para un tramo de la ruta. Puede ser que el archivo de esa línea en la carpeta `rutas/` esté desactualizado o mal armado. Avísale a alguien del equipo técnico con el mensaje de error completo.

**"Modelo no encontrado" o "Lookup table no encontrada"**
Faltan archivos dentro de la carpeta `modelos/`. Si descargaste el repositorio de nuevo o cambiaste de carpeta, revisa que la carpeta `modelos/` se haya copiado completa junto con el código.

**"Archivo de ruta no encontrado"**
Falta el archivo de una línea dentro de la carpeta `rutas/` (tiene que existir `linea_{numero}_1.csv` para ida y `linea_{numero}_2.csv` para vuelta).

**El script tarda muchísimo o parece colgado**
Es normal que tarde varios minutos — simula miles de viajes por línea. Si lleva más de 15-20 minutos sin avanzar, revisa el archivo `simulacion.log` (se genera en la misma carpeta donde corriste el script) para ver en qué se quedó, y avísale al equipo técnico.

## Estructura del proyecto

```
3_simulador_tiempos_arco/
├── scripts/
│   ├── simulacion_horarios.py   # Simula la semana configurada
│   └── generar_analisis.py      # Genera el notebook de graficos
├── modelos/                      # Modelo entrenado y sus tablas de apoyo (no se toca a mano)
├── rutas/                        # Un CSV por linea y sentido, con la secuencia de paraderos
├── resultados/                   # Se genera al correr la simulacion (un Excel por linea, por semana)
├── analisis/                     # CSV de datos reales (opcional) + notebook de graficos generado
├── requirements.txt              # Librerias necesarias
├── simulacion.log                # Se genera al correr la simulacion
└── README.md                     # Este archivo
```

Los contenidos de `resultados/` y `analisis/*.ipynb` no quedan guardados en el control de versiones (se regeneran cada vez que corres los scripts).
