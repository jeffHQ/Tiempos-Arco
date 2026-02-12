"""
Script de Simulacion de Horarios de Transporte
==============================================

Simula tiempos de viaje para una semana completa de la linea 305
usando lookup tables livianas (sin cargar df_modelado).

Genera 8+ archivos de salida con programaciones y detallados.
"""

import pandas as pd
import numpy as np
import pickle
import logging
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACION DE PARAMETROS
# ============================================================================

LINEA = 305
ANIO = 2026
MES = 2
NUM_SEMANA = 4              # Semana 4 de febrero = Feb 23 - Mar 1
INTERVALO_MINUTOS = 1       # 1 o 5 minutos entre salidas
HORA_INICIO = 4             # 04:00
HORA_FIN = 24               # Incluye 00:00 (medianoche)

# Rutas de archivos
RUTA_MODELO = 'modelos/mejor_modelo.pkl'
RUTA_ARCO_STATS = 'modelos/arco_stats.parquet'
RUTA_ARCO_FRANJA = 'modelos/arco_franja_stats.parquet'
RUTA_ARCO_DIA = 'modelos/arco_dia_stats.parquet'
RUTA_FERIADOS = 'modelos/feriados.pkl'
RUTA_RUTAS = 'rutas/'
CARPETA_SALIDA = 'resultados/'

# ============================================================================
# CONFIGURACION DE LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('simulacion.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# FUNCIONES DE CARGA
# ============================================================================

def cargar_artefactos():
    """Carga modelo, features y lookup tables."""
    logger.info("Cargando artefactos...")

    with open(RUTA_MODELO, 'rb') as f:
        modelo_info = pickle.load(f)
    modelo = modelo_info['modelo']
    features = modelo_info['features']
    logger.info(f"Modelo: {modelo_info['nombre']}, {len(features)} features")

    arco_stats = pd.read_parquet(RUTA_ARCO_STATS)
    # arco_stats ya tiene arco_id como index
    logger.info(f"arco_stats: {len(arco_stats)} arcos")

    arco_franja_stats = pd.read_parquet(RUTA_ARCO_FRANJA).set_index(['arco_id', 'franja_horaria'])
    logger.info(f"arco_franja_stats: {len(arco_franja_stats)} combinaciones")

    arco_dia_stats = pd.read_parquet(RUTA_ARCO_DIA).set_index(['arco_id', 'DiaSemana_num'])
    logger.info(f"arco_dia_stats: {len(arco_dia_stats)} combinaciones")

    with open(RUTA_FERIADOS, 'rb') as f:
        feriados = pickle.load(f)
    logger.info(f"Feriados: {len(feriados)} fechas")

    return modelo, features, arco_stats, arco_franja_stats, arco_dia_stats, feriados


def cargar_ruta(linea, sentido):
    """Carga la secuencia ordenada de arco_ids para una linea y sentido."""
    archivo = os.path.join(RUTA_RUTAS, f'linea_{linea}_{sentido}.csv')
    df = pd.read_csv(archivo)
    arcos = df['arco_id'].tolist()
    logger.info(f"Ruta linea {linea} sentido {sentido}: {len(arcos)} arcos")
    return arcos


# ============================================================================
# FUNCIONES DE CALCULO
# ============================================================================

def calcular_fechas_semana(anio, mes, num_semana):
    """
    Calcula las 7 fechas de la semana N del mes M.

    Returns:
        dict {date: tipo_dia} donde tipo_dia es 'habil', 'sabado', 'domingo' o 'feriado'
    """
    # Encontrar el primer lunes del mes (en o despues del dia 1)
    primer_dia = date(anio, mes, 1)
    dia_semana_primero = primer_dia.weekday()  # 0=Lun, 6=Dom
    if dia_semana_primero == 0:
        primer_lunes = primer_dia
    else:
        primer_lunes = primer_dia + timedelta(days=(7 - dia_semana_primero))
    lunes_semana_n = primer_lunes + timedelta(weeks=num_semana - 1)

    fechas = {}
    for i in range(7):
        fecha = lunes_semana_n + timedelta(days=i)
        dia_semana = fecha.weekday()
        if dia_semana == 5:
            tipo = 'sabado'
        elif dia_semana == 6:
            tipo = 'domingo'
        else:
            tipo = 'habil'
        fechas[fecha] = tipo

    return fechas


def clasificar_franja(hora_decimal):
    """Clasifica la hora decimal en una franja horaria."""
    if 6 <= hora_decimal < 9:
        return 'PicoManana'
    elif 9 <= hora_decimal < 12:
        return 'ValleManana'
    elif 12 <= hora_decimal < 14:
        return 'Mediodia'
    elif 14 <= hora_decimal < 17:
        return 'ValleTarde'
    elif 17 <= hora_decimal < 21:
        return 'PicoTarde'
    else:
        return 'Noche'


def obtener_temporada(mes):
    """Obtiene la temporada segun el mes."""
    if mes in [1, 2]:
        return 'T1_Ene-Feb'
    elif mes in [3, 4, 5, 6, 7]:
        return 'T2_Mar-Jul'
    elif mes in [8, 9, 10, 11]:
        return 'T3_Ago-Nov'
    else:
        return 'T4_Dic'


def crear_features_arco(arco_id, fecha, hora_decimal, sentido, linea,
                        posicion, total_arcos, retraso_acumulado,
                        arco_stats, arco_franja_stats, arco_dia_stats, feriados):
    """
    Construye un dict con las 40 features usando SOLO lookup tables.

    Args:
        arco_id: Identificador normalizado del arco
        fecha: date object
        hora_decimal: Hora en formato decimal (ej: 8.5 = 08:30)
        sentido: 1 (ida) o 2 (vuelta)
        linea: Numero de linea
        posicion: Posicion del arco en el viaje (1-based)
        total_arcos: Total de arcos en el viaje
        retraso_acumulado: Retraso acumulado en segundos
        arco_stats, arco_franja_stats, arco_dia_stats, feriados: lookup tables

    Returns:
        dict con las 40 features
    """
    stats = arco_stats.loc[arco_id]
    dia_semana = fecha.weekday()
    hora = int(hora_decimal)
    minuto = int((hora_decimal % 1) * 60)
    mes = fecha.month

    franja = clasificar_franja(hora_decimal)
    temporada = obtener_temporada(mes)

    # Historicos desde lookup
    tiempo_hist = float(stats['tiempo_historico_arco'])

    try:
        tiempo_hist_franja = float(
            arco_franja_stats.loc[(arco_id, franja), 'tiempo_historico_arco_franja']
        )
    except KeyError:
        tiempo_hist_franja = tiempo_hist

    try:
        tiempo_hist_dia = float(
            arco_dia_stats.loc[(arco_id, dia_semana), 'tiempo_historico_arco_dia']
        )
    except KeyError:
        tiempo_hist_dia = tiempo_hist

    return {
        # Temporales
        'Hora_Partida': hora_decimal,
        'Minuto_Partida': float(minuto),
        'hora_sin': np.sin(2 * np.pi * hora_decimal / 24),
        'hora_cos': np.cos(2 * np.pi * hora_decimal / 24),
        'DiaSemana_num': dia_semana,
        'dia_mes': fecha.day,
        'mes': mes,
        'es_fin_semana': int(dia_semana >= 5),
        'es_hora_pico': int((6 <= hora_decimal <= 9) or (17 <= hora_decimal <= 20)),
        'es_inicio_mes': int(fecha.day <= 5),
        'es_fin_mes': int(fecha.day >= 25),
        'es_feriado': int(fecha in feriados),
        # Linea y sentido
        'linea': linea,
        'sentido': sentido,
        # Encoding
        'arco_id_encoded': int(stats['arco_id_encoded']),
        'parada_desde_encoded': int(stats['parada_desde_encoded']),
        'parada_hacia_encoded': int(stats['parada_hacia_encoded']),
        # Historicos
        'tiempo_historico_arco': tiempo_hist,
        'tiempo_historico_arco_franja': tiempo_hist_franja,
        'tiempo_historico_arco_dia': tiempo_hist_dia,
        'std_historico_arco': float(stats['std_historico_arco']),
        'percentil_25_arco': float(stats['percentil_25_arco']),
        'percentil_75_arco': float(stats['percentil_75_arco']),
        'percentil_90_arco': float(stats['percentil_90_arco']),
        # Secuenciales
        'posicion_arco_en_viaje': posicion,
        'total_arcos_viaje': total_arcos,
        'es_primer_arco': int(posicion == 1),
        'es_ultimo_arco': int(posicion == total_arcos),
        'progreso_viaje': float(posicion) / float(total_arcos),
        'retraso_acumulado': retraso_acumulado,
        # Franjas one-hot
        'franja_Noche': int(franja == 'Noche'),
        'franja_PicoManana': int(franja == 'PicoManana'),
        'franja_ValleManana': int(franja == 'ValleManana'),
        'franja_Mediodia': int(franja == 'Mediodia'),
        'franja_ValleTarde': int(franja == 'ValleTarde'),
        'franja_PicoTarde': int(franja == 'PicoTarde'),
        # Temporadas one-hot
        'temporada_T1_Ene-Feb': int(temporada == 'T1_Ene-Feb'),
        'temporada_T2_Mar-Jul': int(temporada == 'T2_Mar-Jul'),
        'temporada_T3_Ago-Nov': int(temporada == 'T3_Ago-Nov'),
        'temporada_T4_Dic': int(temporada == 'T4_Dic'),
    }


# ============================================================================
# FUNCIONES DE SIMULACION
# ============================================================================

def convertir_lookups_a_dicts(arco_franja_stats, arco_dia_stats):
    """Convierte lookup DataFrames indexados a dicts nativos para acceso rapido."""
    franja_dict = {}
    for (arco_id, franja), row in arco_franja_stats.iterrows():
        franja_dict[(arco_id, franja)] = float(row['tiempo_historico_arco_franja'])

    dia_dict = {}
    for (arco_id, dia), row in arco_dia_stats.iterrows():
        dia_dict[(arco_id, dia)] = float(row['tiempo_historico_arco_dia'])

    return franja_dict, dia_dict


def precomputar_arcos(arcos, sentido, linea, arco_stats, features):
    """
    Pre-computa features estaticas por arco (encoding, historicos, secuenciales).
    """
    total_arcos = len(arcos)
    feature_idx = {f: i for i, f in enumerate(features)}
    arcos_precomp = []

    for i, arco_id in enumerate(arcos):
        posicion = i + 1
        stats = arco_stats.loc[arco_id]

        # Template de valores: array con features estaticas ya puestas
        vals = np.zeros(len(features), dtype=np.float64)

        # Linea y sentido
        vals[feature_idx['linea']] = linea
        vals[feature_idx['sentido']] = sentido

        # Encoding
        vals[feature_idx['arco_id_encoded']] = int(stats['arco_id_encoded'])
        vals[feature_idx['parada_desde_encoded']] = int(stats['parada_desde_encoded'])
        vals[feature_idx['parada_hacia_encoded']] = int(stats['parada_hacia_encoded'])

        # Historicos que no dependen de hora/dia
        vals[feature_idx['tiempo_historico_arco']] = float(stats['tiempo_historico_arco'])
        vals[feature_idx['std_historico_arco']] = float(stats['std_historico_arco'])
        vals[feature_idx['percentil_25_arco']] = float(stats['percentil_25_arco'])
        vals[feature_idx['percentil_75_arco']] = float(stats['percentil_75_arco'])
        vals[feature_idx['percentil_90_arco']] = float(stats['percentil_90_arco'])

        # Secuenciales
        vals[feature_idx['posicion_arco_en_viaje']] = posicion
        vals[feature_idx['total_arcos_viaje']] = total_arcos
        vals[feature_idx['es_primer_arco']] = int(posicion == 1)
        vals[feature_idx['es_ultimo_arco']] = int(posicion == total_arcos)
        vals[feature_idx['progreso_viaje']] = float(posicion) / float(total_arcos)
        vals[feature_idx['retraso_acumulado']] = 0.0

        arcos_precomp.append({
            'arco_id': arco_id,
            'posicion': posicion,
            'template': vals.copy(),
            'tiempo_hist': float(stats['tiempo_historico_arco']),
        })

    return arcos_precomp, feature_idx


def simular_viaje(arcos_precomp, feature_idx, fecha, hora_inicio_decimal,
                  booster, franja_dict, dia_dict, feriados):
    """
    Simula un viaje completo usando arrays numpy pre-computados.
    """
    dia_semana = fecha.weekday()
    mes = fecha.month
    es_feriado = int(fecha in feriados)
    es_fin_semana = int(dia_semana >= 5)
    es_inicio_mes = int(fecha.day <= 5)
    es_fin_mes = int(fecha.day >= 25)

    # Temporada
    if mes in (1, 2):
        temporada = 'T1_Ene-Feb'
    elif mes in (3, 4, 5, 6, 7):
        temporada = 'T2_Mar-Jul'
    elif mes in (8, 9, 10, 11):
        temporada = 'T3_Ago-Nov'
    else:
        temporada = 'T4_Dic'

    fi = feature_idx
    TWO_PI_24 = 2 * np.pi / 24

    hora_actual = hora_inicio_decimal
    resultados = []

    for ap in arcos_precomp:
        vals = ap['template'].copy()
        arco_id = ap['arco_id']
        tiempo_hist = ap['tiempo_hist']

        hora_decimal = hora_actual
        minuto = int((hora_decimal % 1) * 60)

        vals[fi['Hora_Partida']] = hora_decimal
        vals[fi['Minuto_Partida']] = float(minuto)
        vals[fi['hora_sin']] = np.sin(TWO_PI_24 * hora_decimal)
        vals[fi['hora_cos']] = np.cos(TWO_PI_24 * hora_decimal)
        vals[fi['DiaSemana_num']] = dia_semana
        vals[fi['dia_mes']] = fecha.day
        vals[fi['mes']] = mes
        vals[fi['es_fin_semana']] = es_fin_semana
        vals[fi['es_hora_pico']] = int((6 <= hora_decimal <= 9) or (17 <= hora_decimal <= 20))
        vals[fi['es_inicio_mes']] = es_inicio_mes
        vals[fi['es_fin_mes']] = es_fin_mes
        vals[fi['es_feriado']] = es_feriado

        # Franja
        if 6 <= hora_decimal < 9:
            franja = 'PicoManana'
        elif 9 <= hora_decimal < 12:
            franja = 'ValleManana'
        elif 12 <= hora_decimal < 14:
            franja = 'Mediodia'
        elif 14 <= hora_decimal < 17:
            franja = 'ValleTarde'
        elif 17 <= hora_decimal < 21:
            franja = 'PicoTarde'
        else:
            franja = 'Noche'

        # Franja one-hot
        vals[fi['franja_Noche']] = 0
        vals[fi['franja_PicoManana']] = 0
        vals[fi['franja_ValleManana']] = 0
        vals[fi['franja_Mediodia']] = 0
        vals[fi['franja_ValleTarde']] = 0
        vals[fi['franja_PicoTarde']] = 0
        vals[fi[f'franja_{franja}']] = 1

        # Temporada one-hot
        vals[fi['temporada_T1_Ene-Feb']] = 0
        vals[fi['temporada_T2_Mar-Jul']] = 0
        vals[fi['temporada_T3_Ago-Nov']] = 0
        vals[fi['temporada_T4_Dic']] = 0
        vals[fi[f'temporada_{temporada}']] = 1

        # Dict lookups (much faster than DataFrame .loc)
        vals[fi['tiempo_historico_arco_franja']] = franja_dict.get(
            (arco_id, franja), tiempo_hist
        )
        vals[fi['tiempo_historico_arco_dia']] = dia_dict.get(
            (arco_id, dia_semana), tiempo_hist
        )

        # Predict using booster.inplace_predict (fast, no feature name check)
        tiempo_seg = float(booster.inplace_predict(vals.reshape(1, -1))[0])
        if tiempo_seg < 0:
            tiempo_seg = 0.0

        hora_llegada = hora_actual + (tiempo_seg / 3600.0)

        resultados.append({
            'arco_id': arco_id,
            'orden': ap['posicion'],
            'hora_inicio_decimal': hora_actual,
            'hora_llegada_decimal': hora_llegada,
            'tiempo_arco_seg': tiempo_seg,
        })

        hora_actual = hora_llegada

    return resultados


def generar_horarios_salida():
    """Genera lista de horas decimales de salida."""
    horarios = []
    h = HORA_INICIO
    m = 0
    while h < HORA_FIN or (h == HORA_FIN and m == 0):
        horarios.append(h + m / 60.0)
        m += INTERVALO_MINUTOS
        if m >= 60:
            m = 0
            h += 1
    return horarios


def decimal_a_hhmm(hora_decimal):
    """Convierte hora decimal a formato HH:MM."""
    hora_decimal = hora_decimal % 24
    h = int(hora_decimal)
    m = int((hora_decimal - h) * 60)
    return f"{h:02d}:{m:02d}"


def simular_semana_completa(modelo, features, arco_stats, arco_franja_stats,
                            arco_dia_stats, feriados, arcos_ida, arcos_vuelta,
                            fechas_semana):
    """
    Simula todos los viajes de la semana.

    Returns:
        DataFrame con todos los resultados arco por arco
    """
    horarios = generar_horarios_salida()
    dias_nombres = ['Lunes', 'Martes', 'Miercoles', 'Jueves', 'Viernes', 'Sabado', 'Domingo']

    # Pre-compute static features for both directions
    logger.info("Pre-computando features estaticas por arco...")
    ida_precomp, feature_idx = precomputar_arcos(arcos_ida, 1, LINEA, arco_stats, features)
    vuelta_precomp, _ = precomputar_arcos(arcos_vuelta, 2, LINEA, arco_stats, features)
    sentidos = [(1, ida_precomp, 'ida'), (2, vuelta_precomp, 'vuelta')]

    # Convert lookup DataFrames to dicts for fast access
    franja_dict, dia_dict = convertir_lookups_a_dicts(arco_franja_stats, arco_dia_stats)

    # Get raw booster for fast numpy prediction (skip feature name validation)
    booster = modelo.get_booster()
    booster.feature_names = None

    total_viajes = len(fechas_semana) * len(horarios) * len(sentidos)
    logger.info(f"Total viajes a simular: {total_viajes:,}")
    logger.info(f"Horarios de salida: {len(horarios)} ({decimal_a_hhmm(horarios[0])} - {decimal_a_hhmm(horarios[-1])})")

    all_rows = []
    pbar = tqdm(total=total_viajes, desc="Simulando semana")

    for fecha, tipo_dia in fechas_semana.items():
        dia_nombre = dias_nombres[fecha.weekday()]
        for hora_sal in horarios:
            for sentido_num, arcos_precomp, direccion in sentidos:
                resultados_viaje = simular_viaje(
                    arcos_precomp=arcos_precomp,
                    feature_idx=feature_idx,
                    fecha=fecha,
                    hora_inicio_decimal=hora_sal,
                    booster=booster,
                    franja_dict=franja_dict,
                    dia_dict=dia_dict,
                    feriados=feriados,
                )

                for r in resultados_viaje:
                    r['fecha'] = fecha
                    r['dia'] = dia_nombre
                    r['tipo_dia'] = tipo_dia
                    r['hora_salida_viaje'] = hora_sal
                    r['sentido'] = sentido_num
                    r['direccion'] = direccion

                all_rows.extend(resultados_viaje)
                pbar.update(1)

    pbar.close()
    df = pd.DataFrame(all_rows)
    logger.info(f"Simulacion completada: {len(df):,} registros de arcos")
    return df


# ============================================================================
# FUNCIONES DE SALIDA
# ============================================================================

def generar_resumen_viajes(df):
    """
    Archivo 1: resumen_viajes.csv - Resumen de TODOS los viajes simulados.
    """
    resumen = df.groupby(
        ['fecha', 'dia', 'tipo_dia', 'hora_salida_viaje', 'sentido', 'direccion']
    ).agg(
        tiempo_total_seg=('tiempo_arco_seg', 'sum'),
        hora_llegada_decimal=('hora_llegada_decimal', 'last'),
    ).reset_index()

    resumen['hora_salida'] = resumen['hora_salida_viaje'].apply(decimal_a_hhmm)
    resumen['hora_llegada'] = resumen['hora_llegada_decimal'].apply(decimal_a_hhmm)
    resumen['tiempo_total_min'] = (resumen['tiempo_total_seg'] / 60).round(2)
    resumen['fecha'] = resumen['fecha'].astype(str)

    cols = ['fecha', 'dia', 'tipo_dia', 'hora_salida', 'hora_llegada',
            'direccion', 'tiempo_total_seg', 'tiempo_total_min']
    return resumen[cols].sort_values(['fecha', 'hora_salida', 'direccion']).reset_index(drop=True)


def generar_detallado_viajes(df):
    """
    Archivo 2: detallado_viajes.csv - Detalle arco por arco de TODOS los viajes.
    """
    det = df.copy()
    det['hora_salida_viaje_hhmm'] = det['hora_salida_viaje'].apply(decimal_a_hhmm)
    det['hora_inicio'] = det['hora_inicio_decimal'].apply(decimal_a_hhmm)
    det['hora_llegada'] = det['hora_llegada_decimal'].apply(decimal_a_hhmm)
    det['tiempo_arco_min'] = (det['tiempo_arco_seg'] / 60).round(2)
    det['fecha'] = det['fecha'].astype(str)

    cols = ['fecha', 'dia', 'tipo_dia', 'hora_salida_viaje_hhmm', 'direccion',
            'arco_id', 'orden', 'hora_inicio', 'hora_llegada',
            'tiempo_arco_seg', 'tiempo_arco_min']
    return det[cols].rename(columns={'hora_salida_viaje_hhmm': 'hora_salida_viaje'}).reset_index(drop=True)


def generar_programacion_dia_habil(df, arcos_ida, arcos_vuelta):
    """
    Archivos 3 y 4: programacion y detallado de dia habil generico.
    Agrupa predicciones Lun-Vie por arco + hora exacta, calcula P99,
    reconstruye viajes usando P99 por hora.
    """
    df_habil = df[df['tipo_dia'] == 'habil'].copy()
    if df_habil.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Hora exacta de cada arco segun su hora_inicio_decimal
    df_habil['hora'] = df_habil['hora_inicio_decimal'].astype(int)

    # P99 por arco + hora + sentido
    p99 = df_habil.groupby(['arco_id', 'hora', 'sentido']).agg(
        tiempo_arco_p99_seg=('tiempo_arco_seg', lambda x: np.percentile(x, 99))
    ).reset_index()

    # Reconstruir viajes: para cada hora de salida, usar P99 del arco segun franja
    horarios = generar_horarios_salida()
    sentidos = [(1, arcos_ida, 'ida'), (2, arcos_vuelta, 'vuelta')]

    prog_rows = []
    det_rows = []

    for hora_sal in horarios:
        for sentido_num, arcos, direccion in sentidos:
            hora_actual = hora_sal
            tiempo_total = 0.0

            for i, arco_id in enumerate(arcos):
                posicion = i + 1
                hora_exacta = int(hora_actual)

                # Buscar P99 para este arco + hora + sentido
                mask = (
                    (p99['arco_id'] == arco_id)
                    & (p99['hora'] == hora_exacta)
                    & (p99['sentido'] == sentido_num)
                )
                match = p99[mask]
                if not match.empty:
                    t_seg = float(match.iloc[0]['tiempo_arco_p99_seg'])
                else:
                    # Fallback: P99 del arco sin filtrar hora
                    mask2 = (p99['arco_id'] == arco_id) & (p99['sentido'] == sentido_num)
                    match2 = p99[mask2]
                    if not match2.empty:
                        t_seg = float(match2['tiempo_arco_p99_seg'].max())
                    else:
                        t_seg = 0.0

                hora_llegada = hora_actual + (t_seg / 3600.0)

                det_rows.append({
                    'hora_salida_viaje': decimal_a_hhmm(hora_sal),
                    'direccion': direccion,
                    'arco_id': arco_id,
                    'orden': posicion,
                    'hora_inicio': decimal_a_hhmm(hora_actual),
                    'hora_llegada': decimal_a_hhmm(hora_llegada),
                    'tiempo_arco_p99_seg': round(t_seg, 2),
                    'tiempo_arco_p99_min': round(t_seg / 60, 2),
                    'hora': hora_exacta,
                })

                tiempo_total += t_seg
                hora_actual = hora_llegada

            prog_rows.append({
                'hora_salida': decimal_a_hhmm(hora_sal),
                'hora_llegada': decimal_a_hhmm(hora_actual),
                'direccion': direccion,
                'tiempo_total_seg': round(tiempo_total, 2),
                'tiempo_total_min': round(tiempo_total / 60, 2),
            })

    df_prog = pd.DataFrame(prog_rows)
    df_det = pd.DataFrame(det_rows)
    return df_prog, df_det


def filtrar_dia(df, tipo_dia):
    """
    Filtra el dataframe para un tipo de dia especifico y genera
    programacion + detallado (prediccion directa, sin P99).
    """
    df_filtered = df[df['tipo_dia'] == tipo_dia].copy()
    if df_filtered.empty:
        return pd.DataFrame(), pd.DataFrame()

    prog = generar_resumen_viajes(df_filtered)
    det = generar_detallado_viajes(df_filtered)
    return prog, det


def guardar_todos_los_archivos(df, arcos_ida, arcos_vuelta, fechas_semana):
    """Genera y guarda todos los archivos de salida."""
    os.makedirs(CARPETA_SALIDA, exist_ok=True)

    # Archivo 1: resumen_viajes.csv
    resumen = generar_resumen_viajes(df)
    resumen.to_csv(os.path.join(CARPETA_SALIDA, 'resumen_viajes.csv'), index=False)
    logger.info(f"Guardado: resumen_viajes.csv ({len(resumen)} filas)")

    # Archivo 2: detallado_viajes.csv
    detallado = generar_detallado_viajes(df)
    detallado.to_csv(os.path.join(CARPETA_SALIDA, 'detallado_viajes.csv'), index=False)
    logger.info(f"Guardado: detallado_viajes.csv ({len(detallado)} filas)")

    # Archivos 3-4: dia habil
    prog_habil, det_habil = generar_programacion_dia_habil(df, arcos_ida, arcos_vuelta)
    if not prog_habil.empty:
        prog_habil.to_csv(os.path.join(CARPETA_SALIDA, 'programacion_dia_habil.csv'), index=False)
        logger.info(f"Guardado: programacion_dia_habil.csv ({len(prog_habil)} filas)")
        det_habil.to_csv(os.path.join(CARPETA_SALIDA, 'detallado_dia_habil.csv'), index=False)
        logger.info(f"Guardado: detallado_dia_habil.csv ({len(det_habil)} filas)")

    # Archivos 5-6: sabado
    prog_sab, det_sab = filtrar_dia(df, 'sabado')
    if not prog_sab.empty:
        prog_sab.to_csv(os.path.join(CARPETA_SALIDA, 'programacion_sabado.csv'), index=False)
        logger.info(f"Guardado: programacion_sabado.csv ({len(prog_sab)} filas)")
        det_sab.to_csv(os.path.join(CARPETA_SALIDA, 'detallado_sabado.csv'), index=False)
        logger.info(f"Guardado: detallado_sabado.csv ({len(det_sab)} filas)")

    # Archivos 7-8: domingo
    prog_dom, det_dom = filtrar_dia(df, 'domingo')
    if not prog_dom.empty:
        prog_dom.to_csv(os.path.join(CARPETA_SALIDA, 'programacion_domingo.csv'), index=False)
        logger.info(f"Guardado: programacion_domingo.csv ({len(prog_dom)} filas)")
        det_dom.to_csv(os.path.join(CARPETA_SALIDA, 'detallado_domingo.csv'), index=False)
        logger.info(f"Guardado: detallado_domingo.csv ({len(det_dom)} filas)")

    # Archivos 9-10 (condicional): feriados
    with open(RUTA_FERIADOS, 'rb') as f:
        feriados_set = pickle.load(f)

    for fecha in fechas_semana:
        if fecha in feriados_set:
            df_feriado = df[df['fecha'] == fecha].copy()
            if not df_feriado.empty:
                fecha_str = fecha.isoformat()
                prog_fer = generar_resumen_viajes(df_feriado)
                det_fer = generar_detallado_viajes(df_feriado)
                prog_fer.to_csv(os.path.join(CARPETA_SALIDA, f'programacion_feriado_{fecha_str}.csv'), index=False)
                logger.info(f"Guardado: programacion_feriado_{fecha_str}.csv ({len(prog_fer)} filas)")
                det_fer.to_csv(os.path.join(CARPETA_SALIDA, f'detallado_feriado_{fecha_str}.csv'), index=False)
                logger.info(f"Guardado: detallado_feriado_{fecha_str}.csv ({len(det_fer)} filas)")

    logger.info("Todos los archivos generados.")


# ============================================================================
# MAIN
# ============================================================================

def main():
    logger.info("=" * 80)
    logger.info("SIMULACION DE HORARIOS - LINEA %d", LINEA)
    logger.info("=" * 80)

    # Cargar artefactos
    modelo, features, arco_stats, arco_franja_stats, arco_dia_stats, feriados = cargar_artefactos()

    # Cargar rutas
    arcos_ida = cargar_ruta(LINEA, 1)
    arcos_vuelta = cargar_ruta(LINEA, 2)

    # Validar que todos los arcos existen en lookup tables
    for arco in arcos_ida + arcos_vuelta:
        if arco not in arco_stats.index:
            logger.error(f"Arco '{arco}' no encontrado en arco_stats")
            raise ValueError(f"Arco '{arco}' no encontrado en lookup tables")

    # Calcular fechas de la semana
    fechas_semana = calcular_fechas_semana(ANIO, MES, NUM_SEMANA)
    logger.info("Fechas de la semana %d de %s/%d:", NUM_SEMANA, MES, ANIO)
    for fecha, tipo in fechas_semana.items():
        es_feriado = fecha in feriados
        tipo_mostrar = f"{tipo} (FERIADO)" if es_feriado else tipo
        logger.info(f"  {fecha} ({['Lun','Mar','Mie','Jue','Vie','Sab','Dom'][fecha.weekday()]}) - {tipo_mostrar}")

    # Simular semana completa
    df = simular_semana_completa(
        modelo=modelo,
        features=features,
        arco_stats=arco_stats,
        arco_franja_stats=arco_franja_stats,
        arco_dia_stats=arco_dia_stats,
        feriados=feriados,
        arcos_ida=arcos_ida,
        arcos_vuelta=arcos_vuelta,
        fechas_semana=fechas_semana,
    )

    # Guardar archivos
    guardar_todos_los_archivos(df, arcos_ida, arcos_vuelta, fechas_semana)

    # Resumen final
    logger.info("=" * 80)
    logger.info("SIMULACION COMPLETADA")
    logger.info("=" * 80)
    tiempos = df.groupby(['fecha', 'hora_salida_viaje', 'direccion'])['tiempo_arco_seg'].sum()
    logger.info(f"Tiempo de viaje - min: {tiempos.min()/60:.1f} min, max: {tiempos.max()/60:.1f} min, promedio: {tiempos.mean()/60:.1f} min")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nSimulacion interrumpida por el usuario")
    except Exception as e:
        logger.error(f"\nError en simulacion: {e}", exc_info=True)
        raise
