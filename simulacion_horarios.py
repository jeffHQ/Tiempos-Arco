"""
Script de Simulacion de Horarios de Transporte
==============================================

Simula tiempos de viaje para una semana completa de las lineas configuradas
usando lookup tables livianas (sin cargar df_modelado).

Genera archivos Excel por linea con nombre:
  resultados/simulacion_{LINEA}_semana_{DD}_{MM}.xlsx
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

import lightgbm as lgb

# ============================================================================
# CONFIGURACION DE PARAMETROS
# ============================================================================

LINEAS = [305, 303, 336]    # Lineas a simular (se itera en main)
LINEA  = 305                # Valor por defecto (compatibilidad con modo single)

ANIO = 2026
MES = 3
NUM_SEMANA = 3              # Semana 3 de marzo = Mar 16 - Mar 22
INTERVALO_MINUTOS = 1       # 1 o 5 minutos entre salidas
HORA_INICIO = 4             # 04:00
HORA_FIN = 24               # Incluye 00:00 (medianoche)

MODELO_VERSION = 'v2'       # 'v1' o 'v2'
RUTA_MODELOS   = f'modelos/{MODELO_VERSION}/'

# Rutas de archivos v1
RUTA_MODELO        = 'modelos/v1/mejor_modelo.pkl'
RUTA_ARCO_STATS    = 'modelos/v1/arco_stats.parquet'
RUTA_ARCO_FRANJA   = 'modelos/v1/arco_franja_stats.parquet'
RUTA_ARCO_DIA      = 'modelos/v1/arco_dia_stats.parquet'
RUTA_FERIADOS      = f'modelos/{MODELO_VERSION}/feriados.pkl'
RUTA_RUTAS         = 'rutas/'
CARPETA_SALIDA     = 'resultados/'

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
    """Carga modelo v1 (XGBoost), features y lookup tables."""
    logger.info("Cargando artefactos v1...")

    with open(RUTA_MODELO, 'rb') as f:
        modelo_info = pickle.load(f)
    modelo = modelo_info['modelo']
    features = modelo_info['features']
    logger.info(f"Modelo: {modelo_info['nombre']}, {len(features)} features")

    arco_stats = pd.read_parquet(RUTA_ARCO_STATS)
    logger.info(f"arco_stats: {len(arco_stats)} arcos")

    arco_franja_stats = pd.read_parquet(RUTA_ARCO_FRANJA).set_index(['arco_id', 'franja_horaria'])
    logger.info(f"arco_franja_stats: {len(arco_franja_stats)} combinaciones")

    arco_dia_stats = pd.read_parquet(RUTA_ARCO_DIA).set_index(['arco_id', 'DiaSemana_num'])
    logger.info(f"arco_dia_stats: {len(arco_dia_stats)} combinaciones")

    with open(RUTA_FERIADOS, 'rb') as f:
        feriados = pickle.load(f)
    logger.info(f"Feriados: {len(feriados)} fechas")

    return modelo, features, arco_stats, arco_franja_stats, arco_dia_stats, feriados


def cargar_artefactos_v2():
    """Carga artefactos del modelo v2 (LightGBM)."""
    logger.info("Cargando artefactos v2 (LightGBM)...")
    base = Path('modelos/v2')

    model = lgb.Booster(model_file=str(base / 'modelo_v3.lgb'))
    with open(base / 'model_meta.pkl', 'rb') as f:
        meta = pickle.load(f)
    features = meta['features']
    logger.info(f"Modelo v2: {meta['nombre']}, {len(features)} features")

    with open(base / 'encoders_v3.pkl', 'rb') as f:
        encoders = pickle.load(f)

    lv1        = pd.read_parquet(base / 'lookup_historicos_lv1.parquet')
    lv2        = pd.read_parquet(base / 'lookup_historicos_lv2.parquet')
    lv3        = pd.read_parquet(base / 'lookup_historicos_lv3.parquet')
    l_arco     = pd.read_parquet(base / 'lookup_historicos_arco.parquet')
    l_arco_dia = pd.read_parquet(base / 'lookup_historicos_arco_dia.parquet')
    l_long     = pd.read_parquet(base / 'lookup_longitudes.parquet')
    l_geo      = pd.read_parquet(base / 'lookup_geografico.parquet')

    _lr = pd.read_parquet(base / 'lookup_rolling.parquet')
    _col_anio = [c for c in _lr.columns if 'semana' not in c and 'iso' in c.lower()][0]
    l_rolling = (
        _lr.sort_values([_col_anio, '_semana_iso'])
        .groupby(['linea', 'arco_id', 'hora'])[['rolling_4w_media', 'rolling_8w_media', 'rolling_4w_std']]
        .last()
        .reset_index()
    )
    del _lr
    l_rolling['linea'] = pd.to_numeric(l_rolling['linea']).astype('Int64')

    IDX_LV1      = lv1.set_index(['linea', 'arco_id', 'hora', 'dia_semana'])
    IDX_LV2      = lv2.set_index(['linea', 'arco_id', 'hora'])
    IDX_LV3      = lv3.set_index(['linea', 'arco_id', 'franja'])
    IDX_ARCO     = l_arco.set_index(['linea', 'arco_id'])
    IDX_ARCO_DIA = l_arco_dia.set_index(['linea', 'arco_id', 'dia_semana'])
    IDX_LONG     = l_long.set_index(['linea', 'arco_id'])
    IDX_GEO      = l_geo.set_index('parada_id')
    IDX_ROLL     = l_rolling.set_index(['linea', 'arco_id', 'hora'])

    indices = {
        'lv1': IDX_LV1, 'lv2': IDX_LV2, 'lv3': IDX_LV3,
        'arco': IDX_ARCO, 'arco_dia': IDX_ARCO_DIA,
        'long': IDX_LONG, 'geo': IDX_GEO, 'roll': IDX_ROLL,
    }

    logger.info(f"  lv1: {len(lv1):,} | lv2: {len(lv2):,} | lv3: {len(lv3):,} | rolling: {len(l_rolling):,}")

    with open(base / 'feriados.pkl', 'rb') as f:
        feriados = pickle.load(f)
    logger.info(f"Feriados: {len(feriados)} fechas")

    return model, features, encoders, indices, feriados


def cargar_ruta(linea, sentido):
    """Carga la secuencia ordenada de arco_ids para una linea y sentido."""
    archivo = os.path.join(RUTA_RUTAS, f'linea_{linea}_{sentido}.csv')
    df = pd.read_csv(archivo)
    arcos = df['arco_id'].tolist()
    logger.info(f"Ruta linea {linea} sentido {sentido}: {len(arcos)} arcos")
    return arcos


def cargar_ruta_df(linea, sentido, normalize_arco_id=False):
    """Carga la ruta completa como DataFrame. Normaliza arco_id si se indica."""
    archivo = os.path.join(RUTA_RUTAS, f'linea_{linea}_{sentido}.csv')
    df = pd.read_csv(archivo)[['parada_desde_norm', 'parada_hacia_norm', 'arco_id']].copy()
    if normalize_arco_id:
        df['arco_id'] = df['arco_id'].str.replace(' - ', '-', regex=False)
    logger.info(f"Ruta df linea {linea} sentido {sentido}: {len(df)} arcos")
    return df


# ============================================================================
# FUNCIONES DE CALCULO (compartidas)
# ============================================================================

def calcular_fechas_semana(anio, mes, num_semana):
    """Calcula las 7 fechas de la semana N del mes M."""
    primer_dia = date(anio, mes, 1)
    dia_semana_primero = primer_dia.weekday()
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
    """Clasifica la hora decimal en franja horaria (v1)."""
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
    """Construye un dict con las 40 features usando SOLO lookup tables (v1)."""
    stats = arco_stats.loc[arco_id]
    dia_semana = fecha.weekday()
    hora = int(hora_decimal)
    minuto = int((hora_decimal % 1) * 60)
    mes = fecha.month
    franja = clasificar_franja(hora_decimal)
    temporada = obtener_temporada(mes)

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
        'linea': linea,
        'sentido': sentido,
        'arco_id_encoded': int(stats['arco_id_encoded']),
        'parada_desde_encoded': int(stats['parada_desde_encoded']),
        'parada_hacia_encoded': int(stats['parada_hacia_encoded']),
        'tiempo_historico_arco': tiempo_hist,
        'tiempo_historico_arco_franja': tiempo_hist_franja,
        'tiempo_historico_arco_dia': tiempo_hist_dia,
        'std_historico_arco': float(stats['std_historico_arco']),
        'percentil_25_arco': float(stats['percentil_25_arco']),
        'percentil_75_arco': float(stats['percentil_75_arco']),
        'percentil_90_arco': float(stats['percentil_90_arco']),
        'posicion_arco_en_viaje': posicion,
        'total_arcos_viaje': total_arcos,
        'es_primer_arco': int(posicion == 1),
        'es_ultimo_arco': int(posicion == total_arcos),
        'progreso_viaje': float(posicion) / float(total_arcos),
        'retraso_acumulado': retraso_acumulado,
        'franja_Noche': int(franja == 'Noche'),
        'franja_PicoManana': int(franja == 'PicoManana'),
        'franja_ValleManana': int(franja == 'ValleManana'),
        'franja_Mediodia': int(franja == 'Mediodia'),
        'franja_ValleTarde': int(franja == 'ValleTarde'),
        'franja_PicoTarde': int(franja == 'PicoTarde'),
        'temporada_T1_Ene-Feb': int(temporada == 'T1_Ene-Feb'),
        'temporada_T2_Mar-Jul': int(temporada == 'T2_Mar-Jul'),
        'temporada_T3_Ago-Nov': int(temporada == 'T3_Ago-Nov'),
        'temporada_T4_Dic': int(temporada == 'T4_Dic'),
    }


# ============================================================================
# FUNCIONES AUXILIARES V2
# ============================================================================

def _hora_a_franja_v2(hora):
    """Clasifica la hora entera en franja horaria (v2)."""
    if hora <= 6:
        return 'Valle'
    elif hora <= 9:
        return 'Pico_manana'
    elif hora <= 16:
        return 'Mediodia'
    elif hora <= 20:
        return 'Pico_tarde'
    else:
        return 'Noche'


def _features_temporales_v2(hora_inicio_decimal, fecha, feriados):
    """Extrae los 12 features temporales fijos para un viaje (v2)."""
    h = int(hora_inicio_decimal)
    m = int((hora_inicio_decimal % 1) * 60)
    dow = fecha.weekday()
    return {
        'Hora_Partida':     h,
        'hora_sin':         np.sin(2 * np.pi * h / 24),
        'hora_cos':         np.cos(2 * np.pi * h / 24),
        'DiaSemana_num':    dow,
        'es_fin_semana':    int(dow >= 5),
        'dia_mes':          fecha.day,
        'mes':              fecha.month,
        'es_feriado':       int(fecha in feriados),
        'año':              fecha.year,
        'semana_del_mes':   (fecha.day - 1) // 7 + 1,
        'intervalo_30min':  h * 2 + int(m >= 30),
        'hora_x_diasemana': h * 10 + dow,
    }


def _encode_safe_v2(encoder, value, default=-1):
    """Codificacion segura con fallback para clases desconocidas."""
    if hasattr(encoder, 'classes_'):
        if value in encoder.classes_:
            return int(encoder.transform([value])[0])
        return default
    elif isinstance(encoder, dict):
        return encoder.get(value, default)
    return default


def _lookup_historicos_v2(indices, linea_int, arco, hora, dia_semana, franja):
    """Busqueda jerarquica de historicos: lv1 -> lv2 -> lv3 (v2)."""
    res = {
        'tiempo_historico_arco_hora_diasemana': np.nan,
        'std_historico_arco_hora_diasemana':    np.nan,
        'percentil_95_arco_hora_diasemana':     np.nan,
        'tiempo_historico_arco_hora':           np.nan,
        'nivel_historico':                      3,
        'cv_historico_arco_hora':               np.nan,
    }
    try:
        r = indices['lv1'].loc[(linea_int, arco, hora, dia_semana)]
        res.update({
            'tiempo_historico_arco_hora_diasemana': r['lv1_media'],
            'std_historico_arco_hora_diasemana':    r['lv1_std'],
            'percentil_95_arco_hora_diasemana':     r['lv1_p95'],
            'tiempo_historico_arco_hora':           r['lv1_media'],
            'nivel_historico':                      1,
            'cv_historico_arco_hora': (r['lv1_std'] / r['lv1_media']
                                       if r['lv1_media'] > 0 else 0),
        })
        return res
    except KeyError:
        pass
    try:
        r = indices['lv2'].loc[(linea_int, arco, hora)]
        res.update({
            'tiempo_historico_arco_hora': r['lv2_media'],
            'nivel_historico':            2,
            'cv_historico_arco_hora': (r['lv2_std'] / r['lv2_media']
                                       if r['lv2_media'] > 0 else 0),
        })
        return res
    except KeyError:
        pass
    try:
        indices['lv3'].loc[(linea_int, arco, franja)]
        res['nivel_historico'] = 3
    except KeyError:
        pass
    return res


def _lookup_arco_base_v2(indices, linea_int, arco, dia_semana):
    """Historicos base del arco (v2)."""
    res = {
        'tiempo_historico_arco':     np.nan,
        'tiempo_historico_arco_dia': np.nan,
        'std_historico_arco':        np.nan,
        'percentil_25_arco':         np.nan,
        'percentil_75_arco':         np.nan,
        'percentil_90_arco':         np.nan,
    }
    try:
        r = indices['arco'].loc[(linea_int, arco)]
        res.update({
            'tiempo_historico_arco': r['tiempo_historico_arco'],
            'std_historico_arco':    r['std_historico_arco'],
            'percentil_25_arco':     r['percentil_25_arco'],
            'percentil_75_arco':     r['percentil_75_arco'],
            'percentil_90_arco':     r['percentil_90_arco'],
        })
    except KeyError:
        pass
    try:
        r = indices['arco_dia'].loc[(linea_int, arco, dia_semana)]
        res['tiempo_historico_arco_dia'] = r['tiempo_historico_arco_dia']
    except KeyError:
        pass
    return res


def _lookup_rolling_v2(indices, linea_int, arco, hora):
    """Features de rolling window (v2)."""
    res = {
        'rolling_4semanas_media_arco_hora': np.nan,
        'rolling_4semanas_std_arco_hora':   np.nan,
        'rolling_8semanas_media_arco_hora': np.nan,
    }
    try:
        r = indices['roll'].loc[(linea_int, arco, hora)]
        res.update({
            'rolling_4semanas_media_arco_hora': r['rolling_4w_media'],
            'rolling_4semanas_std_arco_hora':   r['rolling_4w_std'],
            'rolling_8semanas_media_arco_hora': r['rolling_8w_media'],
        })
    except KeyError:
        pass
    return res


def _lookup_geo_v2(indices, encoders, parada_desde):
    """Features geograficos del arco (v2)."""
    res = {'distrito_desde_encoded': 0, 'es_avenida_principal': 0}
    try:
        g = indices['geo'].loc[parada_desde]
        enc_distrito = encoders['distrito_desde']
        distrito_str = g['distrito'] if g['distrito'] else 'Sin_dato'
        if distrito_str in enc_distrito.classes_:
            res['distrito_desde_encoded'] = int(enc_distrito.transform([distrito_str])[0])
        res['es_avenida_principal'] = int(g['es_avenida_principal'])
    except KeyError:
        pass
    return res


# ============================================================================
# FUNCIONES DE SIMULACION V1
# ============================================================================

def convertir_lookups_a_dicts(arco_franja_stats, arco_dia_stats):
    """Convierte lookup DataFrames a dicts nativos para acceso rapido."""
    franja_dict = {}
    for (arco_id, franja), row in arco_franja_stats.iterrows():
        franja_dict[(arco_id, franja)] = float(row['tiempo_historico_arco_franja'])

    dia_dict = {}
    for (arco_id, dia), row in arco_dia_stats.iterrows():
        dia_dict[(arco_id, dia)] = float(row['tiempo_historico_arco_dia'])

    return franja_dict, dia_dict


def precomputar_arcos(arcos, sentido, linea, arco_stats, features):
    """Pre-computa features estaticas por arco (v1)."""
    total_arcos = len(arcos)
    feature_idx = {f: i for i, f in enumerate(features)}
    arcos_precomp = []

    for i, arco_id in enumerate(arcos):
        posicion = i + 1
        stats = arco_stats.loc[arco_id]

        vals = np.zeros(len(features), dtype=np.float64)
        vals[feature_idx['linea']] = linea
        vals[feature_idx['sentido']] = sentido
        vals[feature_idx['arco_id_encoded']] = int(stats['arco_id_encoded'])
        vals[feature_idx['parada_desde_encoded']] = int(stats['parada_desde_encoded'])
        vals[feature_idx['parada_hacia_encoded']] = int(stats['parada_hacia_encoded'])
        vals[feature_idx['tiempo_historico_arco']] = float(stats['tiempo_historico_arco'])
        vals[feature_idx['std_historico_arco']] = float(stats['std_historico_arco'])
        vals[feature_idx['percentil_25_arco']] = float(stats['percentil_25_arco'])
        vals[feature_idx['percentil_75_arco']] = float(stats['percentil_75_arco'])
        vals[feature_idx['percentil_90_arco']] = float(stats['percentil_90_arco'])
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
    """Simula un viaje completo con arrays numpy pre-computados (v1)."""
    dia_semana = fecha.weekday()
    mes = fecha.month
    es_feriado = int(fecha in feriados)
    es_fin_semana = int(dia_semana >= 5)
    es_inicio_mes = int(fecha.day <= 5)
    es_fin_mes = int(fecha.day >= 25)

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

        vals[fi['franja_Noche']] = 0
        vals[fi['franja_PicoManana']] = 0
        vals[fi['franja_ValleManana']] = 0
        vals[fi['franja_Mediodia']] = 0
        vals[fi['franja_ValleTarde']] = 0
        vals[fi['franja_PicoTarde']] = 0
        vals[fi[f'franja_{franja}']] = 1

        vals[fi['temporada_T1_Ene-Feb']] = 0
        vals[fi['temporada_T2_Mar-Jul']] = 0
        vals[fi['temporada_T3_Ago-Nov']] = 0
        vals[fi['temporada_T4_Dic']] = 0
        vals[fi[f'temporada_{temporada}']] = 1

        vals[fi['tiempo_historico_arco_franja']] = franja_dict.get(
            (arco_id, franja), tiempo_hist
        )
        vals[fi['tiempo_historico_arco_dia']] = dia_dict.get(
            (arco_id, dia_semana), tiempo_hist
        )

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


# ============================================================================
# FUNCIONES DE SIMULACION V2
# ============================================================================

def precomputar_arcos_v2(ruta_df, sentido, linea, indices_v2, encoders_v2):
    """Pre-computa features estaticas por arco para el modelo v2."""
    n_arcos = len(ruta_df)
    linea_int = int(linea)
    linea_enc = _encode_safe_v2(encoders_v2['linea'], str(linea))
    sent_enc  = _encode_safe_v2(encoders_v2['sentido'], sentido)
    arcos_precomp = []

    for pos, row in enumerate(ruta_df.itertuples(index=False)):
        arco  = row.arco_id
        desde = row.parada_desde_norm
        hacia = row.parada_hacia_norm

        static = {}
        static['arco_id_encoded']      = _encode_safe_v2(encoders_v2['arco_id'], arco)
        static['parada_desde_encoded'] = _encode_safe_v2(encoders_v2['parada_desde_norm'], desde)
        static['parada_hacia_encoded'] = _encode_safe_v2(encoders_v2['parada_hacia_norm'], hacia)
        static['linea_encoded']        = linea_enc
        static['sentido_encoded']      = sent_enc

        try:
            r = indices_v2['arco'].loc[(linea_int, arco)]
            static['tiempo_historico_arco'] = r['tiempo_historico_arco']
            static['std_historico_arco']    = r['std_historico_arco']
            static['percentil_25_arco']     = r['percentil_25_arco']
            static['percentil_75_arco']     = r['percentil_75_arco']
            static['percentil_90_arco']     = r['percentil_90_arco']
        except KeyError:
            static['tiempo_historico_arco'] = np.nan
            static['std_historico_arco']    = np.nan
            static['percentil_25_arco']     = np.nan
            static['percentil_75_arco']     = np.nan
            static['percentil_90_arco']     = np.nan

        try:
            dist = indices_v2['long'].loc[(linea_int, arco), 'distancia_arco_metros']
            static['distancia_arco_metros'] = float(dist)
            static['tiene_longitud']        = 1
        except KeyError:
            static['distancia_arco_metros'] = -1.0
            static['tiene_longitud']        = 0

        static.update(_lookup_geo_v2(indices_v2, encoders_v2, desde))

        static['posicion_arco_en_viaje'] = pos
        static['total_arcos_viaje']      = n_arcos
        static['es_primer_arco']         = int(pos == 0)
        static['es_ultimo_arco']         = int(pos == n_arcos - 1)
        static['progreso_viaje']         = pos / n_arcos

        arcos_precomp.append({
            'arco_id':      arco,
            'parada_desde': desde,
            'parada_hacia': hacia,
            'static_feats': static,
            'linea_int':    linea_int,
        })

    return arcos_precomp


def simular_viaje_v2(arcos_precomp_v2, fecha, hora_inicio_decimal,
                     model_v2, features_v2, encoders_v2, indices_v2, feriados):
    """
    Simula un viaje completo con el modelo LightGBM v2.
    Hora_Partida fija en la hora de salida del viaje para todos los arcos.
    """
    dia_semana = fecha.weekday()
    hora_int   = int(hora_inicio_decimal)
    franja     = _hora_a_franja_v2(hora_int)
    temp_feats = _features_temporales_v2(hora_inicio_decimal, fecha, feriados)

    rows = []
    for ap in arcos_precomp_v2:
        arco      = ap['arco_id']
        linea_int = ap['linea_int']
        feat = {}
        feat.update(temp_feats)
        feat.update(_lookup_historicos_v2(
            indices_v2, linea_int, arco, hora_int, dia_semana, franja
        ))
        try:
            r = indices_v2['arco_dia'].loc[(linea_int, arco, dia_semana)]
            feat['tiempo_historico_arco_dia'] = r['tiempo_historico_arco_dia']
        except KeyError:
            feat['tiempo_historico_arco_dia'] = np.nan
        feat.update(_lookup_rolling_v2(indices_v2, linea_int, arco, hora_int))
        feat.update(ap['static_feats'])
        rows.append(feat)

    df_feats = pd.DataFrame(rows)
    preds = np.clip(model_v2.predict(df_feats[features_v2]), 5, None)

    hora_actual = hora_inicio_decimal
    resultados  = []
    for idx, ap in enumerate(arcos_precomp_v2):
        tiempo_seg   = float(preds[idx])
        hora_llegada = hora_actual + (tiempo_seg / 3600.0)
        resultados.append({
            'arco_id':              ap['arco_id'],
            'orden':                ap['static_feats']['posicion_arco_en_viaje'] + 1,
            'hora_inicio_decimal':  hora_actual,
            'hora_llegada_decimal': hora_llegada,
            'tiempo_arco_seg':      tiempo_seg,
        })
        hora_actual = hora_llegada

    return resultados


def _simular_semana_v2_batch(linea, sentido, ruta_df, fechas_semana, horarios,
                              model_v2, features_v2, encoders_v2, indices_v2, feriados):
    """
    Version optimizada: agrupa trips por slot unico (fecha, hora_int, half_flag).
    Dentro del mismo slot, Hora_Partida y todos los lookups son identicos,
    por lo que las predicciones de cada arco son constantes.
    Reduce las llamadas a model.predict() de ~16800 a ~280 por sentido.
    """
    arcos_precomp = precomputar_arcos_v2(ruta_df, sentido, linea, indices_v2, encoders_v2)
    n_arcos   = len(arcos_precomp)
    linea_int = int(linea)

    # Slots unicos: (fecha, hora_int, half_flag) — half_flag cambia intervalo_30min
    # Se incluye HORA_FIN para cubrir el viaje de medianoche (hora_decimal=24.0)
    fechas_list = sorted(fechas_semana.keys())
    unique_slots = []
    for fecha in fechas_list:
        for h in range(HORA_INICIO, HORA_FIN + 1):
            unique_slots.append((fecha, h, 0))
            unique_slots.append((fecha, h, 30))
    n_slots = len(unique_slots)

    # Construir matriz de features: (n_slots * n_arcos) filas
    feature_rows = []
    for fecha, hora_int, half in unique_slots:
        dia_semana  = fecha.weekday()
        hora_decimal = hora_int + half / 60.0
        franja      = _hora_a_franja_v2(hora_int)
        temp        = _features_temporales_v2(hora_decimal, fecha, feriados)

        # Precomputar lookups que dependen de (hora_int, dia_semana) para cada arco
        for ap in arcos_precomp:
            arco = ap['arco_id']
            feat = {}
            feat.update(temp)
            feat.update(_lookup_historicos_v2(
                indices_v2, linea_int, arco, hora_int, dia_semana, franja
            ))
            try:
                r = indices_v2['arco_dia'].loc[(linea_int, arco, dia_semana)]
                feat['tiempo_historico_arco_dia'] = r['tiempo_historico_arco_dia']
            except KeyError:
                feat['tiempo_historico_arco_dia'] = np.nan
            feat.update(_lookup_rolling_v2(indices_v2, linea_int, arco, hora_int))
            feat.update(ap['static_feats'])
            feature_rows.append(feat)

    # Prediccion en batch: (n_slots * n_arcos) filas -> (n_slots, n_arcos)
    df_feats    = pd.DataFrame(feature_rows)
    preds_flat  = np.clip(model_v2.predict(df_feats[features_v2]), 5, None)
    preds_matrix = preds_flat.reshape(n_slots, n_arcos)  # (n_slots, n_arcos)

    # Mapa rapido slot -> fila en preds_matrix
    slot_to_row = {slot: i for i, slot in enumerate(unique_slots)}

    # Construir DataFrames de salida por dia (operaciones vectorizadas)
    dias_nombres = ['Lunes', 'Martes', 'Miercoles', 'Jueves', 'Viernes', 'Sabado', 'Domingo']
    direccion    = 'ida' if sentido == 1 else 'vuelta'
    ordenes      = np.array([ap['static_feats']['posicion_arco_en_viaje'] + 1
                             for ap in arcos_precomp])
    arco_ids     = [ap['arco_id'] for ap in arcos_precomp]
    horarios_arr = np.array(horarios)
    n_horarios   = len(horarios_arr)

    all_dfs = []
    for fecha, tipo_dia in fechas_semana.items():
        dia_nombre = dias_nombres[fecha.weekday()]

        # Construir (n_horarios, n_arcos) de tiempos predichos
        preds_day = np.zeros((n_horarios, n_arcos))
        for h_idx, hora_sal in enumerate(horarios):
            hora_int = int(hora_sal)
            half = 30 if int((hora_sal % 1) * 60) >= 30 else 0
            slot_row = slot_to_row.get((fecha, hora_int, half))
            if slot_row is not None:
                preds_day[h_idx] = preds_matrix[slot_row]

        # Vectorizado: horas acumuladas
        cumhours = np.hstack([
            np.zeros((n_horarios, 1)),
            np.cumsum(preds_day, axis=1)
        ]) / 3600.0  # (n_horarios, n_arcos+1)

        hora_inicio_mat  = horarios_arr[:, None] + cumhours[:, :-1]  # (n_horarios, n_arcos)
        hora_llegada_mat = horarios_arr[:, None] + cumhours[:, 1:]

        df_day = pd.DataFrame({
            'arco_id':              np.tile(arco_ids, n_horarios),
            'orden':                np.tile(ordenes, n_horarios),
            'hora_inicio_decimal':  hora_inicio_mat.ravel(),
            'hora_llegada_decimal': hora_llegada_mat.ravel(),
            'tiempo_arco_seg':      preds_day.ravel(),
            'hora_salida_viaje':    np.repeat(horarios_arr, n_arcos),
        })
        df_day['fecha']     = fecha
        df_day['dia']       = dia_nombre
        df_day['tipo_dia']  = tipo_dia
        df_day['sentido']   = sentido
        df_day['direccion'] = direccion

        all_dfs.append(df_day)

    return pd.concat(all_dfs, ignore_index=True)


# ============================================================================
# FUNCIONES DE HORARIOS
# ============================================================================

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
    total_min = round((hora_decimal % 24) * 60)
    h, m = divmod(total_min, 60)
    return f"{int(h):02d}:{int(m):02d}"


def simular_semana_completa(linea, feriados, arcos_ida, arcos_vuelta, fechas_semana,
                             # v1 args
                             modelo=None, features=None, arco_stats=None,
                             arco_franja_stats=None, arco_dia_stats=None,
                             # v2 args
                             model_v2=None, features_v2=None, encoders_v2=None,
                             indices_v2=None, ruta_ida_df=None, ruta_vuelta_df=None):
    """Simula todos los viajes de la semana para la version de modelo activa."""
    horarios = generar_horarios_salida()
    total_viajes = len(fechas_semana) * len(horarios) * 2
    logger.info(f"Linea {linea}: {total_viajes:,} viajes | {len(horarios)} salidas "
                f"({decimal_a_hhmm(horarios[0])}-{decimal_a_hhmm(horarios[-1])})")

    if MODELO_VERSION == 'v1':
        dias_nombres = ['Lunes', 'Martes', 'Miercoles', 'Jueves', 'Viernes', 'Sabado', 'Domingo']
        ida_precomp, feature_idx = precomputar_arcos(arcos_ida, 1, linea, arco_stats, features)
        vuelta_precomp, _ = precomputar_arcos(arcos_vuelta, 2, linea, arco_stats, features)
        franja_dict, dia_dict = convertir_lookups_a_dicts(arco_franja_stats, arco_dia_stats)
        booster = modelo.get_booster()
        booster.feature_names = None
        sentidos = [(1, ida_precomp, 'ida'), (2, vuelta_precomp, 'vuelta')]

        all_rows = []
        pbar = tqdm(total=total_viajes, desc=f"Simulando L{linea} v1")
        for fecha, tipo_dia in fechas_semana.items():
            dia_nombre = dias_nombres[fecha.weekday()]
            for hora_sal in horarios:
                for sentido_num, arcos_precomp, direccion in sentidos:
                    res = simular_viaje(
                        arcos_precomp, feature_idx, fecha, hora_sal,
                        booster, franja_dict, dia_dict, feriados
                    )
                    for r in res:
                        r.update({'fecha': fecha, 'dia': dia_nombre, 'tipo_dia': tipo_dia,
                                  'hora_salida_viaje': hora_sal, 'sentido': sentido_num,
                                  'direccion': direccion})
                    all_rows.extend(res)
                    pbar.update(1)
        pbar.close()
        df = pd.DataFrame(all_rows)

    elif MODELO_VERSION == 'v2':
        logger.info(f"Linea {linea}: simulacion v2 batch (predicciones vectorizadas)...")
        dfs = []
        for sentido_num, ruta_df, direccion in [
            (1, ruta_ida_df, 'ida'),
            (2, ruta_vuelta_df, 'vuelta'),
        ]:
            df_s = _simular_semana_v2_batch(
                linea, sentido_num, ruta_df, fechas_semana, horarios,
                model_v2, features_v2, encoders_v2, indices_v2, feriados
            )
            dfs.append(df_s)
        df = pd.concat(dfs, ignore_index=True)

    else:
        raise ValueError(f"MODELO_VERSION desconocida: '{MODELO_VERSION}'")

    logger.info(f"Linea {linea}: {len(df):,} registros de arcos generados")
    return df


# ============================================================================
# FUNCIONES DE SALIDA
# ============================================================================

def generar_resumen_viajes(df):
    """Archivo resumen_viajes: un row por viaje."""
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
    """Archivo detallado_viajes: un row por arco por viaje."""
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
    Programacion P99 para dia habil: programa conservadora usando el
    percentil 99 de tiempos por arco+hora observados en la semana.
    """
    df_habil = df[df['tipo_dia'] == 'habil'].copy()
    if df_habil.empty:
        return pd.DataFrame(), pd.DataFrame()

    df_habil['hora'] = df_habil['hora_inicio_decimal'].astype(int)

    p99 = df_habil.groupby(['arco_id', 'hora', 'sentido']).agg(
        tiempo_arco_p99_seg=('tiempo_arco_seg', lambda x: np.percentile(x, 99))
    ).reset_index()

    horarios = generar_horarios_salida()
    sentidos = [(1, arcos_ida, 'ida'), (2, arcos_vuelta, 'vuelta')]
    prog_rows, det_rows = [], []

    for hora_sal in horarios:
        for sentido_num, arcos, direccion in sentidos:
            hora_actual = hora_sal
            tiempo_total = 0.0
            for i, arco_id in enumerate(arcos):
                hora_exacta = int(hora_actual)
                mask = ((p99['arco_id'] == arco_id)
                        & (p99['hora'] == hora_exacta)
                        & (p99['sentido'] == sentido_num))
                match = p99[mask]
                if not match.empty:
                    t_seg = float(match.iloc[0]['tiempo_arco_p99_seg'])
                else:
                    mask2 = (p99['arco_id'] == arco_id) & (p99['sentido'] == sentido_num)
                    match2 = p99[mask2]
                    t_seg = float(match2['tiempo_arco_p99_seg'].max()) if not match2.empty else 0.0

                hora_llegada = hora_actual + (t_seg / 3600.0)
                det_rows.append({
                    'hora_salida_viaje': decimal_a_hhmm(hora_sal),
                    'direccion': direccion,
                    'arco_id': arco_id,
                    'orden': i + 1,
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

    return pd.DataFrame(prog_rows), pd.DataFrame(det_rows)


def filtrar_dia(df, tipo_dia):
    """Filtra por tipo de dia y devuelve (programacion, detallado)."""
    df_f = df[df['tipo_dia'] == tipo_dia].copy()
    if df_f.empty:
        return pd.DataFrame(), pd.DataFrame()
    return generar_resumen_viajes(df_f), generar_detallado_viajes(df_f)


DIR_ORDER = {'ida': 0, 'vuelta': 1}


def _sort_df(df, cols):
    """Ordena DataFrame por columnas, poniendo 'ida' antes de 'vuelta'."""
    df = df.copy()
    if 'direccion' in cols:
        df['_dir_order'] = df['direccion'].map(DIR_ORDER)
        sort_cols = [c if c != 'direccion' else '_dir_order' for c in cols]
        df = df.sort_values(sort_cols).drop(columns='_dir_order')
    else:
        df = df.sort_values(cols)
    return df.reset_index(drop=True)


def guardar_excel_linea(df, arcos_ida, arcos_vuelta, feriados, archivo_excel):
    """
    Guarda todos los resultados de una linea en un Excel multi-hoja,
    con el mismo formato que simulacion_305_semana_23_02.xlsx.
    Incluye todas las hojas excepto detallado_viajes (demasiado grande).
    """
    os.makedirs(CARPETA_SALIDA, exist_ok=True)

    prog_habil, det_habil = generar_programacion_dia_habil(df, arcos_ida, arcos_vuelta)
    prog_sab, det_sab     = filtrar_dia(df, 'sabado')
    prog_dom, det_dom     = filtrar_dia(df, 'domingo')
    resumen               = generar_resumen_viajes(df)
    detallado             = generar_detallado_viajes(df)

    hojas = [
        ('prog_dia_habil', prog_habil,  ['hora_salida', 'direccion']),
        ('det_dia_habil',  det_habil,   ['hora_salida_viaje', 'direccion', 'orden']),
        ('prog_sabado',    prog_sab,    ['hora_salida', 'direccion']),
        ('det_sabado',     det_sab,     ['hora_salida_viaje', 'direccion', 'orden']),
        ('prog_domingo',   prog_dom,    ['hora_salida', 'direccion']),
        ('det_domingo',    det_dom,     ['hora_salida_viaje', 'direccion', 'orden']),
        ('resumen_viajes', resumen,     ['fecha', 'hora_salida', 'direccion']),
        ('detallado_viajes', detallado, ['fecha', 'hora_salida_viaje', 'direccion', 'orden']),
    ]

    # Hojas de feriados (condicional)
    fechas_unicas = df['fecha'].unique()
    for fecha in fechas_unicas:
        if fecha in feriados:
            df_fer = df[df['fecha'] == fecha].copy()
            if not df_fer.empty:
                fecha_str = str(fecha)
                hojas.append((f'prog_feriado_{fecha_str}',
                               generar_resumen_viajes(df_fer),
                               ['hora_salida', 'direccion']))

    with pd.ExcelWriter(archivo_excel, engine='openpyxl') as writer:
        for nombre_hoja, df_hoja, cols_orden in hojas:
            if df_hoja.empty:
                continue
            df_sorted = _sort_df(df_hoja, cols_orden)
            df_sorted.to_excel(writer, sheet_name=nombre_hoja, index=False)
            logger.info(f"  [{nombre_hoja}]: {len(df_sorted):,} filas")

    logger.info(f"Guardado: {archivo_excel}")


def guardar_todos_los_archivos(df, arcos_ida, arcos_vuelta, fechas_semana):
    """Guarda todos los CSVs de resultados (modo single-linea)."""
    os.makedirs(CARPETA_SALIDA, exist_ok=True)

    resumen = generar_resumen_viajes(df)
    resumen.to_csv(os.path.join(CARPETA_SALIDA, 'resumen_viajes.csv'), index=False)
    logger.info(f"Guardado: resumen_viajes.csv ({len(resumen)} filas)")

    detallado = generar_detallado_viajes(df)
    detallado.to_csv(os.path.join(CARPETA_SALIDA, 'detallado_viajes.csv'), index=False)
    logger.info(f"Guardado: detallado_viajes.csv ({len(detallado)} filas)")

    prog_habil, det_habil = generar_programacion_dia_habil(df, arcos_ida, arcos_vuelta)
    if not prog_habil.empty:
        prog_habil.to_csv(os.path.join(CARPETA_SALIDA, 'programacion_dia_habil.csv'), index=False)
        det_habil.to_csv(os.path.join(CARPETA_SALIDA, 'detallado_dia_habil.csv'), index=False)

    prog_sab, det_sab = filtrar_dia(df, 'sabado')
    if not prog_sab.empty:
        prog_sab.to_csv(os.path.join(CARPETA_SALIDA, 'programacion_sabado.csv'), index=False)
        det_sab.to_csv(os.path.join(CARPETA_SALIDA, 'detallado_sabado.csv'), index=False)

    prog_dom, det_dom = filtrar_dia(df, 'domingo')
    if not prog_dom.empty:
        prog_dom.to_csv(os.path.join(CARPETA_SALIDA, 'programacion_domingo.csv'), index=False)
        det_dom.to_csv(os.path.join(CARPETA_SALIDA, 'detallado_domingo.csv'), index=False)

    with open(RUTA_FERIADOS, 'rb') as f:
        feriados_set = pickle.load(f)
    for fecha in fechas_semana:
        if fecha in feriados_set:
            df_fer = df[df['fecha'] == fecha].copy()
            if not df_fer.empty:
                fecha_str = fecha.isoformat()
                generar_resumen_viajes(df_fer).to_csv(
                    os.path.join(CARPETA_SALIDA, f'programacion_feriado_{fecha_str}.csv'), index=False)
                generar_detallado_viajes(df_fer).to_csv(
                    os.path.join(CARPETA_SALIDA, f'detallado_feriado_{fecha_str}.csv'), index=False)

    logger.info("Todos los archivos CSV generados.")


# ============================================================================
# MAIN
# ============================================================================

def main():
    logger.info("=" * 80)
    logger.info("SIMULACION MULTI-LINEA  [modelo %s]  lineas: %s",
                MODELO_VERSION, LINEAS)
    logger.info("=" * 80)

    fechas_semana = calcular_fechas_semana(ANIO, MES, NUM_SEMANA)
    lunes = min(fechas_semana.keys())
    semana_str = f"{lunes.day:02d}_{lunes.month:02d}"

    logger.info("Fechas semana %d de %d/%d: %s - %s",
                NUM_SEMANA, MES, ANIO, lunes, max(fechas_semana.keys()))
    for fecha, tipo in fechas_semana.items():
        logger.info("  %s (%s) - %s", fecha,
                    ['Lun','Mar','Mie','Jue','Vie','Sab','Dom'][fecha.weekday()], tipo)

    # Cargar artefactos una sola vez
    if MODELO_VERSION == 'v1':
        modelo, features, arco_stats, arco_franja_stats, arco_dia_stats, feriados = cargar_artefactos()
        artefactos_v1 = dict(modelo=modelo, features=features, arco_stats=arco_stats,
                             arco_franja_stats=arco_franja_stats, arco_dia_stats=arco_dia_stats)
    elif MODELO_VERSION == 'v2':
        model_v2, features_v2, encoders_v2, indices_v2, feriados = cargar_artefactos_v2()
        artefactos_v2 = dict(model_v2=model_v2, features_v2=features_v2,
                             encoders_v2=encoders_v2, indices_v2=indices_v2)
    else:
        raise ValueError(f"MODELO_VERSION desconocida: '{MODELO_VERSION}'")

    for linea_actual in LINEAS:
        logger.info("-" * 60)
        logger.info("Procesando linea %d ...", linea_actual)

        if MODELO_VERSION == 'v1':
            arcos_ida    = cargar_ruta(linea_actual, 1)
            arcos_vuelta = cargar_ruta(linea_actual, 2)

            for arco in arcos_ida + arcos_vuelta:
                if arco not in arco_stats.index:
                    logger.error("Arco '%s' no encontrado en arco_stats (L%d)", arco, linea_actual)
                    raise ValueError(f"Arco '{arco}' no encontrado")

            df = simular_semana_completa(
                linea=linea_actual, feriados=feriados,
                arcos_ida=arcos_ida, arcos_vuelta=arcos_vuelta,
                fechas_semana=fechas_semana,
                **artefactos_v1,
            )

        else:  # v2
            ruta_ida_df    = cargar_ruta_df(linea_actual, 1, normalize_arco_id=True)
            ruta_vuelta_df = cargar_ruta_df(linea_actual, 2, normalize_arco_id=True)
            arcos_ida    = ruta_ida_df['arco_id'].tolist()
            arcos_vuelta = ruta_vuelta_df['arco_id'].tolist()

            df = simular_semana_completa(
                linea=linea_actual, feriados=feriados,
                arcos_ida=arcos_ida, arcos_vuelta=arcos_vuelta,
                fechas_semana=fechas_semana,
                ruta_ida_df=ruta_ida_df, ruta_vuelta_df=ruta_vuelta_df,
                **artefactos_v2,
            )

        # Guardar Excel con nombre por convencion
        archivo_excel = os.path.join(
            CARPETA_SALIDA, f'simulacion_{linea_actual}_semana_{semana_str}.xlsx'
        )
        guardar_excel_linea(df, arcos_ida, arcos_vuelta, feriados, archivo_excel)

        tiempos = df.groupby(['fecha', 'hora_salida_viaje', 'direccion'])['tiempo_arco_seg'].sum()
        logger.info("L%d tiempos: min=%.1f min  max=%.1f min  media=%.1f min",
                    linea_actual,
                    tiempos.min() / 60, tiempos.max() / 60, tiempos.mean() / 60)

    logger.info("=" * 80)
    logger.info("SIMULACION COMPLETADA — %d lineas", len(LINEAS))
    logger.info("=" * 80)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nSimulacion interrumpida por el usuario")
    except Exception as e:
        logger.error(f"\nError en simulacion: {e}", exc_info=True)
        raise
