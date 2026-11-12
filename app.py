"""
Parrilla Adaptativa - PMV (versión 3)
Planificación adaptativa de contenido con IA para marcas de accesorios hechos a mano.
Caso piloto: Carolina Atelier.

Ejecutar en local:  streamlit run app.py
"""
import json
import re
from copy import deepcopy
from datetime import date, timedelta
from pathlib import Path

import requests

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Parrilla Adaptativa", page_icon="🗓️", layout="wide")

# ---------------------------------------------------------------------------
# Configuración base (todo se puede editar desde la app en "Mis formatos y reglas")
# ---------------------------------------------------------------------------
COLUMNAS = [
    "fecha", "titulo", "formato", "angulo", "pilar", "vistas", "espectadores",
    "interacciones", "pct_inter_no_seguidores", "envios", "guardados",
    "comentarios", "planeado",
]
OPCIONALES = ["tipo", "personalidad", "pagado"]  # columnas que el CSV puede traer o no
TIPOS = ["Reel", "Carrusel", "Historia"]

# Marcadores disponibles en la estructura:
# {gancho} {angulo} {cta} {personalidad} {voz} {musica} {vestuario}
_ARTELIER = [
    ["0-2 s", "Firma fija de la serie: «Artelier · Episodio N» (igual en todos los episodios)"],
    ["2-8 s", "Inspiración en pantalla: {angulo}. Voz en off ({voz}): «{gancho}»"],
    ["8-18 s", "Board de inspiración: colores, texturas y materiales. La voz cuenta la historia de la inspiración"],
    ["18-35 s", "Proceso: manos de la diseñadora creando la pieza (en Artelier sí se muestra la técnica)"],
    ["35-50 s", "Revelación: mujer con personalidad {personalidad} ({vestuario}) usando la pieza. Música: {musica}"],
    ["Cierre", "Frase fija de cierre de la serie + «Pieza única» o «Disponible en la web». {cta}"],
]
FORMATOS_BASE = {
    # ---------------- Reels (30 a 60 segundos) ----------------
    "Artelier (pieza nueva)": {"tipo": "Reel", "horas": 8.0, "permite_tecnica": True,
                               "usa_personalidad": True, "estructura": _ARTELIER},
    "Artelier (modelo existente)": {"tipo": "Reel", "horas": 4.0, "permite_tecnica": True,
                                    "usa_personalidad": True, "estructura": _ARTELIER},
    "Primer plano de producto": {
        "tipo": "Reel", "horas": 1.0, "permite_tecnica": False, "usa_personalidad": False,
        "estructura": [
            ["0-3 s", "Gancho visual: detalle de la pieza en movimiento + texto: «{gancho}»"],
            ["3-15 s", "Recorrido lento por la pieza mostrando {angulo}"],
            ["15-25 s", "La pieza puesta, en un contexto cotidiano"],
            ["Cierre", "{cta}"]]},
    "Proceso (detrás de cámaras)": {
        "tipo": "Reel", "horas": 2.0, "permite_tecnica": False, "usa_personalidad": False,
        "estructura": [
            ["0-3 s", "Resultado final en pantalla + texto: «{gancho}»"],
            ["3-20 s", "Tres momentos del proceso SIN revelar la técnica, enfocados en {angulo}"],
            ["20-30 s", "Revelación de la pieza terminada"],
            ["Cierre", "{cta}"]]},
    "Transformación (antes y después)": {
        "tipo": "Reel", "horas": 1.5, "permite_tecnica": False, "usa_personalidad": True,
        "estructura": [
            ["0-3 s", "Antes: look sin accesorio + texto: «{gancho}»"],
            ["3-6 s", "Transición al ritmo de {musica}"],
            ["6-25 s", "Después: el mismo look con la pieza, con estilo {personalidad} ({vestuario})"],
            ["Cierre", "{cta}"]]},
    "Prueba en la vida real": {
        "tipo": "Reel", "horas": 1.5, "permite_tecnica": False, "usa_personalidad": False,
        "estructura": [
            ["0-3 s", "Reto en pantalla: «{gancho}»"],
            ["3-25 s", "La pieza durante el día real: {angulo}"],
            ["25-35 s", "Estado de la pieza al final del día"],
            ["Cierre", "{cta}"]]},
    "Persona luciendo la pieza": {
        "tipo": "Reel", "horas": 1.0, "permite_tecnica": False, "usa_personalidad": True,
        "estructura": [
            ["0-3 s", "Mujer de estilo {personalidad} acomodándose la pieza + texto: «{gancho}»"],
            ["3-20 s", "Momentos de su día con la pieza puesta ({vestuario}); música: {musica}"],
            ["Cierre", "{cta}"]]},
    "Inspiración temática": {
        "tipo": "Reel", "horas": 1.5, "permite_tecnica": False, "usa_personalidad": False,
        "estructura": [
            ["0-3 s", "La inspiración en pantalla: {angulo} + texto: «{gancho}»"],
            ["3-15 s", "Paralelo entre la inspiración y los detalles de la pieza"],
            ["15-25 s", "La pieza terminada"],
            ["Cierre", "{cta}"]]},
    "Mensaje de marca": {
        "tipo": "Reel", "horas": 1.0, "permite_tecnica": False, "usa_personalidad": False,
        "estructura": [
            ["0-3 s", "Frase que rompe una creencia: «{gancho}»"],
            ["3-25 s", "El mensaje de la marca sobre {angulo}, con las piezas de fondo"],
            ["Cierre", "{cta}"]]},
    "Tutorial": {
        "tipo": "Reel", "horas": 1.5, "permite_tecnica": False, "usa_personalidad": False,
        "estructura": [
            ["0-3 s", "Resultado final primero + texto: «{gancho}»"],
            ["3-25 s", "Pasos numerados (máximo 3) aplicados a {angulo}"],
            ["Cierre", "{cta}"]]},
    "Anuncio o promoción": {"proponer": False,
        "tipo": "Reel", "horas": 1.0, "permite_tecnica": False, "usa_personalidad": False,
        "estructura": [
            ["0-3 s", "El anuncio en pantalla: {angulo}"],
            ["3-15 s", "Dónde, cuándo y qué piezas"],
            ["Cierre", "{cta}"]]},
    # ---------------- Carruseles ----------------
    "Carrusel ¿Cuál eres tú? (identidad)": {
        "tipo": "Carrusel", "horas": 1.5, "permite_tecnica": False, "usa_personalidad": True,
        "estructura": [
            ["Lámina 1", "Portada: «{gancho}»"],
            ["Láminas 2-5", "Una pieza por personalidad, empezando por {personalidad}: cómo se ve y con qué outfit"],
            ["Lámina 6", "Mensaje de identidad: usar lo que te hace sentir tú misma"],
            ["Lámina 7", "{cta}"]]},
    "Carrusel board de inspiración": {
        "tipo": "Carrusel", "horas": 1.0, "permite_tecnica": False, "usa_personalidad": False,
        "estructura": [
            ["Lámina 1", "Portada con la inspiración: {angulo}"],
            ["Láminas 2-4", "Colores, texturas y referencias del board"],
            ["Lámina 5", "La pieza resultante"],
            ["Lámina 6", "{cta}"]]},
    "Carrusel educativo": {
        "tipo": "Carrusel", "horas": 1.5, "permite_tecnica": False, "usa_personalidad": False,
        "estructura": [
            ["Lámina 1", "Portada: «{gancho}»"],
            ["Láminas 2-5", "Un consejo por lámina sobre {angulo}"],
            ["Lámina 6", "Resumen"],
            ["Lámina 7", "{cta}"]]},
    "Carrusel de producto o catálogo": {
        "tipo": "Carrusel", "horas": 1.0, "permite_tecnica": False, "usa_personalidad": False,
        "estructura": [
            ["Lámina 1", "Portada: la pieza protagonista + «{gancho}»"],
            ["Láminas 2-4", "Detalles, colores y la pieza puesta en un uso cotidiano"],
            ["Lámina 5", "{cta}"]]},
    "Carrusel mensaje de marca": {
        "tipo": "Carrusel", "horas": 1.0, "permite_tecnica": False, "usa_personalidad": False,
        "estructura": [
            ["Lámina 1", "Frase que rompe una creencia: «{gancho}»"],
            ["Láminas 2-4", "El mensaje de la marca sobre {angulo}, una idea por lámina"],
            ["Lámina 5", "{cta}"]]},
    # ---------------- Historias ----------------
    "Historia de interacción (encuesta o pregunta)": {
        "tipo": "Historia", "horas": 0.25, "permite_tecnica": False, "usa_personalidad": True,
        "estructura": [["Historia", "Encuesta: ¿eres más {personalidad} o …? (sticker de encuesta sobre {angulo})"]]},
    "Historia detrás de cámaras": {
        "tipo": "Historia", "horas": 0.25, "permite_tecnica": False, "usa_personalidad": False,
        "estructura": [["Historia", "Un momento del taller hoy, relacionado con {angulo}, sin revelar la técnica"]]},
    "Historia de producto": {
        "tipo": "Historia", "horas": 0.25, "permite_tecnica": False, "usa_personalidad": False,
        "estructura": [["Historia", "La pieza en un uso cotidiano + sticker de enlace a la web"]]},
}

PERSONALIDADES_BASE = [
    {"personalidad": "Romántica / girly", "voz": "dulce y cálida", "musica": "pop suave y brillante",
     "vestuario": "rosa, pasteles, lazos y texturas suaves"},
    {"personalidad": "Rockera", "voz": "segura y directa, con ritmo marcado", "musica": "rock con guitarras",
     "vestuario": "negro, cuero, botas y metales"},
    {"personalidad": "Clásica elegante", "voz": "pausada y sofisticada", "musica": "piano o jazz suave",
     "vestuario": "neutros, sastre y perlas"},
    {"personalidad": "Bohemia", "voz": "relajada y narrativa", "musica": "acústica o folk",
     "vestuario": "tonos tierra, flores y tejidos"},
    {"personalidad": "Minimalista", "voz": "serena, con frases cortas", "musica": "electrónica suave",
     "vestuario": "monocromático y líneas limpias"},
    {"personalidad": "Artística / extravagante", "voz": "expresiva y juguetona", "musica": "ecléctica",
     "vestuario": "mucho color y mezcla de texturas"},
]

# Banco de categorías para que la app sugiera inspiraciones cuando la marca no tiene una
BANCO_INSPIRACION = [
    "un paisaje de Colombia", "una flor colombiana", "una década de la moda", "un género musical",
    "un movimiento artístico", "un elemento del mar", "una ciudad y su arquitectura", "una estación del año",
    "una textura de la naturaleza", "un oficio tradicional", "una época del cine", "un ave colombiana",
]

# Reglas de lenguaje: se buscan como palabra completa o raíz (inicio de palabra).
# aplica_tecnica = True marca las reglas de técnica, que no aplican en formatos que permiten mostrarla.
REGLAS_BASE = [
    {"evitar": "alta costura", "usar": "alta moda (para los productos)", "es_tecnica": False},
    {"evitar": "bisutería", "usar": "accesorios de diseño de autor", "es_tecnica": False},
    {"evitar": "barat", "usar": "no usar: destaca el diseño y la calidad", "es_tecnica": False},
    {"evitar": "económic", "usar": "no usar: destaca el diseño y la calidad", "es_tecnica": False},
    {"evitar": "artesan", "usar": "hecho a mano", "es_tecnica": False},
    {"evitar": "personaliza", "usar": "co-creación / co-crear", "es_tecnica": False},
    {"evitar": "navidad", "usar": "fin de año", "es_tecnica": False},
    {"evitar": "navideñ", "usar": "fin de año", "es_tecnica": False},
    {"evitar": "halloween", "usar": "el nombre del mes (octubre)", "es_tecnica": False},
    {"evitar": "semana santa", "usar": "el nombre del mes", "es_tecnica": False},
    {"evitar": "amor y amistad", "usar": "el nombre del mes (septiembre)", "es_tecnica": False},
    {"evitar": "san valentín", "usar": "el nombre del mes (febrero)", "es_tecnica": False},
    {"evitar": "bendici", "usar": "no usar referencias religiosas", "es_tecnica": False},
    {"evitar": "dios", "usar": "no usar referencias religiosas", "es_tecnica": False},
    {"evitar": "gold filled", "usar": "alambre dorado (hasta tener certificado del proveedor)", "es_tecnica": False},
    {"evitar": "goldfilled", "usar": "alambre dorado (hasta tener certificado del proveedor)", "es_tecnica": False},
    {"evitar": "bañado en oro", "usar": "alambre dorado (hasta tener certificado del proveedor)", "es_tecnica": False},
    {"evitar": "alambrismo", "usar": "no mencionar la técnica", "es_tecnica": True},
    {"evitar": "nylon", "usar": "solo materiales de la descripción del producto", "es_tecnica": True},
]

PILARES_BASE = ["Identidad (propósito)", "Hecho a mano en Colombia", "Uso cotidiano", "Co-creación",
                "Diseño de autor"]

GANCHOS = [
    "Nadie te cuenta esto sobre {angulo}",
    "Lo que pasa cuando {angulo}",
    "¿Te atreves con {angulo}?",
    "Probamos {angulo} para que tú no tengas que hacerlo",
    "Esto no es lo que esperabas de {angulo}",
]
GANCHOS_PERSONALIDAD = [
    "Todo empezó con {angulo}",
    "¿Qué pasa si {angulo} se convierte en una pieza?",
    "Esta pieza nació de {angulo}",
]
CTA = {
    "Alcance": "Llamado implícito a compartir: «Envíaselo a quien necesita ver esto»",
    "Comunidad": "Pregunta abierta para comentar: «¿Tú cuál elegirías?»",
    "Conversión": "Invitación suave: «La tuya se co-crea en la web, enlace en el perfil»",
}
DIAS = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]


# ---------------------------------------------------------------------------
# Lógica (independiente de la interfaz)
# ---------------------------------------------------------------------------
def validar_datos(df):
    return [c for c in COLUMNAS if c not in df.columns]


def preparar(df, formatos):
    df = df.copy()
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce", dayfirst=True)
    # Conteos: se quitan puntos y comas de miles (1.047 o 1,047 -> 1047)
    for c in ["vistas", "espectadores", "interacciones", "envios", "guardados", "comentarios"]:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(r"[.,\s]", "", regex=True), errors="coerce").fillna(0)
    # Porcentaje: se acepta coma decimal (12,5 -> 12.5)
    df["pct_inter_no_seguidores"] = pd.to_numeric(
        df["pct_inter_no_seguidores"].astype(str).str.replace("%", "").str.replace(",", ".").str.strip(),
        errors="coerce")  # puede quedar vacío: Meta no entrega este dato
    df["planeado"] = df["planeado"].astype(str).str.strip().str.lower().isin(["si", "sí", "1", "true"])
    if "tipo" not in df.columns:
        df["tipo"] = df["formato"].map(lambda f: formatos.get(f, {}).get("tipo", "Reel"))
    df["tipo"] = df["tipo"].fillna("Reel").replace("", "Reel")
    if "personalidad" not in df.columns:
        df["personalidad"] = ""
    df["personalidad"] = df["personalidad"].fillna("")
    if "pagado" not in df.columns:
        df["pagado"] = "no"
    df["pagado"] = df["pagado"].astype(str).str.strip().str.lower().isin(["si", "sí", "1", "true"])
    return df.sort_values("fecha").reset_index(drop=True)


REGLAS_CLASIFICACION = [  # (tipo, palabras clave, formato) en orden de prioridad
    ("Reel", ["feria", "disponible a partir", "página web", "pagina web", "lanzamiento"], "Anuncio o promoción"),
    ("Reel", ["inspirada", "inspiración", "inspiracion", "inspirándonos", "inspirado"], "Inspiración temática"),
    ("Reel", ["proceso", "process", "cobra vida", "creando", "pedido en proceso"], "Proceso (detrás de cámaras)"),
    ("Reel", ["no es el accesorio", "cómo te hace sentir", "historia", "hecho en colombia", "zona de confort",
              "genérico"], "Mensaje de marca"),
    ("Reel", ["tips", "paso a paso", "cómo usar"], "Tutorial"),
    ("Reel", ["cuál eliges", "un plan simple", "outfit"], "Persona luciendo la pieza"),
    ("Carrusel", ["tips", "historia", "estilos de", "sabías"], "Carrusel educativo"),
    ("Carrusel", ["inspiración", "inspiracion", "inspirada"], "Carrusel board de inspiración"),
    ("Carrusel", ["evolucionando", "no necesitas", "creemos"], "Carrusel mensaje de marca"),
]
DEFECTO_TIPO = {"Reel": "Primer plano de producto", "Carrusel": "Carrusel de producto o catálogo",
                "Historia": "Historia de producto"}


def clasificar(texto, tipo, formatos):
    t = str(texto).lower()
    for tp, claves, fmt in REGLAS_CLASIFICACION:
        if tp == tipo and fmt in formatos and any(k in t for k in claves):
            return fmt
    return DEFECTO_TIPO.get(tipo, list(formatos)[0])


def es_exportacion_meta(df):
    return {"hora de publicación", "alcance", "veces que se compartió"} <= {c.lower() for c in df.columns}


def desde_meta(df, formatos):
    """Convierte la exportación de Meta Business Suite (Estadísticas > Contenido) al formato de la app."""
    df = df.rename(columns={c: c.lower() for c in df.columns})
    # Solo la cuenta propia (la más frecuente): las colaboraciones con otras cuentas se excluyen
    if "nombre de usuario de la cuenta" in df.columns:
        propia = df["nombre de usuario de la cuenta"].mode()[0]
        df = df[df["nombre de usuario de la cuenta"] == propia]
    tipos = {"reel de instagram": "Reel", "secuencia de instagram": "Carrusel", "imagen de instagram": "Carrusel",
             "historia de instagram": "Historia"}
    tipo = df["tipo de publicación"].str.lower().map(tipos).fillna("Reel")
    desc = df["descripción"].fillna("")
    num = lambda c: pd.to_numeric(df.get(c, 0), errors="coerce").fillna(0)
    out = pd.DataFrame({
        "fecha": pd.to_datetime(df["hora de publicación"], format="%m/%d/%Y %H:%M", errors="coerce")
                   .dt.strftime("%d/%m/%Y"),
        "titulo": desc.str.split("\n").str[0].str.strip().str[:70],
        "formato": [clasificar(d, t, formatos) for d, t in zip(desc, tipo)],
        "angulo": "", "pilar": "",
        "vistas": num("visualizaciones"), "espectadores": num("alcance"),
        "interacciones": num("me gusta") + num("veces que se compartió") + num("comentarios") + num("veces que se guardó"),
        "pct_inter_no_seguidores": np.nan,
        "envios": num("veces que se compartió"), "guardados": num("veces que se guardó"),
        "comentarios": num("comentarios"), "planeado": "no", "tipo": tipo.values, "personalidad": "",
    })
    # Posible pauta: alcance mayor que las visualizaciones o muy por encima de lo normal de la cuenta
    mediana = out["espectadores"].median()
    out["pagado"] = np.where((out["espectadores"] > out["vistas"]) | (out["espectadores"] > 4 * mediana), "si", "no")
    return out.reset_index(drop=True)


def diagnostico(df, por="formato"):
    g = df.groupby(por).agg(
        piezas=("titulo", "count"), espectadores=("espectadores", "sum"), envios=("envios", "sum"),
        inter_no_seg=("pct_inter_no_seguidores", "mean"), comentarios=("comentarios", "sum"))
    g["tasa_envios_%"] = (g["envios"] / g["espectadores"].replace(0, np.nan) * 100).round(2)
    g["inter_no_seguidores_%"] = g["inter_no_seg"].round(1)
    g["tasa_comentarios_%"] = (g["comentarios"] / g["espectadores"].replace(0, np.nan) * 100).round(2)
    g["confiabilidad"] = np.where(g["piezas"] >= 5, "Suficiente", "Pocos datos")
    return g[["piezas", "tasa_envios_%", "inter_no_seguidores_%", "tasa_comentarios_%", "confiabilidad"]] \
        .sort_values("tasa_envios_%", ascending=False)


def posteriores(df, columna, opciones, descuento, fuerza_prior=200):
    """Modelo Beta-Binomial sobre la tasa de envíos, para cada opción de `columna`
    (un formato o una personalidad).
    - Prior empírico: toda opción parte de la tasa global de los datos, con un peso
      equivalente a `fuerza_prior` espectadores (una opción nueva no parte de un valor irreal).
    - Memoria que se desvanece: peso = descuento ** (posición desde la pieza más reciente)."""
    df = df.sort_values("fecha").reset_index(drop=True)
    tasa = df["envios"].sum() / max(df["espectadores"].sum(), 1) if len(df) else 0.01
    a0, b0 = max(tasa * fuerza_prior, 0.5), max((1 - tasa) * fuerza_prior, 0.5)
    w = descuento ** ((len(df) - 1) - np.arange(len(df)))
    filas = []
    for o in opciones:
        m = (df[columna] == o).to_numpy() if len(df) else np.array([], dtype=bool)
        exitos = float((w[m] * df.loc[m, "envios"]).sum()) if len(df) else 0.0
        fracasos = float((w[m] * (df.loc[m, "espectadores"] - df.loc[m, "envios"]).clip(lower=0)).sum()) \
            if len(df) else 0.0
        a, b = a0 + exitos, b0 + fracasos
        filas.append({"opcion": o, "alpha": a, "beta": b, "media_%": a / (a + b) * 100, "piezas": int(m.sum())})
    return pd.DataFrame(filas).set_index("opcion")


def thompson(post, n, rng, excluir=(), variar=True):
    """En cada espacio muestrea una tasa probable por opción y elige la mayor:
    explota lo que funciona sin dejar de explorar las opciones con pocos datos.
    Con variar=True no repite una opción en la misma semana mientras queden otras disponibles."""
    elegidos = []
    for _ in range(n):
        usadas = set(excluir) | set(elegidos) if variar else set(excluir)
        familia = lambda o: str(o).split(" (")[0]  # p. ej., las dos versiones de Artelier son una familia
        familias = {familia(u) for u in usadas}
        candidatas = post[[familia(o) not in familias for o in post.index]]
        if candidatas.empty:
            candidatas = post
        muestras = {o: rng.beta(r.alpha, r.beta) for o, r in candidatas.iterrows()}
        elegidos.append(max(muestras, key=muestras.get))
    return elegidos


def saturacion(muestra, candidatos):
    total = muestra["cantidad"].sum()
    conteo = muestra.groupby("angulo")["cantidad"].sum()
    filas = []
    for a in sorted(set(candidatos) | set(conteo.index)):
        c = float(conteo.get(a, 0))
        filas.append({"angulo": a, "veces_en_muestra": int(c),
                      "saturacion_%": round(c / total * 100, 1) if total else 0.0,
                      "es_propio": a in candidatos})
    out = pd.DataFrame(filas).sort_values("saturacion_%")
    out["nivel"] = pd.cut(out["saturacion_%"], [-0.1, 10, 25, 100], labels=["Baja", "Media", "Alta"])
    return out


def revisar_reglas(texto, reglas, permite_tecnica):
    t = texto.lower()
    alertas = []
    for r in reglas:
        clave = str(r.get("evitar", "")).strip().lower()
        if not clave or (permite_tecnica and r.get("es_tecnica", False)):
            continue
        if re.search(r"\b" + re.escape(clave), t):
            alertas.append(f"Evita «{clave}…»: {r.get('usar', '')}")
    return alertas


def partir_angulo(linea):
    """«Referencia interna | lo que se dice en público». Sin «|», todo es público."""
    if "|" in str(linea):
        ref, pub = [x.strip() for x in str(linea).split("|", 1)]
        return ref, limpiar_publico(pub or ref)
    return "", limpiar_publico(linea)


def limpiar_publico(texto):
    return re.sub(r"^\s*inspiraci[oó]n\s*:\s*", "", str(texto).strip(), flags=re.IGNORECASE)


def guion(nombre, formato, angulo, pilar, objetivo, persona, rng):
    referencia, angulo = partir_angulo(angulo)
    usa_p = bool(formato.get("usa_personalidad", False) and persona and persona.get("personalidad"))
    gancho = rng.choice(GANCHOS_PERSONALIDAD if usa_p else GANCHOS).format(angulo=angulo.lower())
    cta = CTA[objetivo]
    if usa_p and objetivo == "Alcance":
        cta = f"Llamado implícito a compartir: «Envíaselo a tu amiga más {persona['personalidad'].lower()}»"
    datos = {"gancho": gancho, "angulo": angulo.lower(), "cta": cta,
             "personalidad": persona["personalidad"] if usa_p else "",
             "voz": persona.get("voz", "") if usa_p else "", "musica": persona.get("musica", "") if usa_p else "",
             "vestuario": persona.get("vestuario", "") if usa_p else ""}
    pasos = []
    for t, d in formato["estructura"]:
        try:
            pasos.append((t, d.format(**datos)))
        except (KeyError, IndexError, ValueError):
            pasos.append((t, d))
    return {"formato": nombre, "tipo": formato.get("tipo", "Reel"), "angulo": angulo, "referencia": referencia,
            "pilar": pilar,
            "objetivo": objetivo, "personalidad": datos["personalidad"], "gancho": gancho, "pasos": pasos,
            "horas": float(formato["horas"]), "permite_tecnica": bool(formato.get("permite_tecnica")),
            "aprobada": False}


def prompt_ia(idea, marca, reglas):
    pasos = "\n".join(f"- {t}: {d}" for t, d in idea["pasos"])
    reglas_txt = "; ".join(f"evita «{r['evitar']}» ({r['usar']})" for r in reglas
                           if r.get("evitar") and not (idea["permite_tecnica"] and r.get("es_tecnica")))
    extra = (f"La voz y el estilo deben sentirse como una mujer de personalidad {idea['personalidad']}. "
             if idea.get("personalidad") else "")
    duracion = "Duración: entre 30 y 60 segundos. " if idea["tipo"] == "Reel" else ""
    return (
        f"Eres guionista de contenido para {marca}, una marca de accesorios hechos a mano con diseño de autor. "
        f"Su propósito: que cada mujer pierda el miedo a usar lo que la hace sentir ella misma.\n"
        f"Mejora este {idea['tipo'].lower()} de formato «{idea['formato']}» sobre «{idea['angulo']}» "
        f"(pilar: {idea['pilar']}; objetivo: {idea['objetivo']}). {extra}{duracion}\n"
        f"Mantén la estructura, haz el gancho más específico y conserva un llamado a la acción implícito.\n"
        f"Reglas de marca: {reglas_txt}.\n\n{pasos}"
    )


GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta"


@st.cache_data(ttl=3600, show_spinner=False)
def modelo_gemini(clave):
    """Busca un modelo Flash disponible para la clave (los nombres cambian con el tiempo)."""
    r = requests.get(f"{GEMINI_URL}/models", params={"key": clave}, timeout=20)
    r.raise_for_status()
    nombres = [m["name"] for m in r.json().get("models", [])
               if "generateContent" in m.get("supportedGenerationMethods", []) and "flash" in m["name"]]
    estables = [n for n in nombres if "preview" not in n and "exp" not in n and "lite" not in n]
    elegido = (sorted(estables) or sorted(nombres))
    if not elegido:
        raise RuntimeError("No hay modelos Flash disponibles para esta clave.")
    return elegido[-1]


def escribir_guion_ia(idea, marca, reglas, clave):
    instruccion = prompt_ia(idea, marca, reglas) + (
        "\n\nEntrega el guion COMPLETO y listo para grabar, en español de Colombia, con estas secciones:\n"
        "1. Gancho (texto exacto en pantalla de los primeros 3 segundos).\n"
        "2. Guion por bloques de tiempo: qué se ve (toma), texto en pantalla y voz en off palabra por palabra.\n"
        "3. Música o ambiente sugerido.\n"
        "4. Descripción para la publicación (máximo 3 líneas) y 5 hashtags.\n"
        "No inventes datos de precios, materiales ni certificaciones que no aparezcan aquí.")
    modelo = modelo_gemini(clave)
    r = requests.post(f"{GEMINI_URL}/{modelo}:generateContent", params={"key": clave}, timeout=60,
                      json={"contents": [{"parts": [{"text": instruccion}]}]})
    if r.status_code == 429:
        raise RuntimeError("Se alcanzó el límite gratuito de la IA por hoy. Intenta más tarde o usa "
                           "«Instrucción para tu asistente de IA».")
    r.raise_for_status()
    partes = r.json()["candidates"][0]["content"]["parts"]
    return "".join(p.get("text", "") for p in partes).strip()


def clave_ia():
    try:
        return st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        return ""


def indicadores_plan(reg, planeadas):
    publicadas = len(reg)
    segun_plan = int(reg["planeado"].sum()) if publicadas else 0
    return {"adherencia": segun_plan / planeadas if planeadas else 0.0,
            "improvisacion": (publicadas - segun_plan) / publicadas if publicadas else 0.0}


def generar_semana(df, formatos, personalidades, piezas, angulos, pilares, descuento, rng):
    ideas = []
    pers_names = [p["personalidad"] for p in personalidades if p.get("personalidad")]
    df = df[~df["pagado"]] if "pagado" in df.columns else df
    pers_post = posteriores(df, "personalidad", pers_names, descuento) if pers_names else None
    por_nombre = {p["personalidad"]: p for p in personalidades}
    for tipo in TIPOS:
        n = int(piezas.get(tipo, 0))
        # Los anuncios (ferias, lanzamientos) dependen de eventos: no los propone el motor
        opciones = [f for f, v in formatos.items() if v.get("tipo", "Reel") == tipo and v.get("proponer", True)]
        if n <= 0 or not opciones:
            continue
        datos_tipo = df[df["tipo"] == tipo]
        # 1) Formatos fijos (por ejemplo, una serie que debe salir cada semana)
        fijos = []
        for f in opciones:
            fijos += [f] * int(formatos[f].get("fijo_semana", 0) or 0)
        fijos = fijos[:n]
        # 2) El resto de espacios los decide el motor adaptativo
        libres = n - len(fijos)
        elegidos = fijos + (thompson(posteriores(datos_tipo, "formato", opciones, descuento), libres, rng,
                                     excluir=set(fijos), variar=False) if libres > 0 else [])
        objetivos = (["Alcance", "Comunidad", "Conversión"] * 10)[:n]
        for f, o in zip(elegidos, objetivos):
            persona = por_nombre.get(thompson(pers_post, 1, rng)[0]) if pers_post is not None else None
            ideas.append(guion(f, formatos[f], rng.choice(angulos), rng.choice(pilares), o, persona, rng))
    return ideas


# ---------------------------------------------------------------------------
# Estado de la sesión (los datos solo viven en esta sesión; no se guardan en el servidor)
# ---------------------------------------------------------------------------
ss = st.session_state
ss.setdefault("acepta", False)
ss.setdefault("df", None)
ss.setdefault("marca", {"nombre": "Carolina Atelier", "pilares": list(PILARES_BASE), "horas": 12.0,
                        "piezas": {"Reel": 3, "Carrusel": 2, "Historia": 6}})
ss.setdefault("formatos", deepcopy(FORMATOS_BASE))
ss.setdefault("personalidades", deepcopy(PERSONALIDADES_BASE))
ss.setdefault("reglas", deepcopy(REGLAS_BASE))
ss.setdefault("muestra", pd.DataFrame({"angulo": ["(ejemplo) 5 peinados con diadema", "(ejemplo) Get ready with me"],
                                       "cantidad": [0, 0]}))
ss.setdefault("candidatos", ["Inspiración: el mar de 7 colores"])
ss.setdefault("ideas", [])
ss.setdefault("parrilla", None)
ss.setdefault("registro", pd.DataFrame(columns=COLUMNAS + OPCIONALES))
ss.setdefault("descuento", 0.9)
ss.setdefault("meta_raw", None)
rng = np.random.default_rng()

# ---------------------------------------------------------------------------
# Bienvenida y autorización
# ---------------------------------------------------------------------------
st.sidebar.title("🗓️ Parrilla Adaptativa")
st.sidebar.caption("PMV · Caso piloto Carolina Atelier")

if not ss.acepta:
    st.title("Planifica tu contenido con tus propios datos")
    st.write("Esta aplicación analiza el desempeño de tu contenido, detecta ángulos poco explotados, "
             "sugiere personalidades e inspiraciones y arma una parrilla semanal con guiones que se ajusta "
             "con cada resultado.")
    st.info("**Tratamiento de datos (Ley 1581 de 2012).** Los datos que cargues se procesan solo "
            "durante esta sesión y no se almacenan en el servidor. La aplicación nunca pide "
            "contraseñas. Puedes descargar tus resultados y borrar todo al cerrar la pestaña.")
    if st.checkbox("Autorizo el tratamiento de mis datos para este análisis"):
        if st.button("Comenzar", type="primary"):
            ss.acepta = True
            st.rerun()
    st.stop()

pantalla = st.sidebar.radio("Recorrido", [
    "1. Perfil de marca", "2. Mis formatos y reglas", "3. Cargar datos", "4. Diagnóstico",
    "5. Radar de saturación", "6. Ideas y guiones", "7. Parrilla semanal", "8. Registrar resultados"])
if ss.df is not None:
    st.sidebar.success(f"{len(ss.df)} piezas cargadas")
else:
    st.sidebar.warning("Sin datos cargados")

# ---------------------------------------------------------------------------
# 1. Perfil de marca
# ---------------------------------------------------------------------------
if pantalla.startswith("1"):
    st.header("1. Perfil de marca")
    m = ss.marca
    m["nombre"] = st.text_input("Nombre de la marca", m["nombre"])
    pil = st.text_input("Propósito y pilares de marca (separados por coma)", ", ".join(m["pilares"]))
    m["pilares"] = [p.strip() for p in pil.split(",") if p.strip()]
    m["horas"] = st.number_input("Horas disponibles por semana para contenido (entre todo el equipo)",
                                 0.5, 80.0, float(m["horas"]), 0.5)
    st.subheader("Piezas por semana")
    c1, c2, c3 = st.columns(3)
    m["piezas"]["Reel"] = c1.number_input("Reels", 0, 14, int(m["piezas"]["Reel"]))
    m["piezas"]["Carrusel"] = c2.number_input("Carruseles", 0, 14, int(m["piezas"]["Carrusel"]))
    m["piezas"]["Historia"] = c3.number_input("Historias", 0, 50, int(m["piezas"]["Historia"]))
    st.success("Perfil guardado en la sesión.")

# ---------------------------------------------------------------------------
# 2. Mis formatos y reglas
# ---------------------------------------------------------------------------
elif pantalla.startswith("2"):
    st.header("2. Mis formatos y reglas")
    t1, t2, t3, t4 = st.tabs(["Formatos", "Personalidades", "Reglas de lenguaje", "Guardar / cargar"])

    with t1:
        st.write("Formatos que la app puede proponer. Ajusta las horas reales o crea formatos nuevos.")
        st.caption("**horas**: tiempo real de producción de una pieza · **fijo_semana**: cuántas piezas de este "
                   "formato deben salir sí o sí cada semana (0 = lo decide el motor) · **permite_tecnica**: solo en "
                   "formatos donde se puede mostrar la técnica · **usa_personalidad**: la pieza se adapta a una "
                   "personalidad (voz, música y vestuario).")
        resumen = pd.DataFrame([{"formato": k, "tipo": v.get("tipo", "Reel"), "horas": v["horas"],
                                 "fijo_semana": int(v.get("fijo_semana", 0) or 0),
                                 "permite_tecnica": v["permite_tecnica"], "usa_personalidad": v["usa_personalidad"]}
                                for k, v in ss.formatos.items()])
        editado = st.data_editor(resumen, disabled=["formato", "tipo"], hide_index=True, width="stretch",
                                 key="ed_fmt",
                                 column_config={"fijo_semana": st.column_config.NumberColumn(min_value=0, max_value=7,
                                                                                             step=1)})
        for r in editado.itertuples():
            ss.formatos[r.formato].update(horas=float(r.horas), fijo_semana=int(r.fijo_semana or 0),
                                          permite_tecnica=bool(r.permite_tecnica),
                                          usa_personalidad=bool(r.usa_personalidad))
        ver = st.selectbox("Ver la estructura de", list(ss.formatos))
        st.table(pd.DataFrame(ss.formatos[ver]["estructura"], columns=["Tiempo / lámina", "Qué pasa"]))

        st.subheader("Crear un formato nuevo")
        with st.form("nuevo_formato", clear_on_submit=True):
            c1, c2 = st.columns(2)
            nombre = c1.text_input("Nombre del formato")
            tipo = c2.selectbox("Tipo", TIPOS)
            horas = st.number_input("Horas de producción", 0.25, 20.0, 2.0, 0.25)
            c3, c4 = st.columns(2)
            tec = c3.checkbox("Permite mostrar técnica")
            per = c4.checkbox("Usa personalidad (tono, música y vestuario)")
            bloques = st.text_area(
                "Bloques, uno por línea, con el formato  tiempo | qué pasa",
                "0-3 s | Gancho: «{gancho}»\n3-25 s | Desarrollo sobre {angulo}\nCierre | {cta}",
                help="Puedes usar {gancho} {angulo} {cta} {personalidad} {voz} {musica} {vestuario}")
            if st.form_submit_button("Agregar formato", type="primary"):
                estructura = [[p.strip() for p in l.split("|", 1)] for l in bloques.splitlines() if "|" in l]
                if not nombre.strip() or not estructura:
                    st.error("Escribe un nombre y al menos un bloque con el formato  tiempo | qué pasa")
                else:
                    ss.formatos[nombre.strip()] = {"tipo": tipo, "horas": horas, "permite_tecnica": tec,
                                                   "usa_personalidad": per, "estructura": estructura}
                    st.success(f"Formato «{nombre.strip()}» agregado.")
                    st.rerun()

    with t2:
        st.write("Catálogo de personalidades. **La app elige cuál proponer** con el motor adaptativo: prueba "
                 "todas, mide cuáles generan más envíos y las propone más. Puedes agregar o quitar personalidades.")
        pdf = st.data_editor(pd.DataFrame(ss.personalidades), num_rows="dynamic", hide_index=True,
                             width="stretch", key="ed_per")
        ss.personalidades = [r for r in pdf.fillna("").to_dict("records") if str(r["personalidad"]).strip()]

    with t3:
        st.write("Palabras que nunca deben aparecer. La app busca la palabra completa o su raíz "
                 "(«artesan» detecta artesanía y artesanal). Las reglas marcadas como «es_tecnica» no aplican "
                 "en los formatos que permiten mostrar técnica, como Artelier.")
        rdf = st.data_editor(pd.DataFrame(ss.reglas), num_rows="dynamic", hide_index=True, width="stretch",
                             key="ed_reg",
                             column_config={"es_tecnica": st.column_config.CheckboxColumn(default=False)})
        ss.reglas = [r for r in rdf.fillna({"usar": "", "es_tecnica": False}).to_dict("records")
                     if str(r.get("evitar", "")).strip()]
        vacias = [r["evitar"] for r in ss.reglas if not str(r.get("usar", "")).strip()]
        if vacias:
            st.warning(f"Estas reglas no tienen texto en «usar»: {', '.join(vacias)}")

    with t4:
        st.write("Los cambios solo duran en esta sesión. Descarga tu configuración para volver a cargarla después.")
        config = {"marca": ss.marca, "formatos": ss.formatos, "personalidades": ss.personalidades,
                  "reglas": ss.reglas}
        st.download_button("Descargar mi configuración", json.dumps(config, ensure_ascii=False, indent=2),
                           "configuracion_marca.json", "application/json")
        arch = st.file_uploader("Cargar una configuración guardada", type="json")
        if arch and st.button("Aplicar configuración"):
            cfg = json.load(arch)
            for k in ["marca", "formatos", "personalidades", "reglas"]:
                if k in cfg:
                    ss[k] = cfg[k]
            st.success("Configuración cargada.")

# ---------------------------------------------------------------------------
# 3. Cargar datos
# ---------------------------------------------------------------------------
elif pantalla.startswith("3"):
    st.header("3. Cargar datos de tu contenido")
    st.download_button("Descargar plantilla CSV",
                       pd.DataFrame(columns=COLUMNAS + OPCIONALES).to_csv(index=False).encode("utf-8"),
                       "plantilla_contenido.csv", "text/csv")
    st.caption("En «formato» usa exactamente uno de estos nombres: " + "; ".join(ss.formatos) +
               ". Las columnas «tipo» (Reel, Carrusel o Historia) y «personalidad» son opcionales.")
    origen = st.radio("¿Cómo quieres cargar los datos?", ["Subir mi CSV", "Usar datos de ejemplo (ficticios)"],
                      horizontal=True)
    df = None
    if origen == "Subir mi CSV":
        archivo = st.file_uploader("Archivo CSV con las columnas de la plantilla", type="csv")
        if archivo:
            # Detecta el separador (coma o punto y coma, como guarda Excel en español)
            try:
                df = pd.read_csv(archivo, sep=None, engine="python", encoding="utf-8-sig")
            except UnicodeDecodeError:
                archivo.seek(0)
                df = pd.read_csv(archivo, sep=None, engine="python", encoding="latin-1")
            if es_exportacion_meta(df):
                ss.meta_raw = desde_meta(df, ss.formatos)
                df = None
            else:
                df.columns = [str(c).strip().lower() for c in df.columns]
    else:
        try:
            df = pd.read_csv(Path(__file__).parent / "datos_ejemplo.csv")
            st.caption("Datos inventados solo para demostrar el funcionamiento. No son de ninguna marca real.")
        except FileNotFoundError:
            st.error("No se encontró datos_ejemplo.csv junto a app.py.")
    if origen == "Subir mi CSV" and ss.get("meta_raw") is not None:
        st.info("Detecté una **exportación de Meta Business Suite**. La app clasificó cada publicación por formato "
                "y marcó las que parecen pautadas (alcance mayor que las visualizaciones). **Revisa y corrige** las "
                "columnas «formato» y «pagado» antes de usar los datos.")
        edit = st.data_editor(
            ss.meta_raw, width="stretch", hide_index=True, key="ed_meta",
            column_config={"formato": st.column_config.SelectboxColumn(options=list(ss.formatos), required=True),
                           "pagado": st.column_config.SelectboxColumn(options=["si", "no"], required=True),
                           "planeado": st.column_config.SelectboxColumn(options=["si", "no"], required=True)})
        if st.button("Usar estos datos", type="primary"):
            ss.df = preparar(edit, ss.formatos)
            st.success(f"{len(ss.df)} publicaciones cargadas ({int(ss.df['pagado'].sum())} marcadas como pagadas, "
                       "que no entran al aprendizaje).")
        st.download_button("Descargar datos revisados (CSV)", edit.to_csv(index=False).encode("utf-8"),
                           "contenido_revisado.csv", "text/csv")
    if df is not None:
        faltan = validar_datos(df)
        if faltan:
            st.error(f"Faltan columnas: {', '.join(faltan)}. Revisa la plantilla.")
        else:
            desconocidos = sorted(set(df["formato"].dropna()) - set(ss.formatos))
            if desconocidos:
                st.warning(f"Formatos que no existen en «Mis formatos»: {', '.join(desconocidos)}. "
                           "Corrige el nombre en el CSV o créalos en el paso 2.")
            ss.df = preparar(df, ss.formatos)
            st.success(f"{len(ss.df)} piezas cargadas correctamente.")
            st.dataframe(ss.df[COLUMNAS + OPCIONALES], width="stretch")

# ---------------------------------------------------------------------------
# 4. Diagnóstico
# ---------------------------------------------------------------------------
elif pantalla.startswith("4"):
    st.header("4. Diagnóstico")
    if ss.df is None:
        st.warning("Primero carga tus datos en el paso 3.")
        st.stop()
    base_df = ss.df[~ss.df["pagado"]]
    if ss.df["pagado"].sum():
        st.caption(f"Se excluyen {int(ss.df['pagado'].sum())} publicaciones pagadas: la pauta no refleja el "
                   "desempeño orgánico del formato.")
    sin = int((base_df["espectadores"] <= 0).sum())
    if sin:
        st.warning(f"{sin} piezas no tienen espectadores: no aportan al aprendizaje del motor. "
                   "Completa espectadores y envíos en tu CSV y vuelve a cargarlo.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Piezas orgánicas analizadas", len(base_df))
    c2.metric("Tasa de envíos global", f"{base_df['envios'].sum() / max(base_df['espectadores'].sum(), 1) * 100:.2f} %")
    pct = base_df["pct_inter_no_seguidores"].mean()
    c3.metric("Interacción de no seguidores (promedio)", "No disponible" if pd.isna(pct) else f"{pct:.1f} %")
    st.subheader("Por tipo de contenido")
    st.dataframe(diagnostico(base_df, "tipo"), width="stretch")
    st.subheader("Por formato")
    d = diagnostico(base_df, "formato")
    st.dataframe(d, width="stretch")
    st.bar_chart(d["tasa_envios_%"])
    con_p = base_df[base_df["personalidad"] != ""]
    if len(con_p):
        st.subheader("Por personalidad")
        st.dataframe(diagnostico(con_p, "personalidad"), width="stretch")
    st.info("Las opciones con menos de 5 piezas tienen resultados poco confiables: el motor las seguirá explorando.")
    st.subheader("Motor adaptativo")
    ss.descuento = st.slider("Peso de los datos antiguos (1 = todos pesan igual; menor = se adapta más rápido)",
                             0.70, 1.00, float(ss.descuento), 0.01)
    post = posteriores(base_df, "formato", list(ss.formatos), ss.descuento)
    st.dataframe(post[["piezas", "media_%"]].rename(columns={"media_%": "tasa_envios_esperada_%"}).round(2),
                 width="stretch")
    st.caption("Las opciones sin datos parten de la tasa global de la cuenta y el motor las irá probando.")

# ---------------------------------------------------------------------------
# 5. Radar de saturación
# ---------------------------------------------------------------------------
elif pantalla.startswith("5"):
    st.header("5. Radar de saturación")
    st.write("Revisa en Instagram el contenido más visible de una temática y anota cuántas veces aparece cada "
             "ángulo. Las filas «(ejemplo)» son solo guía: reemplázalas con tu conteo real.")
    st.text_input("Temática analizada", "Accesorios para el cabello")
    ss.muestra = st.data_editor(ss.muestra, num_rows="dynamic", width="stretch",
                                column_config={"cantidad": st.column_config.NumberColumn(min_value=0, step=1)})
    st.caption("Escribe un tema o inspiración por fila y presiona **Enter**. No tiene que ser una inspiración: "
               "puede ser cualquier tema, como «texturas del terciopelo» o «cómo combinarla con ropa de oficina». "
               "Si la inspiración es una persona o una marca, escribe **referencia interna | lo que se dice en "
               "público**. Ejemplo: «Shakira | la danza árabe y sus vestuarios con brillo». El guion solo usa la parte "
               "pública; la referencia queda como nota para el equipo.")
    cand = st.data_editor(pd.DataFrame({"tema o inspiración": ss.candidatos or [""]}), num_rows="dynamic",
                          hide_index=True, width="stretch", key="ed_cand")
    ss.candidatos = [str(c).strip() for c in cand["tema o inspiración"].fillna("") if str(c).strip()]
    if st.button("Sugerir una inspiración"):
        sug = "Inspiración: " + rng.choice(BANCO_INSPIRACION)
        ss.candidatos.append(sug)
        st.rerun()
    sat = saturacion(ss.muestra.dropna(), ss.candidatos)
    st.dataframe(sat, width="stretch")
    if ss.muestra["cantidad"].fillna(0).sum() == 0:
        st.info("La muestra está en cero: ingresa el conteo real para que el radar mida la saturación.")
    bajos = sat[(sat["nivel"] == "Baja") & sat["es_propio"]]["angulo"].tolist()
    st.success(f"Ángulos de baja saturación disponibles: {', '.join(bajos) if bajos else 'ninguno'}")

# ---------------------------------------------------------------------------
# 6. Ideas y guiones
# ---------------------------------------------------------------------------
elif pantalla.startswith("6"):
    st.header("6. Ideas y guiones")
    if ss.df is None:
        st.warning("Primero carga tus datos en el paso 3.")
        st.stop()
    sat = saturacion(ss.muestra.dropna(), ss.candidatos)
    angulos = sat[(sat["nivel"] != "Alta") & sat["es_propio"]]["angulo"].tolist() or ss.candidatos
    if not angulos:
        angulos = ["Inspiración: " + x for x in BANCO_INSPIRACION]
    pilares = ss.marca["pilares"] or ["Marca"]
    with st.expander("➕ Agregar una idea puntual (sin cambiar el resto de la semana)"):
        with st.form("puntual", clear_on_submit=True):
            c1, c2 = st.columns(2)
            f_p = c1.selectbox("Formato", list(ss.formatos))
            o_p = c2.selectbox("Objetivo", list(CTA))
            tema_p = st.text_input("Tema o inspiración", placeholder="Ejemplo: texturas del terciopelo y la pedrería")
            c3, c4 = st.columns(2)
            per_p = c3.selectbox("Personalidad (si el formato la usa)", ["La elige la app"] +
                                 [p["personalidad"] for p in ss.personalidades])
            fecha_p = c4.date_input("Fecha de publicación", date.today() + timedelta(days=1))
            if st.form_submit_button("Agregar idea", type="primary"):
                if not tema_p.strip():
                    st.error("Escribe el tema de la idea.")
                else:
                    porn = {p["personalidad"]: p for p in ss.personalidades}
                    persona = porn.get(per_p) or (ss.personalidades[rng.integers(len(ss.personalidades))]
                                                  if ss.personalidades else None)
                    nueva = guion(f_p, ss.formatos[f_p], tema_p, pilares[0], o_p, persona, rng)
                    nueva.update(puntual=True, fecha_fija=fecha_p)
                    ss.ideas.append(nueva)
                    st.success("Idea agregada. Revísala abajo, escríbele el guion y apruébala.")
    if st.button("Generar la semana", type="primary"):
        puntuales = [x for x in ss.ideas if x.get("puntual")]
        ss.ideas = puntuales + generar_semana(ss.df, ss.formatos, ss.personalidades, ss.marca["piezas"], angulos, pilares,
                                  ss.descuento, rng)
    personas = {p["personalidad"]: p for p in ss.personalidades}
    for tipo in TIPOS:
        grupo = [(i, x) for i, x in enumerate(ss.ideas) if x["tipo"] == tipo]
        if grupo:
            st.subheader(f"{tipo}s")
        for i, idea in grupo:
            titulo = f"{idea['formato']} · {idea['angulo']} · objetivo {idea['objetivo']}"
            if idea.get("referencia"):
                titulo += f" · ref. interna: {idea['referencia']}"
            if idea.get("puntual"):
                titulo = "📌 " + titulo + f" · {idea['fecha_fija']:%d/%m}"
            if idea.get("personalidad"):
                titulo += f" · {idea['personalidad']}"
            with st.expander(titulo, expanded=not idea["aprobada"]):
                texto = st.text_area("Guion (editable)", "\n".join(f"{t}: {d}" for t, d in idea["pasos"]),
                                     key=f"g{i}", height=180)
                for a in revisar_reglas(texto, ss.reglas, idea["permite_tecnica"]):
                    st.warning(a)
                if st.button("✨ Escribir guion completo con IA", key=f"ia{i}"):
                    clave = clave_ia()
                    if not clave:
                        st.info("La IA no está configurada en esta app. Usa «Instrucción para tu asistente de IA» "
                                "o configura la clave GEMINI_API_KEY en los Secrets de Streamlit.")
                    else:
                        try:
                            with st.spinner("Escribiendo el guion…"):
                                idea["pasos"] = [(t, d) for t, d in idea["pasos"]]
                                idea["guion_ia"] = escribir_guion_ia(
                                    {**idea, "pasos": [(t, d) for t, d in
                                                       [l.split(": ", 1) if ": " in l else ("", l)
                                                        for l in texto.splitlines()]]},
                                    ss.marca["nombre"] or "la marca", ss.reglas, clave)
                        except Exception as e:
                            st.error(f"No se pudo generar el guion: {e}")
                if idea.get("guion_ia"):
                    idea["guion_ia"] = st.text_area("Guion completo (IA, editable antes de aprobar)",
                                                    idea["guion_ia"], key=f"gi{i}", height=320)
                    for a in revisar_reglas(idea["guion_ia"], ss.reglas, idea["permite_tecnica"]):
                        st.warning(a)
                    st.caption("Revisa el guion antes de aprobarlo: la IA puede equivocarse. "
                               "No incluyas datos personales de clientes en las instrucciones.")
                c1, c2, c3, c4 = st.columns(4)
                if c1.button("Aprobar", key=f"a{i}"):
                    idea["aprobada"] = True
                    idea["texto"] = idea.get("guion_ia") or texto
                if c2.button("Regenerar", key=f"r{i}"):
                    persona = personas.get(idea.get("personalidad")) or (
                        ss.personalidades[rng.integers(len(ss.personalidades))] if ss.personalidades else None)
                    ss.ideas[i] = guion(idea["formato"], ss.formatos[idea["formato"]], rng.choice(angulos),
                                        rng.choice(pilares), idea["objetivo"], persona, rng)
                    st.rerun()
                c3.download_button("Instrucción para tu asistente de IA",
                                   prompt_ia(idea, ss.marca["nombre"] or "la marca", ss.reglas),
                                   f"prompt_{i+1}.txt", key=f"p{i}")
                if c4.button("Descartar", key=f"d{i}"):
                    ss.ideas.pop(i)
                    st.rerun()
                if idea["aprobada"]:
                    st.success("Aprobada")

# ---------------------------------------------------------------------------
# 7. Parrilla semanal
# ---------------------------------------------------------------------------
elif pantalla.startswith("7"):
    st.header("7. Parrilla semanal")
    aprobadas = [x for x in ss.ideas if x.get("aprobada")]
    if not aprobadas:
        st.warning("Aprueba al menos una idea en el paso 6.")
        st.stop()
    inicio = st.date_input("Semana que inicia el", date.today())
    filas = []
    for tipo in TIPOS:
        grupo = [x for x in aprobadas if x["tipo"] == tipo]
        for j, x in enumerate(grupo):
            dia = x.get("fecha_fija") or inicio + timedelta(days=(j * 7) // len(grupo))
            filas.append({"fecha": dia, "dia": DIAS[dia.weekday()], "tipo": tipo, "formato": x["formato"],
                          "angulo": x["angulo"], "referencia_interna": x.get("referencia", ""), "personalidad": x.get("personalidad", ""), "pilar": x["pilar"],
                          "objetivo": x["objetivo"], "horas": x["horas"], "gancho": x["gancho"]})
    ss.parrilla = pd.DataFrame(filas).sort_values(["fecha", "tipo"]).reset_index(drop=True)
    horas = ss.parrilla["horas"].sum()
    if horas > ss.marca["horas"]:
        st.error(f"La parrilla requiere {horas:g} h y tienes {ss.marca['horas']:g} h. "
                 "Descarta una idea en el paso 6 o ajusta tus horas en el paso 1.")
    else:
        st.success(f"La parrilla requiere {horas:g} h de {ss.marca['horas']:g} h disponibles.")
    st.dataframe(ss.parrilla, width="stretch")
    st.download_button("Descargar parrilla (CSV)", ss.parrilla.to_csv(index=False).encode("utf-8"),
                       "parrilla_semanal.csv", "text/csv")

# ---------------------------------------------------------------------------
# 8. Registrar resultados
# ---------------------------------------------------------------------------
elif pantalla.startswith("8"):
    st.header("8. Registrar resultados (a los 7 días de publicar)")
    with st.form("nuevo"):
        c1, c2, c3 = st.columns(3)
        titulo = c1.text_input("Título de la pieza")
        formato = c2.selectbox("Formato", list(ss.formatos))
        persona = c3.selectbox("Personalidad (si aplica)", ["Ninguna"] + [p["personalidad"] for p in ss.personalidades])
        persona = "" if persona == "Ninguna" else persona
        angulo = st.text_input("Ángulo o inspiración")
        c4, c5, c6, c7 = st.columns(4)
        esp = c4.number_input("Espectadores", 0, step=1)
        env = c5.number_input("Envíos", 0, step=1)
        inter = c6.number_input("Interacciones", 0, step=1)
        nos = c7.number_input("% interacciones de no seguidores", 0.0, 100.0)
        planeado = st.radio("¿Estaba en la parrilla?", ["Sí", "No (improvisado)"], horizontal=True)
        if st.form_submit_button("Guardar resultado", type="primary"):
            fila = {"fecha": date.today().isoformat(), "titulo": titulo, "formato": formato, "angulo": angulo,
                    "pilar": "", "vistas": 0, "espectadores": esp, "interacciones": inter,
                    "pct_inter_no_seguidores": nos, "envios": env, "guardados": 0, "comentarios": 0,
                    "planeado": "si" if planeado == "Sí" else "no",
                    "tipo": ss.formatos[formato].get("tipo", "Reel"), "personalidad": persona}
            ss.registro = pd.concat([ss.registro, pd.DataFrame([fila])], ignore_index=True)
            if ss.df is not None:
                ss.df = preparar(pd.concat([ss.df[COLUMNAS + OPCIONALES], pd.DataFrame([fila])], ignore_index=True),
                                 ss.formatos)
            st.success("Resultado guardado. El motor adaptativo ya lo tiene en cuenta.")
    planeadas = len(ss.parrilla) if ss.parrilla is not None else 0
    reg = preparar(ss.registro, ss.formatos) if len(ss.registro) else ss.registro.assign(
        planeado=pd.Series(dtype=bool))
    ind = indicadores_plan(reg, planeadas)
    c1, c2 = st.columns(2)
    c1.metric("Adherencia a la parrilla", f"{ind['adherencia']*100:.0f} %",
              help="Piezas publicadas según la parrilla ÷ piezas planeadas")
    c2.metric("Improvisación", f"{ind['improvisacion']*100:.0f} %",
              help="Piezas publicadas sin planificación ÷ total de piezas publicadas")
    if len(ss.registro):
        st.dataframe(ss.registro, width="stretch")
        base = ss.df[COLUMNAS + OPCIONALES] if ss.df is not None else ss.registro
        st.download_button("Descargar datos actualizados (CSV)", base.to_csv(index=False).encode("utf-8"),
                           "contenido_actualizado.csv", "text/csv")
