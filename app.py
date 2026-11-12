"""
Parrilla Adaptativa - PMV
Planificación adaptativa de contenido con IA para marcas artesanales emergentes.
Caso piloto: Carolina Atelier.

Ejecutar en local:  streamlit run app.py
"""
import io
from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Parrilla Adaptativa", page_icon="🗓️", layout="wide")

# ---------------------------------------------------------------------------
# Configuración base
# ---------------------------------------------------------------------------
COLUMNAS = [
    "fecha", "titulo", "formato", "angulo", "pilar", "vistas", "espectadores",
    "interacciones", "pct_inter_no_seguidores", "envios", "guardados",
    "comentarios", "planeado",
]

# Biblioteca de formatos: estructura del guion y horas estimadas de producción
FORMATOS = {
    "Primer plano de producto": {
        "horas": 1.0,
        "estructura": [
            ("0-3 s", "Gancho visual: detalle de la pieza en movimiento + texto: «{gancho}»"),
            ("3-10 s", "Recorrido lento por la pieza mostrando {angulo}"),
            ("10-15 s", "La pieza puesta, en un contexto cotidiano"),
            ("Cierre", "{cta}"),
        ],
    },
    "Proceso (detrás de cámaras)": {
        "horas": 2.0,
        "estructura": [
            ("0-3 s", "Resultado final en pantalla + texto: «{gancho}»"),
            ("3-12 s", "Tres momentos del proceso sin revelar la técnica, enfocados en {angulo}"),
            ("12-18 s", "Revelación de la pieza terminada"),
            ("Cierre", "{cta}"),
        ],
    },
    "Transformación (antes y después)": {
        "horas": 1.5,
        "estructura": [
            ("0-3 s", "Antes: look sin accesorio + texto: «{gancho}»"),
            ("3-6 s", "Transición al ritmo del audio"),
            ("6-12 s", "Después: el mismo look con la pieza, mostrando {angulo}"),
            ("Cierre", "{cta}"),
        ],
    },
    "Tutorial": {
        "horas": 1.5,
        "estructura": [
            ("0-3 s", "Resultado final primero + texto: «{gancho}»"),
            ("3-15 s", "Pasos numerados (máximo 3) aplicados a {angulo}"),
            ("15-20 s", "Resultado desde dos ángulos de cámara"),
            ("Cierre", "{cta}"),
        ],
    },
    "Serie por episodios": {
        "horas": 2.0,
        "estructura": [
            ("0-3 s", "Marca fija de la serie (mismo inicio en cada episodio) + «{gancho}»"),
            ("3-12 s", "Historia del episodio: {angulo}"),
            ("12-18 s", "La pieza como desenlace del episodio"),
            ("Cierre", "Adelanto del próximo episodio. {cta}"),
        ],
    },
    "Prueba en la vida real": {
        "horas": 1.5,
        "estructura": [
            ("0-3 s", "Reto en pantalla: «{gancho}»"),
            ("3-15 s", "La pieza durante el día real: {angulo}"),
            ("15-20 s", "Estado de la pieza al final del día"),
            ("Cierre", "{cta}"),
        ],
    },
}

GANCHOS = [
    "Nadie te cuenta esto sobre {angulo}",
    "Lo que pasa cuando {angulo}",
    "¿Te atreves con {angulo}?",
    "Probamos {angulo} para que tú no tengas que hacerlo",
    "Esto no es lo que esperabas de {angulo}",
]

CTA = {
    "Alcance": "Llamado implícito a compartir: «Envíaselo a quien necesita ver esto»",
    "Comunidad": "Pregunta abierta para comentar: «¿Tú cuál elegirías?»",
    "Conversión": "Invitación suave: «La tuya se diseña en la web, enlace en el perfil»",
}

REGLAS_DEFECTO = {
    "alta costura": "alta moda",
    "la clienta": "el cliente",
    "las clientas": "los clientes",
}

DIAS = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]


# ---------------------------------------------------------------------------
# Lógica (independiente de la interfaz)
# ---------------------------------------------------------------------------
def validar_datos(df: pd.DataFrame) -> list:
    faltan = [c for c in COLUMNAS if c not in df.columns]
    return faltan


def preparar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce", dayfirst=True)
    num = ["vistas", "espectadores", "interacciones", "pct_inter_no_seguidores",
           "envios", "guardados", "comentarios"]
    for c in num:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df["planeado"] = df["planeado"].astype(str).str.strip().str.lower().isin(["si", "sí", "1", "true"])
    df = df.sort_values("fecha").reset_index(drop=True)
    esp = df["espectadores"].replace(0, np.nan)
    df["tasa_envios"] = (df["envios"] / esp).fillna(0)
    df["tasa_comentarios"] = (df["comentarios"] / esp).fillna(0)
    return df


def diagnostico(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("formato").agg(
        reels=("titulo", "count"),
        espectadores=("espectadores", "sum"),
        envios=("envios", "sum"),
        inter_no_seg=("pct_inter_no_seguidores", "mean"),
        comentarios=("comentarios", "sum"),
    )
    g["tasa_envios_%"] = (g["envios"] / g["espectadores"].replace(0, np.nan) * 100).round(2)
    g["inter_no_seguidores_%"] = g["inter_no_seg"].round(1)
    g["tasa_comentarios_%"] = (g["comentarios"] / g["espectadores"].replace(0, np.nan) * 100).round(2)
    g["confiabilidad"] = np.where(g["reels"] >= 5, "Suficiente", "Pocos datos")
    return g[["reels", "tasa_envios_%", "inter_no_seguidores_%", "tasa_comentarios_%", "confiabilidad"]] \
        .sort_values("tasa_envios_%", ascending=False)


def posteriores(df: pd.DataFrame, descuento: float) -> pd.DataFrame:
    """Modelo Beta-Binomial por formato sobre la tasa de envíos.
    Los reels recientes pesan más: peso = descuento ** (posición desde el más reciente)."""
    df = df.sort_values("fecha").reset_index(drop=True)
    edad = (len(df) - 1) - np.arange(len(df))
    w = descuento ** edad
    filas = []
    for f in FORMATOS:
        m = df["formato"] == f
        exitos = float((w[m] * df.loc[m, "envios"]).sum())
        fracasos = float((w[m] * (df.loc[m, "espectadores"] - df.loc[m, "envios"]).clip(lower=0)).sum())
        a, b = 1 + exitos, 1 + fracasos
        filas.append({"formato": f, "alpha": a, "beta": b, "media_%": a / (a + b) * 100,
                      "reels": int(m.sum())})
    return pd.DataFrame(filas).set_index("formato")


def thompson(post: pd.DataFrame, n: int, rng) -> list:
    """Elige n formatos: en cada espacio muestrea la tasa probable de cada formato y gana la mayor.
    Así explota lo que funciona sin dejar de explorar formatos con pocos datos."""
    elegidos = []
    for _ in range(n):
        muestras = {f: rng.beta(r.alpha, r.beta) for f, r in post.iterrows()}
        elegidos.append(max(muestras, key=muestras.get))
    return elegidos


def saturacion(muestra: pd.DataFrame, candidatos: list) -> pd.DataFrame:
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


def revisar_reglas(texto: str, reglas: dict) -> list:
    t = texto.lower()
    return [f"Usa «{bien}» en lugar de «{mal}»" for mal, bien in reglas.items() if mal in t]


def guion(formato, angulo, pilar, objetivo, rng) -> dict:
    gancho = rng.choice(GANCHOS).format(angulo=angulo.lower())
    pasos = [(t, d.format(gancho=gancho, angulo=angulo.lower(), cta=CTA[objetivo]))
             for t, d in FORMATOS[formato]["estructura"]]
    return {"formato": formato, "angulo": angulo, "pilar": pilar, "objetivo": objetivo,
            "gancho": gancho, "pasos": pasos, "horas": FORMATOS[formato]["horas"]}


def prompt_ia(idea: dict, marca: str, reglas: dict) -> str:
    pasos = "\n".join(f"- {t}: {d}" for t, d in idea["pasos"])
    reglas_txt = "; ".join(f"di «{b}» y no «{m}»" for m, b in reglas.items())
    return (
        f"Eres guionista de reels para {marca}, una marca artesanal.\n"
        f"Mejora este guion de formato «{idea['formato']}» sobre el ángulo «{idea['angulo']}» "
        f"(pilar: {idea['pilar']}; objetivo: {idea['objetivo']}).\n"
        f"Mantén la estructura, haz el gancho más específico y conserva un llamado a la acción implícito.\n"
        f"Reglas de marca: {reglas_txt}. No reveles técnicas de fabricación.\n\n{pasos}"
    )


def indicadores_plan(reg: pd.DataFrame, planeadas: int) -> dict:
    publicadas = len(reg)
    segun_plan = int(reg["planeado"].sum()) if publicadas else 0
    return {
        "adherencia": segun_plan / planeadas if planeadas else 0.0,
        "improvisacion": (publicadas - segun_plan) / publicadas if publicadas else 0.0,
        "publicadas": publicadas, "segun_plan": segun_plan,
    }


# ---------------------------------------------------------------------------
# Estado de la sesión (los datos solo viven en esta sesión; no se guardan en servidor)
# ---------------------------------------------------------------------------
ss = st.session_state
ss.setdefault("acepta", False)
ss.setdefault("df", None)
ss.setdefault("marca", {"nombre": "", "pilares": [], "horas": 4.0, "piezas": 3,
                        "reglas": dict(REGLAS_DEFECTO)})
ss.setdefault("muestra", pd.DataFrame({"angulo": ["5 peinados con diadema", "Get ready with me"],
                                       "cantidad": [12, 8]}))
ss.setdefault("candidatos", ["Diadema que sobrevive a un martes", "Un outfit, tres personas",
                             "Inspirada en la naturaleza colombiana"])
ss.setdefault("ideas", [])
ss.setdefault("parrilla", None)
ss.setdefault("registro", pd.DataFrame(columns=COLUMNAS))
ss.setdefault("descuento", 0.9)
rng = np.random.default_rng()

# ---------------------------------------------------------------------------
# Pantalla 0: bienvenida y autorización
# ---------------------------------------------------------------------------
st.sidebar.title("🗓️ Parrilla Adaptativa")
st.sidebar.caption("PMV · Caso piloto Carolina Atelier")

if not ss.acepta:
    st.title("Planifica tu contenido con tus propios datos")
    st.write("Esta aplicación analiza el desempeño de tus reels, detecta ángulos poco explotados "
             "y arma una parrilla semanal con ideas y guiones que se ajusta con cada resultado.")
    st.info("**Tratamiento de datos (Ley 1581 de 2012).** Los datos que cargues se procesan solo "
            "durante esta sesión y no se almacenan en el servidor. La aplicación nunca pide "
            "contraseñas. Puedes descargar tus resultados y borrar todo al cerrar la pestaña.")
    if st.checkbox("Autorizo el tratamiento de mis datos para este análisis"):
        if st.button("Comenzar", type="primary"):
            ss.acepta = True
            st.rerun()
    st.stop()

pantalla = st.sidebar.radio("Recorrido", [
    "1. Perfil de marca", "2. Cargar datos", "3. Diagnóstico", "4. Radar de saturación",
    "5. Ideas y guiones", "6. Parrilla semanal", "7. Registrar resultados"])

if ss.df is not None:
    st.sidebar.success(f"{len(ss.df)} reels cargados")
else:
    st.sidebar.warning("Sin datos cargados")

# ---------------------------------------------------------------------------
# 1. Perfil de marca
# ---------------------------------------------------------------------------
if pantalla.startswith("1"):
    st.header("1. Perfil de marca")
    m = ss.marca
    m["nombre"] = st.text_input("Nombre de la marca", m["nombre"] or "Carolina Atelier")
    pil = st.text_input("Pilares de marca (separados por coma)",
                        ", ".join(m["pilares"]) or "Hecho a mano, Uso cotidiano, Co-creación")
    m["pilares"] = [p.strip() for p in pil.split(",") if p.strip()]
    c1, c2 = st.columns(2)
    m["horas"] = c1.number_input("Horas disponibles por semana para contenido", 0.5, 40.0, float(m["horas"]), 0.5)
    m["piezas"] = c2.number_input("Piezas por semana", 1, 14, int(m["piezas"]))
    st.subheader("Reglas de lenguaje de la marca")
    reglas_df = pd.DataFrame({"evitar": list(m["reglas"].keys()), "usar": list(m["reglas"].values())})
    reglas_df = st.data_editor(reglas_df, num_rows="dynamic", width="stretch")
    m["reglas"] = {str(r.evitar).lower(): str(r.usar) for r in reglas_df.itertuples() if str(r.evitar).strip()}
    st.success("Perfil guardado en la sesión.")

# ---------------------------------------------------------------------------
# 2. Cargar datos
# ---------------------------------------------------------------------------
elif pantalla.startswith("2"):
    st.header("2. Cargar datos de los reels")
    plantilla = pd.DataFrame(columns=COLUMNAS).to_csv(index=False).encode("utf-8")
    st.download_button("Descargar plantilla CSV", plantilla, "plantilla_reels.csv", "text/csv")
    origen = st.radio("¿Cómo quieres cargar los datos?",
                      ["Subir mi CSV", "Usar datos de ejemplo (ficticios)"], horizontal=True)
    df = None
    if origen == "Subir mi CSV":
        archivo = st.file_uploader("Archivo CSV con las columnas de la plantilla", type="csv")
        if archivo:
            df = pd.read_csv(archivo)
    else:
        try:
            df = pd.read_csv("datos_ejemplo.csv")
            st.caption("Datos inventados solo para demostrar el funcionamiento. No son de ninguna marca real.")
        except FileNotFoundError:
            st.error("No se encontró datos_ejemplo.csv junto a app.py.")
    if df is not None:
        faltan = validar_datos(df)
        if faltan:
            st.error(f"Faltan columnas: {', '.join(faltan)}. Revisa la plantilla.")
        else:
            ss.df = preparar(df)
            st.success(f"{len(ss.df)} reels cargados correctamente.")
            st.dataframe(ss.df[COLUMNAS], width="stretch")

# ---------------------------------------------------------------------------
# 3. Diagnóstico
# ---------------------------------------------------------------------------
elif pantalla.startswith("3"):
    st.header("3. Diagnóstico por formato")
    if ss.df is None:
        st.warning("Primero carga tus datos en el paso 2.")
        st.stop()
    d = diagnostico(ss.df)
    c1, c2, c3 = st.columns(3)
    c1.metric("Reels analizados", len(ss.df))
    tot = ss.df["envios"].sum() / max(ss.df["espectadores"].sum(), 1) * 100
    c2.metric("Tasa de envíos global", f"{tot:.2f} %")
    c3.metric("Interacción de no seguidores (promedio)", f"{ss.df['pct_inter_no_seguidores'].mean():.1f} %")
    st.dataframe(d, width="stretch")
    st.bar_chart(d["tasa_envios_%"])
    if (d["confiabilidad"] == "Pocos datos").any():
        st.info("Los formatos con menos de 5 reels tienen resultados poco confiables: "
                "el motor adaptativo los seguirá explorando.")
    st.subheader("Motor adaptativo")
    ss.descuento = st.slider("Peso de los datos antiguos (1 = todos pesan igual; menor = se adapta más rápido)",
                             0.70, 1.00, float(ss.descuento), 0.01)
    post = posteriores(ss.df, ss.descuento)
    st.dataframe(post[["reels", "media_%"]].rename(columns={"media_%": "tasa_envios_esperada_%"}).round(2),
                 width="stretch")

# ---------------------------------------------------------------------------
# 4. Radar de saturación
# ---------------------------------------------------------------------------
elif pantalla.startswith("4"):
    st.header("4. Radar de saturación")
    st.write("Revisa en Instagram los reels más visibles de una temática y anota cuántas veces "
             "aparece cada ángulo. Luego escribe los ángulos que tu marca podría usar.")
    st.text_input("Temática analizada", "Accesorios para el cabello")
    ss.muestra = st.data_editor(ss.muestra, num_rows="dynamic", width="stretch",
                                column_config={"cantidad": st.column_config.NumberColumn(min_value=0, step=1)})
    cand = st.text_area("Ángulos propios candidatos (uno por línea)", "\n".join(ss.candidatos))
    ss.candidatos = [c.strip() for c in cand.splitlines() if c.strip()]
    sat = saturacion(ss.muestra.dropna(), ss.candidatos)
    st.dataframe(sat, width="stretch")
    bajos = sat[(sat["nivel"] == "Baja") & sat["es_propio"]]["angulo"].tolist()
    st.success(f"Ángulos de baja saturación disponibles: {', '.join(bajos) if bajos else 'ninguno'}")

# ---------------------------------------------------------------------------
# 5. Ideas y guiones
# ---------------------------------------------------------------------------
elif pantalla.startswith("5"):
    st.header("5. Ideas y guiones")
    if ss.df is None:
        st.warning("Primero carga tus datos en el paso 2.")
        st.stop()
    sat = saturacion(ss.muestra.dropna(), ss.candidatos)
    bajos = sat[(sat["nivel"] != "Alta") & sat["es_propio"]]["angulo"].tolist() or ss.candidatos
    pilares = ss.marca["pilares"] or ["Marca"]
    if st.button("Generar ideas para la semana", type="primary"):
        post = posteriores(ss.df, ss.descuento)
        formatos = thompson(post, int(ss.marca["piezas"]), rng)
        objetivos = (["Alcance", "Comunidad", "Conversión"] * 5)[: len(formatos)]
        ss.ideas = [dict(guion(f, rng.choice(bajos), rng.choice(pilares), o, rng), aprobada=False)
                    for f, o in zip(formatos, objetivos)]
    for i, idea in enumerate(ss.ideas):
        with st.expander(f"Idea {i+1}: {idea['formato']} · {idea['angulo']} · objetivo {idea['objetivo']}",
                         expanded=not idea["aprobada"]):
            texto = st.text_area("Guion (editable)", "\n".join(f"{t}: {d}" for t, d in idea["pasos"]),
                                 key=f"g{i}", height=150)
            alertas = revisar_reglas(texto, ss.marca["reglas"])
            for a in alertas:
                st.warning(a)
            c1, c2, c3, c4 = st.columns(4)
            if c4.button("Descartar", key=f"d{i}"):
                ss.ideas.pop(i)
                st.rerun()
            if c1.button("Aprobar", key=f"a{i}"):
                idea["aprobada"] = True
                idea["texto"] = texto
            if c2.button("Regenerar", key=f"r{i}"):
                ss.ideas[i] = dict(guion(idea["formato"], rng.choice(bajos), rng.choice(pilares),
                                         idea["objetivo"], rng), aprobada=False)
                st.rerun()
            c3.download_button("Instrucción para tu asistente de IA",
                               prompt_ia(idea, ss.marca["nombre"] or "la marca", ss.marca["reglas"]),
                               f"prompt_idea_{i+1}.txt", key=f"p{i}")
            if idea["aprobada"]:
                st.success("Aprobada")

# ---------------------------------------------------------------------------
# 6. Parrilla semanal
# ---------------------------------------------------------------------------
elif pantalla.startswith("6"):
    st.header("6. Parrilla semanal")
    aprobadas = [x for x in ss.ideas if x.get("aprobada")]
    if not aprobadas:
        st.warning("Aprueba al menos una idea en el paso 5.")
        st.stop()
    inicio = st.date_input("Semana que inicia el", date.today())
    paso = max(1, 7 // len(aprobadas))
    filas = [{"fecha": inicio + timedelta(days=i * paso), "dia": DIAS[(inicio + timedelta(days=i * paso)).weekday()],
              "formato": x["formato"], "angulo": x["angulo"], "pilar": x["pilar"],
              "objetivo": x["objetivo"], "horas": x["horas"], "gancho": x["gancho"]}
             for i, x in enumerate(aprobadas)]
    ss.parrilla = pd.DataFrame(filas)
    horas = ss.parrilla["horas"].sum()
    if horas > ss.marca["horas"]:
        st.error(f"La parrilla requiere {horas} h y tienes {ss.marca['horas']} h. "
                 "Quita una pieza en el paso 5 o aumenta tus horas en el paso 1.")
    else:
        st.success(f"La parrilla requiere {horas} h de {ss.marca['horas']} h disponibles.")
    st.dataframe(ss.parrilla, width="stretch")
    st.download_button("Descargar parrilla (CSV)", ss.parrilla.to_csv(index=False).encode("utf-8"),
                       "parrilla_semanal.csv", "text/csv")

# ---------------------------------------------------------------------------
# 7. Registrar resultados
# ---------------------------------------------------------------------------
elif pantalla.startswith("7"):
    st.header("7. Registrar resultados (a los 7 días de publicar)")
    with st.form("nuevo"):
        c1, c2, c3 = st.columns(3)
        titulo = c1.text_input("Título del reel")
        formato = c2.selectbox("Formato", list(FORMATOS))
        angulo = c3.text_input("Ángulo")
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
                    "planeado": "si" if planeado == "Sí" else "no"}
            ss.registro = pd.concat([ss.registro, pd.DataFrame([fila])], ignore_index=True)
            if ss.df is not None:
                ss.df = preparar(pd.concat([ss.df[COLUMNAS], pd.DataFrame([fila])], ignore_index=True))
            st.success("Resultado guardado. El motor adaptativo ya lo tiene en cuenta.")
    planeadas = len(ss.parrilla) if ss.parrilla is not None else 0
    reg = preparar(ss.registro) if len(ss.registro) else ss.registro.assign(planeado=pd.Series(dtype=bool))
    ind = indicadores_plan(reg, planeadas)
    c1, c2 = st.columns(2)
    c1.metric("Adherencia a la parrilla", f"{ind['adherencia']*100:.0f} %",
              help="Piezas publicadas según la parrilla ÷ piezas planeadas")
    c2.metric("Improvisación", f"{ind['improvisacion']*100:.0f} %",
              help="Piezas publicadas sin planificación ÷ total de piezas publicadas")
    if len(ss.registro):
        st.dataframe(ss.registro, width="stretch")
        st.download_button("Descargar datos actualizados (CSV)",
                           ss.df[COLUMNAS].to_csv(index=False).encode("utf-8") if ss.df is not None
                           else ss.registro.to_csv(index=False).encode("utf-8"),
                           "reels_actualizado.csv", "text/csv")
