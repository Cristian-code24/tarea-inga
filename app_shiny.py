"""
=============================================================================
app.py — Exposición Cálculo III (UNJFSC)
Regresión Lineal Múltiple · Shiny for Python + Plotly
=============================================================================
"""

from __future__ import annotations

import pathlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from shiny import App, reactive, render, ui
from shinywidgets import output_widget, render_widget

# ─────────────────────────────────────────────────────────────
# 1. CARGA DE DATOS Y MODELO OLS
# ─────────────────────────────────────────────────────────────
HERE = pathlib.Path(__file__).parent
CSV  = HERE / "datos-modelo.csv"

df = pd.read_csv(CSV)
df.columns = df.columns.str.strip()
df = df.dropna().apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)

x1_data = df["x1_COD_TARIFA"].values
x2_data = df["x2_SUMINISTROS"].values
y_data  = df["y_PROMEDIO_CONSUMO"].values
n       = len(y_data)

X_mat         = np.column_stack([np.ones(n), x1_data, x2_data])
beta, _, _, _ = np.linalg.lstsq(X_mat, y_data, rcond=None)
b0, b1, b2    = float(beta[0]), float(beta[1]), float(beta[2])

y_pred    = X_mat @ beta
residuos  = y_data - y_pred
ss_res    = float(np.sum(residuos**2))
ss_tot    = float(np.sum((y_data - np.mean(y_data))**2))
r2        = 1.0 - ss_res / ss_tot
rmse      = float(np.sqrt(ss_res / n))

# Rangos para sliders
X1_MIN, X1_MAX = int(x1_data.min()), int(x1_data.max())
X2_MIN, X2_MAX = int(x2_data.min()), int(x2_data.max())
X1_DEF = int(np.median(x1_data))
X2_DEF = int(np.median(x2_data))

# Signo de coeficientes para mostrar
s1 = "+" if b1 >= 0 else "−"
s2 = "+" if b2 >= 0 else "−"
eq_str = f"ŷ = {b0:.4f}  {s1}  {abs(b1):.4f}·x₁  {s2}  {abs(b2):.6f}·x₂"

# ─────────────────────────────────────────────────────────────
# 2. PALETA DE COLORES
# ─────────────────────────────────────────────────────────────
C_DARK  = "#1a3a8f"
C_MID   = "#2563eb"
C_LIGHT = "#60a5fa"
C_BG    = "#f0f4ff"
C_WHITE = "#ffffff"
C_GRAY  = "#374151"
C_CARD  = "#e8edf8"

# ─────────────────────────────────────────────────────────────
# 3. CSS PERSONALIZADO
# ─────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
@import url(https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap);

* { box-sizing: border-box; }
body, .shiny-bound-input { font-family: 'Inter', sans-serif !important; }

/* App background */
body { background: linear-gradient(160deg, #f0f4ff 0%, #e8edf8 100%) !important; }

/* Sidebar */
.sidebar {
    background: white !important;
    border-right: 2px solid #bfdbfe !important;
    box-shadow: 4px 0 16px rgba(37,99,235,.08) !important;
    padding: 1.5rem 1.2rem !important;
}

/* Banner */
.app-banner {
    background: linear-gradient(135deg,#1a3a8f 0%,#2563eb 60%,#3b82f6 100%);
    border-radius: 14px; padding: 1.4rem 2rem; margin-bottom: 1.2rem;
    box-shadow: 0 6px 24px rgba(37,99,235,.28); color: white;
}
.app-banner h2 { margin:0; font-size:1.5rem; font-weight:700; letter-spacing:-.4px; }
.app-banner p  { margin:.25rem 0 0; font-size:.95rem; opacity:.85; font-weight:300; }

/* Prediction card */
.pred-card {
    background: linear-gradient(135deg,#eff6ff,#dbeafe);
    border: 2px solid #93c5fd; border-radius: 12px;
    padding: 1.2rem 1.4rem; margin-top: 1.2rem;
    box-shadow: 0 2px 12px rgba(37,99,235,.15);
    text-align: center;
}
.pred-card .pred-label { font-size:.78rem; color:#1e3a8a; font-weight:600;
    text-transform:uppercase; letter-spacing:.6px; }
.pred-card .pred-value { font-size:2.4rem; font-weight:700; color:#1e3a8a;
    margin-top:.2rem; }
.pred-card .pred-sub   { font-size:.8rem; color:#4b5563; margin-top:.15rem; }

/* Info box */
.ibox {
    background: white; border-radius: 12px; padding: 1.3rem 1.6rem;
    box-shadow: 0 2px 12px rgba(0,0,0,.06); border-top: 3px solid #2563eb;
    margin-bottom: 1rem; line-height: 1.8; color: #374151;
}
.ibox h4 { color:#1a3a8f; margin-top:0; font-size:1rem; font-weight:700; }
.ibox code { background:#eff6ff; padding:.1rem .4rem; border-radius:4px;
    font-size:.88rem; color:#1e3a8a; }
.ibox ul { padding-left:1.2rem; } .ibox li { margin-bottom:.45rem; }

/* Equation box */
.eq-box {
    background: linear-gradient(135deg,#1a3a8f,#2563eb);
    border-radius: 12px; padding: 1.2rem 1.8rem; text-align:center;
    color: white; font-size: 1.25rem; font-weight: 700;
    letter-spacing: .3px; margin: 1rem 0;
    box-shadow: 0 4px 16px rgba(37,99,235,.35);
}

/* Metric row */
.metric-row { display:flex; gap:.8rem; margin:1rem 0; flex-wrap:wrap; }
.metric-chip {
    flex:1; min-width:120px; background:white; border-radius:10px;
    padding:.8rem 1rem; border-left:4px solid #2563eb;
    box-shadow: 0 2px 8px rgba(0,0,0,.06);
}
.metric-chip .m-lbl { font-size:.72rem; color:#6b7280; font-weight:600;
    text-transform:uppercase; letter-spacing:.5px; }
.metric-chip .m-val { font-size:1.35rem; font-weight:700; color:#1e3a8a; }

/* Section title */
.sec-title {
    font-size:1.05rem; font-weight:700; color:#1e3a8a;
    border-bottom:2px solid #bfdbfe; padding-bottom:.35rem;
    margin: 1.4rem 0 .8rem;
}

/* Sidebar slider labels */
.control-label { color: #1a3a8f !important; font-weight:600 !important; font-size:.88rem !important; }
.irs--shiny .irs-bar { background:#2563eb !important; }
.irs--shiny .irs-handle { border-color:#2563eb !important; }
.irs--shiny .irs-from, .irs--shiny .irs-to, .irs--shiny .irs-single {
    background:#2563eb !important;
}

/* Nav tabs */
.nav-tabs .nav-link { color: #4b5563 !important; font-weight:600; }
.nav-tabs .nav-link.active {
    background: linear-gradient(135deg,#1a3a8f,#2563eb) !important;
    color: white !important; border-radius:8px 8px 0 0 !important;
}

/* Plotly chart card wrapper */
.chart-card {
    background: white; border-radius:14px; padding:.8rem;
    box-shadow: 0 4px 20px rgba(0,0,0,.08); margin-top:.8rem;
}

/* Footer */
.app-footer {
    text-align:center; color:#9ca3af; font-size:.75rem;
    margin-top:1.5rem; padding-top:.8rem; border-top:1px solid #e5e7eb;
}
</style>
"""

# ─────────────────────────────────────────────────────────────
# 4. UI
# ─────────────────────────────────────────────────────────────
app_ui = ui.page_sidebar(

    # ── Sidebar ──────────────────────────────────────────────
    ui.sidebar(
        ui.HTML("""
        <div style="text-align:center; margin-bottom:1.4rem;">
            <div style="font-size:2.2rem; margin-bottom:.3rem;">📐</div>
            <div style="font-size:.95rem; font-weight:700; color:#1a3a8f;">Cálculo III</div>
            <div style="font-size:.78rem; color:#6b7280; font-weight:400;">UNJFSC · Simulador OLS</div>
        </div>
        <hr style="border-color:#bfdbfe; margin:.8rem 0 1.2rem;">
        """),

        ui.h6("📊 Parámetros de Simulación", style="color:#1a3a8f;font-weight:700;margin-bottom:.8rem;"),

        ui.input_slider(
            "slider_x1", "x₁ — Código de Tarifa",
            min=X1_MIN, max=X1_MAX, value=X1_DEF, step=1,
        ),
        ui.HTML(f'<div style="font-size:.72rem;color:#6b7280;margin-top:-.4rem;margin-bottom:.8rem;">Rango real: [{X1_MIN}, {X1_MAX}]</div>'),

        ui.input_slider(
            "slider_x2", "x₂ — Suministros",
            min=X2_MIN, max=X2_MAX, value=X2_DEF, step=1,
        ),
        ui.HTML(f'<div style="font-size:.72rem;color:#6b7280;margin-top:-.4rem;margin-bottom:1rem;">Rango real: [{X2_MIN}, {X2_MAX}]</div>'),

        ui.hr(),

        # Predicción en sidebar
        ui.output_ui("pred_card_sidebar"),

        ui.hr(),

        ui.HTML(f"""
        <div style="font-size:.75rem;color:#9ca3af;line-height:1.6;">
            <b style="color:#4b5563;">Modelo OLS</b><br>
            n = {n} observaciones<br>
            R² = {r2:.4f}<br>
            RMSE = {rmse:.4f} kWh
        </div>
        """),

        width=300,
        style="background:white; border-right:2px solid #bfdbfe; padding:1.5rem 1.2rem;",
    ),

    # ── Panel principal ───────────────────────────────────────
    ui.HTML(CUSTOM_CSS),

    # Banner
    ui.HTML(f"""
    <div class="app-banner">
        <h2>📐 Cálculo III &mdash; Funciones de Varias Variables</h2>
        <p>Regresión Lineal Múltiple por Mínimos Cuadrados Ordinarios &nbsp;·&nbsp; UNJFSC</p>
    </div>
    """),

    # Tabs
    ui.navset_tab(

        # ── Tab 1: Teoría ─────────────────────────────────────
        ui.nav_panel(
            "📘 Teoría y Modelo",

            # Ecuación del plano
            ui.HTML(f'<div class="sec-title">Ecuación del Plano de Regresión</div>'),
            ui.HTML(f'<div class="eq-box">{eq_str}</div>'),

            # Métricas
            ui.HTML(f"""
            <div class="metric-row">
              <div class="metric-chip">
                <div class="m-lbl">β₀ Intercepto</div>
                <div class="m-val">{b0:.4f}</div>
              </div>
              <div class="metric-chip">
                <div class="m-lbl">β₁ x₁_COD_TARIFA</div>
                <div class="m-val">{b1:.4f}</div>
              </div>
              <div class="metric-chip">
                <div class="m-lbl">β₂ x₂_SUMINISTROS</div>
                <div class="m-val">{b2:.6f}</div>
              </div>
              <div class="metric-chip">
                <div class="m-lbl">R² Ajuste</div>
                <div class="m-val">{r2:.4f}</div>
              </div>
            </div>
            """),

            # Caja de teoría
            ui.HTML(f"""
            <div class="ibox">
              <h4>① Identificación de Variables y Dominio</h4>
              <ul>
                <li><b>y → y_PROMEDIO_CONSUMO</b>: consumo energético promedio (kWh) — <em>variable dependiente</em>.</li>
                <li><b>x₁ → x1_COD_TARIFA</b>: código de tarifa eléctrica del usuario — <em>variable independiente</em>.
                    Dominio observado: <code>[{X1_MIN}, {X1_MAX}]</code>.</li>
                <li><b>x₂ → x2_SUMINISTROS</b>: número de suministros activos — <em>variable independiente</em>.
                    Dominio observado: <code>[{X2_MIN}, {X2_MAX}]</code>.</li>
                <li>Conjunto de datos: <b>{n} observaciones</b> válidas tras limpieza de valores nulos.</li>
              </ul>

              <h4>② Aplicación de Mínimos Cuadrados Ordinarios (OLS)</h4>
              <ul>
                <li>Se modela la relación como un <b>plano en ℝ³</b>: &nbsp; ŷ = β₀ + β₁·x₁ + β₂·x₂</li>
                <li>Se minimiza la <b>función de error cuadrático</b>:
                    S(β) = Σᵢ (yᵢ − β₀ − β₁·x₁ᵢ − β₂·x₂ᵢ)²</li>
                <li>Derivando parcialmente e igualando a cero se obtiene el sistema de <b>ecuaciones normales</b>:
                    <code>(XᵀX)·β = Xᵀy</code>, resuelto mediante descomposición SVD con
                    <code>numpy.linalg.lstsq</code>.</li>
              </ul>

              <h4>③ Significado de los Coeficientes</h4>
              <ul>
                <li><b>β₀ = {b0:.4f}</b>: consumo base estimado cuando x₁ = 0 y x₂ = 0.</li>
                <li><b>β₁ = {b1:.4f}</b>: pendiente parcial ∂ŷ/∂x₁ — por cada unidad adicional de
                    código de tarifa, el consumo varía <b>{b1:.4f} kWh</b> (con x₂ fijo).</li>
                <li><b>β₂ = {b2:.6f}</b>: pendiente parcial ∂ŷ/∂x₂ — por cada suministro adicional,
                    el consumo varía <b>{b2:.6f} kWh</b> (con x₁ fijo).</li>
                <li><b>R² = {r2:.4f}</b>: el modelo explica el <b>{r2*100:.1f}%</b> de la variabilidad
                    total del consumo. RMSE = {rmse:.4f} kWh.</li>
              </ul>
            </div>
            """),

            # Predicción reactiva dinámica
            ui.HTML('<div class="sec-title">Predicción con los Valores del Simulador</div>'),
            ui.output_ui("pred_dinamica"),

            # Tabla de datos
            ui.HTML('<div class="sec-title">Tabla de Datos — Real vs. Predicho</div>'),
            ui.output_data_frame("tabla_datos"),
        ),

        # ── Tab 2: Curvas de Nivel ────────────────────────────
        ui.nav_panel(
            "🗺️ Curvas de Nivel",
            ui.HTML('<div class="sec-title">Mapa de Curvas de Nivel — Consumo Predicho</div>'),
            ui.HTML('<div class="chart-card">'),
            output_widget("plot_contour"),
            ui.HTML('</div>'),
            ui.HTML("""
            <div class="ibox" style="margin-top:.8rem;">
              <h4>📖 Cómo leer este gráfico</h4>
              <ul>
                <li>Cada <b>curva de nivel</b> une los puntos (x₁, x₂) donde el consumo predicho es constante.</li>
                <li><span style="color:#b91c1c;font-weight:600">Colores cálidos (rojo/naranja)</span>
                    → zonas de <b>mayor consumo</b> — la función "sube".</li>
                <li><span style="color:#1d4ed8;font-weight:600">Colores fríos (azul)</span>
                    → zonas de <b>menor consumo</b> — la función "baja".</li>
                <li>La <b>estrella ⭐ amarilla</b> muestra la posición del punto simulado con los sliders.</li>
                <li>Concepto de Cálculo III: función escalar f: ℝ² → ℝ; las curvas son conjuntos {(x₁,x₂) | f = k}.</li>
              </ul>
            </div>
            """),
        ),

        # ── Tab 3: Superficie 3D ──────────────────────────────
        ui.nav_panel(
            "🌐 Superficie 3D",
            ui.HTML('<div class="sec-title">Plano de Regresión en ℝ³ + Datos Reales</div>'),
            ui.HTML('<div class="chart-card">'),
            output_widget("plot_surface3d"),
            ui.HTML('</div>'),
            ui.HTML("""
            <div class="ibox" style="margin-top:.8rem;">
              <h4>📖 Interpretación de la Superficie 3D</h4>
              <ul>
                <li>El <b>plano azul translúcido</b> es la superficie ŷ = β₀ + β₁·x₁ + β₂·x₂ obtenida por OLS.</li>
                <li>Los <b>puntos esféricos</b> son las observaciones reales; el color indica el residuo (y − ŷ):
                    <span style="color:#16a34a;font-weight:600">verde</span> (sobreestimado) |
                    <span style="color:#dc2626;font-weight:600">rojo</span> (subestimado).</li>
                <li>Las <b>líneas punteadas</b> conectan cada punto real con su proyección en el plano — los residuos que OLS minimizó.</li>
                <li>La <b>estrella ⭐ amarilla</b> muestra la predicción del punto simulado por los sliders.</li>
                <li>Puedes <b>rotar e interactuar</b> arrastrando con el mouse.</li>
              </ul>
            </div>
            """),
        ),
    ),

    ui.HTML("""
    <div class="app-footer">
        Sustentación Cálculo III &mdash; UNJFSC &mdash; Regresión OLS &mdash;
        Shiny for Python + Plotly
    </div>
    """),

    title="Cálculo III — OLS",
    fillable=False,
)


# ─────────────────────────────────────────────────────────────
# 5. SERVER
# ─────────────────────────────────────────────────────────────
def server(input, output, session):

    # ── Predicción reactiva ───────────────────────────────────
    @reactive.calc
    def y_simulado():
        return float(b0 + b1 * input.slider_x1() + b2 * input.slider_x2())

    # ── Tarjeta de predicción en sidebar ─────────────────────
    @output
    @render.ui
    def pred_card_sidebar():
        yhat = y_simulado()
        return ui.HTML(f"""
        <div class="pred-card">
          <div class="pred-label">ŷ Consumo Simulado</div>
          <div class="pred-value">{yhat:.2f}</div>
          <div class="pred-sub">kWh &nbsp;·&nbsp;
              x₁={input.slider_x1()} &nbsp; x₂={input.slider_x2()}</div>
        </div>
        """)

    # ── Predicción reactiva en Tab 1 ─────────────────────────
    @output
    @render.ui
    def pred_dinamica():
        yhat  = y_simulado()
        xi1   = input.slider_x1()
        xi2   = input.slider_x2()
        return ui.HTML(f"""
        <div style="display:flex;gap:1rem;flex-wrap:wrap;margin-bottom:1rem;">
          <div class="metric-chip" style="border-left-color:#16a34a;flex:2;">
            <div class="m-lbl">Valores ingresados en Slider</div>
            <div style="font-size:1rem;color:#374151;margin-top:.2rem;">
                x₁ = <b>{xi1}</b> &nbsp;·&nbsp; x₂ = <b>{xi2}</b>
            </div>
          </div>
          <div class="metric-chip" style="border-left-color:#f59e0b;flex:2;">
            <div class="m-lbl">Consumo Predicho ŷ</div>
            <div class="m-val" style="color:#1e3a8a;font-size:1.6rem;">{yhat:.4f} kWh</div>
          </div>
          <div class="metric-chip" style="border-left-color:#2563eb;flex:3;">
            <div class="m-lbl">Sustitución en la Ecuación</div>
            <div style="font-size:.85rem;color:#374151;margin-top:.2rem;font-family:monospace;">
                ŷ = {b0:.4f} {s1} {abs(b1):.4f}·{xi1} {s2} {abs(b2):.6f}·{xi2}<br>
                ŷ = {b0:.4f} {s1} {abs(b1)*xi1:.4f} {s2} {abs(b2)*xi2:.6f}<br>
                <b>ŷ = {yhat:.4f} kWh</b>
            </div>
          </div>
        </div>
        """)

    # ── Tabla de datos ────────────────────────────────────────
    @output
    @render.data_frame
    def tabla_datos():
        d = df.copy()
        d["ŷ_PREDICHO"]   = np.round(y_pred, 4)
        d["Residuo (y−ŷ)"] = np.round(y_data - y_pred, 4)
        return render.DataGrid(d, width="100%")

    # ── Gráfico 2: Curvas de Nivel ────────────────────────────
    @output
    @render_widget
    def plot_contour():
        xi1  = input.slider_x1()
        xi2  = input.slider_x2()
        yhat = y_simulado()

        g1 = np.linspace(x1_data.min(), x1_data.max(), 250)
        g2 = np.linspace(x2_data.min(), x2_data.max(), 250)
        G1, G2 = np.meshgrid(g1, g2)
        Zg     = b0 + b1 * G1 + b2 * G2

        fig = go.Figure()

        # Contour heatmap
        fig.add_trace(go.Contour(
            x=g1, y=g2, z=Zg,
            colorscale="RdYlBu_r",
            contours=dict(
                coloring="heatmap",
                showlabels=True,
                labelfont=dict(size=10, color="white", family="Inter"),
            ),
            colorbar=dict(
                title=dict(text="ŷ Consumo<br>(kWh)", side="right"),
                tickfont=dict(family="Inter", size=10),
                len=0.85,
            ),
            hovertemplate="x₁: %{x:.0f}<br>x₂: %{y:.0f}<br>ŷ: %{z:.2f} kWh<extra></extra>",
        ))

        # Datos reales
        fig.add_trace(go.Scatter(
            x=x1_data, y=x2_data, mode="markers+text",
            marker=dict(size=11, color=y_data, colorscale="RdYlBu_r",
                        line=dict(color="white", width=1.5)),
            text=[f"{v:.0f}" for v in y_data],
            textposition="top center",
            textfont=dict(size=8, color="#1e3a8a"),
            name="Datos Reales",
            hovertemplate="Dato real<br>x₁=%{x}<br>x₂=%{y}<br>y=%{marker.color:.2f}<extra></extra>",
        ))

        # Punto simulado ⭐
        fig.add_trace(go.Scatter(
            x=[xi1], y=[xi2], mode="markers+text",
            marker=dict(size=18, color="gold", symbol="star",
                        line=dict(color="#b45309", width=2)),
            text=[f"⭐ ŷ={yhat:.1f}"],
            textposition="top center",
            textfont=dict(size=11, color="#92400e", family="Inter"),
            name=f"Simulado (x₁={xi1}, x₂={xi2})",
            hovertemplate=f"Punto simulado<br>x₁={xi1}<br>x₂={xi2}<br>ŷ={yhat:.4f} kWh<extra></extra>",
        ))

        fig.update_layout(
            height=520,
            xaxis=dict(title="x₁ — Código de Tarifa",
                       title_font=dict(family="Inter", size=12)),
            yaxis=dict(title="x₂ — Suministros",
                       title_font=dict(family="Inter", size=12)),
            paper_bgcolor="white", plot_bgcolor="#f8fafc",
            font=dict(family="Inter"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="right", x=1, font=dict(size=11)),
            margin=dict(l=60, r=20, t=20, b=60),
        )
        return fig

    # ── Gráfico 3: Superficie 3D ──────────────────────────────
    @output
    @render_widget
    def plot_surface3d():
        xi1  = input.slider_x1()
        xi2  = input.slider_x2()
        yhat = y_simulado()

        s1v = np.linspace(x1_data.min() - 2, x1_data.max() + 2, 60)
        s2v = np.linspace(x2_data.min() - 200, x2_data.max() + 200, 60)
        S1, S2 = np.meshgrid(s1v, s2v)
        Zs     = b0 + b1 * S1 + b2 * S2

        fig = go.Figure()

        # Plano de regresión
        fig.add_trace(go.Surface(
            x=S1, y=S2, z=Zs,
            colorscale="Blues", opacity=0.68, showscale=True,
            colorbar=dict(title=dict(text="ŷ Plano<br>OLS", side="right"),
                          len=0.55, x=1.02,
                          tickfont=dict(family="Inter", size=9)),
            hovertemplate="x₁:%{x:.1f}<br>x₂:%{y:.0f}<br>ŷ:%{z:.2f}<extra>Plano OLS</extra>",
            name="Plano de Regresión",
        ))

        # Puntos reales coloreados por residuo
        res = y_data - y_pred
        fig.add_trace(go.Scatter3d(
            x=x1_data, y=x2_data, z=y_data, mode="markers",
            marker=dict(
                size=7, color=res, colorscale="RdYlGn",
                cmin=-float(np.max(np.abs(res))),
                cmax=float(np.max(np.abs(res))),
                line=dict(color="white", width=0.8),
                colorbar=dict(title=dict(text="Residuo<br>(y−ŷ)", side="right"),
                              x=1.12, len=0.45, y=0.25,
                              tickfont=dict(family="Inter", size=9)),
            ),
            name="Datos Reales",
            hovertemplate="x₁:%{x}<br>x₂:%{y}<br>y real:%{z:.2f}<extra>Dato Real</extra>",
        ))

        # Líneas de residuo (punto → plano)
        for xi1r, xi2r, yir, ypir in zip(x1_data, x2_data, y_data, y_pred):
            fig.add_trace(go.Scatter3d(
                x=[xi1r, xi1r], y=[xi2r, xi2r], z=[yir, ypir], mode="lines",
                line=dict(color="rgba(180,0,0,0.35)", width=1.5, dash="dot"),
                showlegend=False, hoverinfo="skip",
            ))

        # Punto simulado ⭐
        fig.add_trace(go.Scatter3d(
            x=[xi1], y=[xi2], z=[yhat], mode="markers+text",
            marker=dict(size=14, color="gold", symbol="diamond",
                        line=dict(color="#b45309", width=2)),
            text=[f"ŷ={yhat:.1f}"],
            textposition="top center",
            textfont=dict(size=10, color="#92400e"),
            name=f"Simulado (x₁={xi1}, x₂={xi2})",
            hovertemplate=f"Punto simulado<br>x₁={xi1}<br>x₂={xi2}<br>ŷ={yhat:.4f}<extra></extra>",
        ))

        fig.update_layout(
            height=600,
            scene=dict(
                xaxis=dict(title=dict(text="x₁ Cód.Tarifa",
                                  font=dict(family="Inter", size=10)),
                       backgroundcolor="#f0f4ff", gridcolor="#cbd5e1",
                       showbackground=True),
                yaxis=dict(title=dict(text="x₂ Suministros",
                                  font=dict(family="Inter", size=10)),
                       backgroundcolor="#f0f4ff", gridcolor="#cbd5e1",
                       showbackground=True),
                zaxis=dict(title=dict(text="y Consumo (kWh)",
                                  font=dict(family="Inter", size=10)),
                       backgroundcolor="#f0f4ff", gridcolor="#cbd5e1",
                       showbackground=True),
                camera=dict(eye=dict(x=1.7, y=-1.7, z=1.1)),
                aspectmode="auto",
            ),
            paper_bgcolor="white",
            font=dict(family="Inter"),
            legend=dict(orientation="h", yanchor="top", y=-0.04,
                        xanchor="center", x=0.5, font=dict(size=10)),
            margin=dict(l=0, r=0, t=10, b=10),
        )
        return fig


# ─────────────────────────────────────────────────────────────
# 6. ARRANQUE
# ─────────────────────────────────────────────────────────────
app = App(app_ui, server)
