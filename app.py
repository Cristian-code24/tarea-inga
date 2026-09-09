"""
=============================================================================
app.py — Exposición Cálculo III · Carlos Inga (UNJFSC)
Regresión OLS · Shiny for Python · Plotly · MathJax LaTeX
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
# 1. DATOS Y MODELO OLS
# ─────────────────────────────────────────────────────────────
HERE = pathlib.Path(__file__).parent
df   = pd.read_csv(HERE / "datos-modelo.csv")
df.columns = df.columns.str.strip()
df   = df.dropna().apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)

# Adaptación a la notación de la clase de Cálculo III (x, y, z)
x_data = df["x1_COD_TARIFA"].values.astype(float)
y_data = df["x2_SUMINISTROS"].values.astype(float)
z_data = df["y_PROMEDIO_CONSUMO"].values.astype(float) # Variable dependiente Z
n       = len(z_data)

X_mat         = np.column_stack([np.ones(n), x_data, y_data])
beta, _, _, _ = np.linalg.lstsq(X_mat, z_data, rcond=None)
b0, b1, b2    = float(beta[0]), float(beta[1]), float(beta[2])

z_pred   = X_mat @ beta
residuos = z_data - z_pred
ss_res   = float(np.sum(residuos**2))
ss_tot   = float(np.sum((z_data - np.mean(z_data))**2))
r2       = 1.0 - ss_res / ss_tot
rmse     = float(np.sqrt(ss_res / n))

X_MIN, X_MAX = int(x_data.min()), int(x_data.max())
Y_MIN, Y_MAX = int(y_data.min()), int(y_data.max())
X_DEF = int(np.median(x_data))
Y_DEF = int(np.median(y_data))

# Signos para display
s1_latex = "+" if b1 >= 0 else "-"
s2_latex = "+" if b2 >= 0 else "-"

# ─────────────────────────────────────────────────────────────
# 2. MATHJAX + CSS
# ─────────────────────────────────────────────────────────────
MATHJAX_SCRIPT = """
<script>
window.MathJax = {
  tex: {
    inlineMath: [['\\\\(', '\\\\)']],
    displayMath: [['\\\\[', '\\\\]']],
    processEscapes: true
  },
  options: {
    skipHtmlTags: ['script','noscript','style','textarea','pre']
  }
};
</script>
<script id="MathJax-script" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
</script>
"""

CSS = """
<style>
@import url(https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=STIX+Two+Math&display=swap);
*, body { font-family: 'Inter', sans-serif !important; }
body { background: linear-gradient(160deg,#f0f4ff 0%,#e8edf8 100%) !important; }

/* MathJax override para que use fuente matemática */
.MathJax { font-family: 'STIX Two Math', serif !important; }

.banner {
    background: linear-gradient(135deg,#0f2557 0%,#1a3a8f 40%,#2563eb 100%);
    border-radius: 14px; padding: 1.3rem 2rem; margin-bottom: 1.1rem;
    box-shadow: 0 6px 24px rgba(37,99,235,.3); color: white;
}
.banner h2 { margin:0; font-size:1.45rem; font-weight:700; letter-spacing:-.4px; }
.banner p  { margin:.2rem 0 0; font-size:.9rem; opacity:.85; font-weight:300; }

.pred-box {
    background: linear-gradient(135deg,#0f2557,#2563eb);
    border-radius: 12px; padding: 1rem 1.2rem; margin-top: 1rem;
    color: white; text-align: center;
    box-shadow: 0 4px 14px rgba(37,99,235,.35);
}
.pred-box .lbl { font-size:.72rem; font-weight:600; text-transform:uppercase;
                 letter-spacing:.6px; opacity:.8; }
.pred-box .val { font-size:2.1rem; font-weight:700; margin:.15rem 0; }
.pred-box .sub { font-size:.78rem; opacity:.75; }

/* Caja de fase académica */
.fase-box {
    background: white; border-radius: 12px; padding: 1.4rem 1.6rem;
    box-shadow: 0 2px 12px rgba(0,0,0,.06);
    margin-bottom: 1rem; line-height: 1.8; color: #374151;
    border-left: 4px solid #2563eb;
    position: relative;
}
.fase-box.fase-1 { border-left-color: #1a3a8f; }
.fase-box.fase-2 { border-left-color: #2563eb; }
.fase-box.fase-3 { border-left-color: #3b82f6; }
.fase-box.fase-4 { border-left-color: #f59e0b; }

.fase-num {
    display: inline-flex; align-items: center; justify-content: center;
    width: 28px; height: 28px; border-radius: 50%;
    background: linear-gradient(135deg,#1a3a8f,#2563eb);
    color: white; font-weight: 700; font-size: .82rem;
    margin-right: .5rem; vertical-align: middle;
}
.fase-title {
    font-size: 1.05rem; font-weight: 700; color: #0f2557;
    margin-bottom: .6rem;
}
.fase-box h5 {
    font-size: .88rem; font-weight: 600; color: #1e3a8a;
    margin: .8rem 0 .4rem; border-bottom: 1px solid #e5e7eb;
    padding-bottom: .2rem;
}
.fase-box ul { padding-left: 1.1rem; }
.fase-box li { margin-bottom: .4rem; font-size: .9rem; }
.fase-box code {
    background: #eff6ff; padding: .1rem .35rem; border-radius: 4px;
    font-size: .84rem; color: #1e3a8a;
}

/* Caja de ecuación principal */
.eq-display {
    background: linear-gradient(135deg,#0f2557,#1a3a8f);
    border-radius: 12px; padding: 1.2rem 1.5rem; text-align: center;
    color: white; font-size: 1.1rem;
    margin: .9rem 0; box-shadow: 0 4px 14px rgba(37,99,235,.3);
    overflow-x: auto;
}

/* Caja de ecuación dentro de fase */
.eq-inner {
    background: #f0f4ff; border-radius: 8px; padding: .9rem 1.2rem;
    text-align: center; margin: .6rem 0; border: 1px solid #bfdbfe;
    overflow-x: auto;
}

/* Caja de predicción dinámica */
.eq-prediction {
    background: linear-gradient(135deg,#fef3c7,#fde68a);
    border-radius: 8px; padding: .9rem 1.2rem;
    text-align: center; margin: .6rem 0;
    border: 2px solid #f59e0b;
    overflow-x: auto;
}

.eq-result {
    background: linear-gradient(135deg,#dcfce7,#bbf7d0);
    border-radius: 8px; padding: .9rem 1.2rem;
    text-align: center; margin: .6rem 0;
    border: 2px solid #16a34a;
    overflow-x: auto;
}

.ibox {
    background: white; border-radius: 12px; padding: 1.2rem 1.5rem;
    box-shadow: 0 2px 12px rgba(0,0,0,.06); border-top: 3px solid #2563eb;
    margin-bottom: 1rem; line-height: 1.75; color: #374151;
}
.ibox h4 { color:#1a3a8f; margin-top:0; font-size:.98rem; font-weight:700; }
.ibox ul { padding-left:1.1rem; } .ibox li { margin-bottom:.4rem; }
.ibox code { background:#eff6ff; padding:.1rem .35rem; border-radius:4px;
             font-size:.86rem; color:#1e3a8a; }

.chips { display:flex; gap:.7rem; flex-wrap:wrap; margin:.9rem 0; }
.chip {
    flex:1; min-width:110px; background:white; border-radius:10px;
    padding:.7rem .9rem; border-left:4px solid #2563eb;
    box-shadow:0 2px 8px rgba(0,0,0,.06);
}
.chip .cl { font-size:.7rem; color:#6b7280; font-weight:600;
            text-transform:uppercase; letter-spacing:.5px; }
.chip .cv { font-size:1.3rem; font-weight:700; color:#1e3a8a; }

.stt {
    font-size:1.02rem; font-weight:700; color:#1e3a8a;
    border-bottom:2px solid #bfdbfe; padding-bottom:.3rem;
    margin:1.2rem 0 .7rem;
}
.chart-wrap {
    background:white; border-radius:13px; padding:.7rem;
    box-shadow:0 3px 16px rgba(0,0,0,.08); margin-top:.7rem;
}

.footer {
    text-align:center; color:#9ca3af; font-size:.73rem;
    margin-top:1.4rem; padding-top:.7rem; border-top:1px solid #e5e7eb;
}
</style>
"""

# ─────────────────────────────────────────────────────────────
# 3. UI
# ─────────────────────────────────────────────────────────────
app_ui = ui.page_sidebar(

    # ─── Sidebar ─────────────────────────────────────────────
    ui.sidebar(
        ui.HTML("""
        <div style="text-align:center;margin-bottom:1.2rem;">
          <div style="font-size:2rem;margin-bottom:.25rem;">📐</div>
          <div style="font-weight:700;color:#0f2557;font-size:.93rem;">Carlos Inga</div>
          <div style="color:#6b7280;font-size:.75rem;">Cálculo III · UNJFSC</div>
        </div>
        <hr style="border-color:#bfdbfe;margin:.6rem 0 1rem;">
        """),

        ui.h6("🎛️ Escenario Simulado", style="color:#0f2557;font-weight:700;margin-bottom:.7rem;"),

        ui.input_slider("x", "x — Código de Tarifa",
                        min=X_MIN, max=X_MAX, value=X_DEF, step=1),
        ui.HTML(f'<div style="font-size:.7rem;color:#6b7280;margin:-.35rem 0 .7rem">'
                f'Rango real: [{X_MIN}, {X_MAX}]</div>'),

        ui.input_slider("y", "y — Suministros",
                        min=Y_MIN, max=Y_MAX, value=Y_DEF, step=1),
        ui.HTML(f'<div style="font-size:.7rem;color:#6b7280;margin:-.35rem 0 1rem">'
                f'Rango real: [{Y_MIN}, {Y_MAX}]</div>'),

        ui.hr(),
        ui.output_ui("sidebar_pred"),
        ui.hr(),

        ui.HTML(f"""
        <div style="font-size:.73rem;color:#9ca3af;line-height:1.65;">
          <b style="color:#4b5563;">Modelo OLS</b><br>
          n = {n} obs. válidas<br>
          R² = {r2:.4f}<br>
          RMSE = {rmse:.4f} kWh
        </div>
        """),
        width=290,
        style="background:white;border-right:2px solid #bfdbfe;padding:1.4rem 1.1rem;",
    ),

    # ─── Panel principal ──────────────────────────────────────
    ui.HTML(MATHJAX_SCRIPT),
    ui.HTML(CSS),

    ui.HTML(f"""
    <div class="banner">
      <h2>📐 Cálculo III — Regresión Multivariable &nbsp;·&nbsp; UNJFSC</h2>
      <p>Carlos Inga &nbsp;·&nbsp; Función de Dos Variables · Plano de Mínimos Cuadrados (OLS)</p>
    </div>
    """),

    ui.navset_tab(

        # ══ TAB 1: Reporte de Análisis Matemático ═══════════════
        ui.nav_panel("📘 Reporte de Análisis Matemático",

            # ── FASE 1: Variables del Modelo ──
            ui.HTML(f"""
            <div class="fase-box fase-1">
              <div class="fase-title"><span class="fase-num">1</span>Variables del Modelo Matemático</div>
              <p>Nuestro modelo utiliza una función matemática de dos variables para predecir el consumo de energía. Las variables son las siguientes:</p>
              <ul>
                <li><b>Variable Independiente (x):</b> Código de tarifa. Tú la controlas moviendo el primer slider (barra deslizante).</li>
                <li><b>Variable Independiente (y):</b> Número de suministros. Tú la controlas moviendo el segundo slider.</li>
                <li><b>Variable Dependiente (z):</b> Consumo promedio en kWh. Es el resultado que calcula la función dependiendo de los valores que elijas para <b>x</b> e <b>y</b>.</li>
              </ul>
            </div>
            """),

            # ── FASE 2: Dominio y Restricciones ──
            ui.HTML(f"""
            <div class="fase-box fase-2">
              <div class="fase-title"><span class="fase-num">2</span>Dominio y Restricciones</div>
              <h5>Dominio Matemático</h5>
              <p>Matemáticamente, como es una función polinómica de primer grado (un plano plano), no tiene divisiones por cero ni raíces cuadradas. Por esto, el dominio matemático es todo el espacio bidimensional:</p>
              <div class="eq-inner" style="font-family: 'STIX Two Math', serif;">
                \\[
                Dom f = \\mathbb{{R}}^2 = \\{{ (x, y) \\in \\mathbb{{R}}^2 \\}}
                \\]
              </div>
              
              <h5>Restricciones del Problema (Realidad)</h5>
              <p>Aunque matemáticamente podemos poner cualquier número, en la vida real <b>no existen suministros negativos ni infinitos</b>. Por lo tanto, nuestro dominio real está restringido por los datos físicos que hemos recolectado:</p>
              <ul>
                <li>La variable <b>x</b> está restringida al intervalo <b>[{X_MIN}, {X_MAX}]</b>.</li>
                <li>La variable <b>y</b> está restringida al intervalo <b>[{Y_MIN}, {Y_MAX}]</b>.</li>
              </ul>
            </div>
            """),

            # ── FASE 3: Regla de Correspondencia y Evaluación ──
            ui.HTML('<div class="fase-box fase-3">'),
            ui.HTML('<div class="fase-title"><span class="fase-num" style="background:linear-gradient(135deg,#3b82f6,#2563eb);">3</span>Regla de Correspondencia y Evaluación Dinámica</div>'),
            ui.HTML('<p>Al mover los sliders en la barra lateral, estás eligiendo un nuevo par de valores <b>(x, y)</b>. El programa toma automáticamente la <b>regla de correspondencia</b> de nuestro modelo matemático y sustituye esas variables en la ecuación para encontrar el valor de la variable dependiente <b>z</b> (el consumo).</p>'),
            ui.output_ui("fase4_contenido"),
            ui.HTML('</div>'),

            ui.HTML('<div class="stt">Tabla de Datos Completa</div>'),
            ui.output_data_frame("tabla"),
        ),

        # ══ TAB 2: Corte Transversal 2D ═════════════════════════
        ui.nav_panel("📈 Corte Transversal 2D",

            ui.HTML('<div class="stt">Corte de la Función — variable "x" fija, variando "y"</div>'),
            ui.HTML('<div class="chart-wrap">'),
            output_widget("plot_2d"),
            ui.HTML('</div>'),
        ),

        # ══ TAB 3: Comparativa Histórica ════════════════════════
        ui.nav_panel("📊 Comparativa Histórica",
            ui.HTML('<div class="stt">Consumo Real (Z) vs. Función Teórica f(x,y)</div>'),
            ui.HTML('<div class="chart-wrap">'),
            output_widget("plot_barras"),
            ui.HTML('</div>'),
        ),

        # ══ TAB 4: Superficie 3D y Nube de Puntos ═══════════════
        ui.nav_panel("🌐 Superficie 3D y Dominio",
            ui.HTML("""
            <div class="fase-box fase-2" style="margin-top: 1rem;">
              <div class="fase-title">Interpretación del Gráfico 3D</div>
              <p>Este gráfico representa el modelo matemático completo en el espacio tridimensional (x, y, z):</p>
              <ul>
                <li><b>El Plano (Azul):</b> Es nuestra función matemática. Muestra todos los posibles resultados de consumo según la tarifa y suministros. Las líneas sobre el plano son <b>curvas de nivel</b>, que conectan puntos con el mismo consumo.</li>
                <li><b>Puntos Grises:</b> Son los datos <b>reales</b> históricos que usamos para crear el modelo. Observa cómo flotan cerca del plano.</li>
                <li><b>Punto Rojo (Diamante):</b> Es el punto <b>simulado</b>. Este punto se desliza sobre el plano azul cada vez que mueves los sliders de la izquierda, mostrando exactamente dónde se evalúa la función.</li>
              </ul>
            </div>
            """),
            ui.HTML('<div class="chart-wrap">'),
            output_widget("plot_3d"),
            ui.HTML('</div>'),
        ),
    ),

    ui.HTML('<div class="footer">Cálculo III · Carlos Inga · UNJFSC · Shiny for Python + Plotly + MathJax</div>'),

    title="Cálculo III — Carlos Inga · UNJFSC",
    fillable=False,
)


# ─────────────────────────────────────────────────────────────
# 4. SERVER
# ─────────────────────────────────────────────────────────────
def server(input, output, session):

    # ── Predicción reactiva ───────────────────────────────────
    @reactive.calc
    def zhat():
        return float(b0 + b1 * input.x() + b2 * input.y())

    # ── Predicción sidebar ────────────────────────────────────
    @output
    @render.ui
    def sidebar_pred():
        v = zhat()
        return ui.HTML(f"""
        <div class="pred-box">
          <div class="lbl">z Calculado f(x,y)</div>
          <div class="val">{v:.2f}</div>
          <div class="sub">kWh &nbsp;·&nbsp; x={input.x()} &nbsp; y={input.y()}</div>
        </div>
        """)

    # ── FASE 4: Pronóstico Dinámico (reactivo con LaTeX) ──────
    @output
    @render.ui
    def fase4_contenido():
        xi = input.x()
        yi = input.y()
        zp = zhat()

        term1 = b1 * xi
        term2 = b2 * yi
        s1_t = "+" if term1 >= 0 else "-"
        s2_t = "+" if term2 >= 0 else "-"

        return ui.HTML(f"""
        <h5>Valores Actuales (Par ordenado del Dominio)</h5>
        <div class="chips">
          <div class="chip" style="border-left-color:#f59e0b;">
            <div class="cl">x (Cód. Tarifa)</div>
            <div class="cv">{xi}</div>
          </div>
          <div class="chip" style="border-left-color:#f59e0b;">
            <div class="cl">y (Suministros)</div>
            <div class="cv">{yi}</div>
          </div>
        </div>

        <h5>Paso 1 — Regla de correspondencia</h5>
        <div class="eq-inner">
          \\[
          f(x, y) = {b0:.4f} \\;{s1_latex}\\; {abs(b1):.4f} x
          \\;{s2_latex}\\; {abs(b2):.6f} y
          \\]
        </div>

        <h5>Paso 2 — Evaluando la función en f({xi}, {yi})</h5>
        <div class="eq-prediction">
          \\[
          f({xi}, {yi}) = {b0:.4f} \\;{s1_latex}\\; {abs(b1):.4f}({xi})
          \\;{s2_latex}\\; {abs(b2):.6f}({yi})
          \\]
        </div>

        <h5>Paso 3 — Resultado del Consumo "z"</h5>
        <div class="eq-result">
          \\[
          \\boxed{{\\;
          z = f({xi},\\, {yi}) = {zp:.4f} \\;\\text{{kWh}}
          \\;}}
          \\]
        </div>

        <script>
          if (window.MathJax && window.MathJax.typesetPromise) {{
            window.MathJax.typesetPromise();
          }}
        </script>
        """)

    # ── Tabla Tab 1 ───────────────────────────────────────────
    @output
    @render.data_frame
    def tabla():
        d = df.copy()
        d["z_CALCULADO_f(x,y)"]    = np.round(z_pred, 4)
        d["Residuo"] = np.round(z_data - z_pred, 4)
        return render.DataGrid(d, width="100%")

    # ── TAB 2: Gráfico de línea 2D (corte x=fijo) ───────────
    @output
    @render_widget
    def plot_2d():
        xi = float(input.x())
        yi = float(input.y())
        zp  = zhat()

        y_range = np.linspace(Y_MIN, Y_MAX, 400)
        z_line   = b0 + b1 * xi + b2 * y_range

        poly_coefs = np.polyfit(y_data, z_data, 4)
        z_curva = np.polyval(poly_coefs, y_range)

        fig = go.Figure()

        # Área bajo la curva
        fig.add_trace(go.Scatter(
            x=y_range, y=z_line,
            mode="lines",
            fill='tozeroy',
            fillcolor='rgba(255, 71, 126, 0.05)',
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=False,
            hoverinfo="skip"
        ))

        # Línea del plano
        fig.add_trace(go.Scatter(
            x=y_range, y=z_line,
            mode="lines",
            name=f"Plano",
            line=dict(color="#ff477e", width=4),
            hovertemplate=f"y=%{{x:.0f}}<br>z=%{{y:.2f}} kWh<extra>Plano (x={xi:.0f})</extra>",
        ))

        # Curva de tendencia real
        fig.add_trace(go.Scatter(
            x=y_range, y=z_curva,
            mode="lines",
            name="Curva Real",
            line=dict(color="#8b5cf6", width=3, dash="dot"),
            hovertemplate=f"y=%{{x:.0f}}<br>z_curva=%{{y:.2f}} kWh<extra>Tendencia</extra>",
        ))

        num_points = 8
        y_points = np.linspace(Y_MIN, Y_MAX, num_points)
        z_points_curva = np.polyval(poly_coefs, y_points)
        
        fig.add_trace(go.Scatter(
            x=y_points, y=z_points_curva,
            mode="markers",
            name="Puntos Proyectados",
            marker=dict(size=9, color="#080c16", line=dict(color="#00f0ff", width=2.5)),
            hoverinfo="skip",
        ))

        fig.add_trace(go.Scatter(
            x=[yi], y=[zp],
            mode="markers+text",
            name=f"Punto Actual",
            marker=dict(size=20, color="#080c16", symbol="square",
                        line=dict(color="#ff477e", width=4)),
            text=[f"  z = {zp:,.2f} "],
            textposition="middle right",
            textfont=dict(size=12, color="#ff477e", family="monospace"),
            hovertemplate=f"Punto Actual<br>y={yi:.0f}<br>z={zp:.4f} kWh<extra></extra>",
        ))

        fig.add_annotation(
            x=0.03, y=0.92, xref="paper", yref="paper",
            text=(
                "<b>GUÍA DE VARIABLES (Corte 2D):</b><br><br>"
                "<b>Eje Vertical (z)</b> = Consumo<br>"
                "<b>Eje Horizontal (y)</b> = Suministros<br>"
                f"<b>Parámetro Fijo (x)</b> = Código Tarifa ({xi:.0f})<br><br>"
                "<i>La línea recta muestra el corte del plano, mientras que<br>"
                "la curva punteada muestra la tendencia orgánica.</i>"
            ),
            showarrow=False, align="left", bgcolor="rgba(10, 14, 23, 0.75)",
            bordercolor="#334155", borderwidth=1, font=dict(color="white", size=11),
            xanchor="left", yanchor="top", borderpad=12
        )

        fig.update_layout(
            title=dict(
                text=f"<span style='color:#ff477e;'>■</span> Corte del Plano (x constante)<br><br><span style='font-size:16px; color:white;'><b>z = f(x, y) — donde x = <span style='color:#ff477e;'>{xi:.0f}</span></b></span>",
                font=dict(family="monospace", size=14, color="#ff477e"),
                x=0.01, y=0.98, xanchor="left", yanchor="top"
            ),
            height=580,
            xaxis=dict(title="<b>y — Suministros</b>", title_font=dict(size=12, color="white"), gridcolor="#1e293b"),
            yaxis=dict(title="<b>z — Consumo f(x,y)</b>", title_font=dict(size=12, color="white"), gridcolor="#1e293b"),
            paper_bgcolor="#080c16", plot_bgcolor="#080c16",
            showlegend=False, margin=dict(l=70, r=40, t=110, b=80),
        )
        return fig

    # ── TAB 3: Comparativa Histórica (Barras) ────────────────
    @output
    @render_widget
    def plot_barras():
        xi, yi = float(input.x()), float(input.y())
        zp       = zhat()

        obs_labels = [f"O{i+1}" for i in range(n)]
        obs_labels_all = obs_labels + ["★ Simulado"]

        z_real_all = list(z_data) + [None]
        z_pred_all = list(z_pred) + [None]
        z_sim_all  = [None] * n + [zp]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Z Real", x=obs_labels_all, y=z_real_all,
            marker=dict(color=["#1e3a8a"] * n + ["rgba(0,0,0,0)"])
        ))
        fig.add_trace(go.Bar(
            name="Z Calculado", x=obs_labels_all, y=z_pred_all,
            marker=dict(color=["#60a5fa"] * n + ["rgba(0,0,0,0)"])
        ))
        fig.add_trace(go.Bar(
            name=f"Simulado f({xi:.0f}, {yi:.0f})",
            x=obs_labels_all, y=z_sim_all,
            marker=dict(color=["rgba(0,0,0,0)"] * n + ["#ef4444"])
        ))

        fig.update_layout(
            barmode="group", height=500,
            xaxis=dict(title="Puntos del Dominio (x,y)", gridcolor="#e5e7eb"),
            yaxis=dict(title="Consumo Z", gridcolor="#e5e7eb"),
            paper_bgcolor="white", plot_bgcolor="white",
        )
        return fig

    # ── TAB 4: Superficie 3D ──────────────────────────────────
    @output
    @render_widget
    def plot_3d():
        xi, yi = float(input.x()), float(input.y())
        zp       = zhat()

        svx = np.linspace(x_data.min() - 3, x_data.max() + 3, 55)
        svy = np.linspace(y_data.min() - 300, y_data.max() + 300, 55)
        SX, SY = np.meshgrid(svx, svy)
        SZ = b0 + b1 * SX + b2 * SY

        fig = go.Figure()
        
        # 1. Plano f(x,y) con curvas de nivel
        fig.add_trace(go.Surface(
            x=SX, y=SY, z=SZ,
            colorscale="Blues", opacity=0.75,
            name="Plano f(x,y)",
            contours=dict(
                z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True)
            )
        ))
        
        # 2. Datos Reales (Gris)
        fig.add_trace(go.Scatter3d(
            x=x_data, y=y_data, z=z_data,
            mode="markers", 
            marker=dict(size=6, color="#9ca3af", line=dict(color="#4b5563", width=1), opacity=0.8),
            name="Z Real",
            hovertemplate="Tarifa: %{x}<br>Suministros: %{y}<br>Consumo Real: %{z} kWh<extra></extra>"
        ))
        
        # 3. Punto Simulado (Rojo)
        fig.add_trace(go.Scatter3d(
            x=[xi], y=[yi], z=[zp],
            mode="markers+text", 
            marker=dict(size=12, color="#ef4444", symbol="diamond", line=dict(color="#7f1d1d", width=2)),
            text=[f"Simulado: {zp:.1f} kWh"], textposition="top center",
            textfont=dict(color="#ef4444", size=14, family="Arial Black"),
            name="Simulado f(x,y)",
            hovertemplate="Tarifa: %{x}<br>Suministros: %{y}<br>Consumo Simulado: %{z:.2f} kWh<extra></extra>"
        ))
        
        # 4. Línea de proyección para el punto simulado (caída al plano XY)
        fig.add_trace(go.Scatter3d(
            x=[xi, xi], y=[yi, yi], z=[0, zp],
            mode="lines",
            line=dict(color="#ef4444", width=3, dash="dot"),
            showlegend=False,
            hoverinfo="skip"
        ))

        fig.update_layout(
            height=650,
            scene=dict(
                xaxis=dict(title="x (Tarifa)", gridcolor="#e5e7eb", backgroundcolor="#f8fafc"),
                yaxis=dict(title="y (Suministros)", gridcolor="#e5e7eb", backgroundcolor="#f8fafc"),
                zaxis=dict(title="z (Consumo)", gridcolor="#e5e7eb", backgroundcolor="#f8fafc"),
                camera=dict(
                    eye=dict(x=1.6, y=-1.6, z=0.8) # Mejor ángulo de inicio
                )
            ),
            paper_bgcolor="white",
            legend=dict(yanchor="top", y=0.95, xanchor="left", x=0.05),
            margin=dict(l=0, r=0, t=20, b=0)
        )
        return fig

app = App(app_ui, server)
