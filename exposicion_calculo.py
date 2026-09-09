"""
=============================================================================
exposicion_calculo.py
─────────────────────────────────────────────────────────────────────────────
Script de exposición de Cálculo III — Funciones de Varias Variables
Regresión Lineal Múltiple por Mínimos Cuadrados Ordinarios (OLS)

Dependencias: pandas · numpy · matplotlib   (ver requirements.txt)
Uso          : python exposicion_calculo.py
=============================================================================
"""

# ─── Librerías ───────────────────────────────────────────────────────────────
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import ticker
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401  (registro 3D)

warnings.filterwarnings("ignore")

# ─── Constantes de diseño ────────────────────────────────────────────────────
AZUL_OSCURO   = "#1a3a8f"
AZUL_MEDIO    = "#2563eb"
AZUL_CLARO    = "#60a5fa"
GRIS_FONDO    = "#f0f4ff"
GRIS_TEXTO    = "#374151"
BLANCO        = "#ffffff"
ROJO_RESIDUO  = "#ef4444"

# ─── 1. CARGA Y LIMPIEZA DE DATOS ────────────────────────────────────────────
ARCHIVO_CSV = "datos-modelo.csv"

try:
    df = pd.read_csv(ARCHIVO_CSV)
except FileNotFoundError:
    sys.exit(
        f"\n[ERROR] No se encontró '{ARCHIVO_CSV}'.\n"
        "  Asegúrate de ejecutar el script desde el mismo directorio que el CSV.\n"
    )

# Limpieza: eliminar espacios en nombres de columnas y filas con nulos
df.columns = df.columns.str.strip()
df = df.dropna()
df = df.apply(pd.to_numeric, errors="coerce").dropna()

# Vectores de trabajo
x1     = df["x1_COD_TARIFA"].values       # variable independiente 1
x2     = df["x2_SUMINISTROS"].values      # variable independiente 2
y_real = df["y_PROMEDIO_CONSUMO"].values  # variable dependiente
n      = len(y_real)

# ─── 2. REGRESIÓN OLS (Mínimos Cuadrados Ordinarios) ─────────────────────────
# Matriz de diseño X  →  columna de unos | x1 | x2
X_diseno = np.column_stack([np.ones(n), x1, x2])

# Ecuaciones normales:  β = (XᵀX)⁻¹ Xᵀy  →  resueltas por lstsq (SVD)
beta, _, _, _ = np.linalg.lstsq(X_diseno, y_real, rcond=None)
b0, b1, b2    = beta

# Valores predichos y residuos
y_pred   = X_diseno @ beta
residuos = y_real - y_pred

# Métricas de bondad de ajuste
ss_res = np.sum(residuos ** 2)
ss_tot = np.sum((y_real - np.mean(y_real)) ** 2)
r2     = 1.0 - ss_res / ss_tot
rmse   = np.sqrt(ss_res / n)

# ─── 3. FIGURA PRINCIPAL (2×2 grid) ──────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")

fig = plt.figure(
    figsize=(18, 12),
    facecolor=GRIS_FONDO,
    constrained_layout=False,
)
fig.suptitle(
    "CÁLCULO III — Regresión Lineal Múltiple en Funciones de Varias Variables",
    fontsize=16, fontweight="bold", color=AZUL_OSCURO,
    y=0.99,
)

# GridSpec: fila 0 normal | fila 1 normal ; col 0 normal | col 1 normal
gs = gridspec.GridSpec(
    2, 2,
    figure=fig,
    left=0.06, right=0.97,
    top=0.94, bottom=0.06,
    hspace=0.38, wspace=0.30,
)

# ─────────────────────────────────────────────────────────────────────────────
# CUADRANTE 1 (Top-Left) — Modelo Matemático (texto)
# ─────────────────────────────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor(BLANCO)
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.axis("off")

# Borde decorativo del panel
for spine in ax1.spines.values():
    spine.set_edgecolor(AZUL_MEDIO)
    spine.set_linewidth(1.5)

# Título del cuadrante
ax1.text(
    0.5, 0.97,
    "① Modelo Matemático — Resolución OLS",
    ha="center", va="top",
    fontsize=12, fontweight="bold", color=AZUL_OSCURO,
    transform=ax1.transAxes,
)

# Construir signo legible para la ecuación
s1 = "+" if b1 >= 0 else "−"
s2 = "+" if b2 >= 0 else "−"

# Bloque de texto con viñetas
texto = (
    f"  ECUACIÓN DEL PLANO DE REGRESIÓN:\n"
    f"  ŷ = {b0:.4f}  {s1}  {abs(b1):.4f}·x₁  {s2}  {abs(b2):.6f}·x₂\n\n"

    f"  ─ IDENTIFICACIÓN DE VARIABLES ─\n"
    f"  • y  →  y_PROMEDIO_CONSUMO  [variable dependiente]\n"
    f"  • x₁ →  x1_COD_TARIFA       [variable independiente]\n"
    f"           dominio: [{int(x1.min())}, {int(x1.max())}]\n"
    f"  • x₂ →  x2_SUMINISTROS      [variable independiente]\n"
    f"           dominio: [{int(x2.min())}, {int(x2.max())}]\n"
    f"  • n  =  {n} observaciones válidas\n\n"

    f"  ─ MÉTODO DE MÍNIMOS CUADRADOS (OLS) ─\n"
    f"  • Se plantea el modelo:  ŷ = β₀ + β₁x₁ + β₂x₂\n"
    f"  • Se minimiza la suma de cuadrados de residuos:\n"
    f"    S(β) = Σ(yᵢ − β₀ − β₁x₁ᵢ − β₂x₂ᵢ)²\n"
    f"  • Derivando e igualando a cero se obtiene el\n"
    f"    sistema matricial de ecuaciones normales:\n"
    f"    (XᵀX)·β = Xᵀy    [resuelto por SVD]\n\n"

    f"  ─ SIGNIFICADO DE LOS COEFICIENTES ─\n"
    f"  • β₀ = {b0:.4f}: consumo base (x₁=0, x₂=0)\n"
    f"  • β₁ = {b1:.4f}: Δŷ por cada unidad Δx₁ (x₂ fijo)\n"
    f"  • β₂ = {b2:.6f}: Δŷ por cada unidad Δx₂ (x₁ fijo)\n\n"

    f"  ─ BONDAD DE AJUSTE ─\n"
    f"  • R²   = {r2:.4f}  ({r2*100:.1f}% variabilidad explicada)\n"
    f"  • RMSE = {rmse:.4f} kWh  (error cuadrático medio)\n"
)

ax1.text(
    0.03, 0.91, texto,
    ha="left", va="top",
    fontsize=8.2,
    color=GRIS_TEXTO,
    fontfamily="monospace",
    transform=ax1.transAxes,
    linespacing=1.55,
)

# ─────────────────────────────────────────────────────────────────────────────
# CUADRANTE 2 (Top-Right) — Curvas de Nivel (contourf)
# ─────────────────────────────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor(GRIS_FONDO)

# Malla para evaluar el plano
g1 = np.linspace(x1.min(), x1.max(), 300)
g2 = np.linspace(x2.min(), x2.max(), 300)
G1, G2 = np.meshgrid(g1, g2)
Zg     = b0 + b1 * G1 + b2 * G2

# Mapa de contorno relleno
cf = ax2.contourf(G1, G2, Zg, levels=20, cmap="RdYlBu_r", alpha=0.92)
cs = ax2.contour( G1, G2, Zg, levels=20, colors="white", linewidths=0.5, alpha=0.5)
ax2.clabel(cs, inline=True, fontsize=6.5, fmt="%.0f", colors="white")

# Colorbar
cbar = fig.colorbar(cf, ax=ax2, pad=0.02, shrink=0.92)
cbar.set_label("ŷ  Consumo Promedio (kWh)", fontsize=8, color=GRIS_TEXTO)
cbar.ax.tick_params(labelsize=7.5)

# Puntos reales superpuestos
sc = ax2.scatter(
    x1, x2, c=y_real, cmap="RdYlBu_r",
    edgecolors="white", linewidths=1.2,
    s=80, zorder=5,
)
for i, (xi1, xi2, yi) in enumerate(zip(x1, x2, y_real)):
    ax2.annotate(
        f"{yi:.0f}",
        (xi1, xi2), textcoords="offset points", xytext=(4, 4),
        fontsize=6.5, color=AZUL_OSCURO, fontweight="bold",
    )

ax2.set_title(
    "② Curvas de Nivel\n"
    "Rojo/naranja = mayor consumo ↑    Azul = menor consumo ↓",
    fontsize=9.5, fontweight="bold", color=AZUL_OSCURO, pad=8,
)
ax2.set_xlabel("x₁ — Código de Tarifa", fontsize=9, color=GRIS_TEXTO)
ax2.set_ylabel("x₂ — Suministros",       fontsize=9, color=GRIS_TEXTO)
ax2.tick_params(labelsize=8)
ax2.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

# ─────────────────────────────────────────────────────────────────────────────
# CUADRANTE 3 (Bottom-Left) — Barras Agrupadas (Real vs Predicho)
# ─────────────────────────────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
ax3.set_facecolor(BLANCO)

indices = np.arange(n)
ancho   = 0.38

barras_real  = ax3.bar(indices - ancho / 2, y_real,  ancho, label="y Real (CSV)",     color=AZUL_OSCURO, alpha=0.88)
barras_pred  = ax3.bar(indices + ancho / 2, y_pred,  ancho, label="ŷ Predicho (OLS)", color=AZUL_CLARO,  alpha=0.88)

# Etiquetas de residuo sobre cada par
for i, (yr, yp) in enumerate(zip(y_real, y_pred)):
    err = yr - yp
    color_err = ROJO_RESIDUO if abs(err) > rmse else "#16a34a"
    ax3.text(
        i, max(yr, yp) + 18,
        f"Δ{err:+.0f}", ha="center", va="bottom",
        fontsize=6.5, color=color_err, fontweight="bold",
    )

# Eje X con etiquetas "Obs N / T{cod}"
etiquetas = [f"O{i+1}\nT{int(df.reset_index(drop=True).loc[i,'x1_COD_TARIFA'])}" for i in range(n)]
ax3.set_xticks(indices)
ax3.set_xticklabels(etiquetas, fontsize=6.5)
ax3.set_title(
    "③ Precisión — Real vs. Predicho\n"
    "Δ verde = residuo < RMSE  |  Δ rojo = residuo > RMSE",
    fontsize=9.5, fontweight="bold", color=AZUL_OSCURO, pad=8,
)
ax3.set_ylabel("Consumo Promedio (kWh)", fontsize=9, color=GRIS_TEXTO)
ax3.set_xlabel("Observación  /  Código de Tarifa", fontsize=9, color=GRIS_TEXTO)
ax3.legend(fontsize=8, framealpha=0.9)
ax3.tick_params(axis="y", labelsize=8)
ax3.yaxis.grid(True, linestyle="--", alpha=0.6)
ax3.set_axisbelow(True)

# Línea de RMSE de referencia
ax3.axhline(rmse, linestyle=":", color=ROJO_RESIDUO, linewidth=1.2, label=f"RMSE = {rmse:.1f}")

# ─────────────────────────────────────────────────────────────────────────────
# CUADRANTE 4 (Bottom-Right) — Superficie 3D
# ─────────────────────────────────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1], projection="3d")
ax4.set_facecolor(GRIS_FONDO)

# Malla para la superficie
s1v = np.linspace(x1.min() - 2, x1.max() + 2, 60)
s2v = np.linspace(x2.min() - 200, x2.max() + 200, 60)
S1, S2 = np.meshgrid(s1v, s2v)
Zs     = b0 + b1 * S1 + b2 * S2

# Plano de regresión
surf = ax4.plot_surface(
    S1, S2, Zs,
    cmap="Blues", alpha=0.60,
    linewidth=0, antialiased=True,
)

# Puntos reales coloreados por residuo
norm_res = (residuos - residuos.min()) / ((residuos.max() - residuos.min()) + 1e-9)
colores_pts = plt.cm.RdYlGn(norm_res)

sc3d = ax4.scatter(
    x1, x2, y_real,
    c=residuos, cmap="RdYlGn",
    s=60, edgecolors="white", linewidths=0.8,
    zorder=5, depthshade=True,
)

# Líneas verticales de residuo (punto real → proyección en el plano)
for xi1, xi2, yi, ypi in zip(x1, x2, y_real, y_pred):
    ax4.plot(
        [xi1, xi1], [xi2, xi2], [yi, ypi],
        color="rgba(180,0,0,0.5)" if False else "#ef444488",
        linestyle=":", linewidth=1.2,
    )

ax4.set_title(
    "④ Superficie 3D\nPlano OLS + datos reales (color = residuo)",
    fontsize=9.5, fontweight="bold", color=AZUL_OSCURO, pad=10,
)
ax4.set_xlabel("x₁ Cód. Tarifa", fontsize=8, labelpad=6)
ax4.set_ylabel("x₂ Suministros",  fontsize=8, labelpad=6)
ax4.set_zlabel("y Consumo",        fontsize=8, labelpad=6)
ax4.tick_params(labelsize=7)
ax4.view_init(elev=22, azim=-55)

# Colorbar del scatter 3D (residuos)
cbar3d = fig.colorbar(sc3d, ax=ax4, pad=0.08, shrink=0.65, aspect=15)
cbar3d.set_label("Residuo (y − ŷ)", fontsize=7.5, color=GRIS_TEXTO)
cbar3d.ax.tick_params(labelsize=7)

# ─── Pie de página ───────────────────────────────────────────────────────────
fig.text(
    0.5, 0.01,
    f"Sustentación Cálculo III  ·  Regresión Multivariable OLS  ·  "
    f"β₀={b0:.2f}  β₁={b1:.4f}  β₂={b2:.6f}  ·  R²={r2:.4f}  ·  n={n}",
    ha="center", va="bottom",
    fontsize=7.5, color="#6b7280", style="italic",
)

# ─── Mostrar ventana interactiva ──────────────────────────────────────────────
plt.show()
