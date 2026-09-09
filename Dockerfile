# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile  —  Exposición Cálculo III
# Imagen mínima con Python 3.11 + soporte gráfico X11 para matplotlib
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# ── Instalar dependencias del sistema para matplotlib / Tkinter / X11 ────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-tk \
        libx11-6 \
        libxext6 \
        libxrender1 \
        libxft2 \
        fontconfig \
        fonts-dejavu-core \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ── Directorio de trabajo dentro del contenedor ───────────────────────────────
WORKDIR /app

# ── Copiar dependencias y el proyecto ────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY datos-modelo.csv        .
COPY exposicion_calculo.py   .

# ── Variable de entorno: usar backend Tk (ventana nativa via X11) ─────────────
ENV MPLBACKEND=TkAgg

# ── Comando de arranque ───────────────────────────────────────────────────────
CMD ["python", "exposicion_calculo.py"]
