# =========================================================
# FFT ANALYZER PRO — VERSÃO LABORATORIAL
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.fft import fft, fftfreq
from scipy.io import wavfile
from scipy.signal import spectrogram
from scipy.spatial.distance import cosine

from io import BytesIO

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer
)

from reportlab.lib.styles import getSampleStyleSheet

# =========================================================
# CONFIG
# =========================================================

st.set_page_config(
    page_title="FFT Analyzer Lab",
    layout="wide"
)

st.title("FFT Analyzer Lab")
st.write(
    "Análise espectral científica para sinais,"
    " vibração, sensores, RF e dados laboratoriais."
)

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("Configurações")

log_fft = st.sidebar.checkbox(
    "Escala log FFT",
    value=True
)

normalizar = st.sidebar.checkbox(
    "Normalizar FFT",
    value=True
)

mostrar_espectrograma = st.sidebar.checkbox(
    "Mostrar espectrograma",
    value=True
)

limite_plot = st.sidebar.slider(
    "Amostras exibidas no gráfico",
    100,
    10000,
    5000
)

# =========================================================
# GERADOR DE SINAL
# =========================================================

st.sidebar.title("Gerador de sinal")

tipo_sinal = st.sidebar.selectbox(
    "Tipo de sinal",
    [
        "Carregar externo",
        "Seno",
        "Ruído branco",
        "Ruído rosa"
    ]
)

freq_base = st.sidebar.slider(
    "Frequência base",
    1,
    10000,
    440
)

duracao = st.sidebar.slider(
    "Duração (s)",
    1,
    20,
    3
)

sample_rate = 44100

# =========================================================
# FUNÇÕES
# =========================================================

def gerar_seno(freq, duracao):

    t = np.linspace(
        0,
        duracao,
        int(sample_rate * duracao)
    )

    sinal = np.sin(
        2 * np.pi * freq * t
    )

    return sinal

def gerar_ruido_branco(duracao):

    return np.random.normal(
        0,
        0.3,
        int(sample_rate * duracao)
    )

def gerar_ruido_rosa(duracao):

    n = int(sample_rate * duracao)

    white = np.random.randn(n)

    fft_data = np.fft.rfft(white)

    freqs = np.fft.rfftfreq(n)

    freqs[0] = 1

    fft_data = fft_data / np.sqrt(freqs)

    pink = np.fft.irfft(fft_data)

    pink = pink / np.max(np.abs(pink))

    return pink

# =========================================================
# ENTRADA
# =========================================================

uploaded_file = None
uploaded_file_B = None

if tipo_sinal == "Carregar externo":

    uploaded_file = st.file_uploader(
        "Escolha CSV ou WAV",
        type=["csv", "wav"]
    )

else:

    if tipo_sinal == "Seno":
        signal = gerar_seno(
            freq_base,
            duracao
        )

    elif tipo_sinal == "Ruído branco":
        signal = gerar_ruido_branco(
            duracao
        )

    elif tipo_sinal == "Ruído rosa":
        signal = gerar_ruido_rosa(
            duracao
        )

# =========================================================
# COMPARAÇÃO A/B
# =========================================================

st.header("Comparação A/B")

comparar = st.checkbox(
    "Ativar comparação"
)

if comparar:

    uploaded_file_B = st.file_uploader(
        "Escolha segundo arquivo",
        type=["csv", "wav"]
    )

# =========================================================
# LEITURA ARQUIVO A
# =========================================================

sample_rate_A = sample_rate

if tipo_sinal == "Carregar externo":

    if uploaded_file is not None:

        if uploaded_file.name.endswith(".csv"):

            df = pd.read_csv(uploaded_file)

            signal = df.iloc[:, 0].values

            sample_rate_A = len(signal)

        elif uploaded_file.name.endswith(".wav"):

            sample_rate_A, signal = wavfile.read(
                uploaded_file
            )

            if len(signal.shape) > 1:
                signal = signal[:, 0]

# =========================================================
# LEITURA ARQUIVO B
# =========================================================

signal_B = None

if comparar and uploaded_file_B is not None:

    if uploaded_file_B.name.endswith(".csv"):

        df_B = pd.read_csv(uploaded_file_B)

        signal_B = df_B.iloc[:, 0].values

        sample_rate_B = len(signal_B)

    elif uploaded_file_B.name.endswith(".wav"):

        sample_rate_B, signal_B = wavfile.read(
            uploaded_file_B
        )

        if len(signal_B.shape) > 1:
            signal_B = signal_B[:, 0]

# =========================================================
# FFT PRINCIPAL
# =========================================================

if (
    (tipo_sinal != "Carregar externo")
    or
    (uploaded_file is not None)
):

    N = len(signal)

    yf = fft(signal)

    xf = fftfreq(
        N,
        1 / sample_rate_A
    )

    magnitude = np.abs(yf)

    positive = xf >= 0

    xf = xf[positive]
    magnitude = magnitude[positive]

    # =====================================================
    # NORMALIZAÇÃO
    # =====================================================

    if normalizar:
        magnitude = (
            magnitude /
            np.max(magnitude)
        )

    dominant_freq = xf[np.argmax(magnitude)]

    max_magnitude = np.max(magnitude)

    # =====================================================
    # ANÁLISE AUTOMÁTICA
    # =====================================================

    st.header("Análise automática")

    baixa_freq = np.mean(magnitude[:100])

    alta_freq = np.mean(magnitude[-100:])

    energia_total = np.sum(magnitude)

    energia_baixa = np.sum(magnitude[:500])

    percentual_baixa = (
        energia_baixa /
        energia_total
    ) * 100

    if baixa_freq > alta_freq:

        st.success(
            "Predomínio de baixas frequências."
        )

    else:

        st.warning(
            "Predomínio de altas frequências."
        )

    st.info(
        f"Energia concentrada "
        f"em baixas frequências: "
        f"{percentual_baixa:.2f}%"
    )

    if np.max(np.abs(signal)) > 0.95:

        st.warning(
            "Possível saturação ou clipping."
        )

    # =====================================================
    # TOP FREQUÊNCIAS
    # =====================================================

    st.header("Principais frequências")

    idx = np.argsort(magnitude)[-10:]

    top_freqs = xf[idx]
    top_mags = magnitude[idx]

    top_df = pd.DataFrame({

        "Frequência (Hz)": top_freqs,
        "Magnitude": top_mags

    })

    st.dataframe(

        top_df.sort_values(
            by="Magnitude",
            ascending=False
        )

    )

    # =====================================================
    # MÉTRICAS
    # =====================================================

    st.header("Métricas")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Frequência dominante",
        f"{dominant_freq:.2f} Hz"
    )

    col2.metric(
        "Magnitude máxima",
        f"{max_magnitude:.4f}"
    )

    col3.metric(
        "Duração",
        f"{N/sample_rate_A:.2f} s"
    )

    col4.metric(
        "Número de amostras",
        N
    )

    # =====================================================
    # SIMILARIDADE A/B
    # =====================================================

    if signal_B is not None:

        N_B = len(signal_B)

        yf_B = fft(signal_B)

        xf_B = fftfreq(
            N_B,
            1 / sample_rate_B
        )

        magnitude_B = np.abs(yf_B)

        positive_B = xf_B >= 0

        xf_B = xf_B[positive_B]
        magnitude_B = magnitude_B[positive_B]

        if normalizar:

            magnitude_B = (
                magnitude_B /
                np.max(magnitude_B)
            )

        # alinhamento tamanho

        min_len = min(
            len(magnitude),
            len(magnitude_B)
        )

        magA = magnitude[:min_len]
        magB = magnitude_B[:min_len]

        similarity = (
            1 - cosine(magA, magB)
        ) * 100

        st.header("Similaridade espectral")

        st.metric(
            "Similaridade A/B",
            f"{similarity:.2f}%"
        )

    # =====================================================
    # GRÁFICOS
    # =====================================================

    colA, colB = st.columns(2)

    # =====================================================
    # SINAL
    # =====================================================

    with colA:

        st.subheader("Sinal Original")

        fig1, ax1 = plt.subplots(
            figsize=(8, 4)
        )

        signal_plot = signal[:limite_plot]

        ax1.plot(signal_plot)

        ax1.set_xlabel("Amostras")
        ax1.set_ylabel("Amplitude")

        st.pyplot(fig1)

    # =====================================================
    # FFT
    # =====================================================

    with colB:

        st.subheader("FFT")

        fig2, ax2 = plt.subplots(
            figsize=(8, 4)
        )

        ax2.plot(
            xf,
            magnitude,
            label="Arquivo A"
        )

        pico_idx = np.argmax(magnitude)

        ax2.scatter(
            xf[pico_idx],
            magnitude[pico_idx],
            s=100
        )

        ax2.annotate(
            f"{xf[pico_idx]:.1f} Hz",
            (
                xf[pico_idx],
                magnitude[pico_idx]
            )
        )

        if signal_B is not None:

            ax2.plot(
                xf_B,
                magnitude_B,
                label="Arquivo B"
            )

        if log_fft:
            ax2.set_yscale("log")

        ax2.set_xlabel("Frequência (Hz)")
        ax2.set_ylabel("Magnitude")

        ax2.legend()

        st.pyplot(fig2)

    # =====================================================
    # ESPECTROGRAMA
    # =====================================================

    if mostrar_espectrograma:

        st.subheader("Espectrograma")

        f, t_spec, Sxx = spectrogram(
            signal,
            sample_rate_A
        )

        fig3, ax3 = plt.subplots(
            figsize=(12, 5)
        )

        pcm = ax3.pcolormesh(
            t_spec,
            f,
            10 * np.log10(Sxx + 1e-10),
            shading='gouraud'
        )

        fig3.colorbar(
            pcm,
            ax=ax3,
            label="Potência (dB)"
        )

        ax3.set_ylabel("Frequência (Hz)")
        ax3.set_xlabel("Tempo (s)")

        st.pyplot(fig3)

    # =====================================================
    # EXPORTAÇÃO
    # =====================================================

    st.header("Exportação")

    # =====================================================
    # PNG FFT
    # =====================================================

    png_buffer = BytesIO()

    fig2.savefig(
        png_buffer,
        format="png"
    )

    st.download_button(
        label="Baixar FFT PNG",
        data=png_buffer.getvalue(),
        file_name="fft.png",
        mime="image/png"
    )

    # =====================================================
    # CSV FREQUÊNCIAS
    # =====================================================

    csv_export = top_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="Baixar frequências CSV",
        data=csv_export,
        file_name="frequencias.csv",
        mime="text/csv"
    )

    # =====================================================
    # PDF
    # =====================================================

    pdf_buffer = BytesIO()

    doc = SimpleDocTemplate(
        pdf_buffer
    )

    styles = getSampleStyleSheet()

    elements = []

    elements.append(

        Paragraph(
            "Relatório FFT Analyzer Lab",
            styles['Title']
        )

    )

    elements.append(Spacer(1, 20))

    elements.append(

        Paragraph(
            f"Frequência dominante: "
            f"{dominant_freq:.2f} Hz",
            styles['BodyText']
        )

    )

    elements.append(

        Paragraph(
            f"Magnitude máxima: "
            f"{max_magnitude:.4f}",
            styles['BodyText']
        )

    )

    elements.append(

        Paragraph(
            f"Percentual baixa frequência: "
            f"{percentual_baixa:.2f}%",
            styles['BodyText']
        )

    )

    if signal_B is not None:

        elements.append(

            Paragraph(
                f"Similaridade espectral: "
                f"{similarity:.2f}%",
                styles['BodyText']
            )

        )

    doc.build(elements)

    st.download_button(
        label="Gerar relatório PDF",
        data=pdf_buffer.getvalue(),
        file_name="relatorio_fft.pdf",
        mime="application/pdf"
    )