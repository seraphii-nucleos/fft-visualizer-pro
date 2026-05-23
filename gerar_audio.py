import numpy as np
from scipy.io.wavfile import write

sample_rate = 44100

duracao = 3

frequencia = 440

t = np.linspace(
    0,
    duracao,
    int(sample_rate * duracao),
    endpoint=False
)

audio = 0.5 * np.sin(
    2 * np.pi * frequencia * t
)

audio = np.int16(audio * 32767)

write(
    "teste.wav",
    sample_rate,
    audio
)

print("Áudio WAV criado!")