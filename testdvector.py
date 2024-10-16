import torch
import torchaudio

wav2mel = torch.jit.load("wav2mel.pt")
dvector = torch.jit.load("dvector.pt").eval()

wav_tensor, sample_rate = torchaudio.load("example.wav")
mel_tensor = wav2mel(wav_tensor, sample_rate)  # shape: (frames, mel_dim)
emb_tensor = dvector.embed_utterance(mel_tensor)  # shape: (emb_dim)