"""
MVP -- Sistema de Deteccion de Lavado de Dinero en Remesas
Proyecto 2, CC3092 Deep Learning

Interfaz Streamlit que carga los modelos entrenados (Etapa A: autoencoder LSTM,
Etapa B: clasificador con atencion) y explica una alerta para cualquier cuenta
del conjunto de prueba.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import matplotlib.pyplot as plt

BASE = Path(__file__).parent
DATA_DIR = BASE / "data"
MODELS_DIR = DATA_DIR / "models"
TYPE_NAMES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

st.set_page_config(page_title="Detector de Lavado de Dinero en Remesas", page_icon="\U0001F50D", layout="wide")


# ---------------------------------------------------------------------------
# Modelos (mismas clases que en el Componente 2 / 3)
# ---------------------------------------------------------------------------
def sequence_mask(lengths, T, device):
    return (torch.arange(T, device=device)[None, :] < lengths[:, None].to(device)).float()


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, lengths):
        B, T, _ = x.shape
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.encoder(packed)
        z = h_n[-1]
        decoder_input = z.unsqueeze(1).repeat(1, T, 1)
        packed_dec_in = pack_padded_sequence(decoder_input, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_dec_out, _ = self.decoder(packed_dec_in)
        out, _ = pad_packed_sequence(packed_dec_out, batch_first=True, total_length=T)
        recon = self.output_layer(out)
        return recon, z


class AttnTwoStageClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, use_anomaly_score=True):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.attn = nn.Linear(hidden_dim, 1)
        self.use_anomaly_score = use_anomaly_score
        head_in = hidden_dim + (1 if use_anomaly_score else 0)
        self.head = nn.Sequential(nn.Linear(head_in, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1))

    def pool(self, x, lengths):
        T = x.shape[1]
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.encoder(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=T)
        mask = sequence_mask(lengths, T, x.device)
        scores = self.attn(out).squeeze(-1)
        scores = scores.masked_fill(mask == 0, float("-inf"))
        alpha = torch.softmax(scores, dim=1)
        context = (alpha.unsqueeze(-1) * out).sum(dim=1)
        return context, alpha

    def forward_with_attention(self, x, lengths, anomaly_score=None):
        context, alpha = self.pool(x, lengths)
        z = torch.cat([context, anomaly_score.unsqueeze(1)], dim=1) if self.use_anomaly_score else context
        return self.head(z).squeeze(1), alpha


# ---------------------------------------------------------------------------
# Carga de datos y modelos (cacheada)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_models():
    with open(MODELS_DIR / "config.json") as f:
        cfg = json.load(f)
    ae = LSTMAutoencoder(cfg["INPUT_DIM"], cfg["HIDDEN_DIM"])
    ae.load_state_dict(torch.load(MODELS_DIR / "stage_a_autoencoder.pt", map_location="cpu"))
    ae.eval()
    clf = AttnTwoStageClassifier(cfg["INPUT_DIM"], cfg["HIDDEN_DIM"], use_anomaly_score=True)
    clf.load_state_dict(torch.load(MODELS_DIR / "stage_b_attention.pt", map_location="cpu"))
    clf.eval()
    return ae, clf, cfg


@st.cache_resource
def load_data(_cfg):
    """Carga eventos una sola vez y precomputa solo lo barato (matriz de features vectorizada
    + indice de posiciones por cuenta). Las secuencias por cuenta se arman bajo demanda
    (ver get_account_data) para que el primer arranque de la app sea rapido con 105K cuentas."""
    events = pd.read_parquet(DATA_DIR / "account_events.parquet")
    labels = pd.read_parquet(DATA_DIR / "account_labels_splits.parquet")
    events = events.sort_values(["account", "step"]).reset_index(drop=True)

    N_TYPES = _cfg["N_TYPES"]
    FEATURE_COLS = _cfg["FEATURE_COLS"]
    type_onehot = np.zeros((len(events), N_TYPES), dtype=np.float32)
    type_onehot[np.arange(len(events)), events["type_code"].values] = 1.0
    feat_matrix = np.concatenate([
        type_onehot,
        events[["direction"]].values.astype(np.float32),
        events[FEATURE_COLS].values.astype(np.float32),
        events[["near_threshold"]].values.astype(np.float32),
    ], axis=1)

    account_row_idx = events.groupby("account", sort=False).indices
    raw_cols = events[["step", "direction", "type_code", "amount", "isFraud", "near_threshold"]].to_numpy()
    raw_col_names = ["step", "direction", "type_code", "amount", "isFraud", "near_threshold"]

    return labels, feat_matrix, raw_cols, raw_col_names, account_row_idx


def get_account_data(feat_matrix, raw_cols, raw_col_names, account_row_idx, max_len, acc):
    idx = account_row_idx[acc]
    seq = feat_matrix[idx]
    raw = pd.DataFrame(raw_cols[idx], columns=raw_col_names)
    if len(seq) > max_len:
        seq = seq[-max_len:]
        raw = raw.iloc[-max_len:].reset_index(drop=True)
    return seq, raw


def get_tensors(seq):
    x = torch.from_numpy(seq).unsqueeze(0)
    length = torch.tensor([seq.shape[0]])
    return x, length


def run_inference(ae, clf, cfg, seq):
    x, lengths = get_tensors(seq)
    with torch.no_grad():
        recon, _ = ae(x, lengths)
        se = ((recon - x) ** 2).mean(dim=-1)
        mask = sequence_mask(lengths, x.shape[1], x.device)
        err_t = (se * mask).squeeze(0).numpy()[: lengths.item()]
        raw_score = err_t.sum() / lengths.item()
        score_z = (raw_score - cfg["recon_error_mean_train"]) / cfg["recon_error_std_train"]
        logit, alpha = clf.forward_with_attention(x, lengths, torch.tensor([score_z], dtype=torch.float32))
        prob = torch.sigmoid(logit).item()
        alpha = alpha.squeeze(0).numpy()[: lengths.item()]
    return prob, raw_score, score_z, err_t, alpha


TYPOLOGY_HINTS = {
    "near_threshold": "un monto justo debajo del umbral de reporte obligatorio de $10,000 (posible estructuracion/smurfing)",
    "high_freq": "una concentracion inusual de transacciones en poco tiempo (posible layering/capas)",
    "large_amount": "un monto atipicamente alto respecto al resto del historial de la cuenta",
}


def generate_explanation(acc, raw, alpha, prob, score_z, n_events):
    top_idx = int(np.argmax(alpha))
    txn = raw.iloc[top_idx]
    tipo = TYPE_NAMES[int(txn["type_code"])]
    direccion = "recibio" if txn["direction"] == 1 else "envio"
    veredicto = "sospechosa de lavado de dinero" if prob >= 0.5 else "dentro de un patron normal"

    razones = []
    if txn["near_threshold"] == 1:
        razones.append(TYPOLOGY_HINTS["near_threshold"])
    if n_events >= 10:
        razones.append(TYPOLOGY_HINTS["high_freq"])
    if txn["amount"] > raw["amount"].median() * 3:
        razones.append(TYPOLOGY_HINTS["large_amount"])
    razon_txt = "; ".join(razones) if razones else "un patron que se desvia del comportamiento tipico de la cuenta"

    return (
        f"La cuenta **{acc}** fue clasificada como **{veredicto}** con una probabilidad de **{prob:.0%}** "
        f"(score de anomalia de la Etapa A: {score_z:.1f} desviaciones estandar sobre el promedio de cuentas normales). "
        f"La transaccion que mas peso en esta alerta fue una operacion de tipo **{tipo}** donde la cuenta {direccion} "
        f"**${txn['amount']:,.2f}** (transaccion {top_idx + 1} de {n_events} en su historial). "
        f"Esta transaccion presenta {razon_txt}."
    )


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("\U0001F50D Detector de Lavado de Dinero en Remesas")
st.caption("Proyecto 2 -- CC3092 Deep Learning. Sistema de deteccion en dos etapas: autoencoder (Etapa A) + clasificador con atencion (Etapa B).")

ae, clf, cfg = load_models()
labels, feat_matrix, raw_cols, raw_col_names, account_row_idx = load_data(cfg)
test_accs = labels.loc[labels["split"] == "test", "account"].tolist()

st.sidebar.header("Seleccionar cuenta")
try:
    selected_cases = pd.read_parquet(DATA_DIR / "selected_cases.parquet")
    quick_options = {f"{r.caso} -- {r.account} ({r.outcome})": r.account for r in selected_cases.itertuples()}
except FileNotFoundError:
    quick_options = {}

mode = st.sidebar.radio("Modo de seleccion", ["Casos del reporte", "Buscar por ID", "Aleatoria"])

if mode == "Casos del reporte" and quick_options:
    label_sel = st.sidebar.selectbox("Caso", list(quick_options.keys()))
    account_id = quick_options[label_sel]
elif mode == "Buscar por ID":
    account_id = st.sidebar.selectbox("Cuenta (conjunto de prueba)", sorted(test_accs))
else:
    if "random_acc" not in st.session_state or st.sidebar.button("\U0001F3B2 Elegir otra cuenta aleatoria"):
        st.session_state["random_acc"] = np.random.choice(test_accs)
    account_id = st.session_state["random_acc"]

st.sidebar.markdown(f"**Cuenta seleccionada:** `{account_id}`")
st.sidebar.markdown(f"**Total cuentas de prueba:** {len(test_accs):,}")
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Umbral Etapa A:** score de anomalia > "
    f"{cfg['best_threshold_stage_a']:.3f} (elegido maximizando F1 en validacion)."
)

# --- Inference ---
seq, raw = get_account_data(feat_matrix, raw_cols, raw_col_names, account_row_idx, cfg["MAX_SEQ_LEN"], account_id)
prob, raw_score, score_z, err_t, alpha = run_inference(ae, clf, cfg, seq)
n_events = len(raw)
real_label = int(labels.set_index("account").loc[account_id, "has_fraud"])

col1, col2, col3, col4 = st.columns(4)
col1.metric("Etiqueta real", "SOSPECHOSA" if real_label else "normal")
col2.metric("Prob. de lavado (Etapa B)", f"{prob:.1%}")
col3.metric("Score anomalia (Etapa A, z)", f"{score_z:.2f}")
col4.metric("N. transacciones", n_events)

if prob >= 0.5:
    st.error("⚠️ ALERTA: el sistema clasifica esta cuenta como sospechosa de lavado de dinero.")
else:
    st.success("✅ El sistema no encuentra evidencia suficiente para marcar esta cuenta como sospechosa.")

st.markdown("### Explicacion en lenguaje natural")
st.markdown(generate_explanation(account_id, raw, alpha, prob, score_z, n_events))

st.markdown("### Secuencia de transacciones y mapa de calor")
display_df = raw.copy()
display_df["step"] = display_df["step"].astype(int)
display_df["tipo"] = display_df["type_code"].apply(lambda t: TYPE_NAMES[int(t)])
display_df["direccion"] = display_df["direction"].astype(int).map({0: "saliente", 1: "entrante"})
display_df["peso_atencion"] = alpha
display_df["error_reconstruccion"] = err_t
display_df["cerca_umbral"] = display_df["near_threshold"].astype(int).map({0: "", 1: "⚠️"})
display_df["fraude_marcado"] = display_df["isFraud"].astype(int).map({0: "", 1: "⚠️"})
display_df = display_df[["step", "tipo", "direccion", "amount", "cerca_umbral", "fraude_marcado", "peso_atencion", "error_reconstruccion"]]
display_df.columns = ["Step", "Tipo", "Direccion", "Monto", "Cerca umbral $10k", "Fraude real", "Peso atencion", "Error reconstruccion"]

st.dataframe(
    display_df.style.background_gradient(subset=["Peso atencion"], cmap="Reds")
    .background_gradient(subset=["Error reconstruccion"], cmap="Oranges")
    .format({"Monto": "${:,.2f}", "Peso atencion": "{:.3f}", "Error reconstruccion": "{:.3f}"}),
    use_container_width=True,
)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 4), sharex=True)
colors = ["#C44E52" if f == 1 else "#4C72B0" for f in raw["isFraud"]]
ax1.bar(range(n_events), alpha, color=colors)
ax1.set_ylabel("peso atencion")
ax2.bar(range(n_events), err_t, color=colors)
ax2.set_ylabel("error recon.")
ax2.set_xlabel("posicion en la secuencia (mas reciente a la derecha)")
plt.tight_layout()
st.pyplot(fig)

st.caption(
    "Barras rojas = transaccion marcada como fraude en los datos. "
    "Cuando el peso de atencion (Etapa B) y el error de reconstruccion (Etapa A) coinciden en la misma transaccion, "
    "ambas senales del sistema respaldan la misma explicacion."
)

with st.expander("Sobre este sistema"):
    st.markdown(
        "- **Etapa A:** autoencoder LSTM entrenado solo con cuentas normales; el error de reconstruccion es el score de anomalia.\n"
        "- **Etapa B:** clasificador con atencion, con transfer learning desde la Etapa A, combinado con el score de anomalia.\n"
        "- Ver el notebook `02_componente2_deteccion_dos_etapas.ipynb` para el experimento de ablacion: en este dataset, "
        "el sistema de dos etapas no supera a un clasificador entrenado desde cero en las metricas agregadas, pero preserva "
        "interpretabilidad (dos senales independientes) y la capacidad de operar sin etiquetas.\n"
        "- Dataset: PaySim (secuencias construidas a nivel de **cuenta**, no de remitente -- ver Componente 1)."
    )
