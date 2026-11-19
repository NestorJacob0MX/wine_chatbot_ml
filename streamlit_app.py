import os
from typing import Any, Dict

import requests
import streamlit as st

API_URL = os.getenv("CHATBOT_API_URL", "http://127.0.0.1:8000").rstrip("/")
REQUEST_TIMEOUT = 300

st.set_page_config(
    page_title="SommelierAI · Wine Chat",
    page_icon="🍷",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    :root { color-scheme: dark; }
    .stApp {
        background:
            radial-gradient(ellipse at 50% -18%, rgba(132, 72, 111, .24), transparent 48%),
            #101114;
        color: #f2f2f4;
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stMainBlockContainer"] { max-width: 860px; padding-top: 2rem; }
    .brand {
        display: flex; align-items: center; gap: .8rem;
        font-size: 1.65rem; font-weight: 650; letter-spacing: -.03em;
    }
    .brand-mark {
        display: inline-flex; align-items: center; justify-content: center;
        width: 2.5rem; height: 2.5rem; border-radius: 14px;
        background: linear-gradient(140deg, #a85579, #604b93);
        box-shadow: 0 8px 28px rgba(145, 71, 123, .24);
    }
    .subtitle { color: #a6a7af; margin: .25rem 0 1.2rem 3.3rem; }
    .welcome-card {
        border: 1px solid rgba(255,255,255,.09); border-radius: 18px;
        padding: 1.15rem 1.3rem; margin: 1.5rem 0;
        background: rgba(255,255,255,.035); color: #c9c9d0;
        line-height: 1.65;
    }
    [data-testid="stChatMessage"] {
        border: 1px solid rgba(255,255,255,.06);
        background: rgba(255,255,255,.035);
        border-radius: 18px; padding: .8rem 1rem; margin-bottom: .85rem;
    }
    [data-testid="stChatInput"] {
        border-color: rgba(255,255,255,.15); border-radius: 18px;
    }
    [data-testid="stChatInput"] textarea { color: #f2f2f4; }
    .stButton button {
        border-radius: 12px; border: 1px solid rgba(255,255,255,.14);
        background: rgba(255,255,255,.06); color: #f2f2f4;
    }
    .stButton button:hover { border-color: #b879a2; color: white; }
    .caption { color: #92939c; font-size: .82rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="brand"><span class="brand-mark">🍷</span>SommelierAI</div>'
    '<div class="subtitle">Una buena conversación merece un buen vino.</div>',
    unsafe_allow_html=True,
)


def create_session() -> str:
    response = requests.post(f"{API_URL}/api/sessions", timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return str(response.json()["session_id"])


if "messages" not in st.session_state:
    st.session_state.messages = []
if "closed" not in st.session_state:
    st.session_state.closed = False

if "session_id" not in st.session_state:
    try:
        with st.spinner("Preparando una nueva conversación…"):
            st.session_state.session_id = create_session()
    except requests.RequestException as exc:
        st.error(
            "No pude conectar con el API local. Inicia el servidor con "
            "`uvicorn api.main:app --host 127.0.0.1 --port 8000` y vuelve a cargar esta página."
        )
        st.caption(f"Detalle de conexión: {exc}")
        st.stop()

with st.sidebar:
    st.markdown("### Tu conversación")
    st.markdown(
        "Cuéntame si buscas un vino para degustar o para acompañar tu comida."
    )
    st.caption("En español los precios se muestran en MXN (1 USD = 20 MXN).")
    st.caption("In English, prices are shown in USD.")
    if st.button("＋  Nueva conversación", use_container_width=True):
        try:
            delete_response = requests.delete(
                f"{API_URL}/api/sessions/{st.session_state.session_id}",
                timeout=10,
            )
            if delete_response.status_code != 404:
                delete_response.raise_for_status()
            st.session_state.session_id = create_session()
            st.session_state.messages = []
            st.session_state.closed = False
            st.rerun()
        except requests.RequestException as exc:
            st.error(f"No pude iniciar otra conversación: {exc}")

if not st.session_state.messages:
    st.markdown(
        """
        <div class="welcome-card">
        <strong>¡Bienvenido/a!</strong><br>
        Escribe un saludo para comenzar. Te preguntaré si quieres un vino para
        degustar o para acompañar tu comida; después afinaremos la recomendación
        según tu estilo, platillo, horario y presupuesto.
        </div>
        """,
        unsafe_allow_html=True,
    )

for item in st.session_state.messages:
    with st.chat_message(item["role"]):
        st.markdown(item["content"])

if st.session_state.closed:
    st.info("La conversación terminó. Inicia una nueva desde el panel lateral cuando quieras.")

prompt = st.chat_input(
    "Escribe un mensaje…",
    disabled=st.session_state.closed,
)

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    try:
        response = requests.post(
            f"{API_URL}/api/sessions/{st.session_state.session_id}/messages",
            json={"message": prompt},
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        payload: Dict[str, Any] = response.json()
        st.session_state.messages.append(
            {"role": "assistant", "content": payload["message"]}
        )
        st.session_state.closed = bool(payload["closed"])
        st.session_state.currency = payload["currency"]
        st.rerun()
    except requests.RequestException as exc:
        detail = str(exc)
        if isinstance(exc, requests.HTTPError) and exc.response is not None:
            try:
                detail = exc.response.json().get("detail", detail)
            except ValueError:
                pass
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": f"No pude procesar el mensaje: {detail}",
            }
        )
        st.rerun()
