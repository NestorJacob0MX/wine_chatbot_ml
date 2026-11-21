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
if "show_food_camera" not in st.session_state:
    st.session_state.show_food_camera = False
if "food_input_version" not in st.session_state:
    st.session_state.food_input_version = 0
if st.session_state.pop("clear_food_inputs", False):
    st.session_state.food_input_version += 1

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
            st.session_state.show_food_camera = False
            st.session_state.clear_food_inputs = True
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
        if item.get("image"):
            st.image(item["image"], use_container_width=True)
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

def analyze_food_image(selected_image: Any) -> None:
    image_bytes = selected_image.getvalue()
    mime_type = selected_image.type or "image/jpeg"
    filename = getattr(selected_image, "name", None) or "food-image.jpg"
    st.session_state.messages.append(
        {
            "role": "user",
            "content": "Quiero una recomendación para este platillo.",
            "image": image_bytes,
        }
    )
    try:
        with st.spinner("Reconociendo el platillo y buscando un maridaje…"):
            response = requests.post(
                f"{API_URL}/api/sessions/{st.session_state.session_id}/images",
                files={"image": (filename, image_bytes, mime_type)},
                timeout=REQUEST_TIMEOUT,
            )
        response.raise_for_status()
        payload: Dict[str, Any] = response.json()
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": (
                    f"**Platillo identificado:** {payload['food_label']} "
                    f"({payload['confidence']:.0%} de confianza)\n\n"
                    f"{payload['message']}"
                ),
            }
        )
        st.session_state.closed = bool(payload["closed"])
        st.session_state.currency = payload["currency"]
        st.session_state.show_food_camera = False
        st.session_state.clear_food_inputs = True
        st.rerun()
    except requests.RequestException as exc:
        detail = str(exc)
        if isinstance(exc, requests.HTTPError) and exc.response is not None:
            try:
                detail = exc.response.json().get("detail", detail)
            except ValueError:
                pass
        st.error(f"No pude analizar la imagen: {detail}")


if not st.session_state.closed:
    st.markdown("---")
    st.markdown("### Comparte tu platillo")
    st.caption("Sube una imagen o toma una foto; usar la cámara es opcional.")
    upload_column, camera_column = st.columns(2, gap="large")

    with upload_column:
        st.markdown("#### Subir una imagen")
        uploaded_file = st.file_uploader(
            "Elige una imagen del platillo",
            type=["jpg", "jpeg", "png", "webp"],
            key=f"food_image_upload_{st.session_state.food_input_version}",
        )
        if uploaded_file is not None:
            st.image(
                uploaded_file,
                caption="Vista previa de la imagen",
                use_container_width=True,
            )
            if st.button(
                "Analizar imagen subida",
                type="primary",
                use_container_width=True,
                key="analyze_uploaded_food",
            ):
                analyze_food_image(uploaded_file)

    with camera_column:
        st.markdown("#### Tomar una foto")
        if not st.session_state.show_food_camera:
            st.caption("La cámara permanece apagada hasta que decidas activarla.")
            if st.button(
                "📷",
                help="Activar cámara",
                use_container_width=True,
                key="activate_food_camera",
            ):
                st.session_state.show_food_camera = True
                st.rerun()
        else:
            if st.button(
                "Desactivar cámara",
                use_container_width=True,
                key="deactivate_food_camera",
            ):
                st.session_state.show_food_camera = False
                st.session_state.food_input_version += 1
                st.rerun()
            camera_image = st.camera_input(
                "Captura la imagen del platillo",
                key=f"food_camera_input_{st.session_state.food_input_version}",
            )
            if camera_image is not None:
                st.image(
                    camera_image,
                    caption="Vista previa de la foto",
                    use_container_width=True,
                )
                if st.button(
                    "Analizar foto",
                    type="primary",
                    use_container_width=True,
                    key="analyze_camera_food",
                ):
                    analyze_food_image(camera_image)
