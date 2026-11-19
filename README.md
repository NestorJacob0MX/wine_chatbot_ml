# SommelierAI

## Ejecutar la aplicación web local

Instala las dependencias desde la raíz del proyecto:

```bash
python -m pip install -r requirements.txt
```

Inicia el API en una terminal:

```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

El API prepara los datos, ajusta TF-IDF y entrena el modelo al iniciar. Espera
hasta que Uvicorn indique que la aplicación está lista. El pipeline se mantiene
en memoria durante la vida de ese proceso y se reutiliza en todos los mensajes
y conversaciones.

En otra terminal, desde la misma raíz del proyecto, inicia Streamlit:

```bash
streamlit run streamlit_app.py
```

Abre la dirección local que muestra Streamlit, normalmente
`http://localhost:8501`. La documentación del API está en
`http://127.0.0.1:8000/docs`.

No uses varios workers de Uvicorn para esta ejecución local: cada proceso
tendría su propio pipeline en memoria y entrenaría una copia del modelo.

## Conversación y monedas

Escribe un saludo para comenzar. El asistente preguntará si buscas un vino para
degustar o para acompañar alimentos. Para degustación pedirá el estilo o tipo de
vino; para maridaje preguntará por el platillo y adapta la pregunta a la hora
local.

Las conversaciones tienen sesiones independientes y comparten el recomendador
entrenado en el proceso del API. El idioma y los precios se manejan por sesión:
en español los precios se presentan como MXN usando la conversión fija
`1 USD = 20 MXN` y los presupuestos en pesos se convierten a USD para filtrar el
catálogo; en inglés se mantienen y se muestran en USD.
