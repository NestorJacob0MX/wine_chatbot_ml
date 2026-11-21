# SommelierAI

## Ejecutar la aplicación web local

Ejecuta estos pasos desde la raíz del proyecto, donde están `requirements.txt`,
`streamlit_app.py` y las carpetas `api/` y `pipeline/`.

### 1. Crear y activar el entorno virtual

En macOS o Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

En Windows (PowerShell):

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

Activa el entorno virtual en cada terminal que uses para ejecutar el proyecto.

### 2. Instalar dependencias

Con el entorno virtual activado:

```bash
python -m pip install -r requirements.txt
```

### 3. Habilitar recomendaciones por imagen (opcional)

Para clasificar platillos mediante imágenes, entrena ResNet50 una vez:

```bash
python -m image_layer.training.train_food_classifier
```

El entrenamiento requiere el dataset `dataset/dataset_food`, puede descargar
los pesos iniciales de ResNet50 y guarda el modelo en
`artifacts/food_classifier_resnet50.pt`. Si solo necesitas recomendaciones por
texto, puedes omitir este paso.

### 4. Iniciar el API

Abre una terminal, activa el entorno virtual y ejecuta desde la raíz:

```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

El API carga y prepara el catálogo, ajusta TF-IDF, ejecuta el clustering y
carga el clasificador de imágenes si existe el checkpoint. Espera hasta que
Uvicorn indique que la aplicación está lista. Mantén esta terminal abierta.

Comprueba el estado del API en `http://127.0.0.1:8000/health`. Para usar
imágenes, la respuesta debe incluir `"image_classifier_ready": true`. La
documentación interactiva del API está en `http://127.0.0.1:8000/docs`.

### 5. Iniciar la interfaz web

Abre otra terminal, activa también el entorno virtual y ejecuta desde la raíz:

```bash
streamlit run streamlit_app.py
```

Abre la dirección local que muestra Streamlit, normalmente
`http://localhost:8501`. Desde el chat puedes escribir mensajes, subir la
imagen de un platillo o activar la cámara con el botón correspondiente.

Mantén ambas terminales abiertas mientras uses la aplicación. No inicies varias
instancias del API en el mismo puerto: si el puerto `8000` está ocupado, detén
la instancia anterior con `Ctrl+C` o configura otro puerto y actualiza
`CHATBOT_API_URL` para Streamlit. Usa un solo worker de Uvicorn para evitar
cargar una copia del pipeline por proceso.

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

## Clasificación de platillos por imagen

El dataset de imágenes está en `dataset/dataset_food`, organizado en carpetas
por clase. Para volver a entrenar ResNet50 con pesos preentrenados de ImageNet
y ajustar la última etapa del modelo:

```bash
python -m image_layer.training.train_food_classifier
```

El entrenamiento separa el 20% de cada clase para validación, aplica aumento
de datos y guarda el mejor checkpoint en
`artifacts/food_classifier_resnet50.pt`, además de métricas en un archivo JSON.
El primer entrenamiento descarga los pesos públicos de ResNet50 si no están en
la caché local de Torch. Puedes ajustar el número de épocas, batch size, ruta
del dataset y salida con `--epochs`, `--batch-size`, `--dataset` y `--output`.

El entrenamiento de la CNN se ejecuta explícitamente y no al abrir la
interfaz. Puede tardar varios minutos y descargar los pesos iniciales de
ResNet50. Después de entrenar o cambiar el checkpoint, inicia o reinicia el API:
`WineChatbotPipeline.run()` carga el checkpoint junto con las capas de datos,
NLP, clustering y recomendación, una sola vez por proceso. La UI permite subir
JPG/PNG/WebP o tomar una foto desde la cámara; el endpoint
`POST /api/sessions/{session_id}/images` devuelve la clase detectada, confianza
y recomendación. El API limita las imágenes a 10 MB. Si el checkpoint todavía
no existe, la conversación de texto funciona y `/health` reporta
`image_classifier_ready: false`.

Para usar otro checkpoint, define `FOOD_CLASSIFIER_CHECKPOINT` antes de iniciar
Uvicorn.
