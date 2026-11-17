# TacticAI — despliegue

Frontend/functions: Netlify.
Worker: servicio persistente con Docker y volumen para /data y /models.

Variables Netlify: OPENAI_API_KEY, TACTICAI_OPENAI_MODEL, TACTICAI_WORKER_URL, TACTICAI_WORKER_TOKEN.
Variables worker: TACTICAI_DATA, TACTICAI_WORKER_TOKEN y variables del detector.

No subir .env ni pesos del modelo al repositorio.