# TacticAI Vision Worker

FastAPI worker persistente para vídeo. Recibe partidos, conserva proyectos, genera muestreo temporal, eventos candidatos y evidencia.

El pipeline actual es deliberadamente conservador: decodificación y muestreo son hechos; las acciones futbolísticas permanecen como candidate hasta incorporar validación específica.

## Arranque

docker build -t tacticai-worker ./worker
docker run -p 8080:8080 -v $(pwd)/data:/data --env-file .env tacticai-worker

GET /health