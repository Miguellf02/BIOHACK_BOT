# Deep-Biohacking Tracker API & Bot

Pipeline híbrido de Machine Learning local e Inteligencia Artificial Estructurada diseñado para la monitorización continua de series temporales biométricas, estimación de fatiga neuromuscular (RPE) y optimización del split de entrenamiento.

Este proyecto se encuentra en fase de desarrollo activo (MVP/Prueba de concepto). No está diseñado ni optimizado para entornos de producción de alta concurrencia. Su arquitectura está orientada a la ejecución local con persistencia en archivos JSON.

---

## Estructura del Archivo CSV

El sistema requiere una ventana móvil mínima de 14 observaciones para calcular coeficientes estables sin sobreajuste. Al inicializar el sistema, se debe subir un archivo con extensión `.csv` que cumpla estrictamente con la estructura descrita a continuación.

### Columnas Obligatorias (Separadas por comas)
* `fecha`: Cadena en formato ISO 8601 (`AAAA-MM-DD`).
* `horas_sueno`: Valor decimal entre `0.0` y `24.0`.
* `calidad_sueno`: Valor entero entre `1` y `10`.
* `proteinas_g`: Valor decimal equivalente a los gramos consumidos.
* `carbohidratos_g`: Valor decimal equivalente a los gramos consumidos.
* `grasas_g`: Valor decimal equivalente a los gramos consumidos.
* `tipo_entrenamiento`: Cadena restringida estrictamente a los valores `Fuerza`, `Boxeo` o `Descanso`.
* `volumen_total_kg`: Valor decimal o entero con el tonelaje total levantado.
* `rpe`: Valor entero entre `1` y `10` (Rate of Perceived Exertion).

### Ejemplo de Contenido Válido
Cree un archivo de texto plano, llámelo `entrenamiento.csv` y asegúrese de que contenga un mínimo de 14 líneas de datos siguiendo este formato:

```csv
fecha,horas_sueno,calidad_sueno,proteinas_g,carbohidratos_g,grasas_g,tipo_entrenamiento,volumen_total_kg,rpe
2026-05-01,8.2,9,170,350,75,Fuerza,4500,5
2026-05-02,8.0,8,165,320,70,Fuerza,4600,5
2026-05-03,8.5,9,175,360,80,Fuerza,4800,4
2026-05-04,8.0,9,165,340,75,Descanso,0,3
2026-05-05,8.1,8,170,330,72,Fuerza,4700,5
2026-05-06,8.3,9,170,350,75,Boxeo,0,4
2026-05-07,8.7,10,180,380,80,Descanso,0,2
2026-05-08,8.2,9,168,340,74,Fuerza,4900,4
2026-05-09,8.0,8,165,330,70,Fuerza,5000,4
2026-05-10,8.4,9,175,360,78,Fuerza,5100,4
2026-05-11,8.1,9,170,350,75,Boxeo,0,4
2026-05-12,8.5,9,172,360,76,Descanso,0,3
2026-05-13,8.3,9,170,340,72,Fuerza,4800,4
2026-05-14,8.6,10,175,350,75,Descanso,0,2

 Instrucciones de Ejecución
El proyecto se encuentra completamente contenerizado mediante Docker y su despliegue se gestiona a través de Docker Compose.

1. Configuración de Variables de Entorno
Cree un archivo llamado .env en la raíz del proyecto. Este archivo contiene credenciales confidenciales y está excluido del control de versiones mediante el archivo .gitignore.

Fragmento de código
TELEGRAM_BOT_TOKEN=tu_token_de_telegram_aqui
OPENAI_API_KEY=tu_api_key_de_openai_aqui
2. Despliegue de la Infraestructura
Para compilar las imágenes a través de la estrategia multi-stage y levantar los servicios en segundo plano, ejecute el siguiente comando en la terminal:

Bash
docker compose up --build
Este comando expone de manera automática:

El servicio api (FastAPI) en el puerto 8000.

El servicio bot (Telegram Bot Engine) en modo long polling activo, mapeando el volumen local ./user_data para garantizar la persistencia de los registros JSON.

3. Interacción con el Bot
Acceda al chat del bot en Telegram y envíe el comando /start para recibir el bloque informativo.

Adjunte el archivo .csv configurado previamente como un documento dentro del chat. El sistema procesará el pipeline analítico y devolverá el reporte de rendimiento inicial.

Para la actualización diaria continua, ejecute el comando interactivo /hoy sin necesidad de volver a subir un archivo.

Sintaxis del comando:

Plaintext
/hoy [sueño_hrs] [calidad_1-10] [proteina_g] [carbo_g] [grasa_g] [Fuerza/Boxeo/Descanso] [volumen_kg] [rpe_1-10]
Ejemplo práctico:

Plaintext
/hoy 8.5 9 175 320 75 Fuerza 4800 4