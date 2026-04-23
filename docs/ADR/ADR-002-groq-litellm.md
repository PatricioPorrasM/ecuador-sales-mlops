# ADR-002: Groq + LiteLLM como Proveedor del Agente IA

## Metadatos

| Campo | Detalle |
|---|---|
| **ID** | ADR-002 |
| **Título** | Selección de Groq como proveedor LLM y LiteLLM como capa de abstracción para el agente IA |
| **Estado** | Aceptado |
| **Fecha** | 2025 |
| **Autores** | Equipo de Arquitectura |
| **Revisores** | — |

---

## Contexto

El sistema requiere un modelo de lenguaje grande (LLM) para potenciar el agente IA que implementa el patrón ReAct. Este agente es responsable de interpretar preguntas en lenguaje natural, razonar sobre las entidades presentes (provincia, mes), construir el payload de inferencia y formular respuestas comprensibles para el usuario. El LLM es un componente central del sistema y su selección impacta directamente en: latencia de respuesta percibida por el usuario, costo operacional, facilidad de integración y portabilidad futura.

Se evaluaron tres enfoques:

1. **Groq + LiteLLM** — Groq como proveedor de inferencia LLM con hardware LPU, LiteLLM como SDK de abstracción multi-proveedor
2. **OpenAI API directa** — Llamadas directas a `gpt-4o-mini` o `gpt-3.5-turbo` usando el SDK oficial de OpenAI
3. **Ollama local** — Modelos open source ejecutados localmente dentro de Minikube (Llama 3, Mistral)

---

## Decisión

Se adopta **Groq** como proveedor de inferencia LLM con el modelo `llama3-8b-8192`, accedido a través de **LiteLLM** como capa de abstracción, usando la configuración `model="groq/llama3-8b-8192"`.

---

## Justificación

### Por qué Groq sobre OpenAI directo

**Latencia de inferencia.** Groq utiliza hardware LPU (Language Processing Unit) propietario que entrega velocidades de inferencia significativamente superiores a las GPU convencionales usadas por OpenAI. En la práctica, para el modelo `llama3-8b-8192`, Groq genera respuestas en 1-3 segundos frente a los 3-8 segundos típicos de `gpt-3.5-turbo` en OpenAI. Para una interfaz conversacional donde el usuario espera la respuesta, esta diferencia es perceptible y mejora la experiencia.

**Costo.** Groq ofrece un tier gratuito generoso (14,400 requests/día, 6,000 tokens/minuto) que es más que suficiente para un sistema de demostración académico. OpenAI no tiene tier gratuito; cada request tiene costo directo que se acumula durante las demostraciones y pruebas.

**Modelo open source.** `llama3-8b-8192` es un modelo open source de Meta, lo que significa que su arquitectura y comportamiento son auditables y reproducibles. Para un trabajo académico, usar modelos con transparencia metodológica tiene valor epistémico. Los modelos de OpenAI son propietarios y su comportamiento exacto no es reproducible.

**Sin vendor lock-in en el modelo.** Groq ejecuta modelos estándar de la comunidad (Llama 3, Mixtral, Gemma). Si mañana Groq deja de estar disponible, el mismo modelo puede ejecutarse en otro proveedor compatible.

### Por qué LiteLLM sobre el SDK de Groq directamente

**Abstracción de proveedor.** LiteLLM provee una interfaz unificada compatible con OpenAI para más de 100 proveedores LLM. El código del agente usa `litellm.completion(model="groq/llama3-8b-8192", ...)` y si mañana se decide cambiar a `openai/gpt-4o-mini` o `anthropic/claude-3-haiku`, solo cambia el string del modelo, sin modificar el código del agente.

**Portabilidad para el evaluador.** El evaluador del trabajo puede cambiar el proveedor configurando únicamente la variable de entorno, sin entender los detalles de la integración. Esto reduce la barrera para reproducir el sistema.

**Manejo de errores unificado.** LiteLLM normaliza los errores de distintos proveedores en excepciones estándar (`litellm.exceptions.RateLimitError`, `litellm.exceptions.Timeout`), simplificando el manejo de resiliencia en el agente.

**Logging y observabilidad.** LiteLLM tiene integración nativa con callbacks de logging que pueden alimentar métricas de costo, tokens y latencia sin código adicional.

### Por qué Groq + LiteLLM sobre Ollama local

**Recursos de Minikube.** Ejecutar un modelo Llama 3 8B localmente requiere entre 8GB y 16GB de RAM solo para la inferencia del modelo. En un entorno de Minikube en Docker Desktop con recursos compartidos, esto haría que el cluster fuera inoperable. Groq mantiene la inferencia en la nube, dejando los recursos de Minikube para los servicios del sistema.

**Tiempo de setup.** Ollama requiere descargar los pesos del modelo (4-8GB dependiendo de la cuantización) antes de la primera inferencia. Groq está disponible instantáneamente con una API key. En una presentación o demostración, la dependencia de una descarga de varios GB es un riesgo operacional.

**Velocidad de respuesta.** Incluso con GPU, ejecutar un modelo localmente en el entorno de desarrollo produce latencias mayores que Groq con hardware LPU especializado.

---

## Consecuencias

### Positivas
- Latencia de inferencia baja (1-3 segundos) mejora la experiencia del usuario en el chat
- Costo cero durante desarrollo y demostración gracias al tier gratuito de Groq
- El código del agente es agnóstico al proveedor LLM gracias a LiteLLM
- Recursos de Minikube disponibles para los servicios del sistema

### Negativas
- Dependencia de conectividad a internet para el funcionamiento del agente
- El tier gratuito de Groq tiene límites de rate que podrían alcanzarse en pruebas intensivas
- Requiere gestión de una API key adicional (`GROQ_API_KEY`) como Secret en Kubernetes

### Neutrales
- El modelo `llama3-8b-8192` tiene una ventana de contexto de 8,192 tokens, suficiente para el caso de uso pero limitada si se quisiera incluir contexto histórico extenso en el prompt
- Si se necesita cambiar de proveedor, el único cambio necesario es el valor de la variable de entorno `LLM_MODEL` y la API key correspondiente

---

## Alternativas Descartadas

| Alternativa | Razón de descarte |
|---|---|
| OpenAI directa (gpt-3.5-turbo) | Costo por request, vendor lock-in, SDK no portable |
| OpenAI directa (gpt-4o-mini) | Mayor costo, mismos problemas de portabilidad |
| Ollama local (llama3:8b) | Requiere 8-16GB RAM en Minikube, descarga de pesos, latencia mayor |
| Anthropic Claude directa | Sin tier gratuito para volumen de pruebas, vendor lock-in |
| Cohere | Menor comunidad, menos documentación de integración con LiteLLM |
