# ADR-001: Apache Kafka como Broker de Mensajería

## Metadatos

| Campo | Detalle |
|---|---|
| **ID** | ADR-001 |
| **Título** | Selección de Apache Kafka como broker de mensajería para el pipeline de eventos |
| **Estado** | Aceptado |
| **Fecha** | 2025 |
| **Autores** | Equipo de Arquitectura |
| **Revisores** | — |

---

## Contexto

El sistema requiere un mecanismo para registrar y propagar eventos a lo largo del pipeline de consulta del usuario: desde la entrada en la interfaz web, pasando por el razonamiento del agente IA, hasta la generación de la predicción por el modelo. Esta capa de eventos cumple tres propósitos simultáneos: trazabilidad de auditoría, desacoplamiento entre servicios y base para el cálculo de métricas de latencia end-to-end.

Se evaluaron tres alternativas:

1. **Apache Kafka** — broker distribuido orientado a logs, diseñado para streaming de eventos de alto volumen con retención configurable
2. **RabbitMQ** — broker de mensajería orientado a colas, con soporte de múltiples protocolos (AMQP, STOMP, MQTT)
3. **Redis Pub/Sub** — sistema de publicación/suscripción in-memory sobre Redis, sin persistencia de mensajes por defecto

---

## Decisión

Se adopta **Apache Kafka** como broker de mensajería del sistema, desplegado como contenedor Docker y administrado como pod en Kubernetes bajo el namespace `streaming`.

---

## Justificación

### Por qué Kafka sobre RabbitMQ

**Modelo de retención de mensajes.** Kafka persiste los mensajes en disco durante el tiempo configurado (por defecto 7 días), independientemente de si fueron consumidos. Esto permite que nuevos consumidores lean el historial completo de eventos sin necesidad de que el productor reenvíe los mensajes. RabbitMQ elimina los mensajes una vez que son consumidos por todos los suscriptores, lo que imposibilita el replay de eventos para análisis retrospectivo o para agregar nuevos consumidores a futuro.

**Modelo log-append.** El sistema de este proyecto trata los eventos como un log de auditoría inmutable: cada consulta del usuario, cada acción del agente y cada predicción del modelo son hechos que ocurrieron y no deben ser modificados ni eliminados tras su procesamiento. El modelo de log de Kafka es semánticamente correcto para este caso de uso. RabbitMQ está diseñado para colas de trabajo donde los mensajes son tareas a ejecutar, no eventos a registrar.

**Escalabilidad y particionamiento.** Aunque el volumen actual del sistema es bajo, Kafka permite escalar el throughput añadiendo particiones sin cambios en los productores o consumidores. RabbitMQ escala añadiendo nodos al cluster, lo que implica mayor complejidad operacional.

**Idoneidad académica.** En el contexto de una maestría en IA, Kafka representa el estándar de la industria para pipelines de datos en sistemas de machine learning en producción (usado por LinkedIn, Netflix, Uber para sus pipelines de features y monitoreo de modelos). Demostrar su uso correcto tiene mayor valor académico y profesional.

### Por qué Kafka sobre Redis Pub/Sub

**Persistencia.** Redis Pub/Sub no persiste mensajes: si un consumidor no está activo en el momento de la publicación, el mensaje se pierde permanentemente. Kafka garantiza que los mensajes estarán disponibles para ser consumidos incluso si el consumidor estaba caído temporalmente.

**Replay y auditoría.** El caso de uso de trazabilidad requiere poder reconstruir el historial de eventos de una consulta correlacionando mensajes de tres topics por `session_id`. Con Redis Pub/Sub esto es imposible una vez que los mensajes han pasado. Con Kafka, el consumidor puede retroceder en el offset y reprocesar mensajes históricos.

**Separación de responsabilidades.** Redis ya existe en muchos sistemas como caché. Usar Redis Pub/Sub mezcla dos responsabilidades en el mismo servicio, complicando la operación y el monitoreo. Kafka es un sistema dedicado exclusivamente a la mensajería.

---

## Consecuencias

### Positivas
- Trazabilidad completa y duradera del pipeline de consultas
- Posibilidad de agregar nuevos consumidores (analytics, alertas, re-entrenamiento) sin modificar los productores
- Desacoplamiento real entre productores y consumidores: si el `kafka-consumer` está caído, los productores no se ven afectados
- Métricas de lag del consumer disponibles nativamente para monitoreo en Grafana

### Negativas
- Mayor complejidad de despliegue: requiere Zookeeper (o KRaft en versiones recientes) y configuración de topics
- Mayor consumo de recursos en Minikube respecto a RabbitMQ o Redis (aproximadamente 512Mi de RAM para el broker)
- Latencia ligeramente mayor que Redis Pub/Sub para mensajes individuales (no relevante para este caso de uso)

### Neutrales
- Se usa un único nodo de Kafka (replication factor 1) por ser un entorno de desarrollo. En producción se recomendarían al menos 3 brokers para alta disponibilidad.
- Los topics se crean con 1 partición. Aumentar particiones en el futuro requiere reinicio del consumer.

---

## Alternativas Descartadas

| Alternativa | Razón de descarte |
|---|---|
| RabbitMQ | Semántica de cola de trabajo, no de log de eventos. Elimina mensajes tras consumo. |
| Redis Pub/Sub | Sin persistencia. Mensajes se pierden si el consumidor no está activo. |
| HTTP síncrono directo | Acoplamiento fuerte entre servicios. Sin trazabilidad ni capacidad de replay. |
| Amazon SQS / SNS | Servicio gestionado en la nube, no ejecutable localmente en Minikube. |
