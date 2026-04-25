C4Container
    title Diagrama de Contenedores — Ecuador Sales MLOps

    Person(usuario, "Usuario Final")
    Person(datascientist, "Data Scientist")
    System_Ext(groq, "Groq API", "LLM llama-3.1-8b-instant")
    System_Ext(wandb, "Weights & Biases", "Model Registry")

    System_Boundary(k8s, "Kubernetes — Minikube") {

        Container(webui, "web-ui", "Flask · Gunicorn · NodePort 30500", "Interfaz de chat. Orquesta el flujo de consulta y publica eventos a Kafka")
        Container(aiagent, "ai-agent", "Flask · LiteLLM · ReAct", "Agente conversacional. Valida fechas, consulta datos históricos e invoca la inferencia")
        Container(mlinference, "ml-inference", "Flask · XGBoost · Gunicorn", "Sirve predicciones de ventas. Retorna valor, versión del modelo y score de confianza")
        Container(kafka, "Kafka", "Confluent CP 7.6 · StatefulSet", "Bus de eventos: user-requests · agent-actions · model-responses")
        Container(consumer, "kafka-consumer", "Python · kafka-python", "Consume los tres topics y expone métricas de latencia end-to-end a Prometheus")
        Container(prometheus, "Prometheus", "v2.51 · NodePort 30900", "Recolecta métricas de los cuatro servicios cada 15 segundos")
        Container(grafana, "Grafana", "v10.4 · NodePort 30300", "Dashboard de monitoreo en tiempo real con provisioning automático")
        Container(trainer, "model-trainer", "Python · Scikit-learn · XGBoost · K8s Job", "Entrena v1 y v2 secuencialmente, compara RMSE y copia el ganador al PVC")
        ContainerDb(pvc, "models-pvc", "PersistentVolumeClaim · 1 Gi", "Almacena model_production.pkl compartido entre el trainer y la inferencia")
    }

    Rel(usuario, webui, "Consulta en lenguaje natural", "HTTP · NodePort 30500")
    Rel(datascientist, grafana, "Monitoreo de métricas", "HTTP · NodePort 30300")
    Rel(datascientist, trainer, "Lanza entrenamiento", "kubectl apply")

    Rel(webui, aiagent, "POST /process", "HTTP interno · cluster DNS")
    Rel(webui, kafka, "Publica user-requests", "Kafka Protocol · 9092")
    Rel(aiagent, mlinference, "POST /predict", "HTTP interno · cluster DNS")
    Rel(aiagent, groq, "LLM completion — ciclo ReAct", "HTTPS")
    Rel(aiagent, kafka, "Publica agent-actions", "Kafka Protocol · 9092")
    Rel(mlinference, pvc, "Carga model_production.pkl al arrancar", "")
    Rel(mlinference, kafka, "Publica model-responses", "Kafka Protocol · 9092")
    Rel(consumer, kafka, "Consume los tres topics", "Kafka Protocol · 9092")
    Rel(prometheus, webui, "Scrape /metrics", "HTTP · 5002")
    Rel(prometheus, aiagent, "Scrape /metrics", "HTTP · 5001")
    Rel(prometheus, mlinference, "Scrape /metrics", "HTTP · 5000")
    Rel(prometheus, consumer, "Scrape /metrics", "HTTP · 8003")
    Rel(grafana, prometheus, "Consultas PromQL", "HTTP · 9090")
    Rel(trainer, pvc, "Escribe model_production.pkl", "")
    Rel(trainer, wandb, "Registra artefactos y métricas", "HTTPS")
