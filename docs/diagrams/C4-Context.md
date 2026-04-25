C4Context
    title Diagrama de Contexto — Ecuador Sales MLOps

    Person(usuario, "Usuario Final", "Consulta predicciones de ventas del SRI Ecuador en lenguaje natural desde el navegador")
    Person(datascientist, "Data Scientist", "Lanza entrenamientos, monitorea métricas y gestiona el ciclo de vida del modelo")

    System(plataforma, "Ecuador Sales MLOps", "Plataforma MLOps con agente IA conversacional, pipeline de entrenamiento automatizado y monitoreo en tiempo real")

    System_Ext(groq, "Groq API", "Proveedor LLM. Modelo: llama-3.1-8b-instant. Ejecuta el razonamiento ReAct del agente")
    System_Ext(wandb, "Weights & Biases", "Tracking de experimentos ML, registro de métricas y Model Registry con alias de producción")

    Rel(usuario, plataforma, "Realiza consultas en lenguaje natural", "HTTPS · Browser")
    Rel(datascientist, plataforma, "Lanza entrenamientos y revisa dashboards", "kubectl · Browser")
    Rel(plataforma, groq, "Invoca LLM para razonamiento ReAct", "HTTPS · REST")
    Rel(plataforma, wandb, "Registra artefactos y promueve modelos a producción", "HTTPS · REST")
