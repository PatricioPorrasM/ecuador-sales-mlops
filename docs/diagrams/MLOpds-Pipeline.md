flowchart TD
    CSV[("CSV SRI Ecuador\n69 registros\nEne 2020 — Sep 2025")]

    subgraph JOB["K8s Job — model-trainer"]
        direction TB
        V1["train_v1.py\nRandomForest\ninitContainer 1"]
        V2["train_v2.py\nXGBoost\ninitContainer 2"]
        V1 --> V2
        V2 --> CMP
        CMP["compare_and_promote.py\nmain container"]
    end

    subgraph WB["Weights and Biases"]
        A1["Artifact: model-v1\nRMSE · MAE · R²"]
        A2["Artifact: model-v2\nRMSE · MAE · R²"]
        ALIAS["alias: production"]
    end

    subgraph STORAGE["PersistentVolumeClaim — models-pvc"]
        P1["model_v1.pkl"]
        P2["model_v2.pkl"]
        PPROD["model_production.pkl"]
    end

    subgraph SERVING["Inferencia — namespace: inference"]
        INF["ml-inference\nPOST /predict\nGET /ready → 503 si no cargado"]
    end

    subgraph FLUJO["Flujo de predicción"]
        direction LR
        USR(["Usuario"]) --> WUI
        WUI["web-ui\nPOST /chat"] --> AGT
        AGT["ai-agent\nReAct: validate → get_data → predict"] --> INF
    end

    subgraph OBS["Observabilidad"]
        KFK["Kafka\nuser-requests\nagent-actions\nmodel-responses"]
        PROM["Prometheus\nscraping cada 15s"]
        GRAF["Grafana\nDashboard"]
    end

    CSV --> V1
    CSV --> V2
    V1 -->|métricas| A1
    V2 -->|métricas| A2
    V1 -->|serializado| P1
    V2 -->|serializado| P2
    P1 --> CMP
    P2 --> CMP

    CMP --> DEC{RMSE v2 < RMSE v1 x 0.95?}
    DEC -->|"Si — V2 gana"| PPROD
    DEC -->|"No — V1 gana"| PPROD
    DEC -->|ganador| ALIAS

    PPROD -->|carga al arrancar| INF

    WUI -->|eventos| KFK
    AGT -->|eventos| KFK
    INF -->|eventos| KFK
    KFK --> PROM
    WUI -->|/metrics| PROM
    AGT -->|/metrics| PROM
    INF -->|/metrics| PROM
    PROM --> GRAF
