#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# deploy-all.sh — Despliega todo el stack en Minikube en orden correcto
#
# Orden de despliegue:
#   1. namespaces   — base de todo
#   2. storage      — PVs y PVCs (modelos compartidos)
#   3. streaming    — Zookeeper → Kafka → topics-job → kafka-consumer
#   4. inference    — ml-inference (espera a que el PVC esté Bound)
#   5. agent        — ai-agent
#   6. frontend     — web-ui
#   7. monitoring   — Prometheus + Grafana
#   8. training     — model-trainer Job (opcional, con --train)
#
# Uso:
#   bash scripts/deploy-all.sh            # despliega todo (sin el Job de training)
#   bash scripts/deploy-all.sh --train    # incluye el Job de training al final
#   bash scripts/deploy-all.sh --skip-wait  # sin esperar readiness (más rápido)
#
# Prerequisitos:
#   - Minikube corriendo
#   - Imágenes compiladas (bash scripts/build-images.sh)
#   - Secrets creados (bash scripts/setup-minikube.sh)
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BLUE='\033[0;34m'
BOLD='\033[1m'
RESET='\033[0m'

step()    { echo -e "\n${CYAN}${BOLD}[$((++STEP_N))/8] $*${RESET}"; }
ok()      { echo -e "  ${GREEN}✓${RESET} $*"; }
warn()    { echo -e "  ${YELLOW}⚠${RESET}  $*"; }
die()     { echo -e "\n${RED}✗ ERROR: $*${RESET}\n" >&2; exit 1; }
info()    { echo -e "  ${BLUE}→${RESET} $*"; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STEP_N=0
WITH_TRAIN=false
SKIP_WAIT=false

# ── Parsear argumentos ────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --train)      WITH_TRAIN=true;  shift ;;
    --skip-wait)  SKIP_WAIT=true;   shift ;;
    *) die "Argumento desconocido: $1" ;;
  esac
done

# ── Helper: esperar rollout de un Deployment ──────────────────────────────────
wait_rollout() {
  local kind="$1" name="$2" ns="$3"
  local timeout="${4:-120s}"

  if [[ "$SKIP_WAIT" == true ]]; then
    info "skip-wait activo — no esperando $kind/$name"
    return 0
  fi

  info "Esperando rollout de $kind/$name en $ns (timeout $timeout)..."
  if kubectl rollout status "$kind/$name" -n "$ns" --timeout="$timeout"; then
    ok "$name listo"
  else
    warn "$name no completó el rollout en $timeout — continúa de todos modos"
  fi
}

# ── Helper: esperar condición de un pod ───────────────────────────────────────
wait_pods_ready() {
  local label="$1" ns="$2" timeout="${3:-120s}"

  [[ "$SKIP_WAIT" == true ]] && return 0

  info "Esperando pods con label $label en $ns..."
  kubectl wait pod \
    --for=condition=ready \
    -l "$label" \
    -n "$ns" \
    --timeout="$timeout" 2>/dev/null \
    && ok "Pods listos" \
    || warn "Timeout esperando pods $label — continúa de todos modos"
}

# ── Helper: esperar Job completo ──────────────────────────────────────────────
wait_job() {
  local job="$1" ns="$2" timeout="${3:-300s}"

  [[ "$SKIP_WAIT" == true ]] && return 0

  info "Esperando Job $job en $ns (timeout $timeout)..."
  kubectl wait job/"$job" \
    --for=condition=complete \
    -n "$ns" \
    --timeout="$timeout" 2>/dev/null \
    && ok "Job $job completado" \
    || warn "Job $job no completó en $timeout — revisa: kubectl logs -n $ns job/$job"
}

# ── Verificaciones previas ────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo -e "${BOLD}  Ecuador Sales MLOps — Deploy en Minikube${RESET}"
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"

command -v kubectl &>/dev/null || die "kubectl no encontrado en el PATH"
STATUS=$(minikube status --format='{{.Host}}' 2>/dev/null || echo "")
[[ "$STATUS" == "Running" ]] || die "Minikube no está corriendo"
ok "Cluster: $(kubectl config current-context) | Node: $(minikube ip)"
[[ "$WITH_TRAIN" == true ]] && info "Modo --train activo: el Job de training se ejecutará al final"

# ═══════════════════════════════════════════════════════════════════════════════
# PASO 1 — Namespaces
# ═══════════════════════════════════════════════════════════════════════════════
step "Namespaces"

kubectl apply -f "$REPO_ROOT/k8s/namespaces.yaml"
ok "streaming, frontend, agent, inference, monitoring, training"

# ═══════════════════════════════════════════════════════════════════════════════
# PASO 2 — Storage (ConfigMaps, Secrets, PVCs)
# ═══════════════════════════════════════════════════════════════════════════════
step "Storage y configuración"

kubectl apply -f "$REPO_ROOT/k8s/configmap-global.yaml"
ok "ConfigMaps aplicados"

kubectl apply -f "$REPO_ROOT/k8s/pvc-models.yaml"
ok "PersistentVolumes y PVCs aplicados"

# Verificar que los Secrets existen (setup-minikube.sh los crea)
for ns_secret in "training/api-keys" "agent/api-keys"; do
  ns="${ns_secret%%/*}"; sname="${ns_secret##*/}"
  if ! kubectl get secret "$sname" -n "$ns" &>/dev/null; then
    warn "Secret $sname no encontrado en $ns — ejecuta setup-minikube.sh primero"
  else
    ok "Secret $sname en $ns existe"
  fi
done

# Esperar a que los PVCs estén Bound
if [[ "$SKIP_WAIT" == false ]]; then
  for ns_pvc in "inference/models-pvc" "training/models-pvc"; do
    ns="${ns_pvc%%/*}"; pvc="${ns_pvc##*/}"
    for i in $(seq 1 20); do
      PHASE=$(kubectl get pvc "$pvc" -n "$ns" -o jsonpath='{.status.phase}' 2>/dev/null || echo "Pending")
      [[ "$PHASE" == "Bound" ]] && { ok "PVC $pvc ($ns) → Bound"; break; }
      [[ $i -eq 20 ]] && warn "PVC $pvc ($ns) sigue en $PHASE — continúa de todos modos"
      sleep 3
    done
  done
fi

# ═══════════════════════════════════════════════════════════════════════════════
# PASO 3 — Streaming (Zookeeper → Kafka → Topics → Consumer)
# ═══════════════════════════════════════════════════════════════════════════════
step "Streaming (Kafka)"

kubectl apply -f "$REPO_ROOT/k8s/kafka/zookeeper.yaml"
ok "Zookeeper aplicado"

kubectl apply -f "$REPO_ROOT/k8s/kafka/kafka.yaml"
ok "Kafka StatefulSet aplicado"

wait_pods_ready "app=zookeeper" "streaming" "90s"
wait_pods_ready "app=kafka"     "streaming" "180s"

# Re-crear el topics-job si ya existe completado/fallido
kubectl delete job kafka-topics-init -n streaming --ignore-not-found=true
kubectl apply -f "$REPO_ROOT/k8s/kafka/topics-job.yaml"
ok "topics-job aplicado"
wait_job "kafka-topics-init" "streaming" "120s"

kubectl apply -f "$REPO_ROOT/k8s/kafka-consumer/deployment.yaml"
ok "kafka-consumer aplicado"
wait_rollout "deployment" "kafka-consumer" "streaming" "90s"

# ═══════════════════════════════════════════════════════════════════════════════
# PASO 4 — Inference (ml-inference)
# ═══════════════════════════════════════════════════════════════════════════════
step "Inference (ml-inference)"

kubectl apply -f "$REPO_ROOT/k8s/ml-inference/deployment.yaml"
ok "ml-inference Deployment y Service aplicados"
wait_rollout "deployment" "ml-inference" "inference" "120s"

# ═══════════════════════════════════════════════════════════════════════════════
# PASO 5 — Agent (ai-agent)
# ═══════════════════════════════════════════════════════════════════════════════
step "Agent (ai-agent)"

kubectl apply -f "$REPO_ROOT/k8s/ai-agent/deployment.yaml"
ok "ai-agent Deployment y Service aplicados"
wait_rollout "deployment" "ai-agent" "agent" "120s"

# ═══════════════════════════════════════════════════════════════════════════════
# PASO 6 — Frontend (web-ui)
# ═══════════════════════════════════════════════════════════════════════════════
step "Frontend (web-ui)"

kubectl apply -f "$REPO_ROOT/k8s/web-ui/deployment.yaml"
ok "web-ui Deployment y NodePort 30500 aplicados"
wait_rollout "deployment" "web-ui" "frontend" "90s"

# ═══════════════════════════════════════════════════════════════════════════════
# PASO 7 — Monitoring (Prometheus + Grafana)
# ═══════════════════════════════════════════════════════════════════════════════
step "Monitoring (Prometheus + Grafana)"

kubectl apply -f "$REPO_ROOT/k8s/monitoring/prometheus.yaml"
ok "Prometheus ConfigMap, Deployment y NodePort 30900 aplicados"

kubectl apply -f "$REPO_ROOT/k8s/monitoring/grafana.yaml"
ok "Grafana ConfigMaps, Deployment y NodePort 30300 aplicados"

wait_rollout "deployment" "prometheus" "monitoring" "90s"
wait_rollout "deployment" "grafana"    "monitoring" "90s"

# ═══════════════════════════════════════════════════════════════════════════════
# PASO 8 — Training Job (solo con --train)
# ═══════════════════════════════════════════════════════════════════════════════
step "Training"

if [[ "$WITH_TRAIN" == true ]]; then
  # Borrar job previo si existe
  kubectl delete job model-trainer -n training --ignore-not-found=true
  kubectl apply -f "$REPO_ROOT/k8s/training/model-trainer-job.yaml"
  ok "model-trainer Job lanzado"
  info "Sigue el progreso con:"
  info "  kubectl logs -n training job/model-trainer -c train-v1 -f"
  info "  kubectl logs -n training job/model-trainer -c train-v2 -f"
  info "  kubectl logs -n training job/model-trainer -c compare-and-promote -f"
  wait_job "model-trainer" "training" "600s"
else
  info "Skipping training Job (usa --train para ejecutarlo)"
  info "  bash scripts/deploy-all.sh --train"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# RESUMEN FINAL — URLs de acceso
# ═══════════════════════════════════════════════════════════════════════════════
MINIKUBE_IP=$(minikube ip 2>/dev/null || echo "<minikube-ip>")

echo ""
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo -e "${GREEN}${BOLD}  Deploy completado${RESET}"
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo ""
echo -e "  ${BOLD}URLs de acceso:${RESET}"
echo ""
echo -e "  ${GREEN}●${RESET} Web UI (chat)      http://${MINIKUBE_IP}:30500"
echo -e "  ${BLUE}●${RESET} Grafana            http://${MINIKUBE_IP}:30300  (admin/admin)"
echo -e "  ${BLUE}●${RESET} Prometheus         http://${MINIKUBE_IP}:30900"
echo ""
echo -e "  ${BOLD}Servicios internos (kubectl port-forward):${RESET}"
echo ""
echo -e "  kubectl port-forward -n inference svc/ml-inference 5000:5000"
echo -e "  kubectl port-forward -n agent     svc/ai-agent     5001:5001"
echo -e "  kubectl port-forward -n streaming svc/kafka-consumer 8003:8003"
echo ""
echo -e "  ${BOLD}Estado del cluster:${RESET}"
echo ""
kubectl get deployments -A \
  --field-selector="metadata.namespace!=kube-system" \
  -o custom-columns='NAMESPACE:.metadata.namespace,NAME:.metadata.name,READY:.status.readyReplicas,DESIRED:.spec.replicas' \
  2>/dev/null || true
echo ""
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo ""
