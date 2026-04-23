#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# setup-minikube.sh — Prepara el cluster Minikube para Ecuador Sales MLOps
#
# Qué hace:
#   1. Verifica que Minikube está corriendo
#   2. Habilita addons: ingress y metrics-server
#   3. Crea los 6 namespaces del proyecto
#   4. Aplica ConfigMaps globales
#   5. Crea los Secrets (requiere WANDB_API_KEY y GROQ_API_KEY en el entorno)
#   6. Crea los PersistentVolumes y PersistentVolumeClaims para modelos
#
# Uso:
#   export WANDB_API_KEY="tu_clave"
#   export GROQ_API_KEY="tu_clave"
#   bash scripts/setup-minikube.sh
#
# Prerequisitos:
#   - minikube instalado y con un perfil activo
#   - kubectl en el PATH
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Colores ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

step()  { echo -e "\n${CYAN}${BOLD}▶ $*${RESET}"; }
ok()    { echo -e "  ${GREEN}✓${RESET} $*"; }
warn()  { echo -e "  ${YELLOW}⚠${RESET}  $*"; }
die()   { echo -e "\n${RED}✗ ERROR: $*${RESET}\n" >&2; exit 1; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ── 1. Verificar Minikube ─────────────────────────────────────────────────────
step "Verificando Minikube"

command -v minikube &>/dev/null || die "minikube no encontrado en el PATH"
command -v kubectl  &>/dev/null || die "kubectl no encontrado en el PATH"

STATUS=$(minikube status --format='{{.Host}}' 2>/dev/null || echo "")
if [[ "$STATUS" != "Running" ]]; then
  warn "Minikube no está corriendo. Iniciando..."
  minikube start --cpus=4 --memory=6g --disk-size=20g
  ok "Minikube iniciado"
else
  ok "Minikube ya está corriendo ($(minikube ip))"
fi

# ── 2. Habilitar addons ───────────────────────────────────────────────────────
step "Habilitando addons"

for addon in ingress metrics-server; do
  ENABLED=$(minikube addons list -o json 2>/dev/null \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('$addon',{}).get('Status',''))" 2>/dev/null || echo "")
  if [[ "$ENABLED" == "enabled" ]]; then
    ok "addon $addon ya habilitado"
  else
    minikube addons enable "$addon"
    ok "addon $addon habilitado"
  fi
done

# ── 3. Crear namespaces ───────────────────────────────────────────────────────
step "Creando namespaces"

kubectl apply -f "$REPO_ROOT/k8s/namespaces.yaml"
ok "Namespaces aplicados: streaming, frontend, agent, inference, monitoring, training"

# Esperar a que todos los namespaces estén activos
for ns in streaming frontend agent inference monitoring training; do
  kubectl wait --for=jsonpath='{.status.phase}'=Active \
    namespace/"$ns" --timeout=30s &>/dev/null
  ok "namespace/$ns → Active"
done

# ── 4. Aplicar ConfigMaps globales ────────────────────────────────────────────
step "Aplicando ConfigMaps"

kubectl apply -f "$REPO_ROOT/k8s/configmap-global.yaml"
ok "ConfigMaps aplicados en: frontend, agent, inference, streaming, training"

# ── 5. Crear Secrets con claves de API ────────────────────────────────────────
step "Creando Secrets de API"

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  die "WANDB_API_KEY no está definida. Ejecuta: export WANDB_API_KEY='tu_clave'"
fi
if [[ -z "${GROQ_API_KEY:-}" ]]; then
  die "GROQ_API_KEY no está definida. Ejecuta: export GROQ_API_KEY='tu_clave'"
fi

# Upsert idempotente: --dry-run=client genera el YAML, apply lo aplica o actualiza
kubectl create secret generic api-keys \
  --from-literal=WANDB_API_KEY="$WANDB_API_KEY" \
  --namespace=training \
  --dry-run=client -o yaml | kubectl apply -f -
ok "Secret api-keys creado/actualizado en namespace training"

kubectl create secret generic api-keys \
  --from-literal=GROQ_API_KEY="$GROQ_API_KEY" \
  --namespace=agent \
  --dry-run=client -o yaml | kubectl apply -f -
ok "Secret api-keys creado/actualizado en namespace agent"

# ── 6. Crear directorio en el nodo y PVCs ─────────────────────────────────────
step "Creando almacenamiento para modelos"

# El directorio HostPath debe existir en el nodo Minikube antes de crear los PVs
minikube ssh -- sudo mkdir -p /mnt/data/ecuador-models
minikube ssh -- sudo chmod 777 /mnt/data/ecuador-models
ok "Directorio /mnt/data/ecuador-models creado en el nodo Minikube"

kubectl apply -f "$REPO_ROOT/k8s/pvc-models.yaml"
ok "PersistentVolumes y PersistentVolumeClaims aplicados"

# Esperar a que los PVCs se vinculen
for ns_pvc in "inference/models-pvc" "training/models-pvc"; do
  ns="${ns_pvc%%/*}"
  pvc="${ns_pvc##*/}"
  echo -n "  Esperando PVC $pvc en $ns..."
  for i in $(seq 1 30); do
    PHASE=$(kubectl get pvc "$pvc" -n "$ns" -o jsonpath='{.status.phase}' 2>/dev/null || echo "")
    if [[ "$PHASE" == "Bound" ]]; then
      echo -e " ${GREEN}Bound${RESET}"
      break
    fi
    echo -n "."
    sleep 2
    if [[ $i -eq 30 ]]; then
      echo ""
      warn "PVC $pvc en $ns no se vinculó en 60s — verifica con: kubectl describe pvc $pvc -n $ns"
    fi
  done
done

# ── Resumen ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo -e "${GREEN}${BOLD}  Setup completado${RESET}"
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo ""
echo "  Próximos pasos:"
echo "    1. bash scripts/build-images.sh   # compilar imágenes Docker"
echo "    2. bash scripts/deploy-all.sh     # desplegar todos los servicios"
echo ""
