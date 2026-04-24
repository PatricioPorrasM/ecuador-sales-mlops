#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# build-images.sh — Compila las imágenes Docker dentro del daemon de Minikube
#
# Qué hace:
#   1. Apunta el cliente Docker al daemon interno de Minikube
#   2. Hace docker build de cada servicio con tag :latest
#   3. Verifica que todas las imágenes existen al final
#
# Uso:
#   bash scripts/build-images.sh
#   bash scripts/build-images.sh --service web-ui   # compilar uno solo
#
# Las imágenes quedan disponibles directamente para Kubernetes sin push a
# ningún registry (imagePullPolicy: IfNotPresent en los manifiestos).
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

step()   { echo -e "\n${CYAN}${BOLD}▶ $*${RESET}"; }
ok()     { echo -e "  ${GREEN}✓${RESET} $*"; }
warn()   { echo -e "  ${YELLOW}⚠${RESET}  $*"; }
die()    { echo -e "\n${RED}✗ ERROR: $*${RESET}\n" >&2; exit 1; }
header() { echo -e "\n${BOLD}── $* ──${RESET}"; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ── Servicios disponibles ─────────────────────────────────────────────────────
declare -A SERVICES=(
  [model-trainer]="ecuador-sales/model-trainer"
  [ml-inference]="ecuador-sales/ml-inference"
  [ai-agent]="ecuador-sales/ai-agent"
  [web-ui]="ecuador-sales/web-ui"
  [kafka-consumer]="ecuador-sales/kafka-consumer"
)

# Build order: trainer primero para que la imagen esté lista antes de inference
BUILD_ORDER=(model-trainer ml-inference ai-agent web-ui kafka-consumer)

# ── Parsear argumento --service ───────────────────────────────────────────────
TARGET_SERVICE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --service)
      TARGET_SERVICE="$2"
      shift 2
      ;;
    *)
      die "Argumento desconocido: $1. Uso: $0 [--service <nombre>]"
      ;;
  esac
done

if [[ -n "$TARGET_SERVICE" && -z "${SERVICES[$TARGET_SERVICE]:-}" ]]; then
  die "Servicio desconocido: $TARGET_SERVICE. Opciones: ${!SERVICES[*]}"
fi

# ── 1. Verificar Minikube ─────────────────────────────────────────────────────
step "Verificando Minikube"

command -v minikube &>/dev/null || die "minikube no encontrado en el PATH"
command -v docker   &>/dev/null || die "docker no encontrado en el PATH"

STATUS=$(minikube status --format='{{.Host}}' 2>/dev/null || echo "")
[[ "$STATUS" == "Running" ]] || die "Minikube no está corriendo. Ejecuta: minikube start"
ok "Minikube corriendo en $(minikube ip)"

# ── 2. Verificar Docker Desktop ───────────────────────────────────────────────
step "Verificando Docker Desktop"

# En Windows con driver Docker, minikube docker-env tiene incompatibilidades de
# versión de API. Usamos Docker Desktop para el build y minikube image load
# para transferir las imágenes al cluster — método más fiable en Windows.
unset DOCKER_HOST DOCKER_TLS_VERIFY DOCKER_CERT_PATH
docker info &>/dev/null || die "Docker Desktop no está corriendo. Inícialo antes de continuar."
ok "Docker Desktop disponible"

# ── 3. Compilar imágenes ──────────────────────────────────────────────────────
step "Compilando imágenes y cargándolas en Minikube"

FAILED=()
BUILT=()

build_service() {
  local svc="$1"
  local image="${SERVICES[$svc]}"
  local context="$REPO_ROOT/services/$svc"

  header "Building $svc → $image:latest"

  if [[ ! -f "$context/Dockerfile" ]]; then
    warn "Sin Dockerfile en $context — saltando $svc"
    return
  fi

  if docker build \
      --tag "$image:latest" \
      --label "build.commit=$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo 'unknown')" \
      --label "build.service=$svc" \
      --label "build.timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      "$context"; then
    ok "$image:latest compilada en Docker Desktop"
    # Cargar la imagen en el daemon de Minikube
    if minikube image load "$image:latest"; then
      ok "$image:latest cargada en Minikube"
      BUILT+=("$svc")
    else
      warn "Falló la carga de $image:latest en Minikube"
      FAILED+=("$svc")
    fi
  else
    warn "Falló la compilación de $svc"
    FAILED+=("$svc")
  fi
}

if [[ -n "$TARGET_SERVICE" ]]; then
  build_service "$TARGET_SERVICE"
else
  for svc in "${BUILD_ORDER[@]}"; do
    build_service "$svc"
  done
fi

# ── 4. Verificar que las imágenes existen en Minikube ────────────────────────
step "Verificando imágenes en Minikube"

ALL_OK=true
for svc in "${BUILD_ORDER[@]}"; do
  [[ -n "$TARGET_SERVICE" && "$svc" != "$TARGET_SERVICE" ]] && continue

  image="${SERVICES[$svc]}"
  if minikube image ls 2>/dev/null | grep -q "$image"; then
    ok "$image:latest  ✓ en Minikube"
  else
    warn "$image:latest NO encontrada en Minikube"
    ALL_OK=false
  fi
done

# ── Resumen ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"

if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo -e "${RED}${BOLD}  Compilación con errores${RESET}"
  echo -e "  Fallaron: ${FAILED[*]}"
else
  echo -e "${GREEN}${BOLD}  Todas las imágenes compiladas correctamente${RESET}"
fi
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"

echo ""
echo "  Imágenes disponibles en Minikube:"
minikube image ls 2>/dev/null | grep "ecuador-sales" | sort | sed 's/^/    /'

echo ""
if [[ "$ALL_OK" == true && ${#FAILED[@]} -eq 0 ]]; then
  echo "  Próximo paso:"
  echo "    bash scripts/deploy-all.sh"
fi
echo ""

[[ ${#FAILED[@]} -eq 0 ]] || exit 1
