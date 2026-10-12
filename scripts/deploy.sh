#!/usr/bin/env bash
# Build, push and deploy one variant. Deploying creates a billable Cloud Run service.
#   scripts/deploy.sh cpu-repack | cpu-original | l4-original
#
#   cpu-repack    1 vCPU, 2 GiB, request-based billing   int8/int4 repack via fbgemm
#   cpu-original  1 vCPU, 4 GiB, request-based billing   float32 original (2.6 GB peak RSS locally)
#   l4-original   4 vCPU, 16 GiB, 1 L4, instance-based   bf16 original; 4 vCPU / 16 GiB is the L4 minimum
#
# Every service: public (--allow-unauthenticated), --min-instances 0, --max-instances 1, so an
# idle service costs nothing and a stuck benchmark cannot fan out. Take it down with scripts/down.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
VARIANT=${1:?cpu-repack | cpu-original | l4-original}
PROJECT=${PROJECT:-$(gcloud config get-value project 2>/dev/null)}
REGION=${REGION:-us-central1}                 # Artifact Registry and Cloud Build
SERVICE_REGION=${SERVICE_REGION:-$REGION}      # where the service runs; GPU quota is per region
REPO=${REPO:-embedding-cloudrun}
SERVICE=emb-${VARIANT}
IMAGE=$REGION-docker.pkg.dev/$PROJECT/$REPO/embeddinggemma2-$VARIANT:latest

case $VARIANT in
  cpu-repack)   MODEL=repack;   TORCH_INDEX=https://download.pytorch.org/whl/cpu
                FLAGS=(--cpu 1 --memory 2Gi --concurrency 1 --cpu-throttling --set-env-vars TORCH_THREADS=1) ;;
  cpu-original) MODEL=original; TORCH_INDEX=https://download.pytorch.org/whl/cpu
                FLAGS=(--cpu 1 --memory 4Gi --concurrency 1 --cpu-throttling --set-env-vars TORCH_THREADS=1) ;;
  l4-original)  MODEL=original; TORCH_INDEX=https://download.pytorch.org/whl/cu130
                FLAGS=(--cpu 4 --memory 16Gi --gpu 1 --gpu-type nvidia-l4 --no-gpu-zonal-redundancy
                       --no-cpu-throttling --concurrency 4 --set-env-vars TORCH_THREADS=4) ;;
  *) echo "unknown variant $VARIANT" >&2; exit 2 ;;
esac

[ -d models/$MODEL ] || { echo "models/$MODEL missing; run scripts/prepare_models.sh" >&2; exit 1; }
gcloud artifacts repositories describe "$REPO" --location "$REGION" >/dev/null 2>&1 \
  || gcloud artifacts repositories create "$REPO" --repository-format docker --location "$REGION"
if [ "${BUILD:-$([ $VARIANT = l4-original ] && echo cloud || echo local)}" = cloud ]; then
  # The CUDA image is ~12 GB; build it on Cloud Build instead of local disk.
  gcloud builds submit --project "$PROJECT" --region "$REGION" --config cloudbuild.yaml \
    --substitutions "_IMAGE=$IMAGE,_MODEL=$MODEL,_TORCH_INDEX=$TORCH_INDEX" .
else
  gcloud auth configure-docker "$REGION-docker.pkg.dev" --quiet
  docker build --build-arg MODEL=$MODEL --build-arg TORCH_INDEX=$TORCH_INDEX -t "$IMAGE" .
  docker push "$IMAGE"
fi
gcloud run deploy "$SERVICE" --image "$IMAGE" --region "$SERVICE_REGION" --project "$PROJECT" \
  --allow-unauthenticated --min-instances 0 --max-instances 1 --cpu-boost "${FLAGS[@]}"
gcloud run services describe "$SERVICE" --region "$SERVICE_REGION" --format 'value(status.url)'
