#!/usr/bin/env bash
# Delete the Cloud Run services this project deployed (images stay in Artifact Registry).
set -euo pipefail
for region in ${REGIONS:-us-central1 us-east4}; do
  for v in cpu-repack cpu-original l4-original; do
    gcloud run services describe emb-$v --region "$region" >/dev/null 2>&1 \
      && gcloud run services delete emb-$v --region "$region" --quiet
  done
done
gcloud run services list --filter 'metadata.name~^emb-'
