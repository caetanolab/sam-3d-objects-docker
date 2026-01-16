HF_TOKEN="$1"
IMAGE="sam-3d-objects:latest"
VOL="$(pwd)/checkpoints"
TAG="hf"

mkdir -p "$VOL"

docker run --rm -it \
  --gpus all \
  -e HF_TOKEN="$HF_TOKEN" \
  -v "$VOL:/workspace/sam-3d-objects/checkpoints" \
  "$IMAGE" \
  bash -lc '
set -e
source /opt/conda/etc/profile.d/conda.sh
conda activate sam3d-objects
cd /workspace/sam-3d-objects

# pick whichever CLI exists
if command -v hf >/dev/null; then
  HFCLI=hf
else
  HFCLI=huggingface-cli
fi

$HFCLI download \
  --repo-type model \
  --token "$HF_TOKEN" \
  --local-dir checkpoints/'"$TAG"'-download \
  --max-workers 1 \
  facebook/sam-3d-objects

mv checkpoints/'"$TAG"'-download/checkpoints checkpoints/'"$TAG"'
rm -rf checkpoints/'"$TAG"'-download
'
