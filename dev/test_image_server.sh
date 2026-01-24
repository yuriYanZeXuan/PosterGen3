# 写死本地权重路径（按你的机器目录）
export ZIMAGE_MODEL_PATH="/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/Z-Image"
export QWEN_EDIT_MODEL_ID="/mnt/tidalfs-bdsz01/usr/tusen/yanzexuan/weight/qwen_edit_2511"

HOST="127.0.0.1"
PORT="9101"
BASE="http://${HOST}:${PORT}"

SERVER_PID=""
if ! curl -fsS "$BASE/health" >/dev/null 2>&1; then
  # 约定：从 PosterGen3/dev 目录执行（相对路径）
  python3 "./image_server.py" --host "$HOST" --port "$PORT" &
  SERVER_PID="$!"
  trap '[[ -n "${SERVER_PID:-}" ]] && kill "$SERVER_PID" >/dev/null 2>&1 || true' EXIT INT TERM
fi

for _ in {1..60}; do curl -fsS "$BASE/health" >/dev/null && break; sleep 0.5; done

GEN_JSON='{"prompt":"A clean product photo of a red ceramic mug on a white background, studio lighting, high detail","height":512,"width":512,"num_inference_steps":9,"guidance_scale":0.0,"seed":42,"out_dir":null,"engine":"zimage"}'
GEN_PATH="$(curl -sS -X POST "$BASE/v1/generate" -H "Content-Type: application/json" -d "$GEN_JSON" | python3 -c 'import json,sys;print(json.load(sys.stdin)["path"])')"

EDIT_PATH="$(curl -sS -X POST "$BASE/v1/edit" \
  -F engine=qwen_edit \
  -F 'prompt=Make the mug look like it is made of transparent glass, keep the same shape.' \
  -F remove_bg=true \
  -F "image=@${GEN_PATH}" | python3 -c 'import json,sys;print(json.load(sys.stdin)["path"])')"

echo "$GEN_PATH"
echo "$EDIT_PATH"

