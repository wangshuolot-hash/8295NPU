#!/system/bin/sh
# SA8295P 车机 NPU llama-server 一键启动（需 root adb）
# 用法: sh /data/local/tmp/start_npu_server.sh
# 日志: /data/local/tmp/llm-server-npu.log
pkill -9 llama-server 2>/dev/null
sleep 1

export HEXAGON_NPU_FORCE_ARCH=v68
export ADSP_LIBRARY_PATH=/data/local/tmp
export LD_LIBRARY_PATH=/data/local/tmp/llm-hex

nohup /data/local/tmp/llm-hex/llama-server \
  -m /data/local/tmp/llm/gguf/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  -dev hexagon-npu -ngl 99 -c 2048 \
  --host 127.0.0.1 --port 8080 \
  > /data/local/tmp/llm-server-npu.log 2>&1 &

echo "llama-server starting, pid=$!"
echo "waiting for /health (cold start ~40-60s) ..."
i=0
while [ $i -lt 90 ]; do
  sleep 2
  if curl -s http://127.0.0.1:8080/health | grep -q ok; then
    echo "NPU server READY at 127.0.0.1:8080"
    exit 0
  fi
  i=$((i+1))
done
echo "ERROR: server not ready in 180s, check /data/local/tmp/llm-server-npu.log"
exit 1
