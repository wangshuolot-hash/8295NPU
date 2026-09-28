# PC 侧一键拉起车机 NPU server + adb forward（Unity Editor 直连 127.0.0.1:8080 用）
# 前置: 车机 adb 已连接（adb root）
adb push C:\Projects\llm-npu\deploy-hex\start_npu_server.sh /data/local/tmp/start_npu_server.sh
adb shell "sh /data/local/tmp/start_npu_server.sh"
if ($LASTEXITCODE -ne 0) { throw "car-side server start failed" }
adb forward tcp:8080 tcp:8080
Write-Host ""
Write-Host "OK: NPU server on car, adb forward tcp:8080 -> car 8080"
Write-Host "Unity Editor / PC 浏览器可用 http://127.0.0.1:8080/v1/chat/completions"
Write-Host "车机日志: adb shell tail -f /data/local/tmp/llm-server-npu.log"
