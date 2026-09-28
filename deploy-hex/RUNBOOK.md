# Hexagon FastRPC 通路 台架恢复运行手册

> 生成时间：2026-09-24。产物：C:\Projects\llm-npu\deploy-hex\
> 背景：QNN 原生算子路线在 v68 已定性死路（Cast/pack/ConvLayer 三类算子 v68 skel 全缺），
> 本路线 = fork 自带 raw fastRPC 后端（hexagon-npu 设备 + fork 自有 HVX 内核 DSP skel）。

## 部署产物清单（deploy-hex\）

| 文件 | 说明 | 目标路径 |
|------|------|----------|
| libhexagon_npu_skel_v68.so | DSP 侧 skel（250KB，Hexagon ELF，含全部 HVX 内核） | /data/local/tmp/ |
| llama-cli (128MB Debug) | 全量日志版 | /data/local/tmp/llm-hex/ |
| llama-server (140MB Debug) | NPU server 候选 | /data/local/tmp/llm-hex/ |
| llama-bench (62MB Debug) | 性能基准 | /data/local/tmp/llm-hex/ |
| libgcc.so | DSP 侧备用库（skel DT_NEEDED；车机 CDSP 运行时大概率自带，缺时才推） | /data/local/tmp/（按需） |

车上已有：fp16 + q4_k_m gguf（/data/local/tmp/llm/gguf/）。

## 一键部署（台架恢复后执行）

```powershell
$d = "C:\Projects\llm-npu\deploy-hex"
adb push "$d\libhexagon_npu_skel_v68.so" /data/local/tmp/libhexagon_npu_skel_v68.so
adb shell "mkdir -p /data/local/tmp/llm-hex"
adb push "$d\llama-cli" /data/local/tmp/llm-hex/llama-cli
adb push "$d\llama-server" /data/local/tmp/llm-hex/llama-server
adb push "$d\llama-bench" /data/local/tmp/llm-hex/llama-bench
adb shell "chmod +x /data/local/tmp/llm-hex/*"
```

注意：adb push 必须带完整目标路径（推目录会静默失败）。

## 第一轮验证（fp16 模型 + hexagon-npu）

```powershell
adb shell "logcat -c; cd /data/local/tmp && ADSP_LIBRARY_PATH=/data/local/tmp timeout 240 ./llm-hex/llama-cli -m /data/local/tmp/llm/gguf/qwen2.5-0.5b-instruct-fp16.gguf -dev hexagon-npu -ngl 99 -p 'The capital of France is' -n 8 --verbose < /dev/null > /data/local/tmp/hex_run1.log 2>&1; echo EXIT_CODE:`$?; logcat -d > /data/local/tmp/hex_logcat1.txt; echo -n paris=; grep -ac 'Paris' /data/local/tmp/hex_run1.log; echo -n device_open=; grep -ac 'NPU device opened successfully' /data/local/tmp/hex_run1.log; echo -n graph_create=; grep -ac 'creating new graph' /data/local/tmp/hex_run1.log; grep -a 'ERROR\|Unable to open\|unsupported' /data/local/tmp/hex_run1.log | head -6; grep -a 'per second\|eval time' /data/local/tmp/hex_run1.log | head -4"
```

### 判读
- 成功标志：`paris≥1` + `device_open=1` + eval time 性能行
- skel 加载失败（0x3 / dlerr）：DSP 缺 libgcc.so → 推备用库 `adb push $d\libgcc.so /data/local/tmp/libgcc.so` 重试
- 首开被拒（AEE_ECONNREFUSED）：代码内置自动调 `DSPRPC_CONTROL_UNSIGNED_MODULE` 重开（enable_unsigned_dsp_module），若仍失败查 ADSP_LIBRARY_PATH 是否绝对路径
- rpcmem 分配失败（vendor QoS bug）：加 `LD_PRELOAD=/data/local/tmp/libcdsprpc_guard2.so` 重试（guard-v2 仍在车上）

## 关键运行时机制（已在代码中确认）

1. `get_dsp_arch`：运行时 DSPRPC_GET_DSP_INFO 探测架构（0x68）→ 自动选 `libhexagon_npu_skel_v68.so` URI
2. skel URI：`file:///libhexagon_npu_skel_v68.so?npu_device_skel_handle_invoke&_modver=1.0` + `_dom=cdsp`（CDSP_DOMAIN_ID）
3. 未签名模块：ECONNREFUSED 时自动 enable unsigned module（车机未签名 skel 可跑，已验证过 calculator）
4. 算子支持：DSP 侧实时查询（npu_device_device_support_op），host 白名单 MUL_MAT/ADD/SUB/MUL/RMS_NORM/FLASH_ATTN/ROPE/GLU/GET_ROWS/SET_ROWS/CPY
5. 权重路径：rpcmem(ION) 共享内存，DSP 直接读 host 侧权重（无逐 token 拷贝）
6. fp16 走 copy_row_f16（v68 原生支持）；q8_0 反量化已改 v68 标量回退（性能降级仅影响 q8_0，q4_0/q4_K 走查表路径未动）

## 通过后下一步

1. llama-bench 对比 CPU 基线（49 tok/s @ q4km）：`./llm-hex/llama-bench -m fp16.gguf -dev hexagon-npu -ngl 99 -p 512 -n 128`
2. llama-server 上 8080 替换 CPU 调试靶机：`./llm-hex/llama-server -m fp16.gguf -dev hexagon-npu -ngl 99 --host 127.0.0.1 --port 8080`（先杀旧 CPU server）
3. Unity 客户端零改动直接连（客户端指向 127.0.0.1:8080）→ 实车全链路验证
4. 若 fp16 内存/带宽不足再试 q4km（DSP 侧原生 Q4_K 反量化路径）

## 本轮构建修复记录（fork 源码，均已在源码中落地）

1. npu/CMakeLists.txt：补 v68 ExternalProject + CMAKE_INSTALL_PREFIX 修 Program Files 越权
2. npu/device/CMakeLists.txt：源码 glob 路径修正（原指向不存在的 common/、device/ 子目录 → 18KB 空壳 skel）
3. npu/device/type_traits.cpp：dequantize_row_q8_0 的 HVX fp16（v73+）改 v68 标量回退
4. shared/CMakeLists.txt：`if(DEFINED ENV{QNN_SDK_PATH})` 笔误改 `ENV{HEXAGON_SDK_ROOT}`
5. qnn/graph.cpp：bake 改 opt-in + weightsPacking=false + opt level=1 + 全 F32→F16 override + 配置开关（QNN 路线遗留，与 hexagon 路线无冲突）
6. qnn-lib.cpp：QNN_SKIP_RPC_POLLING 开关（QNN 路线遗留）
