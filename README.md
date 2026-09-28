# NPU LLM Deployment on SA8295P (llm-npu)

Run a local LLM (Qwen2.5-0.5B-Instruct Q4_K_M) **entirely on the SA8295P in-vehicle NPU (Hexagon CDSP v68)**, serving OpenAI-compatible HTTP requests that drive Unity 3D car-door animations.

在 **SA8295P 车机 NPU（Hexagon CDSP v68）** 上全量运行本地大模型（Qwen2.5-0.5B-Instruct Q4_K_M），对外提供 OpenAI 兼容 HTTP 接口，驱动 Unity 3D 车门动画。

- Final deliverable: intent recognition + free-form Q&A, no cloud dependency
- 最终形态：本地意图解析 + 自由问答，全程无云端依赖

| Verified latency (on-vehicle) | Value |
|---|---|
| Intent recognition (warm, KV-cached) | ~5 s |
| Intent recognition (cold, full prefill) | ~41 s |
| Test parity (MUL_MAT per-op vs CPU) | 404/404 pass |

> A detailed internal handbook (architecture, all source patches, build/deploy/test guides, full pitfall log) is maintained separately and intentionally not published with this repo. This README covers the essentials.
> 详细的项目手册（架构、补丁清单、构建/部署/测试、踩坑全集）作为内部文档另行维护，有意不随公开仓库分发。本 README 覆盖要点。

---

## 1. What is in this repo / 仓库内容

```
llm-npu/
├── llama-qnn/llama.cpp-dev-refactoring/   # Main source: chraac/llama.cpp fork + our 16 patches
│                                          # 主源码树：fork + 全部 16 处补丁
│   └── ggml/src/ggml-qnn/npu/             #   hexagon-npu FastRPC backend (host + DSP device + IDL)
├── deploy-hex/                           # Deploy scripts + RUNBOOK (binaries are built, not committed)
│   ├── start_npu_server.sh                #   on-vehicle one-click server start
│   └── Start-NpuServer.ps1                #   PC-side: push + start + adb forward
├── unity-demo/                           # Unity 6 voice-door demo project (client side)
│                                          # Unity 6 语音车门示例工程（客户端）
├── docs/                                 # Internal handbook (not published) / 内部手册（不随仓库分发）
├── shim/                                 # libcdsprpc binary patch (QNN-era legacy, kept for reference)
├── ion_probe/                            # GVM ION heap probe (bring-up tool)
├── gguf_scan.py                          # GGUF tensor-type scanner
├── llama-qnn-fork.zip                    # Untouched fork snapshot (28 MB, in case upstream disappears)
└── .gitignore                            # Excludes SDKs / NDK / models / build outputs (see §2)
```

Build outputs and third-party SDKs are **not** committed — see the dependency table below.
构建产物与第三方 SDK **不入库**，见下方依赖清单。

## 2. Dependencies to obtain separately / 需另行获取的依赖

All are excluded by `.gitignore`; download them into the repo root with the exact folder names below, then builds work out of the box.
以下均被 `.gitignore` 排除；按下表下载到仓库根目录并保持文件夹名一致，即可直接构建。

| Component / 组件 | Version | Size | Download / 下载 | Target path / 目标路径 |
|---|---|---|---|---|
| Hexagon SDK (Windows) | 6.6.0.0 | 5.4 GB | [Qualcomm Developer Network](https://developer.qualcomm.com/software/hexagon-dsp-sdk) (account required / 需高通账号) | `hexagon-sdk-6.6.0.0/` |
| Qualcomm AI Runtime (QAIRT) SDK | 2.39.0.250926 | 3 GB | [Qualcomm AI Stack](https://developer.qualcomm.com/software/qualcomm-ai-stack/qairt-sdk) (account required / 需高通账号) | `qairt-2.39.0.250926/` |
| Android NDK | r23b | 2.4 GB | [NDK older releases](https://developer.android.com/ndk/downloads/older_releases) | `ndk-r23b/` |
| Model GGUF | Qwen2.5-0.5B-Instruct Q4_K_M | 491 MB | [HuggingFace](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF) (download all shards and merge / 分片下载后合并) | `models/qwen2.5-0.5b-instruct-q4_k_m.gguf` |

> QAIRT is only used for its headers/runtime libs at link time — the DSP kernel is ours, not QNN's. This is the key difference from the dead-end QNN route.
> QAIRT 仅在链接期借用其头文件/运行时库——DSP 内核完全自研，不依赖 QNN 算子。这是与已被放弃的 QNN 路线的本质区别。

## 3. Build / 构建

Windows host, Android aarch64 target.

```powershell
# 1) Environment — adjust <REPO> to wherever you cloned this repo
#    环境变量（把 <REPO> 替换成你的仓库克隆路径）
$repo = "C:\Projects\llm-npu"          # e.g. / 说明
$env:HEXAGON_SDK_ROOT   = "$repo\hexagon-sdk-6.6.0.0"
$env:HEXAGON_TOOLS_ROOT = "$env:HEXAGON_SDK_ROOT\tools\HEXAGON_Tools\19.0.07"
$env:QNN_SDK_ROOT       = "$repo\qairt-2.39.0.250926"
$src = "$repo\llama-qnn\llama.cpp-dev-refactoring"

# 2) One-time CMake configure (host side, NPU-only build; uses the NDK toolchain)
#    首次配置（host 侧，NPU-Only 构建；使用 NDK 自带工具链）
$ndk = "$repo\ndk-r23b"
cmake -S $src -B "$src\build-hex" `
  -DCMAKE_TOOLCHAIN_FILE="$ndk\build\cmake\android.toolchain.cmake" `
  -DANDROID_ABI=arm64-v8a `
  -DGGML_QNN_ENABLE_HEXAGON_BACKEND=ON `
  -DGGML_HEXAGON_ENABLE_QUANTIZED_TENSORS=ON `
  -DGGML_HEXAGON_NPU_ONLY=ON `
  -DLLAMA_BUILD_TESTS=ON `
  -DCMAKE_BUILD_TYPE=Debug

# 3) Build the v68 DSP skel (ExternalProject) + host binaries
#    构建 v68 DSP skel 与 host 三件套
cmake --build "$src\build-hex" --target hexagon_npu_skel_v68 llama-cli llama-server llama-bench test-backend-ops -j 20
```

Artifacts / 产物:

| Artifact | Path |
|---|---|
| DSP skel | `build-hex\ggml\src\ggml-qnn\npu\hexagon_npu_skel_v68-prefix\src\hexagon_npu_skel_v68-build\libhexagon_npu_skel_v68.so` |
| Host binaries | `build-hex\bin\{llama-server, llama-cli, llama-bench, test-backend-ops}` |
| OpenMP runtime | copy from `ndk-r23b\toolchains\llvm\prebuilt\windows-x86_64\lib\clang\12.0.8\lib\linux\aarch64\libomp.so` |

> SDK path changed after configuring? Delete `hexagon_npu_skel_v68-prefix` and rebuild the ExternalProject — its CMake cache stores absolute paths.
> SDK 移动路径后需删除 `hexagon_npu_skel_v68-prefix` 重建（子构建缓存存有绝对路径）。

## 4. Deploy to the vehicle / 部署到车机

Target: SA8295P (Android 12 GVM, root adb). One command from PC:
目标：SA8295P（Android 12 GVM，root adb），PC 侧一条命令：

```powershell
# Push skel + binaries + model once (first time) / 首次推送产物与模型
adb push <skel-path> /data/local/tmp/libhexagon_npu_skel_v68.so
adb push build-hex\bin\llama-server /data/local/tmp/llm-hex/llama-server
adb push build-hex\bin\llama-cli  /data/local/tmp/llm-hex/llama-cli
adb push build-hex\bin\llama-bench /data/local/tmp/llm-hex/llama-bench
adb push build-hex\bin\test-backend-ops /data/local/tmp/llm-hex/test-backend-ops
adb push ndk-r23b\...\libomp.so /data/local/tmp/llm-hex/libomp.so
adb push models\qwen2.5-0.5b-instruct-q4_k_m.gguf /data/local/tmp/llm/gguf/
adb shell "chmod 755 /data/local/tmp/llm-hex/*"

# Start the server (waits for /health, ~40-60 s cold start)
# 启动 server（内置健康检查等待，冷启动约 40-60 秒）
powershell -File deploy-hex\Start-NpuServer.ps1
```

Runtime environment (set by the start script / 由启动脚本设置):
运行环境变量：

```sh
HEXAGON_NPU_FORCE_ARCH=v68        # vendor arch probe returns 0xffffffff on this GVM
ADSP_LIBRARY_PATH=/data/local/tmp # DSP skel search path (must be absolute)
LD_LIBRARY_PATH=/data/local/tmp/llm-hex
```

## 5. Verify / 验证

```sh
# Per-op numeric parity vs CPU (gold standard) / 逐算子对拍（数值金标准）
adb shell 'HEXAGON_NPU_FORCE_ARCH=v68 ADSP_LIBRARY_PATH=/data/local/tmp \
  LD_LIBRARY_PATH=/data/local/tmp/llm-hex timeout 180 \
  /data/local/tmp/llm-hex/test-backend-ops test -o MUL_MAT -b hexagon-npu 2>&1 | tail -8'
# expect / 期望: 404/404 tests passed

# End-to-end intent via HTTP / 意图端到端
curl -s http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d @deploy-hex/req_fewshot.json
```

## 6. Client integration (Unity or any HTTP client) / 客户端集成（Unity 或任意 HTTP 客户端）

This repo ships the model-serving side **plus a ready-made Unity 6 client**:
[`unity-demo/`](unity-demo/) — a model-agnostic voice-door project (import any car model,
assign its door Transform, press Play; see its own README for details).
本仓库包含模型服务侧，并附带一个开箱即用的 Unity 6 客户端：
[`unity-demo/`](unity-demo/) —— 不绑定具体车模的语音车门工程（导入任意车模、拖入门 Transform、按 Play 即可；详见其内 README）。

For any other language/framework / 其他语言或框架的对接要点:

Recommended client pattern (language-agnostic) / 推荐客户端模式（语言无关）：

- Send `POST /v1/chat/completions` with a few-shot intent prompt: a system message defining the
  labels (`open_door` / `close_door` / `unknown`) followed by 3-4 user/assistant example pairs.
  Small (≤1B) models sit on the intent boundary without examples — few-shot is required, not optional.
  发送 `POST /v1/chat/completions`，system 定义意图标签（open_door / close_door / unknown）+ 3~4 轮 user/assistant 示例对。
  小模型（≤1B）无示例时意图判定落在 logits 边界——few-shot 是必选项。
- On-vehicle app: talk to `http://127.0.0.1:8080/v1/chat/completions` directly (loopback, no network).
  车机端应用：直连 `http://127.0.0.1:8080/v1/chat/completions`（本机回环，不依赖网络）。
- Desktop/editor client (e.g. a Unity Editor on the dev PC): same URL after `adb forward tcp:8080 tcp:8080`.
  PC 端客户端（如 Unity Editor）：执行 `adb forward tcp:8080 tcp:8080` 后使用同一地址。
- Always implement a local fallback (drive the action directly) when the server is unreachable —
  the demo must never block on AI availability.
  服务不可达时必须有本地降级直控——演示链路永不因 AI 缺席而中断。

## 7. Troubleshooting / 常见问题

| Symptom / 现象 | Fix / 解决 |
|---|---|
| `INSTALL_FAILED_INSUFFICIENT_STORAGE` when installing the APK | Clean `/data/local/tmp` legacy artifacts (never touch `titanqnxx` — not ours) 清理历史产物；titanqnxx 非本项目文件勿动 |
| Server not ready in 180 s | `adb shell tail -100 /data/local/tmp/llm-server-npu.log` |
| MUL_MAT rejected by NPU | Rebuild the skel — `HTP_CMAKE_ARGS` must pass `GGML_HEXAGON_ENABLE_QUANTIZED_TENSORS` |
| Intent returns "unknown" | Use the few-shot prompt (0.5B models sit on the intent boundary; see §6) 0.5B 模型在意图边界，必须用 few-shot |
| Skel build fails after SDK relocation | Delete `hexagon_npu_skel_v68-prefix`, rebuild the ExternalProject 清缓存重建 |

For the full pitfall log (vendor libs, v68 ISA limits, toolchain quirks, shell traps) see the internal handbook.
完整踩坑记录（vendor 库、v68 指令限制、工具链、shell 陷阱）见内部手册。

## 8. Key facts for maintainers / 维护者须知

- The fork's hexagon-npu backend is a **from-scratch FastRPC path** (own IDL + own DSP kernels), entirely different from the abandoned QNN route — the vendor v68 skel is missing the ops the QNN serializer emits (Cast / pack_fp16 / ConvLayer), so that route is a dead end on this chip.
  hexagon-npu 后端是**从零自建的 FastRPC 通路**（自有 IDL + 自有 DSP 内核），与已放弃的 QNN 路线完全不同；vendor v68 skel 缺少 QNN 序列化器所需的算子（Cast / pack_fp16 / ConvLayer），该路线在此芯片上是死路。
- The lm_head Q8_0 GEMV uses our **fused v68 kernel** (integer LUT + fp32 HVX synthesis) — the trick that lifted generation speed 4.1×. If you touch it, re-run the per-op parity test first.
  lm_head 的 Q8_0 GEMV 采用**自研 fused 内核**（整型查表 + f32 HVX 合成），生成速度提升 4.1×；改动前务必先跑逐算子对拍。
- Performance roadmap (pending decision): Release build, q4_K fused GEMM, VTCM slice tuning, KV warm-up. Current: pp64 = 2.79 t/s, tg16 = 1.26 t/s.
  性能路线图（待决策）：Release 构建、q4_K fused GEMM、VTCM 调优、KV 预热。当前：pp64 = 2.79 t/s，tg16 = 1.26 t/s。

---

*Generated 2026-09-28. All numbers measured on the actual SA8295P bench.*
*生成于 2026-09-28。全部数字为 SA8295P 实车台架实测值。*
