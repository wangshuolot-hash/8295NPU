# unity-demo — Voice Door Agent (Unity 6) / 语音车门示例工程（Unity 6）

A minimal, model-agnostic Unity 6 project that reproduces the voice-driven car-door
demo: simulated voice buttons → LLM intent recognition (OpenAI-compatible HTTP)
→ any car model's door animates.

一个最小化、不绑定具体车模的 Unity 6 工程，复现"语音指令 → 大模型意图解析 → 车门动画"演示：
把任意车模导入后拖一下门，即可接入完整链路。

> This branch of the `llm-npu` repo pairs with the NPU llama-server on the `main`
> branch — but works with **any** OpenAI-compatible endpoint (Ollama, vLLM, cloud…).
> 本分支与 main 分支的车机 NPU llama-server 配套使用，也兼容任意 OpenAI 接口服务（Ollama、vLLM、云端等）。

## Quick start / 快速开始

```text
1) Open the unity-demo/ folder with Unity 6 (6000.0.x or newer).
   用 Unity 6（6000.0.x 及以上）打开 unity-demo/ 文件夹。

2) Menu: Tools → Build Voice Door Demo Scene
   (generates Assets/Scenes/VoiceDoorDemo.unity with all UI pre-wired)
   菜单：Tools → Build Voice Door Demo Scene
   （一键生成带全部 UI 与事件绑定的示例场景）

3) Point AIDoorAgent.endpoint at your server (default 127.0.0.1:8080), press Play.
   将 AIDoorAgent.endpoint 指向你的模型服务（默认 127.0.0.1:8080），按 Play。
```

## Bring your own car / 导入你自己的车模

1. Import any FBX/GLTF/prefab car into the scene — or use a door-less placeholder
   and just watch the intent text.
   导入任意格式车模到场景（或先不放车模，只看意图判定文本）。
2. Select the **Door** GameObject (created by the scene builder), drag your car's
   door Transform onto the **Door Pivot** slot.
   选中场景里的 Door 对象，把车模的"门"Transform 拖到 Door Pivot 引用上。
3. Pick the animation mode / 选择动画模式:
   - **Hinge** (旋转门): set axis + open angle; put the pivot at the hinge
     (an empty parent node at the hinge works best).
     旋转门：设轴向与开角；pivot 最好指向门轴（可在门轴处建空父节点）。
   - **Slide** (滑动门): set the open offset. 滑动门：设开门位移。
4. Press Play → click "模拟语音 · 开门" → the LLM classifies `open_door` → your
   door animates.
   Play → 点"模拟语音 · 开门" → 大模型判定 open_door → 车门动画。

## Connecting to a server / 连接模型服务

| Scenario / 场景 | Steps / 步骤 |
|---|---|
| NPU llama-server on the car (main branch) / 车机 NPU server（本仓库 main 分支） | start it via `deploy-hex/Start-NpuServer.ps1`; if you run this demo in the Editor over USB, run `adb forward tcp:8080 tcp:8080` first. 车机启动服务后，Editor 经 USB 调试需先做端口转发 |
| On-device app / 打包到车机运行 | keep `127.0.0.1:8080` (loopback, no network needed). 保持 127.0.0.1:8080（本机回环，无需网络） |
| Any other OpenAI-compatible server / 其他 OpenAI 兼容服务 | change `AIDoorAgent.endpoint` in the Inspector, e.g. Ollama `http://127.0.0.1:11434/v1/chat/completions`. Inspector 里改 endpoint 即可 |

## Why few-shot? / 为什么必须 few-shot？

Small models (≤1B) sit right on the intent boundary without examples — the same
prompt can flip between `open_door` and `unknown` depending on numerics. The
built-in few-shot examples make the classification stable (verified 3/3 on a
quantized 0.5B model running on a vehicle NPU).
≤1B 的小模型在无示例时意图判定落在 logits 边界，数值精度轻微变化就会翻转结论。
内置 few-shot 示例让判定稳定（0.5B 量化模型在车机 NPU 上实测 3/3 命中）。

## Files / 文件清单

```
unity-demo/
├── Assets/Scripts/
│   ├── AIDoorAgent.cs              # HTTP client: few-shot intent + free-form Q&A / 意图+问答客户端
│   ├── DoorController.cs           # generic door animation (hinge/slide) / 通用门动画
│   └── Editor/DemoSceneBuilder.cs  # Tools menu: one-click demo scene / 一键场景生成
├── Packages/manifest.json          # uGUI + core modules only / 仅 uGUI 与核心模块
└── ProjectSettings/ProjectVersion.txt
```

Graceful degradation is built in: if the server is unreachable, buttons fall back
to driving the door directly — the demo never blocks.
内置降级策略：服务不可达时按钮直接驱动车门，演示永不中断。
