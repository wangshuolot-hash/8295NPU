using System;
using System.Collections;
using System.Text;
using UnityEngine;
using UnityEngine.Networking;
using UnityEngine.UI;

/// <summary>
/// Voice-command agent / 语音指令 Agent（OpenAI 兼容 HTTP 客户端）。
///
/// Flow / 流程: button (simulated voice) → HTTP POST /v1/chat/completions
///   → LLM intent (few-shot: open_door / close_door / unknown) → DoorController.
///   按钮(模拟语音) → HTTP 请求本地大模型 → few-shot 意图判定 → 驱动车门。
///
/// Works with ANY OpenAI-compatible server / 兼容任意 OpenAI 接口服务:
///   - the NPU llama-server from this repo (car HMI, 127.0.0.1:8080)
///     本仓库的车机 NPU llama-server
///   - or Ollama/vLLM/any cloud endpoint — just change <see cref="endpoint"/>
///     或任意云端/本地 OpenAI 兼容服务，改 endpoint 即可
///
/// Editor note / Editor 使用: run `adb forward tcp:8080 tcp:8080` first
/// if the server runs on a USB-connected device / 服务在车机等设备上时先做端口转发。
/// </summary>
public class AIDoorAgent : MonoBehaviour
{
    [SerializeField] private DoorController door;

    [Header("UI (auto-wired by Tools > Build Voice Door Demo Scene)")]
    [SerializeField] private Text statusText;
    [SerializeField] private InputField questionInput;
    [SerializeField] private Text replyText;

    [Header("Service / 服务")]
    [SerializeField] private string endpoint = "http://127.0.0.1:8080/v1/chat/completions";
    [SerializeField] private float timeoutSeconds = 120f;

    private const string DoorSystem =
        "You are a car intent classifier. Given a user command, reply with exactly one intent label: " +
        "open_door, close_door, or unknown.";

    /// <summary>Few-shot examples: required for small (≤1B) models — they sit on the
    /// intent boundary without examples. 小模型必须 few-shot，否则意图判定不稳定。</summary>
    private static readonly (string user, string assistant)[] DoorExamples =
    {
        ("打开车门", "open_door"),
        ("请帮我把车门打开", "open_door"),
        ("请把门关上", "close_door"),
        ("今天天气怎么样", "unknown"),
    };

    private const string ChatSystem = "You are a helpful local assistant in a car. Reply in the user's language, briefly.";

    private bool _busy;

    public void OnOpenDoorButtonClicked() => SendVoiceCommand("请帮我把车门打开");
    public void OnCloseDoorButtonClicked() => SendVoiceCommand("请帮我把车门关上");

    public void OnAskButtonClicked()
    {
        if (_busy) return;
        string question = questionInput != null ? (questionInput.text ?? "").Trim() : "";
        if (question.Length == 0) { SetReply("请先在输入框输入问题 / type a question first"); return; }
        StartCoroutine(RunChat(question));
    }

    private void SendVoiceCommand(string voiceText)
    {
        if (_busy) return;
        StartCoroutine(RunVoice(voiceText));
    }

    private IEnumerator RunVoice(string voiceText)
    {
        _busy = true;
        SetStatus($"语音: \"{voiceText}\"  →  AI thinking… / 思考中…");
        float started = Time.realtimeSinceStartup;
        string content = null;
        yield return PostChat(DoorSystem, DoorExamples, voiceText, 32, c => content = c);
        float ms = (Time.realtimeSinceStartup - started) * 1000f;

        if (content == null)
        {
            // Server unreachable → degrade to direct control so the demo never blocks.
            // 服务不可达 → 降级直控，演示永不中断。
            SetStatus($"AI unavailable ({ms:0}ms) → direct control / 降级直控");
            if (door != null) door.SetDoorOpen(voiceText.Contains("开"));
        }
        else if (content.Contains("close_door"))
        {
            SetStatus($"AI intent: close_door ({ms:0}ms)");
            if (door != null) door.SetDoorOpen(false);
        }
        else if (content.Contains("open_door"))
        {
            SetStatus($"AI intent: open_door ({ms:0}ms)");
            if (door != null) door.SetDoorOpen(true);
        }
        else
        {
            SetStatus($"AI intent: {content} ({ms:0}ms) → fallback direct / 降级直控");
            if (door != null) door.SetDoorOpen(voiceText.Contains("开"));
        }
        _busy = false;
    }

    private IEnumerator RunChat(string question)
    {
        _busy = true;
        SetStatus($"Q: \"{question}\"  →  AI thinking… / 思考中…");
        SetReply("…");
        float started = Time.realtimeSinceStartup;
        string content = null;
        yield return PostChat(ChatSystem, null, question, 48, c => content = c);
        float ms = (Time.realtimeSinceStartup - started) * 1000f;

        if (content == null)
        {
            SetStatus($"Failed ({ms:0}ms)");
            SetReply($"AI unavailable ({ms:0}ms). Check the server / 请检查模型服务是否可达，Editor 还需 adb forward tcp:8080 tcp:8080。");
        }
        else
        {
            SetStatus($"Replied ({ms:0}ms)");
            string reply = Unescape(content);
            if (reply.Length > 400) reply = reply.Substring(0, 400) + "…";
            SetReply(reply + $"\n[{ms:0}ms]");
            if (door != null)
            {
                if (content.Contains("close_door")) door.SetDoorOpen(false);
                else if (content.Contains("open_door")) door.SetDoorOpen(true);
            }
        }
        _busy = false;
    }

    /// <summary>POST OpenAI-compatible chat completions (optional few-shot rounds).</summary>
    private IEnumerator PostChat(string systemPrompt,
                                 (string user, string assistant)[] examples,
                                 string userText, int maxTokens, Action<string> onDone)
    {
        var body = new StringBuilder("{\"messages\":[");
        body.Append("{\"role\":\"system\",\"content\":\"").Append(Escape(systemPrompt)).Append("\"}");
        if (examples != null)
        {
            foreach (var ex in examples)
            {
                body.Append(",{\"role\":\"user\",\"content\":\"").Append(Escape(ex.user)).Append("\"}");
                body.Append(",{\"role\":\"assistant\",\"content\":\"").Append(Escape(ex.assistant)).Append("\"}");
            }
        }
        body.Append(",{\"role\":\"user\",\"content\":\"").Append(Escape(userText)).Append("\"}");
        body.Append("],\"max_tokens\":").Append(maxTokens).Append(",\"temperature\":0}");
        byte[] bodyBytes = Encoding.UTF8.GetBytes(body.ToString());

        using (var req = new UnityWebRequest(endpoint, UnityWebRequest.kHttpVerbPOST))
        {
            req.uploadHandler = new UploadHandlerRaw(bodyBytes);
            req.downloadHandler = new DownloadHandlerBuffer();
            req.SetRequestHeader("Content-Type", "application/json");
            req.timeout = Mathf.CeilToInt(timeoutSeconds);
            yield return req.SendWebRequest();

            if (req.result != UnityWebRequest.Result.Success) onDone(null);
            else onDone(ExtractContent(req.downloadHandler.text));
        }
    }

    private static string Escape(string s) => s.Replace("\\", "\\\\").Replace("\"", "\\\"").Replace("\n", "\\n");

    private static string Unescape(string s) =>
        s.Replace("\\n", "\n").Replace("\\t", "\t").Replace("\\\"", "\"").Replace("\\\\", "\\");

    /// <summary>Tolerant extraction of choices[].message.content / 容错提取回复字段。</summary>
    private static string ExtractContent(string json)
    {
        const string key = "\"content\":\"";
        int i = json.LastIndexOf(key, StringComparison.Ordinal);
        if (i < 0) return json;
        int start = i + key.Length;
        int end = start;
        while (end < json.Length)
        {
            if (json[end] == '"')
            {
                int backslashes = 0;
                for (int k = end - 1; k >= start && json[k] == '\\'; k--) backslashes++;
                if (backslashes % 2 == 0) break;
            }
            end++;
        }
        return end > start ? json.Substring(start, end - start) : json;
    }

    private void SetStatus(string text)
    {
        if (statusText != null) statusText.text = text;
        Debug.Log($"[AIDoorAgent] {text}");
    }

    private void SetReply(string text)
    {
        if (replyText != null) replyText.text = text;
    }
}
