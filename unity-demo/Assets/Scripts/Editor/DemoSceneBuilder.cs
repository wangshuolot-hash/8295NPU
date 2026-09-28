#if UNITY_EDITOR
using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;

/// <summary>
/// One-click demo scene builder / 一键生成示例场景。
/// Menu: Tools → Build Voice Door Demo Scene
/// Generates a ready-to-press-play scene: Canvas with open/close/ask UI wired to
/// <see cref="AIDoorAgent"/>, an EventSystem, camera and light. Saves to
/// Assets/Scenes/VoiceDoorDemo.unity.
/// 生成开箱即用的场景（按钮已绑定 AIDoorAgent、含 EventSystem/相机/灯光），
/// 保存为 Assets/Scenes/VoiceDoorDemo.unity。
/// </summary>
public static class DemoSceneBuilder
{
    private const string ScenePath = "Assets/Scenes/VoiceDoorDemo.unity";

    [MenuItem("Tools/Build Voice Door Demo Scene")]
    public static void Build()
    {
        var scene = EditorSceneManager.NewScene(NewSceneSetup.DefaultGameObjects, NewSceneMode.Single);

        // ---- Agent & Door objects / Agent 与门对象 ----
        var agentGo = new GameObject("AIDoorAgent");
        var doorGo = new GameObject("Door");
        var agent = agentGo.AddComponent<AIDoorAgent>();
        var door = doorGo.AddComponent<DoorController>();
        var soDoor = new SerializedObject(agent);
        soDoor.FindProperty("door").objectReferenceValue = door;
        soDoor.ApplyModifiedPropertiesNoUndo();

        // ---- Canvas ----
        var canvasGo = new GameObject("Canvas", typeof(Canvas), typeof(CanvasScaler), typeof(GraphicRaycaster));
        canvasGo.GetComponent<Canvas>().renderMode = RenderMode.ScreenSpaceOverlay;
        var scaler = canvasGo.GetComponent<CanvasScaler>();
        scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
        scaler.referenceResolution = new Vector2(1920, 720);

        Text status = null, reply = null;
        InputField input = null;
        AIDoorAgent agentRef = agent;

        // Top status bar / 顶部状态栏
        var statusGo = MakeText(canvasGo.transform, "StatusText", "AI voice-command demo — click a button below",
            TextAnchor.MiddleCenter, 30, new Vector2(-760, -60, 1520, 56));
        status = statusGo.GetComponent<Text>();
        status.fontSize = 26;

        // Reply area / 回复区
        var replyGo = MakeText(canvasGo.transform, "ReplyText", "Model reply shows here / 模型回复显示在这里",
            TextAnchor.UpperLeft, 22, new Vector2(-760, -170, 700, 260));
        reply = replyGo.GetComponent<Text>();
        reply.alignment = TextAnchor.UpperLeft;
        reply.horizontalOverflow = HorizontalWrapMode.Wrap;
        reply.verticalOverflow = VerticalWrapMode.Overflow;

        // Input field / 输入框
        var inputGo = new GameObject("QuestionInput", typeof(InputField), typeof(Image));
        inputGo.transform.SetParent(canvasGo.transform, false);
        var inputRect = inputGo.GetComponent<RectTransform>();
        inputRect.anchorMin = inputRect.anchorMax = new Vector2(0.5f, 0f);
        inputRect.pivot = new Vector2(0.5f, 0f);
        inputRect.anchoredPosition = new Vector2(-230, 120);
        inputRect.sizeDelta = new Vector2(560, 56);
        var inputImg = inputGo.GetComponent<Image>();
        inputImg.color = new Color(0.12f, 0.15f, 0.2f, 0.9f);
        var placeholder = MakeText(inputGo.transform, "Placeholder", "Ask anything / 输入问题…",
            TextAnchor.MiddleLeft, 20, new Vector2(0, 0, -20, 0)).GetComponent<Text>();
        placeholder.fontStyle = FontStyle.Italic;
        var placeholderColor = placeholder.color; placeholderColor.a = 0.5f; placeholder.color = placeholderColor;
        var inputText = MakeText(inputGo.transform, "Text", "", TextAnchor.MiddleLeft, 22,
            new Vector2(0, 0, -20, 0)).GetComponent<Text>();
        inputText.supportRichText = false;
        var inputField = inputGo.GetComponent<InputField>();
        inputField.textComponent = inputText;
        inputField.placeholder = placeholder;
        input = inputField;

        // Buttons / 按钮
        var openBtn = MakeButton(canvasGo.transform, "OpenDoorButton", "模拟语音 · 开门", new Vector2(-360, 36), 320, 84);
        var closeBtn = MakeButton(canvasGo.transform, "CloseDoorButton", "模拟语音 · 关门", new Vector2(0, 36), 320, 84);
        var askBtn = MakeButton(canvasGo.transform, "AskButton", "问 AI / Ask", new Vector2(360, 36), 200, 84);
        Bind(openBtn, "OnOpenDoorButtonClicked");
        Bind(closeBtn, "OnCloseDoorButtonClicked");
        Bind(askBtn, "OnAskButtonClicked");

        // EventSystem (uGUI) ---- needed by InputSystem projects too when using uGUI
        if (Object.FindObjectOfType<EventSystem>() == null)
        {
            var es = new GameObject("EventSystem", typeof(EventSystem));
            if (es.GetComponent<StandaloneInputModule>() == null)
                es.AddComponent<StandaloneInputModule>();
        }

        // Wire UI references into the agent / 把 UI 引用注入 Agent
        var so = new SerializedObject(agentRef);
        so.FindProperty("statusText").objectReferenceValue = status;
        so.FindProperty("questionInput").objectReferenceValue = input;
        so.FindProperty("replyText").objectReferenceValue = reply;
        so.ApplyModifiedPropertiesNoUndo();

        Directory.CreateDirectory("Assets/Scenes");
        EditorSceneManager.SaveScene(scene, ScenePath);
        EditorUtility.DisplayDialog("Voice Door Demo",
            "Scene saved to " + ScenePath +
            "\n\nNext / 下一步:\n" +
            "1) Start the LLM server (or any OpenAI-compatible endpoint),\n" +
            "   Editor: adb forward tcp:8080 tcp:8080 if the server is on a device.\n" +
            "   启动模型服务；服务在车机上时 Editor 先做端口转发。\n" +
            "2) Drop your car model into the scene and assign its door Transform\n" +
            "   to the Door component. 导入车模，把门的 Transform 拖到 Door 组件。\n" +
            "3) Press Play. 按 Play 运行。", "OK");
        EditorGUIUtility.PingObject(AssetDatabase.LoadAssetAtPath<SceneAsset>(ScenePath));
    }

    private static GameObject MakeText(Transform parent, string name, string content,
        TextAnchor anchor, int size, Vector4 rect)
    {
        var go = new GameObject(name, typeof(Text));
        go.transform.SetParent(parent, false);
        var t = go.GetComponent<Text>();
        t.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
        t.text = content;
        t.fontSize = size;
        t.alignment = anchor;
        t.color = Color.white;
        var rt = t.rectTransform;
        rt.anchorMin = rt.anchorMax = new Vector2(0.5f, 1f);
        rt.pivot = new Vector2(0.5f, 1f);
        rt.anchoredPosition = new Vector2(rect.x, rect.y);
        rt.sizeDelta = new Vector2(rect.z, rect.w);
        return go;
    }

    private static GameObject MakeButton(Transform parent, string name, string label,
        Vector2 pos, float w, float h)
    {
        var go = new GameObject(name, typeof(Image), typeof(Button));
        go.transform.SetParent(parent, false);
        var img = go.GetComponent<Image>();
        img.color = new Color(0.08f, 0.22f, 0.45f, 0.95f);
        var rt = go.GetComponent<RectTransform>();
        rt.anchorMin = rt.anchorMax = new Vector2(0.5f, 0f);
        rt.pivot = new Vector2(0.5f, 0f);
        rt.anchoredPosition = pos;
        rt.sizeDelta = new Vector2(w, h);
        var txt = MakeText(go.transform, "Text", label, TextAnchor.MiddleCenter, 24, new Vector4(0, 0, 0, 0));
        var trt = txt.GetComponent<RectTransform>();
        trt.anchorMin = Vector2.zero; trt.anchorMax = Vector2.one;
        trt.offsetMin = Vector2.zero; trt.offsetMax = Vector2.zero;
        return go;
    }

    private static void Bind(GameObject button, string method)
    {
        var btn = button.GetComponent<Button>();
        var target = Object.FindObjectOfType<AIDoorAgent>();
        if (target == null) return;
        switch (method)
        {
            case "OnOpenDoorButtonClicked":
                UnityEditor.Events.UnityEventTools.AddPersistentListener(btn.onClick, target.OnOpenDoorButtonClicked);
                break;
            case "OnCloseDoorButtonClicked":
                UnityEditor.Events.UnityEventTools.AddPersistentListener(btn.onClick, target.OnCloseDoorButtonClicked);
                break;
            case "OnAskButtonClicked":
                UnityEditor.Events.UnityEventTools.AddPersistentListener(btn.onClick, target.OnAskButtonClicked);
                break;
        }
    }
}
#endif
