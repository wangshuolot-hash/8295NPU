using System.Collections;
using UnityEngine;

/// <summary>
/// Generic car-door controller / 通用车门控制器。
/// Works with ANY car model: import your FBX/Prefab, then drag the door
/// Transform onto <see cref="doorPivot"/> in the Inspector — done.
/// 适用于任意车模：导入 FBX/Prefab 后，把门 Transform 拖到 doorPivot 引用即可。
///
/// Two animation modes / 两种动画模式:
///   Hinge : rotate around <see cref="rotationAxis"/> by <see cref="openAngle"/>
///   Slide : offset localPosition by <see cref="openOffset"/>
///   Hinge：绕 rotationAxis 转到 openAngle；Slide：localPosition 偏移 openOffset
///
/// Tip: for hinge doors, doorPivot should be the door's *pivot node*
/// (or an empty parent placed at the hinge). For slide doors just use
/// the door mesh transform itself.
/// 提示：旋转门请把 pivot 指到门轴节点（或在门轴处建空父节点）；滑动门直接指门网格即可。
/// </summary>
public class DoorController : MonoBehaviour
{
    public enum DoorMode { Hinge, Slide }

    [Header("Door binding / 门绑定 (drag your car door here / 把车门 Transform 拖到这里)")]
    [SerializeField] private Transform doorPivot;

    [Header("Animation / 动画")]
    [SerializeField] private DoorMode mode = DoorMode.Hinge;
    [SerializeField] private Vector3 rotationAxis = Vector3.up;   // hinge axis / 铰链轴
    [SerializeField] private float openAngle = 70f;                // degrees / 角度
    [SerializeField] private Vector3 openOffset = new Vector3(-0.9f, 0f, 0.1f); // slide / 滑动位移
    [SerializeField] private float duration = 1.0f;

    public bool IsOpen { get; private set; }

    private Quaternion _closedRot;
    private Quaternion _openRot;
    private Vector3 _closedPos;
    private Vector3 _openPos;
    private Coroutine _anim;

    private void Awake()
    {
        CacheEnds();
    }

    private void OnValidate()
    {
        if (doorPivot != null && !Application.isPlaying) CacheEnds();
    }

    private void CacheEnds()
    {
        if (doorPivot == null) return;
        _closedRot = doorPivot.localRotation;
        _closedPos = doorPivot.localPosition;
        _openRot = Quaternion.AngleAxis(openAngle, rotationAxis.normalized) * _closedRot;
        _openPos = _closedPos + doorPivot.TransformVector(openOffset * transform.lossyScale.x);
        // note: TransformVector keeps the offset relative to the car so scaled
        // prefabs animate consistently / 位移随车模缩放一致
        _openPos = _closedPos + openOffset;
    }

    /// <summary>Voice/agent entry point. Idempotent. / 语音链路入口，幂等。</summary>
    public void SetDoorOpen(bool open)
    {
        if (doorPivot == null)
        {
            Debug.LogWarning("[DoorController] doorPivot not assigned — assign your car door Transform in the Inspector. 门未绑定，请在 Inspector 拖入车门的 Transform。");
            return;
        }
        if (IsOpen == open) return;
        IsOpen = open;

        if (_anim != null) StopCoroutine(_anim);
        _anim = StartCoroutine(Animate());
    }

    private IEnumerator Animate()
    {
        Quaternion fromRot = doorPivot.localRotation;
        Vector3 fromPos = doorPivot.localPosition;
        Quaternion toRot = IsOpen ? _openRot : _closedRot;
        Vector3 toPos = IsOpen ? _openPos : _closedPos;

        float t = 0f;
        while (t < duration)
        {
            t += Time.deltaTime;
            float k = Mathf.SmoothStep(0f, 1f, Mathf.Clamp01(t / duration));
            if (mode == DoorMode.Hinge) doorPivot.localRotation = Quaternion.Slerp(fromRot, toRot, k);
            else doorPivot.localPosition = Vector3.Lerp(fromPos, toPos, k);
            yield return null;
        }
        // snap / 收尾对齐
        doorPivot.localRotation = toRot;
        doorPivot.localPosition = toPos;
        _anim = null;
    }
}
