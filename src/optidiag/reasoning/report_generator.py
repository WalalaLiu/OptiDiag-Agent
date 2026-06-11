"""Chinese diagnosis report generation for agent-facing responses."""

from __future__ import annotations

from typing import Dict, Iterable, List


ISSUE_CN = {
    "over_exposure": "过曝",
    "under_exposure": "欠曝",
    "gaussian_noise": "高斯噪声",
    "poisson_noise": "泊松噪声",
    "dark_noise": "暗噪声",
    "background_gradient": "背景不均匀",
    "blur_defocus": "离焦或模糊",
    "misalignment": "光轴偏移",
    "rotation_tilt": "图像旋转倾斜",
    "cropping_incomplete": "图样裁切不完整",
    "low_contrast": "条纹对比度偏低",
}

SEVERITY_CN = {
    "low": "轻微",
    "medium": "中等",
    "high": "严重",
}

CAUSES = {
    "over_exposure": ["相机曝光时间或光源功率偏高", "中央主极大强度超过探测器动态范围"],
    "under_exposure": ["曝光时间偏短或光源强度不足", "光路遮挡或入射光能量偏低"],
    "gaussian_noise": ["相机读出噪声较明显", "环境振动或电子噪声导致背景起伏"],
    "poisson_noise": ["光子计数不足", "弱光条件下采集导致散粒噪声明显"],
    "dark_noise": ["暗电流或热噪声偏高", "未进行暗场扣除或相机温度较高"],
    "background_gradient": ["环境杂散光进入光路", "屏幕或相机背景照明不均匀"],
    "blur_defocus": ["观察屏或相机未处在清晰成像位置", "透镜焦距或相机焦点未调准"],
    "misalignment": ["孔径、透镜和相机中心未完全共轴", "实验台或支架位置发生偏移"],
    "rotation_tilt": ["相机或孔径板发生旋转", "条纹方向与探测器坐标轴不平行"],
    "cropping_incomplete": ["相机视场没有覆盖完整衍射图样", "主极大或高级次条纹靠近画面边缘"],
    "low_contrast": ["相干性不足或背景光偏强", "孔径边缘污染、光路未充分准直"],
}

SUGGESTIONS = {
    "over_exposure": ["适当降低曝光时间或光源强度，避免中央主极大饱和", "固定曝光参数后重新采集一张图像用于对比"],
    "under_exposure": ["适当增加曝光时间或光源强度", "检查孔径、透镜和屏幕之间是否存在遮挡"],
    "gaussian_noise": ["提高信号强度后降低相机增益", "多次采集取平均以降低随机读出噪声"],
    "poisson_noise": ["增加光子通量或延长曝光时间", "在不过曝的前提下提高入射光强"],
    "dark_noise": ["采集暗场图像并做背景扣除", "降低相机温度或缩短过长曝光时间"],
    "background_gradient": ["遮挡环境杂散光并保持背景均匀", "重新采集背景图用于平场校正"],
    "blur_defocus": ["重新调节透镜焦距或相机焦点", "观察中心主极大边缘是否变清晰后再采集"],
    "misalignment": ["微调孔径、透镜和相机中心，使主极大回到图像中心", "确认光路元件固定后再采集"],
    "rotation_tilt": ["旋正相机或孔径板，使条纹方向与图像坐标轴一致", "记录调整前后条纹角度变化"],
    "cropping_incomplete": ["扩大相机视场或重新居中衍射图样", "确保中央主极大和主要条纹完整进入画面"],
    "low_contrast": ["减小环境背景光并提高光源相干性", "清洁孔径板并重新准直入射光"],
}


def collect_unique(items: Iterable[str]) -> List[str]:
    """Preserve order while removing duplicates."""
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result


def build_diagnosis_sentence(issues: List[Dict[str, object]]) -> str:
    """Build a compact Chinese diagnosis sentence from detected issues."""
    if not issues:
        return "图像质量总体可用，未发现明显过曝、模糊、偏移或背景异常。"

    parts = []
    for issue in issues[:3]:
        issue_type = str(issue["type"])
        severity = str(issue["severity"])
        parts.append(f"{SEVERITY_CN.get(severity, severity)}的{ISSUE_CN.get(issue_type, issue_type)}")
    return "图像疑似存在" + "、".join(parts) + "。"


def build_causes_and_suggestions(issues: List[Dict[str, object]]) -> tuple[List[str], List[str]]:
    """Return Chinese possible causes and operating suggestions."""
    causes: List[str] = []
    suggestions: List[str] = []
    for issue in issues:
        issue_type = str(issue["type"])
        causes.extend(CAUSES.get(issue_type, []))
        suggestions.extend(SUGGESTIONS.get(issue_type, []))

    if not suggestions:
        suggestions = ["保持当前光路和曝光参数，并采集一张重复图像用于稳定性对比"]
    return collect_unique(causes), collect_unique(suggestions)
