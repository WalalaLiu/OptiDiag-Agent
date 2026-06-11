# OpenAPI 接入说明

本文用于把本地 FastAPI 服务接入 `open.zju.edu.cn` 智能体插件。MVP 阶段推荐先在本机或云服务器上启动服务，再把 OpenAPI schema 提供给智能体平台识别工具接口。

## 服务入口

智能体插件主要调用：

```text
POST /analyze
```

健康检查：

```text
GET /health
```

OpenAPI schema：

```text
GET /openapi.json
```

## 输入方式

`/analyze` 使用 `multipart/form-data`。本阶段同时支持上传文件和图片 URL：

- `multipart/form-data`
- `file`：二进制图片上传字段，推荐优先使用
- `image_url`：可选，当平台只能传图片 URL 时使用
- `experiment_type`：可选，默认 `diffraction`

优先级规则：

1. 如果请求里包含 `file`，服务端分析 `file`。
2. 如果没有 `file`，但包含 `image_url`，服务端下载并分析该图片。
3. 如果两者都没有，返回 HTTP 400。

也就是说，智能体平台如果支持文件上传，应优先把用户上传的实验图像作为 `file` 传入；如果平台只能给出附件 URL，则把 URL 作为 `image_url` 传入。

## curl 测试

文件上传：

```bash
curl -X POST "http://127.0.0.1:8000/analyze" \
  -F "file=@data/simulated/smoke/images/00000_single_slit_over_exposure.png" \
  -F "experiment_type=diffraction"
```

图片 URL：

```bash
curl -X POST "http://127.0.0.1:8000/analyze" \
  -F "image_url=https://example.com/diffraction.png" \
  -F "experiment_type=diffraction"
```

## 返回 JSON 字段解释

- `image_type`：图像类型。没有训练模型时通常为 `unknown_or_estimated`；演示脚本可使用仿真真值。
- `confidence`：模型置信度。无模型时为 `0.0`。
- `issues`：检测到的误差/噪声列表，每项包含 `type`、`severity` 和 `score`。
- `metrics`：可解释图像指标，包括饱和像素比例、中心偏移、条纹可见度、拉普拉斯方差、信噪比等。
- `diagnosis`：中文诊断摘要。
- `possible_causes`：可能原因列表。
- `suggestions`：实验操作建议列表。
- `need_reacquire`：是否建议重新采集图像。
- `model_available`：服务端是否加载了神经网络权重。
- `fallback`：当前诊断路径。无模型时为 `rule_based`；加载模型后可为 `model`。

## 智能体提示词建议

建议在智能体系统提示词中加入：

```text
当用户上传光学衍射实验图像时，调用 OptiDiag 插件的 /analyze 接口。
收到 JSON 后，不要只判断合格/不合格。
请先概括图像中最主要的问题，再引用 metrics 中的关键指标解释依据，
最后给出可以直接执行的实验调整建议。
如果 need_reacquire 为 true，请明确建议用户调整参数后重新采集对照图像。
```

回答模板建议：

```text
1. 图像诊断摘要
2. 主要问题及严重程度
3. 关键指标依据
4. 可能原因
5. 下一步实验操作建议
```

## 导出 OpenAPI schema

```bash
python scripts/export_openapi.py --out docs/openapi.json
```

FastAPI 运行时也会直接提供：

```text
http://127.0.0.1:8000/openapi.json
```

把该 schema 提供给智能体平台后，应确认平台识别到 `POST /analyze` 的 `multipart/form-data` 参数。如果平台不支持文件字段，可先让智能体传入 `image_url`。
