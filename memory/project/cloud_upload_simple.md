---
name: cloud-upload-always-http-put
description: AI Studio 云上传模块统一走 HTTP PUT 到 pre-signed URL
type: project
---

ocean.cloud 上传到 AI Studio 时，所有文件（不论大小）统一走 HTTP PUT 到 LFS pre-signed URL。

- 不依赖 `baidubce` BCE SDK（其 `put_super_obejct_from_file` 有 typo）
- 不区分 STS multipart 或 HTTP 直传路径
- `requests.put(data=generator)` 流式上传，适配任意文件大小
- 进度条每传完一个 8MB 分片刷新一次，刷新频率随网速变化

**Why:** 简化代码、消除未验证的 BCE 签名路径、避免 SDK 拼写错误依赖。

**How to apply:** 任何对上传模块的修改都遵循「统一 HTTP PUT 到 pre-signed URL」原则，不再引入 BOS REST API 路径。
