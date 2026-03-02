```markdown
# 🧠 Obsidian 配置指南 for GoodFDE

**2026 年最实用 Obsidian 配置手册**  
专为 AI FDE / 全栈工程师学习者设计  
一键 Clone 即用，3 分钟搭建你的 AI 第二大脑

![Obsidian](https://img.shields.io/badge/Obsidian-7C3AED?style=flat&logo=obsidian&logoColor=white)
![GitHub stars](https://img.shields.io/github/stars/findpsyche/GoodFDE?style=social)
![Version](https://img.shields.io/badge/版本-2026.03-blue)
![License](https://img.shields.io/badge/License-MIT-blue.svg)

---

## 📖 目录

- [🎯 为什么用 Obsidian 管理 GoodFDE？](#-为什么用-obsidian-管理-goodfde)
- [📥 安装 Obsidian](#-安装-obsidian)
- [⚙️ 基础设置](#-基础设置)
- [🔌 插件安装方法（2026 社区最新总结）](#-插件安装方法2026-社区最新总结)
- [🔥 你的核心插件配置](#-你的核心插件配置)
- [🚀 2026 推荐插件清单](#-2026-推荐插件清单)
- [📋 最佳实践（AI 工程师专用）](#-最佳实践ai-工程师专用)
- [🛠 3 分钟上手 GoodFDE](#-3-分钟上手-goodfde)
- [🤝 贡献 & 交流](#-贡献--交流)

---

## 🎯 为什么用 Obsidian 管理 GoodFDE？

- **本地 Markdown + Git**：笔记永远属于你，随便同步到 GitHub
- **知识图谱**：一键看到「vLLM → RAG → Multi-Agent → 生产部署」的完整关联
- **Claudian 插件**：Claude 直接嵌入 Obsidian，读写文件、执行命令、帮你写代码、生成学习计划
- **Dataview + Templater**：自动生成学习仪表盘、每日复盘、进度追踪
- **与 GoodFDE 完美适配**：所有文档打开即用，图谱、双链、Canvas 全部生效

---

## 📥 安装 Obsidian

1. 官网下载最新版：https://obsidian.md/download
2. 安装完成后 → **Open folder as vault** → 选择你克隆的 `GoodFDE` 文件夹

---

## ⚙️ 基础设置（必做）

**Settings → Core plugins** 全部开启：
- Daily notes
- Templates
- Backlinks
- Graph view
- Canvas
- Command palette
- Page preview
- Starred
- Search
- Tags

**外观推荐**：
- 主题：**Minimal** 或 **AnuPpuccin**（Community themes 搜索安装）
- 开启 Readable line length + Show inline title

---

## 🔌 插件安装方法（2026 社区真实总结）

### 方法一：社区浏览器（最推荐，90% 用户使用）

1. Settings → **Community plugins**
2. 点击 **Turn off Restricted Mode**（关闭安全模式）
3. 点击 **Browse**
4. 搜索插件名称 → Install → Enable

### 方法二：BRAT（你已安装，测试版神器）

1. Settings → Community plugins → BRAT
2. **Add a beta plugin for testing**
3. 输入 GitHub 仓库地址（例如 `yishentian/claudian`）
4. Add → 自动下载最新版 → Enable

### 方法三：手动安装（离线 / 私有插件）

1. 下载插件 Releases 的 zip 包
2. 解压到 `.obsidian/plugins/插件ID/` 文件夹
3. 重启 Obsidian → Enable

---

## 🔥 你的核心插件配置（已安装）

| 插件                  | 版本     | 核心用途                          | 推荐指数 |
|-----------------------|----------|-----------------------------------|----------|
| **BRAT**              | 1.4.1    | 安装测试版插件                    | ★★★★★    |
| **Calendar**          | 1.5.10   | 每日笔记日历导航                  | ★★★★★    |
| **Claudian**          | 1.3.67   | Claude 嵌入协作（读写文件、执行命令） | ★★★★★    |
| **Dataview**          | 0.5.68   | SQL 查询 + 自动仪表盘             | ★★★★★    |
| **Templater**         | 2.17.0   | 高级模板，一键生成结构化笔记      | ★★★★★    |

---

## 🚀 2026 推荐插件（继续安装）

**必装（5 分钟）**：
- Tasks（任务管理）
- Periodic Notes（周/月报）
- QuickAdd（快捷创建笔记）
- Style Settings（主题微调）
- Linter（自动格式化）
- Obsidian Git（Obsidian 内 commit）

**AI 学习专属**：
- Excalidraw（画架构图）
- Make.md（文件夹图标美化）
- Smart Connections（AI 智能关联）

---

## 📋 最佳实践（AI 工程师专用）

1. **组织结构**：少用文件夹，多用 MOC + 双链
2. **每日流程**：
   - Calendar 点击今天 → Templater 插入「学习模板」
   - 阅读 GoodFDE 文档 → 用 `[[双链]]` 关联
   - Claudian 命令：「请总结这篇 vLLM 笔记并生成代码示例」
3. **Dataview 仪表盘示例**（新建 `Dashboard.md`）：
   ```dataview
   TABLE file.mtime AS "最后更新", tags
   FROM "learning"
   SORT file.mtime DESC
   LIMIT 15
   ```
4. **版本控制**：安装 Obsidian Git，每天 `Ctrl+Shift+G` 一键提交到 GitHub
5. **性能建议**：插件总数控制在 15 个以内，定期清理

---

## 🛠 3 分钟上手 GoodFDE

```bash
git clone https://github.com/findpsyche/GoodFDE.git
cd GoodFDE
# 用 Obsidian 打开此文件夹
```

1. 打开 `AI_FDE全栈学习路径_2025-2026.md`
2. 右上角开启 **Graph view**
3. 按 `Ctrl/Cmd + P` 输入 “Daily note” 创建今日笔记
4. 开始你的 100 天 AI FDE 学习之旅！

---

## 🤝 贡献 & 交流

- 发现好插件？提交 PR 到本文件
- 配置问题 / 想分享你的 Vault？欢迎 [Issues](https://github.com/findpsyche/GoodFDE/issues)
- Discussions 区一起讨论学习进度

**最后更新**：2026 年 3 月  
**基于你的真实插件截图 + 社区最新经验**

---

**⭐ 如果这个配置对你有帮助，请给主仓库点个 Star！**
```

---
