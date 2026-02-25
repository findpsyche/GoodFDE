规划 虚拟文件系统 任务委派 上下文令牌管理 代码执行 人在循环中

---

后端：深度代理通过类似工具向代理暴露文件系统表面`ls`,`read_file`,`write_file`,`edit_file`,`glob`并且`grep`
`read_file`工具原生支持图像文件(`.png`,`.jpg`,`.jpeg`,`.gif`,`.webp`在所有后端,以多模式内容块的形式返回。

---
子代理subagent
why:**context bloat problem**子代理解决了上下文膨胀问题
子代理：深度代理可以创建子代理来委派工作。您可以在以下选项中指定自定义子代理`subagents`参数。[子代理对于上下文隔离](https://www.dbreunig.com/2025/06/26/how-to-fix-your-context.html#context-quarantine)(保持主要代理人的上下文清晰)以及提供专门说明非常有用。
when:
- ✅ 多步任务,会使主代理的上下文变得混乱
- 需要自定义说明或工具的专用域名
- 需要不同模型功能的任务
- ✅ 当您希望将主要代理人员专注于高层协调时
**何时不使用子代理:**
- ❌ 简单、单步任务
- ❌ 需要保持中间语境
- ❌ 当开销大于收益时

how:字段的字典：
subagent:
name `description` `system_prompt` `tools` `model` `middleware` `interrupt_on`  `skills` 

CompiledSubAgent:
`name`  `description` `runnable` 

---
Human in loop
某些工具操作可能较为敏感,在执行前需要人工审批。深度代理通过朗格法的中断功能支持人为的工作流程。您可以使用以下选项来配置哪些工具需要审批`interrupt_on`参数。

---
长期记忆：
`CompositeBackend`那条路`/memories/`通往一个`StoreBackend`:

### 。短期(过渡)文件系统

- 存放于代理人的状态(通过`StateBackend`)
- 仅在单个线程内坚持
- 线程结束时,文件会丢失
- 通过标准路径访问:`/notes.txt`,`/workspace/draft.md`

2。长期(持久)文件系统

- 存放于朗格法商店(via)`StoreBackend`)
- 贯穿所有线索和对话
- 幸存者代理人重新开始
- 通过前缀路径访问`/memories/`:`/memories/preferences.txt`



---
skills
- A`SKILL.md`包含有关该技能的指令和元数据的文件
- 附加脚本(可选)
- 附加参考信息,例如文档(可选)
- 附加资源,例如模板和其他资源(可选)

当代理启动时,它会读取每个节点的正面`SKILL.md`文件

---
sandbox
在深度代理中,**sandboxes are [backends](https://docs.langchain.com/oss/python/deepagents/backends)**沙盒是定义代理运行环境的后端。与其他仅公开文件操作的后端(State、Filesystem、Store)不同,沙盒后端还向代理提供`execute`用于运行 shell 命令的工具。配置沙盒后端时,代理将获得:

- 所有标准文件系统工具`ls`,`read_file`,`write_file`,`edit_file`,`glob`,`grep`)
- `execute`用于在沙盒中运行任意 shell 命令的工具
- 保护主机系统的安全边界

### `execute`方法

沙盒后端具有简单的架构:提供商必须实现的唯一方法是`execute()`运行一个shell命令并返回其输出。所有其他文件系统操作——`read`,`write`,`edit`,`ls`,`glob`,`grep`——建立在`execute()`由`BaseSandbox`基类,它通过在沙盒中构造脚本并运行它们`execute()`。