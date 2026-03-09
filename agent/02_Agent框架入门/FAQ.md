# 常见问题解答 (FAQ)

## 📋 目录
- [安装和配置](#安装和配置)
- [API相关](#api相关)
- [Agent问题](#agent问题)
- [工具开发](#工具开发)
- [性能优化](#性能优化)
- [错误处理](#错误处理)

---

## 安装和配置

### Q1: 如何安装依赖包？
**A:** 使用以下命令安装所有依赖：
```bash
pip install -r requirements.txt
```

如果遇到网络问题，可以使用国内镜像：
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

### Q2: 提示"ModuleNotFoundError: No module named 'langchain'"
**A:** 这是因为没有安装langchain包。解决方法：
```bash
pip install langchain langchain-openai langchain-anthropic
```

---

### Q3: 如何配置API密钥？
**A:**
1. 复制环境变量模板：
   ```bash
   cp .env.example .env
   ```

2. 编辑`.env`文件，填入你的API密钥：
   ```
   OPENAI_API_KEY=sk-your-key-here
   ANTHROPIC_API_KEY=sk-ant-your-key-here
   ```

3. 确保`.env`文件在项目根目录

---

### Q4: 如何验证环境配置是否正确？
**A:** 运行测试脚本：
```bash
python test_setup.py
```

这会检查：
- Python版本
- 依赖包安装
- 环境变量配置
- 基础功能测试

---

## API相关

### Q5: 提示"AuthenticationError: Invalid API key"
**A:** API密钥配置错误。检查：
1. `.env`文件是否存在
2. API密钥是否正确（没有多余空格）
3. API密钥是否有效（未过期、有额度）

测试API密钥：
```python
from openai import OpenAI
client = OpenAI()  # 会自动读取环境变量
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "test"}]
)
print(response)
```

---

### Q6: 提示"RateLimitError: Rate limit exceeded"
**A:** 超过了API调用频率限制。解决方法：
1. 等待一段时间后重试
2. 实现重试机制：
   ```python
   from langchain.llms import OpenAI
   from langchain.callbacks import get_openai_callback

   llm = OpenAI(
       max_retries=3,
       request_timeout=60
   )
   ```

3. 升级API套餐获得更高限额

---

### Q7: API调用成本太高怎么办？
**A:** 成本控制策略：

1. **使用便宜的模型**：
   ```python
   # 使用gpt-4o-mini而不是gpt-4o
   llm = ChatOpenAI(model="gpt-4o-mini")
   ```

2. **设置max_tokens限制**：
   ```python
   llm = ChatOpenAI(
       model="gpt-4o-mini",
       max_tokens=500  # 限制输出长度
   )
   ```

3. **使用缓存**：
   ```python
   from langchain.cache import InMemoryCache
   import langchain
   langchain.llm_cache = InMemoryCache()
   ```

4. **优先使用Workflow而不是Agent**

5. **设置max_iterations**：
   ```python
   agent_executor = AgentExecutor(
       agent=agent,
       tools=tools,
       max_iterations=3  # 限制迭代次数
   )
   ```

---

## Agent问题

### Q8: Agent一直循环，不返回结果
**A:** 这是常见问题。原因和解决方法：

**原因1：工具描述不清晰**
```python
# ❌ 不好的描述
Tool(name="Search", func=search, description="搜索")

# ✅ 好的描述
Tool(
    name="Search",
    func=search,
    description="在互联网上搜索信息。输入应该是搜索关键词，例如: 'Python教程', 'AI新闻'"
)
```

**原因2：任务太复杂**
- 将复杂任务分解为多个简单任务
- 或者使用Workflow代替Agent

**原因3：没有设置max_iterations**
```python
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    max_iterations=5,  # 添加这个
    early_stopping_method="generate"
)
```

---

### Q9: Agent选择了错误的工具
**A:** 改进工具描述：

```python
# 详细说明工具的用途和输入格式
Tool(
    name="Calculator",
    func=calculate,
    description="""用于数学计算。

支持的运算:
- 基本运算: +, -, *, /, **
- 数学函数: sqrt(), sin(), cos()
- 常量: pi, e

输入格式: 数学表达式字符串
示例输入: "2+2", "sqrt(16)", "sin(pi/2)"

何时使用: 当需要进行数学计算时使用此工具。
"""
)
```

---

### Q10: Agent返回的答案不准确
**A:** 优化策略：

1. **改进System Prompt**：
   ```python
   prompt = ChatPromptTemplate.from_messages([
       ("system", "你是一个专业的AI助手。请仔细思考，给出准确的答案。"),
       ("human", "{input}")
   ])
   ```

2. **降低temperature**：
   ```python
   llm = ChatOpenAI(
       model="gpt-4o-mini",
       temperature=0  # 更确定性的输出
   )
   ```

3. **使用更强大的模型**：
   ```python
   llm = ChatOpenAI(model="gpt-4o")  # 而不是gpt-4o-mini
   ```

4. **添加验证步骤**：
   ```python
   # 让Agent验证自己的答案
   verification_prompt = "请验证上述答案是否正确"
   ```

---

### Q11: 如何查看Agent的推理过程？
**A:** 启用verbose模式：

```python
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,  # 显示详细过程
    return_intermediate_steps=True  # 返回中间步骤
)

result = agent_executor.invoke({"input": "问题"})

# 查看中间步骤
for step in result['intermediate_steps']:
    print(f"Action: {step[0].tool}")
    print(f"Input: {step[0].tool_input}")
    print(f"Output: {step[1]}")
```

---

## 工具开发

### Q12: 如何开发自定义工具？
**A:** 完整示例：

```python
from langchain.agents import Tool
from typing import Optional

def my_custom_tool(input_text: str) -> str:
    """
    自定义工具函数

    Args:
        input_text: 输入文本

    Returns:
        处理结果
    """
    try:
        # 你的工具逻辑
        result = f"处理结果: {input_text}"
        return result
    except Exception as e:
        return f"错误: {str(e)}"

# 创建Tool对象
tool = Tool(
    name="MyCustomTool",
    func=my_custom_tool,
    description="""这是我的自定义工具。

功能: 描述工具的功能
输入: 描述输入格式
输出: 描述输出格式
示例: 提供使用示例
"""
)

# 在Agent中使用
tools = [tool]
```

---

### Q13: 工具如何处理错误？
**A:** 完善的错误处理：

```python
def robust_tool(input_text: str) -> str:
    """带完善错误处理的工具"""

    # 1. 输入验证
    if not input_text or not isinstance(input_text, str):
        return "错误: 输入必须是非空字符串"

    try:
        # 2. 主要逻辑
        result = process(input_text)

        # 3. 结果验证
        if not result:
            return "警告: 未找到结果"

        return f"成功: {result}"

    except ValueError as e:
        return f"输入错误: {str(e)}"
    except ConnectionError as e:
        return f"网络错误: {str(e)}"
    except Exception as e:
        return f"未知错误: {str(e)}"
```

---

### Q14: 工具可以调用外部API吗？
**A:** 可以，示例：

```python
import requests

def api_tool(query: str) -> str:
    """调用外部API的工具"""
    try:
        response = requests.get(
            "https://api.example.com/search",
            params={"q": query},
            timeout=5
        )
        response.raise_for_status()

        data = response.json()
        return f"结果: {data['result']}"

    except requests.Timeout:
        return "错误: API请求超时"
    except requests.RequestException as e:
        return f"错误: API请求失败 - {str(e)}"
```

---

## 性能优化

### Q15: 如何提高Agent响应速度？
**A:** 优化策略：

1. **使用更快的模型**：
   ```python
   llm = ChatOpenAI(model="gpt-4o-mini")  # 比gpt-4o快
   ```

2. **减少工具数量**：
   ```python
   # 只添加必要的工具
   tools = [essential_tool1, essential_tool2]
   ```

3. **使用缓存**：
   ```python
   from langchain.cache import InMemoryCache
   import langchain
   langchain.llm_cache = InMemoryCache()
   ```

4. **并发处理**：
   ```python
   import asyncio

   async def process_multiple():
       tasks = [agent.ainvoke(input) for input in inputs]
       results = await asyncio.gather(*tasks)
       return results
   ```

5. **优化Prompt长度**：
   - 使用简洁的描述
   - 避免冗余信息

---

### Q16: Memory占用太多怎么办？
**A:** Memory优化：

1. **使用WindowMemory**：
   ```python
   from langchain.memory import ConversationBufferWindowMemory

   memory = ConversationBufferWindowMemory(
       k=5  # 只保留最近5轮对话
   )
   ```

2. **使用SummaryMemory**：
   ```python
   from langchain.memory import ConversationSummaryMemory

   memory = ConversationSummaryMemory(llm=llm)
   ```

3. **定期清理**：
   ```python
   # 每N轮对话后清理
   if conversation_count > 10:
       memory.clear()
       conversation_count = 0
   ```

---

## 错误处理

### Q17: 如何处理网络超时？
**A:** 设置超时和重试：

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o-mini",
    request_timeout=30,  # 30秒超时
    max_retries=3  # 重试3次
)

# 或者手动重试
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
def call_with_retry():
    return agent_executor.invoke({"input": "问题"})
```

---

### Q18: 如何处理解析错误？
**A:** 启用错误处理：

```python
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    handle_parsing_errors=True,  # 自动处理解析错误
    verbose=True
)

# 或者自定义错误处理
def custom_error_handler(error):
    return f"发生错误，请重新表述问题: {error}"

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    handle_parsing_errors=custom_error_handler
)
```

---

### Q19: 如何记录日志便于调试？
**A:** 配置日志系统：

```python
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agent.log'),
        logging.StreamHandler()
    ]
)

# 使用callbacks
from langchain.callbacks import StdOutCallbackHandler

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    callbacks=[StdOutCallbackHandler()],
    verbose=True
)
```

---

### Q20: 遇到"Context length exceeded"错误
**A:** 这是因为输入太长。解决方法：

1. **减少Memory保存的历史**：
   ```python
   memory = ConversationBufferWindowMemory(k=3)
   ```

2. **使用SummaryMemory**：
   ```python
   memory = ConversationSummaryMemory(llm=llm)
   ```

3. **分块处理长文本**：
   ```python
   from langchain.text_splitter import RecursiveCharacterTextSplitter

   splitter = RecursiveCharacterTextSplitter(
       chunk_size=1000,
       chunk_overlap=100
   )
   chunks = splitter.split_text(long_text)
   ```

4. **使用支持更长上下文的模型**：
   ```python
   llm = ChatOpenAI(model="gpt-4o")  # 支持128K tokens
   ```

---

## 其他问题

### Q21: 如何在生产环境部署Agent？
**A:** 生产部署建议：

1. **使用FastAPI封装**：
   ```python
   from fastapi import FastAPI

   app = FastAPI()

   @app.post("/chat")
   async def chat(message: str):
       result = agent_executor.invoke({"input": message})
       return {"response": result['output']}
   ```

2. **添加监控**：
   - 记录所有请求和响应
   - 监控API调用次数和成本
   - 设置告警

3. **实现缓存**：
   - 缓存常见问题的答案
   - 使用Redis等缓存系统

4. **错误处理和降级**：
   - 完善的异常处理
   - API失败时的降级方案

---

### Q22: 如何测试Agent？
**A:** 测试策略：

```python
import pytest

def test_agent_basic():
    """测试基础功能"""
    result = agent_executor.invoke({"input": "2+2等于几？"})
    assert "4" in result['output']

def test_agent_with_tool():
    """测试工具调用"""
    result = agent_executor.invoke({"input": "计算10*5"})
    assert "50" in result['output']

def test_agent_error_handling():
    """测试错误处理"""
    result = agent_executor.invoke({"input": "无效输入@#$%"})
    assert result is not None
```

---

### Q23: 推荐的学习资源？
**A:**

**官方文档**：
- [LangChain文档](https://python.langchain.com/)
- [LangChain API参考](https://api.python.langchain.com/)

**在线课程**：
- DeepLearning.AI - LangChain系列
- Udemy - LangChain实战

**社区**：
- [LangChain Discord](https://discord.gg/langchain)
- [GitHub Discussions](https://github.com/langchain-ai/langchain/discussions)
- Reddit r/LangChain

**博客**：
- LangChain官方博客
- Towards Data Science
- Medium #LangChain

---

### Q24: 如何获取帮助？
**A:**

1. **查看文档**：先查看本项目的文档和官方文档
2. **搜索Issues**：在GitHub上搜索类似问题
3. **社区提问**：在Discord或Reddit提问
4. **提交Issue**：如果是bug，提交GitHub Issue

**提问技巧**：
- 提供完整的错误信息
- 提供最小可复现示例
- 说明你已经尝试的解决方法
- 提供环境信息（Python版本、包版本等）

---

## 📞 联系方式

如果以上FAQ没有解决你的问题：

1. 查看项目的其他文档
2. 在GitHub提交Issue
3. 加入Discord社区讨论
4. 查看官方文档

---

**最后更新**: 2026-01-25
**版本**: 1.0
