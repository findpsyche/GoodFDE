# Web2 全栈开发完整图景
> 目标：面对任何Web2前后端、中间件、DevOps追问都能接住

---

## 一、全局架构图：一个请求的完整生命周期

```
用户浏览器
    ↓ HTTPS请求
[CDN / 云WAF] — 静态资源缓存 + DDoS防护
    ↓
[Nginx 反向代理] — 负载均衡 / SSL终止 / 静态文件服务 / 限流
    ↓ upstream转发
[后端应用服务器] — FastAPI/Flask/Django/Express/Spring Boot
    ↓                ↓                ↓
[关系数据库]    [缓存层]         [消息队列]
PostgreSQL/     Redis/           Celery+RabbitMQ/
MySQL           Memcached        Kafka
    ↓                               ↓
[对象存储]                    [异步Worker]
MinIO/OSS/S3                  后台任务处理
    ↓
[监控体系]
Prometheus + Grafana / ELK / OpenTelemetry
```

**一句话总结：** 浏览器 → CDN → Nginx → 后端API → 数据库/缓存/队列 → 返回响应。每一层都有明确职责，这就是「关注点分离」。

---

## 二、前端框架横向对比

### 2.1 四大前端框架

| 维度         | React                  | Vue 3              | Angular               | Svelte          |
| ---------- | ---------------------- | ------------------ | --------------------- | --------------- |
| 开发者        | Meta (Facebook)        | 尤雨溪(独立)            | Google                | Rich Harris     |
| 核心理念       | UI = f(state)          | 渐进式框架              | 全功能平台                 | 编译时框架           |
| 学习曲线       | 中等(JSX)                | 低(模板语法)            | 高(TypeScript+RxJS+DI) | 低               |
| 虚拟DOM      | ✅ Fiber架构              | ✅ 响应式+虚拟DOM        | ✅ 增量DOM               | ❌ 编译时生成DOM操作    |
| 状态管理       | useState/Zustand/Redux | ref/reactive/Pinia | Service+RxJS          | $: 响应式声明        |
| 生态规模       | 最大                     | 大(中国尤其强)           | 大(企业级)                | 小但增长快           |
| 包体积        | ~42KB                  | ~33KB              | ~143KB                | ~1.6KB(运行时几乎为0) |
| 适用场景       | 大型SPA、跨平台(RN)          | 中小项目、快速开发          | 大型企业应用                | 性能敏感、小项目        |
| TypeScript | 支持好                    | 支持好(Vue3原生TS)      | 原生TS                  | 支持              |

**面试回答模板：**
```
"React生态最大、招聘市场最广，适合大型项目和团队协作；
 Vue学习曲线低、中文社区强，适合快速开发和中小项目；
 Angular是全功能平台，适合大型企业级应用但学习成本高；
 Svelte编译时框架，运行时几乎为零，性能最优但生态最小。
 我的项目用React，因为和AI应用生态(Next.js/Vercel)配合最好。"
```

### 2.2 全栈框架对比

| 维度    | Next.js            | Nuxt 3                | Remix           | SvelteKit    |
| ----- | ------------------ | --------------------- | --------------- | ------------ |
| 基于    | React              | Vue 3                 | React           | Svelte       |
| 渲染模式  | SSR/SSG/ISR/CSR全支持 | SSR/SSG/CSR           | SSR为主           | SSR/SSG/CSR  |
| 路由    | 文件系统路由(App Router) | 文件系统路由                | 文件系统路由          | 文件系统路由       |
| API路由 | ✅ Route Handlers   | ✅ server/api/         | ✅ loader/action | ✅ +server.ts |
| 部署    | Vercel最优/自托管可      | 任意                    | 任意              | 任意           |
| 数据获取  | Server Components  | useFetch/useAsyncData | loader函数        | load函数       |
| 市场份额  | 最大                 | Vue生态最大               | 增长中             | 增长中          |

### 2.3 CSR vs SSR vs SSG vs ISR

```
CSR (Client-Side Rendering) — 客户端渲染
  浏览器下载空HTML → 下载JS → JS执行渲染页面
  优点：交互流畅、服务器压力小
  缺点：首屏白屏、SEO差
  适用：后台管理系统、SPA应用

SSR (Server-Side Rendering) — 服务端渲染
  服务器执行React/Vue → 生成完整HTML → 发送给浏览器 → 客户端hydrate
  优点：首屏快、SEO好
  缺点：服务器压力大、TTFB增加
  适用：内容型网站、需要SEO的页面

SSG (Static Site Generation) — 静态生成
  构建时生成所有HTML → 部署到CDN
  优点：最快（纯静态）、最便宜
  缺点：数据更新需要重新构建
  适用：博客、文档站、营销页

ISR (Incremental Static Regeneration) — 增量静态再生
  SSG + 按需更新：首次请求返回缓存页面，后台重新生成
  优点：兼顾SSG的速度和数据新鲜度
  缺点：Next.js特有概念
  适用：电商产品页、新闻列表
```

**面试一句话：** "CSR适合后台系统，SSR适合需要SEO的动态页面，SSG适合内容不常变的页面，ISR是SSG的升级版支持增量更新。我的AI应用用SSR因为需要流式输出。"

---

## 三、前端工程化

### 3.1 构建工具对比

| 维度 | Vite | Webpack | Turbopack | esbuild |
|------|------|---------|-----------|---------|
| 开发者 | 尤雨溪 | Tobias Koppers | Vercel | Evan Wallace |
| 语言 | Go(esbuild)+Rust(SWC) | JavaScript | Rust | Go |
| 开发服务器 | 原生ESM，按需编译 | 全量打包 | 增量编译 | 极快但功能少 |
| HMR速度 | 极快(<50ms) | 慢(秒级) | 快 | 不支持 |
| 生产构建 | Rollup | 自身 | 开发中 | 快但插件少 |
| 配置复杂度 | 低 | 高 | 低(Next.js内置) | 低 |
| 生态 | 快速增长 | 最成熟 | Next.js专用 | 底层工具 |

**为什么Vite取代Webpack：**
```
Webpack的问题：开发服务器启动时需要打包所有模块 → 项目越大启动越慢
Vite的解法：利用浏览器原生ESM，开发时不打包，按需编译请求的模块
  → 启动时间从分钟级降到秒级
  → HMR从秒级降到毫秒级
```

### 3.2 TypeScript

```
为什么用TypeScript？
  JavaScript是动态类型 → 运行时才发现类型错误
  TypeScript是静态类型 → 编译时就能发现错误
  → 大型项目必备：减少bug、提升可维护性、IDE智能提示

核心概念：
  - interface / type：定义数据结构
  - generics：泛型，复用类型逻辑
  - union types：联合类型 string | number
  - type guards：类型守卫，运行时类型检查
  - utility types：Partial<T>, Pick<T,K>, Omit<T,K>
```

### 3.3 CSS方案对比

| 方案 | 原理 | 优点 | 缺点 | 适用 |
|------|------|------|------|------|
| Tailwind CSS | 原子化CSS类 | 开发快、包小、一致性 | 类名长、学习成本 | 现代项目首选 |
| CSS Modules | 局部作用域CSS | 无冲突、标准CSS | 需要配置 | React项目 |
| styled-components | CSS-in-JS | 动态样式、组件化 | 运行时开销 | 需要动态主题 |
| Sass/Less | CSS预处理器 | 变量、嵌套、mixin | 需要编译 | 传统项目 |
| UnoCSS | 原子化CSS(按需) | 比Tailwind更快 | 生态小 | 性能极致 |

### 3.4 ESLint + Prettier

```
ESLint：代码质量检查（找bug、强制规范）
Prettier：代码格式化（统一风格）
两者配合：ESLint管逻辑，Prettier管格式，互不冲突

配置：
  .eslintrc.js → 规则配置
  .prettierrc → 格式配置
  husky + lint-staged → git commit时自动检查
```

---

## 四、状态管理

### 4.1 React 状态管理演进

```
useState（组件内状态）
  ↓ 状态需要跨组件共享
useContext（简单全局状态）
  ↓ 状态逻辑复杂、性能问题
Zustand（轻量级状态库，推荐）
  ↓ 超大型应用、需要中间件
Redux Toolkit（重量级，企业标准）
```

| 方案                    | 复杂度 | 包大小   | 适用场景            |
| --------------------- | --- | ----- | --------------- |
| useState + useContext | 最低  | 0     | 小项目、少量全局状态      |
| Zustand               | 低   | ~1KB  | 中型项目、推荐默认选择     |
| Jotai                 | 低   | ~2KB  | 原子化状态、细粒度更新     |
| Redux Toolkit         | 中   | ~11KB | 大型项目、需要DevTools |
| MobX                  | 中   | ~16KB | 响应式编程偏好         |

### 4.2 Vue 状态管理

```
ref/reactive（组件内响应式状态）
  ↓ 跨组件共享
Pinia（Vue3官方推荐，替代Vuex）
  - 类型安全、DevTools支持
  - 比Vuex更简洁（去掉了mutations）
```

---

## 五、后端框架横向对比

| 维度 | FastAPI | Flask | Django | Express.js | Spring Boot |
|------|---------|-------|--------|------------|-------------|
| 语言 | Python | Python | Python | Node.js | Java/Kotlin |
| 异步 | ✅ 原生ASGI | ❌ WSGI(同步) | 部分(ASGI可选) | ✅ 事件循环 | ✅ WebFlux |
| 类型验证 | ✅ Pydantic自动 | ❌ 手动 | ✅ DRF Serializer | ❌ 手动 | ✅ Bean Validation |
| API文档 | ✅ 自动OpenAPI | ❌ 需flask-restx | ✅ DRF自动 | ❌ 需swagger | ✅ SpringDoc |
| ORM | SQLAlchemy | SQLAlchemy | 自带Django ORM | Prisma/TypeORM | JPA/Hibernate |
| 学习曲线 | 低 | 最低 | 中(全家桶) | 低 | 高(Spring生态) |
| 性能 | 高(异步) | 低(同步) | 中 | 高(V8引擎) | 高(JVM) |
| 适用 | AI/API服务 | 小项目/原型 | 全栈Web | 全栈/实时 | 企业级 |

**面试关键点：为什么AI应用选FastAPI？**
```
1. 原生异步：LLM API调用是IO密集型，async/await天然适配
2. Pydantic集成：LLM输出验证、API Schema定义一套模型搞定
3. 流式响应：StreamingResponse原生支持SSE，实现打字机效果
4. 自动文档：开发调试效率高
5. 轻量：不像Django带一堆用不上的功能
```

---

## 六、API设计规范

### 6.1 RESTful API 设计

```
核心原则：资源(名词) + HTTP方法(动词)

GET    /api/users          → 获取用户列表
GET    /api/users/{id}     → 获取单个用户
POST   /api/users          → 创建用户
PUT    /api/users/{id}     → 全量更新用户
PATCH  /api/users/{id}     → 部分更新用户
DELETE /api/users/{id}     → 删除用户

状态码规范：
  200 OK           → 成功
  201 Created      → 创建成功
  204 No Content   → 删除成功
  400 Bad Request  → 参数错误
  401 Unauthorized → 未认证
  403 Forbidden    → 无权限
  404 Not Found    → 资源不存在
  422 Unprocessable → 验证失败
  429 Too Many Req → 限流
  500 Internal     → 服务器错误
```

### 6.2 RESTful vs GraphQL

| 维度 | RESTful | GraphQL |
|------|---------|---------|
| 数据获取 | 固定端点返回固定结构 | 客户端指定需要的字段 |
| Over-fetching | 常见(返回多余字段) | 不存在 |
| Under-fetching | 常见(需要多次请求) | 不存在(一次查询) |
| 缓存 | HTTP缓存天然支持 | 需要额外处理 |
| 学习成本 | 低 | 中(Schema定义) |
| 适用 | 大多数场景 | 复杂嵌套数据、移动端 |

### 6.3 API版本管理

```
方案1：URL路径版本（最常用）
  /api/v1/users
  /api/v2/users

方案2：Header版本
  Accept: application/vnd.api+json;version=2

方案3：查询参数
  /api/users?version=2

推荐方案1，简单直观。
```

### 6.4 统一响应格式

```python
# FastAPI 统一响应
class ApiResponse(BaseModel):
    code: int = 200
    message: str = "success"
    data: Any = None

class PageResponse(ApiResponse):
    total: int
    page: int
    page_size: int

# 统一错误响应
class ErrorResponse(BaseModel):
    code: int
    message: str
    detail: str | None = None
```

---

## 七、认证鉴权

### 7.1 JWT vs Session vs OAuth2.0

| 维度 | JWT | Session | OAuth2.0 |
|------|-----|---------|----------|
| 存储位置 | 客户端(localStorage/Cookie) | 服务端(内存/Redis) | 授权服务器 |
| 无状态 | ✅ 服务端不存储 | ❌ 服务端存储session | 取决于实现 |
| 扩展性 | 好(无状态，天然支持分布式) | 差(需要共享session) | 好 |
| 安全性 | 中(无法主动失效) | 高(服务端可控) | 高 |
| 适用 | API服务、微服务 | 传统Web应用 | 第三方登录 |

### 7.2 JWT 完整流程

```
1. 用户登录：POST /api/auth/login {username, password}
2. 服务端验证密码 → 生成JWT Token
   JWT = Header.Payload.Signature
   Header: {"alg": "HS256", "typ": "JWT"}
   Payload: {"sub": "user_id", "exp": 过期时间, "role": "admin"}
   Signature: HMAC-SHA256(Header.Payload, SECRET_KEY)
3. 返回Token给客户端
4. 客户端每次请求带上: Authorization: Bearer <token>
5. 服务端验证签名 + 检查过期时间

刷新机制（Access Token + Refresh Token）：
  Access Token: 短期(15min-1h)，用于API访问
  Refresh Token: 长期(7-30天)，用于刷新Access Token
  → Access Token过期 → 用Refresh Token获取新的Access Token
  → Refresh Token过期 → 重新登录
```

```python
# FastAPI JWT实现
from jose import jwt
from datetime import datetime, timedelta

SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"

def create_access_token(data: dict, expires_delta: timedelta = timedelta(hours=1)):
    to_encode = data.copy()
    to_encode["exp"] = datetime.utcnow() + expires_delta
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(token: str):
    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    return payload

# 依赖注入获取当前用户
async def get_current_user(token: str = Depends(oauth2_scheme)):
    payload = verify_token(token)
    user = await get_user(payload["sub"])
    if not user:
        raise HTTPException(status_code=401)
    return user
```

### 7.3 OAuth2.0 授权码流程

```
1. 用户点击"用GitHub登录"
2. 重定向到GitHub授权页面
   GET https://github.com/login/oauth/authorize?
     client_id=xxx&redirect_uri=xxx&scope=user:email
3. 用户同意授权 → GitHub回调你的redirect_uri，带上code
4. 你的后端用code换取access_token
   POST https://github.com/login/oauth/access_token
     {client_id, client_secret, code}
5. 用access_token调用GitHub API获取用户信息
6. 创建/关联本地用户，生成你自己的JWT
```

---

## 八、数据库对比

### 8.1 关系型数据库

| 维度 | PostgreSQL | MySQL |
|------|-----------|-------|
| 特点 | 功能最全、标准兼容 | 速度快、生态大 |
| JSON支持 | ✅ JSONB(可索引) | ✅ JSON(功能弱) |
| 向量扩展 | ✅ pgvector | ❌ 需第三方 |
| 全文搜索 | ✅ 内置tsvector | ✅ 内置FULLTEXT |
| 事务 | MVCC，严格ACID | InnoDB支持ACID |
| 复杂查询 | 强(CTE/窗口函数/递归) | 中等 |
| 适用 | AI应用、复杂业务 | Web应用、高并发读 |

**面试关键：为什么AI应用选PostgreSQL？**
```
1. pgvector扩展：向量和业务数据在同一个数据库，JOIN查询方便
2. JSONB：存储LLM的非结构化输出，支持索引查询
3. 复杂查询能力：窗口函数做数据分析、CTE做递归查询
4. 严格的数据完整性：ACID事务保证数据一致性
```

### 8.2 Redis

```
本质：内存键值存储，支持多种数据结构

核心数据结构：
  String → 缓存、计数器、分布式锁
  Hash   → 对象存储（用户信息）
  List   → 消息队列、最近访问
  Set    → 标签、去重
  ZSet   → 排行榜、延迟队列
  Stream → 消息流（类Kafka）

AI应用中的用途：
  1. LLM响应缓存：相同问题直接返回缓存（语义缓存）
  2. Session存储：用户对话历史
  3. 限流：Token Bucket算法控制API调用频率
  4. 分布式锁：防止重复处理
  5. Pub/Sub：实时通知

关键命令：
  SET key value EX 3600    → 设置带过期时间的缓存
  GET key                  → 获取缓存
  INCR key                 → 原子计数
  LPUSH / RPOP             → 队列操作
  ZADD / ZRANGE            → 有序集合操作
```

### 8.3 MongoDB

```
本质：文档型数据库，存储JSON文档（BSON）

适用场景：
  - 数据结构不固定（Schema-less）
  - 嵌套文档多（如日志、事件）
  - 读多写少、水平扩展需求

不适用：
  - 需要复杂JOIN的场景
  - 强事务需求
  - 数据关系复杂

AI应用中：适合存储LLM对话日志、非结构化数据
但PostgreSQL + JSONB通常是更好的选择（一个数据库搞定）
```

---

## 九、缓存策略

### 9.1 缓存三大问题

```
缓存穿透（Cache Penetration）
  问题：查询不存在的数据 → 缓存没有 → 每次都打数据库
  解决：
    1. 布隆过滤器：快速判断key是否存在
    2. 缓存空值：不存在的key也缓存（设短过期时间）

缓存击穿（Cache Breakdown）
  问题：热点key过期瞬间 → 大量请求同时打数据库
  解决：
    1. 互斥锁：只允许一个请求回源，其他等待
    2. 永不过期：热点数据不设过期，后台异步更新

缓存雪崩（Cache Avalanche）
  问题：大量key同时过期 → 数据库瞬间压力暴增
  解决：
    1. 过期时间加随机值：避免同时过期
    2. 多级缓存：本地缓存 + Redis + 数据库
    3. 限流降级：保护数据库
```

### 9.2 缓存更新策略

```
Cache Aside（旁路缓存）— 最常用
  读：先读缓存 → 没有则读DB → 写入缓存
  写：先更新DB → 再删除缓存（不是更新缓存！）
  为什么删除而不是更新？避免并发下的数据不一致

Read/Write Through（读写穿透）
  缓存层代理所有读写，应用只和缓存交互
  适用：缓存中间件支持（如Redis + 自定义逻辑）

Write Behind（异步写回）
  写操作只写缓存，异步批量写DB
  优点：写性能极高
  缺点：数据可能丢失
  适用：日志、计数器等允许少量丢失的场景
```

---

## 十、消息队列

### 10.1 为什么需要消息队列

```
三大作用：
  1. 异步处理：耗时操作不阻塞主流程（如发邮件、生成报告）
  2. 削峰填谷：突发流量先进队列，消费者按能力处理
  3. 服务解耦：生产者和消费者不直接依赖

AI应用场景：
  - LLM批量推理任务排队
  - 文档入库（分块→向量化→存储）异步处理
  - 评测任务异步执行
```

### 10.2 Celery + RabbitMQ/Redis

```python
# Celery：Python异步任务框架
from celery import Celery

app = Celery('tasks', broker='redis://localhost:6379/0')

@app.task
def process_document(doc_id: str):
    """异步处理文档入库"""
    doc = load_document(doc_id)
    chunks = split_document(doc)
    embeddings = embed_chunks(chunks)
    store_vectors(embeddings)
    return {"status": "done", "chunks": len(chunks)}

# 调用
result = process_document.delay("doc_123")  # 异步发送
result.get(timeout=300)  # 等待结果
```

### 10.3 RabbitMQ vs Kafka

| 维度 | RabbitMQ | Kafka |
|------|----------|-------|
| 模型 | 消息队列(点对点/发布订阅) | 分布式日志(发布订阅) |
| 消息保留 | 消费后删除 | 持久化保留(可回溯) |
| 吞吐量 | 万级/秒 | 百万级/秒 |
| 延迟 | 微秒级 | 毫秒级 |
| 消息顺序 | 队列内有序 | 分区内有序 |
| 适用 | 任务队列、RPC | 日志收集、事件流、大数据 |
| 运维复杂度 | 低 | 高(ZooKeeper/KRaft) |

**面试一句话：** "RabbitMQ适合任务队列场景（如Celery后端），Kafka适合高吞吐的事件流场景（如日志收集、实时数据管道）。小规模用RabbitMQ或Redis作为broker就够了。"

---

## 十一、Nginx 反向代理与负载均衡

### 11.1 Nginx 核心功能

```
1. 反向代理：客户端 → Nginx → 后端服务器
   隐藏后端真实IP，统一入口

2. 负载均衡：分发请求到多个后端实例
   算法：轮询、加权轮询、IP哈希、最少连接

3. 静态文件服务：直接返回HTML/CSS/JS/图片
   不经过后端应用，性能极高

4. SSL终止：Nginx处理HTTPS，后端用HTTP
   减轻后端SSL计算负担

5. 限流：防止恶意请求打垮服务器
```

### 11.2 Nginx 配置示例

```nginx
# /etc/nginx/nginx.conf

# 负载均衡配置
upstream backend {
    # 轮询（默认）
    server 127.0.0.1:8001;
    server 127.0.0.1:8002;
    server 127.0.0.1:8003;
    
    # 加权轮询（权重高的分配更多请求）
    # server 127.0.0.1:8001 weight=3;
    # server 127.0.0.1:8002 weight=1;
    
    # IP哈希（同一IP总是访问同一后端）
    # ip_hash;
    
    # 最少连接（分配给连接数最少的后端）
    # least_conn;
}

server {
    listen 80;
    server_name example.com;
    
    # 静态文件
    location /static/ {
        alias /var/www/static/;
        expires 30d;  # 缓存30天
    }
    
    # API请求代理到后端
    location /api/ {
        proxy_pass http://backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        
        # 超时设置（LLM请求可能很长）
        proxy_read_timeout 300s;
        proxy_connect_timeout 10s;
    }
    
    # 限流（每秒最多10个请求）
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;
    location /api/chat {
        limit_req zone=api_limit burst=20 nodelay;
        proxy_pass http://backend;
    }
    
    # SSE流式响应（LLM打字机效果）
    location /api/chat/stream {
        proxy_pass http://backend;
        proxy_buffering off;  # 关键：禁用缓冲
        proxy_cache off;
        proxy_set_header Connection '';
        chunked_transfer_encoding on;
    }
}

# HTTPS配置
server {
    listen 443 ssl http2;
    server_name example.com;
    
    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    
    # 其他配置同上...
}
```

---

## 十二、Docker 最佳实践

### 12.1 多阶段构建

```dockerfile
# 前端 Dockerfile（多阶段构建）
# 阶段1：构建
FROM node:18-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
RUN npm run build

# 阶段2：运行
FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]

# 优点：最终镜像只包含nginx和构建产物，体积小
```

```dockerfile
# 后端 Dockerfile
FROM python:3.11-slim

WORKDIR /app

# 先复制依赖文件（利用Docker缓存）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 再复制代码
COPY . .

# 非root用户运行（安全）
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 12.2 docker-compose.yml

```yaml
version: '3.8'

services:
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "80:80"
    depends_on:
      - backend
    networks:
      - app-network

  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@postgres:5432/db
      - REDIS_URL=redis://redis:6379/0
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_started
    volumes:
      - ./backend:/app  # 开发时挂载代码（热重载）
    networks:
      - app-network

  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_DB: hydro_ai
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U user"]
      interval: 5s
      timeout: 5s
      retries: 5
    networks:
      - app-network

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redisdata:/data
    networks:
      - app-network

volumes:
  pgdata:
  redisdata:

networks:
  app-network:
    driver: bridge
```

### 12.3 .dockerignore

```
# 减小构建上下文，加快构建速度
node_modules
__pycache__
*.pyc
.git
.env
.vscode
*.log
dist
build
```

---

## 十三、CI/CD

### 13.1 GitHub Actions 示例

```yaml
# .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest
      
      - name: Run tests
        run: pytest tests/
      
      - name: Lint
        run: |
          pip install ruff
          ruff check .

  build-and-push:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Build Docker image
        run: docker build -t myapp:${{ github.sha }} .
      
      - name: Push to registry
        run: |
          echo ${{ secrets.DOCKER_PASSWORD }} | docker login -u ${{ secrets.DOCKER_USERNAME }} --password-stdin
          docker push myapp:${{ github.sha }}

  deploy:
    needs: build-and-push
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to server
        uses: appleboy/ssh-action@master
        with:
          host: ${{ secrets.SERVER_HOST }}
          username: ${{ secrets.SERVER_USER }}
          key: ${{ secrets.SSH_PRIVATE_KEY }}
          script: |
            cd /app
            docker-compose pull
            docker-compose up -d
            docker system prune -f
```

### 13.2 CI/CD 流程

```
代码提交 → GitHub
    ↓
自动触发 GitHub Actions
    ↓
1. 运行测试（pytest/jest）
2. 代码检查（ruff/eslint）
3. 构建Docker镜像
4. 推送到镜像仓库（Docker Hub/阿里云ACR）
    ↓
5. SSH到生产服务器
6. 拉取最新镜像
7. docker-compose up -d（滚动更新）
8. 健康检查
    ↓
部署完成，发送通知（钉钉/Slack）
```

---

## 十四、监控体系

### 14.1 监控三大支柱

```
Metrics（指标）
  - 数值型时序数据：CPU、内存、QPS、延迟
  - 工具：Prometheus + Grafana

Logs（日志）
  - 文本型事件记录：错误日志、访问日志
  - 工具：ELK (Elasticsearch + Logstash + Kibana)

Traces（链路追踪）
  - 分布式请求追踪：一个请求经过哪些服务
  - 工具：Jaeger、Zipkin、OpenTelemetry
```

### 14.2 Prometheus + Grafana

```python
# FastAPI集成Prometheus
from prometheus_client import Counter, Histogram, generate_latest
from fastapi import FastAPI, Response

app = FastAPI()

# 定义指标
request_count = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint'])
request_duration = Histogram('http_request_duration_seconds', 'HTTP request duration')

@app.middleware("http")
async def metrics_middleware(request, call_next):
    with request_duration.time():
        response = await call_next(request)
    request_count.labels(method=request.method, endpoint=request.url.path).inc()
    return response

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'fastapi'
    static_configs:
      - targets: ['backend:8000']
    scrape_interval: 15s
```

### 14.3 关键监控指标

```
系统层：
  - CPU使用率、内存使用率、磁盘IO
  - 网络流量、TCP连接数

应用层：
  - QPS（每秒请求数）
  - 响应时间（P50/P95/P99）
  - 错误率（4xx/5xx比例）
  - 并发连接数

AI应用特有：
  - LLM API调用延迟
  - Token消耗速率
  - 向量检索延迟
  - 缓存命中率
  - 队列积压数量
```

### 14.4 日志聚合（ELK）

```
应用日志 → Filebeat → Logstash → Elasticsearch → Kibana可视化

结构化日志示例：
{
  "timestamp": "2024-01-01T12:00:00Z",
  "level": "INFO",
  "service": "backend",
  "request_id": "abc123",
  "user_id": "user_456",
  "endpoint": "/api/chat",
  "duration_ms": 1234,
  "llm_tokens": 500,
  "message": "Chat request completed"
}
```

---

## 十五、从0到1完整开发流程

### 15.1 项目初始化

```bash
# 1. 创建项目结构
mkdir my-ai-app && cd my-ai-app
mkdir frontend backend

# 2. 前端初始化（Next.js）
cd frontend
npx create-next-app@latest . --typescript --tailwind --app
npm install axios zustand

# 3. 后端初始化（FastAPI）
cd ../backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install fastapi uvicorn sqlalchemy asyncpg redis python-jose

# 4. 数据库初始化
docker run -d --name postgres -e POSTGRES_PASSWORD=password -p 5432:5432 pgvector/pgvector:pg16
docker run -d --name redis -p 6379:6379 redis:7-alpine

# 5. Git初始化
git init
echo "venv/\nnode_modules/\n.env\n__pycache__/" > .gitignore
git add .
git commit -m "Initial commit"
```

### 15.2 开发阶段

```
第1天：搭建基础架构
  - FastAPI项目结构（routers/models/schemas/services）
  - 数据库连接（SQLAlchemy AsyncSession）
  - 统一响应格式
  - 全局异常处理
  - CORS配置

第2-3天：核心业务逻辑
  - 用户认证（JWT）
  - 数据库模型设计
  - CRUD API实现
  - LLM集成（OpenAI/Claude SDK）

第4-5天：前端开发
  - 页面布局（Tailwind CSS）
  - API调用封装（axios）
  - 状态管理（Zustand）
  - 流式响应处理（SSE）

第6天：RAG/Agent功能
  - 向量数据库集成（ChromaDB/Milvus）
  - Embedding模型加载
  - Tool Calling实现
  - 评测框架搭建

第7天：部署上线
  - Dockerfile编写
  - docker-compose配置
  - Nginx反向代理
  - 域名+SSL证书
  - 监控配置
```

### 15.3 技术选型决策树

```
前端框架选择：
  需要SEO？
    是 → Next.js (SSR)
    否 → 需要极致性能？
           是 → Svelte
           否 → React (CRA/Vite)

后端框架选择：
  语言偏好？
    Python → 需要异步？
               是 → FastAPI
               否 → Flask（小项目）/ Django（全栈）
    Node.js → Express / Nest.js
    Java → Spring Boot

数据库选择：
  需要向量检索？
    是 → PostgreSQL + pgvector
    否 → 需要复杂查询？
           是 → PostgreSQL
           否 → MySQL

状态管理选择：
  项目规模？
    小 → useState + useContext
    中 → Zustand
    大 → Redux Toolkit
```

---

## 十六、20个高频面试问题速查

### 16.1 前端问题

**Q1: React的虚拟DOM是什么？为什么需要它？**
```
虚拟DOM是真实DOM的JavaScript对象表示。
React通过diff算法比较新旧虚拟DOM，只更新变化的部分。
优点：减少直接操作DOM（慢），批量更新，跨平台（React Native）。
```

**Q2: useEffect的依赖数组是什么？**
```
依赖数组决定effect何时重新执行：
  [] → 只在mount时执行一次
  [dep1, dep2] → dep1或dep2变化时执行
  不传 → 每次render都执行（通常是bug）
```

**Q3: SSR和CSR的区别？**
```
CSR：浏览器下载JS → 执行渲染 → 首屏慢、SEO差
SSR：服务器生成HTML → 浏览器直接显示 → 首屏快、SEO好
我的AI应用用SSR因为需要流式输出，服务端渲染更合适。
```

**Q4: 如何优化React性能？**
```
1. React.memo：避免不必要的重渲染
2. useMemo/useCallback：缓存计算结果/函数
3. 代码分割：React.lazy + Suspense
4. 虚拟列表：react-window处理长列表
5. 避免内联对象/函数：每次render都是新引用
```

### 16.2 后端问题

**Q5: FastAPI为什么比Flask快？**
```
FastAPI基于ASGI（异步），Flask基于WSGI（同步）。
异步可以在IO等待时处理其他请求，同步会阻塞线程。
LLM API调用是IO密集型，异步天然适配。
```

**Q6: 什么是N+1查询问题？如何解决？**
```
问题：查询N个用户 → 每个用户再查询关联数据 → N+1次查询
解决：
  1. 预加载（eager loading）：SQLAlchemy的joinedload
  2. 批量查询：一次查出所有关联数据
  3. DataLoader（GraphQL）：批量+缓存
```

**Q7: JWT的优缺点？**
```
优点：无状态、跨域友好、天然支持分布式
缺点：无法主动失效（需要黑名单）、payload不能太大
适用：API服务、微服务架构
```

**Q8: 如何防止SQL注入？**
```
1. 使用ORM（SQLAlchemy）：自动参数化查询
2. 参数化查询：cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
3. 输入验证：Pydantic自动验证类型
4. 最小权限：数据库用户只给必要权限
```

### 16.3 数据库问题

**Q9: 数据库索引的原理？**
```
索引是B+树结构，叶子节点存数据指针。
查询时：O(log N)复杂度，比全表扫描O(N)快。
代价：写入变慢（需要维护索引）、占用空间。
何时加索引：WHERE/JOIN/ORDER BY的字段。
```

**Q10: Redis为什么快？**
```
1. 纯内存操作（ns级延迟）
2. 单线程模型（无锁竞争）
3. IO多路复用（epoll）
4. 高效数据结构（SDS、跳表、压缩列表）
```

**Q11: 缓存穿透/击穿/雪崩的区别？**
```
穿透：查不存在的key → 每次打DB → 布隆过滤器/缓存空值
击穿：热点key过期 → 瞬间打DB → 互斥锁/永不过期
雪崩：大量key同时过期 → DB压力暴增 → 过期时间加随机值
```

### 16.4 系统设计问题

**Q12: 如何设计一个高并发的秒杀系统？**
```
1. 前端：按钮防抖、验证码
2. 网关：限流（令牌桶）
3. 缓存：Redis预减库存
4. 队列：请求进MQ异步处理
5. 数据库：乐观锁（version字段）防止超卖
6. 降级：库存不足直接返回，不查DB
```

**Q13: 如何实现分布式锁？**
```
Redis方案：
  SET lock_key unique_value NX EX 30
  NX：不存在才设置（原子操作）
  EX：过期时间（防止死锁）
  unique_value：防止误删别人的锁
  
释放锁：Lua脚本保证原子性
  if redis.call("get", KEYS[1]) == ARGV[1] then
      return redis.call("del", KEYS[1])
  end
```

**Q14: 如何设计一个短链接服务？**
```
1. 长链接 → 哈希 → 短码（6位62进制）
2. 存储：Redis（短码→长链接）+ MySQL（持久化）
3. 重定向：302（临时）vs 301（永久，可缓存）
4. 统计：点击数、来源、时间 → 异步写入
5. 过期：TTL自动清理
```

### 16.5 AI应用问题

**Q15: 如何优化LLM响应速度？**
```
1. 流式输出：SSE实现打字机效果
2. 缓存：相似问题直接返回缓存（语义缓存）
3. Prompt优化：减少不必要的上下文
4. 模型选择：简单任务用小模型
5. 并发控制：限制同时调用数，避免排队
```

**Q16: RAG系统如何评估效果？**
```
检索质量：Recall@K（召回率）、MRR（首个正确结果排名）
生成质量：Faithfulness（忠实度）、Relevance（相关性）
端到端：人工评分、A/B测试
我的项目用自研评测框架，包含这三个维度。
```

**Q17: 如何处理LLM的幻觉问题？**
```
1. RAG：注入事实知识
2. Prompt约束："不知道就说不知道"
3. 输出验证：Pydantic验证格式
4. LLM-as-Judge：用另一个LLM评估忠实度
5. 人工反馈：RLHF微调
```

### 16.6 DevOps问题

**Q18: Docker和虚拟机的区别？**
```
虚拟机：完整OS，资源隔离强，启动慢（分钟级），体积大（GB）
Docker：共享宿主机内核，轻量，启动快（秒级），体积小（MB）
适用：Docker适合微服务，VM适合需要完全隔离的场景
```

**Q19: 如何实现零停机部署？**
```
1. 滚动更新：逐个替换实例，保持服务可用
2. 蓝绿部署：新版本部署到绿环境，测试通过后切流量
3. 金丝雀发布：新版本先给5%流量，逐步放量
4. 健康检查：新实例ready后才接入流量
```

**Q20: 如何排查线上问题？**
```
1. 查日志：ELK搜索错误日志、request_id追踪
2. 看监控：Grafana查看QPS/延迟/错误率突变
3. 链路追踪：Jaeger查看请求经过哪些服务
4. 数据库：慢查询日志、锁等待
5. 回滚：快速恢复服务，再慢慢排查
```

---

## 十七、面试回答模板

### 17.1 项目介绍（1分钟版）

```
"我最近做的项目是一个医疗设备管理系统的AI层。

技术栈是FastAPI后端 + React前端 + PostgreSQL + Redis + ChromaDB。

核心功能是基于RAG的智能问答和故障诊断Agent。
RAG部分用了两阶段检索：BGE向量粗检索 + BGE-Reranker精排。
Agent部分用Tool Calling驱动4个专业Agent协作。

工程化方面做了完整的评测体系（Recall/MRR/NDCG/Faithfulness）、
三级异常处理、AI交互日志追踪，以及Docker三容器部署。

这个项目让我深刻理解了可追溯、可评测的LLM应用开发。"
```

### 17.2 技术选型回答框架

```
面试官："为什么选X而不是Y？"

回答结构：
1. 先讲X解决什么根本问题（第一性原理）
2. 列举候选方案（Y、Z等）
3. 排除标准（为什么不选Y）
4. 选X的决定性理由（具体，不是"综合考虑"）
5. 代价和升级路径（你知道它不完美）

示例：
"为什么选FastAPI而不是Flask？
 FastAPI基于ASGI异步，Flask基于WSGI同步。
 LLM API调用是IO密集型，平均2-5秒响应时间。
 同步框架会阻塞线程，10个并发就需要10个线程。
 异步框架可以在IO等待时处理其他请求，单进程支持数百并发。
 另外FastAPI的Pydantic集成对LLM输出验证特别友好。
 代价是调试稍复杂，但对AI应用来说异步是必选项。"
```

---

## 十八、总结：核心能力地图

```
前端能力：
  ✅ React/Vue基础 + Hooks
  ✅ 状态管理（Zustand/Pinia）
  ✅ SSR/CSR理解
  ✅ Tailwind CSS
  ✅ 流式响应处理（SSE）

后端能力：
  ✅ FastAPI异步编程
  ✅ RESTful API设计
  ✅ JWT认证
  ✅ SQLAlchemy ORM
  ✅ 异常处理与日志

数据库能力：
  ✅ PostgreSQL（JSONB/窗口函数）
  ✅ Redis（缓存/队列/限流）
  ✅ 缓存策略（穿透/击穿/雪崩）

中间件能力：
  ✅ Nginx（反向代理/负载均衡/限流）
  ✅ Celery + RabbitMQ（异步任务）
  ✅ 消息队列原理

DevOps能力：
  ✅ Docker（多阶段构建/Compose）
  ✅ CI/CD（GitHub Actions）
  ✅ 监控（Prometheus/Grafana/ELK）

AI应用能力：
  ✅ LLM集成（流式输出）
  ✅ RAG全链路
  ✅ Agent/Tool Calling
  ✅ 评测体系
```

**面试前最后检查：**
- [ ] 能画出完整的请求链路图
- [ ] 能解释每个技术选型的理由
- [ ] 能说出3个以上的性能优化点
- [ ] 能回答"如果重来会怎么做"
- [ ] 准备好2-3个反问问题

---

**文档完成时间：** 2026-02-28  
**适用岗位：** 清华水利AI平台研发（Python工程 + RAG/Agent + Web全栈）  
**配合文档：** 
- `清华岗位_完整冲刺手册_含Challenger拷问.md`（项目深挖+LLM应用）
- `Part5_Challenger拷问手册_独立版.md`（技术决策追问）
- `面试知识清单_LLM_Web系统开发.md`（LLM×Web基础）
- `LLM_训练与模型完整图景.md`（待创建，补LLM训练盲区）
