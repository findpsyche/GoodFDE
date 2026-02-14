# 20天填鸭式训练方案：大模型SFT工程师 & 代码能力评估工程师

> 目标岗位：字节跳动 CQC 模型运营 + 月之暗面 Kimi For Coding 系统工程师
> 训练周期：2月19日 - 3月10日（共20天）
> 训练强度：每天10-14小时，填鸭式高密度学习

---

## 总览：四大阶段

| 阶段 | 天数 | 主题 | 目标 |
|------|------|------|------|
| 第一阶段 | Day 1-4 | 计算机基础与编程能力 | 打牢底层认知，熟练Python/SQL |
| 第二阶段 | Day 5-8 | 数学基础与机器学习 | 掌握ML核心算法与数学直觉 |
| 第三阶段 | Day 9-14 | 深度学习与大模型核心 | 精通Transformer、预训练、SFT、RLHF |
| 第四阶段 | Day 15-20 | 大模型应用与岗位实战 | PE、RAG、Agent、评估体系、内容安全 |

---

## 第一阶段：计算机基础与编程能力（Day 1-4）

---

### Day 1（2月19日）：计算机体系结构与操作系统基础

**学习目标：** 建立计算机底层运行原理的完整认知，理解程序从代码到执行的全过程。

#### 核心知识点

**1. 冯诺依曼体系结构**
- 五大组件：运算器、控制器、存储器、输入设备、输出设备
- 存储程序概念：指令和数据以同等地位存储在内存中，按地址访问
- CPU工作流程：取指（Fetch）-> 译码（Decode）-> 执行（Execute）-> 写回（Write Back）
- 指令集架构（ISA）：CISC（x86）vs RISC（ARM），理解为什么现在AI芯片多用RISC思想
- 存储层次结构：寄存器（1ns）-> L1 Cache（2ns）-> L2 Cache（7ns）-> L3 Cache（20ns）-> 内存（100ns）-> SSD（100us）-> HDD（10ms）
- 为什么重要：大模型推理时的KV Cache、GPU显存层次都源于这个思想

**2. 操作系统核心概念**
- 进程与线程的区别：进程是资源分配的基本单位，线程是CPU调度的基本单位
- 进程状态转换：新建 -> 就绪 -> 运行 -> 阻塞 -> 终止
- 进程间通信（IPC）：管道、消息队列、共享内存、信号量、Socket
- 线程同步：互斥锁（Mutex）、读写锁、条件变量、信号量
- 死锁四个必要条件：互斥、占有且等待、不可抢占、循环等待
- 死锁预防与避免：银行家算法
- 协程（Coroutine）：用户态轻量级线程，Python的asyncio就是协程，大模型推理服务常用

**3. 内存管理**
- 虚拟内存：为每个进程提供独立的地址空间，通过页表映射到物理内存
- 分页机制：将虚拟地址空间划分为固定大小的页（通常4KB），物理内存划分为页框
- 页面置换算法：FIFO、LRU（最近最少使用）、LFU（最不经常使用）
- 内存分配：malloc/free的底层实现，内存池，内存碎片问题
- 为什么重要：训练大模型时GPU显存管理、模型参数加载、KV Cache的内存分配都依赖这些概念

**4. 文件系统与I/O**
- 文件系统层次：VFS -> 具体文件系统（ext4/NTFS）-> 块设备驱动 -> 硬件
- I/O模型五种：阻塞I/O、非阻塞I/O、I/O多路复用（select/poll/epoll）、信号驱动I/O、异步I/O
- 零拷贝技术：mmap、sendfile，减少用户态和内核态之间的数据拷贝
- 为什么重要：大模型训练中数据加载的I/O瓶颈，数据Pipeline的设计

**推荐资源：**
- CSAPP（深入理解计算机系统）第1、6、9章
- B站：王道考研操作系统（快速过核心章节）
- 小林coding图解系统

**练习任务：**
1. 画出冯诺依曼体系结构图，标注数据流和控制流
2. 用Python的multiprocessing和threading模块分别创建多进程和多线程程序，对比CPU密集型和IO密集型任务的性能差异
3. 实现一个LRU Cache（Python OrderedDict）
4. 用top/htop命令观察系统进程状态

**自测清单：**
- [ ] 能解释CPU执行一条指令的完整流程
- [ ] 能说清进程和线程的区别及使用场景
- [ ] 能解释虚拟内存的工作原理
- [ ] 能说出死锁的四个条件和预防方法
- [ ] 理解存储层次结构对大模型训练/推理的影响


---

### Day 2（2月20日）：计算机网络与通信基础

**学习目标：** 掌握网络通信的核心协议和原理，理解分布式系统通信基础。

#### 核心知识点

**1. OSI七层模型与TCP/IP四层模型**
- 应用层：HTTP、HTTPS、DNS、FTP、SMTP——用户直接接触的协议
- 传输层：TCP（可靠传输）、UDP（快速但不可靠）——端到端通信
- 网络层：IP协议、路由选择、ICMP（ping的原理）——主机到主机寻址
- 数据链路层：以太网帧、MAC地址、ARP协议——局域网内通信
- 物理层：比特流传输，网线、光纤、无线电波
- 记忆口诀：应传网数物（应用-传输-网络-数据链路-物理）

**2. TCP协议深入**
- 三次握手详解：
  - 第一次：客户端发送SYN=1, seq=x（我要建立连接）
  - 第二次：服务端发送SYN=1, ACK=1, seq=y, ack=x+1（同意，我也要建立连接）
  - 第三次：客户端发送ACK=1, seq=x+1, ack=y+1（确认）
  - 为什么不是两次：防止已失效的连接请求到达服务端导致资源浪费
- 四次挥手详解：
  - FIN -> ACK -> FIN -> ACK
  - TIME_WAIT状态：等待2MSL（最大报文段生存时间），确保最后一个ACK到达
- 可靠传输机制：序列号、确认号、超时重传、快速重传（3个重复ACK）
- 流量控制：滑动窗口机制，接收方通过窗口大小控制发送速率
- 拥塞控制四个阶段：慢启动（指数增长）-> 拥塞避免（线性增长）-> 快速重传 -> 快速恢复

**3. HTTP/HTTPS协议**
- HTTP请求方法语义：GET（获取）、POST（创建）、PUT（全量更新）、PATCH（部分更新）、DELETE（删除）
- HTTP状态码：200 OK、301永久重定向、302临时重定向、400错误请求、401未授权、403禁止、404未找到、500服务器错误、502网关错误、503服务不可用
- HTTP/1.1：持久连接（Keep-Alive）、管道化
- HTTP/2：多路复用（一个TCP连接上并行多个请求）、头部压缩、服务器推送
- HTTP/3：基于QUIC（UDP），解决TCP队头阻塞问题
- HTTPS = HTTP + TLS加密：
  - 对称加密（AES）：加解密用同一个密钥，速度快
  - 非对称加密（RSA/ECDSA）：公钥加密私钥解密，用于密钥交换
  - TLS握手：协商加密算法 -> 服务端发证书 -> 客户端验证证书 -> 生成对称密钥 -> 加密通信
- RESTful API设计：资源导向URL、无状态、统一接口、JSON数据格式

**4. DNS解析与CDN**
- DNS递归查询过程：浏览器缓存 -> 本地hosts文件 -> 本地DNS服务器 -> 根DNS -> 顶级域DNS -> 权威DNS
- DNS记录类型：A（域名->IPv4）、AAAA（域名->IPv6）、CNAME（别名）、MX（邮件）、TXT（文本）
- CDN原理：将内容缓存到离用户最近的边缘节点，通过DNS智能解析引导用户访问最近节点

**5. 分布式系统通信**
- RPC（远程过程调用）：像调用本地函数一样调用远程服务
  - gRPC：Google开源，基于HTTP/2和Protocol Buffers，高性能
  - Thrift：Facebook开源，支持多语言
- 消息队列：解耦生产者和消费者，异步处理
  - Kafka：高吞吐、持久化、分布式，适合日志和流数据
  - RabbitMQ：支持复杂路由，适合业务消息
- 负载均衡策略：轮询、加权轮询、最少连接、一致性哈希
- 为什么重要：大模型分布式训练（数据并行、模型并行、流水线并行）依赖高效节点间通信；模型推理服务需要负载均衡和消息队列

**推荐资源：**
- 《计算机网络：自顶向下方法》第1-3章
- B站：湖科大计算机网络
- 小林coding图解网络

**练习任务：**
1. 用Wireshark抓包观察TCP三次握手和HTTP请求
2. 用Python requests库发送GET/POST请求，观察请求头和响应头
3. 用Python socket模块实现TCP客户端-服务端通信
4. 用nslookup和traceroute理解DNS解析和路由

**自测清单：**
- [ ] 能画出TCP三次握手和四次挥手时序图
- [ ] 能解释HTTPS加密过程
- [ ] 能说清HTTP/1.1、HTTP/2、HTTP/3的区别
- [ ] 能解释DNS解析的完整过程
- [ ] 理解RPC和消息队列在分布式系统中的作用

---

### Day 3（2月21日）：数据结构与算法

**学习目标：** 掌握核心数据结构和算法思想，建立算法思维，为理解模型算法打基础。

#### 核心知识点

**1. 基础数据结构**
- 数组（Array）：连续内存存储，O(1)随机访问，O(n)插入删除。Python的list底层是动态数组，自动扩容（当前容量不够时分配1.125倍空间）
- 链表（LinkedList）：非连续内存，每个节点存储数据和指向下一节点的指针。O(1)头部插入删除，O(n)随机访问。单链表、双链表、循环链表
- 栈（Stack）：后进先出LIFO。应用：函数调用栈、表达式求值、括号匹配、DFS
- 队列（Queue）：先进先出FIFO。应用：BFS、任务调度、消息队列。变体：双端队列deque、优先队列
- 哈希表（HashMap）：
  - 原理：通过哈希函数将key映射到数组索引，平均O(1)查找/插入/删除
  - 哈希冲突解决：链地址法（每个桶挂链表）、开放寻址法（线性探测、二次探测）
  - 负载因子：元素数/桶数，超过阈值（如0.75）时扩容rehash
  - Python dict底层：开放寻址法，3.7+保证插入顺序
- 堆（Heap）：
  - 完全二叉树，用数组存储。父节点i的左子节点2i+1，右子节点2i+2
  - 最小堆：父节点 <= 子节点；最大堆：父节点 >= 子节点
  - 操作：插入O(logn)上浮、删除堆顶O(logn)下沉、建堆O(n)
  - 应用：优先队列、Top-K问题、堆排序、Dijkstra算法

**2. 树与图**
- 二叉树遍历：
  - 前序（根左右）、中序（左根右）、后序（左右根）：递归和迭代（用栈）实现
  - 层序遍历：用队列BFS实现
- 二叉搜索树（BST）：左子树所有节点 < 根 < 右子树所有节点，中序遍历有序
- 平衡二叉树：AVL树（严格平衡，旋转操作）、红黑树（近似平衡，Java TreeMap底层）
- B树/B+树：多路平衡搜索树，数据库索引的底层结构。B+树叶子节点用链表连接，范围查询高效
- 字典树（Trie）：前缀树，用于字符串检索、自动补全。NLP中的分词算法会用到
- 图的表示：邻接矩阵（稠密图）、邻接表（稀疏图）
- 图的遍历：BFS（层序，用队列）、DFS（深度，用栈/递归）
- 最短路径：Dijkstra（单源非负权）、Bellman-Ford（单源可负权）、Floyd（多源）
- 拓扑排序：DAG（有向无环图）的线性排序，应用：任务依赖、编译顺序
- 为什么重要：图神经网络GNN、知识图谱、Transformer的注意力机制可视为全连接图上的消息传递

**3. 排序算法**
- O(n^2)排序：冒泡排序、选择排序、插入排序（小数据量或近乎有序时高效）
- O(nlogn)排序：
  - 归并排序：分治思想，稳定排序，需要O(n)额外空间
  - 快速排序：分治思想，选pivot分区，平均O(nlogn)最坏O(n^2)，不稳定
  - 堆排序：利用堆结构，原地排序，不稳定
- 非比较排序：计数排序O(n+k)、桶排序O(n+k)、基数排序O(d*(n+k))
- 排序稳定性：相等元素的相对顺序是否保持。归并稳定，快排/堆排不稳定

**4. 核心算法思想**
- 分治法（Divide and Conquer）：将问题分解为规模更小的子问题，递归求解再合并
  - 例：归并排序、快排、大整数乘法
- 动态规划（Dynamic Programming）：
  - 核心：最优子结构 + 重叠子问题
  - 步骤：定义状态 -> 写状态转移方程 -> 确定边界条件 -> 确定计算顺序
  - 经典题：斐波那契、背包问题、最长公共子序列LCS、最长递增子序列LIS、编辑距离
  - 编辑距离特别重要：NLP中衡量两个字符串相似度的基础度量
- 贪心算法：每步选局部最优，期望全局最优。需要证明贪心选择性质
  - 例：活动选择、Huffman编码、Dijkstra算法
- 二分查找：有序数组O(logn)查找，关键在于边界条件处理（左闭右开vs左闭右闭）
- 回溯法：系统搜索所有可能解，不满足条件时剪枝回退
  - 例：N皇后、全排列、组合总和

**5. 复杂度分析**
- 时间复杂度：O(1) < O(logn) < O(n) < O(nlogn) < O(n^2) < O(2^n) < O(n!)
- 空间复杂度：算法额外使用的内存
- 均摊分析：动态数组扩容的均摊O(1)

**推荐资源：**
- LeetCode Hot 100题单
- 代码随想录（B站+网站）
- 《算法导论》选读

**练习任务：**
1. 手写快速排序、归并排序、二分查找
2. LeetCode 146 LRU Cache
3. LeetCode 72 编辑距离（DP）
4. LeetCode 215 数组中第K大元素（堆/快速选择）
5. LeetCode 200 岛屿数量（BFS/DFS）
6. 每天至少刷10道LeetCode

**自测清单：**
- [ ] 能手写快排并分析复杂度
- [ ] 能解释哈希表原理和冲突解决
- [ ] 能用DP解决编辑距离问题
- [ ] 能实现BFS和DFS
- [ ] 能分析常见算法的时间和空间复杂度


---

### Day 4（2月22日）：Python编程进阶与SQL

**学习目标：** 达到熟练使用Python进行数据处理和分析的水平，掌握SQL核心操作。

#### 核心知识点

**1. Python核心语法与高级特性**
- 数据类型深入：
  - list vs tuple：list可变，tuple不可变（可作为dict的key）
  - dict：3.7+保证插入顺序，底层哈希表，查找O(1)
  - set：无序不重复集合，底层也是哈希表，支持交并差集运算
- 列表推导式与生成器表达式：
  - 列表推导：[x**2 for x in range(10) if x % 2 == 0]，一次性生成全部结果
  - 生成器表达式：(x**2 for x in range(10))，惰性求值，节省内存
  - 字典推导：{k: v for k, v in zip(keys, values)}
- 生成器（Generator）：
  - yield关键字：函数执行到yield暂停，下次调用从暂停处继续
  - 应用场景：处理大文件、无限序列、数据Pipeline
  - send()方法：向生成器发送值
- 装饰器（Decorator）：
  - 本质：接受函数作为参数，返回新函数的高阶函数
  - @语法糖：@decorator等价于func = decorator(func)
  - functools.wraps：保留原函数的元信息
  - 带参数的装饰器：三层嵌套函数
  - 常用场景：日志记录、性能计时、缓存（@lru_cache）、权限验证
- 上下文管理器：
  - with语句自动管理资源的获取和释放
  - __enter__和__exit__方法
  - contextlib.contextmanager装饰器简化实现
- 面向对象编程：
  - 类与实例：__init__构造方法、self引用
  - 继承与多态：super()调用父类方法、方法重写
  - 魔术方法：__repr__（开发者表示）、__str__（用户表示）、__len__、__getitem__、__iter__、__next__
  - PyTorch的Dataset类就需要实现__len__和__getitem__
  - dataclass装饰器：简化数据类的定义
- 类型提示（Type Hints）：
  - 基本语法：def func(x: int, y: str = "default") -> bool:
  - 泛型：List[int]、Dict[str, Any]、Optional[str]、Union[int, str]
  - 提高代码可读性和IDE支持

**2. Python数据处理生态**
- NumPy核心操作：
  - 创建数组：np.array(), np.zeros(), np.ones(), np.arange(), np.linspace(), np.random.randn()
  - 数组属性：shape, dtype, ndim, size
  - 索引与切片：a[1:3, :], a[a > 0]（布尔索引），a[[0,2,4]]（花式索引）
  - 广播机制（Broadcasting）：不同形状数组运算时自动扩展维度的规则
    - 规则1：维度数不同时，在较小数组的shape前面补1
    - 规则2：某个维度大小为1时，沿该维度复制扩展
    - 例：(3,4) + (4,) -> (3,4) + (1,4) -> (3,4) + (3,4)
  - 矩阵运算：np.dot(), np.matmul(), @运算符, np.linalg.inv()（求逆）, np.linalg.eig()（特征值）
  - 聚合运算：np.sum(), np.mean(), np.std(), np.max(), np.argmax()（返回最大值索引）
  - reshape与维度操作：reshape(), flatten(), ravel(), squeeze(), expand_dims(), transpose()
  - 为什么重要：PyTorch的Tensor操作与NumPy几乎一致

- Pandas核心操作：
  - 数据结构：Series（一维带标签）、DataFrame（二维表格）
  - 读写数据：pd.read_csv(), pd.read_json(), pd.read_excel(), pd.read_sql(), df.to_csv()
  - 数据选择：
    - df["col"]或df.col：选择列
    - df.loc[行标签, 列标签]：基于标签选择
    - df.iloc[行索引, 列索引]：基于位置选择
    - df.query("col > 5 and col2 == 'A'")：条件查询
  - 数据清洗：
    - 缺失值：df.isnull(), df.dropna(), df.fillna(value/method)
    - 重复值：df.duplicated(), df.drop_duplicates()
    - 类型转换：df.astype(), pd.to_datetime(), pd.to_numeric()
    - 字符串处理：df["col"].str.lower(), .str.contains(), .str.replace(), .str.split()
  - 分组聚合：
    - df.groupby("col").agg({"val": ["mean", "sum", "count"]})
    - transform()：返回与原DataFrame同样大小的结果
    - apply()：对每个分组应用自定义函数
  - 合并操作：
    - pd.merge(df1, df2, on="key", how="left/right/inner/outer")
    - pd.concat([df1, df2], axis=0/1)
  - 透视表：pd.pivot_table(df, values="val", index="row", columns="col", aggfunc="mean")
  - 时间序列：pd.date_range(), df.resample(), df.rolling()

**3. Python并发编程**
- GIL（全局解释器锁）：
  - CPython中同一时刻只有一个线程执行Python字节码
  - CPU密集型任务：多线程无法利用多核，应使用多进程
  - IO密集型任务：多线程有效，因为IO等待时会释放GIL
- 多线程threading：适合IO密集型（网络请求、文件读写）
- 多进程multiprocessing：适合CPU密集型（数据处理、模型推理）
  - Pool进程池：pool.map(), pool.apply_async()
  - 进程间通信：Queue, Pipe, Manager
- 异步编程asyncio：
  - async def定义协程函数，await等待异步操作
  - asyncio.gather()并发执行多个协程
  - aiohttp：异步HTTP客户端
  - 适合高并发IO场景：大量API调用、爬虫
- concurrent.futures：统一的线程池/进程池接口
  - ThreadPoolExecutor, ProcessPoolExecutor
  - executor.submit(), executor.map()

**4. SQL核心操作**
- 基础查询：
  - SELECT col1, col2 FROM table WHERE condition ORDER BY col LIMIT n
  - DISTINCT去重、LIKE模糊匹配（%任意字符、_单个字符）
  - IN、BETWEEN、IS NULL
- 聚合与分组：
  - 聚合函数：COUNT(), SUM(), AVG(), MAX(), MIN()
  - GROUP BY分组 + HAVING过滤（WHERE过滤行，HAVING过滤组）
  - 执行顺序：FROM -> WHERE -> GROUP BY -> HAVING -> SELECT -> ORDER BY -> LIMIT
- 多表连接（JOIN）：
  - INNER JOIN：两表都有匹配的行
  - LEFT JOIN：左表全部 + 右表匹配的行（不匹配为NULL）
  - RIGHT JOIN：右表全部 + 左表匹配的行
  - FULL OUTER JOIN：两表全部
  - CROSS JOIN：笛卡尔积
  - 自连接：表与自身连接，用于层级关系或前后行比较
- 子查询：
  - WHERE子查询：WHERE col IN (SELECT ...)
  - FROM子查询：FROM (SELECT ...) AS subquery
  - EXISTS/NOT EXISTS：检查子查询是否返回结果
  - 相关子查询vs非相关子查询
- 窗口函数（重点，面试高频）：
  - 语法：函数名() OVER (PARTITION BY col ORDER BY col ROWS/RANGE ...)
  - 排名函数：
    - ROW_NUMBER()：连续排名1,2,3,4
    - RANK()：并列排名1,1,3,4（跳过）
    - DENSE_RANK()：并列排名1,1,2,3（不跳过）
    - NTILE(n)：分成n组
  - 偏移函数：
    - LAG(col, n)：取前n行的值
    - LEAD(col, n)：取后n行的值
    - FIRST_VALUE(), LAST_VALUE()
  - 聚合窗口函数：
    - SUM(col) OVER (ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)：累计求和
    - AVG(col) OVER (ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)：7日移动平均
  - 应用场景：用户留存分析、连续登录天数、排名问题、同比环比计算
- 索引与优化：
  - B+树索引原理：叶子节点有序链表，支持范围查询
  - 聚簇索引（主键索引，数据按索引顺序存储）vs 非聚簇索引（二级索引，存储指向数据的指针）
  - EXPLAIN执行计划：查看查询是全表扫描还是索引扫描
  - 索引优化原则：最左前缀匹配、覆盖索引、避免索引失效（函数操作、隐式类型转换）

**5. Python与SQL结合的数据分析工作流**
- 连接数据库：sqlalchemy.create_engine()
- 读取数据：pd.read_sql(query, engine)
- 典型工作流：SQL提取原始数据 -> Pandas清洗转换 -> NumPy数值计算 -> Matplotlib可视化
- 为什么重要：字节跳动JD明确要求"基于数据独立分析业务问题"

**推荐资源：**
- 《流畅的Python》第1-7章
- Pandas官方文档 10 Minutes to Pandas
- LeetCode SQL题库（至少做20道）
- Mode Analytics SQL教程

**练习任务：**
1. 用Pandas完成完整数据清洗流程：读取CSV -> 处理缺失值 -> 去重 -> 类型转换 -> 分组统计 -> 导出
2. 用NumPy实现矩阵乘法，与np.dot()对比验证
3. 写一个带参数的装饰器实现函数计时和日志功能
4. 实现一个生成器函数逐行读取大文件并统计词频
5. SQL练习：部门最高工资、连续登录天数、用窗口函数计算用户7日留存率
6. 用asyncio + aiohttp并发请求多个URL

**自测清单：**
- [ ] 能熟练使用生成器、装饰器、上下文管理器
- [ ] 能用Pandas完成数据清洗和分组聚合
- [ ] 能解释NumPy广播机制的规则
- [ ] 能写出包含JOIN、子查询、窗口函数的复杂SQL
- [ ] 能解释Python GIL的影响和多进程/多线程/协程的选择策略


---

## 第二阶段：数学基础与机器学习（Day 5-8）

---

### Day 5（2月23日）：数学基础——线性代数与概率统计

**学习目标：** 掌握机器学习和深度学习所需的数学基础，建立直觉理解而非死记公式。

#### 核心知识点

**1. 线性代数核心**
- 向量与向量空间：
  - 向量的几何意义：方向 + 大小
  - 向量加法：平行四边形法则
  - 数乘：缩放向量
  - 点积（内积）：a . b = |a||b|cos(theta)，衡量两个向量的相似度
    - 余弦相似度 = a . b / (|a| * |b|)，NLP中词向量相似度计算的基础
  - 线性无关：一组向量中没有任何一个可以被其他向量线性表示
  - 基（Basis）：线性无关的向量组，张成整个空间
- 矩阵运算：
  - 矩阵乘法：(m,n) x (n,p) = (m,p)，第i行第j列 = 第一个矩阵第i行与第二个矩阵第j列的点积
  - 矩阵乘法的意义：线性变换，将一个空间映射到另一个空间
  - 神经网络的本质：一系列矩阵乘法（线性变换）+ 非线性激活函数
  - 转置：行列互换，(A^T)^T = A，(AB)^T = B^T A^T
  - 逆矩阵：AA^(-1) = I，只有方阵且行列式不为0时存在
  - 行列式：det(A)，几何意义是线性变换对面积/体积的缩放因子
- 特征值与特征向量：
  - 定义：Av = lambda * v，v是特征向量，lambda是特征值
  - 意义：找到矩阵变换中只改变大小不改变方向的向量
  - 特征值分解：A = P * Lambda * P^(-1)
  - 应用：PCA降维、PageRank算法、协方差矩阵分析
- 奇异值分解（SVD）：
  - A = U * Sigma * V^T，适用于任意矩阵（不要求方阵）
  - U：左奇异向量，Sigma：奇异值对角矩阵，V：右奇异向量
  - 截断SVD：只保留前k个最大奇异值，实现降维和压缩
  - 应用：推荐系统、LSA（潜在语义分析）、图像压缩、降噪
- 范数（Norm）：
  - L1范数：|x| = sum(|xi|)，曼哈顿距离，产生稀疏解
  - L2范数：||x|| = sqrt(sum(xi^2))，欧氏距离，平滑解
  - Frobenius范数：矩阵所有元素平方和的平方根
  - 在ML中：L1正则化（Lasso，特征选择）、L2正则化（Ridge，防过拟合）

**2. 概率论与统计**
- 基本概念：
  - 样本空间、事件、概率公理（非负性、规范性、可加性）
  - 条件概率：P(A|B) = P(A and B) / P(B)
  - 独立性：P(A and B) = P(A) * P(B)
  - 全概率公式：P(B) = sum(P(B|Ai) * P(Ai))
- 贝叶斯定理（核心中的核心）：
  - P(A|B) = P(B|A) * P(A) / P(B)
  - 后验 = 似然 * 先验 / 证据
  - 贝叶斯思维：先有一个先验信念 -> 观察到新证据 -> 更新为后验信念
  - 这是机器学习的核心思想：从数据中学习就是不断更新我们对模型参数的信念
  - 朴素贝叶斯分类器：假设特征条件独立，直接应用贝叶斯定理
- 常见概率分布：
  - 伯努利分布：单次二值试验，P(X=1) = p，二分类的基础
  - 二项分布：n次独立伯努利试验中成功的次数
  - 正态分布（高斯分布）：N(mu, sigma^2)，自然界最常见的分布
    - 中心极限定理：大量独立随机变量之和趋近正态分布
    - 标准正态分布：N(0, 1)
  - 均匀分布：每个值等概率，随机初始化权重时常用
  - 多项式分布：伯努利分布的多类推广，softmax输出就是多项式分布
  - 指数分布、泊松分布：事件发生的时间间隔和频率
- 统计量：
  - 期望 E[X]：加权平均值
  - 方差 Var(X) = E[(X - mu)^2]：衡量数据离散程度
  - 标准差 sigma = sqrt(Var(X))
  - 协方差 Cov(X,Y) = E[(X-mu_x)(Y-mu_y)]：衡量两变量线性相关性
  - 相关系数 rho = Cov(X,Y) / (sigma_x * sigma_y)，取值[-1, 1]
- 参数估计：
  - 最大似然估计（MLE）：theta_MLE = argmax P(D|theta)
    - 对数似然：将连乘转化为求和，便于计算和优化
    - 交叉熵损失函数 = 负对数似然
  - 最大后验估计（MAP）：theta_MAP = argmax P(theta|D) = argmax P(D|theta) * P(theta)
    - MAP = MLE + 先验（等价于正则化）
    - L2正则化等价于高斯先验，L1正则化等价于拉普拉斯先验

**3. 微积分与优化**
- 导数与梯度：
  - 标量函数的导数：f'(x) = lim (f(x+h) - f(x)) / h
  - 偏导数：多变量函数对某一变量的导数
  - 梯度：所有偏导数组成的向量，指向函数增长最快的方向
  - 梯度的反方向是函数下降最快的方向——梯度下降的理论基础
- 链式法则（Chain Rule）：
  - d(f(g(x)))/dx = f'(g(x)) * g'(x)
  - 这是反向传播（Backpropagation）的数学基础
  - 深度网络中：损失对第一层参数的梯度 = 每层局部梯度的连乘
- 梯度下降法：
  - 更新规则：theta = theta - alpha * gradient(L(theta))
  - 批量梯度下降（BGD）：用全部数据计算梯度，稳定但慢
  - 随机梯度下降（SGD）：用单个样本，快但噪声大
  - 小批量梯度下降（Mini-batch SGD）：用一个batch，最常用
  - 学习率alpha的重要性：太大震荡发散，太小收敛慢
- 优化器进阶：
  - Momentum：引入动量，加速收敛，减少震荡
  - AdaGrad：自适应学习率，对稀疏特征友好
  - RMSProp：解决AdaGrad学习率单调递减问题
  - Adam：Momentum + RMSProp，最常用的优化器
  - AdamW：Adam + 权重衰减（解耦正则化），大模型训练标配
- 凸优化vs非凸优化：
  - 凸函数：任意两点连线在函数图像上方，局部最优 = 全局最优
  - 深度学习是非凸优化：存在大量局部最优和鞍点
  - 实践中：SGD的噪声反而帮助跳出局部最优

**4. 信息论基础**
- 信息量：I(x) = -log P(x)，小概率事件信息量大
- 熵（Entropy）：H(X) = -sum(P(x) * log P(x))
  - 衡量随机变量的不确定性/信息量
  - 均匀分布熵最大，确定事件熵为0
- 交叉熵（Cross Entropy）：H(P, Q) = -sum(P(x) * log Q(x))
  - 用分布Q编码分布P的平均编码长度
  - 分类任务的损失函数就是交叉熵：真实标签P是one-hot，模型输出Q是softmax概率
  - 二分类交叉熵：-[y*log(p) + (1-y)*log(1-p)]
- KL散度（KL Divergence）：KL(P||Q) = sum(P(x) * log(P(x)/Q(x)))
  - 衡量两个分布的"距离"（非对称：KL(P||Q) \!= KL(Q||P)）
  - KL散度 = 交叉熵 - 熵，即 KL(P||Q) = H(P,Q) - H(P)
  - 最小化交叉熵等价于最小化KL散度（因为H(P)是常数）
  - 应用：VAE的损失函数、RLHF中的KL惩罚项、知识蒸馏
- 互信息（Mutual Information）：I(X;Y) = H(X) - H(X|Y)
  - 衡量两个变量共享的信息量
  - 应用：特征选择、信息瓶颈理论

**推荐资源：**
- 3Blue1Brown《线性代数的本质》系列（强烈推荐，建立直觉）
- 3Blue1Brown《微积分的本质》系列
- 《统计学习方法》附录A
- 《深度学习》花书第2-4章
- StatQuest YouTube频道（统计和ML概念讲解）

**练习任务：**
1. 用NumPy实现矩阵特征值分解和SVD，验证A = U * Sigma * V^T
2. 手推逻辑回归的交叉熵损失函数对权重的梯度
3. 用Python从零实现梯度下降法求解线性回归（不用sklearn）
4. 计算两个概率分布的KL散度和交叉熵，验证KL = CE - H
5. 实现Adam优化器的更新步骤

**自测清单：**
- [ ] 能解释矩阵乘法的几何意义（线性变换）
- [ ] 能用贝叶斯定理解决实际问题
- [ ] 能推导梯度下降的更新公式
- [ ] 能解释交叉熵损失函数为什么适合分类任务
- [ ] 能说清L1/L2正则化与范数、贝叶斯先验的关系
- [ ] 能解释Adam优化器的工作原理

---

### Day 6（2月24日）：机器学习核心算法（上）

**学习目标：**
- 掌握线性回归和逻辑回归的数学推导与求解方法
- 理解决策树的构建过程（ID3/C4.5/CART）及剪枝策略
- 掌握集成学习的两大范式：Bagging（随机森林）与Boosting（GBDT/XGBoost/LightGBM）
- 熟练运用模型评估指标体系，理解偏差-方差权衡
- 建立从传统ML到大模型的损失函数与评估体系的关联认知

#### 核心知识点

**1. 线性回归**
- 模型假设：y = wᵀx + b，假设误差服从高斯分布
- 最小二乘法：最小化残差平方和 ∑(yᵢ - ŷᵢ)²
- 正规方程：w = (XᵀX)⁻¹Xᵀy，适用于特征维度不太高的场景
- 梯度下降求解：批量梯度下降BGD、随机梯度下降SGD、小批量梯度下降Mini-batch SGD
  - 学习率选择、收敛条件、学习率衰减策略
- 正则化：L1（Lasso，产生稀疏解）、L2（Ridge，权重衰减）、Elastic Net
- **为什么重要：** 线性回归是理解所有回归问题的基础，梯度下降是深度学习优化的核心方法，大模型训练本质上就是大规模梯度下降

**2. 逻辑回归**
- Sigmoid函数：σ(z) = 1/(1+e⁻ᶻ)，将线性输出映射到(0,1)概率
- 决策边界：wᵀx + b = 0 定义的超平面
- 交叉熵损失函数：L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]
  - 为什么不用MSE：非凸优化问题，梯度消失
- 多分类扩展：Softmax回归，softmax(zᵢ) = eᶻⁱ / ∑eᶻʲ
- 梯度推导：∂L/∂w = (ŷ - y)·x，形式简洁优美
- **为什么重要：** 逻辑回归的交叉熵损失和Softmax直接用于大模型的语言建模目标（预测下一个token的概率分布）

**3. 决策树**
- ID3算法：基于信息增益选择特征，H(D) = -∑pᵢlog₂pᵢ
- C4.5算法：使用信息增益比，解决ID3偏向多值特征的问题
- CART算法：分类用基尼系数 Gini(D) = 1 - ∑pᵢ²，回归用MSE
- 树的构建：递归二分、特征选择、停止条件
- 剪枝策略：预剪枝（限制深度/叶子数/最小样本数）、后剪枝（代价复杂度剪枝）
- **为什么重要：** 决策树是集成学习的基学习器，理解决策树才能理解XGBoost等面试高频算法

**4. 随机森林**
- Bagging思想：Bootstrap有放回采样，训练多棵树，投票/平均
- 特征随机选择：每次分裂随机选择√p个特征（分类）或p/3个特征（回归）
- OOB（Out-of-Bag）评估：约36.8%的样本未被采样，可作为验证集
- 优点：抗过拟合、可并行、能处理高维数据、可评估特征重要性
- **为什么重要：** Bagging思想在大模型中也有体现，如多个模型的集成投票、Self-Consistency等

**5. GBDT / XGBoost / LightGBM**
- Boosting思想：串行训练，每棵树拟合前一轮的残差（负梯度）
- GBDT：梯度提升决策树，用梯度近似残差
- XGBoost改进：
  - 二阶泰勒展开（利用Hessian信息）
  - 正则化项：Ω(f) = γT + ½λ∑wⱼ²（T为叶子数，w为叶子权重）
  - 列采样、缺失值处理、并行化特征分裂
- LightGBM改进：
  - 直方图加速：将连续特征离散化为bins
  - GOSS（基于梯度的单边采样）：保留大梯度样本
  - EFB（互斥特征捆绑）：减少特征数
  - Leaf-wise生长策略（vs XGBoost的Level-wise）
- **为什么重要：** XGBoost/LightGBM是工业界表格数据的首选模型，面试必考；Boosting的残差拟合思想与大模型的迭代优化有相通之处

**6. 模型评估指标**
- 混淆矩阵：TP/FP/TN/FN四个基本量
- 准确率 Accuracy = (TP+TN)/(TP+FP+TN+FN)，不平衡数据下会误导
- 精确率 Precision = TP/(TP+FP)，关注"预测为正的有多少是对的"
- 召回率 Recall = TP/(TP+FN)，关注"实际为正的有多少被找到"
- F1-Score = 2·P·R/(P+R)，精确率和召回率的调和平均
- AUC-ROC曲线：TPR vs FPR，面积越大越好，不受阈值影响
- PR曲线：Precision vs Recall，不平衡数据下比ROC更有参考价值
- **为什么重要：** 大模型评估（如代码生成的Pass@k、内容安全的精确率/召回率）都基于这些基础指标

**7. 偏差-方差权衡**
- 偏差（Bias）：模型预测值与真实值的差距，反映欠拟合程度
- 方差（Variance）：模型在不同训练集上预测的波动，反映过拟合程度
- 总误差 = 偏差² + 方差 + 不可约噪声
- 欠拟合：高偏差低方差 → 增加模型复杂度、增加特征
- 过拟合：低偏差高方差 → 正则化、增加数据、Dropout、早停
- **为什么重要：** 大模型的Scaling Laws本质上就是在探索偏差-方差权衡在超大规模下的表现

**8. 交叉验证**
- K-Fold交叉验证：数据分K份，轮流做验证集，取K次结果的平均
- 分层K-Fold（Stratified K-Fold）：保持每折中类别比例一致
- 留一法（LOOCV）：K=N，计算量大但偏差最小
- 时间序列交叉验证：只能用过去的数据预测未来，不能随机打乱
- **为什么重要：** 交叉验证是模型选择和超参数调优的标准方法，面试中经常考察

**推荐资源：**
- 周志华《机器学习》（西瓜书）第3-5章
- 李航《统计学习方法》第3-6章
- scikit-learn官方文档中的算法示例
- Kaggle Learn课程：Intro to Machine Learning
- 陈天奇XGBoost论文原文（理解二阶泰勒展开）

**练习任务：**
1. 用Python从零实现逻辑回归（不用sklearn），在二分类数据集上训练并画出决策边界
2. 手推XGBoost的目标函数，理解二阶泰勒展开的推导过程
3. 用sklearn分别训练随机森林和LightGBM，对比在同一数据集上的表现
4. 在不平衡数据集上计算Accuracy/Precision/Recall/F1/AUC，体会各指标的差异
5. 实现5-Fold交叉验证，对比与单次train/test split的结果稳定性

**自测清单：**
- [ ] 能推导逻辑回归的梯度更新公式
- [ ] 能解释信息增益和基尼系数的区别
- [ ] 能说清Bagging和Boosting的核心区别
- [ ] 能解释XGBoost相比GBDT的三个主要改进
- [ ] 能在不平衡数据场景下选择合适的评估指标
- [ ] 能解释偏差-方差权衡与模型复杂度的关系

---

### Day 7（2月25日）：机器学习核心算法（下）

**学习目标：**
- 掌握SVM的最大间隔原理与核函数技巧
- 理解KNN、朴素贝叶斯等经典分类算法的原理与适用场景
- 掌握K-Means、DBSCAN等聚类算法的核心思想
- 理解PCA、t-SNE等降维方法的数学原理
- 掌握EM算法的推导思路与GMM应用
- 建立完整的特征工程方法论

#### 核心知识点

**1. SVM支持向量机**
- 核心思想：找到最大间隔超平面，max 2/‖w‖，等价于 min ½‖w‖²
- 支持向量：距离超平面最近的样本点，决定了决策边界
- 对偶问题：引入拉格朗日乘子α，转化为对偶优化问题
  - 对偶形式只依赖样本间的内积 xᵢᵀxⱼ，这是核技巧的基础
- 核函数（Kernel Trick）：
  - 线性核：K(x,z) = xᵀz
  - 多项式核：K(x,z) = (xᵀz + c)ᵈ
  - RBF高斯核：K(x,z) = exp(-γ‖x-z‖²)，最常用，可映射到无穷维
  - 核函数条件：Mercer定理，核矩阵半正定
- 软间隔：引入松弛变量ξᵢ，允许部分样本违反间隔约束，C参数控制惩罚力度
- SMO算法：每次选两个变量优化，高效求解对偶问题
- **为什么重要：** SVM的核函数思想启发了后续很多"隐式高维映射"的方法；面试中SVM的推导是经典考题

**2. KNN（K近邻）**
- 原理：找到K个最近邻居，投票（分类）或平均（回归）
- 距离度量：欧氏距离、曼哈顿距离、闵可夫斯基距离、余弦相似度
- K值选择：K太小→过拟合（对噪声敏感），K太大→欠拟合（决策边界模糊）
- KD树加速：将空间递归划分，查询时间从O(n)降到O(logn)
- Ball Tree：适用于高维数据，比KD树更高效
- 缺点：计算量大、对特征尺度敏感（需标准化）、高维失效（维度灾难）
- **为什么重要：** KNN的思想在向量检索（RAG系统的核心）中直接应用，理解KNN有助于理解ANN近似最近邻搜索

**3. K-Means聚类**
- Lloyd算法：随机初始化→分配样本到最近中心→更新中心→迭代直到收敛
- 目标函数：最小化簇内平方和 J = ∑∑‖xᵢ - μₖ‖²
- K-Means++初始化：选择彼此远离的初始中心，避免局部最优
- 肘部法则（Elbow Method）：画J vs K曲线，找拐点确定最优K
- 轮廓系数（Silhouette Score）：衡量聚类质量，[-1,1]越大越好
- 缺点：需预设K、假设簇为球形、对异常值敏感
- **为什么重要：** K-Means在文本聚类、用户分群、数据预处理中广泛使用

**4. DBSCAN密度聚类**
- 核心概念：ε邻域、MinPts最小点数
- 点的分类：核心点（ε邻域内≥MinPts个点）、边界点、噪声点
- 算法流程：从核心点出发，密度可达的点归为同一簇
- 优点：不需预设簇数、能发现任意形状的簇、能识别噪声点
- 缺点：对ε和MinPts敏感、不适合密度差异大的数据
- HDBSCAN：层次化DBSCAN，自动确定密度阈值
- **为什么重要：** DBSCAN在异常检测、空间数据分析中常用

**5. PCA主成分分析**
- 目标：找到方差最大的投影方向，实现降维
- 数学推导：对协方差矩阵 C = (1/n)XᵀX 做特征值分解
- 主成分：特征值最大的前k个特征向量
- 方差解释比：λₖ/∑λᵢ，累计方差解释比达到95%时确定维度
- 步骤：中心化→计算协方差矩阵→特征值分解→选取前k个主成分→投影
- 白化（Whitening）：在PCA基础上除以√λ，使各维度方差为1
- **为什么重要：** PCA是最基础的降维方法，理解它有助于理解Embedding空间的几何结构

**6. t-SNE高维可视化**
- 核心思想：保持高维空间中点的局部邻域关系，映射到2D/3D
- 高维：用高斯分布衡量点对相似度 pᵢⱼ
- 低维：用t分布（自由度为1）衡量点对相似度 qᵢⱼ
- 目标：最小化 KL(P‖Q)，用梯度下降优化
- 困惑度（Perplexity）参数：控制关注局部vs全局结构，通常5-50
- 缺点：非确定性、不保持全局结构、计算量大O(n²)
- UMAP：更快、更好地保持全局结构的替代方案
- **为什么重要：** t-SNE/UMAP是可视化Embedding空间的标准工具，在分析大模型表示时常用

**7. 朴素贝叶斯**
- 贝叶斯定理：P(y|x) = P(x|y)P(y) / P(x)
- 条件独立假设：P(x₁,x₂,...,xₙ|y) = ∏P(xᵢ|y)，大幅简化计算
- 三种常见模型：
  - 高斯朴素贝叶斯：连续特征，假设P(xᵢ|y)服从高斯分布
  - 多项式朴素贝叶斯：适合文本分类（词频特征）
  - 伯努利朴素贝叶斯：适合二值特征
- 拉普拉斯平滑：P(xᵢ|y) = (count(xᵢ,y)+α) / (count(y)+α·|V|)，避免零概率
- **为什么重要：** 朴素贝叶斯是文本分类的经典baseline，理解贝叶斯思想对理解概率图模型和生成模型有帮助

**8. EM算法**
- 适用场景：含隐变量的概率模型参数估计
- E步（Expectation）：固定参数θ，计算隐变量的后验分布（期望）
- M步（Maximization）：固定隐变量期望，最大化似然函数更新θ
- 收敛性：每次迭代保证似然函数不减，但可能收敛到局部最优
- GMM高斯混合模型：
  - 假设数据由K个高斯分布混合生成
  - E步：计算每个样本属于各高斯分量的后验概率（责任度）
  - M步：更新各高斯分量的均值、协方差和混合权重
- **为什么重要：** EM算法的思想在很多地方出现，如VAE的变分推断、半监督学习等

**9. 特征工程**
- 特征选择方法：
  - 过滤法（Filter）：方差阈值、相关系数、互信息、卡方检验
  - 包裹法（Wrapper）：递归特征消除RFE、前向/后向选择
  - 嵌入法（Embedded）：L1正则化、树模型特征重要性
- 特征提取：PCA、LDA（线性判别分析）、自编码器
- 特征缩放：
  - 标准化（Z-score）：(x-μ)/σ，均值0方差1
  - 归一化（Min-Max）：(x-min)/(max-min)，缩放到[0,1]
  - 鲁棒缩放：用中位数和IQR，对异常值鲁棒
- 数据预处理Pipeline：缺失值处理→异常值处理→编码（One-Hot/Label/Target）→缩放→特征选择
- **为什么重要：** 特征工程决定了传统ML模型的上限；在大模型时代，数据预处理和特征设计仍然是SFT数据质量的关键

**推荐资源：**
- 周志华《机器学习》第6章（SVM）、第9章（聚类）、第10章（降维）
- 李航《统计学习方法》第7章（SVM）、第9章（EM算法）
- scikit-learn官方文档：Clustering、Decomposition、Feature Selection
- StatQuest YouTube频道（直观理解PCA、t-SNE）
- Distill.pub上的t-SNE交互式可视化文章

**练习任务：**
1. 用sklearn训练SVM（线性核和RBF核），可视化不同核函数的决策边界
2. 实现K-Means算法（不用sklearn），在Iris数据集上聚类并可视化
3. 对高维数据做PCA降维，画出方差解释比曲线，确定最优维度
4. 用t-SNE将MNIST手写数字降到2D并可视化，调整Perplexity观察效果
5. 构建一个完整的特征工程Pipeline：缺失值填充→编码→缩放→特征选择

**自测清单：**
- [ ] 能解释SVM最大间隔的数学含义和对偶问题的推导思路
- [ ] 能说清RBF核为什么能映射到无穷维空间
- [ ] 能解释K-Means的收敛性和K-Means++的改进
- [ ] 能区分PCA和t-SNE的适用场景
- [ ] 能推导EM算法在GMM上的E步和M步
- [ ] 能设计一个完整的特征工程流程

---

### Day 8（2月26日）：机器学习实战与sklearn

**学习目标：**
- 掌握sklearn完整工作流：从数据加载到模型部署的全流程
- 熟练使用Pipeline和ColumnTransformer构建可复现的ML流水线
- 掌握超参数调优方法（GridSearch/RandomSearch/Bayesian Optimization）
- 理解集成学习的高级方法（Stacking）和不平衡数据处理策略
- 掌握模型解释性工具（SHAP/LIME）的使用与解读
- 了解A/B测试的统计学基础

#### 核心知识点

**1. sklearn完整工作流**
- 数据加载：`load_*`内置数据集、`fetch_*`远程数据集、pandas读取CSV/SQL
- 数据探索：描述性统计、缺失值检查、类别分布、相关性矩阵
- 数据预处理：
  - 缺失值：`SimpleImputer`（均值/中位数/众数/常数填充）、`KNNImputer`
  - 编码：`LabelEncoder`（标签）、`OneHotEncoder`（独热）、`OrdinalEncoder`（有序）
  - 缩放：`StandardScaler`、`MinMaxScaler`、`RobustScaler`
- 数据划分：`train_test_split`（注意`stratify`参数保持类别比例）、`KFold`、`StratifiedKFold`
- 模型训练：`model.fit(X_train, y_train)`
- 模型评估：`model.score()`、`classification_report`、`confusion_matrix`、`roc_auc_score`
- **为什么重要：** sklearn是ML实验的标准工具，面试中经常要求手写完整的ML Pipeline

**2. Pipeline构建**
- `Pipeline`：将预处理和模型串联，`Pipeline([('scaler', StandardScaler()), ('clf', SVC())])`
- `make_pipeline`：简化写法，自动命名步骤
- `ColumnTransformer`：对不同类型的列应用不同的预处理
  - 数值列：缺失值填充 + 标准化
  - 类别列：缺失值填充 + 独热编码
  - 示例：`ColumnTransformer([('num', num_pipeline, num_cols), ('cat', cat_pipeline, cat_cols)])`
- `FunctionTransformer`：自定义转换函数
- Pipeline的好处：
  - 防止数据泄露（预处理参数只在训练集上fit）
  - 代码可复现、可序列化（`joblib.dump`）
  - 与交叉验证无缝集成
- **为什么重要：** Pipeline是工业级ML项目的标准实践，也是面试中展示工程能力的关键

**3. 超参数调优**
- `GridSearchCV`：穷举搜索所有参数组合
  - 优点：保证找到网格内最优解
  - 缺点：参数多时计算量爆炸（指数增长）
  - 示例：`GridSearchCV(model, param_grid, cv=5, scoring='f1')`
- `RandomizedSearchCV`：随机采样参数组合
  - 优点：相同计算预算下通常比GridSearch效果更好
  - 可指定`n_iter`控制搜索次数
  - 连续参数可用分布采样：`uniform`、`loguniform`
- Bayesian Optimization（Optuna/Hyperopt）：
  - 用概率代理模型（如高斯过程/TPE）建模目标函数
  - 根据采集函数（Expected Improvement）选择下一个评估点
  - Optuna示例：定义`objective`函数 → `study.optimize(objective, n_trials=100)`
  - 优点：样本效率高，适合评估代价大的模型
- 学习曲线：训练集大小 vs 训练/验证得分，诊断欠拟合/过拟合
- 验证曲线：超参数值 vs 训练/验证得分，找最优超参数范围
- **为什么重要：** 超参数调优直接影响模型性能，Optuna在大模型训练中也广泛用于调学习率、batch size等

**4. 模型选择策略**
- No Free Lunch定理：没有一个算法在所有问题上都最优
- 模型选择经验法则：
  - 数据量小 + 特征少：SVM、逻辑回归
  - 数据量大 + 表格数据：XGBoost/LightGBM
  - 需要解释性：决策树、逻辑回归、线性模型
  - 高维稀疏数据：朴素贝叶斯、线性SVM
- sklearn的`DummyClassifier`/`DummyRegressor`：建立baseline
- 多模型对比：用相同的交叉验证策略，统计检验（配对t检验/Wilcoxon）比较差异显著性
- **为什么重要：** 面试中经常问"给定场景你会选什么模型"，需要有系统的选择框架

**5. 集成学习深入**
- Bagging：并行训练多个基学习器，降低方差
  - `BaggingClassifier`/`BaggingRegressor`
  - 随机森林是Bagging + 特征随机选择的特例
- Boosting：串行训练，每轮关注前一轮的错误样本
  - AdaBoost：调整样本权重
  - GBDT/XGBoost/LightGBM：拟合残差（负梯度）
- Stacking（堆叠）：
  - 第一层：多个不同类型的基学习器
  - 第二层：用基学习器的预测作为特征，训练元学习器
  - `StackingClassifier(estimators=[...], final_estimator=LogisticRegression())`
  - 关键：第一层必须用交叉验证的预测，避免数据泄露
- Voting：简单投票（hard）或概率平均（soft）
- **为什么重要：** 集成学习是Kaggle竞赛的制胜法宝，Stacking思想在大模型中也有应用（如多模型融合）

**6. 不平衡数据处理**
- 问题：正负样本比例悬殊（如1:100），模型倾向预测多数类
- 数据层面：
  - 过采样SMOTE：在少数类样本间插值生成新样本
    - 变体：Borderline-SMOTE、ADASYN（自适应合成）
  - 欠采样：随机欠采样、Tomek Links、ENN（编辑最近邻）
  - 组合：SMOTE + Tomek Links / SMOTE + ENN
  - `imbalanced-learn`库：`SMOTE()`、`RandomUnderSampler()`
- 算法层面：
  - 代价敏感学习：`class_weight='balanced'`，自动调整类别权重
  - Focal Loss：γ参数降低易分类样本的损失权重，关注难样本
    - FL(pₜ) = -αₜ(1-pₜ)^γ · log(pₜ)
  - 阈值调整：不用默认0.5，根据PR曲线选择最优阈值
- 评估层面：用F1/AUC-PR而非Accuracy
- **为什么重要：** 不平衡数据在工业界极其常见（欺诈检测、内容安全分类），Focal Loss在大模型训练中也有应用

**7. 模型解释性**
- 特征重要性：
  - 树模型内置：`model.feature_importances_`（基于分裂增益或不纯度减少）
  - Permutation Importance：打乱某特征后性能下降程度
- SHAP（SHapley Additive exPlanations）：
  - 基于博弈论Shapley值，公平分配每个特征对预测的贡献
  - `shap.TreeExplainer`（树模型快速计算）、`shap.KernelExplainer`（通用）
  - 可视化：`summary_plot`（全局）、`force_plot`（单样本）、`dependence_plot`（交互）
- LIME（Local Interpretable Model-agnostic Explanations）：
  - 在预测点附近采样，用简单模型（线性模型）局部近似
  - 适用于任何黑盒模型
- Partial Dependence Plot（PDP）：展示特征与预测的边际关系
- **为什么重要：** 模型解释性在金融、医疗等领域是合规要求；在大模型评估中，理解模型为什么做出某个预测也越来越重要

**8. A/B测试基础**
- 假设检验框架：
  - 零假设H₀：两组无差异
  - 备择假设H₁：两组有差异
  - 显著性水平α（通常0.05）：犯第一类错误（拒绝真H₀）的概率
  - p值：在H₀为真时，观察到当前或更极端结果的概率
  - p < α → 拒绝H₀，认为差异显著
- 常用检验：
  - t检验：比较两组均值（连续变量）
  - 卡方检验：比较两组比例（分类变量）
  - Mann-Whitney U检验：非参数检验
- 置信区间：参数的区间估计，95%CI表示重复实验95%的CI会包含真值
- 样本量计算：效应量（Effect Size）、统计功效（Power，通常0.8）、显著性水平 → 最小样本量
- 多重比较问题：多次检验需要Bonferroni校正或FDR控制
- **为什么重要：** A/B测试是模型上线决策的标准方法，字节跳动等公司的模型运营岗位需要掌握

**推荐资源：**
- scikit-learn官方文档（User Guide + API Reference）
- Optuna官方文档和教程
- SHAP官方GitHub仓库和文档
- Kaggle竞赛Notebook（学习实战Pipeline）
- 《Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow》第2-7章
- imbalanced-learn官方文档

**练习任务：**
1. 用ColumnTransformer + Pipeline构建一个完整的ML流水线（包含数值和类别特征处理）
2. 用Optuna对LightGBM做超参数调优，对比与GridSearchCV的效率
3. 在不平衡数据集上对比：原始数据 vs SMOTE vs class_weight='balanced' 的效果
4. 用SHAP分析一个XGBoost模型，画出summary_plot和force_plot
5. 实现一个简单的A/B测试：生成模拟数据，计算p值和置信区间，判断差异是否显著

**自测清单：**
- [ ] 能用Pipeline + ColumnTransformer构建完整的预处理流水线
- [ ] 能解释GridSearch和RandomSearch的优缺点
- [ ] 能说清Stacking的工作原理和防止数据泄露的方法
- [ ] 能解释SMOTE的原理和适用场景
- [ ] 能用SHAP解释模型预测结果
- [ ] 能设计一个A/B测试并正确解读p值

---

## 第三阶段：深度学习与大模型（Day 9-14）

---

### Day 9（2月27日）：深度学习基础——神经网络与反向传播

**学习目标：**
- 理解感知机到多层感知机的演进，掌握万能近似定理
- 深入理解前向传播与反向传播的数学推导和计算图
- 掌握各种激活函数、损失函数、权重初始化方法的原理与选择依据
- 理解BatchNorm/LayerNorm/RMSNorm的区别与适用场景
- 掌握PyTorch基础操作，能搭建简单的神经网络

#### 核心知识点

**1. 感知机与多层感知机MLP**
- 感知机：单层线性分类器，y = sign(wᵀx + b)，只能解决线性可分问题
- XOR问题：单层感知机无法解决，需要多层网络
- 多层感知机（MLP）：输入层→隐藏层（一层或多层）→输出层
- 万能近似定理：一个足够宽的单隐藏层MLP可以近似任意连续函数
  - 但实践中深层网络比宽层网络更高效（参数效率更高）
- 全连接层：每个神经元与上一层所有神经元相连，参数量 = 输入维度 × 输出维度 + 偏置
- **为什么重要：** MLP是所有深度学习模型的基础组件，Transformer中的FFN层就是两层MLP

**2. 激活函数**
- Sigmoid：σ(x) = 1/(1+e⁻ˣ)，输出(0,1)
  - 缺点：梯度消失（|x|大时梯度趋近0）、输出非零中心、指数运算慢
- Tanh：tanh(x) = (eˣ-e⁻ˣ)/(eˣ+e⁻ˣ)，输出(-1,1)，零中心
  - 缺点：仍有梯度消失问题
- ReLU：f(x) = max(0,x)
  - 优点：计算简单、缓解梯度消失、稀疏激活
  - 缺点：Dead ReLU问题（负区间梯度为0，神经元永久死亡）
- Leaky ReLU：f(x) = max(αx, x)，α通常0.01，解决Dead ReLU
- GELU：x·Φ(x)，Φ是标准正态CDF，平滑版ReLU
  - BERT、GPT等大模型的默认激活函数
- Swish/SiLU：x·σ(x)，自门控激活函数
  - 在某些场景下优于ReLU和GELU
- **为什么重要：** 激活函数的选择直接影响训练稳定性和模型性能，GELU是当前大模型的标准选择

**3. 前向传播与反向传播**
- 前向传播：输入→逐层计算→输出→计算损失
  - z⁽ˡ⁾ = W⁽ˡ⁾a⁽ˡ⁻¹⁾ + b⁽ˡ⁾，a⁽ˡ⁾ = f(z⁽ˡ⁾)
- 反向传播（Backpropagation）：
  - 核心：链式法则，∂L/∂W⁽ˡ⁾ = ∂L/∂z⁽ˡ⁾ · ∂z⁽ˡ⁾/∂W⁽ˡ⁾
  - 从输出层向输入层逐层传播梯度
  - 时间复杂度：与前向传播同阶，一次前向+一次反向
- 计算图：将计算过程表示为有向无环图（DAG）
  - 每个节点是一个操作，边是数据流
  - 反向传播就是在计算图上反向遍历，应用链式法则
- 自动微分（Autograd）：
  - 前向模式：从输入到输出计算导数（适合输入少输出多）
  - 反向模式：从输出到输入计算导数（适合输入多输出少，即深度学习场景）
  - PyTorch的`autograd`就是反向模式自动微分
- **为什么重要：** 反向传播是深度学习训练的核心算法，理解它才能理解梯度消失/爆炸、残差连接等概念

**4. 损失函数**
- MSE（均方误差）：L = (1/n)∑(yᵢ-ŷᵢ)²，用于回归任务
  - 对异常值敏感，梯度与误差成正比
- 交叉熵损失：L = -∑yᵢlog(ŷᵢ)，用于分类任务
  - 二分类：L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]
  - 多分类：L = -∑yᵢlog(softmax(zᵢ))
  - 与KL散度的关系：CE(p,q) = H(p) + KL(p‖q)
- Focal Loss：FL = -αₜ(1-pₜ)^γ · log(pₜ)
  - γ>0时降低易分类样本的权重，关注难样本
  - 用于不平衡分类，在目标检测（RetinaNet）中提出
- 语言模型损失：本质是交叉熵，预测下一个token的概率分布
- **为什么重要：** 损失函数设计直接决定模型学什么，大模型的预训练损失（语言建模）和微调损失（SFT/DPO）都基于交叉熵

**5. 权重初始化**
- 为什么重要：初始化不当→梯度消失或爆炸→训练失败
- 全零初始化：所有神经元学到相同的东西（对称性问题），绝对不能用
- 随机初始化：打破对称性，但方差需要控制
- Xavier初始化（Glorot）：W ~ N(0, 2/(nᵢₙ+nₒᵤₜ))
  - 适用于Sigmoid/Tanh激活函数
  - 保持前向传播和反向传播的方差一致
- He初始化（Kaiming）：W ~ N(0, 2/nᵢₙ)
  - 适用于ReLU系列激活函数
  - 考虑了ReLU将一半值置零的特性
- **为什么重要：** 正确的初始化是训练成功的前提，大模型通常使用He初始化或特定的缩放初始化

**6. 归一化技术**
- BatchNorm（批归一化）：
  - 对每个特征维度，在batch维度上归一化：x̂ = (x-μ_B)/√(σ²_B+ε)
  - 可学习参数γ和β：y = γx̂ + β
  - 训练时用batch统计量，推理时用全局移动平均
  - 优点：加速收敛、允许更大学习率、有轻微正则化效果
  - 缺点：依赖batch size，batch小时统计量不稳定
- LayerNorm（层归一化）：
  - 对每个样本，在特征维度上归一化
  - 不依赖batch size，适合序列模型和Transformer
  - Transformer的标准归一化方法
- RMSNorm：
  - 简化版LayerNorm，只做缩放不做平移：x̂ = x / RMS(x)，RMS(x) = √(1/n·∑xᵢ²)
  - 计算更快，效果与LayerNorm相当
  - LLaMA等现代大模型的默认选择
- **为什么重要：** 归一化是深度网络训练的关键技术，LayerNorm/RMSNorm是Transformer的核心组件

**7. 正则化与训练稳定性**
- Dropout：训练时随机将一部分神经元输出置零（概率p，通常0.1-0.5）
  - 推理时不Dropout，但输出乘以(1-p)（或训练时除以(1-p)，即Inverted Dropout）
  - 等价于训练了指数级多个子网络的集成
- 梯度消失：深层网络中梯度逐层衰减，底层几乎无法更新
  - 原因：Sigmoid/Tanh的梯度<1，连乘后趋近0
  - 解决：ReLU、残差连接、归一化、合适的初始化
- 梯度爆炸：梯度逐层放大，参数更新过大
  - 解决：梯度裁剪（Gradient Clipping）、权重正则化、归一化
- 残差连接（Residual Connection）：y = F(x) + x
  - 梯度可以通过跳跃连接直接传到底层，缓解梯度消失
  - ResNet的核心创新，也是Transformer的标准组件
- **为什么重要：** 这些技术是训练深层网络的基础保障，大模型训练中梯度裁剪和残差连接是必备的

**8. 学习率调度**
- 固定学习率：简单但通常不是最优
- Step Decay：每隔N个epoch将学习率乘以γ（如0.1）
- Cosine Annealing：lr = lr_min + ½(lr_max - lr_min)(1 + cos(πt/T))
  - 平滑衰减，大模型训练的常用策略
- Warmup：训练初期从很小的学习率线性增长到目标学习率
  - 避免初期大梯度导致训练不稳定
  - 大模型标配：Warmup + Cosine Decay
- OneCycleLR：先升后降，一个周期内完成训练
- ReduceLROnPlateau：验证集指标不再提升时降低学习率
- **为什么重要：** 学习率调度直接影响训练效果和收敛速度，Warmup+Cosine是大模型训练的标准配置

**9. PyTorch基础**
- Tensor操作：创建、索引、切片、reshape、广播、GPU转移（`.to('cuda')`）
- autograd：`requires_grad=True`、`loss.backward()`、`optimizer.step()`
  - `torch.no_grad()`：推理时关闭梯度计算，节省内存
  - `detach()`：从计算图中分离tensor
- `nn.Module`：自定义模型的基类
  - `__init__`中定义层，`forward`中定义前向传播
  - `model.parameters()`获取所有可训练参数
- `DataLoader`：批量加载数据，支持shuffle、多进程加载
  - `Dataset`：自定义数据集，实现`__len__`和`__getitem__`
- 训练循环模板：`for epoch → for batch → forward → loss → backward → step → zero_grad`
- **为什么重要：** PyTorch是大模型训练和微调的主流框架，必须熟练掌握

**推荐资源：**
- 3Blue1Brown《神经网络》系列视频（直观理解反向传播）
- PyTorch官方教程（60分钟入门）
- 李沐《动手学深度学习》第3-5章
- CS231n课程笔记（斯坦福，CNN方向但基础部分通用）
- 论文：《Batch Normalization》、《Deep Residual Learning》

**练习任务：**
1. 用PyTorch从零实现一个两层MLP，在MNIST上训练并达到97%+准确率
2. 手动实现反向传播：对一个简单的两层网络，手算梯度并与PyTorch autograd对比验证
3. 对比不同激活函数（ReLU/GELU/Sigmoid）在同一网络上的训练曲线
4. 实现Warmup + Cosine学习率调度，画出学习率变化曲线
5. 对比有无BatchNorm/Dropout的训练效果

**自测清单：**
- [ ] 能手推一个两层网络的反向传播过程
- [ ] 能解释ReLU相比Sigmoid的优势以及Dead ReLU问题
- [ ] 能说清BatchNorm和LayerNorm的区别及各自适用场景
- [ ] 能解释残差连接为什么能缓解梯度消失
- [ ] 能写出PyTorch的标准训练循环
- [ ] 能解释Warmup的作用和Cosine学习率调度的公式

---

### Day 10（2月28日）：深度学习进阶——CNN与RNN

**学习目标：**
- 掌握CNN的核心操作（卷积、池化、感受野）及经典架构演进
- 理解RNN的序列建模能力及其梯度消失问题
- 掌握LSTM和GRU的门控机制设计
- 理解Seq2Seq和注意力机制的起源
- 建立从CNN/RNN到Transformer的演进认知

#### 核心知识点

**1. 卷积操作**
- 卷积核（Filter/Kernel）：一个小的权重矩阵，在输入上滑动做元素乘加
- 关键参数：
  - 卷积核大小：常用3×3、5×5、1×1
  - 步长（Stride）：滑动步幅，stride=2可替代池化做下采样
  - 填充（Padding）：same padding保持尺寸不变，valid padding不填充
  - 膨胀（Dilation）：扩大感受野而不增加参数
- 输出尺寸公式：O = (I - K + 2P) / S + 1
- 参数量计算：K×K×Cᵢₙ×Cₒᵤₜ + Cₒᵤₜ（偏置）
- 1×1卷积：不改变空间尺寸，用于通道数变换和跨通道信息融合
- 深度可分离卷积（Depthwise Separable）：先逐通道卷积再1×1卷积，大幅减少参数量（MobileNet）
- **为什么重要：** 卷积操作是理解局部特征提取的基础，虽然NLP已转向Transformer，但CNN在视觉领域仍是主流

**2. 池化与感受野**
- 最大池化（Max Pooling）：取窗口内最大值，保留最显著特征
- 平均池化（Average Pooling）：取窗口内平均值
- 全局平均池化（GAP）：将整个特征图平均为一个值，替代全连接层
- 感受野（Receptive Field）：输出特征图上一个像素对应输入图像的区域大小
  - 深层网络的感受野更大，能捕获更全局的信息
  - 3个3×3卷积的感受野 = 1个7×7卷积，但参数更少
- **为什么重要：** 感受野的概念类似于Transformer中注意力的"关注范围"，理解它有助于理解模型如何聚合信息

**3. 经典CNN架构演进**
- LeNet-5（1998）：最早的CNN，手写数字识别，2个卷积层+3个全连接层
- AlexNet（2012）：深度学习复兴的标志，ReLU+Dropout+数据增强+GPU训练
- VGGNet（2014）：全部使用3×3卷积，证明深度的重要性（VGG-16/19）
- GoogLeNet/Inception（2014）：Inception模块，多尺度卷积并行
- ResNet（2015）：残差连接，解决深层网络退化问题，可训练152层+
  - 核心公式：y = F(x) + x（恒等映射的跳跃连接）
  - Bottleneck结构：1×1降维→3×3卷积→1×1升维
- EfficientNet（2019）：复合缩放（深度×宽度×分辨率），NAS搜索最优架构
- **为什么重要：** ResNet的残差连接直接被Transformer采用，理解CNN架构演进有助于理解深度学习的设计哲学

**4. RNN序列建模**
- 基本RNN：hₜ = tanh(Wₕhₜ₋₁ + Wₓxₜ + b)
  - 隐藏状态hₜ携带历史信息，实现序列建模
  - 参数共享：所有时间步共享同一组权重
- 展开（Unrolling）：将RNN沿时间步展开，等价于一个很深的网络
- 梯度消失问题：
  - 反向传播时梯度需要经过多个时间步的连乘
  - 如果∂hₜ/∂hₜ₋₁的特征值<1，梯度指数衰减→长距离依赖学不到
  - 如果特征值>1，梯度指数增长→梯度爆炸（可用梯度裁剪缓解）
- 双向RNN（Bi-RNN）：同时从前向后和从后向前处理序列，拼接两个方向的隐藏状态
- **为什么重要：** RNN的梯度消失问题是LSTM/GRU和后来Transformer出现的直接动机

**5. LSTM（长短期记忆网络）**
- 核心设计：引入细胞状态Cₜ（信息高速公路）和三个门控机制
- 遗忘门（Forget Gate）：fₜ = σ(Wf·[hₜ₋₁, xₜ] + bf)
  - 决定丢弃细胞状态中的哪些信息
- 输入门（Input Gate）：iₜ = σ(Wi·[hₜ₋₁, xₜ] + bi)
  - 决定哪些新信息写入细胞状态
  - 候选值：C̃ₜ = tanh(Wc·[hₜ₋₁, xₜ] + bc)
- 细胞状态更新：Cₜ = fₜ ⊙ Cₜ₋₁ + iₜ ⊙ C̃ₜ
- 输出门（Output Gate）：oₜ = σ(Wo·[hₜ₋₁, xₜ] + bo)
  - hₜ = oₜ ⊙ tanh(Cₜ)
- 为什么能缓解梯度消失：细胞状态的更新是加法操作（而非乘法），梯度可以沿细胞状态直接流动
- **为什么重要：** LSTM的门控思想影响了后续很多架构设计，包括Transformer中的门控FFN

**6. GRU（门控循环单元）**
- 简化版LSTM：将遗忘门和输入门合并为更新门，去掉细胞状态
- 更新门：zₜ = σ(Wz·[hₜ₋₁, xₜ])
- 重置门：rₜ = σ(Wr·[hₜ₋₁, xₜ])
- 候选隐藏状态：h̃ₜ = tanh(W·[rₜ ⊙ hₜ₋₁, xₜ])
- 隐藏状态更新：hₜ = (1-zₜ) ⊙ hₜ₋₁ + zₜ ⊙ h̃ₜ
- 参数比LSTM少约25%，效果通常相当
- **为什么重要：** GRU展示了"简化不一定降低性能"的设计理念

**7. Seq2Seq与编码器-解码器架构**
- 编码器（Encoder）：将输入序列编码为固定长度的上下文向量c
  - 通常取最后一个时间步的隐藏状态：c = hₜ
- 解码器（Decoder）：以上下文向量c为初始状态，自回归生成输出序列
  - 每一步的输入是上一步的输出（Teacher Forcing：训练时用真实标签）
- 信息瓶颈问题：所有输入信息压缩到一个固定长度向量c中，长序列信息丢失严重
- 应用：机器翻译、文本摘要、对话生成
- **为什么重要：** Seq2Seq是Transformer编码器-解码器架构的前身，理解它的局限性才能理解注意力机制的动机

**8. 注意力机制的起源（Bahdanau Attention）**
- 动机：解决Seq2Seq的信息瓶颈问题，让解码器在每一步"关注"编码器的不同位置
- 计算过程：
  - 对齐分数：eᵢⱼ = a(sᵢ₋₁, hⱼ)，a是一个小型神经网络
  - 注意力权重：αᵢⱼ = softmax(eᵢⱼ)
  - 上下文向量：cᵢ = ∑αᵢⱼhⱼ（编码器隐藏状态的加权和）
  - 解码器输入：将cᵢ与当前输入拼接
- Luong Attention：简化版，用点积或双线性形式计算对齐分数
- 注意力的可视化：注意力权重矩阵可以展示输入输出的对齐关系
- **为什么重要：** Bahdanau Attention是Transformer自注意力机制的直接前身，理解它是理解Transformer的关键

**9. 为什么CNN和RNN被Transformer取代**
- RNN的问题：
  - 顺序计算，无法并行化→训练慢
  - 长距离依赖仍然困难（即使LSTM也有限制，通常有效距离<100）
  - 梯度消失/爆炸问题未完全解决
- CNN用于NLP的问题：
  - 感受野有限，需要很多层才能捕获长距离依赖
  - 不够灵活，固定的卷积核大小
- Transformer的优势：
  - 自注意力机制：任意两个位置直接交互，O(1)的路径长度
  - 完全并行化：所有位置同时计算
  - 灵活的注意力模式：可以学习任意的依赖关系
- **为什么重要：** 理解这个演进过程是面试中的经典问题，也是理解大模型架构选择的基础

**推荐资源：**
- CS231n课程（斯坦福，CNN部分）
- CS224n课程（斯坦福，RNN/LSTM/Attention部分）
- colah's blog：《Understanding LSTM Networks》（经典图解LSTM）
- 李沐《动手学深度学习》第6-10章
- 论文：《Deep Residual Learning for Image Recognition》（ResNet）
- 论文：《Neural Machine Translation by Jointly Learning to Align and Translate》（Bahdanau Attention）

**练习任务：**
1. 用PyTorch实现一个简单的CNN（2个卷积层+池化+全连接），在CIFAR-10上训练
2. 手算一个3×3卷积在5×5输入上的输出（stride=1, padding=0），验证输出尺寸公式
3. 用PyTorch实现LSTM，在文本分类任务上训练（如IMDB情感分析）
4. 实现Bahdanau Attention机制，可视化注意力权重矩阵
5. 对比RNN/LSTM/GRU在同一序列任务上的训练速度和效果

**自测清单：**
- [ ] 能计算卷积层的输出尺寸和参数量
- [ ] 能解释ResNet残差连接的设计动机和数学原理
- [ ] 能画出LSTM的结构图并解释三个门的作用
- [ ] 能说清RNN梯度消失的原因和LSTM如何缓解
- [ ] 能解释Seq2Seq的信息瓶颈问题和注意力机制的解决方案
- [ ] 能说清Transformer相比RNN的三个核心优势

---

### Day 11（3月1日）：Transformer架构精讲

**学习目标：**
- 完整掌握自注意力机制的数学推导（Q/K/V、缩放点积、多头注意力）
- 理解Transformer编码器和解码器的完整架构
- 掌握位置编码的多种方案（正弦/余弦、RoPE、ALiBi）
- 理解高效注意力的优化方法（Flash Attention、稀疏注意力）
- 掌握KV Cache、GQA/MQA等推理优化技术

#### 核心知识点

**1. 自注意力机制（Self-Attention）**
- 核心思想：序列中每个位置都可以直接关注其他所有位置，计算加权表示
- Q/K/V矩阵：
  - Query（查询）：Q = XWQ，"我在找什么信息"
  - Key（键）：K = XWK，"我有什么信息可以提供"
  - Value（值）：V = XWV，"我实际提供的信息内容"
  - WQ、WK、WV是可学习的投影矩阵，维度为d_model × d_k
- 缩放点积注意力：Attention(Q,K,V) = softmax(QKᵀ/√d_k)V
  - 为什么要缩放√d_k：当d_k较大时，点积值的方差为d_k，softmax会进入饱和区（梯度极小），除以√d_k使方差回到1
  - softmax将注意力分数归一化为概率分布
- 注意力矩阵：n×n的矩阵，(i,j)表示位置i对位置j的关注程度
- 自注意力 vs 交叉注意力：
  - 自注意力：Q/K/V来自同一序列
  - 交叉注意力：Q来自解码器，K/V来自编码器（用于编码器-解码器架构）
- **为什么重要：** 自注意力是Transformer和所有大模型的核心计算单元，面试必考的推导题

**2. 多头注意力（Multi-Head Attention）**
- 动机：单个注意力头只能关注一种模式，多头可以同时关注不同类型的关系
- 计算过程：
  - 将Q/K/V分别投影到h个不同的子空间：Qᵢ = QWᵢQ，Kᵢ = KWᵢK，Vᵢ = VWᵢV
  - 每个头独立计算注意力：headᵢ = Attention(Qᵢ, Kᵢ, Vᵢ)
  - 拼接所有头的输出：MultiHead = Concat(head₁,...,headₕ)WO
- 参数量：每个头的维度 d_k = d_model / h，总参数量不变
- 不同头学到的模式：语法关系、语义关系、位置关系、共指关系等
- **为什么重要：** 多头注意力让模型能同时捕获多种不同的依赖关系，是Transformer表达能力的关键

**3. Transformer完整架构**
- 编码器栈（Encoder Stack）：
  - N个相同的编码器层堆叠（原始论文N=6）
  - 每层：多头自注意力 → Add & Norm → FFN → Add & Norm
  - 输入可以看到所有位置（双向注意力）
- 解码器栈（Decoder Stack）：
  - N个相同的解码器层堆叠
  - 每层：掩码多头自注意力 → Add & Norm → 交叉注意力 → Add & Norm → FFN → Add & Norm
  - 掩码自注意力：因果掩码（Causal Mask），位置i只能看到≤i的位置，防止信息泄露
- FFN（前馈网络）：两层MLP，FFN(x) = max(0, xW₁+b₁)W₂+b₂
  - 中间维度通常是d_model的4倍（如d_model=768，FFN中间维度=3072）
  - 现代变体：用GELU替代ReLU，用GLU门控线性单元
- Add & Norm：残差连接 + 层归一化
- **为什么重要：** 完整理解Transformer架构是理解所有大模型的前提

**4. 位置编码**
- 为什么需要：自注意力是置换不变的（permutation invariant），不包含位置信息
- 正弦/余弦位置编码（原始Transformer）：
  - PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
  - PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
  - 优点：可以外推到训练时未见过的长度（理论上）
  - 缺点：实际外推能力有限
- RoPE（旋转位置编码）：
  - 将位置信息编码为旋转矩阵，作用在Q和K上
  - 核心性质：内积只依赖相对位置 qₘᵀkₙ = f(xₘ, xₙ, m-n)
  - LLaMA、Qwen等主流大模型的标准选择
  - 支持通过NTK-aware缩放等方法扩展上下文长度
- ALiBi（Attention with Linear Biases）：
  - 不修改Q/K，直接在注意力分数上加线性偏置：score(i,j) = qᵢᵀkⱼ - m·|i-j|
  - m是每个头不同的斜率，距离越远惩罚越大
  - 优点：天然支持长度外推，实现简单
  - BLOOM等模型使用
- **为什么重要：** 位置编码的选择直接影响模型的长文本处理能力，RoPE是当前面试热点

**5. Pre-Norm vs Post-Norm**
- Post-Norm（原始Transformer）：x → SubLayer → Add → LayerNorm
  - 训练不稳定，需要Warmup
  - 理论上表达能力更强
- Pre-Norm（现代主流）：x → LayerNorm → SubLayer → Add
  - 训练更稳定，可以用更大的学习率
  - 梯度直接通过残差连接流动，不经过LayerNorm
  - GPT系列、LLaMA等大模型的标准选择
- **为什么重要：** Pre-Norm是大模型训练稳定性的关键设计选择

**6. MoE混合专家模型**
- 动机：增加模型容量而不成比例增加计算量
- 架构：将FFN替换为多个"专家"FFN + 一个门控网络（Router）
- 门控网络：G(x) = softmax(TopK(x·Wg))，选择Top-K个专家
  - 通常K=1或K=2，即每个token只激活1-2个专家
- 负载均衡：辅助损失函数确保各专家被均匀使用，避免"专家坍塌"
- 代表模型：Mixtral 8x7B（8个专家，每次激活2个，实际计算量≈14B）
- 优点：参数量大但推理计算量可控
- 缺点：通信开销大、负载均衡困难、训练不稳定
- **为什么重要：** MoE是当前大模型扩展的重要方向，GPT-4据传也使用了MoE架构

**7. 注意力复杂度与高效注意力**
- 标准注意力复杂度：O(n²·d)，n是序列长度，d是维度
  - 内存：需要存储n×n的注意力矩阵
  - 当n=128K时，注意力矩阵需要约64GB内存（FP32）
- Flash Attention：
  - 核心思想：利用GPU内存层次结构（SRAM vs HBM），分块计算注意力
  - 不显式存储n×n注意力矩阵，而是分块计算并在SRAM中完成softmax
  - IO复杂度从O(n²)降到O(n²d/M)，M是SRAM大小
  - Flash Attention 2/3：进一步优化并行度和计算效率
  - 已成为大模型训练和推理的标准组件
- Sparse Attention：只计算部分位置对的注意力
  - 局部注意力：每个位置只关注周围窗口内的位置
  - 全局注意力：部分特殊token（如[CLS]）关注所有位置
  - Longformer、BigBird等模型使用
- Sliding Window Attention：
  - 每个位置只关注固定窗口大小W内的位置
  - 复杂度O(n·W)，线性于序列长度
  - Mistral模型使用，配合滚动KV Cache
- **为什么重要：** 高效注意力是支撑大模型处理长文本的关键技术，Flash Attention是面试热点

**8. KV Cache与推理优化**
- KV Cache：
  - 自回归生成时，每生成一个新token需要计算注意力
  - 之前token的K和V不会变化，可以缓存复用
  - 只需计算新token的Q，与缓存的K/V做注意力
  - 将每步的计算量从O(n²)降到O(n)
  - 内存占用：2 × n_layers × n_heads × seq_len × d_head × dtype_size
- GQA（Grouped Query Attention）：
  - 多个Q头共享一组K/V头
  - 如8个Q头共享1组K/V → KV Cache减少8倍
  - LLaMA 2 70B、Mistral等模型使用
- MQA（Multi-Query Attention）：
  - 所有Q头共享同一组K/V
  - KV Cache最小，但可能损失一些表达能力
  - GQA是MQA和MHA的折中方案
- **为什么重要：** KV Cache是大模型推理的核心优化，GQA/MQA直接影响推理速度和内存占用，面试必考

**推荐资源：**
- 论文：《Attention Is All You Need》（原始Transformer论文，必读）
- Jay Alammar博客：《The Illustrated Transformer》（最佳图解）
- 论文：《RoFormer: Enhanced Transformer with Rotary Position Embedding》（RoPE）
- 论文：《FlashAttention: Fast and Memory-Efficient Exact Attention》
- 李沐精读Transformer论文视频
- Andrej Karpathy：《Let's build GPT from scratch》视频

**练习任务：**
1. 用PyTorch从零实现缩放点积注意力和多头注意力
2. 实现正弦/余弦位置编码，可视化不同位置和维度的编码值
3. 实现因果掩码（Causal Mask），验证解码器的自回归特性
4. 计算一个7B参数模型在不同序列长度下的KV Cache内存占用
5. 阅读Transformer原始论文，画出完整的架构图并标注每个组件的维度变化

**自测清单：**
- [ ] 能完整推导缩放点积注意力的公式并解释为什么要除以√d_k
- [ ] 能解释多头注意力的动机和计算过程
- [ ] 能说清Pre-Norm和Post-Norm的区别
- [ ] 能解释RoPE的核心思想和相比正弦位置编码的优势
- [ ] 能解释Flash Attention的核心优化思路
- [ ] 能计算KV Cache的内存占用并解释GQA的优化原理

---

### Day 12（3月2日）：预训练语言模型

**学习目标：**
- 理解语言模型的发展脉络：从N-gram到神经语言模型再到预训练大模型
- 掌握词向量（Word2Vec/GloVe）的训练原理
- 深入理解BERT和GPT两大预训练范式的区别
- 理解Scaling Laws和预训练数据工程的重要性
- 建立从预训练到微调的完整认知框架

#### 核心知识点

**1. 语言模型基础**
- 语言模型的定义：对序列的概率建模 P(w₁,w₂,...,wₙ) = ∏P(wₜ|w₁,...,wₜ₋₁)
- N-gram语言模型：
  - 马尔可夫假设：P(wₜ|w₁,...,wₜ₋₁) ≈ P(wₜ|wₜ₋ₙ₊₁,...,wₜ₋₁)
  - Bigram：P(wₜ|wₜ₋₁)，Trigram：P(wₜ|wₜ₋₂,wₜ₋₁)
  - 问题：数据稀疏、无法捕获长距离依赖、存储量随N指数增长
  - 平滑技术：Laplace平滑、Kneser-Ney平滑
- 神经语言模型（Bengio 2003）：用神经网络学习词的分布式表示
  - 输入：前n-1个词的词向量拼接
  - 输出：softmax预测下一个词的概率分布
  - 突破：词向量可以捕获语义相似性
- 困惑度（Perplexity）：PPL = exp(-1/N · ∑log P(wₜ|context))
  - 直觉：模型在每个位置平均需要从多少个词中选择
  - PPL越低越好，等价于交叉熵损失的指数
  - 大模型评估的核心指标之一
- **为什么重要：** 语言模型是所有大模型的理论基础，困惑度是评估预训练质量的标准指标

**2. Word2Vec词向量**
- 核心思想：用词的上下文来表示词的语义（分布式假设）
- CBOW（Continuous Bag of Words）：用上下文词预测中心词
  - 输入：窗口内的上下文词向量平均
  - 输出：softmax预测中心词
- Skip-gram：用中心词预测上下文词
  - 输入：中心词向量
  - 输出：预测窗口内每个上下文词
  - 通常效果优于CBOW，尤其在小数据集上
- 训练技巧：
  - 负采样（Negative Sampling）：将多分类简化为二分类，大幅加速训练
  - 层次Softmax：用Huffman树替代全量softmax
  - 子采样（Subsampling）：降低高频词的采样概率
- 词向量的性质：king - man + woman ≈ queen（语义算术）
- **为什么重要：** Word2Vec开创了预训练词表示的范式，其思想直接影响了后续所有预训练模型

**3. GloVe与ELMo**
- GloVe（Global Vectors）：
  - 结合全局统计信息（共现矩阵）和局部上下文窗口
  - 目标函数：wᵢᵀw̃ⱼ + bᵢ + b̃ⱼ = log(Xᵢⱼ)，X是共现矩阵
  - 加权函数f(Xᵢⱼ)：降低高频共现对的权重
  - 与Word2Vec效果相当，训练更高效
- ELMo（Embeddings from Language Models）：
  - 用双向LSTM语言模型生成上下文相关的词表示
  - 同一个词在不同上下文中有不同的向量（解决一词多义）
  - 使用方式：将ELMo向量与任务特定的词向量拼接
  - 局限：基于LSTM，双向信息融合不够充分（只是拼接）
- 从静态词向量到动态词向量的演进：Word2Vec/GloVe（静态）→ ELMo（动态）→ BERT（深度双向）
- **为什么重要：** ELMo是预训练+微调范式的先驱，理解它有助于理解BERT的创新之处

**4. BERT：双向预训练**
- 核心创新：用Transformer编码器做深度双向预训练
- 预训练任务：
  - MLM（Masked Language Model）：随机遮盖15%的token，预测被遮盖的token
    - 80%替换为[MASK]、10%替换为随机token、10%保持不变
    - 为什么这样设计：缓解预训练和微调的不一致（微调时没有[MASK]）
  - NSP（Next Sentence Prediction）：预测两个句子是否相邻
    - 后来被证明效果有限，RoBERTa去掉了NSP
- 微调范式：预训练好的BERT + 任务特定的分类头，在下游任务上微调
  - 分类：[CLS] token的表示 → 线性层 → softmax
  - 序列标注：每个token的表示 → 线性层
  - 问答：预测答案的起始和结束位置
- BERT变体：
  - RoBERTa：更多数据、更大batch、去掉NSP、动态masking → 显著提升
  - ALBERT：参数共享（跨层共享）+ 因式分解嵌入 → 参数量大幅减少
  - DeBERTa：解耦注意力（内容和位置分开计算）+ 增强的掩码解码器
- **为什么重要：** BERT开创了"预训练+微调"范式，是NLU任务的里程碑；理解MLM对理解大模型的训练目标设计有帮助

**5. GPT系列：自回归语言模型**
- GPT-1（2018）：
  - 用Transformer解码器做自回归语言建模
  - 预训练：预测下一个token，P(wₜ|w₁,...,wₜ₋₁)
  - 微调：在下游任务上微调（与BERT类似）
- GPT-2（2019）：
  - 更大的模型（1.5B参数）和更多数据（WebText 40GB）
  - 发现：足够大的语言模型可以做Zero-shot任务（不需要微调）
  - "Language Models are Unsupervised Multitask Learners"
- GPT-3（2020）：
  - 175B参数，训练数据300B tokens
  - In-Context Learning涌现：通过Few-shot提示完成任务，无需梯度更新
  - 三种使用方式：Zero-shot、One-shot、Few-shot
  - 涌现能力（Emergent Abilities）：模型规模超过某个阈值后突然出现的能力
- BERT vs GPT的核心区别：
  - BERT：编码器，双向注意力，适合理解任务（NLU）
  - GPT：解码器，因果注意力，适合生成任务（NLG）
  - 趋势：Decoder-Only架构（GPT路线）成为大模型主流
- **为什么重要：** GPT系列定义了当前大模型的主流架构和训练范式，In-Context Learning是大模型最重要的涌现能力

**6. T5：统一文本到文本框架**
- 核心思想：将所有NLP任务统一为"文本到文本"格式
  - 分类："classify: This movie is great" → "positive"
  - 翻译："translate English to French: Hello" → "Bonjour"
  - 摘要："summarize: [长文本]" → "[摘要]"
- 架构：标准的Encoder-Decoder Transformer
- 预训练：Span Corruption（随机遮盖连续的span，预测被遮盖的内容）
- Encoder-Decoder vs Decoder-Only：
  - Encoder-Decoder：输入和输出有明确分离，适合翻译、摘要等
  - Decoder-Only：更简单、更容易扩展、In-Context Learning能力更强
  - 趋势：Decoder-Only成为主流（GPT-4、LLaMA、Qwen等）
- **为什么重要：** T5展示了统一框架的威力，其"文本到文本"思想影响了后续的指令微调设计

**7. 预训练数据工程**
- 数据来源：Common Crawl、Wikipedia、Books、Code（GitHub）、学术论文等
- 数据质量 > 数据数量：
  - 高质量数据（书籍、维基百科）的价值远高于低质量网页
  - 数据质量直接影响模型能力的上限
- 数据配比：不同来源数据的混合比例
  - 代码数据提升推理能力
  - 多语言数据提升跨语言能力
  - 需要实验确定最优配比
- 数据去重：
  - 精确去重：哈希匹配
  - 近似去重：MinHash + LSH（局部敏感哈希）
  - 去重可以显著提升模型质量，减少记忆化
- 数据清洗Pipeline：
  - 语言识别 → 质量过滤（困惑度过滤、启发式规则）→ 去重 → 敏感信息过滤 → 分词
- **为什么重要：** 数据工程是大模型训练中最重要但最容易被忽视的环节，字节跳动等公司的数据运营岗位直接涉及

**8. Scaling Laws**
- OpenAI Scaling Laws（Kaplan et al., 2020）：
  - 模型性能（损失）与模型参数量N、数据量D、计算量C呈幂律关系
  - L(N) ∝ N^(-0.076)，L(D) ∝ D^(-0.095)，L(C) ∝ C^(-0.050)
  - 在固定计算预算下，应该优先增大模型而非数据
- Chinchilla定律（Hoffmann et al., 2022）：
  - 修正了OpenAI的结论：模型参数量和数据量应该等比例增长
  - 最优比例：约20 tokens/parameter
  - Chinchilla（70B参数，1.4T tokens）优于Gopher（280B参数，300B tokens）
  - 影响：后续模型更注重数据量（LLaMA用更多数据训练较小模型）
- 涌现能力与Scaling：
  - 某些能力（如思维链推理）只在模型超过一定规模后才出现
  - 是否真的是"涌现"还是评估指标的阶跃效应，学术界有争议
- **为什么重要：** Scaling Laws指导了大模型的训练资源分配，是理解"为什么要做大模型"的理论基础

**推荐资源：**
- 论文：《BERT: Pre-training of Deep Bidirectional Transformers》
- 论文：《Language Models are Few-Shot Learners》（GPT-3）
- 论文：《Training Compute-Optimal Large Language Models》（Chinchilla）
- Jay Alammar博客：《The Illustrated BERT》《The Illustrated GPT-2》
- 李沐精读BERT和GPT论文视频
- Lilian Weng博客：《Large Language Model》综述

**练习任务：**
1. 用Hugging Face加载预训练BERT，在文本分类任务上微调（如SST-2情感分析）
2. 用GPT-2做文本生成，对比Zero-shot和Few-shot的效果
3. 计算不同规模模型的参数量（如7B/13B/70B），理解参数分布在哪些组件
4. 实现一个简单的MLM预训练任务：随机遮盖token并预测
5. 阅读Chinchilla论文的核心结论，计算给定计算预算下的最优模型大小和数据量

**自测清单：**
- [ ] 能解释困惑度的定义和直觉含义
- [ ] 能说清Word2Vec的CBOW和Skip-gram的区别
- [ ] 能解释BERT的MLM预训练任务和15%遮盖策略的设计原因
- [ ] 能对比BERT和GPT的架构差异和适用场景
- [ ] 能解释为什么Decoder-Only架构成为大模型主流
- [ ] 能说清Chinchilla定律的核心结论及其对大模型训练的影响

---

### Day 13（3月3日）：大模型微调技术（SFT/RLHF）——重点

**学习目标：**
- 掌握指令微调（Instruction Tuning）的数据构造方法和训练流程
- 深入理解LoRA/QLoRA等参数高效微调方法的原理
- 完整掌握RLHF流程：奖励模型训练→PPO优化→KL约束
- 理解DPO等简化对齐方法的原理和优势
- 掌握分布式训练和混合精度训练的核心概念

#### 核心知识点

**1. 指令微调（Instruction Tuning）**
- 目标：让预训练模型学会遵循人类指令，从"续写文本"变为"回答问题"
- 数据格式：
  - 单轮：`{"instruction": "...", "input": "...", "output": "..."}`
  - 多轮：`{"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}`
  - Chat Template：不同模型有不同的模板格式（ChatML、LLaMA格式等）
- 数据构造方法：
  - 人工标注：质量最高但成本高，适合种子数据
  - Self-Instruct：用大模型生成指令-回答对，再人工筛选
  - Evol-Instruct（WizardLM）：逐步增加指令复杂度
  - 从开源数据集收集：Alpaca、ShareGPT、FLAN等
- 数据质量 vs 数据数量：
  - LIMA论文：仅1000条高质量数据就能达到很好的微调效果
  - 数据多样性比数量更重要：覆盖不同任务类型、难度级别、领域
  - 数据去污染：确保评测集数据不在训练集中
- **为什么重要：** SFT数据构造是字节跳动CQC模型运营岗位的核心工作内容，数据质量直接决定模型效果

**2. SFT训练流程**
- 数据准备：收集→清洗→格式化→质量筛选→划分训练/验证集
- 模型选择：选择合适的基座模型（LLaMA/Qwen/Mistral等）
- 超参数设置：
  - 学习率：通常1e-5到5e-5（比预训练小1-2个数量级）
  - Batch Size：根据GPU显存确定，通常用梯度累积模拟大batch
  - Epoch数：通常1-3个epoch，过多会过拟合（灾难性遗忘）
  - 序列长度：根据数据分布确定，通常2048-4096
  - Warmup比例：通常3%-10%
- 训练技巧：
  - 只对assistant回复部分计算损失（不对instruction部分计算）
  - 数据打包（Packing）：将多个短样本拼接到一个序列中，提高GPU利用率
  - 课程学习（Curriculum Learning）：从简单到复杂逐步训练
- 评估：困惑度、人工评估、自动评估（MT-Bench、AlpacaEval）
- **为什么重要：** SFT训练流程是大模型微调的标准实践，面试中经常要求描述完整的训练Pipeline

**3. 参数高效微调PEFT**
- 全量微调的问题：
  - 7B模型全量微调需要约56GB显存（FP16），加上优化器状态需要更多
  - 每个任务保存一份完整模型，存储成本高
- LoRA（Low-Rank Adaptation）：
  - 核心思想：权重更新矩阵ΔW是低秩的，ΔW = BA，B∈R^(d×r)，A∈R^(r×k)，r<<min(d,k)
  - 冻结原始权重W，只训练A和B
  - 推理时合并：W' = W + BA，无额外推理开销
  - 通常应用于注意力层的Q/V投影矩阵
  - 秩r的选择：通常4-64，r越大表达能力越强但参数越多
  - 缩放因子α：ΔW = (α/r)·BA，控制LoRA更新的幅度
  - 参数量：原始d×k → 2×d×r（如d=4096, k=4096, r=16：从16M降到128K）
- QLoRA：
  - 将基座模型量化到4-bit（NF4量化），LoRA适配器保持FP16
  - 双重量化：对量化常数再做一次量化，进一步节省内存
  - 分页优化器：将优化器状态放到CPU内存，GPU显存不足时自动换页
  - 效果：在单张24GB GPU上微调65B模型，效果接近全量微调
- 其他PEFT方法：
  - Adapter：在Transformer层中插入小型瓶颈层（下投影→非线性→上投影）
  - Prefix-Tuning：在每层注意力的K/V前拼接可学习的前缀向量
  - P-Tuning v2：在每层都加入可学习的提示向量
  - IA³：学习三个缩放向量，分别作用于K/V/FFN
- **为什么重要：** LoRA/QLoRA是当前最主流的微调方法，面试必考；理解低秩分解的原理是关键

**4. RLHF完整流程**
- 三个阶段：
  - 阶段1：SFT（已在上面介绍）
  - 阶段2：训练奖励模型（Reward Model）
  - 阶段3：用PPO算法优化策略模型
- 奖励模型训练：
  - 数据：人类对同一prompt的多个回复进行排序（偏好数据）
  - 模型：通常与SFT模型同架构，去掉语言建模头，加一个标量输出头
  - 损失函数：Bradley-Terry模型，L = -log(σ(r(x,yₓ) - r(x,yₗ)))
    - yₓ是偏好回复，yₗ是非偏好回复
  - 训练技巧：数据质量（标注一致性IAA）、正则化防止过拟合
- PPO（Proximal Policy Optimization）：
  - 目标：最大化奖励的同时，不让策略偏离SFT模型太远
  - 目标函数：max E[r(x,y)] - β·KL(π_θ ‖ π_ref)
    - π_θ：当前策略模型
    - π_ref：参考模型（SFT模型的冻结副本）
    - β：KL惩罚系数，控制探索程度
  - PPO的裁剪目标：L^CLIP = min(rₜ·Aₜ, clip(rₜ, 1-ε, 1+ε)·Aₜ)
    - rₜ = π_θ(aₜ|sₜ) / π_old(aₜ|sₜ)，策略比率
    - Aₜ：优势函数（GAE估计）
  - 训练中需要同时维护4个模型：策略模型、参考模型、奖励模型、价值模型
- KL散度约束的作用：
  - 防止奖励黑客（Reward Hacking）：模型找到奖励模型的漏洞而非真正提升质量
  - 保持语言能力：不让模型为了高奖励而退化为重复或无意义的输出
- **为什么重要：** RLHF是ChatGPT成功的关键技术，完整理解三阶段流程是面试重点

**5. DPO（Direct Preference Optimization）**
- 动机：RLHF流程复杂（需要训练奖励模型+PPO），训练不稳定
- 核心思想：将奖励建模和策略优化合并为一步
  - 数学推导：从RLHF的最优策略出发，推导出奖励函数可以用策略的对数概率比表示
  - r(x,y) = β·log(π_θ(y|x)/π_ref(y|x)) + β·log Z(x)
- DPO损失函数：
  - L_DPO = -E[log σ(β·log(π_θ(yₓ|x)/π_ref(yₓ|x)) - β·log(π_θ(yₗ|x)/π_ref(yₗ|x)))]
  - 只需要策略模型和参考模型，不需要奖励模型和价值模型
- 优点：
  - 训练简单：只需偏好数据对，标准的监督学习流程
  - 更稳定：不需要PPO的复杂训练循环
  - 内存效率：只需2个模型（策略+参考）而非4个
- 缺点：
  - 对偏好数据质量更敏感
  - 可能不如RLHF灵活（无法在线采样）
- **为什么重要：** DPO已成为RLHF的主流替代方案，LLaMA 2/3、Qwen等模型都使用了DPO或其变体

**6. 其他对齐方法**
- ORPO（Odds Ratio Preference Optimization）：
  - 将SFT和偏好优化合并为一步
  - 在SFT损失基础上加入odds ratio惩罚项
  - 不需要参考模型，进一步简化流程
- SimPO（Simple Preference Optimization）：
  - 用序列平均对数概率替代DPO中的对数概率
  - 加入长度归一化，避免模型偏好长回复
- Constitutional AI（Anthropic）：
  - 用AI自身来评判和改进回复（RLAIF）
  - 定义一组"宪法"原则，AI根据原则自我批评和修正
- RLAIF（RL from AI Feedback）：
  - 用大模型替代人类标注偏好数据
  - 降低标注成本，但可能引入模型偏见
- **为什么重要：** 对齐技术是大模型安全和质量的关键，了解最新方法有助于面试中展示前沿认知

**7. 分布式训练**
- 数据并行（Data Parallelism, DP）：
  - 每个GPU持有完整模型副本，处理不同的数据batch
  - 梯度在所有GPU间All-Reduce同步
  - 简单但内存效率低（每个GPU都存完整模型）
- 模型并行（Model Parallelism, MP）：
  - 张量并行（Tensor Parallelism）：将单个层的权重矩阵切分到多个GPU
    - 如将FFN的权重按列切分，各GPU计算部分结果后拼接
  - 流水线并行（Pipeline Parallelism）：将不同层分配到不同GPU
    - 微批次（Micro-batch）流水线：减少GPU空闲时间（气泡）
    - GPipe、PipeDream等方案
- ZeRO（Zero Redundancy Optimizer）：
  - ZeRO-1：分片优化器状态
  - ZeRO-2：分片优化器状态 + 梯度
  - ZeRO-3：分片优化器状态 + 梯度 + 模型参数
  - 效果：ZeRO-3可以在N个GPU上训练N倍大的模型
- DeepSpeed：微软的分布式训练框架，实现了ZeRO
- FSDP（Fully Sharded Data Parallel）：PyTorch原生的ZeRO-3实现
- 3D并行：数据并行 × 张量并行 × 流水线并行，训练超大模型的标准方案
- **为什么重要：** 分布式训练是大模型训练的基础设施，面试中经常考察ZeRO和3D并行的原理

**8. 混合精度训练**
- FP32（32位浮点）：标准精度，1个符号位+8个指数位+23个尾数位
- FP16（16位浮点）：半精度，范围小（±65504），容易溢出
- BF16（Brain Float 16）：与FP32相同的指数范围，但尾数精度低
  - 优点：不容易溢出，训练更稳定
  - 缺点：精度略低于FP16
  - 现代GPU（A100+）和大模型训练的首选
- FP8：8位浮点，E4M3（训练前向）和E5M2（训练反向）
  - H100 GPU原生支持，进一步加速训练
- 混合精度训练流程：
  - 模型权重保持FP32主副本
  - 前向和反向传播用FP16/BF16
  - 梯度缩放（Loss Scaling）：放大损失值防止FP16梯度下溢
  - 更新时转回FP32
- **为什么重要：** 混合精度训练是大模型训练的标准配置，BF16 vs FP16的选择是面试常见问题

**推荐资源：**
- 论文：《LoRA: Low-Rank Adaptation of Large Language Models》
- 论文：《QLoRA: Efficient Finetuning of Quantized LLMs》
- 论文：《Training language models to follow instructions with human feedback》（InstructGPT/RLHF）
- 论文：《Direct Preference Optimization》（DPO）
- Hugging Face PEFT库文档和教程
- DeepSpeed官方文档和ZeRO论文
- Lilian Weng博客：《RLHF》综述

**练习任务：**
1. 用Hugging Face PEFT库对一个小模型（如LLaMA-7B）做LoRA微调
2. 手推LoRA的参数量计算：给定d_model=4096, r=16，计算LoRA参数量占比
3. 实现DPO的损失函数：给定偏好数据对，计算DPO loss
4. 构造一个SFT数据集（至少50条），包含不同类型的指令（问答、摘要、翻译、代码）
5. 计算训练一个7B模型所需的GPU显存（FP16全量微调 vs QLoRA）

**自测清单：**
- [ ] 能描述SFT数据构造的完整流程和质量控制方法
- [ ] 能推导LoRA的数学原理并解释秩r的选择
- [ ] 能画出RLHF三阶段流程图并解释每个阶段的目标
- [ ] 能解释DPO相比RLHF的优势和数学推导思路
- [ ] 能说清ZeRO-1/2/3的区别和各自分片的内容
- [ ] 能解释BF16相比FP16的优势

---

### Day 14（3月4日）：大模型推理与部署

**学习目标：**
- 掌握模型量化的主流方法（GPTQ/AWQ/GGUF）及其原理
- 理解推理优化技术（KV Cache、Continuous Batching、Speculative Decoding）
- 熟悉主流推理框架（vLLM、TensorRT-LLM、llama.cpp）的特点
- 掌握各种解码策略的原理和参数调节
- 理解模型服务化的工程实践和评估指标体系
- 了解大模型安全的基本概念和防御方法

#### 核心知识点

**1. 模型量化**
- 量化的目标：用更低精度的数据类型表示权重，减少内存占用和计算量
- INT8量化：
  - 将FP16权重映射到[-128, 127]的整数范围
  - 对称量化：x_q = round(x / scale)，scale = max(|x|) / 127
  - 非对称量化：x_q = round(x / scale) + zero_point
  - LLM.int8()（bitsandbytes）：混合精度分解，异常值用FP16处理
- INT4量化：进一步压缩，每个权重只占4bit
- GPTQ（GPT Quantization）：
  - 基于OBQ（Optimal Brain Quantization）的逐层量化方法
  - 核心思想：量化一个权重时，调整其他权重来补偿量化误差
  - 使用Hessian矩阵的逆来计算最优补偿
  - 需要校准数据集（通常128条样本）
  - 量化速度快（几小时内完成），效果好
- AWQ（Activation-aware Weight Quantization）：
  - 观察：少数"显著"权重（对应激活值大的通道）对模型质量影响最大
  - 策略：对显著权重通道做缩放保护，再统一量化
  - 不需要反向传播，量化速度更快
  - 效果通常优于GPTQ
- GGUF格式（llama.cpp）：
  - GGML的升级格式，支持多种量化级别（Q2_K到Q8_0）
  - 适合CPU推理和边缘设备部署
  - 支持混合量化：不同层使用不同的量化精度
- **为什么重要：** 量化是大模型落地部署的关键技术，面试中经常考察GPTQ和AWQ的原理区别

**2. 推理优化技术**
- KV Cache（Day 11已介绍，此处补充工程细节）：
  - 内存计算：2 × n_layers × seq_len × n_kv_heads × d_head × dtype_bytes
  - 7B模型（32层，32头，128维，FP16）：seq_len=4096时约2GB
  - 优化：GQA/MQA减少KV头数、量化KV Cache（FP8/INT8）
- Continuous Batching（连续批处理）：
  - 传统静态批处理：等所有请求生成完毕才处理下一批
  - 连续批处理：某个请求完成后立即插入新请求，不等其他请求
  - 迭代级调度（Iteration-level Scheduling）：每生成一个token就检查是否有请求完成
  - 效果：吞吐量提升2-5倍
- PagedAttention（vLLM的核心创新）：
  - 借鉴操作系统的虚拟内存和分页机制
  - 将KV Cache分成固定大小的"页"（Block），按需分配
  - 解决KV Cache的内存碎片问题（传统方式需要预分配连续内存）
  - 支持KV Cache共享（如beam search中多个beam共享前缀）
  - 内存利用率从20-40%提升到接近100%
- Speculative Decoding（投机解码）：
  - 用一个小模型（Draft Model）快速生成多个候选token
  - 用大模型（Target Model）并行验证这些候选token
  - 如果小模型预测正确，一次验证通过多个token，加速生成
  - 加速比取决于小模型与大模型的一致率（通常2-3倍）
  - 保证输出分布与大模型完全一致（无损加速）
- **为什么重要：** 推理优化直接影响服务成本和用户体验，PagedAttention和Speculative Decoding是面试热点

**3. 主流推理框架**
- vLLM：
  - 核心特性：PagedAttention、Continuous Batching、Tensor Parallelism
  - 支持OpenAI兼容API、多种模型格式
  - 适合：高吞吐量在线服务
- TensorRT-LLM（NVIDIA）：
  - 基于TensorRT的LLM推理优化
  - 支持INT4/INT8/FP8量化、Inflight Batching
  - 深度优化NVIDIA GPU性能，通常是最快的选择
  - 适合：追求极致性能的生产环境
- llama.cpp：
  - 纯C/C++实现，支持CPU推理
  - GGUF量化格式，支持多种量化级别
  - 适合：边缘设备、个人电脑、无GPU环境
- SGLang：
  - RadixAttention：前缀缓存，复用相同前缀的KV Cache
  - 适合：多轮对话、共享系统提示的场景
- Triton Inference Server（NVIDIA）：
  - 通用模型服务框架，支持多种后端
  - 适合：多模型混合部署的生产环境
- **为什么重要：** 了解不同推理框架的特点有助于根据场景选择合适的方案

**4. 解码策略**
- Greedy Decoding：每步选概率最高的token
  - 简单快速，但容易生成重复、缺乏多样性的文本
- Beam Search：维护K个最优候选序列
  - beam_size=K，每步扩展所有候选，保留得分最高的K个
  - 适合翻译等需要精确输出的任务
  - 不适合开放式生成（倾向于生成短而安全的回复）
- Top-K Sampling：从概率最高的K个token中采样
  - K太小→多样性不足，K太大→可能采到低质量token
  - 问题：不同位置的概率分布差异大，固定K不够灵活
- Top-P（Nucleus）Sampling：从累积概率达到P的最小token集合中采样
  - 自适应：概率集中时选择少，概率分散时选择多
  - 通常P=0.9-0.95
  - 比Top-K更灵活，是当前最常用的采样策略
- Temperature：T参数控制概率分布的"锐度"
  - softmax(zᵢ/T)：T<1→分布更尖锐（更确定），T>1→分布更平坦（更随机）
  - T=0等价于Greedy，T→∞等价于均匀采样
- Repetition Penalty：降低已生成token的概率，减少重复
  - 频率惩罚（Frequency Penalty）：根据出现次数惩罚
  - 存在惩罚（Presence Penalty）：只要出现过就惩罚
- Min-P Sampling：设置最低概率阈值，过滤掉概率过低的token
- **为什么重要：** 解码策略直接影响生成质量，面试中经常考察Top-P和Temperature的原理

**5. 模型服务化**
- API设计：
  - RESTful API：`/v1/chat/completions`（OpenAI兼容格式）
  - 请求参数：model、messages、temperature、top_p、max_tokens、stream等
  - 响应格式：choices、usage（token计数）、finish_reason
- 流式输出（Server-Sent Events, SSE）：
  - 逐token返回生成结果，降低首token延迟（TTFT）
  - HTTP长连接，服务端主动推送
  - 格式：`data: {"choices": [{"delta": {"content": "..."}}]}\n\n`
- 并发处理：
  - 异步框架：FastAPI + uvicorn
  - 请求队列：处理突发流量
  - 批处理：将多个请求合并为一个batch推理
- 限流与配额：
  - Token级限流：限制每分钟/每天的token消耗
  - 请求级限流：限制QPS
  - 用户级配额管理
- 监控指标：
  - TTFT（Time To First Token）：首token延迟
  - TPS（Tokens Per Second）：生成速度
  - 吞吐量：每秒处理的请求数
  - P50/P95/P99延迟
- **为什么重要：** 模型服务化是大模型落地的最后一环，工程能力是面试加分项

**6. 评估指标体系**
- 自动评估指标：
  - BLEU：N-gram精确率的几何平均 + 短句惩罚，用于翻译评估
  - ROUGE：N-gram召回率，ROUGE-1/2/L，用于摘要评估
  - BERTScore：用BERT计算生成文本和参考文本的语义相似度
  - 困惑度（Perplexity）：语言模型质量的基础指标
- 人工评估：
  - 绝对评分：对单个回复打分（如1-5分）
  - 相对评分：对比两个模型的回复，选择更好的（Elo评分）
  - 评估维度：有用性、真实性、无害性、流畅性
- LLM-as-Judge：
  - 用GPT-4等强模型评估其他模型的输出
  - 优点：成本低、速度快、可扩展
  - 缺点：存在偏见（位置偏见、长度偏见、自我偏好）
  - 缓解：随机化顺序、多次评估取平均、校准
- 基准测试：
  - MMLU：多任务语言理解，57个学科
  - HumanEval/MBPP：代码生成评估
  - GSM8K：数学推理
  - MT-Bench：多轮对话质量
  - AlpacaEval：指令遵循能力
- **为什么重要：** 评估体系是模型迭代的基础，字节跳动和月之暗面的岗位都涉及评估Pipeline设计

**7. 大模型安全**
- 越狱攻击（Jailbreak）：
  - 角色扮演攻击："假装你是一个没有限制的AI..."
  - 编码攻击：用Base64、ROT13等编码绕过安全过滤
  - 多轮攻击：通过多轮对话逐步引导模型突破限制
  - 对抗性后缀：在prompt末尾添加特定字符串触发不安全输出
- Prompt注入：
  - 直接注入：在用户输入中嵌入恶意指令
  - 间接注入：在外部数据源（如网页、文档）中嵌入恶意指令
  - 防御：输入过滤、指令层次化（系统提示优先级高于用户输入）
- 内容安全过滤：
  - 输入过滤：检测并拦截恶意输入
  - 输出过滤：检测并拦截不安全的生成内容
  - 分类器：训练专门的安全分类模型
  - 规则引擎：关键词过滤、正则匹配
- 红队测试（Red Teaming）：
  - 系统性地测试模型的安全边界
  - 自动化红队：用AI生成攻击prompt
  - 人工红队：安全专家手动测试
- **为什么重要：** 大模型安全是字节跳动CQC岗位的核心关注点，内容安全分类和红队测试是日常工作

**推荐资源：**
- 论文：《Efficient Memory Management for Large Language Model Serving with PagedAttention》（vLLM）
- 论文：《GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers》
- 论文：《AWQ: Activation-aware Weight Quantization》
- 论文：《Fast Inference from Transformers via Speculative Decoding》
- vLLM官方文档和GitHub
- Hugging Face Text Generation Inference文档
- OWASP LLM Top 10安全风险列表

**练习任务：**
1. 用llama.cpp将一个模型量化为Q4_K_M格式，对比量化前后的困惑度和推理速度
2. 用vLLM部署一个模型，测试Continuous Batching下的吞吐量
3. 实现Top-P和Temperature采样：给定logits，手写采样函数
4. 计算一个13B模型在不同量化精度下的内存占用（FP16/INT8/INT4）
5. 设计一个简单的Prompt注入攻击和防御方案

**自测清单：**
- [ ] 能解释GPTQ和AWQ的核心区别
- [ ] 能说清PagedAttention解决了什么问题以及如何解决
- [ ] 能解释Speculative Decoding的工作原理和加速条件
- [ ] 能说清Top-P采样和Temperature的作用
- [ ] 能描述模型服务化的完整架构（API→队列→推理引擎→监控）
- [ ] 能列举至少3种越狱攻击方式和对应的防御方法

---

## 第四阶段：应用开发与面试冲刺（Day 15-20）

---

### Day 15（3月5日）：Prompt Engineering与应用开发

**学习目标：**
- 系统掌握Prompt Engineering的各种技术（Zero-shot/Few-shot/CoT/ToT）
- 理解高级提示技巧（ReAct、Self-Refine、Reflection）的原理与应用
- 掌握LangChain/LlamaIndex框架的核心概念和使用方法
- 理解函数调用（Function Calling）和结构化输出的实现机制
- 掌握多轮对话管理和上下文窗口优化策略

#### 核心知识点

**1. Prompt Engineering系统化**
- Zero-shot Prompting：直接给出任务描述，不提供示例
  - 适合简单任务或模型能力足够强的场景
  - 技巧：明确任务定义、指定输出格式、设定角色
- Few-shot Prompting：在prompt中提供几个示例（通常3-5个）
  - 示例选择策略：覆盖不同类型、难度递进、与测试样本相似
  - 示例顺序影响：最近的示例影响最大（recency bias）
  - 标签平衡：正负例数量均衡
- Chain-of-Thought（CoT）思维链：
  - 在prompt中展示推理过程，引导模型逐步思考
  - "Let's think step by step"（Zero-shot CoT）
  - 手动编写推理链示例（Manual CoT）
  - 对数学推理、逻辑推理任务效果显著
  - 只在足够大的模型上有效（通常>10B参数）
- Tree-of-Thought（ToT）思维树：
  - 将推理过程组织为树结构，每个节点是一个思考步骤
  - 支持回溯：如果某条路径不通，可以退回尝试其他路径
  - 搜索策略：BFS（广度优先）或DFS（深度优先）
  - 适合需要探索和规划的复杂问题（如24点游戏、创意写作）
- Self-Consistency：
  - 对同一问题多次采样（用较高Temperature），取多数投票的答案
  - 直觉：正确的推理路径更可能被多次采样到
  - 通常采样5-10次，显著提升CoT的准确率
- **为什么重要：** Prompt Engineering是使用大模型的核心技能，CoT和ToT是面试高频考点

**2. Prompt设计原则**
- 角色设定（System Prompt）：
  - 定义模型的身份、能力边界、行为准则
  - 示例："你是一个专业的Python程序员，只回答编程相关问题"
  - 好的角色设定可以显著提升回复质量和一致性
- 任务分解：
  - 将复杂任务拆分为多个简单子任务
  - 每个子任务用独立的prompt处理
  - 用Pipeline串联子任务的输入输出
- 格式约束：
  - 明确指定输出格式（JSON、Markdown、表格等）
  - 提供输出模板或Schema
  - 使用分隔符区分不同部分（如```、---、<tag>）
- 示例选择：
  - 多样性：覆盖不同的输入类型和边界情况
  - 相关性：与实际任务场景相似
  - 质量：示例本身必须正确且高质量
- 负面指令：告诉模型"不要做什么"通常不如告诉"要做什么"有效
- **为什么重要：** 好的Prompt设计是大模型应用开发的基础，也是面试中展示实践经验的关键

**3. 高级Prompt技巧**
- ReAct（Reasoning + Acting）：
  - 交替进行推理（Thought）和行动（Action）
  - 格式：Thought → Action → Observation → Thought → ...
  - 行动可以是调用工具（搜索、计算、查数据库等）
  - 观察是工具返回的结果
  - 是Agent框架的理论基础
- Self-Refine：
  - 模型生成初始回复 → 自我评估 → 根据评估改进 → 迭代
  - 不需要外部反馈，模型自己当"评审"
  - 适合写作、代码生成等可以自我评估的任务
- Reflection：
  - 让模型反思自己的推理过程，识别错误并修正
  - "请检查你的回答是否有错误，如果有请修正"
  - Reflexion框架：将反思结果存入记忆，用于后续决策
- Meta-Prompting：
  - 用大模型生成prompt，再用生成的prompt完成任务
  - 自动化Prompt优化：DSPy、OPRO等框架
  - APE（Automatic Prompt Engineer）：自动搜索最优prompt
- **为什么重要：** 这些高级技巧是构建复杂AI应用的核心方法，ReAct是Agent的理论基础

**4. LangChain / LlamaIndex框架**
- LangChain核心概念：
  - Chain：将多个组件串联，如 Prompt → LLM → OutputParser
  - Agent：根据用户输入动态决定调用哪些工具
  - Memory：管理对话历史，支持多种存储方式
    - ConversationBufferMemory：存储完整对话历史
    - ConversationSummaryMemory：用LLM总结历史
    - ConversationBufferWindowMemory：只保留最近K轮
  - Tool：封装外部功能（搜索、计算、API调用等）
  - Retriever：从知识库检索相关文档（RAG的核心组件）
- LlamaIndex核心概念：
  - 专注于数据索引和检索（RAG场景）
  - Document：原始文档
  - Node：文档分块后的节点
  - Index：索引结构（VectorStoreIndex、TreeIndex等）
  - QueryEngine：查询引擎，组合检索和生成
- LCEL（LangChain Expression Language）：
  - 声明式的链式调用语法：`chain = prompt | llm | parser`
  - 支持流式输出、批处理、异步调用
- **为什么重要：** LangChain/LlamaIndex是大模型应用开发的主流框架，面试中经常考察架构设计

**5. 函数调用与结构化输出**
- Function Calling（函数调用）：
  - 模型根据用户意图，决定调用哪个函数以及传入什么参数
  - API定义：在请求中传入函数的名称、描述、参数Schema（JSON Schema）
  - 模型输出：函数名 + 参数JSON
  - 应用开发者执行函数，将结果返回给模型继续生成
  - 支持并行函数调用（一次调用多个函数）
- 结构化输出（JSON Mode / Structured Output）：
  - 强制模型输出符合指定JSON Schema的结构化数据
  - 实现方式：约束解码（Constrained Decoding），在生成时限制token选择
  - 应用：信息提取、数据转换、API响应生成
- Tool Use vs Function Calling：
  - Tool Use是更通用的概念，Function Calling是其具体实现
  - 工具可以是函数、API、代码执行器、搜索引擎等
- **为什么重要：** 函数调用是构建AI Agent的核心能力，月之暗面Kimi的代码Agent就依赖函数调用机制

**6. 多轮对话管理与上下文优化**
- 对话历史管理：
  - 完整历史：将所有历史消息传入，简单但token消耗大
  - 滑动窗口：只保留最近N轮对话
  - 摘要压缩：用LLM将历史对话总结为简短摘要
  - 混合策略：最近几轮保留完整 + 更早的历史用摘要
- 上下文窗口优化：
  - 问题：上下文窗口有限（4K-128K tokens），需要高效利用
  - 策略：
    - 压缩系统提示：精简角色设定和指令
    - 动态检索：只检索与当前问题相关的上下文（RAG）
    - 分层上下文：重要信息放在开头和结尾（Lost in the Middle问题）
- Lost in the Middle：
  - 研究发现：模型对上下文中间位置的信息关注度最低
  - 对策：将关键信息放在上下文的开头或结尾
- **为什么重要：** 多轮对话管理是聊天应用的核心工程问题，上下文优化直接影响用户体验和成本

**推荐资源：**
- LangChain官方文档和教程
- LlamaIndex官方文档
- 论文：《Chain-of-Thought Prompting Elicits Reasoning in Large Language Models》
- 论文：《ReAct: Synergizing Reasoning and Acting in Language Models》
- 论文：《Tree of Thoughts: Deliberate Problem Solving with Large Language Models》
- OpenAI Function Calling官方文档
- Lilian Weng博客：《Prompt Engineering》综述

**练习任务：**
1. 设计一个CoT prompt，在GSM8K数学题上对比Zero-shot和Few-shot CoT的效果
2. 用LangChain构建一个简单的Agent，能调用搜索和计算工具回答问题
3. 实现Self-Consistency：对同一数学题采样5次，取多数投票答案
4. 用Function Calling实现一个天气查询助手：用户问天气→模型调用天气API→返回结果
5. 实现一个多轮对话管理器：支持滑动窗口和摘要压缩两种策略

**自测清单：**
- [ ] 能解释CoT、ToT、Self-Consistency的原理和适用场景
- [ ] 能设计一个高质量的Few-shot prompt并解释示例选择策略
- [ ] 能说清ReAct框架的Thought-Action-Observation循环
- [ ] 能用LangChain构建一个包含Memory和Tool的Agent
- [ ] 能解释Function Calling的工作流程
- [ ] 能设计多轮对话的上下文管理策略

---

### Day 16（3月6日）：RAG检索增强生成

**学习目标：**
- 掌握RAG的完整架构：索引构建→检索→重排序→生成
- 理解文本分块策略的选择依据和实现方法
- 掌握Embedding模型的选择和向量数据库的使用
- 理解稠密检索、稀疏检索和混合检索的原理与权衡
- 掌握高级RAG技术（Self-RAG、Graph RAG等）
- 能够根据场景选择RAG vs 长上下文 vs 微调

#### 核心知识点

**1. RAG完整架构**
- 离线索引阶段：
  - 文档加载：PDF、Word、HTML、Markdown等格式解析
  - 文本分块（Chunking）：将长文档切分为适当大小的片段
  - 向量化（Embedding）：将文本片段转换为向量表示
  - 索引存储：将向量存入向量数据库，建立索引
- 在线检索阶段：
  - 查询理解：对用户查询做改写、扩展、分解
  - 检索（Retrieval）：从向量数据库中找到最相关的文档片段
  - 重排序（Reranking）：用更精确的模型对检索结果重新排序
  - 生成（Generation）：将检索到的上下文和用户查询一起输入LLM生成回答
- Naive RAG的问题：
  - 检索质量差：分块不合理、查询与文档语义不匹配
  - 生成质量差：检索到的内容不相关或冗余，模型产生幻觉
  - 需要Advanced RAG技术来解决
- **为什么重要：** RAG是大模型落地最广泛的应用模式，几乎所有企业级AI应用都涉及RAG

**2. 文本分块策略**
- 固定大小分块：
  - 按字符数或token数切分，设置重叠（overlap）防止信息断裂
  - 简单快速，但可能在语义不完整的位置切断
  - 常用参数：chunk_size=512 tokens, overlap=50 tokens
- 递归分块（Recursive Splitting）：
  - 按层次化的分隔符递归切分：段落→句子→词
  - LangChain的`RecursiveCharacterTextSplitter`
  - 优先在自然边界（段落、句子）处切分
- 语义分块（Semantic Chunking）：
  - 用Embedding计算相邻句子的语义相似度
  - 在语义变化大的位置切分（相似度低于阈值）
  - 保证每个chunk内语义连贯
- 文档结构分块：
  - 利用文档结构（标题、章节、列表）进行切分
  - Markdown按标题层级切分
  - HTML按DOM结构切分
- 分块大小的权衡：
  - 太小：上下文不完整，检索到的信息碎片化
  - 太大：包含无关信息，降低检索精度，浪费上下文窗口
  - 经验值：256-1024 tokens，根据具体场景调整
- **为什么重要：** 分块质量直接决定检索质量，是RAG系统中最容易被忽视但影响最大的环节

**3. Embedding模型**
- 文本Embedding的目标：将文本映射到稠密向量空间，语义相似的文本距离近
- 主流Embedding模型：
  - OpenAI text-embedding-3-small/large：商业API，效果好但有成本
  - BGE系列（BAAI）：开源，中英文效果优秀，BGE-M3支持多语言+多粒度
  - E5系列（Microsoft）：开源，E5-mistral-7b-instruct效果接近商业模型
  - Sentence-BERT：基于BERT的句子级Embedding，经典方法
  - GTE系列（Alibaba）：开源，多语言支持好
- 选择依据：
  - 语言支持：中文场景优先选BGE或GTE
  - 维度：768-1536维，维度越高表达能力越强但存储和计算成本越高
  - 最大长度：大多数模型支持512 tokens，部分支持8192+
  - MTEB排行榜：Massive Text Embedding Benchmark，标准评测
- 训练Embedding模型：
  - 对比学习：正样本对（语义相似）拉近，负样本对推远
  - 损失函数：InfoNCE / Triplet Loss / Multiple Negatives Ranking Loss
  - 难负例挖掘（Hard Negative Mining）：选择与正样本相似但不相关的负样本
- **为什么重要：** Embedding模型是RAG检索质量的基础，选择合适的模型直接影响系统效果

**4. 向量数据库**
- Faiss（Facebook AI Similarity Search）：
  - 开源库（非独立数据库），支持多种索引类型
  - IVF（倒排文件索引）：先聚类再在簇内搜索
  - HNSW（层次化可导航小世界图）：图索引，查询速度快
  - PQ（乘积量化）：压缩向量，减少内存占用
  - 适合：嵌入到应用中，不需要独立服务
- Milvus：
  - 开源分布式向量数据库，支持亿级向量
  - 支持多种索引（IVF_FLAT、IVF_PQ、HNSW等）
  - 支持标量过滤（混合查询）
  - 适合：大规模生产环境
- Chroma：
  - 轻量级开源向量数据库，API简洁
  - 适合：原型开发和小规模应用
- Pinecone：
  - 全托管云服务，无需运维
  - 适合：不想自建基础设施的团队
- Weaviate：
  - 开源，支持向量+关键词混合搜索
  - 内置模块化的向量化能力
- 选择依据：数据规模、是否需要分布式、是否需要托管、预算
- **为什么重要：** 向量数据库是RAG系统的核心基础设施，面试中经常考察不同方案的选型依据

**5. 检索策略**
- 稠密检索（Dense Retrieval）：
  - 用Embedding模型将查询和文档都转为向量，计算余弦相似度
  - 优点：能捕获语义相似性（同义词、释义）
  - 缺点：对精确匹配（专有名词、数字）效果差
- 稀疏检索（BM25）：
  - 基于词频的经典检索算法
  - BM25(q,d) = ∑IDF(qᵢ)·(f(qᵢ,d)·(k₁+1))/(f(qᵢ,d)+k₁·(1-b+b·|d|/avgdl))
  - 优点：精确匹配能力强、无需训练、可解释
  - 缺点：无法理解语义相似性
- 混合检索（Hybrid Search）：
  - 结合稠密检索和稀疏检索的结果
  - 融合方法：
    - RRF（Reciprocal Rank Fusion）：score = ∑1/(k+rankᵢ)，k通常=60
    - 加权融合：α·dense_score + (1-α)·sparse_score
  - 通常效果优于单一检索方法
- 查询改写（Query Rewriting）：
  - HyDE（Hypothetical Document Embeddings）：先让LLM生成假设性回答，用回答做检索
  - 查询扩展：用LLM将查询扩展为多个子查询
  - 查询分解：将复杂查询分解为多个简单查询，分别检索后合并
- **为什么重要：** 检索策略直接决定RAG的效果上限，混合检索和查询改写是提升效果的关键手段

**6. 重排序（Reranking）**
- 为什么需要重排序：
  - 初始检索（Bi-Encoder）速度快但精度有限
  - 重排序（Cross-Encoder）精度高但速度慢
  - 两阶段策略：先用Bi-Encoder快速召回Top-K，再用Cross-Encoder精排Top-N
- Cross-Encoder重排序：
  - 将查询和文档拼接输入BERT类模型，输出相关性分数
  - 比Bi-Encoder更精确（因为可以做深度交互）
  - 但不能预计算文档向量，每次查询都要重新计算
- Cohere Rerank：商业API，效果好，使用简单
- BGE-Reranker：开源重排序模型，中英文效果优秀
- 重排序的位置：检索Top-50~100 → 重排序 → 取Top-5~10 → 输入LLM
- **为什么重要：** 重排序是提升RAG精度的关键步骤，Bi-Encoder+Cross-Encoder的两阶段架构是标准实践

**7. RAG评估**
- 检索评估：
  - 召回率（Recall@K）：Top-K结果中包含正确文档的比例
  - MRR（Mean Reciprocal Rank）：正确文档排名的倒数的平均
  - NDCG（Normalized Discounted Cumulative Gain）：考虑排名位置的评估指标
- 生成评估：
  - 忠实度（Faithfulness）：生成的回答是否忠于检索到的上下文（不产生幻觉）
  - 相关性（Relevance）：回答是否与用户问题相关
  - 上下文精确度（Context Precision）：检索到的上下文中有多少是有用的
  - 上下文召回率（Context Recall）：回答所需的信息是否都被检索到了
- 评估框架：
  - RAGAS：自动化RAG评估框架，用LLM评估忠实度、相关性等
  - TruLens：RAG应用的评估和监控工具
- **为什么重要：** 没有评估就无法迭代优化，RAG评估体系是构建生产级RAG系统的基础

**8. 高级RAG技术**
- Self-RAG：
  - 模型自己决定是否需要检索（而非每次都检索）
  - 生成特殊token：[Retrieve]（是否检索）、[IsRel]（是否相关）、[IsSup]（是否支持）
  - 训练模型学会自我反思和自我评估
- CRAG（Corrective RAG）：
  - 检索后评估文档质量，如果质量差则触发网络搜索补充
  - 三种判断：Correct（直接用）、Incorrect（网络搜索）、Ambiguous（两者结合）
- Graph RAG：
  - 将文档构建为知识图谱，利用图结构进行检索
  - 适合需要多跳推理的问题（如"A的老板的妻子是谁"）
  - 微软Graph RAG：社区检测+摘要，支持全局性问题
- Multi-Modal RAG：
  - 支持图片、表格、PDF等多模态内容的检索和理解
  - 图片：用多模态Embedding或OCR+文本Embedding
  - 表格：结构化解析后索引
- RAG vs 长上下文 vs 微调：
  - RAG：知识频繁更新、需要引用来源、数据量大
  - 长上下文：数据量适中（<128K tokens）、需要全局理解
  - 微调：需要改变模型行为/风格、特定领域适配
  - 实践中常组合使用：微调+RAG
- **为什么重要：** 高级RAG技术是提升系统效果的关键，Graph RAG和Self-RAG是当前研究热点

**推荐资源：**
- LangChain RAG教程和Cookbook
- LlamaIndex官方文档（RAG部分）
- 论文：《Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks》（RAG原始论文）
- 论文：《Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection》
- 论文：《From Local to Global: A Graph RAG Approach to Query-Focused Summarization》
- RAGAS官方文档
- Pinecone Learning Center的RAG教程

**练习任务：**
1. 用LangChain/LlamaIndex构建一个完整的RAG系统：文档加载→分块→Embedding→检索→生成
2. 对比不同分块策略（固定大小 vs 递归 vs 语义分块）对检索效果的影响
3. 实现混合检索：BM25 + 稠密检索 + RRF融合
4. 用Cross-Encoder实现重排序，对比有无重排序的效果差异
5. 用RAGAS评估你构建的RAG系统的忠实度和相关性

**自测清单：**
- [ ] 能画出RAG的完整架构图并解释每个环节
- [ ] 能解释不同分块策略的优缺点和适用场景
- [ ] 能说清稠密检索和稀疏检索的互补性以及混合检索的融合方法
- [ ] 能解释Bi-Encoder和Cross-Encoder的区别以及两阶段检索的设计
- [ ] 能用RAGAS的指标体系评估RAG系统
- [ ] 能根据场景选择RAG vs 长上下文 vs 微调

---

### Day 17（3月7日）：AI Agent与工具使用

**学习目标：**
- 理解AI Agent的核心架构：感知-规划-行动循环
- 深入掌握ReAct框架的原理和实现
- 了解多Agent系统的设计模式和主流框架
- 理解代码生成Agent的设计与评估方法
- 掌握Agent安全的核心问题和防御策略
- 建立与月之暗面Kimi For Coding岗位的关联认知

#### 核心知识点

**1. Agent架构：感知-规划-行动循环**
- Agent的定义：能够自主感知环境、做出决策、执行行动的AI系统
- 核心循环：
  - 感知（Perception）：接收用户输入和环境反馈
  - 规划（Planning）：分析任务、制定计划、分解子任务
  - 行动（Action）：调用工具、执行代码、与环境交互
  - 观察（Observation）：获取行动结果，更新状态
- Agent vs Chatbot：
  - Chatbot：被动回答问题，单轮交互
  - Agent：主动规划和执行，多步交互，能使用工具
- Agent的记忆系统：
  - 短期记忆：当前对话上下文（工作记忆）
  - 长期记忆：持久化存储的知识和经验（向量数据库、文件系统）
  - 情景记忆：过去的交互经验，用于类似场景的决策参考
- **为什么重要：** Agent是大模型应用的最高级形态，月之暗面Kimi For Coding本质上就是一个代码Agent

**2. ReAct框架：Reasoning + Acting**
- 核心思想：将推理（Reasoning）和行动（Acting）交织进行
- 执行流程：
  - Thought：模型思考当前状态和下一步计划
  - Action：选择并执行一个工具/动作
  - Observation：获取工具返回的结果
  - 重复上述循环直到任务完成
- 示例格式：
  ```
  Question: 2024年诺贝尔物理学奖得主是谁？
  Thought: 我需要搜索最新的诺贝尔物理学奖信息
  Action: Search["2024 Nobel Prize Physics"]
  Observation: 2024年诺贝尔物理学奖授予了John Hopfield和Geoffrey Hinton...
  Thought: 我已经找到了答案
  Action: Finish["John Hopfield和Geoffrey Hinton"]
  ```
- ReAct vs 纯CoT：
  - 纯CoT：只推理不行动，可能产生幻觉（无法获取外部信息）
  - ReAct：推理+行动，可以通过工具获取真实信息，减少幻觉
- ReAct的局限：
  - 推理链过长时容易出错（错误累积）
  - 工具选择错误会导致后续推理偏离
  - 需要精心设计工具描述和示例
- **为什么重要：** ReAct是当前Agent框架的理论基础，LangChain Agent、AutoGPT等都基于ReAct思想

**3. 工具使用（Tool Use）**
- 工具类型：
  - API调用：天气查询、搜索引擎、数据库查询、第三方服务
  - 代码执行：Python解释器、Shell命令、Jupyter Notebook
  - 文件操作：读写文件、创建目录、文件搜索
  - 浏览器操作：网页浏览、表单填写、信息提取
- 工具定义：
  - 名称：简洁明确的工具名
  - 描述：工具的功能说明（模型根据描述选择工具）
  - 参数Schema：JSON Schema定义输入参数
  - 返回格式：工具输出的格式说明
- 工具选择策略：
  - 模型根据任务需求和工具描述，选择最合适的工具
  - 工具描述的质量直接影响选择准确率
  - 可以通过Few-shot示例引导工具选择
- 工具编排：
  - 顺序调用：一个工具的输出作为下一个工具的输入
  - 并行调用：多个独立工具同时执行
  - 条件调用：根据中间结果决定是否调用某个工具
- **为什么重要：** 工具使用能力是Agent的核心，Kimi For Coding的代码执行、文件操作等都是工具使用的具体实现

**4. 多Agent系统**
- 为什么需要多Agent：
  - 单Agent处理复杂任务时推理链过长，容易出错
  - 不同Agent可以专注于不同领域（分工协作）
  - 多Agent可以互相检查和纠错
- AutoGen（Microsoft）：
  - 多Agent对话框架，Agent之间通过消息传递协作
  - 支持人机协作（Human-in-the-loop）
  - 内置代码执行能力
  - 适合：需要多角色协作的复杂任务
- CrewAI：
  - 基于角色的多Agent框架
  - 定义Agent的角色（Role）、目标（Goal）、背景（Backstory）
  - 支持顺序执行和层次化执行
  - 适合：模拟团队协作的场景
- MetaGPT：
  - 模拟软件公司的多Agent系统
  - 角色：产品经理、架构师、工程师、测试工程师
  - 通过SOP（标准操作流程）协调Agent间的协作
  - 适合：软件开发自动化
- 多Agent通信模式：
  - 集中式：一个协调者Agent分配任务给其他Agent
  - 去中心化：Agent之间直接通信
  - 层次化：上级Agent分解任务，下级Agent执行
- **为什么重要：** 多Agent系统是解决复杂任务的前沿方向，理解其设计模式有助于面试中展示系统设计能力

**5. 代码生成Agent**
- 代码补全（Code Completion）：
  - 根据上下文预测下一段代码
  - Fill-in-the-Middle（FIM）：给定前缀和后缀，生成中间代码
  - 代表：GitHub Copilot、Cursor、Kimi Code
- 代码审查（Code Review）：
  - Agent自动审查代码质量、安全性、风格一致性
  - 检测常见问题：未处理的异常、SQL注入、内存泄漏等
  - 提供修改建议和解释
- 测试生成（Test Generation）：
  - 根据源代码自动生成单元测试
  - 覆盖正常路径、边界条件、异常情况
  - 评估：代码覆盖率、变异测试通过率
- 调试Agent：
  - 分析错误信息和堆栈跟踪
  - 定位bug的根因
  - 提出修复方案并验证
- 代码Agent的工作流：
  - 理解需求 → 检索相关代码 → 生成代码 → 执行测试 → 根据结果修正 → 迭代
  - 关键能力：代码理解、代码生成、错误诊断、自我修正
- **为什么重要：** 代码Agent是月之暗面Kimi For Coding岗位的核心产品方向，理解其设计和评估方法是面试关键

**6. Agent评估**
- 任务完成率（Task Completion Rate）：
  - Agent是否成功完成了给定任务
  - 需要明确的成功标准（如代码通过所有测试用例）
- 工具调用准确率：
  - 是否选择了正确的工具
  - 参数是否正确
  - 调用顺序是否合理
- 推理链质量：
  - 推理步骤是否逻辑连贯
  - 是否有不必要的冗余步骤
  - 错误恢复能力：出错后能否自我纠正
- 效率指标：
  - 完成任务所需的步骤数
  - Token消耗量
  - 时间消耗
- Agent评估基准：
  - SWE-bench：真实GitHub Issue修复任务
  - WebArena：网页操作任务
  - GAIA：通用AI助手评估
  - AgentBench：多环境Agent评估
- **为什么重要：** Agent评估是衡量Agent系统质量的关键，SWE-bench是代码Agent的标准评测

**7. Agent安全**
- 权限控制：
  - 最小权限原则：Agent只能访问完成任务所需的最少资源
  - 文件系统隔离：限制Agent可访问的目录范围
  - 网络访问控制：限制Agent可访问的URL和API
  - 操作审批：高风险操作（如删除文件、执行系统命令）需要人工确认
- 沙箱执行：
  - 代码执行在隔离的沙箱环境中进行
  - Docker容器：资源限制（CPU、内存、磁盘）、网络隔离
  - 时间限制：防止无限循环
  - 输出限制：防止大量输出消耗资源
- 输出验证：
  - 检查Agent的输出是否符合预期格式
  - 验证生成的代码是否安全（无恶意操作）
  - 检测Prompt注入攻击（Agent从外部数据中接收到恶意指令）
- 人机协作（Human-in-the-loop）：
  - 关键决策点需要人工确认
  - 异常情况自动暂停并通知人工
  - 审计日志：记录Agent的所有操作
- **为什么重要：** Agent安全是产品化的前提，Kimi For Coding等产品必须确保代码执行的安全性

**推荐资源：**
- 论文：《ReAct: Synergizing Reasoning and Acting in Language Models》
- 论文：《Toolformer: Language Models Can Teach Themselves to Use Tools》
- 论文：《SWE-bench: Can Language Models Resolve Real-World GitHub Issues?》
- AutoGen官方文档和教程
- CrewAI官方文档
- LangChain Agent教程
- Lilian Weng博客：《LLM Powered Autonomous Agents》

**练习任务：**
1. 用LangChain实现一个ReAct Agent，能调用搜索和计算工具回答复杂问题
2. 设计一个代码生成Agent的工作流：需求理解→代码生成→测试→修正
3. 用AutoGen或CrewAI搭建一个简单的多Agent系统（如"程序员+测试员"协作）
4. 分析SWE-bench的评估方法，理解代码Agent的评估标准
5. 设计一个Agent的安全策略：权限控制+沙箱执行+输出验证

**自测清单：**
- [ ] 能画出Agent的感知-规划-行动循环架构图
- [ ] 能解释ReAct框架的Thought-Action-Observation流程
- [ ] 能说清多Agent系统的三种通信模式
- [ ] 能描述代码Agent的完整工作流和关键能力
- [ ] 能用SWE-bench等基准解释Agent评估方法
- [ ] 能设计Agent的安全策略（权限控制、沙箱、审计）

---

### Day 18（3月8日）：代码能力评估体系——重点（对标月之暗面）

**学习目标：**
- 掌握代码评估的多维度体系：正确性、效率、可读性、安全性
- 理解静态分析和动态测试的评估方法论
- 深入掌握HumanEval/MBPP/SWE-bench等主流基准测试
- 理解Pass@k指标的定义和无偏估计方法
- 掌握评估Pipeline的完整设计：从题目生成到指标计算
- 建立持续评估和回归测试的工程实践认知

#### 核心知识点

**1. 代码评估维度**
- 功能正确性（Correctness）：
  - 代码是否能正确完成指定功能
  - 评估方法：单元测试通过率、输入输出匹配
  - 最基础也最重要的维度
- 代码效率（Efficiency）：
  - 时间复杂度和空间复杂度是否合理
  - 评估方法：运行时间对比、内存占用对比、大数据量压力测试
  - 与参考解法的效率比较
- 代码可读性（Readability）：
  - 变量命名是否有意义、代码结构是否清晰、注释是否恰当
  - 评估方法：静态分析工具（pylint、eslint）、代码风格检查
  - 难以完全自动化，通常需要人工或LLM辅助评估
- 代码安全性（Security）：
  - 是否存在SQL注入、XSS、命令注入等安全漏洞
  - 评估方法：静态安全扫描（Bandit、Semgrep）、动态安全测试
  - 在生产代码评估中尤为重要
- 风格一致性（Style Consistency）：
  - 是否遵循项目的编码规范
  - 评估方法：linter工具、AST结构对比
- **为什么重要：** 月之暗面Kimi For Coding需要全面评估代码生成质量，不仅仅是"能不能跑"，还要"跑得好不好"

**2. 评估方法论：静态分析**
- AST（抽象语法树）解析：
  - 将代码解析为树结构，分析代码的结构特征
  - Python：`ast`模块；JavaScript：`@babel/parser`
  - 应用：代码结构相似度、API使用模式分析、代码复杂度计算
- 代码复杂度指标：
  - 圈复杂度（Cyclomatic Complexity）：独立路径数，反映代码分支复杂度
  - 认知复杂度（Cognitive Complexity）：考虑嵌套深度的复杂度度量
  - Halstead指标：基于操作符和操作数的代码度量
- 代码相似度：
  - Token级相似度：编辑距离、BLEU
  - AST级相似度：树编辑距离、子树匹配
  - 语义级相似度：CodeBERTScore（用代码预训练模型计算语义相似度）
- 静态安全分析：
  - Bandit（Python）：检测常见安全问题
  - Semgrep：多语言静态分析，支持自定义规则
  - CodeQL：GitHub的代码安全分析引擎
- **为什么重要：** 静态分析是代码评估Pipeline的第一道关卡，可以快速发现结构性问题

**3. 评估方法论：动态测试**
- 单元测试：
  - 针对单个函数/方法的测试
  - 测试用例设计：正常输入、边界条件、异常输入、大数据量
  - 覆盖率指标：行覆盖率、分支覆盖率、路径覆盖率
- 集成测试：
  - 测试多个模块的协作
  - 适用于评估生成的完整程序或模块
- 模糊测试（Fuzz Testing）：
  - 自动生成大量随机输入，检测崩溃和异常行为
  - 工具：AFL、libFuzzer、Hypothesis（Python属性测试）
  - 适合发现边界条件和未处理的异常
- 差分测试（Differential Testing）：
  - 对同一输入，比较生成代码和参考实现的输出
  - 适合有明确参考答案的场景
- 沙箱执行环境：
  - Docker容器：隔离执行环境，限制资源
  - 时间限制：防止无限循环（通常5-30秒）
  - 内存限制：防止内存溢出（通常256MB-1GB）
  - 网络隔离：防止恶意网络访问
  - 文件系统隔离：只允许访问指定目录
- **为什么重要：** 动态测试是验证代码功能正确性的金标准，沙箱执行是评估Pipeline的核心基础设施

**4. 主流基准测试**
- HumanEval（OpenAI）：
  - 164个Python编程题，每题包含函数签名、docstring和测试用例
  - 评估指标：Pass@k
  - 特点：题目质量高，但规模小、只有Python、难度偏简单
- MBPP（Google）：
  - 974个Python编程题，难度从简单到中等
  - 每题3个测试用例
  - 比HumanEval规模更大，覆盖更多编程模式
- HumanEval+/MBPP+（EvalPlus）：
  - 在原始基准上增加了大量测试用例（80倍+）
  - 发现很多模型在原始基准上"通过"但实际有bug
  - 更严格、更可靠的评估
- SWE-bench：
  - 真实GitHub Issue修复任务（2294个任务）
  - 评估模型解决真实软件工程问题的能力
  - 需要理解代码库、定位bug、编写修复代码
  - SWE-bench Lite：300个精选任务的子集
  - 当前最具挑战性的代码评估基准
- LiveCodeBench：
  - 持续更新的编程竞赛题目（来自LeetCode、Codeforces等）
  - 避免数据污染（题目在模型训练截止日期之后发布）
  - 评估代码生成、自我修复、代码执行预测等多种能力
- MultiPL-E：多语言代码评估，将HumanEval翻译为18种编程语言
- **为什么重要：** 了解主流基准测试是代码评估岗位的基本要求，SWE-bench是月之暗面重点关注的评测

**5. Pass@k指标**
- 定义：从模型生成的k个代码样本中，至少有一个通过所有测试用例的概率
- 朴素估计：生成k个样本，检查是否至少一个通过
  - 问题：方差大，需要大量重复实验
- 无偏估计公式（Chen et al., 2021）：
  - 生成n个样本（n≥k），其中c个通过测试
  - Pass@k = 1 - C(n-c, k) / C(n, k)
  - 其中C(a,b)是组合数
  - 这个估计是无偏的，且方差比朴素估计小
- 实践中的参数选择：
  - 通常n=200, k∈{1, 10, 100}
  - Pass@1：模型一次生成就正确的概率（最实用的指标）
  - Pass@10：10次机会中至少一次正确（反映模型的潜力）
  - Pass@100：100次机会中至少一次正确（反映模型的覆盖能力）
- Temperature的影响：
  - 低Temperature（0.2）：Pass@1高，Pass@100低（生成集中但多样性差）
  - 高Temperature（0.8）：Pass@1低，Pass@100高（生成分散但覆盖广）
  - 评估Pass@1用低Temperature，评估Pass@k(k>1)用高Temperature
- **为什么重要：** Pass@k是代码生成评估的标准指标，理解其无偏估计公式是面试必考内容

**6. 代码质量自动评分**
- CodeBLEU：
  - 在BLEU基础上加入代码特有的匹配维度
  - 四个组成部分：N-gram匹配 + 加权N-gram匹配 + AST匹配 + 数据流匹配
  - 比纯BLEU更适合代码评估，但仍有局限（不能评估功能正确性）
- 功能正确性评估：
  - 测试用例通过率：最直接的正确性指标
  - 部分正确性：通过部分测试用例的比例
  - 编译/语法正确率：代码能否成功编译/解析
- 效率对比评估：
  - 运行时间比较：生成代码 vs 参考解法的运行时间比
  - 大O复杂度分析：通过多组不同规模输入推断时间复杂度
  - 内存使用对比
- LLM-as-Judge用于代码评估：
  - 用GPT-4等强模型评估代码质量（可读性、风格、最佳实践）
  - 优点：可以评估难以自动化的维度
  - 缺点：存在偏见，需要校准
- **为什么重要：** 多维度的代码质量评估是构建完整评估Pipeline的关键

**7. 评估Pipeline设计**
- 完整Pipeline流程：
  - 题目生成/收集 → 代码生成（调用模型API）→ 沙箱执行 → 结果判定 → 指标计算 → 报告生成
- 题目生成：
  - 从编程竞赛平台收集（LeetCode、Codeforces）
  - 从真实代码库提取（GitHub Issue、代码补全场景）
  - 用LLM生成新题目（需要人工验证质量）
  - 题目质量控制：确保题目描述清晰、测试用例充分、难度标注准确
- 代码生成：
  - 批量调用模型API，控制Temperature和采样参数
  - 支持多种prompt格式（函数签名+docstring、自然语言描述、代码补全）
  - 记录生成的token数、延迟等元数据
- 沙箱执行：
  - Docker容器池：预创建容器，减少启动开销
  - 并行执行：多个容器同时执行不同的代码样本
  - 超时处理：超时的代码标记为失败
  - 结果收集：标准输出、标准错误、退出码、运行时间、内存使用
- 结果判定：
  - 精确匹配：输出与预期完全一致
  - 近似匹配：浮点数允许误差、忽略空白字符差异
  - 特殊判定：对于多解问题，需要自定义判定逻辑
- 指标计算与报告：
  - 按题目难度、类型、编程语言分组统计
  - 计算Pass@k、编译率、运行时间分布等
  - 生成可视化报告（趋势图、对比图）
- **为什么重要：** 评估Pipeline设计是月之暗面Kimi For Coding系统工程师的核心工作内容

**8. 多语言评估与持续评估**
- 多语言评估挑战：
  - 不同语言的测试框架不同（pytest/JUnit/go test等）
  - 不同语言的沙箱环境配置不同
  - 需要为每种语言准备编译/运行环境
  - 评估结果的跨语言可比性
- 评估数据构造：
  - 数据污染检测：确保评测题目不在模型训练数据中
    - N-gram重叠检测、语义相似度检测
  - 难度校准：确保不同题目的难度标注一致
  - 测试用例充分性：用变异测试（Mutation Testing）验证测试用例质量
- 持续评估与回归测试：
  - 自动化评估流水线：模型更新后自动触发评估
  - 回归检测：对比新旧版本在相同题目上的表现
  - 性能监控：跟踪模型在各维度上的长期趋势
  - 告警机制：性能下降超过阈值时自动告警
- 大模型代码能力常见问题：
  - 长代码生成质量下降
  - 复杂逻辑推理错误
  - 不常见API使用错误
  - 边界条件处理不当
  - 代码风格不一致
- **为什么重要：** 持续评估是保证模型质量的关键，回归测试防止模型更新引入退化

**推荐资源：**
- 论文：《Evaluating Large Language Models Trained on Code》（Codex/HumanEval）
- 论文：《Is Your Code Generated by ChatGPT Really Correct?》（EvalPlus）
- 论文：《SWE-bench: Can Language Models Resolve Real-World GitHub Issues?》
- 论文：《LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code》
- EvalPlus排行榜：https://evalplus.github.io/leaderboard.html
- BigCode项目和StarCoder论文

**练习任务：**
1. 下载HumanEval数据集，用一个开源模型（如CodeLlama）生成代码并计算Pass@1和Pass@10
2. 实现Pass@k的无偏估计：给定n=200个样本中c个通过，计算Pass@1/10/100
3. 设计一个简单的代码评估沙箱：Docker容器+超时控制+结果收集
4. 用AST解析分析生成代码的结构特征（函数数量、循环嵌套深度、变量命名质量）
5. 设计一个评估Pipeline的架构图，包含题目管理、代码生成、沙箱执行、结果分析四个模块

**自测清单：**
- [ ] 能列举代码评估的五个维度并解释各自的评估方法
- [ ] 能说清HumanEval、MBPP、SWE-bench的区别和适用场景
- [ ] 能推导Pass@k的无偏估计公式并解释Temperature对Pass@k的影响
- [ ] 能设计一个完整的代码评估Pipeline（从题目到报告）
- [ ] 能解释沙箱执行的安全措施和资源限制
- [ ] 能说清数据污染检测的方法和持续评估的工程实践

---

### Day 19（3月9日）：内容安全与模型运营——重点（对标字节跳动）

**学习目标：**
- 掌握内容安全体系的完整架构：文本安全、多模态安全、安全分级
- 理解大模型安全的攻击面和防御方法
- 掌握模型运营的核心流程：数据标注管理、SFT数据运营、模型迭代
- 理解A/B测试与灰度发布在模型上线中的应用
- 掌握Badcase分析方法论：分类、归因、解决方案、验证

#### 核心知识点

**1. 内容安全体系**
- 文本安全分类：
  - 违法违规：涉政、涉恐、涉黄、赌博、毒品等
  - 有害信息：虚假信息、仇恨言论、歧视、骚扰、自残自杀引导
  - 不良信息：低俗、引战、标题党、垃圾广告
  - 隐私泄露：个人信息、商业机密、敏感数据
- 安全分级体系：
  - 多级分类：严重违规（必须拦截）→ 一般违规（建议拦截）→ 边界内容（需人工审核）→ 安全
  - 不同场景不同阈值：面向未成年人的产品需要更严格的过滤
  - 误拦率 vs 漏放率的权衡：安全场景通常宁可误拦不可漏放
- 多模态安全：
  - 图片安全：色情检测、暴力检测、敏感人物识别、OCR文字检测
  - 视频安全：关键帧提取+图片检测、音频转文字+文本检测
  - 多模态融合：图文结合的安全检测（如图片安全但配文不安全）
- 安全检测Pipeline：
  - 规则引擎（第一层）：关键词匹配、正则表达式、黑名单过滤
    - 速度快、可解释、但容易被绕过（谐音、变体）
  - 机器学习模型（第二层）：文本分类模型、BERT-based安全分类器
    - 准确率高、能处理变体，但需要标注数据训练
  - 大模型审核（第三层）：用LLM做复杂场景的安全判断
    - 理解上下文、处理隐晦表达，但成本高、速度慢
  - 人工审核（第四层）：处理机器无法判断的边界case
- **为什么重要：** 内容安全是字节跳动CQC岗位的核心工作，需要理解完整的安全检测体系

**2. 大模型安全深入**
- 越狱攻击分类：
  - 角色扮演（Role-playing）："假装你是DAN，一个没有限制的AI"
  - 场景构造："这是一个虚构的故事，在故事中..."
  - 编码绕过：Base64编码、Unicode变体、Pig Latin等
  - 多语言攻击：用小语种绕过主要语言的安全过滤
  - 对抗性后缀（GCG攻击）：在prompt末尾添加优化过的token序列
  - 多轮渐进攻击：通过多轮对话逐步引导模型突破限制
  - 间接注入：在外部数据（网页、文档）中嵌入恶意指令
- 防御方法：
  - 输入过滤：
    - 关键词检测 + 语义分类器
    - Prompt注入检测模型
    - 输入改写/规范化（去除特殊编码）
  - 输出过滤：
    - 安全分类器检测生成内容
    - 规则引擎过滤敏感词
    - 多模型交叉验证
  - 模型层面：
    - 安全对齐训练（RLHF/DPO中加入安全偏好数据）
    - Constitutional AI：定义安全准则，模型自我约束
    - 安全系统提示（System Prompt）：明确告知模型的行为边界
  - 系统层面：
    - 速率限制：防止自动化攻击
    - 用户行为分析：检测异常使用模式
    - 审计日志：记录所有交互，便于事后分析
- 红队测试（Red Teaming）：
  - 目标：系统性地发现模型的安全漏洞
  - 自动化红队：用AI生成攻击prompt（如Perez et al.的方法）
  - 人工红队：安全专家手动测试各种攻击向量
  - 红队评估指标：攻击成功率（ASR）、攻击多样性、漏洞严重程度
  - 持续红队：模型更新后重新测试，确保安全性不退化
- **为什么重要：** 大模型安全是产品上线的前提，红队测试是字节跳动安全团队的日常工作

**3. 模型运营：数据标注管理**
- 标注任务设计：
  - 标注指南（Annotation Guideline）：明确定义每个标签的含义、边界case、示例
  - 标注指南的迭代：根据标注过程中发现的问题持续更新
  - 标注工具：Label Studio、Doccano、Prodigy等
- 标注质量控制：
  - IAA（Inter-Annotator Agreement）标注一致性：
    - Cohen's Kappa：两个标注者的一致性，κ = (p_o - p_e) / (1 - p_e)
    - Fleiss' Kappa：多个标注者的一致性
    - Krippendorff's Alpha：支持多种数据类型和缺失值
    - 一般要求κ > 0.6为可接受，κ > 0.8为优秀
  - 质量检查流程：
    - 黄金标准题（Gold Standard）：预设正确答案的题目，检测标注者准确率
    - 交叉标注：同一样本由多人标注，取多数投票或专家仲裁
    - 定期校准会议：标注团队讨论分歧case，统一标准
  - 标注者管理：
    - 培训和考核：上岗前培训+考试
    - 绩效追踪：准确率、速度、一致性
    - 淘汰机制：持续低质量的标注者需要重新培训或淘汰
- **为什么重要：** 数据标注质量直接决定SFT和RLHF的效果，标注管理是模型运营的基础工作

**4. SFT数据运营**
- 数据需求分析：
  - 根据模型评估结果识别薄弱能力
  - 根据用户反馈识别高频问题类型
  - 根据业务需求确定优先级
- 数据构造策略：
  - 人工编写：质量最高，适合种子数据和高难度数据
  - 模型生成+人工筛选：效率高，适合大规模数据扩充
  - 用户数据挖掘：从真实用户交互中筛选高质量样本（需脱敏）
  - 数据增强：改写、翻译、难度调整
- 数据质量评估：
  - 指令清晰度：指令是否明确、无歧义
  - 回复质量：回复是否准确、完整、有帮助
  - 格式规范：是否符合Chat Template要求
  - 多样性：是否覆盖足够多的任务类型和难度级别
- 数据版本管理：
  - 数据集版本控制（类似代码版本控制）
  - 数据变更记录：谁在什么时候做了什么修改
  - 数据回滚能力：出问题时可以回退到之前的版本
- 数据配比优化：
  - 不同任务类型的数据比例（问答:摘要:翻译:代码 = ?）
  - 不同难度级别的比例
  - 通过实验确定最优配比
- **为什么重要：** SFT数据运营是字节跳动CQC模型运营岗位的日常核心工作

**5. 模型迭代流程**
- 完整迭代周期：
  - 评估当前模型 → 识别问题 → 制定改进方案 → 数据准备 → 训练 → 评估新模型 → 上线决策
- 问题识别：
  - 自动评估：在标准基准上定期评测
  - 人工评估：抽样检查模型回复质量
  - 用户反馈：收集和分析用户的负面反馈
  - Badcase收集：系统性收集模型表现差的案例
- 改进方案：
  - 数据层面：增加特定类型的训练数据
  - 模型层面：调整训练超参数、更换基座模型
  - 系统层面：优化Prompt、增加后处理规则
- 上线决策：
  - 对比评估：新模型 vs 旧模型在所有维度上的表现
  - 回归检测：确保新模型没有在已有能力上退化
  - 安全评估：确保新模型通过安全测试
  - 灰度发布：先对小部分用户上线，观察效果
- **为什么重要：** 模型迭代是持续提升模型质量的核心流程，理解完整流程是运营岗位的基本要求

**6. A/B测试与灰度发布**
- A/B测试在模型上线中的应用：
  - 实验组：使用新模型
  - 对照组：使用旧模型
  - 随机分流：确保两组用户特征分布一致
- 评估指标：
  - 在线指标：用户满意度（点赞/点踩）、对话轮数、留存率、使用时长
  - 离线指标：自动评估分数、人工评估分数
  - 业务指标：转化率、收入等（如果适用）
- 灰度发布策略：
  - 按比例灰度：1% → 5% → 20% → 50% → 100%
  - 按用户群灰度：先内部用户 → 种子用户 → 全量用户
  - 按地域灰度：先小地域 → 大地域 → 全量
- 监控与回滚：
  - 实时监控关键指标
  - 异常检测：指标突变自动告警
  - 快速回滚：发现问题后立即切回旧模型
- 统计显著性：
  - 样本量足够大才能得出可靠结论
  - p值 < 0.05 且效应量有实际意义才上线
  - 避免过早停止实验（Peeking Problem）
- **为什么重要：** A/B测试是模型上线决策的标准方法，灰度发布是降低上线风险的关键手段

**7. 用户反馈收集与模型监控**
- 用户反馈渠道：
  - 显式反馈：点赞/点踩、评分、文字反馈
  - 隐式反馈：重新生成（regenerate）、编辑回复、对话中断
  - 反馈数据的价值：直接反映用户满意度，是模型改进的重要信号
- 反馈数据处理：
  - 去噪：过滤恶意反馈、误操作
  - 分类：按问题类型分类（事实错误、格式问题、安全问题、能力不足等）
  - 优先级排序：根据频率和严重程度确定处理优先级
- 模型监控：
  - 性能监控：延迟、吞吐量、错误率
  - 质量监控：自动评估指标的趋势
  - 安全监控：安全事件的频率和类型
  - 成本监控：token消耗、GPU使用率
- 告警机制：
  - 阈值告警：指标超过预设阈值
  - 趋势告警：指标持续恶化
  - 异常检测：统计异常（如突然的指标跳变）
- **为什么重要：** 用户反馈和模型监控是模型运营的"眼睛"，没有监控就无法发现和解决问题

**8. Badcase分析方法论**
- Badcase定义：模型表现明显不符合预期的案例
- 分类体系：
  - 事实性错误：模型输出了错误的事实信息（幻觉）
  - 逻辑错误：推理过程有逻辑漏洞
  - 指令不遵循：没有按照用户指令执行
  - 格式错误：输出格式不符合要求
  - 安全问题：输出了不安全的内容
  - 质量问题：回复过于简短、冗长、重复、不相关
  - 能力缺失：模型不具备完成任务的能力
- 归因分析：
  - 数据问题：训练数据中缺少相关样本、数据质量差、数据标注错误
  - 模型问题：模型能力不足、过拟合、灾难性遗忘
  - 系统问题：Prompt设计不当、检索结果不相关、后处理规则缺失
  - 用户问题：指令模糊、超出模型能力范围
- 解决方案：
  - 短期：添加规则过滤、调整Prompt、增加后处理
  - 中期：补充训练数据、微调模型
  - 长期：改进模型架构、升级基座模型
- 验证闭环：
  - 修复后在原始Badcase上验证
  - 在相似case上验证泛化性
  - 回归测试确保没有引入新问题
- Badcase管理系统：
  - 收集：自动收集+人工提交
  - 标注：分类+归因+优先级
  - 追踪：每个Badcase的处理状态和解决方案
  - 统计：Badcase的分布趋势，衡量模型改进效果
- **为什么重要：** Badcase分析是模型持续改进的核心方法论，是字节跳动模型运营岗位的日常工作

**推荐资源：**
- OWASP LLM Top 10安全风险列表
- 论文：《Red Teaming Language Models to Reduce Harms》（Anthropic）
- 论文：《Universal and Transferable Adversarial Attacks on Aligned Language Models》（GCG攻击）
- 字节跳动技术博客（内容安全相关文章）
- Label Studio官方文档
- 《Trustworthy Machine Learning》相关章节

**练习任务：**
1. 设计一个文本安全分类体系：定义至少5个安全类别，每个类别给出3个示例
2. 设计一个标注指南：针对"回复质量评估"任务，定义评分标准（1-5分）和标注规范
3. 计算两个标注者在100个样本上的Cohen's Kappa值
4. 收集10个大模型的Badcase，按分类体系归类并分析原因
5. 设计一个模型上线的A/B测试方案：确定指标、样本量、实验周期、决策标准

**自测清单：**
- [ ] 能描述内容安全检测的四层Pipeline架构
- [ ] 能列举至少5种越狱攻击方式和对应的防御方法
- [ ] 能解释IAA（标注一致性）的计算方法和质量标准
- [ ] 能描述SFT数据运营的完整流程
- [ ] 能设计一个A/B测试方案并解释统计显著性的判断标准
- [ ] 能用Badcase分析方法论对一个具体case进行分类、归因和提出解决方案

---

### Day 20（3月10日）：面试冲刺与项目整合

**学习目标：**
- 掌握STAR法则包装项目经验，准备3个核心项目的完整叙述
- 系统复习高频面试题，建立快速回答的知识框架
- 掌握系统设计题的回答方法论和4个核心系统的设计思路
- 准备行为面试的常见问题和回答策略
- 制定简历优化和投递策略

#### 核心知识点

**1. 项目经验包装（STAR法则）**
- STAR法则：
  - Situation（背景）：项目的业务背景和技术背景
  - Task（任务）：你负责的具体任务和目标
  - Action（行动）：你采取的具体技术方案和实施过程
  - Result（结果）：量化的成果（指标提升、效率改善等）
- 项目一：SFT数据构造项目
  - S：公司/团队需要提升大模型在特定领域的回复质量
  - T：负责设计和实施SFT数据构造Pipeline，目标是构建高质量指令微调数据集
  - A：
    - 设计了多层数据构造策略：种子数据人工编写 + Self-Instruct扩充 + 质量筛选
    - 建立了标注质量控制体系：标注指南编写、IAA一致性检查（κ>0.75）、定期校准会议
    - 实现了数据多样性分析工具：按任务类型、难度、领域统计分布，指导数据补充方向
    - 设计了数据配比实验：通过消融实验确定不同任务类型的最优比例
  - R：构建了X万条高质量SFT数据，模型在MT-Bench上提升了X分，人工评估胜率提升X%
  - 面试要点：强调数据质量控制方法、数据多样性分析、配比优化实验
- 项目二：RAG系统项目
  - S：需要构建一个基于企业知识库的问答系统，解决大模型幻觉和知识时效性问题
  - T：负责设计和实现完整的RAG Pipeline
  - A：
    - 文档处理：实现了多格式解析（PDF/Word/HTML）+ 语义分块策略
    - 检索优化：实现了BM25+稠密检索的混合检索 + Cross-Encoder重排序
    - 生成优化：设计了带引用的生成Prompt，减少幻觉
    - 评估体系：用RAGAS框架建立了忠实度、相关性、上下文精确度的自动评估
    - 迭代优化：通过Badcase分析持续改进分块策略和检索参数
  - R：检索召回率从X%提升到Y%，回答忠实度从X%提升到Y%，用户满意度提升X%
  - 面试要点：强调混合检索的设计、评估体系的建立、迭代优化的方法论
- 项目三：代码评估Pipeline项目
  - S：需要系统性评估代码生成模型的能力，支持模型迭代和选型决策
  - T：负责设计和实现代码评估Pipeline
  - A：
    - 评估框架：支持HumanEval/MBPP/自定义题目集，多语言（Python/Java/C++/Go）
    - 沙箱执行：基于Docker的隔离执行环境，支持并行执行和资源限制
    - 指标体系：Pass@k（无偏估计）、编译率、运行时间分布、代码质量评分
    - 持续评估：模型更新自动触发评估，回归检测+趋势监控+告警
    - 数据污染检测：N-gram重叠检测，确保评测数据不在训练集中
  - R：支持了X个模型版本的评估，发现了Y个关键回归问题，评估效率提升X倍
  - 面试要点：强调Pipeline的完整性、Pass@k的理解、持续评估的工程实践
- **为什么重要：** 项目经验是面试中最重要的考察内容，STAR法则确保回答结构清晰、重点突出

**2. 高频面试题梳理**

**Transformer相关：**
- Q：解释自注意力机制的计算过程
  - A：Q/K/V投影 → 缩放点积 QKᵀ/√d_k → softmax归一化 → 加权求和V → 多头拼接+投影
- Q：为什么要除以√d_k？
  - A：d_k大时点积方差为d_k，softmax进入饱和区梯度极小，除以√d_k使方差回到1
- Q：Pre-Norm和Post-Norm的区别？
  - A：Pre-Norm训练更稳定（梯度直接通过残差流动），Post-Norm理论表达能力更强；现代大模型用Pre-Norm
- Q：RoPE位置编码的核心思想？
  - A：将位置信息编码为旋转矩阵作用在Q/K上，使内积只依赖相对位置；支持长度外推

**SFT/RLHF相关：**
- Q：LoRA的原理是什么？
  - A：假设权重更新ΔW是低秩的，分解为BA（B∈R^(d×r), A∈R^(r×k)），冻结原始权重只训练A和B；推理时合并W'=W+BA无额外开销
- Q：RLHF的三个阶段？
  - A：①SFT指令微调 → ②训练奖励模型（偏好数据+Bradley-Terry模型）→ ③PPO优化（最大化奖励+KL约束）
- Q：DPO相比RLHF的优势？
  - A：将奖励建模和策略优化合并为一步，只需2个模型（策略+参考）而非4个，训练更简单稳定
- Q：SFT数据质量和数量哪个更重要？
  - A：质量更重要（LIMA论文：1000条高质量数据即可），但多样性也很关键

**KV Cache与推理优化：**
- Q：KV Cache的原理？
  - A：自回归生成时缓存已计算的K/V，新token只需计算Q与缓存K/V的注意力，每步从O(n²)降到O(n)
- Q：GQA是什么？
  - A：多个Q头共享一组K/V头，减少KV Cache大小；是MHA和MQA的折中方案
- Q：Flash Attention的核心优化？
  - A：利用GPU SRAM/HBM内存层次，分块计算注意力避免存储完整n×n矩阵，IO复杂度从O(n²)降低

**量化相关：**
- Q：GPTQ和AWQ的区别？
  - A：GPTQ基于Hessian逆补偿量化误差（逐层量化）；AWQ基于激活值感知，对显著权重通道做缩放保护再量化，更快且通常效果更好
- Q：INT4量化后模型质量下降多少？
  - A：取决于量化方法，GPTQ/AWQ在4-bit下通常困惑度增加<1%，但在某些推理任务上可能下降更明显

**解码策略：**
- Q：Top-P和Top-K的区别？
  - A：Top-K固定选K个token，Top-P自适应选累积概率达P的最小集合；Top-P更灵活，概率集中时选少概率分散时选多
- Q：Temperature的作用？
  - A：控制softmax分布锐度，T<1更确定T>1更随机，T=0等价Greedy

**评估指标：**
- Q：Pass@k怎么计算？
  - A：生成n个样本c个通过，Pass@k = 1 - C(n-c,k)/C(n,k)；这是无偏估计
- Q：BLEU和ROUGE的区别？
  - A：BLEU衡量精确率（生成的N-gram有多少在参考中），ROUGE衡量召回率（参考的N-gram有多少在生成中）

**安全相关：**
- Q：常见的越狱攻击方式？
  - A：角色扮演、编码绕过、多轮渐进、对抗性后缀、间接注入
- Q：如何防御Prompt注入？
  - A：输入过滤+指令层次化（系统提示优先级高于用户输入）+输出过滤+行为监控

**3. 系统设计题**

**系统设计一：SFT训练Pipeline**
- 需求：设计一个端到端的SFT训练系统
- 架构：
  - 数据层：数据收集→清洗→格式化→质量筛选→版本管理
  - 训练层：模型选择→超参数配置→分布式训练（DeepSpeed/FSDP）→Checkpoint管理
  - 评估层：自动评估（PPL/MT-Bench）→人工评估→安全评估→回归测试
  - 部署层：模型导出→量化→推理服务→A/B测试→灰度发布
- 关键设计决策：
  - LoRA vs 全量微调：根据数据量和计算资源选择
  - 数据配比：通过消融实验确定
  - 训练策略：学习率调度（Warmup+Cosine）、梯度累积、混合精度
- 监控与迭代：训练loss曲线、评估指标趋势、Badcase分析驱动数据补充

**系统设计二：代码评估系统**
- 需求：设计一个支持多语言、多基准的代码生成评估系统
- 架构：
  - 题目管理：题目库（HumanEval/MBPP/自定义）、难度标注、测试用例管理
  - 代码生成：模型API调用、批量生成、参数控制（Temperature/Top-P/采样数）
  - 沙箱执行：Docker容器池、并行执行、资源限制（CPU/内存/时间）、结果收集
  - 指标计算：Pass@k（无偏估计）、编译率、效率对比、代码质量评分
  - 报告系统：可视化Dashboard、趋势图、模型对比、回归检测告警
- 关键设计决策：
  - 沙箱安全：网络隔离、文件系统隔离、进程限制
  - 并行度：根据GPU/CPU资源动态调整
  - 数据污染：定期检测评测数据是否泄露到训练集
- 扩展性：支持新增编程语言、新增基准测试、自定义评估维度

**系统设计三：RAG问答系统**
- 需求：设计一个企业级知识库问答系统
- 架构：
  - 离线索引：文档解析→分块→Embedding→向量数据库存储
  - 在线查询：查询改写→混合检索（BM25+稠密）→重排序→生成→引用标注
  - 评估监控：RAGAS自动评估、用户反馈收集、Badcase分析
- 关键设计决策：
  - 分块策略：根据文档类型选择（结构化文档用结构分块，非结构化用语义分块）
  - 检索策略：混合检索+RRF融合，Top-50召回→Cross-Encoder重排→Top-5输入LLM
  - 缓存策略：热门查询缓存、相似查询复用
  - 更新策略：增量索引更新，文档变更后自动重新索引
- 质量保障：忠实度检测（检测幻觉）、引用验证、答案一致性检查

**系统设计四：内容安全审核系统**
- 需求：设计一个大模型输出的内容安全审核系统
- 架构：
  - 输入审核：关键词过滤→Prompt注入检测→安全分类器
  - 输出审核：规则引擎→安全分类模型→LLM审核（复杂case）→人工审核（边界case）
  - 反馈闭环：安全事件收集→分类归因→规则/模型更新→验证
- 关键设计决策：
  - 延迟 vs 安全：流式输出场景下如何平衡审核延迟和安全性
  - 分级策略：不同严重程度的内容采用不同处理方式（拦截/警告/标记）
  - 误拦处理：建立申诉机制，误拦case用于改进分类器
- 监控告警：安全事件率、误拦率、漏放率、审核延迟

**4. 行为面试准备**
- 常见问题与回答框架：
  - "介绍一个你最有挑战性的项目"
    - 用STAR法则，重点讲技术挑战和你的解决方案
    - 强调你的独立思考和技术决策能力
  - "遇到过什么技术难题？怎么解决的？"
    - 描述具体的技术问题（如训练不收敛、评估指标异常）
    - 强调排查过程（假设→验证→解决）和学到的经验
  - "你对这个岗位的理解是什么？"
    - 字节CQC：SFT数据运营+内容安全+模型评估迭代，核心是通过数据和评估驱动模型质量提升
    - 月之暗面Kimi For Coding：代码能力评估+代码Agent设计+评估Pipeline工程，核心是构建和优化代码智能系统
  - "你的优势是什么？"
    - 结合具体项目经验说明技术能力
    - 强调学习能力和对AI领域的热情
    - 展示系统性思维（不只是写代码，还能设计Pipeline和评估体系）
  - "你有什么问题想问我们？"
    - 团队目前的技术栈和工作流程
    - 模型迭代的频率和评估方法
    - 团队未来的技术方向和挑战
- 面试礼仪：
  - 准时、着装得体
  - 回答简洁有条理，不要过度展开
  - 不会的问题诚实说不会，但可以说你的思考方向
  - 展示对公司和岗位的了解和热情

**5. 简历优化与投递策略**
- 简历优化要点：
  - 项目经验用STAR法则组织，突出量化结果
  - 技术栈关键词与JD匹配：
    - 字节CQC：SFT、RLHF、数据标注、内容安全、模型评估、A/B测试
    - 月之暗面：代码评估、Pass@k、HumanEval、SWE-bench、Agent、Pipeline
  - 教育背景和相关课程
  - 开源贡献或技术博客（如果有）
- 投递策略：
  - 优先内推：找在职员工内推，通过率远高于海投
  - 投递时间：工作日上午投递，HR处理概率更高
  - 多岗位投递：同一公司可以投递多个相关岗位
  - 跟进：投递一周后如果没有回复，可以适当跟进
- 面试准备时间线：
  - 投递前：简历优化、项目经验梳理
  - 一面前：技术基础复习、高频面试题练习
  - 二面前：系统设计题练习、项目深挖准备
  - HR面前：行为面试准备、薪资期望确定

**推荐资源：**
- 《Cracking the Coding Interview》系统设计部分
- 牛客网/力扣面经（搜索目标公司+岗位的面经）
- GitHub上的AI面试题集合
- 目标公司的技术博客和论文（了解技术方向）
- 模拟面试：找朋友或用AI模拟面试练习

**练习任务：**
1. 用STAR法则写出三个项目的完整叙述（每个项目300-500字），练习口头表达（计时3分钟内）
2. 对着镜子或录音练习回答10个高频技术面试题，确保每个问题能在2分钟内清晰回答
3. 画出4个系统设计题的架构图，练习在白板上边画边讲解
4. 准备5个行为面试问题的回答，用STAR法则组织
5. 优化简历，确保与目标岗位JD的关键词匹配度>80%

**自测清单：**
- [ ] 能用STAR法则在3分钟内清晰讲述每个项目
- [ ] 能在2分钟内回答Transformer/LoRA/RLHF/DPO/KV Cache等核心概念
- [ ] 能在白板上画出SFT Pipeline/代码评估系统/RAG系统/安全审核系统的架构图
- [ ] 能回答"你对这个岗位的理解"并展示与岗位的匹配度
- [ ] 简历已优化，项目经验有量化结果，技术关键词与JD匹配
- [ ] 已完成至少3次模拟面试练习
