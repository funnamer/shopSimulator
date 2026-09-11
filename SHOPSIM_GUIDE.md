# ShopSimulator 启动与配置指南

本文档说明如何在当前仓库中启动 ShopSimulator 文本环境，并使用 DeepSeek 作为被评测模型运行 `single_eval`。

所有命令默认从仓库根目录执行：

```bash
cd /Users/funnamer/Desktop/agent-rl/LongHorizonAgentRL
```

## 1. 目录与进程关系

一次完整评测包含两个进程：

```text
DeepSeek API
    ↑
single_eval/agent.py        被评测 Agent，调用模型并生成动作
    ↓ HTTP :5000
shop_env/shop_env/pack_api.py
    ↓
WebAgentTextEnv             执行搜索、点击、选择规格和购买
    ↓
SQLite FTS5 + Original Reward
```

建议使用两个终端：

- 终端 A：运行购物环境服务；
- 终端 B：运行被评测 Agent。

## 2. Conda 环境

当前使用的环境名称是 `shopsim`：

```bash
conda activate shopsim
python --version
```

推荐 Python 3.10。

安装购物环境依赖：

```bash
pip install -r ShopSimulator/shop_env/requirements.txt
```

安装评测客户端额外需要的依赖：

```bash
pip install openai PyYAML requests
```

安装原始 Reward 使用的中文 spaCy 模型：

```bash
pip install \
  https://github.com/explosion/spacy-models/releases/download/zh_core_web_sm-3.7.0/zh_core_web_sm-3.7.0-py3-none-any.whl
```

验证关键依赖：

```bash
python -c "import gym, flask, spacy, openai, yaml; spacy.load('zh_core_web_sm'); print('dependencies OK')"
```

## 3. 商品数据

环境代码默认读取：

```text
ShopSimulator/shop_env/data/items_eval_train.json
```

仓库提供的是压缩文件：

```text
ShopSimulator/shop_env/data/fine_items_eval_train_all.json.gz
```

如果尚未解压，执行：

```bash
cd ShopSimulator/shop_env/data
gzip -dk fine_items_eval_train_all.json.gz
ln -sfn fine_items_eval_train_all.json items_eval_train.json
cd ../../..
```

执行后应存在：

```text
data/fine_items_eval_train_all.json
data/items_eval_train.json -> fine_items_eval_train_all.json
```

## 4. `.env` 配置

仓库根目录使用 `.env` 保存本地密钥和运行参数：

```dotenv
DEEPSEEK_API_KEY=替换为你的APIKey
DEEPSEEK_BASE_URL=https://api.deepseek.com
SHOPSIM_ENV_MAX_NUM=1
```

`.env` 已被 `.gitignore` 忽略，不应提交到版本库。建议设置权限：

```bash
chmod 600 .env
```

字段说明：

| 字段 | 含义 | 推荐值 |
| --- | --- | --- |
| `DEEPSEEK_API_KEY` | DeepSeek API 密钥 | 从 DeepSeek 控制台获取 |
| `DEEPSEEK_BASE_URL` | OpenAI 兼容接口地址 | `https://api.deepseek.com` |
| `SHOPSIM_ENV_MAX_NUM` | 同时初始化的环境槽位数 | 本地测试用 `1` |

`single_eval/agent.py` 和 `pack_api.py` 都会通过 `python-dotenv` 自动读取仓库根目录的 `.env`。如果终端中已经显式设置了同名环境变量，终端中的值优先。

## 5. 启动购物环境

在终端 A 执行：

```bash
cd /Users/funnamer/Desktop/agent-rl/LongHorizonAgentRL
conda activate shopsim
cd ShopSimulator/shop_env/shop_env
python pack_api.py
```

服务默认监听：

```text
http://127.0.0.1:5000/api/shop_agent
```

首次启动时，新的 SQLite FTS5 搜索引擎会自动构建索引：

```text
ShopSimulator/shop_env/search_engine/products.sqlite3
```

首次构建需要读取完整商品数据，因此会比后续启动慢。商品文件发生变化时，索引会自动重建。

可以在另一个终端检查服务：

```bash
curl -X POST http://127.0.0.1:5000/api/shop_agent \
  -H 'Content-Type: application/json' \
  -d '{"action":"release_all"}'
```

正常响应：

```json
{
  "result": {
    "message": "All environments have been initialized"
  }
}
```

停止服务时，在终端 A 按 `Ctrl+C`。

## 6. 运行 DeepSeek Smoke Test

确认购物环境正在运行后，在终端 B 执行：

```bash
cd /Users/funnamer/Desktop/agent-rl/LongHorizonAgentRL
conda activate shopsim
cd ShopSimulator/single_eval

python agent.py \
  --yaml_name configs/standard/deepseek_smoke.yaml
```

`agent.py` 会自动读取根目录 `.env`，因此这里不需要再次执行 `source .env`。

Smoke 配置通过顺序抽样执行 task 0、1、2：

```yaml
task_nums: 1495
task_selection:
  mode: sequential
  sample_size: 3
  seed: 42
output_path: outputs/test_tool3
```

结果保存在：

```text
ShopSimulator/single_eval/outputs/test_tool3/deepseek-flash/0.json
```

如果结果文件已经存在，断点续跑逻辑会认为 task 0 已完成。需要重新测试时，请先将该 JSON 移到备份位置，然后重新执行命令。

## 7. DeepSeek 评测配置说明

当前配置文件：

```text
ShopSimulator/single_eval/configs/standard/deepseek_smoke.yaml
```

完整结构：

```yaml
env_config:
  base_url: http://127.0.0.1:5000

agent_config:
  model_name: deepseek-flash
  source: deepseek
  model_key_env: DEEPSEEK_API_KEY
  base_url_env: DEEPSEEK_BASE_URL
  thinking: disabled
  max_tokens: 512
  system_prompt: |
    你正在进行网上购物……
  task_nums: 1495
  task_selection:
    mode: sequential
    sample_size: 3
    seed: 42
  output_path: outputs/test_tool3
```

### `env_config`

| 字段 | 含义 |
| --- | --- |
| `base_url` | ShopSimulator HTTP 服务地址；代码会追加 `/api/shop_agent` |
| `if_persona` | 可选；设为 `true` 时启用 Persona 模式 |

Standard 模式不设置 `if_persona`：

```yaml
env_config:
  base_url: http://127.0.0.1:5000
```

Persona 模式设置：

```yaml
env_config:
  base_url: http://127.0.0.1:5000
  if_persona: true
```

### `agent_config`

| 字段 | 含义 |
| --- | --- |
| `model_name` | 传给模型服务的模型名称 |
| `source` | API 类型；当前支持 `deepseek`、`openai`、`idealab` |
| `model_key_env` | 从哪个环境变量读取 API Key |
| `base_url_env` | 从哪个环境变量读取模型服务 URL |
| `thinking` | DeepSeek 思考模式；评测建议使用 `disabled` |
| `max_tokens` | 每一步模型最大输出 token 数 |
| `temperature` | 可选；默认 `0.0`，保证评测更稳定 |
| `tool_choice` | 可选；默认 `required`，要求模型每轮调用一个工具 |
| `system_prompt` | 购物 Agent 的动作规范和任务提示 |
| `task_nums` | 候选任务池大小，对应任务 ID `[0, task_nums)` |
| `task_selection.mode` | 可选；`sequential`（默认）或 `random` |
| `task_selection.sample_size` | 可选；从候选池选择多少个任务，默认选择全部 |
| `task_selection.seed` | 随机模式使用的整数 seed，默认 `0` |
| `output_path` | 结果输出根目录，相对于 `single_eval` 当前目录 |

配置中只保存环境变量名，不直接保存 API Key：

```yaml
model_key_env: DEEPSEEK_API_KEY
base_url_env: DEEPSEEK_BASE_URL
```

## 8. 运行更多任务

建议复制 smoke 配置：

```bash
cd ShopSimulator/single_eval
cp configs/standard/deepseek_smoke.yaml \
   configs/standard/deepseek_eval.yaml
```

然后修改。下面表示从 1495 个候选任务中，用 seed 42 可复现地随机抽取 100 个：

```yaml
task_nums: 1495
task_selection:
  mode: random
  sample_size: 100
  seed: 42
output_path: outputs/standard
```

如果需要按任务 ID 顺序运行前 100 个任务：

```yaml
task_nums: 1495
task_selection:
  mode: sequential
  sample_size: 100
  seed: 42  # sequential 模式会忽略 seed
```

相同的 `task_nums`、`sample_size` 和 `seed` 总会得到相同的随机任务 ID。
实际选择结果还会保存到：

```text
<output_path>/<model_name>/run_metadata/task_selection.json
```

断点续跑时会先按相同配置重新生成完整抽样，再跳过其中已经完成的任务，
因此不会因为部分任务已经完成而改变剩余样本。`sequential` 控制任务 ID 的
选择顺序；使用 `--multithread` 时完成顺序仍由各任务耗时决定，严格串行请不要
传 `--multithread`。

单线程运行：

```bash
python agent.py \
  --yaml_name configs/standard/deepseek_eval.yaml
```

多线程运行：

```bash
python agent.py \
  --yaml_name configs/standard/deepseek_eval.yaml \
  --multithread \
  --max_workers 4
```

并发时需要保证环境槽位数不少于 Agent worker 数。例如：

```dotenv
SHOPSIM_ENV_MAX_NUM=4
```

修改 `.env` 后需要重启购物环境服务。

## 9. Standard 与 Persona

### Standard

模型直接看到完整购物要求，主要测试：

- 查询构造；
- 搜索结果筛选；
- 商品比较；
- 规格选择；
- 购买决策。

配置：

```yaml
env_config:
  base_url: http://127.0.0.1:5000
```

### Persona

模型看到简化购物要求，并在 system message 中获得用户画像，需要从画像推断隐藏偏好。

配置：

```yaml
env_config:
  base_url: http://127.0.0.1:5000
  if_persona: true
```

两种模式使用相同的动作循环和原始 Reward，差异只在模型可见的任务信息。

## 10. 动作格式

模型每一步必须且只能发起一次标准 tool call。单 Agent 模式提供：

```json
{"name": "search", "arguments": {"keywords": "商品关键词"}}
```

或者：

```json
{"name": "click", "arguments": {"value": "当前可点击值"}}
```

多方交互模式还提供：

```json
{"name": "ask_shopper", "arguments": {"question": "需要确认的问题"}}
```

`tool_adapter.py` 会将工具调用转换为环境原有的 `search[...]` / `click[...]`
动作。下一步所需的环境 observation 会以带有对应 `tool_call_id` 的
`role: tool` 消息加入模型上下文；终止状态和 Reward 等结构化信息只在程序
内部处理。购买前仍然必须至少选择一个商品规格。

## 11. 输出文件结构

每个任务生成一个 JSON：

```json
{
  "task_id": 0,
  "reward": 1.0,
  "reward_detail": {
    "r_type": 1.0,
    "r_att": 1.0,
    "r_option": 1.0,
    "r_price": true
  },
  "goal": {},
  "purchase": {},
  "conversation": []
}
```

主要字段：

| 字段 | 含义 |
| --- | --- |
| `reward` | ShopSimulator 原始综合 Reward |
| `reward_detail` | 类别、属性、规格、价格等子分数 |
| `goal` | 数据集目标商品及约束 |
| `purchase` | Agent 实际购买的商品与规格 |
| `conversation` | 完整模型交互轨迹 |

## 12. 搜索配置

当前默认搜索实现是：

```text
SQLite FTS5 + 中文 bigram + 多字段 BM25
```

字段权重定义在：

```text
ShopSimulator/shop_env/web_agent_site/engine/search.py
```

默认权重：

```python
{
    "title": 3.0,
    "brand": 2.0,
    "category": 2.0,
    "model": 2.5,
    "attributes": 1.5,
    "options": 1.2,
    "bullets": 0.8,
}
```

搜索索引不包含任务 goal、reward 或答案字段，防止检索阶段泄漏目标答案。

将旧搜索索引移到备份文件后，下一次启动会自动重建：

```bash
mv ShopSimulator/shop_env/search_engine/products.sqlite3 \
   ShopSimulator/shop_env/search_engine/products.sqlite3.bak
```

只有在确认需要重建时才执行此命令。

## 13. 常见问题

### 无法连接 5000 端口

错误示例：

```text
Connection refused: 127.0.0.1:5000
```

处理：先在终端 A 启动 `pack_api.py`，并保持该进程运行。

### 找不到商品数据

错误示例：

```text
No such file or directory: items_eval_train.json
```

处理：按照“商品数据”章节解压文件并创建软链接。

### 找不到中文 spaCy 模型

错误示例：

```text
Can't find model 'zh_core_web_sm'
```

处理：安装 `zh_core_web_sm-3.7.0` wheel，并用下面的命令验证：

```bash
python -c "import spacy; spacy.load('zh_core_web_sm'); print('OK')"
```

### API Key 未读取

确认 `.env` 位于仓库根目录，而不是 `single_eval` 目录：

```text
LongHorizonAgentRL/.env
```

同时确认配置使用：

```yaml
model_key_env: DEEPSEEK_API_KEY
```

### 显示所有任务都已完成

`single_eval` 会根据输出目录中的 `<task_id>.json` 自动断点续跑。更换 `output_path`，或者备份旧结果后再运行。

### DeepSeek 返回空 content

DeepSeek 当前模型默认可能启用思考模式。评测配置中设置：

```yaml
thinking: disabled
```

确保有限的输出 token 用于生成可执行的 `Thought` 和 `Action`。

## 14. 最短启动流程

终端 A：

```bash
cd /Users/funnamer/Desktop/agent-rl/LongHorizonAgentRL
conda activate shopsim
cd ShopSimulator/shop_env/shop_env
python pack_api.py
```

终端 B：

```bash
cd /Users/funnamer/Desktop/agent-rl/LongHorizonAgentRL
conda activate shopsim
cd ShopSimulator/single_eval
python agent.py --yaml_name configs/standard/deepseek_smoke.yaml
```
