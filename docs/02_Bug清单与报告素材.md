# Bug 清单与报告素材

> 这一份文档可以直接作为报告“debugging analysis”章节的素材库。建议最终报告中把下面条目压缩整理成更自然的叙述。

## 1. 使用方式
每个 bug 建议在报告里写成：
- **位置**
- **类别**：Stage I（管线）或 Stage II（机制）
- **原始错误行为**
- **影响**
- **修复方式**
- **报告可用分析句**

---

## 2. Stage I：基础管线 / 可执行性问题

### 2.1 `TrainTools/train.py`
- 类别：Stage I
- 原始错误行为：`argparse.Namespace({ ... })` 用法错误。
- 影响：训练参数无法正确转成属性访问形式，后续组件读取 `args.xxx` 时存在失效风险。
- 修复方式：改为 `argparse.Namespace(**{...})`。
- 报告句：这是训练主入口的结构性错误，虽然代码表面上能执行到该行，但实际参数对象不满足后续模块的接口预期。

### 2.2 `TrainTools/train_utils.py`
- 类别：Stage I
- 原始错误行为：调用 `loss.item().backward()`。
- 影响：`loss.item()` 返回 Python 标量，无法进行反向传播，训练不能正确更新参数。
- 修复方式：改为 `loss.backward()`，并将梯度裁剪放到优化器 `step()` 前。
- 报告句：该错误直接破坏 backward graph，是导致训练流程逻辑失效的核心 bug 之一。

### 2.3 `EvaluateTools/evaluate.py`
- 类别：Stage I
- 原始错误行为：checkpoint 读取 key 写成 `ckpt["model"]`。
- 影响：与训练保存的 `model_state` 不匹配，评估阶段无法正确加载模型参数。
- 修复方式：改为 `ckpt["model_state"]`。
- 报告句：这是训练—评估接口不一致问题，属于典型的 checkpoint schema 错误。

### 2.4 `Tools/download.py`
- 类别：Stage I
- 原始错误行为：spaCy 默认模型名使用过时的 `en`。
- 影响：在现代环境中依赖安装步骤容易失败，影响 notebook executability。
- 修复方式：改为 `en_core_web_sm`。
- 报告句：该错误不影响模型理论本身，但会直接导致干净环境中的 notebook 无法跑通，因此属于 executability 关键问题。

### 2.5 `assignment1.ipynb`
- 类别：Stage I
- 原始错误行为：原 notebook 配置与当前可运行版本不一致，且训练 scheduler 选择不安全。
- 影响：Tutor 在 Colab 中直接运行时可能遇到路径错配、依赖问题或 checkpoint 保存错误。
- 修复方式：统一 Drive 路径、依赖安装方式、训练参数，并使用可稳定保存的 scheduler。
- 报告句：由于评分方式明确要求 tutors 直接运行 notebook，notebook 本身应被视作提交接口的一部分，而不是可忽略的辅助脚本。

---

## 3. Stage II：机制级实现错误

### 3.1 `Losses/loss.py`
- 类别：Stage II
- 原始错误行为：`F.nll_loss` 的输入和标签顺序写反。
- 影响：loss 计算不符合 PyTorch 接口定义，训练信号错误。
- 修复方式：改为 `F.nll_loss(p1, y1)` 和 `F.nll_loss(p2, y2)`。
- 报告句：该问题会直接破坏监督信号，使模型即使运行也无法遵循正确的优化目标。

### 3.2 `Models/dropout.py`
- 类别：Stage II
- 原始错误行为：inverted dropout 缩放因子除以 `p` 而不是 `1-p`。
- 影响：激活值统计量错误，训练动态被扭曲。
- 修复方式：改为 `x * mask / (1.0 - self.p)`。
- 报告句：这是正则化机制定义错误，不一定立刻报错，但会显著影响训练稳定性。

### 3.3 `Models/Activations/relu.py`
- 类别：Stage II
- 原始错误行为：`clamp(max=0.0)`，实现成了负半轴保留。
- 影响：ReLU 被反向实现，非线性行为完全错误。
- 修复方式：改为 `clamp(min=0.0)`。
- 报告句：激活函数方向颠倒会导致模型表达能力和梯度传播模式偏离理论设计。

### 3.4 `Models/Activations/leakeyReLU.py`
- 类别：Stage II
- 原始错误行为：正负分支条件写反。
- 影响：LeakyReLU 行为错误，负半轴斜率作用到了正半轴。
- 修复方式：改为 `torch.where(x >= 0, x, negative_slope * x)`。
- 报告句：虽然这类错误不一定触发 runtime failure，但它会系统性改变网络的非线性特征。

### 3.5 `Models/embedding.py`
- 类别：Stage II
- 原始错误行为：`transpose(0, 2)` 维度错误；字符嵌入 `permute` 顺序错误。
- 影响：Highway 输入形状错误、字符卷积输入通道语义错位。
- 修复方式：修正为 `transpose(1, 2)` 和 `permute(0, 3, 1, 2)`。
- 报告句：embedding 层是词/字符信息融合入口，维度错位会把后续模块全部带偏。

### 3.6 `Models/conv.py`
- 类别：Stage II
- 原始错误行为：Conv1d unfold 维度错误；Conv2d padding 高宽处理不一致；depthwise/pointwise 顺序错误。
- 影响：卷积核滑窗位置、输出形状与 depthwise separable conv 语义均错误。
- 修复方式：沿正确维度 unfold，修正 2D padding，改为先 depthwise 再 pointwise。
- 报告句：这一组错误既影响形状正确性，也影响卷积模块理论定义，是机制级关键问题。

### 3.7 `Models/qanet.py`
- 类别：Stage II
- 原始错误行为：context word/char embedding 调用对象写反；CQAttention mask 传参次序错误。
- 影响：词向量和字符向量语义错位，注意力 masking 失效。
- 修复方式：恢复正确 embedding 读取顺序，按 `(cmask, qmask)` 传递。
- 报告句：该问题会使模型在最前层就接收到错误模态信息，并破坏后续注意力计算。

### 3.8 `Models/attention.py`
- 类别：Stage II
- 原始错误行为：`A = torch.bmm(Q, S1)` 乘法方向错误。
- 影响：context-query attention 的加权汇聚逻辑错误。
- 修复方式：改为 `A = torch.bmm(S1, Q)`。
- 报告句：错误的 batch matrix multiplication 会使注意力输出语义与论文公式不一致。

### 3.9 `Models/heads.py`
- 类别：Stage II
- 原始错误行为：`torch.cat([M1, M2], dim=0)` 按 batch 拼接；pointer 投影实现不正确。
- 影响：输出头输入张量形状错误，起止位置分布无法正确生成。
- 修复方式：改为按通道拼接，并使用等价的 `einsum` 线性投影。
- 报告句：输出头是 span prediction 的直接接口，形状错误会导致预测分布失真。

### 3.10 `Models/encoder.py`
- 类别：Stage II
- 原始错误行为：位置编码频率张量形状错误；multi-head attention reshape/permute 错误；softmax/mask 轴错误；残差与 norm 索引逻辑错误。
- 影响：self-attention 与 encoder block 的核心数学机制失效。
- 修复方式：重写位置编码广播逻辑，修正 MHA 张量布局，修正 residual/norm 顺序。
- 报告句：这是整个模型的核心机制修复，若该模块错误，即使外围流程能运行，模型也很难表现出合理学习行为。

### 3.11 `Models/Normalizations/layernorm.py`
- 类别：Stage II
- 原始错误行为：`keepdim=False` 造成广播错误；仿射变换写成 `x_norm * bias + weight`。
- 影响：LayerNorm 数学定义错误，并可能触发维度不匹配。
- 修复方式：使用 `keepdim=True` 并改为 `x_norm * weight + bias`。
- 报告句：该问题同时影响数值稳定性和实现正确性，是 normalization 的典型错误案例。

### 3.12 `Models/Normalizations/groupnorm.py`
- 类别：Stage II
- 原始错误行为：分组 reshape 次序错误。
- 影响：group normalization 实际没有按照预期 group 进行。
- 修复方式：改为 `[B, G, C//G, ...]` 布局。
- 报告句：GroupNorm 的关键在于正确分组，reshape 次序错误会完全改变被标准化的统计单元。

### 3.13 `Optimizers/sgd.py`
- 类别：Stage II
- 原始错误行为：weight decay 符号错误。
- 影响：L2 正则方向错误，参数更新偏离理论形式。
- 修复方式：改为 `grad.add(p, alpha=wd)`。
- 报告句：该 bug 不一定导致崩溃，但会使“正则化”变成反向作用。

### 3.14 `Optimizers/sgd_momentum.py`
- 类别：Stage II
- 原始错误行为：velocity state key 不一致；动量更新式符号错误。
- 影响：momentum buffer 无法正确维护，更新规则失真。
- 修复方式：统一使用 `velocity`，并改为 `v = mu*v + grad`。
- 报告句：优化器状态 bug 往往隐蔽，但会强烈影响收敛行为和实验可解释性。

### 3.15 `Optimizers/adam.py`
- 类别：Stage II
- 原始错误行为：state key 使用错误；二阶矩没有平方；bias correction 公式错误；weight decay 符号错误。
- 影响：Adam 退化成错误的近似算法，训练不稳定或无效。
- 修复方式：改为标准 Adam 更新。
- 报告句：由于 Adam 是实验中常见对照组，修复其数学定义对于后续实验结论有效性非常重要。

### 3.16 `Schedulers/lambda_scheduler.py`
- 类别：Stage II
- 原始错误行为：学习率写成 `base_lr + factor`。
- 影响：scheduler 输出不符合乘性调度定义。
- 修复方式：改为 `base_lr * factor`。
- 报告句：学习率调度器虽小，但其错误会系统性扭曲所有训练实验。

### 3.17 `Schedulers/step_scheduler.py`
- 类别：Stage II
- 原始错误行为：衰减公式写成 `base_lr * gamma * k` 而不是指数衰减。
- 影响：step scheduler 行为与定义不一致。
- 修复方式：改为 `base_lr * (gamma ** k)`。
- 报告句：这类公式级错误会直接影响优化路径，是做 scheduler 实验时必须修复的前提。

### 3.18 `Schedulers/cosine_scheduler.py`
- 类别：Stage II
- 原始错误行为：`math.PI` 不存在，且余弦公式缺少 0.5 系数。
- 影响：运行报错或输出曲线错误。
- 修复方式：改为标准 cosine annealing 公式。
- 报告句：该错误同时影响可执行性和机制正确性，属于典型的“实现细节导致理论失真”。

### 3.19 `Schedulers/scheduler.py`
- 类别：Stage II / 可执行性补充
- 原始错误行为：lambda scheduler 用匿名 `lambda`，checkpoint 保存不可 pickle。
- 影响：训练能跑但保存 checkpoint 时失败。
- 修复方式：将匿名 lambda 替换为命名函数，且 notebook 默认使用 `step` scheduler。
- 报告句：这是一个很好的案例，说明某些 bug 只有在完整训练—保存链路上才会暴露出来。

### 3.20 `Models/Initializations/*`
- 类别：Stage II
- 原始错误行为：Kaiming/Xavier 方差公式错误。
- 影响：参数初始化尺度不合理，影响收敛速度和稳定性。
- 修复方式：恢复标准公式。
- 报告句：初始化虽然位于训练开始前，但它深刻影响优化早期动态，因此属于机制修复的重要组成部分。

### 3.21 `EvaluateTools/eval_utils.py`
- 类别：Stage II / Stage I 交叉
- 原始错误行为：`argmax` 沿错误维度计算。
- 影响：预测的 start/end 位置不对应样本序列维，评估指标失真。
- 修复方式：改为沿序列维 `dim=1` 求最大。
- 报告句：这类错误特别隐蔽，因为评估代码能运行，但报告出来的指标没有真实性。

---

## 4. 报告写法建议

### 4.1 Debugging Analysis 章节结构建议
1. **先按阶段分**：Stage I 与 Stage II。
2. **再按模块分组**：training/eval、model core、optimizer/scheduler、notebook/dependency。
3. 每组不要只列 patch，要说明：
   - 该 bug 为什么错；
   - 它会导致什么现象；
   - 你如何判断并修复；
   - 修复后训练/评估出现了哪些改善。

### 4.2 不要只写“修了哪些文件”
PDF 要的是：
- 系统性、完整性；
- 原因分析；
- 对训练行为的影响；
- 修复后的理论一致性。

因此报告建议把“文件修改”翻译成“机制问题”。

