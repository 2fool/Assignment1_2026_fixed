# 最终提交 Checklist

## A. 代码仓库
- [ ] 最终提交仓库固定为 `Assignment1_2026_fixed`
- [ ] 不再混用旧目录 `Assignment1_2026`
- [ ] `assignment1.ipynb` 为最终 grading 入口
- [ ] `requirements.txt` 完整，干净环境可装
- [ ] 不依赖本地隐藏路径或手动补文件

## B. 本地验证
- [ ] 下载数据成功
- [ ] 预处理成功
- [ ] smoke test 训练成功
- [ ] 至少一组较长训练成功
- [ ] `evaluate()` 成功输出指标

## C. Colab 验收验证
- [ ] 全新 runtime 中 clone 正确仓库
- [ ] `cwd` 指向 `Assignment1_2026_fixed`
- [ ] `sys.path[0]` 指向 `Assignment1_2026_fixed`
- [ ] notebook 从安装到评估可完整运行
- [ ] 未导入旧仓库缓存模块

## D. 实验要求
- [ ] 至少完成 3 个实验
- [ ] 每个实验有清晰假设
- [ ] 每个实验有固定对照与控制变量
- [ ] 每个实验有定量结果（loss / F1 / EM）
- [ ] 每个实验有分析，不只是贴结果

## E. 报告要求
- [ ] 报告使用课程模板（LaTeX 或 Word）
- [ ] 第一页显著放置 Google Drive 仓库公开链接
- [ ] 包含 Introduction
- [ ] 包含完整 Debugging Analysis
- [ ] 包含 Experiments 和结果分析
- [ ] 包含 Conclusion / Discussion
- [ ] 不是只写“修了哪些文件”，而是写 bug、影响和修复 reasoning

## F. Google Drive 提交
- [ ] 完整仓库上传到 Google Drive
- [ ] sharing 权限设为 tutor 可访问 / public link 可访问
- [ ] 链接已实际测试可打开
- [ ] 报告中的链接与最终仓库一致

## G. 最后 24 小时建议
- [ ] 在另一台机器或全新 Colab 复跑 notebook
- [ ] 检查路径、依赖、Drive 权限
- [ ] 提前上传，不要等截止前最后几分钟
- [ ] 保留一份本地压缩备份和 git 备份

