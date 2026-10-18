# 资格与待测

## 当前结论

默认关闭，实验性。原分支的45项CPU、组件同分片精确一致、run21服务成功和FA组件窗口15%/20%收益均是**上游固定提交的交接证据**，不是本独立包的新模型实测。上游完整记录见[原交接](upstream-handoff.md)，本包包装检查见[回执](packaging-acceptance.json)。

8请求中3个后续生成序列不同；1K同输入的整块/分片GEMM数值差异已单独观察，不能解释其他形状或全部后续分叉。保留同分片串行/并发逐元素精确检查；不同于未分片OFF的生成序列仅作为诊断。按inference-engine-project开测前冻结的数据集任务指标预算验收，当前默认保留同组Native分数99%；组合共用总预算，不能每项各扣一次。尚无充分任务评分，既不自动通过，也不因Token不同自动判超预算。

## 联合验收基线

Microflow是BetterScale联合扩展MOD。冻结同一版本、模型、设备、图/MTP/缓存及全部非Microflow设置，比较BetterScale＋Microflow OFF与BetterScale＋Microflow ON的边际费用；同宿主Native用于整栈99%总质量预算和总成本参照。独立包不要求先在无BetterScale环境取得单项收益。

## jingyuan后续任务

1. 在固定原共享宿主中从clean wheel完成真实ECPA受管OFF/ON生命周期，验证两个TP rank导入独立包、native摘要、target触发和draft无Microflow op；保持独立编译缓存，退出后核对资源释放与OFF恢复。
2. 原8请求及重复OFF完整复现，分别测1K/2K/4K/8K和未命中回退；保留首token/logprob/全序列、结束原因与实际I/O，不用1K GEMM差异包办所有因果解释。
3. MMLU-Pro、HLE-Verified和FrontierScience适用分项各建匹配Native/ON质量与费用合同；完整任务输出预算与短组件测时分开。记录所有未决/失败/截断与重试，不挑有利窗口。
4. 同组至少3独立匹配启动，报告完整服务窗口和实际Token卡租、输入/输出/共享归属及扰动；组件延迟不得当整模型收益或百万Token降幅。
5. 优先完成上述BetterScale联合OFF/ON验收，再测与Gate/Zero、KV格式等的联合增量收益。拆出通用host capability是后续工程改进，不是当前联合MOD的前置目标；母包内置与独立包重复实现只保留一个。

原Fletcher机器上的`/Users/fletcher-tian/Archives/BetterScale/prefill-microflow-20261018/`仅是作者提供的原件位置，本次没有获得该机器文件，不能声称已独立核完所有原始NPU数据或二进制。仓库只保存已获取分支中的文本交接、源码与公开许可；需要原件时登记转交与摘要，不编造下载地址。
