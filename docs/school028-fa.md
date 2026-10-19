# 学校0.28 FA分片适配

本适配只包含FA预填充投影、归一化与RoPE分片调度。GDN保持原宿主路径，
不使用KKT/WU native库；draft保持原路径。它不是完整GDN+FA Microflow资格。

实验Bundle为`org.vllm-hust.prefill-microflow-school028-fa`，默认关闭，与原Bundle互斥。
原`host-contract.json`和185文件来源约束不变；新Bundle使用独立的逐文件宿主合同，
只接纳已冻结学校版本及BetterScale源文件，不允许通过修改manifest版本范围自动升级。

学校State配置按`betterscale.qwen35_worker.Worker`精确名称识别。
显式注册后，仅在当前进程将该Python导出指向继承原worker的适配类，
不修改site-packages文件。适配类在加载真实权重前安装FA函数，
保留原初始化、State缓存、调度与MTP生命周期。

设备准入保持四卡排他租约、TP2执行，固定E24/R36、每卡20 GiB状态与8 GiB host缓存。
测试使用原MMLU-Pro 200请求、输入哈希不变、无限到达率和out256。
ON/OFF使用独立编译缓存和新worker；同组比较须保留全部排队时间和退出回执。

资格检查使用真实FA权重，分别对1K/2K/4K比较同分片串行与并发执行的五个输出，
必须逐元素一致。该检查不是任务质量评分，也不证明与未分片GEMM逐元素一致。
服务启动后必须取得两rank加载回执、十个target FA层的分片图捕获记录，
并在请求窗口内记录对应shape的真实图重放。仅加载插件或仅捕获图均不算请求阶段生效。

`tools/verify_school_run.py`核验运行来源、请求总数、token总数、控制记录与释放状态。
吞吐、TTFT和TPOT分别比较；未经多次匹配运行和任务质量预算验收，不升级完整MOD资格。

回滚需停止该实例、通过Manager禁用新Bundle，再启动原worker。
不支持在同一进程切换开关。完整GDN移植仍需发布双slab元数据和GDN output writer，
不能仅放宽原宿主版本限制。
