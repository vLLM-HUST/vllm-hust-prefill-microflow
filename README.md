# Prefill Microflow：GDN + FA 预填充分片流水线

负责人：**jingyuan / 田景远（[@CubeLander](https://github.com/CubeLander)）**。

从[BetterScale `lumi/cache-transfer-profile`](https://github.com/vLLM-HUST/BetterScale/tree/lumi/cache-transfer-profile)固定提交`dcac3fa0ebe96cfaa443c0295458ad66822c319a`拆出的独立源码、native构建与ECPA Manifest 0.3 MOD。**实验性、默认关闭；当前没有整模型质量/降本资格。**

当前独立的是包、执行叶子、ECPA选择与构建交付；GDN仍依赖固定BetterScale共享宿主的状态发布机制，不能直接装在任意原生vLLM上。包不包含整个BetterScale，也不自动安装、启用或替换母包；该共享宿主须由操作者单独准备并通过原有准入。所有已安装BetterScale Python/JSON源文件逐字节校验，已加载的内置Microflow会被拒绝，避免重复生效。

## 范围与当前证据

| 项目 | 范围 |
|---|---|
| 模型与并行 | Qwen3.5-35B-A3B、BF16、TP2 |
| GDN | 4K，两段流水线；保留请求相对64-token边界与完整状态 |
| FA | 1K→512×2，2K→1K×2，4K→1K×4，8K→2K×4；其余形状回退原序列 |
| 排除 | draft、MLP、跨层AllReduce；FA attention/KV scatter仍归原consumer |
| native依赖 | 独立构建KKT/WU两个aarch64库；原GDN H/O等共享宿主依赖仍须满足 |
| 原分支已验证 | 上游报告45项CPU测试；同分片串行/并发组件逐元素一致；原整模型能编译、捕获graph并完成服务 |
| 尚待验收 | 原整模型8请求有3个后续序列不同；1K GEMM形状差异已复现，不能解释全部分叉。尚无足够任务级精度结果 |
| 性能边界 | 原组件窗口FA 4K/8K约15%/20%延迟下降，GDN基本持平；不是本独立包的新实测，也不是整模型收益 |

独立包自己的CPU/安装与Manager检查见[包装验收](docs/packaging-acceptance.json)，不继承上游45项为本包的新测试次数。[完整验收边界与待办](docs/qualification.md) · [来源与逐文件差异](provenance.json)。

## 安装与ECPA生命周期

在隔离环境构建/安装本包；不要修改其他线程的已安装运行时：

```sh
export VLLM_HUST_EXT_CONFIG="$TASK_MANAGER_CONFIG"
python -m build
python -m pip install --no-deps dist/vllm_hust_prefill_microflow-0.1.0.dev0-py3-none-any.whl
vllm-hust-ext extension list
vllm-hust-ext extension inspect org.vllm-hust.prefill-microflow
vllm-hust-ext extension validate org.vllm-hust.prefill-microflow
```

先按[native构建步骤](docs/native-build.md)生成库及内容清单。使用`BETTERSCALE_MICROFLOW_NATIVE_DIR`和`HUST_PREFILL_MICROFLOW_NATIVE_MANIFEST`显式指定；无库、未知版本、源字节不符或共享编译缓存均拒绝启动。匹配的原始宿主版本记录在包内`host-contract.json`；学校版/Core0.28等新宿主须另行移植验证，不能直接扩大范围。

```sh
python -m hust_prefill_microflow environment --cache-root "$TASK_CACHE_ROOT" --arm on
```

该命令只打印环境JSON、不启动worker。将结果合入任务私有部署环境，补上native两个路径，然后执行：

```sh
python -m hust_prefill_microflow check
vllm-hust-ext extension enable org.vllm-hust.prefill-microflow
vllm-hust-ext extension check org.vllm-hust.prefill-microflow
vllm-hust-ext extension plan org.vllm-hust.prefill-microflow
vllm-hust-ext run --dry-run -- python -m vllm.entrypoints.openai.api_server --model "$MODEL_PATH"
```

Manager传递启用意图和插件选择，正式运行须使用原已准入共享宿主的完整启动配置/Worker，而不是只复制以上dry-run示例的模型参数。正式启动前仍需设备租约、各rank实际生效见证及冻结数据/评分/费用合同。

回滚：

```sh
vllm-hust-ext extension disable org.vllm-hust.prefill-microflow
python -m hust_prefill_microflow environment --cache-root "$TASK_CACHE_ROOT" --arm off
```

将OFF环境用于**重启后的全新worker**。OFF/ON的vLLM、Triton、Inductor缓存目录分离；不支持在已有worker中热切换。需要移除时先禁用、退出原worker，再`forget`与`pip uninstall vllm-hust-prefill-microflow`。安装、发现、enable/plan、注册、真实作用、质量及性能是不同阶段；本包不伪造`runtime_effective`。

## 组合关系

这是BetterScale内含Microflow的独立候选，母包/其他分支若已加载同一实现只能执行和计益一次。Manifest独占声明Microflow发布入口、FA预填充调度和GDN预填充调度；与其他改写同路径的MOD先审计，未声明冲突不代表兼容。当前尚无Gate/Zero/量化KV等完整组合降本资格，不能直接接入已有成本阶梯。

代码布局、显式import bridge和未完成的原生宿主移植见[架构](docs/architecture.md)。第三方许可及原作者信息保留在`LICENSE`、`licenses/`和源码头；本仓库不包含权重、native二进制或Fletcher本地原始证据包。
