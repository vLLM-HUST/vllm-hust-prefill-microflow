# 独立包边界

10个`microflow*.py`执行叶子位于`hust_prefill_microflow.runtime`，native源码位于`native/`。数学和调度代码保持原来源；只将少数共享支持模块import改为显式BetterScale宿主引用，并将native目录改为必须显式指定。逐文件原始/提取SHA和差异类别见`provenance.json`。

ECPA发现仅导入无副作用的包根。通过Manager选择插件后，`register()`检查开关、固定宿主版本、185个BetterScale Python/JSON文件内容、KKT/WU的ELF架构/内容摘要和独立编译缓存；随后安装一个只处理10个确切模块名的`sys.meta_path`桥。该桥将宿主既有`.microflow*`发布入口导向独立包，其他导入不变，不覆写任何安装文件。数值模块仅在宿主worker真正导入这些入口时加载；这是真实、显式声明的运行时import hook，不是原生vLLM新增的稳定API。

入口调用顺序、宿主元数据/状态发布、GDN output writer和现有H/O库仍归固定共享宿主。已提前加载内置Microflow时拒绝安装；未知宿主/库直接失败，不回退到“表面启用而无作用”。同worker开关/缓存变化要求重启。卸载Python包不会回滚活进程的patch，必须先关闭worker。

当前没有证明本独立包装已在NPU上完成ECPA受管生命周期；上游run21服务成功属于原始集成，不继承为本包资格。下一步为固定原宿主的clean-wheel Manager→两rank→请求→停止/释放→OFF回滚验收。脱离BetterScale共享宿主运行还需独立移植状态发布与GDN writer；这里没有复制整套母包伪装成无依赖子MOD。

## 联合MOD定位

Microflow与BetterScale联合使用，独立的是交付和开关。脱离BetterScale不是当前验收目标；固定BetterScale宿主的OFF→ON为边际成本对照，匹配Native另作整栈总质量/成本参照。内置与提取实现择一，不重复执行。
