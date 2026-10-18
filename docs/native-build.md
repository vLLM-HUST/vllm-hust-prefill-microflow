# 单独构建KKT / WU

仅在已准入Ascend910B2/CANN9.1环境构建，不修改正在测量的共享安装，不在推理时编译。两个库不能替代原共享宿主的GDN H/O或ABI库。

```sh
cmake -S native -B "$TASK_NATIVE_BUILD" \
  -DASCEND_CANN_PACKAGE_PATH="$QUALIFIED_CANN_ROOT" \
  -DCATLASS_ROOT="$PINNED_CATLASS_ROOT"
cmake --build "$TASK_NATIVE_BUILD" -j2
```

使用明确固定的CATLASS提交`41bf90da655bba3c66d0acd7e00abe33960ecfd6`。库输出位置由该CANN构建决定，显式填写实际包含`libbs_microflow_kkt.so`与`libbs_microflow_wu.so`的目录，不假设发行路径。

```sh
python tools/native_manifest.py --native-dir "$TASK_NATIVE_LIB_DIR" --output "$TASK_NATIVE_MANIFEST"
export BETTERSCALE_MICROFLOW_NATIVE_DIR="$TASK_NATIVE_LIB_DIR"
export HUST_PREFILL_MICROFLOW_NATIVE_MANIFEST="$TASK_NATIVE_MANIFEST"
```

清单只是操作者声明并绑定本次构建内容，不授予设备或数值资格。包在启动前验证两库的ELF64/aarch64及SHA，不自动接受其他构建/下载或进行运行时编译。CANN、CATLASS及所含天津大学代码的许可/源码头保留，native二进制不要提交Git。
