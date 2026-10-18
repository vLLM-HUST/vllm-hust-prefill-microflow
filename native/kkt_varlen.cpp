// SPDX-License-Identifier: Apache-2.0
// Separate kernel binary: keep its 1:1 metadata attached to entry index zero.
#include "kkt_kernel.hpp"
extern "C" __global__ __aicore__ void kkt11_varlen(GM_ADDR k,GM_ADDR beta,GM_ADDR g,GM_ADDR mask,GM_ADDR out,GM_ADDR ws,int T,GM_ADDR cu,GM_ADDR indices,int chunks){
 KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);run<1,true>(k,beta,g,mask,out,ws,T,cu,indices,chunks);
}
