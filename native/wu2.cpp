// SPDX-License-Identifier: Apache-2.0
#include "wu_kernel.hpp"
extern "C" __global__ __aicore__ void wu2(GM_ADDR k,GM_ADDR v,GM_ADDR beta,GM_ADDR g,GM_ADDR a,GM_ADDR w,GM_ADDR u,GM_ADDR ws,GM_ADDR cu,GM_ADDR ix,int T,int chunks){
 KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);run<2>(k,v,beta,g,a,w,u,ws,cu,ix,T,chunks);
}
