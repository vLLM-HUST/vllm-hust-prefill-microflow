// SPDX-License-Identifier: Apache-2.0
#pragma once
#define CATLASS_ARCH 2201
#include "kernel_operator.h"
#include "catlass/arch/resource.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "block_mmad_pingpong_tla_multi.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/layout/layout.hpp"
#include "tla/tensor.hpp"
using namespace AscendC; using namespace Catlass;
using Tile=tla::Shape<tla::Int<64>,tla::Int<64>,tla::Int<64>>;
using Policy=Gemm::MmadPingpongTlaMulti<Arch::AtlasA2,true,false>;
using WUCopy=Gemm::Tile::PackedTileCopyTla<Arch::AtlasA2,float,layout::RowMajor,float,layout::RowMajor,float,layout::RowMajor>;
using MM=Gemm::Block::BlockMmadTla<Policy,Tile,Tile,float,float,float,void,WUCopy>;
// Per core two independent A/B/C FP32 slots; Vector owns both producer and epilogue.
template<int R> __aicore__ inline void run(GM_ADDR k,GM_ADDR v,GM_ADDR beta,GM_ADDR g,GM_ADDR a,GM_ADDR w,GM_ADDR u,GM_ADDR ws,GM_ADDR cu,GM_ADDR ix,int T,int chunks){
 Arch::Resource<Arch::AtlasA2> res;
 GlobalTensor<bfloat16_t> kg,vg,bg,ag,wg,ug;
 GlobalTensor<float> gg,sg;GlobalTensor<int64_t> cg,ig;
 kg.SetGlobalBuffer((__gm__ bfloat16_t*)k);vg.SetGlobalBuffer((__gm__ bfloat16_t*)v);
 bg.SetGlobalBuffer((__gm__ bfloat16_t*)beta);ag.SetGlobalBuffer((__gm__ bfloat16_t*)a);
 wg.SetGlobalBuffer((__gm__ bfloat16_t*)w);ug.SetGlobalBuffer((__gm__ bfloat16_t*)u);
 gg.SetGlobalBuffer((__gm__ float*)g);sg.SetGlobalBuffer((__gm__ float*)ws);
 cg.SetGlobalBuffer((__gm__ int64_t*)cu);ig.SetGlobalBuffer((__gm__ int64_t*)ix);
 int core=GetBlockIdx();if ASCEND_IS_AIV {core/=R;}
 // Published indices contain a live prefix followed by empty sentinels.
 // Trim it on device so a fixed graph does not multiply zero tiles for idle rows.
 // Every Cube/Vector participant derives the same count before priming slots.
 while(chunks>0){int seq=ig.GetValue((chunks-1)*2);
  if(cg.GetValue(seq+1)>cg.GetValue(seq))break;
  --chunks;
 }
 int cores=GetBlockNum(),tasks=chunks*16*4,count=(tasks+cores-1-core)/cores;
 if ASCEND_IS_AIC {
  MM mm(res);auto l=tla::MakeLayout<float,layout::RowMajor>(64,64);
  for(int it=0;it<count;++it){int slot=it%2,base=(core*2+slot)*12288;
   CrossCoreWaitFlag(slot);
   auto aa=tla::MakeTensor(sg[base],l,Arch::PositionGM{});
   auto bb=tla::MakeTensor(sg[base+4096],l,Arch::PositionGM{});
   auto cc=tla::MakeTensor(sg[base+8192],l,Arch::PositionGM{});
   mm.preSetFlags();mm(aa,bb,cc,GemmCoord{64,64,64});mm.finalWaitFlags();
   CrossCoreSetFlag<2,PIPE_FIX>(2+slot);
  }
 }
 if ASCEND_IS_AIV {
  constexpr int rows=64/R,n=rows*64;int row0=GetSubBlockIdx()*rows;
  auto ah=res.ubBuf.GetBufferByByte<bfloat16_t>(0);
  auto bh=res.ubBuf.GetBufferByByte<bfloat16_t>(8192);
  auto af=res.ubBuf.GetBufferByByte<float>(16384);
  auto bf=res.ubBuf.GetBufferByByte<float>(32768);
  auto beta16=res.ubBuf.GetBufferByByte<bfloat16_t>(49152);
  auto betaf=res.ubBuf.GetBufferByByte<float>(49408);
  auto gate=res.ubBuf.GetBufferByByte<float>(49664);
  auto scale=res.ubBuf.GetBufferByByte<float>(50176);
  auto scratch=res.ubBuf.GetBufferByByte<uint8_t>(66560);
  uint32_t dst[2]={rows,64},src[2]={rows,1};
  // Prime two slots; thereafter drain one result before reusing that slot.
  for(int step=0;step<count+2;++step){
   if(step>=2){int it=step-2,task=core+it*cores,slot=it%2;
    int chunk=task/64,head=task/4%16,part=task%4;
    int seq=ig.GetValue(2*chunk),token=cg.GetValue(seq)+ig.GetValue(2*chunk+1)*64;
    int valid=Min(rows,Max(0,Min(64,int(cg.GetValue(seq+1))-token)-row0));
    CrossCoreWaitFlag(2+slot);
    DataCopy(af,sg[(core*2+slot)*12288+8192+row0*64],n);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    Cast(ah,af,RoundMode::CAST_RINT,n);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    if(valid>0){DataCopyParams p{(uint16_t)valid,4,0,4};
     int off=(head*T+token+row0)*128+(part%2)*64;
     if(part<2)DataCopy(ug[off],ah,p);else DataCopy(wg[off],ah,p);
    }
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
   }
   if(step<count){int task=core+step*cores,slot=step%2;
    int chunk=task/64,head=task/4%16,part=task%4;
    int seq=ig.GetValue(2*chunk),token=cg.GetValue(seq)+ig.GetValue(2*chunk+1)*64;
    int valid=Min(rows,Max(0,Min(64,int(cg.GetValue(seq+1))-token)-row0));
    Duplicate(ah,bfloat16_t(0),n);Duplicate(bh,bfloat16_t(0),n);Duplicate(beta16,bfloat16_t(0),rows);Duplicate(gate,0.f,rows);
    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    if(valid>0){DataCopyPadParams pad{true,0,0,0};
     DataCopyPad(ah,ag[(token+row0)*16*64+head*64],DataCopyParams{(uint16_t)valid,128,1920,0},pad);
     int stride=part<2?16*128:8*128;
     auto inp=part<2?vg:kg;int h=part<2?head:head/2;
     DataCopyPad(bh,inp[(token+row0)*stride+h*128+part%2*64],DataCopyParams{(uint16_t)valid,128,(uint16_t)((stride-64)*2),0},pad);
     DataCopyPad(beta16,bg[head*T+token+row0],DataCopyParams{1,(uint16_t)(valid*2),0,0},pad);
     if(part>=2)DataCopyPad(gate,gg[head*T+token+row0],DataCopyParams{1,(uint16_t)(valid*4),0,0},pad);
    }
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    Cast(af,ah,RoundMode::CAST_NONE,n);Cast(bf,bh,RoundMode::CAST_NONE,n);Cast(betaf,beta16,RoundMode::CAST_NONE,rows);PipeBarrier<PIPE_V>();
    Broadcast<float,2,1>(scale,betaf,dst,src,scratch);PipeBarrier<PIPE_V>();Mul(bf,bf,scale,n);PipeBarrier<PIPE_V>();
    if(part>=2){Exp(gate,gate,rows);PipeBarrier<PIPE_V>();Broadcast<float,2,1>(scale,gate,dst,src,scratch);PipeBarrier<PIPE_V>();Mul(bf,bf,scale,n);}
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    int base=(core*2+slot)*12288;DataCopy(sg[base+row0*64],af,n);DataCopy(sg[base+4096+row0*64],bf,n);
    CrossCoreSetFlag<2,PIPE_MTE3>(slot);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
   }
  }
 }
 PipeBarrier<PIPE_ALL>();
}
