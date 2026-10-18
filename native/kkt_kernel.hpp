// SPDX-License-Identifier: Apache-2.0
#pragma once
// 1:1 varlen KKT with empty publication-sentinel handling.
#define CATLASS_ARCH 2201
#include "kernel_operator.h"
#include "catlass/arch/resource.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "block_mmad_pingpong_tla_multi.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/layout/layout.hpp"
#include "tla/tensor.hpp"
using namespace AscendC;
using namespace Catlass;
using Tile=tla::Shape<tla::Int<64>,tla::Int<64>,tla::Int<128>>;
using Policy=Gemm::MmadPingpongTlaMulti<Arch::AtlasA2,true,false>;
using TileCopy=Gemm::Tile::PackedTileCopyTla<Arch::AtlasA2,bfloat16_t,layout::RowMajor,bfloat16_t,layout::ColumnMajor,float,layout::RowMajor>;
using MM=Gemm::Block::BlockMmadTla<Policy,Tile,Tile,bfloat16_t,bfloat16_t,float,void,TileCopy>;

template<int R, bool Varlen=false> __aicore__ inline void run(GM_ADDR k,GM_ADDR beta,GM_ADDR g,GM_ADDR mask,GM_ADDR out,GM_ADDR workspace,int T,GM_ADDR cu=nullptr,GM_ADDR indices=nullptr,int chunks=0){
 Arch::Resource<Arch::AtlasA2> res;
 GlobalTensor<bfloat16_t> kg,bg;GlobalTensor<float> gg,mg,og,wg;
 kg.SetGlobalBuffer((__gm__ bfloat16_t*)k); bg.SetGlobalBuffer((__gm__ bfloat16_t*)beta);
 gg.SetGlobalBuffer((__gm__ float*)g); mg.SetGlobalBuffer((__gm__ float*)mask);
 og.SetGlobalBuffer((__gm__ float*)out);wg.SetGlobalBuffer((__gm__ float*)workspace);
 int core=GetBlockIdx();if ASCEND_IS_AIV {core/=R;}
 GlobalTensor<int64_t> cug,ig;
 if constexpr(Varlen){cug.SetGlobalBuffer((__gm__ int64_t*)cu);ig.SetGlobalBuffer((__gm__ int64_t*)indices);}
 if constexpr(Varlen){
  // Keep all participants on the same live task count, before flag priming.
  while(chunks>0){int seq=ig.GetValue((chunks-1)*2);
   if(cug.GetValue(seq+1)>cug.GetValue(seq))break;
   --chunks;
  }
 }
 int tasks=(Varlen?chunks:T/64)*16;int cores=GetBlockNum();
 if ASCEND_IS_AIC {
  MM mm(res);
  auto kl=tla::MakeLayout<bfloat16_t,layout::RowMajor>(64,8*128);
  auto kt=tla::MakeLayout<bfloat16_t,layout::ColumnMajor>(8*128,64);
  auto ol=tla::MakeLayout<float,layout::RowMajor>(64,64);
  int iteration=0;
  for(int task=core;task<tasks;task+=cores,++iteration){
   int slot=iteration%2;CrossCoreWaitFlag(2+slot);
   int token=task/16*64,valid=64;
   if constexpr(Varlen){
    int seq=ig.GetValue(task/16*2),chunk=ig.GetValue(task/16*2+1);
    token=cug.GetValue(seq)+chunk*64;valid=Min(64,int(cug.GetValue(seq+1))-token);
   }
   int offset=(token*8+task%16/2)*128;
   auto a=tla::MakeTensor(kg[offset],kl,Arch::PositionGM{});
   auto b=tla::MakeTensor(kg[offset],kt,Arch::PositionGM{});
   auto c=tla::MakeTensor(wg[(core*2+slot)*4096],ol,Arch::PositionGM{});
   auto aa=tla::GetTile(a,tla::MakeCoord(0,0),tla::MakeShape(valid,128));
   auto bb=tla::GetTile(b,tla::MakeCoord(0,0),tla::MakeShape(128,valid));
   auto cc=tla::GetTile(c,tla::MakeCoord(0,0),tla::MakeShape(valid,valid));
   if(valid>0){mm.preSetFlags();mm(aa,bb,cc,GemmCoord{uint32_t(valid),uint32_t(valid),128});mm.finalWaitFlags();}
   CrossCoreSetFlag<2,PIPE_FIX>(slot);
  }
  CrossCoreWaitFlag(2);CrossCoreWaitFlag(3);
 }
 if ASCEND_IS_AIV {
  constexpr int rows=64/R, n=rows*64;int row0=GetSubBlockIdx()*rows;
  auto av=res.ubBuf.GetBufferByByte<float>(0);
  auto gv=res.ubBuf.GetBufferByByte<float>(16384);
  auto bv=res.ubBuf.GetBufferByByte<bfloat16_t>(16640);
  auto bf=res.ubBuf.GetBufferByByte<float>(16896);
  auto left=res.ubBuf.GetBufferByByte<float>(17408);
  auto right=res.ubBuf.GetBufferByByte<float>(33792);
  auto mv=res.ubBuf.GetBufferByByte<float>(50176);
  auto scratch=res.ubBuf.GetBufferByByte<uint8_t>(66560);
  DataCopy(mv,mg[row0*64],n);SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
  uint32_t dst[2]={rows,64}, srcr[2]={1,64},srcl[2]={rows,1};
  CrossCoreSetFlag<2,PIPE_MTE2>(2);CrossCoreSetFlag<2,PIPE_MTE2>(3);
  int iteration=0;
  for(int task=core;task<tasks;task+=cores,++iteration){
   int slot=iteration%2;CrossCoreWaitFlag(slot);
   int token=task/16*64,valid=64;
   if constexpr(Varlen){
    static_assert(R==1,"Varlen is qualified only for 1:1");
    int seq=ig.GetValue(task/16*2),chunk=ig.GetValue(task/16*2+1);
    token=cug.GetValue(seq)+chunk*64;valid=Min(64,int(cug.GetValue(seq+1))-token);
    // Never read padding from adjacent sequences or stale workspace columns.
    if(valid<64){Duplicate(av,0.0f,n);Duplicate(gv,0.0f,64);Duplicate(bv,bfloat16_t(0),64);}
    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    DataCopyParams ap{(uint16_t)valid,(uint16_t)(valid*4),(uint16_t)((64-valid)*4),(uint16_t)((64-((valid+7)/8*8))/8)};
    DataCopyPadParams pad{true,0,0,0};
    if(valid>0)DataCopyPad(av,wg[(core*2+slot)*4096],ap,pad);
    DataCopyParams gp{1,(uint16_t)(valid*4),0,0},bp{1,(uint16_t)(valid*2),0,0};
    if(valid>0)DataCopyPad(gv,gg[task%16*T+token],gp,pad);
    if(valid>0)DataCopyPad(bv,bg[task%16*T+token],bp,pad);
   }else{
    DataCopy(av,wg[(core*2+slot)*4096+row0*64],n);
    DataCopy(gv,gg[task%16*T+token],64);
    DataCopy(bv,bg[task%16*T+token+row0],rows);
   }
   SetFlag<HardEvent::MTE2_V>(EVENT_ID0);WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
   CrossCoreSetFlag<2,PIPE_MTE2>(2+slot);
   Broadcast<float,2,0>(right,gv,dst,srcr,scratch);
   Broadcast<float,2,1>(left,gv[row0],dst,srcl,scratch);PipeBarrier<PIPE_V>();
   Sub(left,left,right,n);PipeBarrier<PIPE_V>();
   Mins(left,left,0.0f,n);PipeBarrier<PIPE_V>();Exp(left,left,n);PipeBarrier<PIPE_V>();
   Mul(av,av,left,n);Cast(bf,bv,RoundMode::CAST_NONE,rows);PipeBarrier<PIPE_V>();
   Broadcast<float,2,1>(right,bf,dst,srcl,scratch);PipeBarrier<PIPE_V>();
   Mul(av,av,right,n);PipeBarrier<PIPE_V>();Mul(av,av,mv,n);
   SetFlag<HardEvent::V_MTE3>(EVENT_ID0);WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
   DataCopyParams cp{(uint16_t)(Varlen?valid:rows),8,0,120};
   if(valid>0)DataCopy(og[(token+row0)*16*64+task%16*64],av,cp);
   SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
   if constexpr(Varlen){SetFlag<HardEvent::MTE3_V>(EVENT_ID0);WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);}
  }
 }
 PipeBarrier<PIPE_ALL>();
}
