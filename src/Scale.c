#include "TH.h"
#include "luaT.h"

#include <lua.h>
#include <lauxlib.h>
#include <lualib.h>

static int jhu_THScale(lua_State *L) {
  THDoubleTensor *input = (THDoubleTensor *)luaT_checkudata(L, 1,
                                                            "torch.DoubleTensor");

  double *input_data;
  long nframe = 0, dim = 0;
  long t, d;
  
  if(input->nDimension == 1) {
    nframe = 1;
    dim = input->size[0];
  }
  else if (input->nDimension == 2) {
    nframe = input->size[0];
    dim = input->size[1];
  }
  else {
    THArgCheck(0, 2, "vector or matrix input");
  }

  THAssert( THDoubleTensor_isContiguous(input)  );
  double *input_data0  = THDoubleTensor_data(input);
  double sum;
#pragma omp parallel for private(t, d, sum, input_data)
  for (t = 0; t < nframe; t++) {
    sum = 0;
    input_data = input_data0 + dim*t;
    
    for (d = 0; d < dim; d++)
      sum += input_data[d];
    
    for (d = 0; d < dim; d++)
      input_data[d] = input_data[d]/sum;
  }
  
  return 0;
}

static const struct luaL_Reg jhu_THScale__ [] = {
  {"scale", jhu_THScale},
  {0, 0}
};

static void jhu_THScale_init(lua_State *L) {
  int ret = luaT_pushmetatable(L, "torch.DoubleTensor");
  if(ret == 0) {
    THError("problem pushing metatable");
  }
  luaT_registeratname(L, jhu_THScale__, "jhu");
  lua_pop(L, 1);
}
