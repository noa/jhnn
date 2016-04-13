#include "TH.h"
#include "luaT.h"

#include <lua.h>
#include <lauxlib.h>
#include <lualib.h>

static int jhu_THLogSum(lua_State *L) {
  THDoubleTensor *input = (THDoubleTensor *)luaT_checkudata(L, 1,
                                                            "torch.DoubleTensor");
  THDoubleTensor *output = (THDoubleTensor *)luaT_checkudata(L, 2,
                                                             "torch.DoubleTensor");

  double *input_data, *output_data;
  long nframe = 0, dim = 0;
  long t, d;

  if(input->nDimension == 1) {
    nframe = 1;
    dim = input->size[0];
    THAssert( THDoubleTensor_size(output, 0) == 1 );
  }
  else if (input->nDimension == 2) {
    nframe = input->size[0];
    dim = input->size[1];
    THAssert( THDoubleTensor_size(output, 0) == nframe );
  }
  else {
    THArgCheck(0, 2, "vector or matrix input");
  }

  THAssert( THDoubleTensor_isContiguous(input)  );
  THAssert( THDoubleTensor_isContiguous(output) );

  double *input_data0  = THDoubleTensor_data(input);
  double *output_data0 = THDoubleTensor_data(output);

  double logsum;
  double maxInput;
#pragma omp parallel for private(t, d, maxInput, logsum, input_data, output_data)
  for (t = 0; t < nframe; t++) {
    logsum = 0;
    maxInput = -DBL_MAX;
    input_data = input_data0 + dim*t;
    output_data = output_data0 + dim*t;

    for (d = 0; d < dim; d++)
      maxInput = THMax(maxInput, input_data[d]);

    for (d = 0; d < dim; d++)
      logsum += THExpMinusApprox(maxInput-input_data[d]);
    logsum = maxInput + log(logsum);

    output_data0[t] = logsum;
  }

  return 0;
}

static const struct luaL_Reg jhu_THLogSum__ [] = {
  {"logsum", jhu_THLogSum},
  {0, 0}
};

static void jhu_THLogSum_init(lua_State *L) {
  int ret = luaT_pushmetatable(L, "torch.DoubleTensor");
  if(ret == 0) {
    THError("problem pushing metatable");
  }
  luaT_registeratname(L, jhu_THLogSum__, "jhu");
  lua_pop(L, 1);
}
