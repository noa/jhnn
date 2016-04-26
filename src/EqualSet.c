#include "TH.h"
#include "luaT.h"

#include <lua.h>
#include <lauxlib.h>
#include <lualib.h>

static int jhu_THEqualSet(lua_State *L) {
    int narg = lua_gettop(L);
    if (narg != 4) {
        THError("expecting exactly 3 arguments");
    }

    THDoubleTensor *output = (THDoubleTensor *)luaT_checkudata(L, 1,
                                                              "torch.DoubleTensor");
    THDoubleTensor *input = (THDoubleTensor *)luaT_checkudata(L, 2,
                                                              "torch.DoubleTensor");


    int size1 = THDoubleTensor_size(input, 0);
    if(!(size1 == THDoubleTensor_size(output, 0))) THError("size mismatch");
    if(!(THDoubleTensor_nDimension(output) ==
         THDoubleTensor_nDimension(input)
           )) THError("dim mismatch");

    double *input_data = THDoubleTensor_data(input);
    double *output_data = THDoubleTensor_data(output);

    int val1 = lua_tonumber(L, 3);
    int val2 = lua_tonumber(L, 4);

    int t;

#pragma omp parallel for
    for(t=0;t<size1;++t) {
        if((int)input_data[t] == val1) {
            output_data[t] = input_data[t];
        } else {
            output_data[t] = (double)val2;
        }
    }

    return 0;
}

static const struct luaL_Reg jhu_THEqualSet__ [] = {
    {"equalset", jhu_THEqualSet},
    {0, 0}
};

static void jhu_THEqualSet_init(lua_State *L) {
    int ret = luaT_pushmetatable(L, "torch.DoubleTensor");
    if(ret == 0) {
        THError("problem pushing metatable");
    }
    luaT_registeratname(L, jhu_THEqualSet__, "jhu");
    lua_pop(L, 1);
}
