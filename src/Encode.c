#include "TH.h"
#include "luaT.h"

#include <lua.h>
#include <lauxlib.h>
#include <lualib.h>

static int jhu_LongEncode(lua_State *L) {
    int narg = lua_gettop(L);
    if (narg != 4) {
        THError("expecting exactly 4 arguments");
    }

    THLongTensor *input0 = (THLongTensor *)luaT_checkudata(L, 1,
                                                           "torch.LongTensor");
    THLongTensor *input1 = (THLongTensor *)luaT_checkudata(L, 2,
                                                           "torch.LongTensor");
    THLongTensor *output = (THLongTensor *)luaT_checkudata(L, 3,
                                                           "torch.LongTensor");
    long N = lua_tonumber(L, 4);

    long *input_data0 = THLongTensor_data(input0);
    long *input_data1 = THLongTensor_data(input1);
    long *output_data = THLongTensor_data(output);

    int nelem1 = THLongTensor_size(input0, 0);
    int nelem2 = THLongTensor_size(input1, 0);

    int ndim1 = THLongTensor_nDimension(input0);
    int ndim2 = THLongTensor_nDimension(input1);

    if (ndim1 != ndim2)   THError("dim mismatch");
    if (nelem1 != nelem2) THError("size mismatch");

#pragma omp for
    for(int k=0; k<nelem1; k++) {
        output_data[k] = input_data0[k] + (input_data1[k]-1) * N;
    }

    return 0;
}

static int jhu_LongDecode(lua_State *L) {
    int narg = lua_gettop(L);
    if (narg != 4) {
        THError("expecting exactly 4 arguments");
    }

    THLongTensor *input = (THLongTensor *)luaT_checkudata(L, 1,
                                                          "torch.LongTensor");
    THLongTensor *output0 = (THLongTensor *)luaT_checkudata(L, 2,
                                                            "torch.LongTensor");
    THLongTensor *output1 = (THLongTensor *)luaT_checkudata(L, 3,
                                                            "torch.LongTensor");
    long N = lua_tonumber(L, 4);

    long *input_data = THLongTensor_data(input);
    long *output_data0 = THLongTensor_data(output0);
    long *output_data1 = THLongTensor_data(output1);

    int nelem0 = THLongTensor_size(input, 0);
    int nelem1 = THLongTensor_size(output0, 0);
    int nelem2 = THLongTensor_size(output1, 0);

    int ndim1 = THLongTensor_nDimension(output0);
    int ndim2 = THLongTensor_nDimension(output1);

    if (ndim1 != ndim2)   THError("dim mismatch");
    if (nelem0 != nelem1) THError("size mismatch");
    if (nelem1 != nelem2) THError("size mismatch");

#pragma omp for
    for(int k=0; k<nelem1; k++) {
        output_data0[k] = ((input_data[k]-1) % N) + 1;
        output_data1[k] = ((input_data[k]-1) / N) + 1;
    }

    return 0;
}

static int jhu_DoubleEncode(lua_State *L) {
    int narg = lua_gettop(L);
    if (narg != 4) {
        THError("expecting exactly 4 arguments");
    }

    THDoubleTensor *input0 = (THDoubleTensor *)luaT_checkudata(L, 1,
                                                               "torch.DoubleTensor");
    THDoubleTensor *input1 = (THDoubleTensor *)luaT_checkudata(L, 2,
                                                               "torch.DoubleTensor");
    THDoubleTensor *output = (THDoubleTensor *)luaT_checkudata(L, 3,
                                                               "torch.DoubleTensor");
    long N = lua_tonumber(L, 4);

    double *input_data0 = THDoubleTensor_data(input0);
    double *input_data1 = THDoubleTensor_data(input1);
    double *output_data = THDoubleTensor_data(output);

    int nelem1 = THDoubleTensor_size(input0, 0);
    int nelem2 = THDoubleTensor_size(input1, 0);

    int ndim1 = THDoubleTensor_nDimension(input0);
    int ndim2 = THDoubleTensor_nDimension(input1);

    if (ndim1 != ndim2)   THError("dim mismatch");
    if (nelem1 != nelem2) THError("size mismatch");

#pragma omp for
    for(int k=0; k<nelem1; k++) {
        output_data[k] = input_data0[k] + ((long)input_data1[k]-1) * N;
    }

    return 0;
}

static int jhu_DoubleDecode(lua_State *L) {
    int narg = lua_gettop(L);
    if (narg != 4) {
        THError("expecting exactly 4 arguments");
    }

    THDoubleTensor *input = (THDoubleTensor *)luaT_checkudata(L, 1,
                                                          "torch.DoubleTensor");
    THDoubleTensor *output0 = (THDoubleTensor *)luaT_checkudata(L, 2,
                                                            "torch.DoubleTensor");
    THDoubleTensor *output1 = (THDoubleTensor *)luaT_checkudata(L, 3,
                                                            "torch.DoubleTensor");
    long N = lua_tonumber(L, 4);

    double *input_data = THDoubleTensor_data(input);
    double *output_data0 = THDoubleTensor_data(output0);
    double *output_data1 = THDoubleTensor_data(output1);

    int nelem0 = THDoubleTensor_size(input, 0);
    int nelem1 = THDoubleTensor_size(output0, 0);
    int nelem2 = THDoubleTensor_size(output1, 0);

    int ndim1 = THDoubleTensor_nDimension(output0);
    int ndim2 = THDoubleTensor_nDimension(output1);

    if (ndim1 != ndim2)   THError("dim mismatch");
    if (nelem0 != nelem1) THError("size mismatch");
    if (nelem1 != nelem2) THError("size mismatch");

#pragma omp for
    for(int k=0; k<nelem1; k++) {
        output_data0[k] = (((long)input_data[k]-1) % N) + 1;
        output_data1[k] = (((long)input_data[k]-1) / N) + 1;
    }

    return 0;
}

static const struct luaL_Reg jhu_LongEncode__ [] = {
    {"encode", jhu_LongEncode},
    {"decode", jhu_LongDecode},
    {0, 0}
};

static const struct luaL_Reg jhu_DoubleEncode__ [] = {
    {"encode", jhu_DoubleEncode},
    {"decode", jhu_DoubleDecode},
    {0, 0}
};

static void jhu_Encode_init(lua_State *L) {
    // long
    int ret = luaT_pushmetatable(L, "torch.LongTensor");
    if(ret == 0) {
        THError("problem pushing metatable");
    }
    luaT_registeratname(L, jhu_LongEncode__, "jhu");
    lua_pop(L, 1);

    // double
    ret = luaT_pushmetatable(L, "torch.DoubleTensor");
    if(ret == 0) {
        THError("problem pushing metatable");
    }
    luaT_registeratname(L, jhu_DoubleEncode__, "jhu");
    lua_pop(L, 1);
}
