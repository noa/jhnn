#include "TH.h"
#include "luaT.h"
#include "ops.h"

#include <lua.h>
#include <lauxlib.h>
#include <lualib.h>

static int jhu_THLogSample(lua_State *L) {
    lua_getglobal(L, "torch");
    THGenerator *_generator = (THGenerator *)luaT_getfieldcheckudata(L, -1, "_gen", torch_Generator);
    lua_pop(L, 2);

    THDoubleTensor *input = (THDoubleTensor *)luaT_checkudata(L, 1,
                                                              "torch.DoubleTensor");

    THLongTensor *output = (THLongTensor *)luaT_checkudata(L, 2,
                                                           "torch.LongTensor");

    long nframe = 0, dim = 0;
    long t, d;

    if (input->nDimension == 1) {
        nframe = 1;
        dim = input->size[0];
        THAssert( THLongTensor_size(output, 0) == 1 );
    }
    else if (input->nDimension == 2) {
        nframe = input->size[0];
        dim = input->size[1];
        THAssert( THLongTensor_size(output, 0) == nframe );
    }
    else {
        THArgCheck(0, 2, "vector or matrix expected");
    }

    THAssert( THDoubleTensor_isContiguous(input)  );
    THAssert( THLongTensor_isContiguous(output) );

    double *input_data;
    double *input_data0 = THDoubleTensor_data(input);
    long *output_data0 = THLongTensor_data(output);

#pragma omp parallel for private(t, d, input_data)
    for (t = 0; t < nframe; ++t) {
        input_data = input_data0 + dim*t;

        for (d=0; d<dim-1; ++d) {
            log_plus_equals(&input_data[d+1], input_data[d]);
        }

        /* sample a probability mass from a uniform distribution */
        double uniform_sample = log(THRandom_uniform(_generator, 0, 1)) + input_data[dim-1];

        /* Do a binary search for the slot in which the prob falls
           ie cum_dist[row][slot-1] < uniform_prob < cum_distr[row][slot] */
        int left_pointer = 0;
        int right_pointer = dim;
        int mid_pointer;
        double cum_prob;
        int sample_idx;

        while(right_pointer - left_pointer > 0) {
            mid_pointer = left_pointer + (right_pointer - left_pointer) / 2;
            THAssert( mid_pointer >= 0  );
            THAssert( mid_pointer < dim );
            cum_prob = input_data[mid_pointer];
            if (cum_prob < uniform_sample) {
                left_pointer = mid_pointer + 1;
            }
            else {
                right_pointer = mid_pointer;
            }
        }
        sample_idx = left_pointer;
        output_data0[t] = sample_idx + 1; /* increment by 1 for lua compat */
    }

    return 0;
}

static int jhu_THLogSampleD(lua_State *L) {
    lua_getglobal(L, "torch");
    THGenerator *_generator = (THGenerator *)luaT_getfieldcheckudata(L, -1, "_gen", torch_Generator);
    lua_pop(L, 2);

    THDoubleTensor *input = (THDoubleTensor *)luaT_checkudata(L, 1,
                                                              "torch.DoubleTensor");

    THDoubleTensor *output = (THDoubleTensor *)luaT_checkudata(L, 2,
                                                               "torch.DoubleTensor");

    long nframe = 0, dim = 0;
    long t, d;

    if (input->nDimension == 1) {
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
        THArgCheck(0, 2, "vector or matrix expected");
    }

    THAssert( THDoubleTensor_isContiguous(input)  );
    THAssert( THDoubleTensor_isContiguous(output) );

    double *input_data;
    double *input_data0 = THDoubleTensor_data(input);
    double *output_data0 = THDoubleTensor_data(output);

#pragma omp parallel for private(t, d, input_data)
    for (t = 0; t < nframe; ++t) {
        input_data = input_data0 + dim*t;

        for (d=0; d<dim-1; ++d) {
            log_plus_equals(&input_data[d+1], input_data[d]);
        }

        /* sample a probability mass from a uniform distribution */
        double uniform_sample = log(THRandom_uniform(_generator, 0, 1)) + input_data[dim-1];

        /* Do a binary search for the slot in which the prob falls
           ie cum_dist[row][slot-1] < uniform_prob < cum_distr[row][slot] */
        int left_pointer = 0;
        int right_pointer = dim;
        int mid_pointer;
        double cum_prob;
        int sample_idx;

        while(right_pointer - left_pointer > 0) {
            mid_pointer = left_pointer + (right_pointer - left_pointer) / 2;
            THAssert( mid_pointer >= 0  );
            THAssert( mid_pointer < dim );
            cum_prob = input_data[mid_pointer];
            if (cum_prob < uniform_sample) {
                left_pointer = mid_pointer + 1;
            }
            else {
                right_pointer = mid_pointer;
            }
        }
        sample_idx = left_pointer;
        output_data0[t] = sample_idx + 1; /* increment by 1 for lua compat */
    }

    return 0;
}

static const struct luaL_Reg jhu_THLogSample__ [] = {
    {"logsample", jhu_THLogSample},
    {"logsampled", jhu_THLogSampleD},
    {0, 0}
};

static void jhu_THLogSample_init(lua_State *L) {
    int ret = luaT_pushmetatable(L, "torch.DoubleTensor");
    if(ret == 0) {
        THError("problem pushing metatable");
    }
    luaT_registeratname(L, jhu_THLogSample__, "jhu");
    lua_pop(L, 1);
}
