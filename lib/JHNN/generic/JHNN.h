#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/JHNN.h"
#else

TH_API void JHNN_(LookupTable_accGradParameters)(
          JHNNState *state,
          THIndexTensor *input,
          THTensor *gradOutput,
          THTensor *gradWeight,
          THIntegerTensor *count,
          THTensor *sorted,
          THTensor *indices,
          bool scaleGradByFreq,
          int paddingValue,
          real scale);

#endif
