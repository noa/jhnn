#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/LookupTable.c"
#else

static void JHNN_(LookupTable_resetCount)(
          THInteger_t *count_data,
          THIndexTensor *input)
{
  int i;
  THIndex_t *input_data = THIndexTensor_(data)(input);
  long numel = THIndexTensor_(nElement)(input);
  
  for (i = 0; i<numel; i++) {
    long k = input_data[i] - 1;
    count_data[k] = 0;
  }
  for (i = 0; i<numel; i++) {
    long k = input_data[i] - 1;
    count_data[k]++;
  }
}

void JHNN_(LookupTable_accGradParameters)(
          JHNNState *state,
          THIndexTensor *input,
          THTensor *gradOutput,
          THTensor *gradWeight,
          THIntegerTensor *count,
          THTensor *sorted,
          THTensor *indices,
          bool scaleGradByFreq,
          int paddingValue,
          THTensor *scale)
{
  long i,j;
  THInteger_t *count_data = NULL;

  if (scaleGradByFreq) {
    THIntegerTensor_(resize1d)(count, gradWeight->size[0]);
    count_data = THIntegerTensor_(data)(count);
  }

  if (!THTensor_(isContiguous)(gradWeight))
    THError("gradWeight must be contiguous");
  if (!THIndexTensor_(isContiguous)(input))
    THError("input must be contiguous");
  if (!THTensor_(isContiguous)(scale))
    THError("scale must be contiguous");
  if (THIndexTensor_(nDimension)(input) != 2)
    THError("input must be matrix but input dim = %d", THIndexTensor_(nDimension)(input));

  long numi = THIndexTensor_(size)(input, 0);
  long numj = THIndexTensor_(size)(input, 1);

  if (numi != THTensor_(size)(scale, 0))
    THError("size mismatch between input and scale");

  THIndex_t *input_data = THIndexTensor_(data)(input);
  long numel = THIndexTensor_(nElement)(input);
  long numw = THTensor_(size)(gradWeight, 0);

  // check that inputs are all within range
  for (i=0; i<numel; i++) {
    if (input_data[i] < 1 || input_data[i] > numw) {
      THError("input out of range");
    }
  }

  gradOutput = THTensor_(newContiguous)(gradOutput);

  real *gw = THTensor_(data)(gradWeight);
  real *go = THTensor_(data)(gradOutput);
  real *scale_data = THTensor_(data)(scale);
  long stride = THTensor_(stride)(gradWeight, 0);

  if (count_data) {
    JHNN_(LookupTable_resetCount)(count_data, input);
  }

  // TODO: add the OpenMP loop back in
  
  for (i=0; i<numi; i++) {
    for (j=0; j<numj; j++) {
      long k = input_data[j + numj*i] - 1; //  elements in the same row are contiguous in memory for a matrix
      real scale_ = scale_data[i];
      if (count_data) scale_ /= count_data[k];
      THBlas_(axpy)(stride, scale_, go + i*stride, 1, gw + k*stride, 1);
    }
  }

  THTensor_(free)(gradOutput);
}

#endif
