#ifndef JH_H
#define JH_H

#include <stdbool.h>
#include <TH.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define JH_(NAME) TH_CONCAT_3(JH_, Real, NAME)

#define THIndexTensor THLongTensor
#define THIndexTensor_(NAME) THLongTensor_ ## NAME

#define THIntegerTensor THIntTensor
#define THIntegerTensor_(NAME) THIntTensor_ ## NAME

typedef long THIndex_t;
typedef int THInteger_t;
typedef void JHState;

#include "generic/JH.h"
#include <THGenerateFloatTypes.h>

#endif
