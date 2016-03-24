#include "TH.h"
#include "JH.h"

#define torch_(NAME) TH_CONCAT_3(torch_, Real, NAME)
#define jh_(NAME) TH_CONCAT_3(jh_, Real, NAME)

#include "generic/JH.c"
#include "THGenerateFloatTypes.h"
