#include "TH.h"
#include "JHNN.h"

#define torch_(NAME) TH_CONCAT_3(torch_, Real, NAME)
#define nn_(NAME) TH_CONCAT_3(nn_, Real, NAME)

#include "generic/LookupTable.c"
#include "THGenerateFloatTypes.h"
