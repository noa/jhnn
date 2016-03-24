#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/JH.h"
#else

TH_API void JH_(logscale)(THTensor *log_prob_dist);

TH_API void JH_(logsample)(THIndexTensor *self,
                           THGenerator *_generator,
                           THTensor *prob_dist);

#endif
