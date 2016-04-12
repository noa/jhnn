#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/JH.c"
#else

void JH_(logsum)(THTensor *input, THTensor *result) {
  int size = input->nDimension;
  printf("size = %d\n", size);
  return;
}

void JH_(logscale)(THTensor *log_prob_dist) {
  
  return;
}

void JH_(logsample)(THIndexTensor *self, THGenerator *_generator, THTensor *prob_dist) {
  return;
}

#endif
