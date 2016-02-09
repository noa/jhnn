#ifndef __JHU_OPS_H__
#define __JHU_OPS_H__

#include <math.h>

// Numerically stable way of computing f(a) = log(1 + exp(a))
inline double log1pexp(double a) {
  if(a <= -37.0) {
    return exp(a);
  } else if(a > -37.0 && a <= 18.0) {
    return log1p(exp(a));
  } else if(a > 18.0 && a <= 33.3) {
    return a + exp(-a);
  } else if(a > 33.3) {
    return a;
  }
  return a; // shouldn't get here
}

inline double log_add(double a, double b) {
  if (a == -INFINITY) return b;
  if (b == -INFINITY) return a;
  return a>b? a+log1pexp(b-a):  b+log1pexp(a-b);
}

inline void log_plus_equals(double *l1, double l2) {
  *l1 = log_add(*l1,l2);
}

#endif
