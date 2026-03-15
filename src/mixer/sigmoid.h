#ifndef SIGMOID_H
#define SIGMOID_H

#include <vector>

class Sigmoid {
 public:
  Sigmoid(int logit_size);
  float Logit(float p) const;
  static float Logistic(float p);
  static float FastLogistic(float p);

 private:
  float SlowLogit(float p);
  int logit_size_;
  std::vector<float> logit_table_;
  static std::vector<float> logistic_table_;
  static const int logistic_table_size_;
};

#endif
