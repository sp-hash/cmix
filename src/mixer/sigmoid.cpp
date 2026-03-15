#include "sigmoid.h"

#include <math.h>

const int Sigmoid::logistic_table_size_ = 1000001;
std::vector<float> Sigmoid::logistic_table_;

Sigmoid::Sigmoid(int logit_size) : logit_size_(logit_size),
    logit_table_(logit_size, 0) {
  for (int i = 0; i < logit_size_; ++i) {
    logit_table_[i] = SlowLogit((i + 0.5f) / logit_size_);
  }
  if (logistic_table_.empty()) {
    logistic_table_.resize(logistic_table_size_);
    for (int i = 0; i < logistic_table_size_; ++i) {
      float p = (i - (logistic_table_size_ / 2)) * 20.0f / logistic_table_size_;
      logistic_table_[i] = 1.0f / (1.0f + exp(-p));
    }
  }
}

float Sigmoid::Logit(float p) const {
  int index = p * logit_size_;
  if (index >= logit_size_) index = logit_size_ - 1;
  else if (index < 0) index = 0;
  return logit_table_[index];
}

float Sigmoid::FastLogistic(float p) {
  int index = static_cast<int>((p * (logistic_table_size_ / 20.0f)) + (logistic_table_size_ / 2));
  if (index >= logistic_table_size_) return 1.0f;
  if (index <= 0) return 0.0f;
  return logistic_table_[index];
}

float Sigmoid::Logistic(float p) {
  return FastLogistic(p);
}

float Sigmoid::SlowLogit(float p) {
  return log(p / (1 - p));
}
