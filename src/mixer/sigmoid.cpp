#include "sigmoid.h"

#include <math.h>

// optimize: precompute scaling factors for FastLogistic

const int Sigmoid::logistic_table_size_ = 1000001;
std::vector<float> Sigmoid::logistic_table_;
float* Sigmoid::logistic_table_ptr = nullptr;
int Sigmoid::table_size = 0;

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
    logistic_table_ptr = logistic_table_.data();
    table_size = logistic_table_size_;
  }
}

float Sigmoid::Logit(float p) const {
  int index = p * logit_size_;
  if (index >= logit_size_) index = logit_size_ - 1;
  else if (index < 0) index = 0;
  return logit_table_[index];
}

float Sigmoid::FastLogistic(float p) {
  static const float scale = 1000001.0f / 20.0f;
  static const int half = 1000001 / 2;
  int index = static_cast<int>(p * scale) + half;
  if (index >= 1000001) return 1.0f;
  if (index <= 0) return 0.0f;
  return logistic_table_[index];
}

float Sigmoid::FastTanh(float p) {
  return 2.0f * FastLogistic(2.0f * p) - 1.0f;
}

float Sigmoid::Logistic(float p) {
  return FastLogistic(p);
}

float Sigmoid::SlowLogit(float p) {
  return log(p / (1 - p));
}
