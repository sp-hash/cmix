#include "byte-model.h"

#include <numeric>

ByteModel::ByteModel(const std::vector<bool>& vocab) : ex(0),top_(255), mid_(0),
    bot_(0),  vocab_(vocab), probs_(1.0 / 256, 256), dirty_(true) {
  for (int i = 0; i < 257; ++i) cumulative_probs_[i] = 0;
}

void ByteModel::RefreshSums() {
  if (!dirty_) return;
  float sum = 0;
  cumulative_probs_[0] = 0;
  const float* p = &probs_[0];
  for (int i = 0; i < 256; ++i) {
    sum += p[i];
    cumulative_probs_[i + 1] = sum;
  }
  dirty_ = false;
}

const std::valarray<float>& ByteModel::Predict() {
  RefreshSums();
  auto mid = bot_ + ((top_ - bot_) / 2);
  float sum_bot_mid = cumulative_probs_[mid + 1] - cumulative_probs_[bot_];
  float sum_mid_top = cumulative_probs_[top_ + 1] - cumulative_probs_[mid + 1];
  float num = sum_mid_top;
  float denom = sum_bot_mid + num;
  ex = bot_;
  const float* p = &probs_[0];
  float max_prob_val = p[bot_];
  for (int i = bot_ + 1; i <= top_; i++) {
    if (p[i] > max_prob_val) {
      max_prob_val = p[i];
      ex = i;
    }
  }
  if (denom == 0) outputs_[0] = 0.5;
  else outputs_[0] = num / denom;
  return outputs_;
}

const std::valarray<float>& ByteModel::BytePredict() {
  return probs_;
}

void ByteModel::Perceive(int bit) {
  mid_ = bot_ + ((top_ - bot_) / 2);
  if (bit) {
    bot_ = mid_ + 1;
  } else {
    top_ = mid_;
  }
}

void ByteModel::UpdateProbs(const unsigned int* frequencies) {
  top_ = 255;
  bot_ = 0;
  float total_sum = 0;
  float* p = &probs_[0];
  for (int i = 0; i < 256; ++i) {
    if (vocab_[i]) {
      float val = (frequencies[i] < 1) ? 1.0f : static_cast<float>(frequencies[i]);
      p[i] = val;
      total_sum += val;
    } else {
      p[i] = 0;
    }
  }
  if (total_sum > 0) {
    float inv_sum = 1.0f / total_sum;
    float current_sum = 0;
    cumulative_probs_[0] = 0;
    for (int i = 0; i < 256; ++i) {
      p[i] *= inv_sum;
      current_sum += p[i];
      cumulative_probs_[i + 1] = current_sum;
    }
  }
  dirty_ = false;
}

void ByteModel::UpdateProbs(const std::valarray<float>& normalized_probs) {
  top_ = 255;
  bot_ = 0;
  float* p = &probs_[0];
  const float* src = &normalized_probs[0];
  float current_sum = 0;
  cumulative_probs_[0] = 0;
  for (int i = 0; i < 256; ++i) {
    p[i] = vocab_[i] ? src[i] : 0;
    current_sum += p[i];
    cumulative_probs_[i + 1] = current_sum;
  }
  // Optional: re-normalize if some vocab was false
  if (current_sum > 0 && current_sum < 0.999f) {
    float inv_sum = 1.0f / current_sum;
    current_sum = 0;
    for (int i = 0; i < 256; ++i) {
      p[i] *= inv_sum;
      current_sum += p[i];
      cumulative_probs_[i + 1] = current_sum;
    }
  }
  dirty_ = false;
}

void ByteModel::ByteUpdate() {
  top_ = 255;
  bot_ = 0;
  dirty_ = true;
  float* p = &probs_[0];
  for (int i = 0; i < 256; ++i) {
    if (!vocab_[i]) p[i] = 0;
  }
}

