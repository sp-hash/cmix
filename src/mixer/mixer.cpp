#include "mixer.h"

#include "sigmoid.h"

#include <numeric>
#include <math.h>

Mixer::Mixer(const std::valarray<float>& inputs,
    const std::vector<float>& extra_inputs,
    const unsigned long long& context, float learning_rate,
    unsigned int extra_input_size) : inputs_(inputs),
    extra_inputs_vec_(extra_inputs), extra_inputs_(extra_input_size), p_(0.5),
    learning_rate_(learning_rate), context_(context), max_steps_(1), steps_(0),
    last_decay_val_(0), last_decay_steps_(0xFFFFFFFFFFFFFFFFULL),
    cached_data_(nullptr), cached_context_(0xFFFFFFFFFFFFFFFFULL)
    {}

ContextData* Mixer::GetContextData(bool use_cache) {
  if (use_cache && cached_data_ && cached_context_ == context_) {
    return cached_data_;
  }
  ContextData* data;
  unsigned long long limit = 10000;
  if (context_map_.size() >= limit && context_map_.find(context_) == context_map_.end()) {
    data = context_map_[0xDEADBEEF].get();
    if (data == nullptr) {
      context_map_[0xDEADBEEF] = std::unique_ptr<ContextData>(
          new ContextData(inputs_.size(), extra_inputs_.size()));
      data = context_map_[0xDEADBEEF].get();
    }
  } else {
    data = context_map_[context_].get();
    if (data == nullptr) {
      context_map_[context_] = std::unique_ptr<ContextData>(
          new ContextData(inputs_.size(), extra_inputs_.size()));
      data = context_map_[context_].get();
    }
  }
  cached_data_ = data;
  cached_context_ = context_;
  return data;
}

float Mixer::Mix() {
  ContextData* data = GetContextData(true);
  float p = 0;
  const float* w = &data->weights[0];
  const float* inp = &inputs_[0];
  int n = inputs_.size();
  for (int i = 0; i < n; ++i) {
    p += inp[i] * w[i];
  }
  p_ = p;
  n = extra_inputs_.size();
  if (n > 0) {
    const float* e_vec = &extra_inputs_vec_[0];
    float* e_inp = &extra_inputs_[0];
    const float* ew = &data->extra_weights[0];
    float e = 0;
    for (int i = 0; i < n; ++i) {
      e_inp[i] = e_vec[i];
      e += e_inp[i] * ew[i];
    }
    p_ += e;
  }
  return p_;
}

void Mixer::Perceive(int bit) {
  ContextData* data = GetContextData(true);
  float decay;
  if (steps_ == last_decay_steps_) {
    decay = last_decay_val_;
  } else {
    decay = 0.9 / pow(0.0000001 * steps_ + 0.8, 0.8);
    last_decay_steps_ = steps_;
    last_decay_val_ = decay;
  }
  decay *= 1.5 - ((1.0 * data->steps) / max_steps_);
  float update = decay * learning_rate_ * (Sigmoid::Logistic(p_) - bit);
  ++steps_;
  ++data->steps;
  if (data->steps > max_steps_) {
    max_steps_ = data->steps;
  }
  int n = inputs_.size();
  if (n > 0) {
    const float* inp = &inputs_[0];
    float* w = &data->weights[0];
    for (int i = 0; i < n; ++i) w[i] -= update * inp[i];
    if ((data->steps & 1023) == 0) {
      for (int i = 0; i < n; ++i) w[i] *= 1.0f - 3.0e-6f;
    }
  }

  n = extra_inputs_.size();
  if (n > 0) {
    const float* e_inp = &extra_inputs_[0];
    float* ew = &data->extra_weights[0];
    for (int i = 0; i < n; ++i) ew[i] -= update * e_inp[i];
    if ((data->steps & 1023) == 0) {
      for (int i = 0; i < n; ++i) ew[i] *= 1.0f - 3.0e-6f;
    }
  }
}
