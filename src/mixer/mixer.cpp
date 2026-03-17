#include "mixer.h"

#include "sigmoid.h"

#include <numeric>
#include <math.h>
#include <immintrin.h>
#include <cstring>

// AVX2 helpers (safe fallbacks for small n)
static inline float dot_product_f(const float* a, const float* b, int n) {
  int i = 0;
  __m256 vsum2 = _mm256_setzero_ps();
  for (; i + 7 < n; i += 8) {
    __m256 va = _mm256_loadu_ps(a + i);
    __m256 vb = _mm256_loadu_ps(b + i);
    vsum2 = _mm256_add_ps(vsum2, _mm256_mul_ps(va, vb));
  }
  __m128 vlow = _mm256_castps256_ps128(vsum2);
  __m128 vhigh = _mm256_extractf128_ps(vsum2, 1);
  vlow = _mm_add_ps(vlow, vhigh);
  vlow = _mm_hadd_ps(vlow, vlow);
  vlow = _mm_hadd_ps(vlow, vlow);
  float sum = _mm_cvtss_f32(vlow);
  for (; i < n; ++i) sum += a[i] * b[i];
  return sum;
}

static inline void saxpy_f(float* y, const float* x, int n, float a) {
  int i = 0;
  __m256 va = _mm256_set1_ps(a);
  for (; i + 7 < n; i += 8) {
    __m256 vx = _mm256_loadu_ps(x + i);
    __m256 vy = _mm256_loadu_ps(y + i);
    vy = _mm256_add_ps(vy, _mm256_mul_ps(va, vx));
    _mm256_storeu_ps(y + i, vy);
  }
  for (; i < n; ++i) y[i] += a * x[i];
}

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
  const float* w = &data->weights[0];
  const float* inp = &inputs_[0];
  int n = inputs_.size();
  // vectorized dot product
  p_ = (n > 0) ? dot_product_f(inp, w, n) : 0.0f;
  n = extra_inputs_.size();
  if (n > 0) {
    const float* e_vec = &extra_inputs_vec_[0];
    float* e_inp = &extra_inputs_[0];
    const float* ew = &data->extra_weights[0];
    // copy extra inputs for later updates and compute dot product
    std::memcpy(e_inp, e_vec, sizeof(float) * n);
    p_ += dot_product_f(e_inp, ew, n);
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
    // vectorized weight update: w += (-update) * inp
    saxpy_f(w, inp, n, -update);
    if ((data->steps & 1023) == 0) {
      for (int i = 0; i < n; ++i) w[i] *= 1.0f - 3.0e-6f;
    }
  }

  n = extra_inputs_.size();
  if (n > 0) {
    const float* e_inp = &extra_inputs_[0];
    float* ew = &data->extra_weights[0];
    saxpy_f(ew, e_inp, n, -update);
    if ((data->steps & 1023) == 0) {
      for (int i = 0; i < n; ++i) ew[i] *= 1.0f - 3.0e-6f;
    }
  }
}
