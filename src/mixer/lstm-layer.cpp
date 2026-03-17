#include "lstm-layer.h"

#include "sigmoid.h"

#include <math.h>
#include <algorithm>
#include <numeric>
#include <immintrin.h>

namespace {

// Small AVX-512/AVX2 helpers: dot product and saxpy (y += a*x)
static inline float dot_product_f(const float* a, const float* b, int n) {
  int i = 0;
  float sum = 0;
#ifdef __AVX512F__
  __m512 vsum = _mm512_setzero_ps();
  for (; i + 15 < n; i += 16) {
    __m512 va = _mm512_loadu_ps(a + i);
    __m512 vb = _mm512_loadu_ps(b + i);
    vsum = _mm512_add_ps(vsum, _mm512_mul_ps(va, vb));
  }
  sum = _mm512_reduce_add_ps(vsum);
#elif defined(__AVX2__)
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
  sum = _mm_cvtss_f32(vlow);
#endif
  for (; i < n; ++i) sum += a[i] * b[i];
  return sum;
}

static inline void saxpy_f(float* y, const float* x, int n, float a) {
  int i = 0;
#ifdef __AVX512F__
  __m512 va = _mm512_set1_ps(a);
  for (; i + 15 < n; i += 16) {
    __m512 vx = _mm512_loadu_ps(x + i);
    __m512 vy = _mm512_loadu_ps(y + i);
    _mm512_storeu_ps(y + i, _mm512_add_ps(vy, _mm512_mul_ps(va, vx)));
  }
#elif defined(__AVX2__)
  __m256 va2 = _mm256_set1_ps(a);
  for (; i + 7 < n; i += 8) {
    __m256 vx = _mm256_loadu_ps(x + i);
    __m256 vy = _mm256_loadu_ps(y + i);
    _mm256_storeu_ps(y + i, _mm256_add_ps(vy, _mm256_mul_ps(va2, vx)));
  }
#endif
  for (; i < n; ++i) y[i] += a * x[i];
}


void Adam(std::valarray<float>* g, std::valarray<float>* m,
    std::valarray<float>* v, std::valarray<float>* w, float learning_rate,
    float t, unsigned long long update_limit) {
  const float beta1 = 0.025f, beta2 = 0.9999f, eps = 1e-6f; 
  float alpha;
  float b1t, b2t;
  if (t < update_limit) {
    alpha = learning_rate * 0.1f / sqrt(5e-5f * t + 1.0f);
    b1t = 1.0f - pow(beta1, t);
    b2t = 1.0f - pow(beta2, t);
  } else {
    alpha = learning_rate * 0.1f / sqrt(5e-5f * update_limit + 1.0f);
    b1t = 1.0f - pow(beta1, update_limit);
    b2t = 1.0f - pow(beta2, update_limit);
  }


  float* g_ptr = &(*g)[0];
  float* m_ptr = &(*m)[0];
  float* v_ptr = &(*v)[0];
  float* w_ptr = &(*w)[0];
  int n = g->size();

  int i = 0;
#ifdef __AVX512F__
  __m512 v_beta1 = _mm512_set1_ps(beta1);
  __m512 v_1_beta1 = _mm512_set1_ps(1.0f - beta1);
  __m512 v_beta2 = _mm512_set1_ps(beta2);
  __m512 v_1_beta2 = _mm512_set1_ps(1.0f - beta2);
  __m512 v_alpha_b1t = _mm512_set1_ps(alpha / b1t);
  __m512 v_b2t = _mm512_set1_ps(b2t);
  __m512 v_eps = _mm512_set1_ps(eps);

  for (; i + 15 < n; i += 16) {
    __m512 gi = _mm512_loadu_ps(g_ptr + i);
    __m512 mi = _mm512_add_ps(_mm512_mul_ps(_mm512_loadu_ps(m_ptr + i), v_beta1), _mm512_mul_ps(v_1_beta1, gi));
    __m512 vi = _mm512_add_ps(_mm512_mul_ps(_mm512_loadu_ps(v_ptr + i), v_beta2), _mm512_mul_ps(v_1_beta2, _mm512_mul_ps(gi, gi)));
    _mm512_storeu_ps(m_ptr + i, mi);
    _mm512_storeu_ps(v_ptr + i, vi);
    __m512 term = _mm512_div_ps(_mm512_mul_ps(v_alpha_b1t, mi), _mm512_add_ps(_mm512_sqrt_ps(_mm512_div_ps(vi, v_b2t)), v_eps));
    _mm512_storeu_ps(w_ptr + i, _mm512_sub_ps(_mm512_loadu_ps(w_ptr + i), term));
  }
#elif defined(__AVX2__)
  __m256 v_beta1 = _mm256_set1_ps(beta1);
  __m256 v_1_beta1 = _mm256_set1_ps(1.0f - beta1);
  __m256 v_beta2 = _mm256_set1_ps(beta2);
  __m256 v_1_beta2 = _mm256_set1_ps(1.0f - beta2);
  __m256 v_alpha_b1t = _mm256_set1_ps(alpha / b1t);
  __m256 v_b2t = _mm256_set1_ps(b2t);
  __m256 v_eps = _mm256_set1_ps(eps);

  for (; i + 7 < n; i += 8) {
    __m256 gi = _mm256_loadu_ps(g_ptr + i);
    __m256 mi = _mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(m_ptr + i), v_beta1), _mm256_mul_ps(v_1_beta1, gi));
    __m256 vi = _mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(v_ptr + i), v_beta2), _mm256_mul_ps(v_1_beta2, _mm256_mul_ps(gi, gi)));
    _mm256_storeu_ps(m_ptr + i, mi);
    _mm256_storeu_ps(v_ptr + i, vi);
    __m256 term = _mm256_div_ps(_mm256_mul_ps(v_alpha_b1t, mi), _mm256_add_ps(_mm256_sqrt_ps(_mm256_div_ps(vi, v_b2t)), v_eps));
    _mm256_storeu_ps(w_ptr + i, _mm256_sub_ps(_mm256_loadu_ps(w_ptr + i), term));
  }
#endif

  for (; i < n; ++i) {
    float gi = g_ptr[i];
    float mi = m_ptr[i] * beta1 + (1.0f - beta1) * gi;
    float vi = v_ptr[i] * beta2 + (1.0f - beta2) * gi * gi;
    m_ptr[i] = mi;
    v_ptr[i] = vi;
    w_ptr[i] -= alpha * (mi / b1t) / (sqrt(vi / b2t) + eps);
  }
}

}

LstmLayer::LstmLayer(unsigned int input_size, unsigned int auxiliary_input_size,
    unsigned int output_size, unsigned int num_cells, int horizon,
    float gradient_clip, float learning_rate) :
    state_(num_cells), state_error_(num_cells), stored_error_(num_cells),
    tanh_state_(std::valarray<float>(num_cells), horizon),
    input_gate_state_(std::valarray<float>(num_cells), horizon),
    last_state_(std::valarray<float>(num_cells), horizon),
    gradient_clip_(gradient_clip), learning_rate_(learning_rate),
    num_cells_(num_cells), epoch_(0), horizon_(horizon),
    input_size_(auxiliary_input_size), output_size_(output_size),
    forget_gate_(input_size, num_cells, horizon, output_size_ + input_size_),
    input_node_(input_size, num_cells, horizon, output_size_ + input_size_),
    output_gate_(input_size, num_cells, horizon, output_size_ + input_size_) {
  float val = sqrt(6.0f / float(input_size_ + output_size_));
  float low = -val;
  float range = 2 * val;
  for (unsigned int i = 0; i < num_cells_; ++i) {
    for (unsigned int j = 0; j < forget_gate_.weights_[i].size(); ++j) {
      forget_gate_.weights_[i][j] = low + Rand() * range;
      input_node_.weights_[i][j] = low + Rand() * range;
      output_gate_.weights_[i][j] = low + Rand() * range;
    }
    forget_gate_.weights_[i][forget_gate_.weights_[i].size() - 1] = 1;
  }
}

void LstmLayer::ForwardPass(const std::valarray<float>& input, int input_symbol,
    std::valarray<float>* hidden, int hidden_start) {
  last_state_[epoch_] = state_;
  ForwardPass(forget_gate_, input, input_symbol);
  ForwardPass(input_node_, input, input_symbol);
  ForwardPass(output_gate_, input, input_symbol);

  float* fgs_ptr = &forget_gate_.state_[epoch_][0];
  float* ins_ptr = &input_node_.state_[epoch_][0];
  float* ogs_ptr = &output_gate_.state_[epoch_][0];
  float* igs_ptr = &input_gate_state_[epoch_][0];
  float* s_ptr = &state_[0];
  float* ts_ptr = &tanh_state_[epoch_][0];
  float* h_ptr = &(*hidden)[0] + hidden_start;

  const float scale = Sigmoid::table_size / 20.0f;
  const int half = Sigmoid::table_size / 2;
  const float* table = Sigmoid::logistic_table_ptr;
  const int table_size = Sigmoid::table_size;

  unsigned int i = 0;
#ifdef __AVX2__
  __m256 v_scale = _mm256_set1_ps(scale);
  __m256i v_half = _mm256_set1_epi32(half);
  __m256i v_table_size = _mm256_set1_epi32(table_size);
  __m256i v_zero = _mm256_setzero_si256();
  __m256 v_zero_f = _mm256_setzero_ps();
  __m256 v_one_f = _mm256_set1_ps(1.0f);
  __m256 v_two_f = _mm256_set1_ps(2.0f);

  for (; i + 7 < num_cells_; i += 8) {
    __m256 fgs = _mm256_loadu_ps(fgs_ptr + i);
    __m256 ins = _mm256_loadu_ps(ins_ptr + i);
    __m256 ogs = _mm256_loadu_ps(ogs_ptr + i);
    __m256 ss = _mm256_loadu_ps(s_ptr + i);

    auto fast_logistic_v = [&](__m256 p) {
      __m256i idx = _mm256_add_epi32(_mm256_cvtps_epi32(_mm256_mul_ps(p, v_scale)), v_half);
      // clamp idx
      __m256i clamped_idx = _mm256_max_epi32(v_zero, _mm256_min_epi32(idx, _mm256_sub_epi32(v_table_size, _mm256_set1_epi32(1))));
      __m256 res = _mm256_i32gather_ps(table, clamped_idx, 4);
      // handle out of bounds if necessary (though clamp handles it)
      return res;
    };

    __m256 fg = fast_logistic_v(fgs);
    __m256 in_node = _mm256_sub_ps(_mm256_mul_ps(v_two_f, fast_logistic_v(_mm256_mul_ps(v_two_f, ins))), v_one_f);
    __m256 og = fast_logistic_v(ogs);

    _mm256_storeu_ps(fgs_ptr + i, fg);
    _mm256_storeu_ps(ins_ptr + i, in_node);
    _mm256_storeu_ps(ogs_ptr + i, og);

    __m256 ig = _mm256_sub_ps(v_one_f, fg);
    _mm256_storeu_ps(igs_ptr + i, ig);

    __m256 s = _mm256_add_ps(_mm256_mul_ps(ss, fg), _mm256_mul_ps(in_node, ig));
    _mm256_storeu_ps(s_ptr + i, s);

    __m256 ts = _mm256_sub_ps(_mm256_mul_ps(v_two_f, fast_logistic_v(_mm256_mul_ps(v_two_f, s))), v_one_f);
    _mm256_storeu_ps(ts_ptr + i, ts);
    _mm256_storeu_ps(h_ptr + i, _mm256_mul_ps(og, ts));
  }
#endif

  for (; i < num_cells_; ++i) {
    // fg = Sigmoid::FastLogistic(fgs_ptr[i])
    int idx_fg = static_cast<int>(fgs_ptr[i] * scale + 0.5f) + half;
    float fg = (idx_fg >= table_size) ? 1.0f : (idx_fg <= 0) ? 0.0f : table[idx_fg];

    // in_node = Sigmoid::FastTanh(ins_ptr[i])
    int idx_in = static_cast<int>((2.0f * ins_ptr[i]) * scale + 0.5f) + half;
    float in_node = 2.0f * ((idx_in >= table_size) ? 1.0f : (idx_in <= 0) ? 0.0f : table[idx_in]) - 1.0f;

    // og = Sigmoid::FastLogistic(ogs_ptr[i])
    int idx_og = static_cast<int>(ogs_ptr[i] * scale + 0.5f) + half;
    float og = (idx_og >= table_size) ? 1.0f : (idx_og <= 0) ? 0.0f : table[idx_og];

    fgs_ptr[i] = fg;
    ins_ptr[i] = in_node;
    ogs_ptr[i] = og;

    float ig = 1.0f - fg;
    igs_ptr[i] = ig;

    float s = s_ptr[i] * fg + in_node * ig;
    s_ptr[i] = s;

    // ts = Sigmoid::FastTanh(s)
    int idx_s = static_cast<int>((2.0f * s) * scale + 0.5f) + half;
    float ts = 2.0f * ((idx_s >= table_size) ? 1.0f : (idx_s <= 0) ? 0.0f : table[idx_s]) - 1.0f;

    ts_ptr[i] = ts;
    h_ptr[i] = og * ts;
  }

  ++epoch_;
  if (epoch_ == horizon_) epoch_ = 0;
}

void LstmLayer::ForwardPass(NeuronLayer& neurons,
    const std::valarray<float>& input, int input_symbol) {
  int num_inp = input.size();
  const float* inp_ptr = &input[0];
  float* norm_ptr = &neurons.norm_[epoch_][0];
  for (unsigned int i = 0; i < num_cells_; ++i) {
    const float* w = &neurons.weights_[i][0];
    const float* w_off = w + output_size_;
    norm_ptr[i] = w[input_symbol] + dot_product_f(inp_ptr, w_off, num_inp);
  }
  float sq_sum = 1e-10f; // tiny epsilon
  int i_cell = 0;
#ifdef __AVX2__
  __m256 v_sq_sum = _mm256_setzero_ps();
  for (; i_cell + 7 < (int)num_cells_; i_cell += 8) {
    __m256 v_norm = _mm256_loadu_ps(norm_ptr + i_cell);
    v_sq_sum = _mm256_add_ps(v_sq_sum, _mm256_mul_ps(v_norm, v_norm));
  }
  alignas(32) float tmp[8];
  _mm256_storeu_ps(tmp, v_sq_sum);
  sq_sum += (tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7]);
#endif
  for (; (unsigned int)i_cell < num_cells_; ++i_cell) {
    sq_sum += norm_ptr[i_cell] * norm_ptr[i_cell];
  }

  neurons.ivar_[epoch_] = 1.0f / sqrt((sq_sum / num_cells_) + 1e-5f);
  float ivar = neurons.ivar_[epoch_];
  const float* gamma_ptr = &neurons.gamma_[0];
  const float* beta_ptr = &neurons.beta_[0];
  float* state_ptr = &neurons.state_[epoch_][0];

  i_cell = 0;
#ifdef __AVX2__
  __m256 v_ivar = _mm256_set1_ps(ivar);
  for (; i_cell + 7 < (int)num_cells_; i_cell += 8) {
    __m256 v_norm = _mm256_loadu_ps(norm_ptr + i_cell);
    __m256 v_gamma = _mm256_loadu_ps(gamma_ptr + i_cell);
    __m256 v_beta = _mm256_loadu_ps(beta_ptr + i_cell);
    v_norm = _mm256_mul_ps(v_norm, v_ivar);
    _mm256_storeu_ps(norm_ptr + i_cell, v_norm);
    _mm256_storeu_ps(state_ptr + i_cell, _mm256_add_ps(_mm256_mul_ps(v_norm, v_gamma), v_beta));
  }
#endif
  for (; (unsigned int)i_cell < num_cells_; ++i_cell) {
    norm_ptr[i_cell] *= ivar;
    state_ptr[i_cell] = norm_ptr[i_cell] * gamma_ptr[i_cell] + beta_ptr[i_cell];
  }
}

void LstmLayer::ClipGradients(std::valarray<float>* arr) {
  for (unsigned int i = 0; i < arr->size(); ++i) {
    if ((*arr)[i] < -gradient_clip_) (*arr)[i] = -gradient_clip_;
    else if ((*arr)[i] > gradient_clip_) (*arr)[i] = gradient_clip_;
  }
}

void LstmLayer::BackwardPass(const std::valarray<float>&input, int epoch,
    int layer, int input_symbol, std::valarray<float>* hidden_error) {
  if (epoch == (int)horizon_ - 1) {
    stored_error_ = *hidden_error;
    state_error_ = 0;
  } else {
    stored_error_ += *hidden_error;
  }

  const float* ts_ptr = &tanh_state_[epoch][0];
  const float* se_ptr = &stored_error_[0];
  const float* ogs_ptr = &output_gate_.state_[epoch][0];
  const float* ins_ptr = &input_node_.state_[epoch][0];
  const float* igs_ptr = &input_gate_state_[epoch][0];
  const float* ls_ptr = &last_state_[epoch][0];
  const float* fgs_ptr = &forget_gate_.state_[epoch][0];

  float* oge_ptr = &output_gate_.error_[0];
  float* seer_ptr = &state_error_[0];
  float* ine_ptr = &input_node_.error_[0];
  float* fge_ptr = &forget_gate_.error_[0];

  int i_cell = 0;
#ifdef __AVX2__
  __m256 v_one = _mm256_set1_ps(1.0f);
  for (; i_cell + 7 < (int)num_cells_; i_cell += 8) {
    __m256 ts = _mm256_loadu_ps(ts_ptr + i_cell);
    __m256 se = _mm256_loadu_ps(se_ptr + i_cell);
    __m256 ogs = _mm256_loadu_ps(ogs_ptr + i_cell);
    __m256 ins = _mm256_loadu_ps(ins_ptr + i_cell);
    __m256 igs = _mm256_loadu_ps(igs_ptr + i_cell);
    __m256 ls = _mm256_loadu_ps(ls_ptr + i_cell);
    __m256 fgs = _mm256_loadu_ps(fgs_ptr + i_cell);

    // oge_ptr[i] = ts * se * ogs * (1.0f - ogs);
    _mm256_storeu_ps(oge_ptr + i_cell, _mm256_mul_ps(_mm256_mul_ps(ts, se), _mm256_mul_ps(ogs, _mm256_sub_ps(v_one, ogs))));

    // local_state_error = seer_ptr[i] + se * ogs * (1.0f - ts * ts);
    __m256 lse = _mm256_add_ps(_mm256_loadu_ps(seer_ptr + i_cell), _mm256_mul_ps(_mm256_mul_ps(se, ogs), _mm256_sub_ps(v_one, _mm256_mul_ps(ts, ts))));
    _mm256_storeu_ps(seer_ptr + i_cell, lse);

    // ine_ptr[i] = local_state_error * igs * (1.0f - ins * ins);
    _mm256_storeu_ps(ine_ptr + i_cell, _mm256_mul_ps(_mm256_mul_ps(lse, igs), _mm256_sub_ps(v_one, _mm256_mul_ps(ins, ins))));

    // fge_ptr[i] = (ls - ins) * local_state_error * fgs * igs;
    _mm256_storeu_ps(fge_ptr + i_cell, _mm256_mul_ps(_mm256_mul_ps(_mm256_sub_ps(ls, ins), lse), _mm256_mul_ps(fgs, igs)));
  }
#endif

  for (; (unsigned int)i_cell < num_cells_; ++i_cell) {
    float ts = ts_ptr[i_cell];
    float se = se_ptr[i_cell];
    float ogs = ogs_ptr[i_cell];
    float ins = ins_ptr[i_cell];
    float igs = igs_ptr[i_cell];
    float ls = ls_ptr[i_cell];

    oge_ptr[i_cell] = ts * se * ogs * (1.0f - ogs);
    float local_state_error = seer_ptr[i_cell] + se * ogs * (1.0f - ts * ts);
    seer_ptr[i_cell] = local_state_error;
    ine_ptr[i_cell] = local_state_error * igs * (1.0f - ins * ins);
    fge_ptr[i_cell] = (ls - ins) * local_state_error * fgs_ptr[i_cell] * igs;
  }

  *hidden_error = 0;
  if (epoch > 0) {
    unsigned int i_loop = 0;
#ifdef __AVX2__
    for (; i_loop + 7 < num_cells_; i_loop += 8) {
      _mm256_storeu_ps(seer_ptr + i_loop, _mm256_mul_ps(_mm256_loadu_ps(seer_ptr + i_loop), _mm256_loadu_ps(fgs_ptr + i_loop)));
    }
#endif
    for (; i_loop < num_cells_; ++i_loop) seer_ptr[i_loop] *= fgs_ptr[i_loop];
    stored_error_ = 0;
  } else {
    if (update_steps_ < update_limit_) {
      ++update_steps_;
    }
  }

  BackwardPass(forget_gate_, input, epoch, layer, input_symbol, hidden_error);
  BackwardPass(input_node_, input, epoch, layer, input_symbol, hidden_error);
  BackwardPass(output_gate_, input, epoch, layer, input_symbol, hidden_error);

  ClipGradients(&state_error_);
  ClipGradients(&stored_error_);
  ClipGradients(hidden_error);
}

void LstmLayer::BackwardPass(NeuronLayer& neurons,
    const std::valarray<float>&input, int epoch, int layer, int input_symbol,
    std::valarray<float>* hidden_error) {
  if (epoch == (int)horizon_ - 1) {
    neurons.gamma_u_ = 0;
    neurons.beta_u_ = 0;
    for (unsigned int i = 0; i < num_cells_; ++i) {
      neurons.update_[i] = 0;
      int offset = output_size_ + input_size_;
      const float* w = &neurons.weights_[i][offset];
      for (unsigned int j = 0; j < neurons.transpose_.size(); ++j) {
        neurons.transpose_[j][i] = w[j];
      }
    }
  }

  float* bu_ptr = &neurons.beta_u_[0];
  float* gu_ptr = &neurons.gamma_u_[0];
  float* err_ptr = &neurons.error_[0];
  const float* norm_ptr = &neurons.norm_[epoch][0];
  const float* gamma_ptr = &neurons.gamma_[0];
  float dot_err_norm = 0;
  float ivar = neurons.ivar_[epoch];
  unsigned int i = 0;
#ifdef __AVX2__
  __m256 v_dot_err_norm = _mm256_setzero_ps();
  __m256 v_ivar = _mm256_set1_ps(ivar);
  for (; i + 7 < num_cells_; i += 8) {
    __m256 err = _mm256_loadu_ps(err_ptr + i);
    __m256 norm = _mm256_loadu_ps(norm_ptr + i);
    __m256 gamma = _mm256_loadu_ps(gamma_ptr + i);

    _mm256_storeu_ps(bu_ptr + i, _mm256_add_ps(_mm256_loadu_ps(bu_ptr + i), err));
    _mm256_storeu_ps(gu_ptr + i, _mm256_add_ps(_mm256_loadu_ps(gu_ptr + i), _mm256_mul_ps(err, norm)));

    __m256 updated_err = _mm256_mul_ps(_mm256_mul_ps(err, gamma), v_ivar);
    _mm256_storeu_ps(err_ptr + i, updated_err);
    v_dot_err_norm = _mm256_add_ps(v_dot_err_norm, _mm256_mul_ps(updated_err, norm));
  }
  __m128 vlow = _mm256_castps256_ps128(v_dot_err_norm);
  __m128 vhigh = _mm256_extractf128_ps(v_dot_err_norm, 1);
  vlow = _mm_add_ps(vlow, vhigh);
  vlow = _mm_hadd_ps(vlow, vlow);
  vlow = _mm_hadd_ps(vlow, vlow);
  dot_err_norm = _mm_cvtss_f32(vlow);
#endif

  for (; i < num_cells_; ++i) {
    float err = err_ptr[i];
    float norm = norm_ptr[i];
    bu_ptr[i] += err;
    gu_ptr[i] += err * norm;
    err *= gamma_ptr[i] * ivar;
    err_ptr[i] = err;
    dot_err_norm += err * norm;
  }
  dot_err_norm /= static_cast<float>(num_cells_);

  // Vectorized: err_ptr[i] -= dot_err_norm * norm_ptr[i]
  saxpy_f(err_ptr, norm_ptr, num_cells_, -dot_err_norm);

  if (layer > 0) {
    float* herr_ptr = &(*hidden_error)[0];
    for (unsigned int i = 0; i < num_cells_; ++i) {
      const float* trans = &neurons.transpose_[num_cells_ + i][0];
      herr_ptr[i] += dot_product_f(err_ptr, trans, num_cells_);
    }
  }
  if (epoch > 0) {
    float* serr_ptr = &stored_error_[0];
    for (unsigned int i = 0; i < num_cells_; ++i) {
      const float* trans = &neurons.transpose_[i][0];
      serr_ptr[i] += dot_product_f(err_ptr, trans, num_cells_);
    }
  }

  const float* inp_ptr = &input[0];
  int inp_size = input.size();
  for (unsigned int i = 0; i < num_cells_; ++i) {
    float err = err_ptr[i];
    float* up_ptr = &neurons.update_[i][output_size_];
    // vectorized accumulation: up_ptr[j] += err * inp_ptr[j]
    saxpy_f(up_ptr, inp_ptr, inp_size, err);
    neurons.update_[i][input_symbol] += err;
  }

  if (epoch == 0) {
    for (unsigned int i = 0; i < num_cells_; ++i) {
      Adam(&neurons.update_[i], &neurons.m_[i], &neurons.v_[i],
          &neurons.weights_[i], learning_rate_, update_steps_, update_limit_);
    }
    Adam(&neurons.gamma_u_, &neurons.gamma_m_, &neurons.gamma_v_,
        &neurons.gamma_, learning_rate_, update_steps_, update_limit_);
    Adam(&neurons.beta_u_, &neurons.beta_m_, &neurons.beta_v_,
        &neurons.beta_, learning_rate_, update_steps_, update_limit_);
  }
}

std::vector<std::valarray<std::valarray<float>>*> LstmLayer::Weights() {
  std::vector<std::valarray<std::valarray<float>>*> weights;
  weights.push_back(&forget_gate_.weights_);
  weights.push_back(&input_node_.weights_);
  weights.push_back(&output_gate_.weights_);
  return weights;
}
