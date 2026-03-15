#include "lstm-layer.h"

#include "sigmoid.h"

#include <math.h>
#include <algorithm>
#include <numeric>

namespace {

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

  for (int i = 0; i < n; ++i) {
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

  for (unsigned int i = 0; i < num_cells_; ++i) {
    float fg = Sigmoid::Logistic(fgs_ptr[i]);
    float in_node = tanh(ins_ptr[i]);
    float og = Sigmoid::Logistic(ogs_ptr[i]);
    fgs_ptr[i] = fg;
    ins_ptr[i] = in_node;
    ogs_ptr[i] = og;

    float ig = 1.0f - fg;
    igs_ptr[i] = ig;

    float s = s_ptr[i] * fg + in_node * ig;
    s_ptr[i] = s;
    float ts = tanh(s);
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
    float f = w[input_symbol];
    const float* w_off = w + output_size_;
    for (int j = 0; j < num_inp; ++j) {
      f += inp_ptr[j] * w_off[j];
    }
    norm_ptr[i] = f;
  }
  float sq_sum = 0;
  for (unsigned int i = 0; i < num_cells_; ++i) {
    sq_sum += norm_ptr[i] * norm_ptr[i];
  }
  neurons.ivar_[epoch_] = 1.0f / sqrt((sq_sum / num_cells_) + 1e-5f);
  float ivar = neurons.ivar_[epoch_];
  const float* gamma_ptr = &neurons.gamma_[0];
  const float* beta_ptr = &neurons.beta_[0];
  float* state_ptr = &neurons.state_[epoch_][0];
  for (unsigned int i = 0; i < num_cells_; ++i) {
    norm_ptr[i] *= ivar;
    state_ptr[i] = norm_ptr[i] * gamma_ptr[i] + beta_ptr[i];
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

  for (unsigned int i = 0; i < num_cells_; ++i) {
    float ts = ts_ptr[i];
    float se = se_ptr[i];
    float ogs = ogs_ptr[i];
    float ins = ins_ptr[i];
    float igs = igs_ptr[i];
    float ls = ls_ptr[i];

    oge_ptr[i] = ts * se * ogs * (1.0f - ogs);
    float local_state_error = seer_ptr[i] + se * ogs * (1.0f - ts * ts);
    seer_ptr[i] = local_state_error;
    ine_ptr[i] = local_state_error * igs * (1.0f - ins * ins);
    fge_ptr[i] = (ls - ins) * local_state_error * fgs_ptr[i] * igs;
  }

  *hidden_error = 0;
  if (epoch > 0) {
    for (unsigned int i = 0; i < num_cells_; ++i) seer_ptr[i] *= fgs_ptr[i];
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
  for (unsigned int i = 0; i < num_cells_; ++i) {
    float err = err_ptr[i];
    float norm = norm_ptr[i];
    bu_ptr[i] += err;
    gu_ptr[i] += err * norm;
    err *= gamma_ptr[i] * neurons.ivar_[epoch];
    err_ptr[i] = err;
    dot_err_norm += err * norm;
  }
  dot_err_norm /= static_cast<float>(num_cells_);
  for (unsigned int i = 0; i < num_cells_; ++i) {
    err_ptr[i] -= dot_err_norm * norm_ptr[i];
  }

  if (layer > 0) {
    float* herr_ptr = &(*hidden_error)[0];
    for (unsigned int i = 0; i < num_cells_; ++i) {
      float f = 0;
      const float* trans = &neurons.transpose_[num_cells_ + i][0];
      for (unsigned int j = 0; j < num_cells_; ++j) {
        f += err_ptr[j] * trans[j];
      }
      herr_ptr[i] += f;
    }
  }
  if (epoch > 0) {
    float* serr_ptr = &stored_error_[0];
    for (unsigned int i = 0; i < num_cells_; ++i) {
      float f = 0;
      const float* trans = &neurons.transpose_[i][0];
      for (unsigned int j = 0; j < num_cells_; ++j) {
        f += err_ptr[j] * trans[j];
      }
      serr_ptr[i] += f;
    }
  }

  const float* inp_ptr = &input[0];
  int inp_size = input.size();
  for (unsigned int i = 0; i < num_cells_; ++i) {
    float err = err_ptr[i];
    float* up_ptr = &neurons.update_[i][output_size_];
    for (int j = 0; j < inp_size; ++j) {
      up_ptr[j] += err * inp_ptr[j];
    }
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
