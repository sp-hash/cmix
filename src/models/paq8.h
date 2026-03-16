#ifndef PAQ8_H
#define PAQ8_H

#include "model.h"
#include <vector>
#include <memory>
#include <valarray>

namespace paq8 {
  class Predictor;
  void setMaxMem(unsigned long long m);
  unsigned long long getMaxMem();
  void setFileSize(unsigned long long f);
  void setLightMode(bool l);
  bool getLightMode();
}

class PAQ8 : public Model {
 public:
  PAQ8(int memory);
  const std::valarray<float>& Predict();
  unsigned int NumOutputs();
  void Perceive(int bit);
  void ByteUpdate() {};

 private:
  std::unique_ptr<paq8::Predictor> predictor_;
};

#endif
