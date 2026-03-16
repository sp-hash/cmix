#define _CRT_SECURE_NO_WARNINGS

#include <fstream>
#include <ctime>
#include <chrono>
#include <stdio.h>
#include <cstdlib>
#include <vector>
#include <string.h>
#include <intrin.h>

#include "preprocess/preprocessor.h"
#include "coder/encoder.h"
#include "coder/decoder.h"
#include "predictor.h"
#include "models/paq8.h"

namespace {
  const int kMinVocabFileSize = 10000;
}

char* dictionary_path = NULL;

int Help() {
  printf("cmix version 21\n");
  printf("Compress:\n");
  printf("    with dictionary:    cmix -c [dictionary] [input] [output]\n");
  printf("    without dictionary: cmix -c [input] [output]\n");
  printf("    force text-mode:    cmix -t [dictionary] [input] [output]\n");
  printf("    no preprocessing:   cmix -n [input] [output]\n");
  printf("    only preprocessing: cmix -s [dictionary] [input] [output]\n");
  printf("                        cmix -s [input] [output]\n");
  printf("    limit memory:       cmix -m[limit] ... (e.g. -m8G, -m512M)\n");
  printf("Decompress:\n");
  printf("    with dictionary:    cmix -d [dictionary] [input] [output]\n");
  printf("    without dictionary: cmix -d [input] [output]\n");
  return -1;
}

void WriteHeader(unsigned long long length, const std::vector<bool>& vocab,
    bool dictionary_used, std::ofstream* os) {
  for (int i = 4; i >= 0; --i) {
    char c = length >> (8*i);
    if (i == 4) {
      c &= 0x7F;
      if (dictionary_used) c |= 0x80;
    }
    os->put(c);
  }
  if (length < kMinVocabFileSize) return;
  for (int i = 0; i < 32; ++i) {
    unsigned char c = 0;
    for (int j = 0; j < 8; ++j) {
      if (vocab[i * 8 + j]) c += 1<<j;
    }
    os->put(c);
  }
}

void WriteStorageHeader(FILE* out, bool dictionary_used) {
  for (int i = 4; i >= 0; --i) {
    char c = 0;
    if (i == 4 && dictionary_used) c = 0x80;
    putc(c, out);
  }
}

void ReadHeader(std::ifstream* is, unsigned long long* length,
    bool* dictionary_used, std::vector<bool>* vocab) {
  *length = 0;
  for (int i = 0; i <= 4; ++i) {
    *length <<= 8;
    unsigned char c = is->get();
    if (i == 0) {
      if (c&0x80) *dictionary_used = true;
      else *dictionary_used = false;
      c &= 0x7F;
    }
    *length += c;
  }
  if (*length == 0) return;
  if (*length < kMinVocabFileSize) {
    std::fill(vocab->begin(), vocab->end(), true);
    return;
  }
  for (int i = 0; i < 32; ++i) {
    unsigned char c = is->get();
    for (int j = 0; j < 8; ++j) {
      if (c & (1<<j)) (*vocab)[i * 8 + j] = true;
    }
  }
}

void ExtractVocab(unsigned long long input_bytes, std::ifstream* is,
    std::vector<bool>* vocab) {
  const int BUF_SIZE = 1 << 20; // 1MB buffer
  std::vector<char> buffer(BUF_SIZE);
  unsigned long long remaining = input_bytes;

  bool local_vocab[256];
  int count = 0;
  for (int i = 0; i < 256; ++i) {
    local_vocab[i] = (*vocab)[i];
    if (local_vocab[i]) count++;
  }

  while (remaining > 0 && count < 256) {
    size_t to_read = (remaining > BUF_SIZE) ? BUF_SIZE : (size_t)remaining;
    is->read(&buffer[0], to_read);
    size_t bytes_read = is->gcount();
    if (bytes_read == 0) break;
    remaining -= bytes_read;

    for (size_t i = 0; i < bytes_read; ++i) {
      unsigned char c = (unsigned char)buffer[i];
      if (!local_vocab[c]) {
        local_vocab[c] = true;
        count++;
        if (count == 256) break;
      }
    }
  }

  for (int i = 0; i < 256; ++i) {
    (*vocab)[i] = local_vocab[i];
  }
}

void ClearOutput() {
  fprintf(stderr, "\r                     \r");
  fflush(stderr);
}

void Compress(unsigned long long input_bytes, std::ifstream* is,
    std::ofstream* os, unsigned long long* output_bytes, Predictor* p) {
  Encoder e(os, p);
  std::chrono::steady_clock::time_point start_time;
  std::chrono::steady_clock::time_point last_update;
  bool started = false;
  ClearOutput();

  const int BUF_SIZE = 1 << 16;
  std::vector<char> buffer(BUF_SIZE);
  unsigned long long pos = 0;

  while (pos < input_bytes) {
    size_t to_read = (input_bytes - pos > BUF_SIZE) ? BUF_SIZE : (size_t)(input_bytes - pos);
    is->read(&buffer[0], to_read);
    size_t bytes_read = is->gcount();
    if (bytes_read == 0) break;

    for (size_t i = 0; i < bytes_read; ++i) {
      unsigned char c = (unsigned char)buffer[i];
      for (int j = 7; j >= 0; --j) {
        e.Encode((c >> j) & 1);
      }
      pos++;

      auto now = std::chrono::steady_clock::now();
      if (!started) {
        // mark processing start on first processed byte
        start_time = now;
        last_update = now;
        started = true;
      }
      if (started && std::chrono::duration_cast<std::chrono::seconds>(now - last_update).count() >= 5) {
        double frac = 100.0 * pos / input_bytes;
        long long elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
        long long remaining = (pos > 0) ? (elapsed * (input_bytes - pos) / pos) : 0;
        fprintf(stderr, "\rprogress: %.2f%%, ETA: %02lld:%02lld:%02lld", frac, remaining / 3600, (remaining % 3600) / 60, remaining % 60);
        fflush(stderr);
        last_update = now;
      }
    }
  }
  e.Flush();
  *output_bytes = os->tellp();
}

void Decompress(unsigned long long output_length, std::ifstream* is,
                std::ofstream* os, Predictor* p) {
  Decoder d(is, p);
  // Start timing when decompression output actually begins to be produced
  // to avoid including earlier setup time in the ETA calculation.
  std::chrono::steady_clock::time_point start_time;
  std::chrono::steady_clock::time_point last_update;
  bool started = false;
  ClearOutput();

  const int BUF_SIZE = 1 << 16;
  std::vector<char> buffer(BUF_SIZE);
  int buffered = 0;

  for(unsigned long long pos = 0; pos < output_length; ++pos) {
    int byte = 1;
    while (byte < 256) {
      byte += byte + d.Decode();
    }

    buffer[buffered++] = (unsigned char)byte;
    if (buffered == BUF_SIZE) {
      os->write(&buffer[0], BUF_SIZE);
      buffered = 0;
    }

    auto now = std::chrono::steady_clock::now();
    if (!started) {
      start_time = now;
      last_update = now;
      started = true;
    }
    if (started && std::chrono::duration_cast<std::chrono::seconds>(now - last_update).count() >= 5) {
      double frac = 100.0 * pos / output_length;
      long long elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
      long long remaining = (pos > 0) ? (elapsed * (output_length - pos) / pos) : 0;
      fprintf(stderr, "\rprogress: %.2f%%, ETA: %02lld:%02lld:%02lld", frac, remaining / 3600, (remaining % 3600) / 60, remaining % 60);
      fflush(stderr);
      last_update = now;
    }
  }
  if (buffered > 0) {
    os->write(&buffer[0], buffered);
  }
}

bool Store(const std::string& input_path, const std::string& temp_path,
    const std::string& output_path, FILE* dictionary,
    unsigned long long* input_bytes, unsigned long long* output_bytes) {
  FILE* data_in = fopen(input_path.c_str(), "rb");
  if (!data_in) return false;
  FILE* data_out = fopen(output_path.c_str(), "wb");
  if (!data_out) return false;
  fseek(data_in, 0L, SEEK_END);
  *input_bytes = ftell(data_in);
  fseek(data_in, 0L, SEEK_SET);
  WriteStorageHeader(data_out, dictionary != NULL);
  fprintf(stderr, "\rpreprocessing...");
  fflush(stderr);
  preprocessor::Encode(data_in, data_out, false, *input_bytes, temp_path,
      dictionary);
  fseek(data_out, 0L, SEEK_END);
  *output_bytes = ftell(data_out);
  fclose(data_in);
  fclose(data_out);
  return true;
}

bool RunCompression(bool enable_preprocess, bool text_mode,
    const std::string& input_path, const std::string& temp_path,
    const std::string& output_path, FILE* dictionary,
    unsigned long long* input_bytes, unsigned long long* output_bytes) {
  FILE* data_in = fopen(input_path.c_str(), "rb");
  if (!data_in) return false;
  setvbuf(data_in, NULL, _IOFBF, 1 << 20);
  FILE* temp_out = fopen(temp_path.c_str(), "wb");
  if (!temp_out) return false;
  setvbuf(temp_out, NULL, _IOFBF, 1 << 20);

  fseek(data_in, 0L, SEEK_END);
  *input_bytes = ftell(data_in);
  fseek(data_in, 0L, SEEK_SET);

  bool detected_pure_text = false;
  if (enable_preprocess) {
    fprintf(stderr, "\rpreprocessing...");
    fflush(stderr);
    detected_pure_text = preprocessor::Encode(data_in, temp_out, text_mode, *input_bytes, temp_path,
        dictionary);
  } else {
    preprocessor::NoPreprocess(data_in, temp_out, *input_bytes);
  }
  fclose(data_in);
  fclose(temp_out);

  std::ifstream temp_in(temp_path, std::ios::in | std::ios::binary);
  if (!temp_in.is_open()) return false;

  std::ofstream data_out(output_path, std::ios::out | std::ios::binary);
  if (!data_out.is_open()) return false;

  temp_in.seekg(0, std::ios::end);
  unsigned long long temp_bytes = temp_in.tellg();
  temp_in.seekg(0, std::ios::beg);

  std::vector<bool> vocab(256, false);
  if (temp_bytes < kMinVocabFileSize) {
    std::fill(vocab.begin(), vocab.end(), true);
  } else {
    ExtractVocab(temp_bytes, &temp_in, &vocab);
    temp_in.seekg(0, std::ios::beg);
  }

  WriteHeader(temp_bytes, vocab, dictionary != NULL, &data_out);
  // Only use light mode if explicitly requested with -t
  paq8::setLightMode(text_mode);
  Predictor p(vocab, temp_bytes);
  if (enable_preprocess) preprocessor::Pretrain(&p, dictionary);
  Compress(temp_bytes, &temp_in, &data_out, output_bytes, &p);
  temp_in.close();
  data_out.close();
  remove(temp_path.c_str());
  return true;
}

bool RunDecompression(const std::string& input_path,
    const std::string& temp_path, const std::string& output_path,
    FILE* dictionary, unsigned long long* input_bytes,
    unsigned long long* output_bytes) {
  std::ifstream data_in(input_path, std::ios::in | std::ios::binary);
  if (!data_in.is_open()) return false;

  data_in.seekg(0, std::ios::end);
  *input_bytes = data_in.tellg();
  data_in.seekg(0, std::ios::beg);
  std::vector<bool> vocab(256, false);
  bool dictionary_used;
  ReadHeader(&data_in, output_bytes, &dictionary_used, &vocab);
  if (!dictionary_used && dictionary != NULL) return false;
  if (dictionary_used && dictionary == NULL) return false;

  if (*output_bytes == 0) {  // undo store
    data_in.close();
    FILE* in = fopen(input_path.c_str(), "rb");
    if (!in) return false;
    FILE* data_out = fopen(output_path.c_str(), "wb");
    if (!data_out) return false;
    fseek(in, 5L, SEEK_SET);
    fprintf(stderr, "\rdecoding...");
    fflush(stderr);
    preprocessor::Decode(in, data_out, dictionary);
    fseek(data_out, 0L, SEEK_END);
    *output_bytes = ftell(data_out);
    fclose(in);
    fclose(data_out);
    return true;
  }
  paq8::setLightMode(false);
  Predictor p(vocab, *output_bytes);
  if (dictionary_used) preprocessor::Pretrain(&p, dictionary);

  std::ofstream temp_out(temp_path, std::ios::out | std::ios::binary);
  if (!temp_out.is_open()) return false;

  Decompress(*output_bytes, &data_in, &temp_out, &p);
  data_in.close();
  temp_out.close();

  FILE* temp_in = fopen(temp_path.c_str(), "rb");
  if (!temp_in) return false;
  FILE* data_out = fopen(output_path.c_str(), "wb");
  if (!data_out) return false;

  preprocessor::Decode(temp_in, data_out, dictionary);
  fseek(data_out, 0L, SEEK_END);
  *output_bytes = ftell(data_out);
  fclose(temp_in);
  fclose(data_out);
  remove(temp_path.c_str());
  return true;
}

int main(int argc, char* argv[]) {
  // Detect CPU SIMD support and print selected architecture
  auto detect_and_print_simd = [](){
    int info[4] = {0,0,0,0};
    __cpuid(info, 0);
    int nIds = info[0];
    bool sse = false, sse2 = false, sse3 = false, ssse3 = false;
    bool sse41 = false, sse42 = false, avx = false, avx2 = false, avx512f = false;
    if (nIds >= 1) {
      __cpuid(info, 1);
      int ecx = info[2];
      int edx = info[3];
      sse3 = (ecx & (1 << 0)) != 0;
      ssse3 = (ecx & (1 << 9)) != 0;
      sse41 = (ecx & (1 << 19)) != 0;
      sse42 = (ecx & (1 << 20)) != 0;
      avx = (ecx & (1 << 28)) != 0;
      sse = (edx & (1 << 25)) != 0;
      sse2 = (edx & (1 << 26)) != 0;
    }
    if (nIds >= 7) {
      int info7[4] = {0,0,0,0};
      __cpuidex(info7, 7, 0);
      int ebx = info7[1];
      avx2 = (ebx & (1 << 5)) != 0;
      avx512f = (ebx & (1 << 16)) != 0; // AVX-512 Foundation
    }

    unsigned long long xcr0 = 0;
#if defined(_XCR_XFEATURE_ENABLED_MASK)
    xcr0 = _xgetbv(0);
#elif defined(__GNUC__)
    // no-op for compilers without _xgetbv macro
#endif
    bool os_avx = (xcr0 & 0x6) == 0x6; // XCR0[2:1] == 11 for AVX
    bool os_avx512 = (xcr0 & 0xE0) == 0xE0; // XCR0[7:5] == 111 for AVX-512

    bool have_avx = avx && os_avx;
    bool have_avx2 = avx2 && have_avx;
    bool have_avx512 = avx512f && have_avx && os_avx512;

    fprintf(stderr, "SIMD support:\n");
    fprintf(stderr, "  SSE : %s\n", sse ? "YES" : "NO");
    fprintf(stderr, "  SSE2: %s\n", sse2 ? "YES" : "NO");
    fprintf(stderr, "  SSE3: %s\n", sse3 ? "YES" : "NO");
    fprintf(stderr, "  SSSE3: %s\n", ssse3 ? "YES" : "NO");
    fprintf(stderr, "  SSE4.1: %s\n", sse41 ? "YES" : "NO");
    fprintf(stderr, "  SSE4.2: %s\n", sse42 ? "YES" : "NO");
    fprintf(stderr, "  AVX : %s\n", have_avx ? "YES" : "NO");
    fprintf(stderr, "  AVX2: %s\n", have_avx2 ? "YES" : "NO");
    fprintf(stderr, "  AVX-512F: %s\n", have_avx512 ? "YES" : "NO");

    const char* selected = "SCALAR";
    if (have_avx512) selected = "AVX512";
    else if (have_avx2) selected = "AVX2";
    else if (have_avx) selected = "AVX";
    else if (sse42) selected = "SSE4.2";
    else if (sse2) selected = "SSE2";

    fprintf(stderr, "Selected architecture: %s\n", selected);
  };
  detect_and_print_simd();
  if (argc < 4) return Help();

  unsigned long long memory_limit = 0;
  // Accept -m anywhere on the command line (not only as the first arg).
  for (int i = 1; i < argc; ++i) {
    if (argv[i][0] == '-' && argv[i][1] == 'm') {
      char* endptr;
      unsigned long long m = strtoull(argv[i] + 2, &endptr, 10);
      if (*endptr == 'G' || *endptr == 'g') m <<= 30;
      else if (*endptr == 'M' || *endptr == 'm') m <<= 20;
      else if (*endptr == 'K' || *endptr == 'k') m <<= 10;
      if (m > 0) paq8::setMaxMem(m);
      memory_limit = m;
    }
  }

  // Remove any -m arguments from argv so the rest of the parsing (which
  // expects optional single-character flags before the main command) works
  // regardless of -m position.
  int write = 1;
  for (int i = 1; i < argc; ++i) {
    if (argv[i][0] == '-' && argv[i][1] == 'm') continue;
    argv[write++] = argv[i];
  }
  argc = write;

  int arg_idx = 1;

  if (argc - arg_idx < 3 || argc - arg_idx > 4 || strlen(argv[arg_idx]) != 2 || argv[arg_idx][0] != '-' ||
      (argv[arg_idx][1] != 'c' && argv[arg_idx][1] != 'd' && argv[arg_idx][1] != 's' &&
      argv[arg_idx][1] != 'n' && argv[arg_idx][1] != 't')) {
    return Help();
  }

  bool enable_preprocess = true;
  bool text_mode = false;

  // Handle optional flags before the main command (e.g., -t -c input output)
  while (arg_idx < argc && argv[arg_idx][0] == '-' && strlen(argv[arg_idx]) == 2) {
    if (argv[arg_idx][1] == 't') {
      text_mode = true;
      arg_idx++;
    } else {
      break;
    }
  }

  if (argc - arg_idx < 3) {
    return Help();
  }

  if (argv[arg_idx][1] == 'n') enable_preprocess = false;

  // Save the command (c, d, n, s, t)
  char command = argv[arg_idx][1];
  if (command == 't') text_mode = true;

  std::string input_path;
  std::string output_path;
  FILE* dictionary = NULL;

  if (argc - arg_idx == 4) {
    if (command == 'n') return Help();
    dictionary = fopen(argv[arg_idx + 1], "rb");
    if (!dictionary) return Help();
    dictionary_path = argv[arg_idx + 1];
    input_path = argv[arg_idx + 2];
    output_path = argv[arg_idx + 3];
  } else {
    input_path = argv[arg_idx + 1];
    output_path = argv[arg_idx + 2];
  }

  clock_t start = clock();

  std::string temp_path = output_path + ".cmix.temp";

  unsigned long long input_bytes = 0, output_bytes = 0;

  if (command == 's') {
    if (!Store(input_path, temp_path, output_path, dictionary, &input_bytes,
        &output_bytes)) {
      return Help();
    }
  } else if (command == 'c' || command == 'n' || command == 't') {
    if (!RunCompression(enable_preprocess, text_mode, input_path, temp_path,
        output_path, dictionary, &input_bytes, &output_bytes)) {
      return Help();
    }
  } else if (command == 'd') {
    if (!RunDecompression(input_path, temp_path, output_path, dictionary,
        &input_bytes, &output_bytes)) {
      return Help();
    }
  } else {
    return Help();
  }

  long long total_seconds = (long long)((double)clock() - start) / CLOCKS_PER_SEC;
  printf("\r%lld bytes -> %lld bytes in %02lld:%02lld:%02lld.\n",
      input_bytes, output_bytes,
      total_seconds / 3600, (total_seconds % 3600) / 60, total_seconds % 60);

  if (command == 'c' || command == 'n' || command == 't') {
    double cross_entropy = output_bytes;
    cross_entropy /= input_bytes;
    cross_entropy *= 8;
    printf("cross entropy: %.3f\n", cross_entropy);
  }

  return 0;
}
