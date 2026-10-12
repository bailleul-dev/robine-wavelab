#pragma once
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Which upstream implementation to instantiate.
enum nam_ref_mode {
  /// Upstream dispatcher (`nam::get_dsp`): A2-shaped WaveNets take the
  /// upstream A2 fast path, everything else the generic path.
  NAM_REF_MODE_AUTO = 0,
  /// Generic `nam::wavenet::WaveNet` (bypasses the A2 fast path).
  NAM_REF_MODE_GENERIC = 1,
  /// Upstream `A2FastModel` (fails if the model is not A2-shaped).
  NAM_REF_MODE_A2_FAST = 2,
};

typedef struct nam_ref_model nam_ref_model;

/// Build a model from `.nam` JSON text. For a SlimmableContainer, `slim`
/// (0..1) selects the submodel like `ContainerModel::SetSlimmableSize`.
/// The model is reset to `max_block` and prewarmed. On failure returns NULL
/// and writes a message into `err`.
nam_ref_model* nam_ref_create(const char* json, size_t json_len, int mode, double slim, int max_block,
                              char* err, size_t err_len);
void nam_ref_destroy(nam_ref_model* m);
/// Expected sample rate (-1 if unknown).
double nam_ref_sample_rate(const nam_ref_model* m);
int nam_ref_prewarm_samples(nam_ref_model* m);
/// Human readable description of the instantiated class.
const char* nam_ref_description(const nam_ref_model* m);
/// Reset and prewarm again.
void nam_ref_reset(nam_ref_model* m);
/// Process `n` frames (`n <= max_block`).
void nam_ref_process(nam_ref_model* m, const float* input, float* output, int n);

#ifdef __cplusplus
}
#endif
