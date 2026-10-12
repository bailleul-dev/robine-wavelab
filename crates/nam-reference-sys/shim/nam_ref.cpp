// Thin C shim over NeuralAmpModelerCore so the Rust tools can drive the
// upstream implementation in-process with float buffers.
#include "nam_ref.h"

#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "json.hpp"
#include "NAM/dsp.h"
#include "NAM/get_dsp.h"
#include "NAM/wavenet/a2_fast.h"
#include "NAM/wavenet/model.h"

struct nam_ref_model
{
  std::unique_ptr<nam::DSP> dsp;
  double sample_rate = -1.0;
  int max_block = 0;
  std::string description;
  std::vector<NAM_SAMPLE> in;
  std::vector<NAM_SAMPLE> out;
};

namespace
{

nlohmann::json select_submodel(const nlohmann::json& container_config, double slim)
{
  const auto& submodels = container_config.at("submodels");
  if (!submodels.is_array() || submodels.empty())
    throw std::runtime_error("SlimmableContainer has no submodels");
  size_t idx = submodels.size() - 1;
  for (size_t i = 0; i < submodels.size(); i++)
  {
    if (slim < submodels[i].at("max_value").get<double>())
    {
      idx = i;
      break;
    }
  }
  return submodels[idx].at("model");
}

void set_error(char* err, size_t err_len, const std::string& msg)
{
  if (!err || err_len == 0)
    return;
  const size_t n = std::min(err_len - 1, msg.size());
  std::memcpy(err, msg.data(), n);
  err[n] = '\0';
}

} // namespace

extern "C" {

nam_ref_model* nam_ref_create(const char* json, size_t json_len, int mode, double slim, int max_block, char* err,
                              size_t err_len)
{
  try
  {
    nlohmann::json j = nlohmann::json::parse(json, json + json_len);
    std::string arch = j.value("architecture", std::string());
    // Resolve the container ourselves so that every mode sees the same submodel.
    if (arch == "SlimmableContainer")
    {
      nlohmann::json sub = select_submodel(j.at("config"), slim);
      if (!sub.contains("sample_rate") && j.contains("sample_rate"))
        sub["sample_rate"] = j["sample_rate"];
      if (!sub.contains("metadata") && j.contains("metadata"))
        sub["metadata"] = j["metadata"];
      j = std::move(sub);
      arch = j.value("architecture", std::string());
    }

    auto m = std::make_unique<nam_ref_model>();
    m->sample_rate = nam::get_sample_rate_from_nam_file(j);
    m->max_block = max_block;

    if (mode == NAM_REF_MODE_AUTO)
    {
      m->dsp = nam::get_dsp(j);
      int ch = 0;
      const bool fast = arch == "WaveNet" && nam::wavenet::a2_fast::is_a2_shape(j.at("config"), &ch);
      m->description = fast ? ("upstream A2FastModel<" + std::to_string(ch) + "> (auto)") : ("upstream " + arch + " (auto)");
    }
    else
    {
      if (arch != "WaveNet")
        throw std::runtime_error("mode requires a WaveNet model, got " + arch);
      std::vector<float> weights = j.at("weights").get<std::vector<float>>();
      if (mode == NAM_REF_MODE_GENERIC)
      {
        auto cfg = nam::wavenet::parse_config_json(j.at("config"), m->sample_rate);
        m->dsp = cfg.create(std::move(weights), m->sample_rate);
        m->description = "upstream generic WaveNet";
      }
      else if (mode == NAM_REF_MODE_A2_FAST)
      {
        int ch = 0;
        if (!nam::wavenet::a2_fast::is_a2_shape(j.at("config"), &ch))
          throw std::runtime_error("model is not A2-shaped");
        auto cfg = nam::wavenet::a2_fast::create_a2_fast_config(j.at("config"), m->sample_rate);
        m->dsp = cfg->create(std::move(weights), m->sample_rate);
        m->description = "upstream A2FastModel<" + std::to_string(ch) + ">";
      }
      else
      {
        throw std::runtime_error("unknown mode");
      }
    }

    m->in.assign(static_cast<size_t>(max_block), 0.0);
    m->out.assign(static_cast<size_t>(max_block), 0.0);
    m->dsp->ResetAndPrewarm(m->sample_rate, max_block);
    return m.release();
  }
  catch (const std::exception& e)
  {
    set_error(err, err_len, e.what());
    return nullptr;
  }
}

void nam_ref_destroy(nam_ref_model* m)
{
  delete m;
}

double nam_ref_sample_rate(const nam_ref_model* m)
{
  return m->sample_rate;
}

int nam_ref_prewarm_samples(nam_ref_model* m)
{
  return m->dsp->GetPrewarmSamples();
}

const char* nam_ref_description(const nam_ref_model* m)
{
  return m->description.c_str();
}

void nam_ref_reset(nam_ref_model* m)
{
  m->dsp->ResetAndPrewarm(m->sample_rate, m->max_block);
}

void nam_ref_process(nam_ref_model* m, const float* input, float* output, int n)
{
  for (int i = 0; i < n; i++)
    m->in[static_cast<size_t>(i)] = static_cast<NAM_SAMPLE>(input[i]);
  NAM_SAMPLE* in_ptrs[1] = {m->in.data()};
  NAM_SAMPLE* out_ptrs[1] = {m->out.data()};
  m->dsp->process(in_ptrs, out_ptrs, n);
  for (int i = 0; i < n; i++)
    output[i] = static_cast<float>(m->out[static_cast<size_t>(i)]);
}

} // extern "C"
