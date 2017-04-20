// Copyright (c) 2017 Google Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef LIBSPIRV_TOOLS_STATS_STATS_ANALYZER_H_
#define LIBSPIRV_TOOLS_STATS_STATS_ANALYZER_H_

#include <unordered_map>

#include "source/spirv_stats.h"

class StatsAnalyzer {
 public:
  explicit StatsAnalyzer(const libspirv::SpirvStats& stats);

  void WriteVersion(std::ostream& out);
  void WriteGenerator(std::ostream& out);
  void WriteCapability(std::ostream& out);
  void WriteExtension(std::ostream& out);
  void WriteOpcode(std::ostream& out);
  void WriteOpcodeMarkov(std::ostream& out);

 private:
  const libspirv::SpirvStats& stats_;

  uint32_t num_modules_;

  std::unordered_map<uint32_t, double> version_freq_;
  std::unordered_map<uint32_t, double> generator_freq_;
  std::unordered_map<uint32_t, double> capability_freq_;
  std::unordered_map<std::string, double> extension_freq_;
  std::unordered_map<uint32_t, double> opcode_freq_;
};


#endif  // LIBSPIRV_TOOLS_STATS_STATS_ANALYZER_H_
