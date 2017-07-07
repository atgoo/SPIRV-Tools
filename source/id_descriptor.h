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

#ifndef LIBSPIRV_ID_DESCRIPTOR_H_
#define LIBSPIRV_ID_DESCRIPTOR_H_

#include <unordered_map>
#include <vector>

#include "spirv-tools/libspirv.hpp"

namespace libspirv {

class IdDescriptorCollection {
 public:
  IdDescriptorCollection() {
    words_.reserve(16);
  }

  uint32_t IssueNewDescriptor(const spv_parsed_instruction_t& inst);

  uint32_t GetDescriptor(uint32_t id) const {
    const auto it = id_to_descriptor_.find(id);
    if (it == id_to_descriptor_.end())
      return 0;
    return it->second;
  }

 private:
  std::unordered_map<uint32_t, uint32_t> id_to_descriptor_;
  std::vector<uint32_t> words_;
};

}  // namespace libspirv

#endif  // LIBSPIRV_ID_DESCRIPTOR_H_

