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

#ifndef LIBSPIRV_COMP_MARKV_MODEL_H_
#define LIBSPIRV_COMP_MARKV_MODEL_H_

#include <map>
#include <unordered_set>
#include <vector>

#include "spirv/1.2/spirv.h"
#include "spirv-tools/libspirv.h"
#include "util/huffman_codec.h"

namespace spvtools {

// Base class for MARK-V models.
// The class contains encoding/decoding model with various constants and
// codecs used by the compression algorithm.
class MarkvModel {
 public:
  MarkvModel() : operand_chunk_lengths_(
      static_cast<size_t>(SPV_OPERAND_TYPE_NUM_OPERAND_TYPES), 0) {}

  uint32_t model_type() const { return model_type_; }
  uint32_t model_version() const { return model_version_; }

  uint32_t opcode_chunk_length() const { return opcode_chunk_length_; }
  uint32_t num_operands_chunk_length() const { return num_operands_chunk_length_; }
  uint32_t mtf_rank_chunk_length() const { return mtf_rank_chunk_length_; }

  uint32_t u64_chunk_length() const { return u64_chunk_length_; }
  uint32_t s64_chunk_length() const { return s64_chunk_length_; }
  uint32_t s64_block_exponent() const { return s64_block_exponent_; }

  // Returns a codec for common opcode_and_num_operands words for the given
  // previous opcode. May return nullptr if the codec doesn't exist.
  const spvutils::HuffmanCodec<uint64_t>* GetOpcodeAndNumOperandsMarkovHuffmanCodec(
      uint32_t prev_opcode) const {
    if (prev_opcode == SpvOpNop)
      return opcode_and_num_operands_huffman_codec_.get();

    const auto it =
        opcode_and_num_operands_markov_huffman_codecs_.find(prev_opcode);
    if (it == opcode_and_num_operands_markov_huffman_codecs_.end())
      return nullptr;
    return it->second.get();
  }

  // Returns a codec for common non-id words used for given operand slot.
  // Operand slot is defined by the opcode and the operand index.
  // May return nullptr if the codec doesn't exist.
  const spvutils::HuffmanCodec<uint64_t>* GetNonIdWordHuffmanCodec(
      uint32_t opcode, uint32_t operand_index) const {
    const auto it = non_id_word_huffman_codecs_.find(
        std::pair<uint32_t, uint32_t>(opcode, operand_index));
    if (it == non_id_word_huffman_codecs_.end())
      return nullptr;
    return it->second.get();
  }

  // Returns a codec for common id descriptos used for given operand slot.
  // Operand slot is defined by the opcode and the operand index.
  // May return nullptr if the codec doesn't exist.
  const spvutils::HuffmanCodec<uint64_t>* GetIdDescriptorHuffmanCodec(
      uint32_t opcode, uint32_t operand_index) const {
    const auto it = id_descriptor_huffman_codecs_.find(
        std::pair<uint32_t, uint32_t>(opcode, operand_index));
    if (it == id_descriptor_huffman_codecs_.end())
      return nullptr;
    return it->second.get();
  }

  // Returns a codec for common strings used by the given opcode.
  // Operand slot is defined by the opcode and the operand index.
  // May return nullptr if the codec doesn't exist.
  const spvutils::HuffmanCodec<std::string>* GetLiteralStringHuffmanCodec(
      uint32_t opcode) const {
    const auto it = literal_string_huffman_codecs_.find(opcode);
    if (it == literal_string_huffman_codecs_.end())
      return nullptr;
    return it->second.get();
  }

  // Checks if |descriptor| has a coding scheme in any of
  // id_descriptor_huffman_codecs_.
  bool DescriptorHasCodingScheme(uint32_t descriptor) const {
    return descriptors_with_coding_scheme_.count(descriptor);
  }

  // Returns chunk length used for variable length encoding of spirv operand
  // words.
  uint32_t GetOperandVariableWidthChunkLength(spv_operand_type_t type) const {
    return operand_chunk_lengths_.at(static_cast<size_t>(type));
  }

  // Sets model type.
  void SetModelType(uint32_t in_model_type) {
    model_type_ = in_model_type;
  }

  // Sets model version.
  void SetModelVersion(uint32_t in_model_version) {
    model_version_ = in_model_version;
  }

  // Returns value used by Huffman codecs as a signal that a value is not in the
  // coding table.
  static uint64_t GetMarkvNoneOfTheAbove() {
    // Magic number.
    return 1111111111111111111;
  }

 protected:
  // Huffman codec for base-rate of opcode_and_num_operands.
  std::unique_ptr<spvutils::HuffmanCodec<uint64_t>>
      opcode_and_num_operands_huffman_codec_;

  // Huffman codecs for opcode_and_num_operands. The map key is previous opcode.
  std::map<uint32_t, std::unique_ptr<spvutils::HuffmanCodec<uint64_t>>>
      opcode_and_num_operands_markov_huffman_codecs_;

  // Huffman codecs for non-id single-word operand values.
  // The map key is pair <opcode, operand_index>.
  std::map<std::pair<uint32_t, uint32_t>,
      std::unique_ptr<spvutils::HuffmanCodec<uint64_t>>> non_id_word_huffman_codecs_;

  // Huffman codecs for id descriptors. The map key is pair
  // <opcode, operand_index>.
  std::map<std::pair<uint32_t, uint32_t>,
      std::unique_ptr<spvutils::HuffmanCodec<uint64_t>>> id_descriptor_huffman_codecs_;

  // Set of all descriptors which have a coding scheme in any of
  // id_descriptor_huffman_codecs_.
  std::unordered_set<uint32_t> descriptors_with_coding_scheme_;

  // Huffman codecs for literal strings. The map key is the opcode of the
  // current instruction. This assumes, that there is no more than one literal
  // string operand per instruction, but would still work even if this is not
  // the case. Names and debug information strings are not collected.
  std::map<uint32_t, std::unique_ptr<spvutils::HuffmanCodec<std::string>>>
      literal_string_huffman_codecs_;

  // Chunk lengths used for variable width encoding of operands (index is
  // spv_operand_type of the operand).
  std::vector<uint32_t> operand_chunk_lengths_;

  uint32_t opcode_chunk_length_ = 7;
  uint32_t num_operands_chunk_length_ =  3;
  uint32_t mtf_rank_chunk_length_ = 5;

  uint32_t u64_chunk_length_ = 8;
  uint32_t s64_chunk_length_ = 8;
  uint32_t s64_block_exponent_ = 10;

  uint32_t model_type_ = 0;
  uint32_t model_version_ = 0;
};

}  // namespace spvtools

#endif  // LIBSPIRV_COMP_MARKV_MODEL_H_
