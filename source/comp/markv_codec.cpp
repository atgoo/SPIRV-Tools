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

// Contains
//   - SPIR-V to MARK-V encoder
//   - MARK-V to SPIR-V decoder
//
// MARK-V is a compression format for SPIR-V binaries. It strips away
// non-essential information (such as result ids which can be regenerated) and
// uses various bit reduction techiniques to reduce the size of the binary.

#include <algorithm>
#include <cassert>
#include <cstring>
#include <functional>
#include <iostream>
#include <iterator>
#include <list>
#include <memory>
#include <numeric>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "spirv/1.2/spirv.h"

#include "binary.h"
#include "diagnostic.h"
#include "enum_string_mapping.h"
#include "ext_inst.h"
#include "extensions.h"
#include "id_descriptor.h"
#include "instruction.h"
#include "markv.h"
#include "markv_model.h"
#include "opcode.h"
#include "operand.h"
#include "spirv-tools/libspirv.h"
#include "spirv_endian.h"
#include "spirv_validator_options.h"
#include "util/bit_stream.h"
#include "util/huffman_codec.h"
#include "util/move_to_front.h"
#include "util/parse_number.h"
#include "val/instruction.h"
#include "val/validation_state.h"
#include "validate.h"

using libspirv::IdDescriptorCollection;
using libspirv::Instruction;
using libspirv::ValidationState_t;
using libspirv::DiagnosticStream;
using spvutils::BitReaderWord64;
using spvutils::BitWriterWord64;
using spvutils::HuffmanCodec;
using MoveToFront = spvutils::MoveToFront<uint32_t>;
using MultiMoveToFront = spvutils::MultiMoveToFront<uint32_t>;

namespace spvtools {

namespace {

const uint32_t kSpirvMagicNumber = SpvMagicNumber;
const uint32_t kMarkvMagicNumber = 0x07230303;

const uint32_t kMtfForwardDeclared = 0;

// Signals that the value is not in the coding scheme and a fallback method
// needs to be used.
const uint64_t kMarkvNoneOfTheAbove = MarkvModel::GetMarkvNoneOfTheAbove();

// Mtf ranks smaller than this are encoded with Huffman coding.
const uint32_t kMtfSmallestRankEncodedByValue = 10;

// Signals that the mtf rank is too large to be encoded with Huffman.
const uint32_t kMtfRankEncodedByValueSignal =
    std::numeric_limits<uint32_t>::max();

const size_t kCommentNumWhitespaces = 2;

const size_t kByteBreakAfterInstIfLessThanUntilNextByte = 8;

std::map<uint32_t, uint32_t> GetMtfRankHist() {
  return std::map<uint32_t, uint32_t>({
      {1, 50},
      {2, 20},
      {3, 5},
      {4, 5},
      {5, 2},
      {6, 1},
      {7, 1},
      {8, 1},
      {9, 1},
      {kMtfRankEncodedByValueSignal, 10},
  });
}

// Returns true if the opcode has a fixed number of operands. May return a
// false negative.
bool OpcodeHasFixedNumberOfOperands(SpvOp opcode) {
  switch (opcode) {
    // TODO(atgoo@github.com) This is not a complete list.
    case SpvOpNop:
    case SpvOpName:
    case SpvOpUndef:
    case SpvOpSizeOf:
    case SpvOpLine:
    case SpvOpNoLine:
    case SpvOpDecorationGroup:
    case SpvOpExtension:
    case SpvOpExtInstImport:
    case SpvOpMemoryModel:
    case SpvOpCapability:
    case SpvOpTypeVoid:
    case SpvOpTypeBool:
    case SpvOpTypeInt:
    case SpvOpTypeFloat:
    case SpvOpTypeVector:
    case SpvOpTypeMatrix:
    case SpvOpTypeSampler:
    case SpvOpTypeSampledImage:
    case SpvOpTypeArray:
    case SpvOpTypePointer:
    case SpvOpConstantTrue:
    case SpvOpConstantFalse:
    case SpvOpLabel:
    case SpvOpBranch:
    case SpvOpFunction:
    case SpvOpFunctionParameter:
    case SpvOpFunctionEnd:
    case SpvOpBitcast:
    case SpvOpCopyObject:
    case SpvOpTranspose:
    case SpvOpSNegate:
    case SpvOpFNegate:
    case SpvOpIAdd:
    case SpvOpFAdd:
    case SpvOpISub:
    case SpvOpFSub:
    case SpvOpIMul:
    case SpvOpFMul:
    case SpvOpUDiv:
    case SpvOpSDiv:
    case SpvOpFDiv:
    case SpvOpUMod:
    case SpvOpSRem:
    case SpvOpSMod:
    case SpvOpFRem:
    case SpvOpFMod:
    case SpvOpVectorTimesScalar:
    case SpvOpMatrixTimesScalar:
    case SpvOpVectorTimesMatrix:
    case SpvOpMatrixTimesVector:
    case SpvOpMatrixTimesMatrix:
    case SpvOpOuterProduct:
    case SpvOpDot:
      return true;
    default:
      break;
  }
  return false;
}

size_t GetNumBitsToNextByte(size_t bit_pos) { return (8 - (bit_pos % 8)) % 8; }

// Defines and returns current MARK-V version.
uint32_t GetMarkvVersion() {
  const uint32_t kVersionMajor = 1;
  const uint32_t kVersionMinor = 3;
  return kVersionMinor | (kVersionMajor << 16);
}

class MarkvLogger {
 public:
  MarkvLogger(MarkvLogConsumer log_consumer, MarkvDebugConsumer debug_consumer)
      : log_consumer_(log_consumer), debug_consumer_(debug_consumer) {}

  void AppendText(const std::string& str) {
    Append(str);
    use_delimiter_ = false;
  }

  void AppendTextNewLine(const std::string& str) {
    Append(str);
    Append("\n");
    use_delimiter_ = false;
  }

  void AppendBitSequence(const std::string& str) {
    if (debug_consumer_) instruction_bits_ << str;
    if (use_delimiter_) Append("-");
    Append(str);
    use_delimiter_ = true;
  }

  void AppendWhitespaces(size_t num) {
    Append(std::string(num, ' '));
    use_delimiter_ = false;
  }

  void NewLine() {
    Append("\n");
    use_delimiter_ = false;
  }

  bool DebugInstruction(const spv_parsed_instruction_t& inst) {
    bool result = true;
    if (debug_consumer_) {
      result = debug_consumer_(
          std::vector<uint32_t>(inst.words, inst.words + inst.num_words),
          instruction_bits_.str(), instruction_comment_.str());
      instruction_bits_.str(std::string());
      instruction_comment_.str(std::string());
    }
    return result;
  }

 private:
  MarkvLogger(const MarkvLogger&) = delete;
  MarkvLogger(MarkvLogger&&) = delete;
  MarkvLogger& operator=(const MarkvLogger&) = delete;
  MarkvLogger& operator=(MarkvLogger&&) = delete;

  void Append(const std::string& str) {
    if (log_consumer_) log_consumer_(str);
    if (debug_consumer_) instruction_comment_ << str;
  }

  MarkvLogConsumer log_consumer_;
  MarkvDebugConsumer debug_consumer_;

  std::stringstream instruction_bits_;
  std::stringstream instruction_comment_;

  // If true a delimiter will be appended before the next bit sequence.
  // Used to generate outputs like: 1100-0 1110-1-1100-1-1111-0 110-0.
  bool use_delimiter_ = false;
};

// Base class for MARK-V encoder and decoder. Contains common functionality
// such as:
// - Validator connection and validation state.
// - SPIR-V grammar and helper functions.
class MarkvCodecBase {
 public:
  virtual ~MarkvCodecBase() { spvValidatorOptionsDestroy(validator_options_); }

  MarkvCodecBase() = delete;

 protected:
  struct MarkvHeader {
    MarkvHeader() {
      magic_number = kMarkvMagicNumber;
      markv_version = GetMarkvVersion();
      markv_model = 0;
      markv_length_in_bits = 0;
      spirv_version = 0;
      spirv_generator = 0;
    }

    uint32_t magic_number;
    uint32_t markv_version;
    // Magic number to identify or verify MarkvModel used for encoding.
    uint32_t markv_model;
    uint32_t markv_length_in_bits;
    uint32_t spirv_version;
    uint32_t spirv_generator;
  };

  // |model| is owned by the caller, must be not null and valid during the
  // lifetime of the codec.
  explicit MarkvCodecBase(spv_const_context context,
                          spv_validator_options validator_options,
                          const MarkvModel* model)
      : validator_options_(validator_options),
        grammar_(context),
        model_(model),
        mtf_huffman_codec_(GetMtfRankHist()),
        context_(context),
        vstate_(validator_options
                    ? new ValidationState_t(context, validator_options_)
                    : nullptr) {}
/*                    : nullptr),
        mtfs_(1 << IdDescriptorCollection::GetBitWidth()) {}
        */

  // Validates a single instruction and updates validation state of the module.
  // Does nothing and returns SPV_SUCCESS if validator was not created.
  spv_result_t UpdateValidationState(const spv_parsed_instruction_t& inst) {
    if (!vstate_) return SPV_SUCCESS;

    return ValidateInstructionAndUpdateValidationState(vstate_.get(), &inst);
  }

  // Returns instruction which created |id| or nullptr if such instruction was
  // not registered.
  const Instruction* FindDef(uint32_t id) const {
    const auto it = id_to_def_instruction_.find(id);
    if (it == id_to_def_instruction_.end()) return nullptr;
    return it->second;
  }

  // Returns type id of vector type component.
  uint32_t GetVectorComponentType(uint32_t vector_type_id) const {
    const Instruction* type_inst = FindDef(vector_type_id);
    assert(type_inst);
    assert(type_inst->opcode() == SpvOpTypeVector);

    const uint32_t component_type =
        type_inst->word(type_inst->operands()[1].offset);
    return component_type;
  }

  // Process data from the current instruction. This would update MTFs and
  // other data containers.
  void ProcessCurInstruction();

  // Returns words of the current instruction. Decoder has a different
  // implementation and the array is valid only until the previously decoded
  // word.
  virtual const uint32_t* GetInstWords() const { return inst_.words; }

  // Returns the opcode of the previous instruction.
  SpvOp GetPrevOpcode() const {
    if (instructions_.empty()) return SpvOpNop;

    return instructions_.back()->opcode();
  }

  // Returns diagnostic stream, position index is set to instruction number.
  DiagnosticStream Diag(spv_result_t error_code) const {
    return DiagnosticStream({0, 0, instructions_.size()}, context_->consumer,
                            error_code);
  }

  // Returns current id bound.
  uint32_t GetIdBound() const { return id_bound_; }

  // Sets current id bound, expected to be no lower than the previous one.
  void SetIdBound(uint32_t id_bound) {
    assert(id_bound >= id_bound_);
    id_bound_ = id_bound;
    if (vstate_) vstate_->setIdBound(id_bound);
  }

  MoveToFront& GetMtf(uint32_t handle) {
    return mtfs_[handle];
  }

  spv_validator_options validator_options_ = nullptr;
  const libspirv::AssemblyGrammar grammar_;
  MarkvHeader header_;

  // MARK-V model, not owned.
  const MarkvModel* model_ = nullptr;

  // Current instruction, current operand and current operand index.
  spv_parsed_instruction_t inst_;
  spv_parsed_operand_t operand_;
  uint32_t operand_index_;

  // Maps a result ID to its type ID.  By convention:
  //  - a result ID that is a type definition maps to itself.
  //  - a result ID without a type maps to 0.  (E.g. for OpLabel)
  std::unordered_map<uint32_t, uint32_t> id_to_type_id_;

  // Id of the current function or zero if outside of function.
  uint32_t cur_function_id_ = 0;

  // List of ids local to the current function.
  std::vector<uint32_t> ids_local_to_cur_function_;

  // List of instructions in the order they are given in the module.
  std::vector<std::unique_ptr<const Instruction>> instructions_;

  // Container/computer for id descriptors.
  IdDescriptorCollection id_descriptors_;

  // Huffman codec for move-to-front ranks.
  HuffmanCodec<uint32_t> mtf_huffman_codec_;

  // If not nullptr, codec will log comments on the compression process.
  std::unique_ptr<MarkvLogger> logger_;

 private:
  spv_const_context context_ = nullptr;

  std::unique_ptr<ValidationState_t> vstate_;

  // Maps result id to the instruction which defined it.
  std::unordered_map<uint32_t, const Instruction*> id_to_def_instruction_;

  uint32_t id_bound_ = 1;

  // Container for all move-to-front sequences.
  // std::vector<MoveToFront> mtfs_;
  std::map<uint32_t, MoveToFront> mtfs_;
};

// SPIR-V to MARK-V encoder. Exposes functions EncodeHeader and
// EncodeInstruction which can be used as callback by spvBinaryParse.
// Encoded binary is written to an internally maintained bitstream.
// After the last instruction is encoded, the resulting MARK-V binary can be
// acquired by calling GetMarkvBinary().
// The encoder uses SPIR-V validator to keep internal state, therefore
// SPIR-V binary needs to be able to pass validator checks.
// CreateCommentsLogger() can be used to enable the encoder to write comments
// on how encoding was done, which can later be accessed with GetComments().
class MarkvEncoder : public MarkvCodecBase {
 public:
  // |model| is owned by the caller, must be not null and valid during the
  // lifetime of MarkvEncoder.
  MarkvEncoder(spv_const_context context, const MarkvCodecOptions& options,
               const MarkvModel* model)
      : MarkvCodecBase(context, GetValidatorOptions(options), model),
        options_(options) {
    (void)options_;
  }

  // Writes data from SPIR-V header to MARK-V header.
  spv_result_t EncodeHeader(spv_endianness_t /* endian */, uint32_t /* magic */,
                            uint32_t version, uint32_t generator,
                            uint32_t id_bound, uint32_t /* schema */) {
    SetIdBound(id_bound);
    header_.spirv_version = version;
    header_.spirv_generator = generator;
    return SPV_SUCCESS;
  }

  // Creates an internal logger which writes comments on the encoding process.
  void CreateLogger(MarkvLogConsumer log_consumer,
                    MarkvDebugConsumer debug_consumer) {
    logger_.reset(new MarkvLogger(log_consumer, debug_consumer));
    writer_.SetCallback(
        [this](const std::string& str) { logger_->AppendBitSequence(str); });
  }

  // Encodes SPIR-V instruction to MARK-V and writes to bit stream.
  // Operation can fail if the instruction fails to pass the validator or if
  // the encoder stubmles on something unexpected.
  spv_result_t EncodeInstruction(const spv_parsed_instruction_t& inst);

  // Concatenates MARK-V header and the bit stream with encoded instructions
  // into a single buffer and returns it as spv_markv_binary. The returned
  // value is owned by the caller and needs to be destroyed with
  // spvMarkvBinaryDestroy().
  std::vector<uint8_t> GetMarkvBinary() {
    header_.markv_length_in_bits =
        static_cast<uint32_t>(sizeof(header_) * 8 + writer_.GetNumBits());
    header_.markv_model =
        (model_->model_type() << 16) | model_->model_version();

    const size_t num_bytes = sizeof(header_) + writer_.GetDataSizeBytes();
    std::vector<uint8_t> markv(num_bytes);

    assert(writer_.GetData());
    std::memcpy(markv.data(), &header_, sizeof(header_));
    std::memcpy(markv.data() + sizeof(header_), writer_.GetData(),
                writer_.GetDataSizeBytes());
    return markv;
  }

  // Optionally adds disassembly to the comments.
  // Disassembly should contain all instructions in the module separated by
  // \n, and no header.
  void SetDisassembly(std::string&& disassembly) {
    disassembly_.reset(new std::stringstream(std::move(disassembly)));
  }

  // Extracts the next instruction line from the disassembly and logs it.
  void LogDisassemblyInstruction() {
    if (logger_ && disassembly_) {
      std::string line;
      std::getline(*disassembly_, line, '\n');
      logger_->AppendTextNewLine(line);
    }
  }

 private:
  // Creates and returns validator options. Returned value owned by the caller.
  static spv_validator_options GetValidatorOptions(
      const MarkvCodecOptions& options) {
    return options.validate_spirv_binary ? spvValidatorOptionsCreate()
                                         : nullptr;
  }

  // Writes a single word to bit stream. operand_.type determines if the word is
  // encoded and how.
  spv_result_t EncodeNonIdWord(uint32_t word);

  // Writes both opcode and num_operands as a single code.
  // Returns SPV_UNSUPPORTED iff no suitable codec was found.
  spv_result_t EncodeOpcodeAndNumOperands(uint32_t opcode,
                                          uint32_t num_operands);

  // Writes mtf rank to bit stream.
  spv_result_t EncodeMtfRankHuffman(uint32_t rank);

  // Writes id using coding based on mtf associated with the id descriptor.
  // Returns SPV_UNSUPPORTED iff fallback method needs to be used.
  spv_result_t EncodeIdWithDescriptor(uint32_t id);

  // Writes result id of the current instruction if can't be inferred.
  spv_result_t EncodeResultId();

  // Writes ids which are neither type nor result ids.
  spv_result_t EncodeRefId(uint32_t id);

  // Writes bits to the stream until the beginning of the next byte if the
  // number of bits until the next byte is less than |byte_break_if_less_than|.
  void AddByteBreak(size_t byte_break_if_less_than);

  // Encodes a literal number operand and writes it to the bit stream.
  spv_result_t EncodeLiteralNumber(const spv_parsed_operand_t& operand);

  MarkvCodecOptions options_;

  // Bit stream where encoded instructions are written.
  BitWriterWord64 writer_;

  // If not nullptr, disassembled instruction lines will be written to comments.
  // Format: \n separated instruction lines, no header.
  std::unique_ptr<std::stringstream> disassembly_;
};

// Decodes MARK-V buffers written by MarkvEncoder.
class MarkvDecoder : public MarkvCodecBase {
 public:
  // |model| is owned by the caller, must be not null and valid during the
  // lifetime of MarkvEncoder.
  MarkvDecoder(spv_const_context context, const std::vector<uint8_t>& markv,
               const MarkvCodecOptions& options, const MarkvModel* model)
      : MarkvCodecBase(context, GetValidatorOptions(options), model),
        options_(options),
        reader_(markv) {
    (void)options_;
    SetIdBound(1);
    parsed_operands_.reserve(25);
    inst_words_.reserve(25);
  }

  // Creates an internal logger which writes comments on the decoding process.
  void CreateLogger(MarkvLogConsumer log_consumer,
                    MarkvDebugConsumer debug_consumer) {
    logger_.reset(new MarkvLogger(log_consumer, debug_consumer));
  }

  // Decodes SPIR-V from MARK-V and stores the words in |spirv_binary|.
  // Can be called only once. Fails if data of wrong format or ends prematurely,
  // of if validation fails.
  spv_result_t DecodeModule(std::vector<uint32_t>* spirv_binary);

 private:
  // Describes the format of a typed literal number.
  struct NumberType {
    spv_number_kind_t type;
    uint32_t bit_width;
  };

  // Creates and returns validator options. Returned value owned by the caller.
  static spv_validator_options GetValidatorOptions(
      const MarkvCodecOptions& options) {
    return options.validate_spirv_binary ? spvValidatorOptionsCreate()
                                         : nullptr;
  }

  // Reads a single bit from reader_. The read bit is stored in |bit|.
  // Returns false iff reader_ fails.
  bool ReadBit(bool* bit) {
    uint64_t bits = 0;
    const bool result = reader_.ReadBits(&bits, 1);
    if (result) *bit = bits ? true : false;
    return result;
  };

  // Returns ReadBit bound to the class object.
  std::function<bool(bool*)> GetReadBitCallback() {
    return std::bind(&MarkvDecoder::ReadBit, this, std::placeholders::_1);
  }

  // Reads a single non-id word from bit stream. operand_.type determines if
  // the word needs to be decoded and how.
  spv_result_t DecodeNonIdWord(uint32_t* word);

  // Reads and decodes both opcode and num_operands as a single code.
  // Returns SPV_UNSUPPORTED iff no suitable codec was found.
  spv_result_t DecodeOpcodeAndNumberOfOperands(uint32_t* opcode,
                                               uint32_t* num_operands);

  // Reads mtf rank from bit stream.
  spv_result_t DecodeMtfRankHuffman(uint32_t* rank);

  // Reads id using coding based on mtf associated with the id descriptor.
  // Returns SPV_UNSUPPORTED iff fallback method needs to be used.
  spv_result_t DecodeIdWithDescriptor(uint32_t* id);

  // Reads result id of the current instruction if can't be inferred.
  spv_result_t DecodeResultId();

  // Reads id which is neither type nor result id.
  spv_result_t DecodeRefId(uint32_t* id);

  // Reads and discards bits until the beginning of the next byte if the
  // number of bits until the next byte is less than |byte_break_if_less_than|.
  bool ReadToByteBreak(size_t byte_break_if_less_than);

  // Returns instruction words decoded up to this point.
  const uint32_t* GetInstWords() const override { return inst_words_.data(); }

  // Reads a literal number as it is described in |operand| from the bit stream,
  // decodes and writes it to spirv_.
  spv_result_t DecodeLiteralNumber(const spv_parsed_operand_t& operand);

  // Reads instruction from bit stream, decodes and validates it.
  // Decoded instruction is valid until the next call of DecodeInstruction().
  spv_result_t DecodeInstruction();

  // Read operand from the stream decodes and validates it.
  spv_result_t DecodeOperand(size_t operand_offset,
                             const spv_operand_type_t type,
                             spv_operand_pattern_t* expected_operands);

  // Records the numeric type for an operand according to the type information
  // associated with the given non-zero type Id.  This can fail if the type Id
  // is not a type Id, or if the type Id does not reference a scalar numeric
  // type.  On success, return SPV_SUCCESS and populates the num_words,
  // number_kind, and number_bit_width fields of parsed_operand.
  spv_result_t SetNumericTypeInfoForType(spv_parsed_operand_t* parsed_operand,
                                         uint32_t type_id);

  // Records the number type for the current instruction, if it generates a
  // type. For types that aren't scalar numbers, record something with number
  // kind SPV_NUMBER_NONE.
  void RecordNumberType();

  MarkvCodecOptions options_;

  // Temporary sink where decoded SPIR-V words are written. Once it contains the
  // entire module, the container is moved and returned.
  std::vector<uint32_t> spirv_;

  // Bit stream containing encoded data.
  BitReaderWord64 reader_;

  // Temporary storage for operands of the currently parsed instruction.
  // Valid until next DecodeInstruction call.
  std::vector<spv_parsed_operand_t> parsed_operands_;

  // Temporary storage for current instruction words.
  // Valid until next DecodeInstruction call.
  std::vector<uint32_t> inst_words_;

  // Maps a type ID to its number type description.
  std::unordered_map<uint32_t, NumberType> type_id_to_number_type_info_;

  // Maps an ExtInstImport id to the extended instruction type.
  std::unordered_map<uint32_t, spv_ext_inst_type_t> import_id_to_ext_inst_type_;
};

void MarkvCodecBase::ProcessCurInstruction() {
  instructions_.emplace_back(new Instruction(&inst_));

  const SpvOp opcode = SpvOp(inst_.opcode);

  if (inst_.result_id) {
    id_to_def_instruction_.emplace(inst_.result_id, instructions_.back().get());

    // Collect ids local to the current function.
    if (cur_function_id_) {
      ids_local_to_cur_function_.push_back(inst_.result_id);
    }

    // Starting new function.
    if (opcode == SpvOpFunction) {
      cur_function_id_ = inst_.result_id;

      // Store function parameter types in a queue, so that we know which types
      // to expect in the following OpFunctionParameter instructions.
      //const Instruction* def_inst = FindDef(inst_.words[4]);
      assert(def_inst);
      assert(def_inst->opcode() == SpvOpTypeFunction);
    }
  }

  // Remove local ids from MTFs if function end.
  if (opcode == SpvOpFunctionEnd) {
    cur_function_id_ = 0;
    for (uint32_t id : ids_local_to_cur_function_) {
      GetMtf(id_descriptors_.GetDescriptor(id)).Remove(id);
    }
    ids_local_to_cur_function_.clear();
  }

  if (!inst_.result_id) return;

  {
    // Save the result ID to type ID mapping.
    // In the grammar, type ID always appears before result ID.
    // A regular value maps to its type. Some instructions (e.g. OpLabel)
    // have no type Id, and will map to 0. The result Id for a
    // type-generating instruction (e.g. OpTypeInt) maps to itself.
    auto insertion_result = id_to_type_id_.emplace(
        inst_.result_id,
        spvOpcodeGeneratesType(SpvOp(inst_.opcode)) ? inst_.result_id
                                                    : inst_.type_id);
    (void)insertion_result;
    assert(insertion_result.second);
  }

  const uint32_t descriptor = id_descriptors_.ProcessInstruction(inst_);
  assert(descriptor);
  GetMtf(descriptor).Insert(inst_.result_id);
}

spv_result_t MarkvEncoder::EncodeNonIdWord(uint32_t word) {
  auto* codec = model_->GetNonIdWordHuffmanCodec(inst_.opcode, operand_index_);

  if (codec) {
    uint64_t bits = 0;
    size_t num_bits = 0;
    if (codec->Encode(word, &bits, &num_bits)) {
      // Encoding successful.
      writer_.WriteBits(bits, num_bits);
      return SPV_SUCCESS;
    } else {
      // Encoding failed, write kMarkvNoneOfTheAbove flag.
      if (!codec->Encode(kMarkvNoneOfTheAbove, &bits, &num_bits))
        return Diag(SPV_ERROR_INTERNAL)
               << "Non-id word Huffman table for "
               << spvOpcodeString(SpvOp(inst_.opcode)) << " operand index "
               << operand_index_ << " is missing kMarkvNoneOfTheAbove";
      writer_.WriteBits(bits, num_bits);
    }
  }

  // Fallback encoding.
  const size_t chunk_length =
      model_->GetOperandVariableWidthChunkLength(operand_.type);
  if (chunk_length) {
    writer_.WriteVariableWidthU32(word, chunk_length);
  } else {
    writer_.WriteUnencoded(word);
  }
  return SPV_SUCCESS;
}

spv_result_t MarkvDecoder::DecodeNonIdWord(uint32_t* word) {
  auto* codec = model_->GetNonIdWordHuffmanCodec(inst_.opcode, operand_index_);

  if (codec) {
    uint64_t decoded_value = 0;
    if (!codec->DecodeFromStream(GetReadBitCallback(), &decoded_value))
      return Diag(SPV_ERROR_INVALID_BINARY)
             << "Failed to decode non-id word with Huffman";

    if (decoded_value != kMarkvNoneOfTheAbove) {
      // The word decoded successfully.
      *word = uint32_t(decoded_value);
      assert(*word == decoded_value);
      return SPV_SUCCESS;
    }

    // Received kMarkvNoneOfTheAbove signal, use fallback decoding.
  }

  const size_t chunk_length =
      model_->GetOperandVariableWidthChunkLength(operand_.type);
  if (chunk_length) {
    if (!reader_.ReadVariableWidthU32(word, chunk_length))
      return Diag(SPV_ERROR_INVALID_BINARY)
             << "Failed to decode non-id word with varint";
  } else {
    if (!reader_.ReadUnencoded(word))
      return Diag(SPV_ERROR_INVALID_BINARY)
             << "Failed to read unencoded non-id word";
  }
  return SPV_SUCCESS;
}

spv_result_t MarkvEncoder::EncodeOpcodeAndNumOperands(uint32_t opcode,
                                                      uint32_t num_operands) {
  uint64_t bits = 0;
  size_t num_bits = 0;

  const uint32_t word = opcode | (num_operands << 16);

  // First try to use the Markov chain codec.
  auto* codec =
      model_->GetOpcodeAndNumOperandsMarkovHuffmanCodec(GetPrevOpcode());
  if (codec) {
    if (codec->Encode(word, &bits, &num_bits)) {
      // The word was successfully encoded into bits/num_bits.
      writer_.WriteBits(bits, num_bits);
      return SPV_SUCCESS;
    } else {
      // The word is not in the Huffman table. Write kMarkvNoneOfTheAbove
      // and use fallback encoding.
      if (!codec->Encode(kMarkvNoneOfTheAbove, &bits, &num_bits))
        return Diag(SPV_ERROR_INTERNAL)
               << "opcode_and_num_operands Huffman table for "
               << spvOpcodeString(GetPrevOpcode())
               << "is missing kMarkvNoneOfTheAbove";
      writer_.WriteBits(bits, num_bits);
    }
  }

  // Fallback to base-rate codec.
  codec = model_->GetOpcodeAndNumOperandsMarkovHuffmanCodec(SpvOpNop);
  assert(codec);
  if (codec->Encode(word, &bits, &num_bits)) {
    // The word was successfully encoded into bits/num_bits.
    writer_.WriteBits(bits, num_bits);
    return SPV_SUCCESS;
  } else {
    // The word is not in the Huffman table. Write kMarkvNoneOfTheAbove
    // and return false.
    if (!codec->Encode(kMarkvNoneOfTheAbove, &bits, &num_bits))
      return Diag(SPV_ERROR_INTERNAL)
             << "Global opcode_and_num_operands Huffman table is missing "
             << "kMarkvNoneOfTheAbove";
    writer_.WriteBits(bits, num_bits);
    return SPV_UNSUPPORTED;
  }
}

spv_result_t MarkvDecoder::DecodeOpcodeAndNumberOfOperands(
    uint32_t* opcode, uint32_t* num_operands) {
  // First try to use the Markov chain codec.
  auto* codec =
      model_->GetOpcodeAndNumOperandsMarkovHuffmanCodec(GetPrevOpcode());
  if (codec) {
    uint64_t decoded_value = 0;
    if (!codec->DecodeFromStream(GetReadBitCallback(), &decoded_value))
      return Diag(SPV_ERROR_INTERNAL)
             << "Failed to decode opcode_and_num_operands, previous opcode is "
             << spvOpcodeString(GetPrevOpcode());

    if (decoded_value != kMarkvNoneOfTheAbove) {
      // The word was successfully decoded.
      *opcode = uint32_t(decoded_value & 0xFFFF);
      *num_operands = uint32_t(decoded_value >> 16);
      return SPV_SUCCESS;
    }

    // Received kMarkvNoneOfTheAbove signal, use fallback decoding.
  }

  // Fallback to base-rate codec.
  codec = model_->GetOpcodeAndNumOperandsMarkovHuffmanCodec(SpvOpNop);
  assert(codec);
  uint64_t decoded_value = 0;
  if (!codec->DecodeFromStream(GetReadBitCallback(), &decoded_value))
    return Diag(SPV_ERROR_INTERNAL)
           << "Failed to decode opcode_and_num_operands with global codec";

  if (decoded_value == kMarkvNoneOfTheAbove) {
    // Received kMarkvNoneOfTheAbove signal, fallback further.
    return SPV_UNSUPPORTED;
  }

  *opcode = uint32_t(decoded_value & 0xFFFF);
  *num_operands = uint32_t(decoded_value >> 16);
  return SPV_SUCCESS;
}

spv_result_t MarkvEncoder::EncodeMtfRankHuffman(uint32_t rank) {
  uint64_t bits = 0;
  size_t num_bits = 0;
  if (rank < kMtfSmallestRankEncodedByValue) {
    // Encode using Huffman coding.
    if (!mtf_huffman_codec_.Encode(rank, &bits, &num_bits))
      return Diag(SPV_ERROR_INTERNAL)
             << "Failed to encode MTF rank with Huffman";

    writer_.WriteBits(bits, num_bits);
  } else {
    // Encode by value.
    if (!mtf_huffman_codec_.Encode(kMtfRankEncodedByValueSignal,
                                   &bits, &num_bits))
      return Diag(SPV_ERROR_INTERNAL)
             << "Failed to encode kMtfRankEncodedByValueSignal";

    writer_.WriteBits(bits, num_bits);
    writer_.WriteVariableWidthU32(rank - kMtfSmallestRankEncodedByValue,
                                  model_->mtf_rank_chunk_length());
  }
  return SPV_SUCCESS;
}

spv_result_t MarkvDecoder::DecodeMtfRankHuffman(uint32_t* rank) {
  uint32_t decoded_value = 0;
  if (!mtf_huffman_codec_.DecodeFromStream(GetReadBitCallback(), &decoded_value))
    return Diag(SPV_ERROR_INTERNAL) << "Failed to decode MTF rank with Huffman";

  if (decoded_value == kMtfRankEncodedByValueSignal) {
    // Decode by value.
    if (!reader_.ReadVariableWidthU32(rank, model_->mtf_rank_chunk_length()))
      return Diag(SPV_ERROR_INTERNAL)
             << "Failed to decode MTF rank with varint";
    *rank += kMtfSmallestRankEncodedByValue;
  } else {
    // Decode using Huffman coding.
    assert(decoded_value < kMtfSmallestRankEncodedByValue);
    *rank = decoded_value;
  }
  return SPV_SUCCESS;
}

spv_result_t MarkvEncoder::EncodeIdWithDescriptor(uint32_t id) {
  const uint32_t descriptor = id_descriptors_.GetDescriptor(id);
  // Write descriptor even if it is zero.
  writer_.WriteBits(descriptor, IdDescriptorCollection::GetBitWidth());

  if (!descriptor) {
    // The id is forward declared.
    return SPV_UNSUPPORTED;
  }

  assert(GetMtf(descriptor).GetSize() > 0);
  if (GetMtf(descriptor).GetSize() == 1) {
    // If the sequence has only one element no need to write rank, the decoder
    // would make the same decision.
    return SPV_SUCCESS;
  }

  uint32_t rank = 0;
  if (!GetMtf(descriptor).RankFromValue(id, &rank))
    return Diag(SPV_ERROR_INTERNAL) << "Id is not in the MTF sequence";

  return EncodeMtfRankHuffman(rank);
}

spv_result_t MarkvDecoder::DecodeIdWithDescriptor(uint32_t* id) {
  uint64_t value = 0;

  if (!reader_.ReadBits(&value, IdDescriptorCollection::GetBitWidth()))
    return Diag(SPV_ERROR_INTERNAL) << "Failed to read descriptor";

  const uint32_t descriptor = uint32_t(value);
  if (!descriptor) {
    // This is forward declaration of an id, will be handled elsewhere.
    return SPV_UNSUPPORTED;
  }

  assert(GetMtf(descriptor).GetSize() > 0);

  *id = 0;
  uint32_t rank = 0;

  if (GetMtf(descriptor).GetSize() == 1) {
    rank = 1;
  } else {
    const spv_result_t result = DecodeMtfRankHuffman(&rank);
    if (result != SPV_SUCCESS) return result;
  }

  assert(rank);
  if (!GetMtf(descriptor).ValueFromRank(rank, id))
    return Diag(SPV_ERROR_INTERNAL) << "MTF rank is out of bounds";

  return SPV_SUCCESS;
}

spv_result_t MarkvEncoder::EncodeRefId(uint32_t id) {
  {
    // Try to encode using id descriptor mtfs.
    const spv_result_t result = EncodeIdWithDescriptor(id);
    if (result != SPV_UNSUPPORTED) return result;
    // If can't be done continue with other methods.
  }

  assert(spvOperandCanBeForwardDeclaredFunction(
      SpvOp(inst_.opcode))(operand_index_));

  uint32_t rank = 0;

  if (!GetMtf(kMtfForwardDeclared).RankFromValue(id, &rank)) {
    // This is the first occurrence of a forward declared id.
    GetMtf(kMtfForwardDeclared).Insert(id);
    rank = 0;
  }

  writer_.WriteVariableWidthU32(rank, model_->mtf_rank_chunk_length());
  return SPV_SUCCESS;
}

spv_result_t MarkvDecoder::DecodeRefId(uint32_t* id) {
  {
    const spv_result_t result = DecodeIdWithDescriptor(id);
    if (result != SPV_UNSUPPORTED) return result;
  }

  assert(spvOperandCanBeForwardDeclaredFunction(
      SpvOp(inst_.opcode))(operand_index_));
  *id = 0;

  uint32_t rank = 0;

  if (!reader_.ReadVariableWidthU32(&rank, model_->mtf_rank_chunk_length()))
    return Diag(SPV_ERROR_INTERNAL)
        << "Failed to decode MTF rank with varint";

  if (rank == 0) {
    // This is the first occurrence of a forward declared id.
    *id = GetIdBound();
    SetIdBound(*id + 1);
    GetMtf(kMtfForwardDeclared).Insert(*id);
  } else {
    if (!GetMtf(kMtfForwardDeclared).ValueFromRank(rank, id))
      return Diag(SPV_ERROR_INTERNAL) << "MTF rank out of bounds";
  }

  assert(*id);
  return SPV_SUCCESS;
}

spv_result_t MarkvEncoder::EncodeResultId() {
  uint32_t rank = 0;

  const uint32_t num_still_forward_declared =
      GetMtf(kMtfForwardDeclared).GetSize();

  if (num_still_forward_declared) {
    // We write the rank only if kMtfForwardDeclared is not empty. If it is
    // empty the decoder knows that there are no forward declared ids to expect.
    if (GetMtf(kMtfForwardDeclared).RankFromValue(inst_.result_id, &rank)) {
      // This is a definition of a forward declared id. We can remove the id
      // from kMtfForwardDeclared.
      if (!GetMtf(kMtfForwardDeclared).Remove(inst_.result_id))
        return Diag(SPV_ERROR_INTERNAL)
               << "Failed to remove id from kMtfForwardDeclared";
      writer_.WriteBits(1, 1);
      writer_.WriteVariableWidthU32(rank, model_->mtf_rank_chunk_length());
    } else {
      rank = 0;
      writer_.WriteBits(0, 1);
    }
  }

  return SPV_SUCCESS;
}

spv_result_t MarkvDecoder::DecodeResultId() {
  uint32_t rank = 0;

  const uint32_t num_still_forward_declared =
      GetMtf(kMtfForwardDeclared).GetSize();

  if (num_still_forward_declared) {
    // Some ids were forward declared. Check if this id is one of them.
    uint64_t id_was_forward_declared;
    if (!reader_.ReadBits(&id_was_forward_declared, 1))
      return Diag(SPV_ERROR_INVALID_BINARY)
             << "Failed to read id_was_forward_declared flag";

    if (id_was_forward_declared) {
      if (!reader_.ReadVariableWidthU32(&rank, model_->mtf_rank_chunk_length()))
        return Diag(SPV_ERROR_INVALID_BINARY)
               << "Failed to read MTF rank of forward declared id";

      if (rank) {
        // The id was forward declared, recover it from kMtfForwardDeclared.
        if (!GetMtf(kMtfForwardDeclared).ValueFromRank(rank, &inst_.result_id))
          return Diag(SPV_ERROR_INTERNAL)
                 << "Forward declared MTF rank is out of bounds";

        // We can now remove the id from kMtfForwardDeclared.
        if (!GetMtf(kMtfForwardDeclared).Remove(inst_.result_id))
          return Diag(SPV_ERROR_INTERNAL)
                 << "Failed to remove id from kMtfForwardDeclared";
      }
    }
  }

  if (inst_.result_id == 0) {
    // The id was not forward declared, issue a new id.
    inst_.result_id = GetIdBound();
    SetIdBound(inst_.result_id + 1);
  }

  return SPV_SUCCESS;
}

spv_result_t MarkvEncoder::EncodeLiteralNumber(
    const spv_parsed_operand_t& operand) {
  if (operand.number_bit_width <= 32) {
    const uint32_t word = inst_.words[operand.offset];
    return EncodeNonIdWord(word);
  } else {
    assert(operand.number_bit_width <= 64);
    const uint64_t word = uint64_t(inst_.words[operand.offset]) |
                          (uint64_t(inst_.words[operand.offset + 1]) << 32);
    if (operand.number_kind == SPV_NUMBER_UNSIGNED_INT) {
      writer_.WriteVariableWidthU64(word, model_->u64_chunk_length());
    } else if (operand.number_kind == SPV_NUMBER_SIGNED_INT) {
      int64_t val = 0;
      std::memcpy(&val, &word, 8);
      writer_.WriteVariableWidthS64(val, model_->s64_chunk_length(),
                                    model_->s64_block_exponent());
    } else if (operand.number_kind == SPV_NUMBER_FLOATING) {
      writer_.WriteUnencoded(word);
    } else {
      return Diag(SPV_ERROR_INTERNAL) << "Unsupported bit length";
    }
  }
  return SPV_SUCCESS;
}

spv_result_t MarkvDecoder::DecodeLiteralNumber(
    const spv_parsed_operand_t& operand) {
  if (operand.number_bit_width <= 32) {
    uint32_t word = 0;
    const spv_result_t result = DecodeNonIdWord(&word);
    if (result != SPV_SUCCESS) return result;
    inst_words_.push_back(word);
  } else {
    assert(operand.number_bit_width <= 64);
    uint64_t word = 0;
    if (operand.number_kind == SPV_NUMBER_UNSIGNED_INT) {
      if (!reader_.ReadVariableWidthU64(&word, model_->u64_chunk_length()))
        return Diag(SPV_ERROR_INVALID_BINARY) << "Failed to read literal U64";
    } else if (operand.number_kind == SPV_NUMBER_SIGNED_INT) {
      int64_t val = 0;
      if (!reader_.ReadVariableWidthS64(&val, model_->s64_chunk_length(),
                                        model_->s64_block_exponent()))
        return Diag(SPV_ERROR_INVALID_BINARY) << "Failed to read literal S64";
      std::memcpy(&word, &val, 8);
    } else if (operand.number_kind == SPV_NUMBER_FLOATING) {
      if (!reader_.ReadUnencoded(&word))
        return Diag(SPV_ERROR_INVALID_BINARY) << "Failed to read literal F64";
    } else {
      return Diag(SPV_ERROR_INTERNAL) << "Unsupported bit length";
    }
    inst_words_.push_back(static_cast<uint32_t>(word));
    inst_words_.push_back(static_cast<uint32_t>(word >> 32));
  }
  return SPV_SUCCESS;
}

void MarkvEncoder::AddByteBreak(size_t byte_break_if_less_than) {
  const size_t num_bits_to_next_byte =
      GetNumBitsToNextByte(writer_.GetNumBits());
  if (num_bits_to_next_byte == 0 ||
      num_bits_to_next_byte > byte_break_if_less_than)
    return;

  if (logger_) {
    logger_->AppendWhitespaces(kCommentNumWhitespaces);
    logger_->AppendText("<byte break>");
  }

  writer_.WriteBits(0, num_bits_to_next_byte);
}

bool MarkvDecoder::ReadToByteBreak(size_t byte_break_if_less_than) {
  const size_t num_bits_to_next_byte =
      GetNumBitsToNextByte(reader_.GetNumReadBits());
  if (num_bits_to_next_byte == 0 ||
      num_bits_to_next_byte > byte_break_if_less_than)
    return true;

  uint64_t bits = 0;
  if (!reader_.ReadBits(&bits, num_bits_to_next_byte)) return false;

  assert(bits == 0);
  if (bits != 0) return false;

  return true;
}

spv_result_t MarkvEncoder::EncodeInstruction(
    const spv_parsed_instruction_t& inst) {
  SpvOp opcode = SpvOp(inst.opcode);
  inst_ = inst;

  const spv_result_t validation_result = UpdateValidationState(inst);
  if (validation_result != SPV_SUCCESS) return validation_result;

  LogDisassemblyInstruction();

  const spv_result_t opcode_encodig_result =
      EncodeOpcodeAndNumOperands(opcode, inst.num_operands);
  if (opcode_encodig_result < 0) return opcode_encodig_result;

  if (opcode_encodig_result != SPV_SUCCESS) {
    // Fallback encoding for opcode and num_operands.
    writer_.WriteVariableWidthU32(opcode, model_->opcode_chunk_length());

    if (!OpcodeHasFixedNumberOfOperands(opcode)) {
      // If the opcode has a variable number of operands, encode the number of
      // operands with the instruction.

      if (logger_) logger_->AppendWhitespaces(kCommentNumWhitespaces);

      writer_.WriteVariableWidthU16(inst.num_operands,
                                    model_->num_operands_chunk_length());
    }
  }

  // Write operands.
  const uint32_t num_operands = inst_.num_operands;
  for (operand_index_ = 0; operand_index_ < num_operands; ++operand_index_) {
    operand_ = inst_.operands[operand_index_];

    if (logger_) {
      logger_->AppendWhitespaces(kCommentNumWhitespaces);
      logger_->AppendText("<");
      logger_->AppendText(spvOperandTypeStr(operand_.type));
      logger_->AppendText(">");
    }

    switch (operand_.type) {
      case SPV_OPERAND_TYPE_RESULT_ID: {
        const spv_result_t result = EncodeResultId();
        if (result != SPV_SUCCESS) return result;
        break;
      }
      case SPV_OPERAND_TYPE_TYPE_ID: {
        const spv_result_t result = EncodeIdWithDescriptor(inst_.type_id);
        if (result != SPV_SUCCESS) return result;
        break;
      }

      case SPV_OPERAND_TYPE_ID:
      case SPV_OPERAND_TYPE_OPTIONAL_ID:
      case SPV_OPERAND_TYPE_SCOPE_ID:
      case SPV_OPERAND_TYPE_MEMORY_SEMANTICS_ID: {
        const uint32_t id = inst_.words[operand_.offset];
        const spv_result_t result = EncodeRefId(id);
        if (result != SPV_SUCCESS) return result;
        break;
      }

      case SPV_OPERAND_TYPE_LITERAL_INTEGER: {
        const spv_result_t result =
            EncodeNonIdWord(inst_.words[operand_.offset]);
        if (result != SPV_SUCCESS) return result;
        break;
      }

      case SPV_OPERAND_TYPE_TYPED_LITERAL_NUMBER: {
        const spv_result_t result = EncodeLiteralNumber(operand_);
        if (result != SPV_SUCCESS) return result;
        break;
      }

      case SPV_OPERAND_TYPE_LITERAL_STRING: {
        const char* src =
            reinterpret_cast<const char*>(&inst_.words[operand_.offset]);

        auto* codec = model_->GetLiteralStringHuffmanCodec(opcode);
        if (codec) {
          uint64_t bits = 0;
          size_t num_bits = 0;
          const std::string str = src;
          if (codec->Encode(str, &bits, &num_bits)) {
            writer_.WriteBits(bits, num_bits);
            break;
          } else {
            bool result =
                codec->Encode("kMarkvNoneOfTheAbove", &bits, &num_bits);
            (void)result;
            assert(result);
            writer_.WriteBits(bits, num_bits);
          }
        }

        const size_t length = spv_strnlen_s(src, operand_.num_words * 4);
        if (length == operand_.num_words * 4)
          return Diag(SPV_ERROR_INVALID_BINARY)
                 << "Failed to find terminal character of literal string";
        for (size_t i = 0; i < length + 1; ++i) writer_.WriteUnencoded(src[i]);
        break;
      }

      default: {
        for (int i = 0; i < operand_.num_words; ++i) {
          const uint32_t word = inst_.words[operand_.offset + i];
          const spv_result_t result = EncodeNonIdWord(word);
          if (result != SPV_SUCCESS) return result;
        }
        break;
      }
    }
  }

  AddByteBreak(kByteBreakAfterInstIfLessThanUntilNextByte);

  if (logger_) {
    logger_->NewLine();
    logger_->NewLine();
    if (!logger_->DebugInstruction(inst_)) return SPV_REQUESTED_TERMINATION;
  }

  ProcessCurInstruction();

  return SPV_SUCCESS;
}

spv_result_t MarkvDecoder::DecodeModule(std::vector<uint32_t>* spirv_binary) {
  const bool header_read_success =
      reader_.ReadUnencoded(&header_.magic_number) &&
      reader_.ReadUnencoded(&header_.markv_version) &&
      reader_.ReadUnencoded(&header_.markv_model) &&
      reader_.ReadUnencoded(&header_.markv_length_in_bits) &&
      reader_.ReadUnencoded(&header_.spirv_version) &&
      reader_.ReadUnencoded(&header_.spirv_generator);

  if (!header_read_success)
    return Diag(SPV_ERROR_INVALID_BINARY) << "Unable to read MARK-V header";

  if (header_.markv_length_in_bits == 0)
    return Diag(SPV_ERROR_INVALID_BINARY)
           << "Header markv_length_in_bits field is zero";

  if (header_.magic_number != kMarkvMagicNumber)
    return Diag(SPV_ERROR_INVALID_BINARY)
           << "MARK-V binary has incorrect magic number";

  // TODO(atgoo@github.com): Print version strings.
  if (header_.markv_version != GetMarkvVersion())
    return Diag(SPV_ERROR_INVALID_BINARY)
           << "MARK-V binary and the codec have different versions";

  const uint32_t model_type = header_.markv_model >> 16;
  const uint32_t model_version = header_.markv_model & 0xFFFF;
  if (model_type != model_->model_type())
    return Diag(SPV_ERROR_INVALID_BINARY)
           << "MARK-V binary and the codec use different MARK-V models";

  if (model_version != model_->model_version())
    return Diag(SPV_ERROR_INVALID_BINARY)
           << "MARK-V binary and the codec use different versions if the same "
           << "MARK-V model";

  spirv_.reserve(header_.markv_length_in_bits / 2);  // Heuristic.
  spirv_.resize(5, 0);
  spirv_[0] = kSpirvMagicNumber;
  spirv_[1] = header_.spirv_version;
  spirv_[2] = header_.spirv_generator;

  if (logger_) {
    reader_.SetCallback(
        [this](const std::string& str) { logger_->AppendBitSequence(str); });
  }

  while (reader_.GetNumReadBits() < header_.markv_length_in_bits) {
    inst_ = {};
    const spv_result_t decode_result = DecodeInstruction();
    if (decode_result != SPV_SUCCESS) return decode_result;

    const spv_result_t validation_result = UpdateValidationState(inst_);
    if (validation_result != SPV_SUCCESS) return validation_result;
  }

  if (reader_.GetNumReadBits() != header_.markv_length_in_bits ||
      !reader_.OnlyZeroesLeft()) {
    return Diag(SPV_ERROR_INVALID_BINARY)
           << "MARK-V binary has wrong stated bit length "
           << reader_.GetNumReadBits() << " " << header_.markv_length_in_bits;
  }

  // Decoding of the module is finished, validation state should have correct
  // id bound.
  spirv_[3] = GetIdBound();

  *spirv_binary = std::move(spirv_);
  return SPV_SUCCESS;
}

// TODO(atgoo@github.com): The implementation borrows heavily from
// Parser::parseOperand.
// Consider coupling them together in some way once MARK-V codec is more mature.
// For now it's better to keep the code independent for experimentation
// purposes.
spv_result_t MarkvDecoder::DecodeOperand(
    size_t operand_offset, const spv_operand_type_t type,
    spv_operand_pattern_t* expected_operands) {
  const SpvOp opcode = static_cast<SpvOp>(inst_.opcode);

  memset(&operand_, 0, sizeof(operand_));

  assert((operand_offset >> 16) == 0);
  operand_.offset = static_cast<uint16_t>(operand_offset);
  operand_.type = type;

  // Set default values, may be updated later.
  operand_.number_kind = SPV_NUMBER_NONE;
  operand_.number_bit_width = 0;

  const size_t first_word_index = inst_words_.size();

  switch (type) {
    case SPV_OPERAND_TYPE_RESULT_ID: {
      const spv_result_t result = DecodeResultId();
      if (result != SPV_SUCCESS) return result;

      inst_words_.push_back(inst_.result_id);
      SetIdBound(std::max(GetIdBound(), inst_.result_id + 1));
      break;
    }

    case SPV_OPERAND_TYPE_TYPE_ID: {
      const spv_result_t result = DecodeIdWithDescriptor(&inst_.type_id);
      if (result != SPV_SUCCESS) return result;

      inst_words_.push_back(inst_.type_id);
      SetIdBound(std::max(GetIdBound(), inst_.type_id + 1));
      break;
    }

    case SPV_OPERAND_TYPE_ID:
    case SPV_OPERAND_TYPE_OPTIONAL_ID:
    case SPV_OPERAND_TYPE_SCOPE_ID:
    case SPV_OPERAND_TYPE_MEMORY_SEMANTICS_ID: {
      uint32_t id = 0;
      const spv_result_t result = DecodeRefId(&id);
      if (result != SPV_SUCCESS) return result;

      if (id == 0) return Diag(SPV_ERROR_INVALID_BINARY) << "Decoded id is 0";

      if (type == SPV_OPERAND_TYPE_ID || type == SPV_OPERAND_TYPE_OPTIONAL_ID) {
        operand_.type = SPV_OPERAND_TYPE_ID;

        if (opcode == SpvOpExtInst && operand_.offset == 3) {
          // The current word is the extended instruction set id.
          // Set the extended instruction set type for the current
          // instruction.
          auto ext_inst_type_iter = import_id_to_ext_inst_type_.find(id);
          if (ext_inst_type_iter == import_id_to_ext_inst_type_.end()) {
            return Diag(SPV_ERROR_INVALID_ID)
                   << "OpExtInst set id " << id
                   << " does not reference an OpExtInstImport result Id";
          }
          inst_.ext_inst_type = ext_inst_type_iter->second;
        }
      }

      inst_words_.push_back(id);
      SetIdBound(std::max(GetIdBound(), id + 1));
      break;
    }

    case SPV_OPERAND_TYPE_EXTENSION_INSTRUCTION_NUMBER: {
      uint32_t word = 0;
      const spv_result_t result = DecodeNonIdWord(&word);
      if (result != SPV_SUCCESS) return result;

      inst_words_.push_back(word);

      assert(SpvOpExtInst == opcode);
      assert(inst_.ext_inst_type != SPV_EXT_INST_TYPE_NONE);
      spv_ext_inst_desc ext_inst;
      if (grammar_.lookupExtInst(inst_.ext_inst_type, word, &ext_inst))
        return Diag(SPV_ERROR_INVALID_BINARY)
               << "Invalid extended instruction number: " << word;
      spvPushOperandTypes(ext_inst->operandTypes, expected_operands);
      break;
    }

    case SPV_OPERAND_TYPE_LITERAL_INTEGER:
    case SPV_OPERAND_TYPE_OPTIONAL_LITERAL_INTEGER: {
      // These are regular single-word literal integer operands.
      // Post-parsing validation should check the range of the parsed value.
      operand_.type = SPV_OPERAND_TYPE_LITERAL_INTEGER;
      // It turns out they are always unsigned integers!
      operand_.number_kind = SPV_NUMBER_UNSIGNED_INT;
      operand_.number_bit_width = 32;

      uint32_t word = 0;
      const spv_result_t result = DecodeNonIdWord(&word);
      if (result != SPV_SUCCESS) return result;

      inst_words_.push_back(word);
      break;
    }

    case SPV_OPERAND_TYPE_TYPED_LITERAL_NUMBER:
    case SPV_OPERAND_TYPE_OPTIONAL_TYPED_LITERAL_INTEGER: {
      operand_.type = SPV_OPERAND_TYPE_TYPED_LITERAL_NUMBER;
      if (opcode == SpvOpSwitch) {
        // The literal operands have the same type as the value
        // referenced by the selector Id.
        const uint32_t selector_id = inst_words_.at(1);
        const auto type_id_iter = id_to_type_id_.find(selector_id);
        if (type_id_iter == id_to_type_id_.end() || type_id_iter->second == 0) {
          return Diag(SPV_ERROR_INVALID_BINARY)
                 << "Invalid OpSwitch: selector id " << selector_id
                 << " has no type";
        }
        uint32_t type_id = type_id_iter->second;

        if (selector_id == type_id) {
          // Recall that by convention, a result ID that is a type definition
          // maps to itself.
          return Diag(SPV_ERROR_INVALID_BINARY)
                 << "Invalid OpSwitch: selector id " << selector_id
                 << " is a type, not a value";
        }
        if (auto error = SetNumericTypeInfoForType(&operand_, type_id))
          return error;
        if (operand_.number_kind != SPV_NUMBER_UNSIGNED_INT &&
            operand_.number_kind != SPV_NUMBER_SIGNED_INT) {
          return Diag(SPV_ERROR_INVALID_BINARY)
                 << "Invalid OpSwitch: selector id " << selector_id
                 << " is not a scalar integer";
        }
      } else {
        assert(opcode == SpvOpConstant || opcode == SpvOpSpecConstant);
        // The literal number type is determined by the type Id for the
        // constant.
        assert(inst_.type_id);
        if (auto error = SetNumericTypeInfoForType(&operand_, inst_.type_id))
          return error;
      }

      if (auto error = DecodeLiteralNumber(operand_)) return error;

      break;
    }

    case SPV_OPERAND_TYPE_LITERAL_STRING:
    case SPV_OPERAND_TYPE_OPTIONAL_LITERAL_STRING: {
      operand_.type = SPV_OPERAND_TYPE_LITERAL_STRING;
      std::vector<char> str;
      auto* codec = model_->GetLiteralStringHuffmanCodec(inst_.opcode);

      if (codec) {
        std::string decoded_string;
        const bool huffman_result =
            codec->DecodeFromStream(GetReadBitCallback(), &decoded_string);
        assert(huffman_result);
        if (!huffman_result)
          return Diag(SPV_ERROR_INVALID_BINARY)
                 << "Failed to read literal string";

        if (decoded_string != "kMarkvNoneOfTheAbove") {
          std::copy(decoded_string.begin(), decoded_string.end(),
                    std::back_inserter(str));
          str.push_back('\0');
        }
      }

      // The loop is expected to terminate once we encounter '\0' or exhaust
      // the bit stream.
      if (str.empty()) {
        while (true) {
          char ch = 0;
          if (!reader_.ReadUnencoded(&ch))
            return Diag(SPV_ERROR_INVALID_BINARY)
                   << "Failed to read literal string";

          str.push_back(ch);

          if (ch == '\0') break;
        }
      }

      while (str.size() % 4 != 0) str.push_back('\0');

      inst_words_.resize(inst_words_.size() + str.size() / 4);
      std::memcpy(&inst_words_[first_word_index], str.data(), str.size());

      if (SpvOpExtInstImport == opcode) {
        // Record the extended instruction type for the ID for this import.
        // There is only one string literal argument to OpExtInstImport,
        // so it's sufficient to guard this just on the opcode.
        const spv_ext_inst_type_t ext_inst_type =
            spvExtInstImportTypeGet(str.data());
        if (SPV_EXT_INST_TYPE_NONE == ext_inst_type) {
          return Diag(SPV_ERROR_INVALID_BINARY)
                 << "Invalid extended instruction import '" << str.data()
                 << "'";
        }
        // We must have parsed a valid result ID.  It's a condition
        // of the grammar, and we only accept non-zero result Ids.
        assert(inst_.result_id);
        const bool inserted =
            import_id_to_ext_inst_type_.emplace(inst_.result_id, ext_inst_type)
                .second;
        (void)inserted;
        assert(inserted);
      }
      break;
    }

    case SPV_OPERAND_TYPE_CAPABILITY:
    case SPV_OPERAND_TYPE_SOURCE_LANGUAGE:
    case SPV_OPERAND_TYPE_EXECUTION_MODEL:
    case SPV_OPERAND_TYPE_ADDRESSING_MODEL:
    case SPV_OPERAND_TYPE_MEMORY_MODEL:
    case SPV_OPERAND_TYPE_EXECUTION_MODE:
    case SPV_OPERAND_TYPE_STORAGE_CLASS:
    case SPV_OPERAND_TYPE_DIMENSIONALITY:
    case SPV_OPERAND_TYPE_SAMPLER_ADDRESSING_MODE:
    case SPV_OPERAND_TYPE_SAMPLER_FILTER_MODE:
    case SPV_OPERAND_TYPE_SAMPLER_IMAGE_FORMAT:
    case SPV_OPERAND_TYPE_FP_ROUNDING_MODE:
    case SPV_OPERAND_TYPE_LINKAGE_TYPE:
    case SPV_OPERAND_TYPE_ACCESS_QUALIFIER:
    case SPV_OPERAND_TYPE_OPTIONAL_ACCESS_QUALIFIER:
    case SPV_OPERAND_TYPE_FUNCTION_PARAMETER_ATTRIBUTE:
    case SPV_OPERAND_TYPE_DECORATION:
    case SPV_OPERAND_TYPE_BUILT_IN:
    case SPV_OPERAND_TYPE_GROUP_OPERATION:
    case SPV_OPERAND_TYPE_KERNEL_ENQ_FLAGS:
    case SPV_OPERAND_TYPE_KERNEL_PROFILING_INFO: {
      // A single word that is a plain enum value.
      uint32_t word = 0;
      const spv_result_t result = DecodeNonIdWord(&word);
      if (result != SPV_SUCCESS) return result;

      inst_words_.push_back(word);

      // Map an optional operand type to its corresponding concrete type.
      if (type == SPV_OPERAND_TYPE_OPTIONAL_ACCESS_QUALIFIER)
        operand_.type = SPV_OPERAND_TYPE_ACCESS_QUALIFIER;

      spv_operand_desc entry;
      if (grammar_.lookupOperand(type, word, &entry)) {
        return Diag(SPV_ERROR_INVALID_BINARY)
               << "Invalid " << spvOperandTypeStr(operand_.type)
               << " operand: " << word;
      }

      // Prepare to accept operands to this operand, if needed.
      spvPushOperandTypes(entry->operandTypes, expected_operands);
      break;
    }

    case SPV_OPERAND_TYPE_FP_FAST_MATH_MODE:
    case SPV_OPERAND_TYPE_FUNCTION_CONTROL:
    case SPV_OPERAND_TYPE_LOOP_CONTROL:
    case SPV_OPERAND_TYPE_IMAGE:
    case SPV_OPERAND_TYPE_OPTIONAL_IMAGE:
    case SPV_OPERAND_TYPE_OPTIONAL_MEMORY_ACCESS:
    case SPV_OPERAND_TYPE_SELECTION_CONTROL: {
      // This operand is a mask.
      uint32_t word = 0;
      const spv_result_t result = DecodeNonIdWord(&word);
      if (result != SPV_SUCCESS) return result;

      inst_words_.push_back(word);

      // Map an optional operand type to its corresponding concrete type.
      if (type == SPV_OPERAND_TYPE_OPTIONAL_IMAGE)
        operand_.type = SPV_OPERAND_TYPE_IMAGE;
      else if (type == SPV_OPERAND_TYPE_OPTIONAL_MEMORY_ACCESS)
        operand_.type = SPV_OPERAND_TYPE_MEMORY_ACCESS;

      // Check validity of set mask bits. Also prepare for operands for those
      // masks if they have any.  To get operand order correct, scan from
      // MSB to LSB since we can only prepend operands to a pattern.
      // The only case in the grammar where you have more than one mask bit
      // having an operand is for image operands.  See SPIR-V 3.14 Image
      // Operands.
      uint32_t remaining_word = word;
      for (uint32_t mask = (1u << 31); remaining_word; mask >>= 1) {
        if (remaining_word & mask) {
          spv_operand_desc entry;
          if (grammar_.lookupOperand(type, mask, &entry)) {
            return Diag(SPV_ERROR_INVALID_BINARY)
                   << "Invalid " << spvOperandTypeStr(operand_.type)
                   << " operand: " << word << " has invalid mask component "
                   << mask;
          }
          remaining_word ^= mask;
          spvPushOperandTypes(entry->operandTypes, expected_operands);
        }
      }
      if (word == 0) {
        // An all-zeroes mask *might* also be valid.
        spv_operand_desc entry;
        if (SPV_SUCCESS == grammar_.lookupOperand(type, 0, &entry)) {
          // Prepare for its operands, if any.
          spvPushOperandTypes(entry->operandTypes, expected_operands);
        }
      }
      break;
    }
    default:
      return Diag(SPV_ERROR_INVALID_BINARY)
             << "Internal error: Unhandled operand type: " << type;
  }

  operand_.num_words = uint16_t(inst_words_.size() - first_word_index);

  assert(int(SPV_OPERAND_TYPE_FIRST_CONCRETE_TYPE) <= int(operand_.type));
  assert(int(SPV_OPERAND_TYPE_LAST_CONCRETE_TYPE) >= int(operand_.type));

  parsed_operands_.push_back(operand_);

  return SPV_SUCCESS;
}

spv_result_t MarkvDecoder::DecodeInstruction() {
  parsed_operands_.clear();
  inst_words_.clear();

  // Opcode/num_words placeholder, the word will be filled in later.
  inst_words_.push_back(0);

  bool num_operands_still_unknown = true;
  {
    uint32_t opcode = 0;
    uint32_t num_operands = 0;

    const spv_result_t opcode_decoding_result =
        DecodeOpcodeAndNumberOfOperands(&opcode, &num_operands);
    if (opcode_decoding_result < 0) return opcode_decoding_result;

    if (opcode_decoding_result == SPV_SUCCESS) {
      inst_.num_operands = static_cast<uint16_t>(num_operands);
      num_operands_still_unknown = false;
    } else {
      if (!reader_.ReadVariableWidthU32(&opcode,
                                        model_->opcode_chunk_length())) {
        return Diag(SPV_ERROR_INVALID_BINARY)
               << "Failed to read opcode of instruction";
      }
    }

    inst_.opcode = static_cast<uint16_t>(opcode);
  }

  const SpvOp opcode = static_cast<SpvOp>(inst_.opcode);

  spv_opcode_desc opcode_desc;
  if (grammar_.lookupOpcode(opcode, &opcode_desc) != SPV_SUCCESS) {
    return Diag(SPV_ERROR_INVALID_BINARY) << "Invalid opcode";
  }

  spv_operand_pattern_t expected_operands;
  expected_operands.reserve(opcode_desc->numTypes);
  for (auto i = 0; i < opcode_desc->numTypes; i++) {
    expected_operands.push_back(
        opcode_desc->operandTypes[opcode_desc->numTypes - i - 1]);
  }

  if (num_operands_still_unknown) {
    if (!OpcodeHasFixedNumberOfOperands(opcode)) {
      if (!reader_.ReadVariableWidthU16(&inst_.num_operands,
                                        model_->num_operands_chunk_length()))
        return Diag(SPV_ERROR_INVALID_BINARY)
               << "Failed to read num_operands of instruction";
    } else {
      inst_.num_operands = static_cast<uint16_t>(expected_operands.size());
    }
  }

  for (operand_index_ = 0;
       operand_index_ < static_cast<size_t>(inst_.num_operands);
       ++operand_index_) {
    assert(!expected_operands.empty());
    const spv_operand_type_t type =
        spvTakeFirstMatchableOperand(&expected_operands);

    const size_t operand_offset = inst_words_.size();

    const spv_result_t decode_result =
        DecodeOperand(operand_offset, type, &expected_operands);

    if (decode_result != SPV_SUCCESS) return decode_result;
  }

  assert(inst_.num_operands == parsed_operands_.size());

  // Only valid while inst_words_ and parsed_operands_ remain unchanged (until
  // next DecodeInstruction call).
  inst_.words = inst_words_.data();
  inst_.operands = parsed_operands_.empty() ? nullptr : parsed_operands_.data();
  inst_.num_words = static_cast<uint16_t>(inst_words_.size());
  inst_words_[0] = spvOpcodeMake(inst_.num_words, SpvOp(inst_.opcode));

  std::copy(inst_words_.begin(), inst_words_.end(), std::back_inserter(spirv_));

  assert(inst_.num_words ==
             std::accumulate(
                 parsed_operands_.begin(), parsed_operands_.end(), 1,
                 [](int num_words, const spv_parsed_operand_t& operand) {
                   return num_words += operand.num_words;
                 }) &&
         "num_words in instruction doesn't correspond to the sum of num_words"
         "in the operands");

  RecordNumberType();
  ProcessCurInstruction();

  if (!ReadToByteBreak(kByteBreakAfterInstIfLessThanUntilNextByte))
    return Diag(SPV_ERROR_INVALID_BINARY) << "Failed to read to byte break";

  if (logger_) {
    logger_->NewLine();
    std::stringstream ss;
    ss << spvOpcodeString(opcode) << " ";
    for (size_t index = 1; index < inst_words_.size(); ++index)
      ss << inst_words_[index] << " ";
    logger_->AppendText(ss.str());
    logger_->NewLine();
    logger_->NewLine();
    if (!logger_->DebugInstruction(inst_)) return SPV_REQUESTED_TERMINATION;
  }

  return SPV_SUCCESS;
}

spv_result_t MarkvDecoder::SetNumericTypeInfoForType(
    spv_parsed_operand_t* parsed_operand, uint32_t type_id) {
  assert(type_id != 0);
  auto type_info_iter = type_id_to_number_type_info_.find(type_id);
  if (type_info_iter == type_id_to_number_type_info_.end()) {
    return Diag(SPV_ERROR_INVALID_BINARY)
           << "Type Id " << type_id << " is not a type";
  }

  const NumberType& info = type_info_iter->second;
  if (info.type == SPV_NUMBER_NONE) {
    // This is a valid type, but for something other than a scalar number.
    return Diag(SPV_ERROR_INVALID_BINARY)
           << "Type Id " << type_id << " is not a scalar numeric type";
  }

  parsed_operand->number_kind = info.type;
  parsed_operand->number_bit_width = info.bit_width;
  // Round up the word count.
  parsed_operand->num_words = static_cast<uint16_t>((info.bit_width + 31) / 32);
  return SPV_SUCCESS;
}

void MarkvDecoder::RecordNumberType() {
  const SpvOp opcode = static_cast<SpvOp>(inst_.opcode);
  if (spvOpcodeGeneratesType(opcode)) {
    NumberType info = {SPV_NUMBER_NONE, 0};
    if (SpvOpTypeInt == opcode) {
      info.bit_width = inst_.words[inst_.operands[1].offset];
      info.type = inst_.words[inst_.operands[2].offset]
                      ? SPV_NUMBER_SIGNED_INT
                      : SPV_NUMBER_UNSIGNED_INT;
    } else if (SpvOpTypeFloat == opcode) {
      info.bit_width = inst_.words[inst_.operands[1].offset];
      info.type = SPV_NUMBER_FLOATING;
    }
    // The *result* Id of a type generating instruction is the type Id.
    type_id_to_number_type_info_[inst_.result_id] = info;
  }
}

spv_result_t EncodeHeader(void* user_data, spv_endianness_t endian,
                          uint32_t magic, uint32_t version, uint32_t generator,
                          uint32_t id_bound, uint32_t schema) {
  MarkvEncoder* encoder = reinterpret_cast<MarkvEncoder*>(user_data);
  return encoder->EncodeHeader(endian, magic, version, generator, id_bound,
                               schema);
}

spv_result_t EncodeInstruction(void* user_data,
                               const spv_parsed_instruction_t* inst) {
  MarkvEncoder* encoder = reinterpret_cast<MarkvEncoder*>(user_data);
  return encoder->EncodeInstruction(*inst);
}

}  // namespace

spv_result_t SpirvToMarkv(
    spv_const_context context, const std::vector<uint32_t>& spirv,
    const MarkvCodecOptions& options, const MarkvModel& markv_model,
    MessageConsumer message_consumer, MarkvLogConsumer log_consumer,
    MarkvDebugConsumer debug_consumer, std::vector<uint8_t>* markv) {
  spv_context_t hijack_context = *context;
  SetContextMessageConsumer(&hijack_context, message_consumer);

  spv_const_binary_t spirv_binary = {spirv.data(), spirv.size()};

  spv_endianness_t endian;
  spv_position_t position = {};
  if (spvBinaryEndianness(&spirv_binary, &endian)) {
    return DiagnosticStream(position, hijack_context.consumer,
                            SPV_ERROR_INVALID_BINARY)
           << "Invalid SPIR-V magic number.";
  }

  spv_header_t header;
  if (spvBinaryHeaderGet(&spirv_binary, endian, &header)) {
    return DiagnosticStream(position, hijack_context.consumer,
                            SPV_ERROR_INVALID_BINARY)
           << "Invalid SPIR-V header.";
  }

  MarkvEncoder encoder(&hijack_context, options, &markv_model);

  if (log_consumer || debug_consumer) {
    encoder.CreateLogger(log_consumer, debug_consumer);

    spv_text text = nullptr;
    if (spvBinaryToText(&hijack_context, spirv.data(), spirv.size(),
                        SPV_BINARY_TO_TEXT_OPTION_NO_HEADER, &text,
                        nullptr) != SPV_SUCCESS) {
      return DiagnosticStream(position, hijack_context.consumer,
                              SPV_ERROR_INVALID_BINARY)
             << "Failed to disassemble SPIR-V binary.";
    }
    assert(text);
    encoder.SetDisassembly(std::string(text->str, text->length));
    spvTextDestroy(text);
  }

  if (spvBinaryParse(&hijack_context, &encoder, spirv.data(), spirv.size(),
                     EncodeHeader, EncodeInstruction, nullptr) != SPV_SUCCESS) {
    return DiagnosticStream(position, hijack_context.consumer,
                            SPV_ERROR_INVALID_BINARY)
           << "Unable to encode to MARK-V.";
  }

  *markv = encoder.GetMarkvBinary();
  return SPV_SUCCESS;
}

spv_result_t MarkvToSpirv(
    spv_const_context context, const std::vector<uint8_t>& markv,
    const MarkvCodecOptions& options, const MarkvModel& markv_model,
    MessageConsumer message_consumer, MarkvLogConsumer log_consumer,
    MarkvDebugConsumer debug_consumer, std::vector<uint32_t>* spirv) {
  spv_position_t position = {};
  spv_context_t hijack_context = *context;
  SetContextMessageConsumer(&hijack_context, message_consumer);

  MarkvDecoder decoder(&hijack_context, markv, options, &markv_model);

  if (log_consumer || debug_consumer)
    decoder.CreateLogger(log_consumer, debug_consumer);

  if (decoder.DecodeModule(spirv) != SPV_SUCCESS) {
    return DiagnosticStream(position, hijack_context.consumer,
                            SPV_ERROR_INVALID_BINARY)
           << "Unable to decode MARK-V.";
  }

  assert(!spirv->empty());
  return SPV_SUCCESS;
}

}  // namespave spvtools
