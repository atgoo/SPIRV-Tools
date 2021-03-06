// Copyright (c) 2017 Pierre Moreau
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

#include "remove_duplicates_pass.h"

#include <cstring>

#include <algorithm>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "decoration_manager.h"
#include "ir_context.h"
#include "opcode.h"
#include "reflect.h"

namespace spvtools {
namespace opt {

using ir::Instruction;
using ir::Module;
using ir::Operand;
using opt::analysis::DecorationManager;
using opt::analysis::DefUseManager;

Pass::Status RemoveDuplicatesPass::Process(ir::IRContext* ir_context) {
  bool modified = RemoveDuplicateCapabilities(ir_context);
  modified |= RemoveDuplicatesExtInstImports(ir_context);
  modified |= RemoveDuplicateTypes(ir_context);
  modified |= RemoveDuplicateDecorations(ir_context);

  return modified ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

bool RemoveDuplicatesPass::RemoveDuplicateCapabilities(
    ir::IRContext* ir_context) const {
  bool modified = false;

  if (ir_context->capabilities().empty()) {
    return modified;
  }

  std::unordered_set<uint32_t> capabilities;
  for (auto* i = &*ir_context->capability_begin(); i;) {
    auto res = capabilities.insert(i->GetSingleWordOperand(0u));

    if (res.second) {
      // Never seen before, keep it.
      i = i->NextNode();
    } else {
      // It's a duplicate, remove it.
      i = ir_context->KillInst(i);
      modified = true;
    }
  }

  return modified;
}

bool RemoveDuplicatesPass::RemoveDuplicatesExtInstImports(
    ir::IRContext* ir_context) const {
  bool modified = false;

  if (ir_context->ext_inst_imports().empty()) {
    return modified;
  }

  std::unordered_map<std::string, SpvId> ext_inst_imports;
  for (auto* i = &*ir_context->ext_inst_import_begin(); i;) {
    auto res = ext_inst_imports.emplace(
        reinterpret_cast<const char*>(i->GetInOperand(0u).words.data()),
        i->result_id());
    if (res.second) {
      // Never seen before, keep it.
      i = i->NextNode();
    } else {
      // It's a duplicate, remove it.
      ir_context->ReplaceAllUsesWith(i->result_id(), res.first->second);
      i = ir_context->KillInst(i);
      modified = true;
    }
  }

  return modified;
}

bool RemoveDuplicatesPass::RemoveDuplicateTypes(
    ir::IRContext* ir_context) const {
  bool modified = false;

  if (ir_context->types_values().empty()) {
    return modified;
  }

  std::vector<Instruction*> visited_types;
  std::vector<Instruction*> to_delete;
  for (auto* i = &*ir_context->types_values_begin(); i; i = i->NextNode()) {
    // We only care about types.
    if (!spvOpcodeGeneratesType((i->opcode())) &&
        i->opcode() != SpvOpTypeForwardPointer) {
      continue;
    }

    // Is the current type equal to one of the types we have aready visited?
    SpvId id_to_keep = 0u;
    // TODO(dneto0): Use a trie to avoid quadratic behaviour? Extract the
    // ResultIdTrie from unify_const_pass.cpp for this.
    for (auto j : visited_types) {
      if (AreTypesEqual(*i, *j, ir_context)) {
        id_to_keep = j->result_id();
        break;
      }
    }

    if (id_to_keep == 0u) {
      // This is a never seen before type, keep it around.
      visited_types.emplace_back(i);
    } else {
      // The same type has already been seen before, remove this one.
      ir_context->KillNamesAndDecorates(i->result_id());
      ir_context->ReplaceAllUsesWith(i->result_id(), id_to_keep);
      modified = true;
      to_delete.emplace_back(i);
    }
  }

  for (auto i : to_delete) {
    ir_context->KillInst(i);
  }

  return modified;
}

// TODO(pierremoreau): Duplicate decoration groups should be removed. For
// example, in
//     OpDecorate %1 Constant
//     %1 = OpDecorationGroup
//     OpDecorate %2 Constant
//     %2 = OpDecorationGroup
//     OpGroupDecorate %1 %3
//     OpGroupDecorate %2 %4
// group %2 could be removed.
bool RemoveDuplicatesPass::RemoveDuplicateDecorations(
    ir::IRContext* ir_context) const {
  bool modified = false;

  std::vector<const Instruction*> visited_decorations;

  opt::analysis::DecorationManager decoration_manager(ir_context->module());
  for (auto* i = &*ir_context->annotation_begin(); i;) {
    // Is the current decoration equal to one of the decorations we have aready
    // visited?
    bool already_visited = false;
    // TODO(dneto0): Use a trie to avoid quadratic behaviour? Extract the
    // ResultIdTrie from unify_const_pass.cpp for this.
    for (const Instruction* j : visited_decorations) {
      if (decoration_manager.AreDecorationsTheSame(&*i, j, false)) {
        already_visited = true;
        break;
      }
    }

    if (!already_visited) {
      // This is a never seen before decoration, keep it around.
      visited_decorations.emplace_back(&*i);
      i = i->NextNode();
    } else {
      // The same decoration has already been seen before, remove this one.
      modified = true;
      i = ir_context->KillInst(i);
    }
  }

  return modified;
}

bool RemoveDuplicatesPass::AreTypesEqual(const Instruction& inst1,
                                         const Instruction& inst2,
                                         ir::IRContext* context) {
  if (inst1.opcode() != inst2.opcode()) return false;
  if (!ir::IsTypeInst(inst1.opcode())) return false;

  const analysis::Type* type1 =
      context->get_type_mgr()->GetType(inst1.result_id());
  const analysis::Type* type2 =
      context->get_type_mgr()->GetType(inst2.result_id());
  if (type1 && type2 && *type1 == *type2) return true;

  return false;
}

}  // namespace opt
}  // namespace spvtools
