
std::map<uint64_t, uint32_t> GetOpcodeAndNumOperandsHist() {
  return std::map<uint64_t, uint32_t>({
    { CombineOpcodeAndNumOperands(SpvOpName, 2), 5805 },
    { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 5), 5848 },
    { CombineOpcodeAndNumOperands(SpvOpSelectionMerge, 2), 9148 },
    { CombineOpcodeAndNumOperands(SpvOpFunctionEnd, 0), 49767 },
    { CombineOpcodeAndNumOperands(SpvOpReturn, 0), 9562 },
    { CombineOpcodeAndNumOperands(SpvOpAccessChain, 6), 6566 },
    { CombineOpcodeAndNumOperands(SpvOpLoad, 3), 917670 },
    { CombineOpcodeAndNumOperands(SpvOpConstantComposite, 4), 20665 },
    { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 5), 63908 },
    { CombineOpcodeAndNumOperands(SpvOpTypeSampledImage, 2), 11045 },
    { CombineOpcodeAndNumOperands(SpvOpFunctionParameter, 2), 84426 },
    { CombineOpcodeAndNumOperands(SpvOpTypeVoid, 1), 6401 },
    { CombineOpcodeAndNumOperands(SpvOpVariable, 3), 371543 },
    { CombineOpcodeAndNumOperands(SpvOpBranchConditional, 3), 9987 },
    { CombineOpcodeAndNumOperands(SpvOpTypeFunction, 2), 7256 },
    { CombineOpcodeAndNumOperands(SpvOpConstantComposite, 5), 18073 },
    { CombineOpcodeAndNumOperands(SpvOpAccessChain, 4), 427562 },
    { CombineOpcodeAndNumOperands(SpvOpLabel, 1), 77328 },
    { CombineOpcodeAndNumOperands(SpvOpTypeMatrix, 3), 8362 },
    { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 4), 28585 },
    { CombineOpcodeAndNumOperands(SpvOpTypeInt, 3), 12783 },
    { CombineOpcodeAndNumOperands(SpvOpTypeArray, 3), 12325 },
    { CombineOpcodeAndNumOperands(SpvOpCapability, 1), 7622 },
    { CombineOpcodeAndNumOperands(SpvOpConstantComposite, 6), 14472 },
    { CombineOpcodeAndNumOperands(SpvOpTypeFunction, 3), 19783 },
    { CombineOpcodeAndNumOperands(SpvOpTypeVector, 3), 23463 },
    { CombineOpcodeAndNumOperands(SpvOpTypePointer, 3), 117527 },
    { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 7), 88112 },
    { CombineOpcodeAndNumOperands(SpvOpVectorTimesMatrix, 4), 5786 },
    { CombineOpcodeAndNumOperands(SpvOpTypeFloat, 2), 6401 },
    { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 4), 26463 },
    { CombineOpcodeAndNumOperands(SpvOpExtInstImport, 2), 6401 },
    { CombineOpcodeAndNumOperands(SpvOpDecorate, 3), 158529 },
    { CombineOpcodeAndNumOperands(SpvOpMemberDecorate, 4), 604149 },
    { CombineOpcodeAndNumOperands(SpvOpMemoryModel, 2), 6401 },
    { CombineOpcodeAndNumOperands(SpvOpImageSampleDrefExplicitLod, 7), 7119 },
    { CombineOpcodeAndNumOperands(SpvOpTypeImage, 8), 11045 },
    { CombineOpcodeAndNumOperands(SpvOpSampledImage, 4), 29330 },
    { CombineOpcodeAndNumOperands(SpvOpConstant, 3), 238196 },
    { CombineOpcodeAndNumOperands(SpvOpCompositeExtract, 4), 176855 },
    { CombineOpcodeAndNumOperands(SpvOpStore, 2), 457349 },
    { CombineOpcodeAndNumOperands(SpvOpFSub, 4), 54678 },
    { CombineOpcodeAndNumOperands(SpvOpAccessChain, 5), 74353 },
    { CombineOpcodeAndNumOperands(SpvOpCompositeExtract, 5), 6685 },
    { CombineOpcodeAndNumOperands(SpvOpExtInst, 6), 42506 },
    { CombineOpcodeAndNumOperands(SpvOpVectorTimesScalar, 4), 52904 },
    { CombineOpcodeAndNumOperands(SpvOpReturnValue, 1), 40266 },
    { CombineOpcodeAndNumOperands(SpvOpFAdd, 4), 106441 },
    { CombineOpcodeAndNumOperands(SpvOpDecorate, 2), 27139 },
    { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 8), 42240 },
    { CombineOpcodeAndNumOperands(SpvOpFMul, 4), 132212 },
    { CombineOpcodeAndNumOperands(SpvOpFDiv, 4), 14638 },
    { CombineOpcodeAndNumOperands(SpvOpImageSampleExplicitLod, 6), 9853 },
    { CombineOpcodeAndNumOperands(SpvOpImageSampleExplicitLod, 7), 9230 },
    { CombineOpcodeAndNumOperands(SpvOpExtInst, 5), 38407 },
    { CombineOpcodeAndNumOperands(SpvOpFunction, 4), 49767 },
    { CombineOpcodeAndNumOperands(SpvOpBranch, 1), 16344 },
    { CombineOpcodeAndNumOperands(SpvOpFNegate, 3), 9701 },
    { CombineOpcodeAndNumOperands(SpvOpImageSampleImplicitLod, 4), 39072 },
    { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 6), 65100 },
    { CombineOpcodeAndNumOperands(SpvOpDot, 4), 48636 },
    { CombineOpcodeAndNumOperands(SpvOpExtInst, 7), 50592 },
    { CombineOpcodeAndNumOperands(SpvOpMemberDecorate, 3), 29331 },
    { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 6), 58080 },
    { kMarkvNoneOfTheAbove, 265104 },
  });
}

std::map<uint32_t, std::unique_ptr<HuffmanCodec<uint64_t>>>
GetOpcodeAndNumOperandsMarkovHuffmanCodecs() {
  std::map<uint32_t, std::unique_ptr<HuffmanCodec<uint64_t>>> codecs;
  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpCompositeExtract, 4), 27 },
      { CombineOpcodeAndNumOperands(SpvOpFMul, 4), 1979 },
      { CombineOpcodeAndNumOperands(SpvOpStore, 2), 3554 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 6), 1120 },
      { CombineOpcodeAndNumOperands(SpvOpReturnValue, 1), 1040 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 7), 2943 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 5), 6 },
      { CombineOpcodeAndNumOperands(SpvOpLoad, 3), 5271 },
      { CombineOpcodeAndNumOperands(SpvOpVectorTimesScalar, 4), 480 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 4), 2663 },
      { kMarkvNoneOfTheAbove, 1909 },
    }));

    codecs.emplace(SpvOpImageSampleExplicitLod, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpFDiv, 4), 2 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 5), 3 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 4), 58 },
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 4), 240 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 5), 802 },
      { CombineOpcodeAndNumOperands(SpvOpReturnValue, 1), 23 },
      { CombineOpcodeAndNumOperands(SpvOpStore, 2), 2700 },
      { CombineOpcodeAndNumOperands(SpvOpFAdd, 4), 843 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 5), 1809 },
      { CombineOpcodeAndNumOperands(SpvOpImageSampleExplicitLod, 6), 280 },
      { CombineOpcodeAndNumOperands(SpvOpLoad, 3), 3201 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 6), 15 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 6), 400 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 4), 3572 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 7), 185 },
      { CombineOpcodeAndNumOperands(SpvOpFSub, 4), 103 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 7), 2 },
      { CombineOpcodeAndNumOperands(SpvOpFMul, 4), 276 },
      { CombineOpcodeAndNumOperands(SpvOpImageSampleImplicitLod, 4), 8 },
      { CombineOpcodeAndNumOperands(SpvOpVectorTimesScalar, 4), 71 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 6), 5 },
      { CombineOpcodeAndNumOperands(SpvOpImageSampleImplicitLod, 6), 29 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeExtract, 4), 1 },
      { kMarkvNoneOfTheAbove, 1464 },
    }));

    codecs.emplace(SpvOpFDiv, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 6), 2304 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 4), 1596 },
      { CombineOpcodeAndNumOperands(SpvOpImageSampleImplicitLod, 4), 2400 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 4), 9686 },
      { CombineOpcodeAndNumOperands(SpvOpImageSampleExplicitLod, 7), 1160 },
      { CombineOpcodeAndNumOperands(SpvOpImageSampleExplicitLod, 6), 160 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeExtract, 4), 1824 },
      { CombineOpcodeAndNumOperands(SpvOpLoad, 3), 10200 },
      { kMarkvNoneOfTheAbove, 2934 },
    }));

    codecs.emplace(SpvOpSampledImage, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpImageSampleImplicitLod, 4), 1 },
      { CombineOpcodeAndNumOperands(SpvOpFNegate, 3), 71 },
      { CombineOpcodeAndNumOperands(SpvOpFMod, 4), 288 },
      { CombineOpcodeAndNumOperands(SpvOpReturnValue, 1), 1238 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 5), 4078 },
      { CombineOpcodeAndNumOperands(SpvOpStore, 2), 22982 },
      { CombineOpcodeAndNumOperands(SpvOpFDiv, 4), 545 },
      { CombineOpcodeAndNumOperands(SpvOpImageSampleExplicitLod, 6), 4 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 5), 1006 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeExtract, 4), 829 },
      { CombineOpcodeAndNumOperands(SpvOpLoad, 3), 17711 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 6), 1192 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 4), 24358 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 4), 3558 },
      { CombineOpcodeAndNumOperands(SpvOpFAdd, 4), 29220 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 5), 2215 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 6), 294 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 7), 4655 },
      { CombineOpcodeAndNumOperands(SpvOpFSub, 4), 6184 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 7), 999 },
      { CombineOpcodeAndNumOperands(SpvOpFMul, 4), 6619 },
      { CombineOpcodeAndNumOperands(SpvOpVectorTimesScalar, 4), 138 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 6), 1103 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 6), 1528 },
      { CombineOpcodeAndNumOperands(SpvOpDot, 4), 1313 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 8), 75 },
      { kMarkvNoneOfTheAbove, 13221 },
    }));

    codecs.emplace(SpvOpFMul, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpImageSampleImplicitLod, 6), 1 },
      { CombineOpcodeAndNumOperands(SpvOpVectorTimesMatrix, 4), 1 },
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 5), 4 },
      { CombineOpcodeAndNumOperands(SpvOpFSub, 4), 21 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 7), 4069 },
      { CombineOpcodeAndNumOperands(SpvOpReturnValue, 1), 645 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 5), 1786 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeExtract, 4), 3916 },
      { CombineOpcodeAndNumOperands(SpvOpImageSampleImplicitLod, 4), 5776 },
      { CombineOpcodeAndNumOperands(SpvOpStore, 2), 31806 },
      { CombineOpcodeAndNumOperands(SpvOpFDiv, 4), 1150 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 4), 15999 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 6), 69 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 5), 2578 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 4), 2879 },
      { CombineOpcodeAndNumOperands(SpvOpLoad, 3), 15568 },
      { CombineOpcodeAndNumOperands(SpvOpFOrdLessThan, 4), 869 },
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 4), 7 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 8), 64 },
      { CombineOpcodeAndNumOperands(SpvOpFMul, 4), 8446 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 7), 82 },
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 8), 1 },
      { CombineOpcodeAndNumOperands(SpvOpFAdd, 4), 931 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 5), 3517 },
      { CombineOpcodeAndNumOperands(SpvOpImageSampleExplicitLod, 6), 32 },
      { CombineOpcodeAndNumOperands(SpvOpVectorTimesScalar, 4), 58 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 6), 757 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 6), 3770 },
      { CombineOpcodeAndNumOperands(SpvOpDot, 4), 1566 },
      { kMarkvNoneOfTheAbove, 10641 },
    }));

    codecs.emplace(SpvOpFAdd, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpTypeFunction, 11), 1 },
      { CombineOpcodeAndNumOperands(SpvOpTypeFunction, 9), 1 },
      { CombineOpcodeAndNumOperands(SpvOpTypeFunction, 7), 5 },
      { CombineOpcodeAndNumOperands(SpvOpTypeFunction, 8), 109 },
      { CombineOpcodeAndNumOperands(SpvOpTypeFunction, 5), 516 },
      { CombineOpcodeAndNumOperands(SpvOpTypeMatrix, 3), 57 },
      { CombineOpcodeAndNumOperands(SpvOpTypeVector, 3), 1480 },
      { CombineOpcodeAndNumOperands(SpvOpConstantComposite, 5), 1207 },
      { CombineOpcodeAndNumOperands(SpvOpTypeFunction, 3), 6062 },
      { CombineOpcodeAndNumOperands(SpvOpTypeBool, 1), 346 },
      { CombineOpcodeAndNumOperands(SpvOpTypePointer, 3), 9166 },
      { CombineOpcodeAndNumOperands(SpvOpConstant, 3), 21334 },
      { CombineOpcodeAndNumOperands(SpvOpTypeInt, 3), 1358 },
      { CombineOpcodeAndNumOperands(SpvOpTypeImage, 8), 2741 },
      { CombineOpcodeAndNumOperands(SpvOpVariable, 3), 64256 },
      { CombineOpcodeAndNumOperands(SpvOpTypeFunction, 6), 2378 },
      { CombineOpcodeAndNumOperands(SpvOpTypeFunction, 13), 1 },
      { CombineOpcodeAndNumOperands(SpvOpFunction, 4), 54 },
      { CombineOpcodeAndNumOperands(SpvOpTypeSampler, 1), 280 },
      { CombineOpcodeAndNumOperands(SpvOpConstantComposite, 6), 6 },
      { CombineOpcodeAndNumOperands(SpvOpTypeFunction, 4), 1644 },
      { CombineOpcodeAndNumOperands(SpvOpTypeFunction, 10), 285 },
      { CombineOpcodeAndNumOperands(SpvOpTypeFunction, 2), 60 },
      { CombineOpcodeAndNumOperands(SpvOpConstantComposite, 4), 645 },
      { kMarkvNoneOfTheAbove, 11576 },
    }));

    codecs.emplace(SpvOpTypePointer, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 6), 1 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 5), 2072 },
      { CombineOpcodeAndNumOperands(SpvOpReturnValue, 1), 5373 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 7), 4081 },
      { CombineOpcodeAndNumOperands(SpvOpFSub, 4), 1894 },
      { CombineOpcodeAndNumOperands(SpvOpStore, 2), 4759 },
      { CombineOpcodeAndNumOperands(SpvOpFDiv, 4), 1035 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 5), 6190 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 4), 9448 },
      { CombineOpcodeAndNumOperands(SpvOpFMul, 4), 3632 },
      { CombineOpcodeAndNumOperands(SpvOpDot, 4), 3227 },
      { CombineOpcodeAndNumOperands(SpvOpVectorTimesScalar, 4), 1247 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 6), 2188 },
      { CombineOpcodeAndNumOperands(SpvOpFOrdLessThan, 4), 835 },
      { CombineOpcodeAndNumOperands(SpvOpLoad, 3), 4335 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 6), 120 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 5), 2380 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 6), 30 },
      { CombineOpcodeAndNumOperands(SpvOpFAdd, 4), 1185 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 4), 470 },
      { CombineOpcodeAndNumOperands(SpvOpFNegate, 3), 160 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeExtract, 4), 1 },
      { kMarkvNoneOfTheAbove, 5468 },
    }));

    codecs.emplace(SpvOpFSub, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpFNegate, 3), 3 },
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 6), 4 },
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 5), 2 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 5), 6868 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 5), 328 },
      { CombineOpcodeAndNumOperands(SpvOpReturnValue, 1), 1 },
      { CombineOpcodeAndNumOperands(SpvOpBitcast, 3), 640 },
      { CombineOpcodeAndNumOperands(SpvOpStore, 2), 1689 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeExtract, 5), 4128 },
      { CombineOpcodeAndNumOperands(SpvOpImageSampleExplicitLod, 6), 640 },
      { CombineOpcodeAndNumOperands(SpvOpVectorExtractDynamic, 4), 1920 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 5), 241 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 4), 2667 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 6), 31152 },
      { CombineOpcodeAndNumOperands(SpvOpLoad, 3), 1524 },
      { CombineOpcodeAndNumOperands(SpvOpFOrdGreaterThanEqual, 4), 1082 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeExtract, 4), 119412 },
      { CombineOpcodeAndNumOperands(SpvOpShiftLeftLogical, 4), 1644 },
      { CombineOpcodeAndNumOperands(SpvOpImageSampleDrefExplicitLod, 7), 7119 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 4), 995 },
      { CombineOpcodeAndNumOperands(SpvOpImageSampleExplicitLod, 7), 800 },
      { CombineOpcodeAndNumOperands(SpvOpVectorTimesScalar, 4), 108 },
      { CombineOpcodeAndNumOperands(SpvOpFSub, 4), 119 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 7), 6 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 7), 40 },
      { CombineOpcodeAndNumOperands(SpvOpFMul, 4), 388 },
      { kMarkvNoneOfTheAbove, 18354 },
    }));

    codecs.emplace(SpvOpCompositeExtract, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpCompositeExtract, 4), 16 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 6), 105 },
      { CombineOpcodeAndNumOperands(SpvOpLoad, 3), 1657 },
      { CombineOpcodeAndNumOperands(SpvOpStore, 2), 1788 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 4), 536 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 5), 800 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 7), 1 },
      { CombineOpcodeAndNumOperands(SpvOpReturnValue, 1), 883 },
      { kMarkvNoneOfTheAbove, 579 },
    }));

    codecs.emplace(SpvOpVectorTimesMatrix, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpLabel, 1), 16344 },
      { kMarkvNoneOfTheAbove, 1635 },
    }));

    codecs.emplace(SpvOpBranch, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpFunction, 4), 43366 },
      { kMarkvNoneOfTheAbove, 4337 },
    }));

    codecs.emplace(SpvOpFunctionEnd, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpLabel, 1), 9987 },
      { kMarkvNoneOfTheAbove, 999 },
    }));

    codecs.emplace(SpvOpBranchConditional, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpFDiv, 4), 1 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 5), 5 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 4), 18 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 5), 47 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 7), 4 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 6), 832 },
      { CombineOpcodeAndNumOperands(SpvOpImageSampleImplicitLod, 4), 494 },
      { CombineOpcodeAndNumOperands(SpvOpFSub, 4), 407 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 5), 3194 },
      { CombineOpcodeAndNumOperands(SpvOpStore, 2), 13385 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 4), 4715 },
      { CombineOpcodeAndNumOperands(SpvOpFNegate, 3), 1601 },
      { CombineOpcodeAndNumOperands(SpvOpLoad, 3), 15278 },
      { CombineOpcodeAndNumOperands(SpvOpVectorTimesMatrix, 4), 8 },
      { CombineOpcodeAndNumOperands(SpvOpReturnValue, 1), 5045 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeExtract, 4), 247 },
      { CombineOpcodeAndNumOperands(SpvOpFAdd, 4), 115 },
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 4), 48 },
      { CombineOpcodeAndNumOperands(SpvOpFMul, 4), 81 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 7), 151 },
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 3), 16 },
      { CombineOpcodeAndNumOperands(SpvOpVectorTimesScalar, 4), 130 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 6), 8 },
      { CombineOpcodeAndNumOperands(SpvOpSelectionMerge, 2), 16 },
      { CombineOpcodeAndNumOperands(SpvOpDot, 4), 800 },
      { CombineOpcodeAndNumOperands(SpvOpBranch, 1), 18 },
      { kMarkvNoneOfTheAbove, 4668 },
    }));

    codecs.emplace(SpvOpFunctionCall, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpFDiv, 4), 1 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 6), 3 },
      { CombineOpcodeAndNumOperands(SpvOpDot, 4), 1 },
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 5), 16 },
      { CombineOpcodeAndNumOperands(SpvOpFSub, 4), 3 },
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 3), 1 },
      { CombineOpcodeAndNumOperands(SpvOpTypeBool, 1), 1489 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 5), 3168 },
      { CombineOpcodeAndNumOperands(SpvOpTypeMatrix, 3), 1506 },
      { CombineOpcodeAndNumOperands(SpvOpConstantComposite, 6), 2205 },
      { CombineOpcodeAndNumOperands(SpvOpConstantComposite, 4), 626 },
      { CombineOpcodeAndNumOperands(SpvOpConstant, 3), 62291 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 4), 7106 },
      { CombineOpcodeAndNumOperands(SpvOpTypeSampledImage, 2), 4328 },
      { CombineOpcodeAndNumOperands(SpvOpVariable, 3), 222855 },
      { CombineOpcodeAndNumOperands(SpvOpTypeVector, 3), 2603 },
      { CombineOpcodeAndNumOperands(SpvOpLoad, 3), 19794 },
      { CombineOpcodeAndNumOperands(SpvOpFunction, 4), 2834 },
      { CombineOpcodeAndNumOperands(SpvOpTypeImage, 8), 1156 },
      { CombineOpcodeAndNumOperands(SpvOpTypeInt, 3), 6859 },
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 4), 121 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 8), 198 },
      { CombineOpcodeAndNumOperands(SpvOpTypeArray, 3), 21 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 5), 797 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeExtract, 4), 266 },
      { CombineOpcodeAndNumOperands(SpvOpStore, 2), 1045 },
      { CombineOpcodeAndNumOperands(SpvOpTypeSampler, 1), 2116 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 6), 121 },
      { CombineOpcodeAndNumOperands(SpvOpTypePointer, 3), 18656 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 7), 484 },
      { CombineOpcodeAndNumOperands(SpvOpConstantComposite, 5), 1764 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 6), 1 },
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 6), 2 },
      { kMarkvNoneOfTheAbove, 36800 },
    }));

    codecs.emplace(SpvOpVariable, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpLoad, 3), 421125 },
      { CombineOpcodeAndNumOperands(SpvOpStore, 2), 87356 },
      { kMarkvNoneOfTheAbove, 50849 },
    }));

    codecs.emplace(SpvOpAccessChain, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 4), 4 },
      { CombineOpcodeAndNumOperands(SpvOpFSub, 4), 3 },
      { CombineOpcodeAndNumOperands(SpvOpFDiv, 4), 1 },
      { CombineOpcodeAndNumOperands(SpvOpStore, 2), 3424 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 4), 4009 },
      { CombineOpcodeAndNumOperands(SpvOpSelectionMerge, 2), 552 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 5), 184 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 6), 536 },
      { CombineOpcodeAndNumOperands(SpvOpLoad, 3), 29533 },
      { CombineOpcodeAndNumOperands(SpvOpReturnValue, 1), 432 },
      { CombineOpcodeAndNumOperands(SpvOpVariable, 3), 33480 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 6), 2 },
      { CombineOpcodeAndNumOperands(SpvOpVectorTimesScalar, 4), 3 },
      { CombineOpcodeAndNumOperands(SpvOpFNegate, 3), 346 },
      { CombineOpcodeAndNumOperands(SpvOpBranch, 1), 1194 },
      { CombineOpcodeAndNumOperands(SpvOpDot, 4), 492 },
      { CombineOpcodeAndNumOperands(SpvOpLoopMerge, 3), 839 },
      { CombineOpcodeAndNumOperands(SpvOpReturn, 0), 111 },
      { CombineOpcodeAndNumOperands(SpvOpPhi, 6), 290 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeExtract, 4), 138 },
      { CombineOpcodeAndNumOperands(SpvOpKill, 0), 1169 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 7), 365 },
      { CombineOpcodeAndNumOperands(SpvOpFMul, 4), 105 },
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 3), 2 },
      { kMarkvNoneOfTheAbove, 7728 },
    }));

    codecs.emplace(SpvOpLabel, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpLabel, 1), 9 },
      { CombineOpcodeAndNumOperands(SpvOpFunctionEnd, 0), 9553 },
      { kMarkvNoneOfTheAbove, 957 },
    }));

    codecs.emplace(SpvOpReturn, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpFunctionParameter, 2), 42510 },
      { CombineOpcodeAndNumOperands(SpvOpLabel, 1), 7257 },
      { kMarkvNoneOfTheAbove, 4977 },
    }));

    codecs.emplace(SpvOpFunction, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpTypeFunction, 5), 4 },
      { CombineOpcodeAndNumOperands(SpvOpFunction, 4), 1 },
      { CombineOpcodeAndNumOperands(SpvOpTypeVector, 3), 2528 },
      { CombineOpcodeAndNumOperands(SpvOpConstant, 3), 1477 },
      { CombineOpcodeAndNumOperands(SpvOpTypeImage, 8), 4 },
      { CombineOpcodeAndNumOperands(SpvOpTypeInt, 3), 1581 },
      { CombineOpcodeAndNumOperands(SpvOpTypePointer, 3), 12482 },
      { CombineOpcodeAndNumOperands(SpvOpConstantComposite, 6), 828 },
      { CombineOpcodeAndNumOperands(SpvOpTypeFunction, 4), 714 },
      { CombineOpcodeAndNumOperands(SpvOpConstantComposite, 4), 1605 },
      { CombineOpcodeAndNumOperands(SpvOpVariable, 3), 20 },
      { CombineOpcodeAndNumOperands(SpvOpTypeMatrix, 3), 58 },
      { CombineOpcodeAndNumOperands(SpvOpTypeFunction, 3), 132 },
      { kMarkvNoneOfTheAbove, 2245 },
    }));

    codecs.emplace(SpvOpTypeVector, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpFunctionParameter, 2), 41916 },
      { CombineOpcodeAndNumOperands(SpvOpLabel, 1), 42510 },
      { kMarkvNoneOfTheAbove, 8443 },
    }));

    codecs.emplace(SpvOpFunctionParameter, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpLabel, 1), 52 },
      { CombineOpcodeAndNumOperands(SpvOpFunctionEnd, 0), 40214 },
      { kMarkvNoneOfTheAbove, 4027 },
    }));

    codecs.emplace(SpvOpReturnValue, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpTypeFunction, 2), 6401 },
      { kMarkvNoneOfTheAbove, 641 },
    }));

    codecs.emplace(SpvOpTypeVoid, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 7), 1 },
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 5), 4051 },
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 13), 768 },
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 8), 1406 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 8), 1 },
      { CombineOpcodeAndNumOperands(SpvOpFAdd, 4), 64 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 7), 713 },
      { CombineOpcodeAndNumOperands(SpvOpFMul, 4), 1613 },
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 6), 3242 },
      { CombineOpcodeAndNumOperands(SpvOpLoad, 3), 175188 },
      { CombineOpcodeAndNumOperands(SpvOpReturn, 0), 9451 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 4), 2203 },
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 4), 22911 },
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 7), 4918 },
      { CombineOpcodeAndNumOperands(SpvOpFNegate, 3), 956 },
      { CombineOpcodeAndNumOperands(SpvOpBranch, 1), 14018 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 4), 136310 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeExtract, 4), 2738 },
      { CombineOpcodeAndNumOperands(SpvOpFDiv, 4), 572 },
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 11), 352 },
      { CombineOpcodeAndNumOperands(SpvOpVectorTimesScalar, 4), 403 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeExtract, 5), 1440 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 6), 4 },
      { CombineOpcodeAndNumOperands(SpvOpStore, 2), 32814 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 5), 36375 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 6), 1458 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 6), 463 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 5), 895 },
      { CombineOpcodeAndNumOperands(SpvOpFSub, 4), 191 },
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 3), 218 },
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 14), 81 },
      { CombineOpcodeAndNumOperands(SpvOpDot, 4), 5 },
      { CombineOpcodeAndNumOperands(SpvOpVectorTimesMatrix, 4), 76 },
      { CombineOpcodeAndNumOperands(SpvOpSelectionMerge, 2), 48 },
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 9), 297 },
      { CombineOpcodeAndNumOperands(SpvOpReturnValue, 1), 108 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 5), 252 },
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 10), 6 },
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 12), 2 },
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 15), 25 },
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 17), 24 },
      { kMarkvNoneOfTheAbove, 45701 },
    }));

    codecs.emplace(SpvOpStore, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpMemberDecorate, 4), 800 },
      { CombineOpcodeAndNumOperands(SpvOpDecorate, 2), 2 },
      { CombineOpcodeAndNumOperands(SpvOpExecutionMode, 2), 4428 },
      { CombineOpcodeAndNumOperands(SpvOpDecorate, 3), 1166 },
      { kMarkvNoneOfTheAbove, 640 },
    }));

    codecs.emplace(SpvOpEntryPoint, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 11), 14 },
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 8), 26 },
      { CombineOpcodeAndNumOperands(SpvOpIMul, 4), 998 },
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 4), 4243 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 6), 54110 },
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 9), 1 },
      { CombineOpcodeAndNumOperands(SpvOpReturnValue, 1), 13438 },
      { CombineOpcodeAndNumOperands(SpvOpBitcast, 3), 1644 },
      { CombineOpcodeAndNumOperands(SpvOpIAdd, 4), 1154 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeExtract, 4), 35977 },
      { CombineOpcodeAndNumOperands(SpvOpImageSampleExplicitLod, 7), 7270 },
      { CombineOpcodeAndNumOperands(SpvOpSampledImage, 4), 29330 },
      { CombineOpcodeAndNumOperands(SpvOpFMul, 4), 68497 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 7), 63947 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 4), 116977 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 5), 11660 },
      { CombineOpcodeAndNumOperands(SpvOpFSub, 4), 24748 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 5), 7352 },
      { CombineOpcodeAndNumOperands(SpvOpImageSampleExplicitLod, 6), 6836 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 6), 30414 },
      { CombineOpcodeAndNumOperands(SpvOpStore, 2), 99067 },
      { CombineOpcodeAndNumOperands(SpvOpFDiv, 4), 3615 },
      { CombineOpcodeAndNumOperands(SpvOpVectorTimesScalar, 4), 40750 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeExtract, 5), 1117 },
      { CombineOpcodeAndNumOperands(SpvOpVectorTimesMatrix, 4), 3198 },
      { CombineOpcodeAndNumOperands(SpvOpLoad, 3), 98776 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 8), 41896 },
      { CombineOpcodeAndNumOperands(SpvOpFAdd, 4), 42193 },
      { CombineOpcodeAndNumOperands(SpvOpDot, 4), 24825 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 6), 1221 },
      { CombineOpcodeAndNumOperands(SpvOpFOrdLessThan, 4), 2308 },
      { CombineOpcodeAndNumOperands(SpvOpImageSampleImplicitLod, 4), 15753 },
      { CombineOpcodeAndNumOperands(SpvOpFNegate, 3), 5094 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 6), 13841 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 4), 5703 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 7), 6237 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 5), 23796 },
      { CombineOpcodeAndNumOperands(SpvOpSelectionMerge, 2), 550 },
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 5), 1763 },
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 7), 122 },
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 6), 88 },
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 3), 494 },
      { kMarkvNoneOfTheAbove, 91436 },
    }));

    codecs.emplace(SpvOpLoad, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpEntryPoint, 26), 2 },
      { CombineOpcodeAndNumOperands(SpvOpEntryPoint, 17), 102 },
      { CombineOpcodeAndNumOperands(SpvOpEntryPoint, 6), 306 },
      { CombineOpcodeAndNumOperands(SpvOpEntryPoint, 14), 910 },
      { CombineOpcodeAndNumOperands(SpvOpEntryPoint, 19), 321 },
      { CombineOpcodeAndNumOperands(SpvOpEntryPoint, 8), 426 },
      { CombineOpcodeAndNumOperands(SpvOpEntryPoint, 7), 969 },
      { CombineOpcodeAndNumOperands(SpvOpEntryPoint, 10), 153 },
      { CombineOpcodeAndNumOperands(SpvOpEntryPoint, 11), 317 },
      { CombineOpcodeAndNumOperands(SpvOpEntryPoint, 13), 1056 },
      { CombineOpcodeAndNumOperands(SpvOpEntryPoint, 12), 592 },
      { CombineOpcodeAndNumOperands(SpvOpEntryPoint, 18), 183 },
      { CombineOpcodeAndNumOperands(SpvOpEntryPoint, 15), 259 },
      { CombineOpcodeAndNumOperands(SpvOpEntryPoint, 20), 241 },
      { CombineOpcodeAndNumOperands(SpvOpEntryPoint, 21), 93 },
      { CombineOpcodeAndNumOperands(SpvOpEntryPoint, 16), 50 },
      { CombineOpcodeAndNumOperands(SpvOpEntryPoint, 22), 29 },
      { CombineOpcodeAndNumOperands(SpvOpEntryPoint, 5), 322 },
      { CombineOpcodeAndNumOperands(SpvOpEntryPoint, 23), 23 },
      { CombineOpcodeAndNumOperands(SpvOpEntryPoint, 9), 21 },
      { CombineOpcodeAndNumOperands(SpvOpEntryPoint, 4), 3 },
      { CombineOpcodeAndNumOperands(SpvOpEntryPoint, 24), 15 },
      { CombineOpcodeAndNumOperands(SpvOpEntryPoint, 25), 8 },
      { kMarkvNoneOfTheAbove, 641 },
    }));

    codecs.emplace(SpvOpMemoryModel, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpTypeFunction, 3), 1 },
      { CombineOpcodeAndNumOperands(SpvOpTypePointer, 3), 30 },
      { CombineOpcodeAndNumOperands(SpvOpTypeVector, 3), 6370 },
      { kMarkvNoneOfTheAbove, 641 },
    }));

    codecs.emplace(SpvOpTypeFloat, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 7), 6 },
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 4), 32 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 6), 5467 },
      { CombineOpcodeAndNumOperands(SpvOpSelect, 5), 336 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 6), 1343 },
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 8), 1 },
      { CombineOpcodeAndNumOperands(SpvOpFAdd, 4), 4114 },
      { CombineOpcodeAndNumOperands(SpvOpReturnValue, 1), 2365 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 7), 18944 },
      { CombineOpcodeAndNumOperands(SpvOpFSub, 4), 4936 },
      { CombineOpcodeAndNumOperands(SpvOpVectorTimesMatrix, 4), 2503 },
      { CombineOpcodeAndNumOperands(SpvOpLoad, 3), 17658 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 4), 18807 },
      { CombineOpcodeAndNumOperands(SpvOpStore, 2), 33468 },
      { CombineOpcodeAndNumOperands(SpvOpFDiv, 4), 3683 },
      { CombineOpcodeAndNumOperands(SpvOpImageSampleExplicitLod, 6), 1596 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 5), 1945 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 6), 538 },
      { CombineOpcodeAndNumOperands(SpvOpVectorTimesScalar, 4), 296 },
      { CombineOpcodeAndNumOperands(SpvOpImageSampleImplicitLod, 4), 531 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 7), 1410 },
      { CombineOpcodeAndNumOperands(SpvOpFMul, 4), 10780 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 5), 6456 },
      { CombineOpcodeAndNumOperands(SpvOpMatrixTimesVector, 4), 387 },
      { CombineOpcodeAndNumOperands(SpvOpFNegate, 3), 384 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeExtract, 4), 3456 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 4), 2423 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 6), 16 },
      { CombineOpcodeAndNumOperands(SpvOpDot, 4), 4130 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 5), 436 },
      { kMarkvNoneOfTheAbove, 14845 },
    }));

    codecs.emplace(SpvOpCompositeConstruct, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpTypeFunction, 9), 2 },
      { CombineOpcodeAndNumOperands(SpvOpTypeFunction, 14), 1 },
      { CombineOpcodeAndNumOperands(SpvOpTypeFunction, 10), 66 },
      { CombineOpcodeAndNumOperands(SpvOpTypeFunction, 16), 1 },
      { CombineOpcodeAndNumOperands(SpvOpTypePointer, 3), 3798 },
      { CombineOpcodeAndNumOperands(SpvOpTypeFunction, 8), 187 },
      { CombineOpcodeAndNumOperands(SpvOpTypeFunction, 6), 985 },
      { CombineOpcodeAndNumOperands(SpvOpTypeFunction, 5), 2016 },
      { CombineOpcodeAndNumOperands(SpvOpTypeFunction, 3), 10041 },
      { CombineOpcodeAndNumOperands(SpvOpTypeVector, 3), 6139 },
      { CombineOpcodeAndNumOperands(SpvOpTypeFunction, 7), 1330 },
      { CombineOpcodeAndNumOperands(SpvOpTypeMatrix, 3), 1208 },
      { CombineOpcodeAndNumOperands(SpvOpTypeFunction, 2), 795 },
      { CombineOpcodeAndNumOperands(SpvOpTypeInt, 3), 1824 },
      { CombineOpcodeAndNumOperands(SpvOpTypeImage, 8), 744 },
      { CombineOpcodeAndNumOperands(SpvOpTypeFloat, 2), 6383 },
      { CombineOpcodeAndNumOperands(SpvOpTypeFunction, 4), 2777 },
      { CombineOpcodeAndNumOperands(SpvOpTypeFunction, 12), 768 },
      { kMarkvNoneOfTheAbove, 4003 },
    }));

    codecs.emplace(SpvOpTypeFunction, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpMemoryModel, 2), 6401 },
      { kMarkvNoneOfTheAbove, 641 },
    }));

    codecs.emplace(SpvOpExtInstImport, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpTypeSampledImage, 2), 6717 },
      { CombineOpcodeAndNumOperands(SpvOpTypePointer, 3), 4328 },
      { kMarkvNoneOfTheAbove, 1105 },
    }));

    codecs.emplace(SpvOpTypeImage, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpDecorate, 3), 4 },
      { CombineOpcodeAndNumOperands(SpvOpDecorate, 2), 26812 },
      { CombineOpcodeAndNumOperands(SpvOpMemberDecorate, 3), 24447 },
      { CombineOpcodeAndNumOperands(SpvOpMemberDecorate, 4), 582217 },
      { kMarkvNoneOfTheAbove, 63349 },
    }));

    codecs.emplace(SpvOpMemberDecorate, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpCapability, 1), 1221 },
      { CombineOpcodeAndNumOperands(SpvOpExtInstImport, 2), 6401 },
      { kMarkvNoneOfTheAbove, 763 },
    }));

    codecs.emplace(SpvOpCapability, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpTypeFunction, 7), 1 },
      { CombineOpcodeAndNumOperands(SpvOpTypeFunction, 5), 8 },
      { CombineOpcodeAndNumOperands(SpvOpTypeMatrix, 3), 40 },
      { CombineOpcodeAndNumOperands(SpvOpTypeInt, 3), 20 },
      { CombineOpcodeAndNumOperands(SpvOpTypePointer, 3), 319 },
      { CombineOpcodeAndNumOperands(SpvOpConstant, 3), 11718 },
      { CombineOpcodeAndNumOperands(SpvOpTypeVector, 3), 672 },
      { kMarkvNoneOfTheAbove, 1279 },
    }));

    codecs.emplace(SpvOpTypeInt, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpTypeBool, 1), 1111 },
      { CombineOpcodeAndNumOperands(SpvOpConstantComposite, 6), 1090 },
      { CombineOpcodeAndNumOperands(SpvOpConstantComposite, 5), 1540 },
      { CombineOpcodeAndNumOperands(SpvOpTypeVector, 3), 956 },
      { CombineOpcodeAndNumOperands(SpvOpConstantComposite, 4), 3489 },
      { CombineOpcodeAndNumOperands(SpvOpTypePointer, 3), 5054 },
      { CombineOpcodeAndNumOperands(SpvOpTypeArray, 3), 3 },
      { CombineOpcodeAndNumOperands(SpvOpTypeMatrix, 3), 143 },
      { CombineOpcodeAndNumOperands(SpvOpConstant, 3), 30574 },
      { CombineOpcodeAndNumOperands(SpvOpFunction, 4), 18 },
      { CombineOpcodeAndNumOperands(SpvOpTypeImage, 8), 3597 },
      { CombineOpcodeAndNumOperands(SpvOpTypeInt, 3), 11 },
      { CombineOpcodeAndNumOperands(SpvOpVariable, 3), 4858 },
      { kMarkvNoneOfTheAbove, 5283 },
    }));

    codecs.emplace(SpvOpConstantComposite, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpConstantComposite, 5), 4 },
      { CombineOpcodeAndNumOperands(SpvOpVariable, 3), 892 },
      { CombineOpcodeAndNumOperands(SpvOpTypePointer, 3), 6335 },
      { CombineOpcodeAndNumOperands(SpvOpConstant, 3), 3010 },
      { CombineOpcodeAndNumOperands(SpvOpTypeVector, 3), 800 },
      { kMarkvNoneOfTheAbove, 1105 },
    }));

    codecs.emplace(SpvOpTypeSampledImage, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpTypeArray, 3), 3 },
      { CombineOpcodeAndNumOperands(SpvOpConstant, 3), 1 },
      { CombineOpcodeAndNumOperands(SpvOpTypePointer, 3), 18978 },
      { CombineOpcodeAndNumOperands(SpvOpVariable, 3), 1312 },
      { CombineOpcodeAndNumOperands(SpvOpFunction, 4), 3481 },
      { CombineOpcodeAndNumOperands(SpvOpTypeMatrix, 3), 800 },
      { CombineOpcodeAndNumOperands(SpvOpTypeFunction, 3), 2745 },
      { kMarkvNoneOfTheAbove, 3098 },
    }));

    codecs.emplace(SpvOpTypeStruct, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 5), 5 },
      { CombineOpcodeAndNumOperands(SpvOpFAdd, 4), 2 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 6), 3 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeExtract, 4), 54 },
      { CombineOpcodeAndNumOperands(SpvOpFMul, 4), 24 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 4), 13 },
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 4), 240 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 4), 1627 },
      { CombineOpcodeAndNumOperands(SpvOpVectorTimesScalar, 4), 253 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 6), 1152 },
      { CombineOpcodeAndNumOperands(SpvOpLoad, 3), 1722 },
      { CombineOpcodeAndNumOperands(SpvOpFNegate, 3), 384 },
      { CombineOpcodeAndNumOperands(SpvOpDot, 4), 1237 },
      { CombineOpcodeAndNumOperands(SpvOpFSub, 4), 2 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 7), 386 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 5), 11 },
      { CombineOpcodeAndNumOperands(SpvOpStore, 2), 550 },
      { CombineOpcodeAndNumOperands(SpvOpReturnValue, 1), 1598 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 5), 438 },
      { kMarkvNoneOfTheAbove, 971 },
    }));

    codecs.emplace(SpvOpFNegate, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpDecorate, 2), 325 },
      { CombineOpcodeAndNumOperands(SpvOpTypeVoid, 1), 6401 },
      { CombineOpcodeAndNumOperands(SpvOpMemberDecorate, 4), 21122 },
      { CombineOpcodeAndNumOperands(SpvOpMemberDecorate, 3), 4884 },
      { CombineOpcodeAndNumOperands(SpvOpDecorate, 3), 152936 },
      { kMarkvNoneOfTheAbove, 18567 },
    }));

    codecs.emplace(SpvOpDecorate, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpTypeFunction, 3), 801 },
      { CombineOpcodeAndNumOperands(SpvOpConstant, 3), 1435 },
      { CombineOpcodeAndNumOperands(SpvOpTypeInt, 3), 172 },
      { CombineOpcodeAndNumOperands(SpvOpTypePointer, 3), 2291 },
      { CombineOpcodeAndNumOperands(SpvOpTypeFunction, 5), 7 },
      { CombineOpcodeAndNumOperands(SpvOpTypeArray, 3), 2594 },
      { CombineOpcodeAndNumOperands(SpvOpTypeVector, 3), 692 },
      { CombineOpcodeAndNumOperands(SpvOpTypeMatrix, 3), 232 },
      { CombineOpcodeAndNumOperands(SpvOpTypeFunction, 4), 129 },
      { CombineOpcodeAndNumOperands(SpvOpVariable, 3), 4 },
      { CombineOpcodeAndNumOperands(SpvOpFunction, 4), 1 },
      { kMarkvNoneOfTheAbove, 837 },
    }));

    codecs.emplace(SpvOpTypeMatrix, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpTypeVector, 3), 572 },
      { CombineOpcodeAndNumOperands(SpvOpTypeBool, 1), 1275 },
      { CombineOpcodeAndNumOperands(SpvOpTypeFloat, 2), 18 },
      { CombineOpcodeAndNumOperands(SpvOpConstantComposite, 6), 10343 },
      { CombineOpcodeAndNumOperands(SpvOpConstantComposite, 4), 14299 },
      { CombineOpcodeAndNumOperands(SpvOpTypeInt, 3), 955 },
      { CombineOpcodeAndNumOperands(SpvOpTypeImage, 8), 2802 },
      { CombineOpcodeAndNumOperands(SpvOpFunction, 4), 12 },
      { CombineOpcodeAndNumOperands(SpvOpConstant, 3), 104685 },
      { CombineOpcodeAndNumOperands(SpvOpTypePointer, 3), 32681 },
      { CombineOpcodeAndNumOperands(SpvOpConstantComposite, 5), 12838 },
      { CombineOpcodeAndNumOperands(SpvOpVariable, 3), 43866 },
      { CombineOpcodeAndNumOperands(SpvOpTypeMatrix, 3), 4317 },
      { CombineOpcodeAndNumOperands(SpvOpTypeArray, 3), 4638 },
      { kMarkvNoneOfTheAbove, 23575 },
    }));

    codecs.emplace(SpvOpConstant, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpConstant, 3), 3 },
      { CombineOpcodeAndNumOperands(SpvOpTypeArray, 3), 5066 },
      { CombineOpcodeAndNumOperands(SpvOpTypePointer, 3), 21 },
      { kMarkvNoneOfTheAbove, 871 },
    }));

    codecs.emplace(SpvOpTypeArray, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 5), 3 },
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 6), 1 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 6), 11 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 8), 6 },
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 4), 2 },
      { CombineOpcodeAndNumOperands(SpvOpFDiv, 4), 831 },
      { CombineOpcodeAndNumOperands(SpvOpStore, 2), 23930 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 5), 5767 },
      { CombineOpcodeAndNumOperands(SpvOpReturnValue, 1), 5685 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 5), 8822 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 4), 31675 },
      { CombineOpcodeAndNumOperands(SpvOpFSub, 4), 7313 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 7), 1799 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 5), 2385 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 6), 529 },
      { CombineOpcodeAndNumOperands(SpvOpLoad, 3), 27990 },
      { CombineOpcodeAndNumOperands(SpvOpFNegate, 3), 518 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 6), 2 },
      { CombineOpcodeAndNumOperands(SpvOpDot, 4), 880 },
      { CombineOpcodeAndNumOperands(SpvOpVectorTimesScalar, 4), 1751 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 6), 2486 },
      { CombineOpcodeAndNumOperands(SpvOpFMul, 4), 6506 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 7), 253 },
      { CombineOpcodeAndNumOperands(SpvOpFAdd, 4), 184 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 4), 1928 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeExtract, 4), 208 },
      { kMarkvNoneOfTheAbove, 13149 },
    }));

    codecs.emplace(SpvOpExtInst, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 6), 1 },
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 4), 1 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 5), 975 },
      { CombineOpcodeAndNumOperands(SpvOpReturnValue, 1), 901 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 4), 2733 },
      { CombineOpcodeAndNumOperands(SpvOpFAdd, 4), 5213 },
      { CombineOpcodeAndNumOperands(SpvOpLoad, 3), 15195 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 6), 2288 },
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 6), 2 },
      { CombineOpcodeAndNumOperands(SpvOpStore, 2), 13741 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 5), 342 },
      { CombineOpcodeAndNumOperands(SpvOpFDiv, 4), 218 },
      { CombineOpcodeAndNumOperands(SpvOpVectorTimesScalar, 4), 411 },
      { CombineOpcodeAndNumOperands(SpvOpFSub, 4), 2792 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 7), 344 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 4), 7454 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeExtract, 4), 12 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 7), 58 },
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 3), 126 },
      { CombineOpcodeAndNumOperands(SpvOpFMul, 4), 6 },
      { CombineOpcodeAndNumOperands(SpvOpImageSampleImplicitLod, 4), 15 },
      { CombineOpcodeAndNumOperands(SpvOpImageSampleImplicitLod, 6), 4 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 5), 1 },
      { kMarkvNoneOfTheAbove, 5287 },
    }));

    codecs.emplace(SpvOpVectorTimesScalar, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 6), 3 },
      { CombineOpcodeAndNumOperands(SpvOpImageSampleImplicitLod, 6), 1 },
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 5), 9 },
      { CombineOpcodeAndNumOperands(SpvOpFNegate, 3), 178 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 6), 216 },
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 4), 374 },
      { CombineOpcodeAndNumOperands(SpvOpDot, 4), 5184 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeExtract, 4), 2730 },
      { CombineOpcodeAndNumOperands(SpvOpImageSampleImplicitLod, 4), 14094 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 4), 16271 },
      { CombineOpcodeAndNumOperands(SpvOpStore, 2), 61234 },
      { CombineOpcodeAndNumOperands(SpvOpFDiv, 4), 1138 },
      { CombineOpcodeAndNumOperands(SpvOpFAdd, 4), 19098 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 5), 3437 },
      { CombineOpcodeAndNumOperands(SpvOpImageSampleExplicitLod, 6), 305 },
      { CombineOpcodeAndNumOperands(SpvOpFMul, 4), 20427 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 7), 1486 },
      { CombineOpcodeAndNumOperands(SpvOpLoad, 3), 29303 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 6), 3302 },
      { CombineOpcodeAndNumOperands(SpvOpVectorTimesScalar, 4), 4714 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 4), 417 },
      { CombineOpcodeAndNumOperands(SpvOpReturnValue, 1), 1451 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 5), 2921 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 5), 1579 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 6), 1089 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 7), 1747 },
      { CombineOpcodeAndNumOperands(SpvOpFSub, 4), 2529 },
      { kMarkvNoneOfTheAbove, 19535 },
    }));

    codecs.emplace(SpvOpVectorShuffle, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpReturnValue, 1), 4 },
      { CombineOpcodeAndNumOperands(SpvOpStore, 2), 13520 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeExtract, 4), 3358 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 6), 1 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 7), 15178 },
      { CombineOpcodeAndNumOperands(SpvOpLoad, 3), 1515 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 4), 4176 },
      { CombineOpcodeAndNumOperands(SpvOpVectorTimesScalar, 4), 35 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 5), 1 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 6), 942 },
      { CombineOpcodeAndNumOperands(SpvOpFunctionCall, 4), 362 },
      { CombineOpcodeAndNumOperands(SpvOpDot, 4), 2 },
      { CombineOpcodeAndNumOperands(SpvOpFMul, 4), 13 },
      { kMarkvNoneOfTheAbove, 3911 },
    }));

    codecs.emplace(SpvOpImageSampleImplicitLod, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpFMul, 4), 2169 },
      { CombineOpcodeAndNumOperands(SpvOpStore, 2), 1159 },
      { CombineOpcodeAndNumOperands(SpvOpFDiv, 4), 1563 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 5), 880 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 7), 8134 },
      { CombineOpcodeAndNumOperands(SpvOpFSub, 4), 3028 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 6), 1279 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 5), 8612 },
      { CombineOpcodeAndNumOperands(SpvOpVectorShuffle, 6), 4 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 4), 5566 },
      { CombineOpcodeAndNumOperands(SpvOpExtInst, 6), 1554 },
      { CombineOpcodeAndNumOperands(SpvOpVectorTimesScalar, 4), 1080 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 6), 28 },
      { CombineOpcodeAndNumOperands(SpvOpDot, 4), 4974 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 5), 3951 },
      { CombineOpcodeAndNumOperands(SpvOpReturnValue, 1), 2 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 4), 1037 },
      { CombineOpcodeAndNumOperands(SpvOpFAdd, 4), 503 },
      { CombineOpcodeAndNumOperands(SpvOpLoad, 3), 3101 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeExtract, 4), 1 },
      { CombineOpcodeAndNumOperands(SpvOpFNegate, 3), 5 },
      { kMarkvNoneOfTheAbove, 4864 },
    }));

    codecs.emplace(SpvOpDot, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpBranchConditional, 3), 9148 },
      { kMarkvNoneOfTheAbove, 915 },
    }));

    codecs.emplace(SpvOpSelectionMerge, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpFMul, 4), 407 },
      { CombineOpcodeAndNumOperands(SpvOpStore, 2), 814 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 4), 407 },
      { CombineOpcodeAndNumOperands(SpvOpLoad, 3), 407 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 5), 814 },
      { CombineOpcodeAndNumOperands(SpvOpCompositeConstruct, 6), 814 },
      { CombineOpcodeAndNumOperands(SpvOpAccessChain, 4), 3456 },
      { kMarkvNoneOfTheAbove, 712 },
    }));

    codecs.emplace(SpvOpImageSampleDrefExplicitLod, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { CombineOpcodeAndNumOperands(SpvOpMemberDecorate, 4), 5 },
      { CombineOpcodeAndNumOperands(SpvOpDecorate, 3), 23 },
      { CombineOpcodeAndNumOperands(SpvOpMemberName, 3), 30 },
      { CombineOpcodeAndNumOperands(SpvOpName, 2), 5747 },
      { kMarkvNoneOfTheAbove, 581 },
    }));

    codecs.emplace(SpvOpName, std::move(codec));
  }

  return codecs;
}

std::map<uint32_t, std::unique_ptr<HuffmanCodec<std::string>>>
GetLiteralStringHuffmanCodecs() {
  std::map<uint32_t, std::unique_ptr<HuffmanCodec<std::string>>> codecs;
  {
    std::unique_ptr<HuffmanCodec<std::string>> codec(new HuffmanCodec<std::string>({
      { std::string("main"), 4005 },
      { std::string("MainPs"), 1596 },
      { std::string("MainVs"), 800 },
      { std::string("kMarkvNoneOfTheAbove"), 321 },
    }));

    codecs.emplace(SpvOpEntryPoint, std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<std::string>> codec(new HuffmanCodec<std::string>({
      { std::string("GLSL.std.450"), 6401 },
      { std::string("kMarkvNoneOfTheAbove"), 321 },
    }));

    codecs.emplace(SpvOpExtInstImport, std::move(codec));
  }

  return codecs;
}

std::map<std::pair<uint32_t, uint32_t>, std::unique_ptr<HuffmanCodec<uint64_t>>>
GetNonIdWordHuffmanCodecs() {
  std::map<std::pair<uint32_t, uint32_t>, std::unique_ptr<HuffmanCodec<uint64_t>>> codecs;
  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 0, 30 },
      { 97, 36 },
      { 98, 34 },
      { 99, 27 },
      { 100, 28 },
      { 101, 7 },
      { 102, 41 },
      { 104, 43 },
      { 105, 53 },
      { 106, 2 },
      { 107, 12 },
      { 108, 3 },
      { 109, 13 },
      { 110, 26 },
      { 112, 112 },
      { 113, 28 },
      { 114, 19 },
      { 115, 15 },
      { 116, 50 },
      { 117, 6 },
      { 118, 8 },
      { 119, 3 },
      { 120, 60 },
      { 121, 35 },
      { 122, 4 },
      { 12404, 2 },
      { 12642, 2 },
      { 12644, 13 },
      { 12646, 2 },
      { 12648, 3 },
      { 12658, 2 },
      { 12660, 2 },
      { 12897, 2 },
      { 12898, 3 },
      { 12900, 14 },
      { 12904, 2 },
      { 12912, 2 },
      { 12914, 3 },
      { 13154, 2 },
      { 13156, 4 },
      { 13168, 2 },
      { 13412, 4 },
      { 13668, 3 },
      { 16758, 7 },
      { 17014, 7 },
      { 17526, 6 },
      { 21606, 17 },
      { 22633, 3 },
      { 22889, 3 },
      { 23145, 3 },
      { 24930, 4 },
      { 24931, 4 },
      { 24941, 3 },
      { 24944, 5 },
      { 24948, 16 },
      { 24950, 2 },
      { 25202, 3 },
      { 25700, 3 },
      { 25705, 2 },
      { 25714, 45 },
      { 25970, 3 },
      { 25972, 2 },
      { 26214, 3 },
      { 26223, 2 },
      { 26482, 5 },
      { 26728, 2 },
      { 26736, 2 },
      { 26978, 2 },
      { 26989, 3 },
      { 26994, 2 },
      { 26995, 3 },
      { 27499, 6 },
      { 27750, 4 },
      { 28013, 2 },
      { 28020, 2 },
      { 28257, 3 },
      { 28513, 2 },
      { 28515, 3 },
      { 28518, 2 },
      { 28525, 3 },
      { 28530, 50 },
      { 28769, 2 },
      { 28770, 2 },
      { 28771, 8 },
      { 28784, 2 },
      { 28788, 2 },
      { 29283, 8 },
      { 29286, 4 },
      { 29298, 3 },
      { 29796, 3 },
      { 29804, 2 },
      { 29815, 2 },
      { 30051, 7 },
      { 30069, 3 },
      { 30307, 7 },
      { 30325, 29 },
      { 30326, 4 },
      { 30563, 7 },
      { 30583, 2 },
      { 30818, 2 },
      { 30820, 4 },
      { 30836, 2 },
      { 31096, 2 },
      { 31332, 3 },
      { 3158120, 2 },
      { 3158376, 2 },
      { 3171188, 6 },
      { 3223656, 2 },
      { 3223912, 2 },
      { 3306086, 2 },
      { 3371622, 2 },
      { 4280182, 6 },
      { 5460854, 17 },
      { 5657974, 6 },
      { 6383731, 5 },
      { 6450529, 5 },
      { 6512994, 4 },
      { 6513519, 7 },
      { 6647923, 3 },
      { 6648422, 7 },
      { 6710642, 7 },
      { 6710895, 2 },
      { 6711652, 10 },
      { 6777196, 4 },
      { 6778465, 2 },
      { 6779503, 2 },
      { 6910575, 2 },
      { 7102824, 2 },
      { 7106403, 54 },
      { 7168355, 3 },
      { 7168371, 2 },
      { 7173491, 6 },
      { 7234916, 6 },
      { 7234924, 2 },
      { 7235948, 7 },
      { 7239027, 8 },
      { 7368033, 2 },
      { 7498084, 8 },
      { 7499630, 16 },
      { 7500915, 2 },
      { 7562610, 25 },
      { 7563620, 4 },
      { 7565168, 63 },
      { 7565174, 7 },
      { 7565413, 10 },
      { 7566965, 2 },
      { 7630708, 2 },
      { 7696226, 2 },
      { 7881065, 5 },
      { 7881321, 5 },
      { 7890025, 17 },
      { 7890292, 3 },
      { 7946601, 5 },
      { 7946857, 5 },
      { 7954802, 7 },
      { kMarkvNoneOfTheAbove, 82 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpName, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 4, 3468 },
      { 8, 4386 },
      { 9, 236 },
      { 10, 1316 },
      { 13, 222 },
      { 26, 7154 },
      { 27, 232 },
      { 29, 530 },
      { 30, 292 },
      { 31, 3733 },
      { 32, 386 },
      { 37, 1234 },
      { 40, 28229 },
      { 43, 25682 },
      { 46, 23724 },
      { 48, 336 },
      { 49, 1186 },
      { 66, 938 },
      { 67, 682 },
      { 68, 3023 },
      { 69, 22498 },
      { 71, 1838 },
      { kMarkvNoneOfTheAbove, 6576 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpExtInst, 3), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 0, 6401 },
      { kMarkvNoneOfTheAbove, 321 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpMemoryModel, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 1, 6401 },
      { kMarkvNoneOfTheAbove, 321 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpMemoryModel, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 0, 1973 },
      { 4, 4428 },
      { kMarkvNoneOfTheAbove, 321 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpEntryPoint, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 1, 6401 },
      { 32, 1216 },
      { kMarkvNoneOfTheAbove, 382 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpCapability, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 32, 12783 },
      { kMarkvNoneOfTheAbove, 640 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeInt, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 0, 6397 },
      { 1, 6386 },
      { kMarkvNoneOfTheAbove, 640 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeInt, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 32, 6401 },
      { kMarkvNoneOfTheAbove, 321 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeFloat, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 2, 8009 },
      { 3, 6195 },
      { 4, 9259 },
      { kMarkvNoneOfTheAbove, 1174 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeVector, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 2, 291 },
      { 3, 2633 },
      { 4, 5438 },
      { kMarkvNoneOfTheAbove, 419 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeMatrix, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 1, 6917 },
      { 2, 2731 },
      { 3, 1397 },
      { kMarkvNoneOfTheAbove, 553 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeImage, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 0, 10196 },
      { 1, 849 },
      { kMarkvNoneOfTheAbove, 553 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeImage, 3), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 0, 11022 },
      { 1, 23 },
      { kMarkvNoneOfTheAbove, 553 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeImage, 4), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 0, 11045 },
      { kMarkvNoneOfTheAbove, 553 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeImage, 5), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 1, 11045 },
      { kMarkvNoneOfTheAbove, 553 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeImage, 6), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 0, 11045 },
      { kMarkvNoneOfTheAbove, 553 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeImage, 7), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 0, 13057 },
      { 1, 19201 },
      { 2, 32259 },
      { 3, 12845 },
      { 6, 6983 },
      { 7, 33182 },
      { kMarkvNoneOfTheAbove, 5877 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypePointer, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 0, 18082 },
      { 1, 11827 },
      { 2, 11920 },
      { 3, 11416 },
      { 4, 5939 },
      { 5, 5557 },
      { 6, 4930 },
      { 7, 4431 },
      { 8, 3827 },
      { 9, 4032 },
      { 10, 4897 },
      { 11, 2289 },
      { 12, 3796 },
      { 13, 3003 },
      { 14, 2553 },
      { 15, 2681 },
      { 16, 4447 },
      { 17, 4263 },
      { 18, 3772 },
      { 19, 3862 },
      { 20, 3784 },
      { 21, 3611 },
      { 22, 3448 },
      { 23, 3775 },
      { 24, 2835 },
      { 25, 1260 },
      { 26, 1492 },
      { 27, 1966 },
      { 28, 1691 },
      { 29, 2858 },
      { 30, 2374 },
      { 31, 2308 },
      { 32, 1865 },
      { 33, 1163 },
      { 34, 971 },
      { 35, 874 },
      { 36, 812 },
      { 37, 817 },
      { 38, 810 },
      { 39, 729 },
      { 40, 792 },
      { 41, 863 },
      { 42, 863 },
      { 43, 863 },
      { 44, 792 },
      { 45, 792 },
      { 256, 1483 },
      { 507307272, 768 },
      { 864026611, 857 },
      { 925353388, 284 },
      { 981668463, 1788 },
      { 997553156, 384 },
      { 1014330372, 384 },
      { 1020708227, 384 },
      { 1028443341, 496 },
      { 1032953056, 1056 },
      { 1033463938, 384 },
      { 1033463943, 407 },
      { 1039998884, 384 },
      { 1039998950, 407 },
      { 1040187392, 571 },
      { 1042401985, 1056 },
      { 1044220635, 1056 },
      { 1045622707, 384 },
      { 1045622740, 407 },
      { 1048576000, 953 },
      { 1053609165, 1072 },
      { 1053790359, 1056 },
      { 1054448026, 576 },
      { 1055437881, 2952 },
      { 1056300230, 1056 },
      { 1056964608, 5820 },
      { 1058056805, 1056 },
      { 1059286575, 1056 },
      { 1061158912, 542 },
      { 1061997773, 1302 },
      { 1064514355, 502 },
      { 1064854933, 1056 },
      { 1065353216, 6322 },
      { 1065354055, 426 },
      { 1068289229, 288 },
      { 1069547520, 646 },
      { 1073741824, 5071 },
      { 1077936128, 768 },
      { 1082130432, 1338 },
      { 1083416238, 520 },
      { 1090519040, 332 },
      { 1091567616, 1067 },
      { 1098907648, 515 },
      { 1107296256, 495 },
      { 1115422720, 831 },
      { 1115684864, 411 },
      { 1124073472, 1064 },
      { 1132396544, 1069 },
      { 1132462080, 245 },
      { 1140850688, 563 },
      { 1199562752, 1043 },
      { 3179067684, 1056 },
      { 3180973575, 1056 },
      { 3182651297, 1056 },
      { 3196448879, 1056 },
      { 3204448256, 527 },
      { 3204993516, 1056 },
      { 3205248529, 1056 },
      { 3207137644, 1056 },
      { 3208642560, 529 },
      { 3211081967, 1056 },
      { 3212836864, 2736 },
      { 3215772877, 288 },
      { 3221225472, 297 },
      { 3263168512, 428 },
      { 3271557120, 428 },
      { 3332128768, 1550 },
      { kMarkvNoneOfTheAbove, 11910 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpConstant, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 0, 49767 },
      { kMarkvNoneOfTheAbove, 2489 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFunction, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 0, 49743 },
      { 1, 35713 },
      { 2, 13584 },
      { 3, 21875 },
      { 6, 68876 },
      { 7, 181752 },
      { kMarkvNoneOfTheAbove, 18578 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpVariable, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 2, 26812 },
      { 6, 11485 },
      { 11, 3892 },
      { 18, 327 },
      { 30, 52814 },
      { 33, 27011 },
      { 34, 63327 },
      { kMarkvNoneOfTheAbove, 9284 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpDecorate, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 0, 75837 },
      { 1, 19175 },
      { 2, 9951 },
      { 3, 8398 },
      { 4, 6546 },
      { 5, 6184 },
      { 6, 5310 },
      { 7, 3937 },
      { 8, 2109 },
      { 9, 865 },
      { 10, 630 },
      { 11, 437 },
      { 12, 1026 },
      { 13, 904 },
      { 14, 1096 },
      { 15, 2253 },
      { 16, 6573 },
      { 17, 470 },
      { 18, 838 },
      { 19, 470 },
      { 20, 282 },
      { 64, 4908 },
      { kMarkvNoneOfTheAbove, 7927 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpDecorate, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 0, 36584 },
      { 1, 30174 },
      { 2, 29363 },
      { 3, 29356 },
      { 4, 19578 },
      { 5, 19570 },
      { 6, 19560 },
      { 7, 19551 },
      { 8, 19540 },
      { 9, 19536 },
      { 10, 19535 },
      { 11, 19514 },
      { 12, 19514 },
      { 13, 19506 },
      { 14, 19500 },
      { 15, 19529 },
      { 16, 19460 },
      { 17, 19460 },
      { 18, 19457 },
      { 19, 19457 },
      { 20, 19377 },
      { 21, 14493 },
      { 22, 14013 },
      { 23, 23599 },
      { 24, 13613 },
      { 25, 8740 },
      { 26, 8580 },
      { 27, 8580 },
      { 28, 8100 },
      { 29, 8098 },
      { 30, 8092 },
      { 31, 8086 },
      { 32, 8031 },
      { 33, 7935 },
      { 34, 6567 },
      { 35, 5367 },
      { 36, 4887 },
      { 37, 14655 },
      { 38, 4887 },
      { kMarkvNoneOfTheAbove, 31675 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpMemberDecorate, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 4, 29331 },
      { 7, 29331 },
      { 11, 1672 },
      { 35, 573146 },
      { kMarkvNoneOfTheAbove, 31675 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpMemberDecorate, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 0, 26816 },
      { 3, 816 },
      { 16, 44036 },
      { 28, 8095 },
      { 32, 14697 },
      { 36, 3213 },
      { 40, 8066 },
      { 44, 7936 },
      { 48, 14674 },
      { 60, 9763 },
      { 64, 19519 },
      { 68, 1535 },
      { 72, 1546 },
      { 76, 11291 },
      { 80, 14622 },
      { 84, 3215 },
      { 88, 1687 },
      { 92, 6559 },
      { 96, 14638 },
      { 100, 1685 },
      { 104, 1624 },
      { 108, 6406 },
      { 112, 14616 },
      { 116, 1533 },
      { 120, 6406 },
      { 124, 3211 },
      { 128, 19487 },
      { 132, 3211 },
      { 136, 3211 },
      { 140, 11439 },
      { 144, 14601 },
      { 148, 6553 },
      { 152, 8169 },
      { 156, 11288 },
      { 160, 14619 },
      { 172, 1700 },
      { 176, 14591 },
      { 188, 1534 },
      { 192, 19475 },
      { 200, 1603 },
      { 204, 6569 },
      { 208, 9700 },
      { 220, 1854 },
      { 224, 9708 },
      { 232, 1854 },
      { 236, 2403 },
      { 240, 4405 },
      { 248, 2883 },
      { 256, 12320 },
      { 272, 5607 },
      { 288, 5367 },
      { 292, 5204 },
      { 296, 5367 },
      { 300, 5204 },
      { 304, 5047 },
      { 316, 4884 },
      { 320, 5047 },
      { 332, 5047 },
      { 336, 4888 },
      { 348, 4884 },
      { 352, 4887 },
      { 364, 4884 },
      { 368, 4887 },
      { 372, 4884 },
      { 376, 4884 },
      { 384, 4887 },
      { 392, 4884 },
      { 400, 4888 },
      { 416, 4887 },
      { 424, 4884 },
      { 432, 4887 },
      { 448, 4887 },
      { 460, 4884 },
      { 464, 4887 },
      { 468, 4884 },
      { 472, 4884 },
      { 476, 4884 },
      { 480, 4887 },
      { 488, 4884 },
      { 492, 4884 },
      { 496, 4887 },
      { 512, 4887 },
      { 640, 4884 },
      { kMarkvNoneOfTheAbove, 30208 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpMemberDecorate, 3), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 0, 132059 },
      { 1, 4160 },
      { 2, 18691 },
      { 3, 12782 },
      { 4, 27525 },
      { 5, 235 },
      { kMarkvNoneOfTheAbove, 9773 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpVectorShuffle, 4), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 0, 8100 },
      { 1, 127483 },
      { 2, 4260 },
      { 3, 23168 },
      { 4, 4986 },
      { 5, 27455 },
      { kMarkvNoneOfTheAbove, 9773 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpVectorShuffle, 5), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 0, 5897 },
      { 1, 2257 },
      { 2, 90845 },
      { 3, 5310 },
      { 4, 2501 },
      { 5, 1727 },
      { 6, 21815 },
      { kMarkvNoneOfTheAbove, 6518 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpVectorShuffle, 6), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 0, 2050 },
      { 1, 2395 },
      { 2, 813 },
      { 3, 33496 },
      { 4, 235 },
      { 5, 2727 },
      { 6, 524 },
      { kMarkvNoneOfTheAbove, 2113 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpVectorShuffle, 7), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 0, 48636 },
      { 1, 48680 },
      { 2, 49119 },
      { 3, 30585 },
      { 4, 1496 },
      { 5, 3416 },
      { 6, 1608 },
      { kMarkvNoneOfTheAbove, 9178 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpCompositeExtract, 3), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 0, 3511 },
      { 1, 1431 },
      { 2, 1431 },
      { 3, 312 },
      { kMarkvNoneOfTheAbove, 335 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpCompositeExtract, 4), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 1, 35 },
      { kMarkvNoneOfTheAbove, 2 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpImageSampleImplicitLod, 4), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 2, 9853 },
      { 10, 9230 },
      { kMarkvNoneOfTheAbove, 955 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpImageSampleExplicitLod, 4), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 2, 7119 },
      { kMarkvNoneOfTheAbove, 356 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpImageSampleDrefExplicitLod, 5), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 0, 9148 },
      { kMarkvNoneOfTheAbove, 458 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpSelectionMerge, 1), std::move(codec));
  }

  return codecs;
}

std::map<std::pair<uint32_t, uint32_t>, std::unique_ptr<HuffmanCodec<uint64_t>>>
GetIdDescriptorHuffmanCodecs() {
  std::map<std::pair<uint32_t, uint32_t>, std::unique_ptr<HuffmanCodec<uint64_t>>> codecs;
  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 4228502127, 6401 },
      { kMarkvNoneOfTheAbove, 321 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpExtInstImport, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 51257 },
      { 1951208733, 71472 },
      { 2320303498, 3631 },
      { 3334207724, 5145 },
      { kMarkvNoneOfTheAbove, 6576 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpExtInst, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 25054578, 928 },
      { 295018543, 1512 },
      { 345779370, 672 },
      { 439764402, 688 },
      { 599185303, 1596 },
      { 602134513, 865 },
      { 779021139, 1722 },
      { 1015552308, 1596 },
      { 1236389532, 1596 },
      { 1350769073, 696 },
      { 1401529900, 684 },
      { 1422734249, 840 },
      { 1830851200, 960 },
      { 1834771669, 876 },
      { 1847537868, 768 },
      { 1868255565, 660 },
      { 1906736845, 828 },
      { 1955104493, 1644 },
      { 2032069771, 1019 },
      { 2161102232, 1012 },
      { 2176287978, 931 },
      { 2244470522, 1596 },
      { 2406519603, 696 },
      { 2532518896, 1828 },
      { 2533958702, 771 },
      { 2789375411, 1356 },
      { 2890522199, 876 },
      { 2921996537, 832 },
      { 2962487375, 684 },
      { 3485833663, 856 },
      { 3534235309, 1440 },
      { 3561482820, 1015 },
      { 4018237905, 1656 },
      { 4253534934, 796 },
      { kMarkvNoneOfTheAbove, 6576 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpExtInst, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 4228502127, 131505 },
      { kMarkvNoneOfTheAbove, 6576 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpExtInst, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 139011596, 3239 },
      { 181902171, 1598 },
      { 206688607, 768 },
      { 296981500, 1415 },
      { 810488476, 4543 },
      { 870594305, 4101 },
      { 1015552308, 1596 },
      { 1367301635, 3696 },
      { 1570165302, 883 },
      { 1695321385, 674 },
      { 1742737136, 4878 },
      { 1901166356, 904 },
      { 1980341560, 799 },
      { 2055836767, 780 },
      { 2096388952, 6812 },
      { 2124837447, 1596 },
      { 2176287978, 928 },
      { 2269114589, 931 },
      { 2321729979, 1079 },
      { 2572638469, 2064 },
      { 2614879967, 3644 },
      { 2622612602, 948 },
      { 2639720559, 960 },
      { 2855506940, 5465 },
      { 2919796598, 1356 },
      { 2970183398, 872 },
      { 3044188332, 1012 },
      { 3091876332, 1682 },
      { 3104643263, 959 },
      { 3187066832, 2144 },
      { 3487022798, 4166 },
      { 3714664910, 1019 },
      { 4174489262, 1596 },
      { kMarkvNoneOfTheAbove, 6576 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpExtInst, 4), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 72782198, 576 },
      { 139011596, 1359 },
      { 296981500, 2068 },
      { 333855951, 782 },
      { 505940164, 1020 },
      { 538168945, 936 },
      { 571687023, 780 },
      { 603915804, 936 },
      { 790502615, 1015 },
      { 810488476, 929 },
      { 823178554, 674 },
      { 1302662663, 480 },
      { 1367301635, 3714 },
      { 1378082995, 1020 },
      { 1410311776, 936 },
      { 1443829854, 1068 },
      { 1476058635, 696 },
      { 1684282922, 768 },
      { 1726093781, 672 },
      { 1847537868, 528 },
      { 1901166356, 1009 },
      { 2096388952, 2136 },
      { 2162986400, 535 },
      { 2250225826, 1020 },
      { 2282454607, 936 },
      { 2348201466, 1080 },
      { 2597020383, 1824 },
      { 2630465393, 672 },
      { 2817335337, 856 },
      { 2855506940, 23318 },
      { 3122368657, 1194 },
      { 3154597438, 936 },
      { 3187066832, 1202 },
      { 3233393284, 1109 },
      { 3244209297, 674 },
      { 3251128023, 3889 },
      { 3496407048, 828 },
      { 3534235309, 504 },
      { 3536471583, 768 },
      { 3692647551, 567 },
      { 3928764629, 1020 },
      { 3994511488, 1020 },
      { 4018237905, 636 },
      { 4026740269, 1068 },
      { kMarkvNoneOfTheAbove, 4655 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpExtInst, 5), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 30614927, 320 },
      { 114047017, 260 },
      { 139011596, 268 },
      { 296981500, 21976 },
      { 429277936, 432 },
      { 508217552, 2063 },
      { 759179275, 384 },
      { 810488476, 355 },
      { 870594305, 312 },
      { 955292137, 464 },
      { 1022915255, 695 },
      { 1209418480, 388 },
      { 1279584219, 408 },
      { 1287937401, 528 },
      { 1319785741, 327 },
      { 1541020250, 431 },
      { 1684282922, 637 },
      { 1901166356, 820 },
      { 2004567202, 312 },
      { 2119793999, 537 },
      { 2197882533, 320 },
      { 2407299925, 328 },
      { 2538917932, 600 },
      { 2683080096, 535 },
      { 2683267775, 288 },
      { 2854085372, 528 },
      { 2861331635, 312 },
      { 3547456240, 891 },
      { 3753486980, 1104 },
      { 3782778655, 336 },
      { 3990925720, 600 },
      { 4170820740, 260 },
      { kMarkvNoneOfTheAbove, 2530 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpExtInst, 6), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 3056042030, 6401 },
      { kMarkvNoneOfTheAbove, 321 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeVoid, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 789872778, 6386 },
      { 2430404313, 6397 },
      { kMarkvNoneOfTheAbove, 640 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeInt, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 1951208733, 6401 },
      { kMarkvNoneOfTheAbove, 321 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeFloat, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 166253838, 831 },
      { 679771963, 6193 },
      { 1247793383, 1632 },
      { 2261697609, 729 },
      { 2263349224, 1298 },
      { 2320303498, 6376 },
      { 3334207724, 6401 },
      { kMarkvNoneOfTheAbove, 1174 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeVector, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 789872778, 2362 },
      { 1415510495, 833 },
      { 1951208733, 18970 },
      { 2430404313, 1298 },
      { kMarkvNoneOfTheAbove, 1174 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeVector, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 1146476634, 286 },
      { 2160380860, 5438 },
      { 3800912395, 1334 },
      { 3802564010, 1299 },
      { kMarkvNoneOfTheAbove, 419 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeMatrix, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 1299 },
      { 3334207724, 7058 },
      { kMarkvNoneOfTheAbove, 419 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeMatrix, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 1343794461, 3603 },
      { 2984325996, 6045 },
      { 3998230222, 1397 },
      { kMarkvNoneOfTheAbove, 553 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeImage, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 1951208733, 11045 },
      { kMarkvNoneOfTheAbove, 553 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeImage, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 3866587616, 1397 },
      { 3868239231, 3603 },
      { 3869890846, 6045 },
      { kMarkvNoneOfTheAbove, 553 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeSampledImage, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 1343794461, 3603 },
      { 2984325996, 6045 },
      { 3998230222, 1397 },
      { kMarkvNoneOfTheAbove, 553 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeSampledImage, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 1389644742, 4884 },
      { 3232633974, 816 },
      { 3278176820, 1481 },
      { 3648138580, 164 },
      { 3697687030, 4897 },
      { kMarkvNoneOfTheAbove, 617 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeArray, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 1951208733, 836 },
      { 2160380860, 4908 },
      { 3334207724, 6553 },
      { kMarkvNoneOfTheAbove, 617 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeArray, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 144116905, 816 },
      { 827246872, 187 },
      { 2715370488, 1481 },
      { 2798552666, 9783 },
      { kMarkvNoneOfTheAbove, 617 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeArray, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 40653745, 4925 },
      { 119981689, 1481 },
      { 153085016, 320 },
      { 545678922, 772 },
      { 899570100, 360 },
      { 929101967, 414 },
      { 1070791291, 4884 },
      { 1100599986, 1012 },
      { 1103903216, 648 },
      { 1154919607, 4884 },
      { 1258105452, 320 },
      { 1369578001, 528 },
      { 1674803691, 1368 },
      { 1989520052, 811 },
      { 2924146124, 4873 },
      { 3345288309, 260 },
      { 3357250579, 280 },
      { 3410158390, 1200 },
      { 3489360962, 768 },
      { 3495967422, 270 },
      { 3504981554, 480 },
      { 3705139860, 483 },
      { 3822983876, 280 },
      { 4141567741, 312 },
      { 4240893633, 338 },
      { kMarkvNoneOfTheAbove, 1732 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 5696 },
      { 1951208733, 3540 },
      { 2160380860, 4884 },
      { 3278176820, 1481 },
      { 3334207724, 19022 },
      { kMarkvNoneOfTheAbove, 1732 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 8918 },
      { 1951208733, 1978 },
      { 2160380860, 4884 },
      { 2320303498, 4235 },
      { 3232633974, 811 },
      { 3334207724, 7392 },
      { kMarkvNoneOfTheAbove, 1411 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 4267 },
      { 1951208733, 8413 },
      { 2160380860, 4887 },
      { 2320303498, 4885 },
      { 3334207724, 4175 },
      { kMarkvNoneOfTheAbove, 1332 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 3), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 6501 },
      { 1951208733, 3526 },
      { 2160380860, 4884 },
      { 2320303498, 5702 },
      { 3334207724, 5509 },
      { kMarkvNoneOfTheAbove, 1308 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 4), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 6489 },
      { 1951208733, 8135 },
      { 2263349224, 640 },
      { 2320303498, 199 },
      { 3334207724, 9983 },
      { kMarkvNoneOfTheAbove, 1273 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 5), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 4876 },
      { 1951208733, 8117 },
      { 2320303498, 313 },
      { 3334207724, 11970 },
      { kMarkvNoneOfTheAbove, 1265 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 6), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 4888 },
      { 1951208733, 12934 },
      { 2320303498, 2838 },
      { 3334207724, 4183 },
      { kMarkvNoneOfTheAbove, 1244 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 7), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 4878 },
      { 1951208733, 13392 },
      { 2320303498, 783 },
      { 3334207724, 4943 },
      { kMarkvNoneOfTheAbove, 1200 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 8), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 1951208733, 11875 },
      { 2320303498, 3159 },
      { 3334207724, 7615 },
      { kMarkvNoneOfTheAbove, 1135 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 9), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 4876 },
      { 1951208733, 8080 },
      { 2320303498, 122 },
      { 3334207724, 7786 },
      { kMarkvNoneOfTheAbove, 1045 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 10), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 4889 },
      { 1951208733, 8366 },
      { 2320303498, 1603 },
      { 3334207724, 4945 },
      { kMarkvNoneOfTheAbove, 992 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 11), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 9766 },
      { 1951208733, 9696 },
      { kMarkvNoneOfTheAbove, 976 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 12), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 4940 },
      { 1951208733, 12938 },
      { 2320303498, 1603 },
      { kMarkvNoneOfTheAbove, 976 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 13), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 5003 },
      { 1951208733, 8001 },
      { 2320303498, 4876 },
      { 3334207724, 1600 },
      { kMarkvNoneOfTheAbove, 976 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 14), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 6252 },
      { 1951208733, 6752 },
      { 2320303498, 6473 },
      { kMarkvNoneOfTheAbove, 976 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 15), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 10957 },
      { 1951208733, 8103 },
      { 2320303498, 400 },
      { kMarkvNoneOfTheAbove, 975 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 16), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 5684 },
      { 1951208733, 7769 },
      { 2320303498, 1123 },
      { 3334207724, 4884 },
      { kMarkvNoneOfTheAbove, 974 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 17), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 5044 },
      { 1951208733, 13533 },
      { 3334207724, 880 },
      { kMarkvNoneOfTheAbove, 974 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 18), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 240 },
      { 1951208733, 18337 },
      { 2320303498, 880 },
      { kMarkvNoneOfTheAbove, 973 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 19), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 4886 },
      { 1951208733, 14171 },
      { 2320303498, 400 },
      { kMarkvNoneOfTheAbove, 973 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 20), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 326 },
      { 1951208733, 8723 },
      { 2320303498, 10328 },
      { kMarkvNoneOfTheAbove, 969 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 21), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 1951208733, 4522 },
      { 2320303498, 4884 },
      { 3334207724, 5033 },
      { kMarkvNoneOfTheAbove, 725 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 22), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 505 },
      { 1951208733, 3591 },
      { 2320303498, 5044 },
      { 3334207724, 4873 },
      { kMarkvNoneOfTheAbove, 701 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 23), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 1374 },
      { 1951208733, 2562 },
      { 2160380860, 4873 },
      { 2320303498, 160 },
      { 3334207724, 4884 },
      { kMarkvNoneOfTheAbove, 693 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 24), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 1254 },
      { 1951208733, 2442 },
      { 2320303498, 5044 },
      { 3334207724, 4873 },
      { kMarkvNoneOfTheAbove, 681 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 25), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 585 },
      { 1951208733, 3268 },
      { 2320303498, 4887 },
      { kMarkvNoneOfTheAbove, 438 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 26), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 1534 },
      { 1951208733, 2159 },
      { 2320303498, 4887 },
      { kMarkvNoneOfTheAbove, 430 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 27), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 6138 },
      { 1951208733, 2440 },
      { kMarkvNoneOfTheAbove, 430 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 28), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 583 },
      { 1951208733, 7509 },
      { kMarkvNoneOfTheAbove, 406 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 29), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 1368 },
      { 1951208733, 6715 },
      { kMarkvNoneOfTheAbove, 405 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 30), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 1200 },
      { 1951208733, 6828 },
      { 2320303498, 64 },
      { kMarkvNoneOfTheAbove, 405 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 31), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 480 },
      { 1951208733, 7455 },
      { 2320303498, 151 },
      { kMarkvNoneOfTheAbove, 405 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 32), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 1951208733, 6567 },
      { 2320303498, 1464 },
      { kMarkvNoneOfTheAbove, 402 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 33), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 1951208733, 480 },
      { 2320303498, 7455 },
      { kMarkvNoneOfTheAbove, 397 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 34), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 1951208733, 4884 },
      { 2320303498, 1683 },
      { kMarkvNoneOfTheAbove, 329 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 35), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 1951208733, 4884 },
      { 2320303498, 480 },
      { kMarkvNoneOfTheAbove, 269 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 36), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 4887 },
      { kMarkvNoneOfTheAbove, 245 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 37), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 1389644742, 4884 },
      { kMarkvNoneOfTheAbove, 245 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 38), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 3697687030, 4884 },
      { kMarkvNoneOfTheAbove, 245 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 39), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 2320303498, 3 },
      { kMarkvNoneOfTheAbove, 1 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 40), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 2320303498, 3 },
      { kMarkvNoneOfTheAbove, 1 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 41), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 3 },
      { kMarkvNoneOfTheAbove, 1 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 42), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 3 },
      { kMarkvNoneOfTheAbove, 1 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 43), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 3 },
      { kMarkvNoneOfTheAbove, 1 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 44), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 3 },
      { kMarkvNoneOfTheAbove, 1 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 45), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 3 },
      { kMarkvNoneOfTheAbove, 1 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 46), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 3 },
      { kMarkvNoneOfTheAbove, 1 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 47), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 3 },
      { kMarkvNoneOfTheAbove, 1 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 48), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 3 },
      { kMarkvNoneOfTheAbove, 1 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 49), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 3 },
      { kMarkvNoneOfTheAbove, 1 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 50), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 3 },
      { kMarkvNoneOfTheAbove, 1 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeStruct, 51), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 135486769, 1366 },
      { 436416061, 1012 },
      { 440421571, 4893 },
      { 450406196, 821 },
      { 503094540, 1200 },
      { 543621065, 2396 },
      { 827698488, 6315 },
      { 907126242, 4335 },
      { 908777857, 4804 },
      { 910429472, 3182 },
      { 1195077267, 640 },
      { 1294403159, 6212 },
      { 1296054774, 5978 },
      { 1297706389, 5520 },
      { 1322549027, 1863 },
      { 1784441183, 1476 },
      { 2080953106, 1564 },
      { 2194691858, 3649 },
      { 2448331885, 1298 },
      { 2466255445, 768 },
      { 2468230023, 1389 },
      { 2547657777, 5278 },
      { 2549309392, 4383 },
      { 2550961007, 4146 },
      { 2894051250, 1184 },
      { 2934934694, 1399 },
      { 2936586309, 1248 },
      { 2938237924, 1310 },
      { 3094180193, 1356 },
      { 3094857332, 1341 },
      { 3095831808, 2396 },
      { 3183924418, 3057 },
      { 3282979782, 871 },
      { 3561562003, 6401 },
      { 3563213618, 1833 },
      { 3564865233, 1557 },
      { 4028622909, 4188 },
      { 4039938779, 811 },
      { 4050155669, 812 },
      { kMarkvNoneOfTheAbove, 5877 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypePointer, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 119981689, 1476 },
      { 162255877, 2396 },
      { 545678922, 772 },
      { 679771963, 18246 },
      { 789872778, 814 },
      { 1100599986, 1012 },
      { 1103903216, 772 },
      { 1154919607, 4188 },
      { 1343794461, 1356 },
      { 1674803691, 1366 },
      { 1951208733, 15502 },
      { 1989520052, 811 },
      { 2160380860, 1738 },
      { 2263349224, 1938 },
      { 2320303498, 15715 },
      { 2924146124, 3057 },
      { 2984325996, 2396 },
      { 3334207724, 23625 },
      { 3410158390, 1200 },
      { 3489360962, 852 },
      { 3800912395, 1474 },
      { 3802564010, 955 },
      { 3866587616, 821 },
      { 3868239231, 1863 },
      { 3869890846, 3649 },
      { 4240893633, 676 },
      { kMarkvNoneOfTheAbove, 5877 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypePointer, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 75986790, 6401 },
      { 95470391, 768 },
      { 170378107, 304 },
      { 172029722, 679 },
      { 204234270, 368 },
      { 205885885, 312 },
      { 244668133, 768 },
      { 265778447, 360 },
      { 753954113, 267 },
      { 1000070091, 800 },
      { 1671139745, 338 },
      { 1774874546, 240 },
      { 1776526161, 308 },
      { 1887808856, 866 },
      { 1889460471, 1596 },
      { 1917966999, 530 },
      { 2044728014, 528 },
      { 2192810893, 312 },
      { 2293247016, 779 },
      { 2503194620, 807 },
      { 2608484640, 768 },
      { 2615111110, 445 },
      { 2668769415, 768 },
      { 2759951687, 1866 },
      { 2761603302, 2564 },
      { 2856623532, 280 },
      { 2956189845, 1596 },
      { 3447882276, 400 },
      { 3633746133, 1604 },
      { 3635397748, 1603 },
      { 3710645347, 222 },
      { 3712296962, 536 },
      { 3727494858, 812 },
      { 3747079365, 796 },
      { 3748965853, 225 },
      { 3750617468, 269 },
      { 4024173916, 324 },
      { 4215670524, 789 },
      { 4217322139, 985 },
      { kMarkvNoneOfTheAbove, 2049 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeFunction, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 545678922, 768 },
      { 679771963, 15639 },
      { 929101967, 414 },
      { 1951208733, 2890 },
      { 2320303498, 4116 },
      { 3056042030, 9500 },
      { 3334207724, 4994 },
      { 3357250579, 280 },
      { 3705139860, 483 },
      { 3800912395, 871 },
      { 3802564010, 211 },
      { kMarkvNoneOfTheAbove, 2049 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeFunction, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 436416061, 324 },
      { 679771963, 1576 },
      { 815757910, 312 },
      { 827698488, 1674 },
      { 1294403159, 4840 },
      { 1296054774, 10733 },
      { 1297706389, 5622 },
      { 1525861001, 190 },
      { 1579585816, 312 },
      { 1675764636, 338 },
      { 1824016656, 260 },
      { 2194691858, 707 },
      { 2320303498, 1315 },
      { 2881886868, 224 },
      { 2926633629, 672 },
      { 3249265647, 312 },
      { 3334207724, 1394 },
      { 3674863070, 224 },
      { 4050155669, 241 },
      { 4141567741, 936 },
      { 4155122613, 196 },
      { kMarkvNoneOfTheAbove, 1687 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeFunction, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 892 },
      { 827698488, 2209 },
      { 1294403159, 1252 },
      { 1296054774, 6035 },
      { 1297706389, 810 },
      { 1951208733, 603 },
      { 2320303498, 1632 },
      { 3334207724, 480 },
      { kMarkvNoneOfTheAbove, 697 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeFunction, 3), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 657 },
      { 827698488, 1143 },
      { 1294403159, 688 },
      { 1296054774, 5321 },
      { 1951208733, 58 },
      { 2194691858, 156 },
      { 2320303498, 555 },
      { 3334207724, 86 },
      { kMarkvNoneOfTheAbove, 434 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeFunction, 4), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 827698488, 1994 },
      { 1294403159, 800 },
      { 1296054774, 2978 },
      { 1297706389, 73 },
      { 1951208733, 264 },
      { kMarkvNoneOfTheAbove, 307 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeFunction, 5), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 827698488, 1584 },
      { 1294403159, 72 },
      { 1296054774, 792 },
      { 1951208733, 141 },
      { 3334207724, 159 },
      { kMarkvNoneOfTheAbove, 138 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeFunction, 6), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 827698488, 1051 },
      { 1951208733, 209 },
      { 2320303498, 156 },
      { kMarkvNoneOfTheAbove, 72 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeFunction, 7), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 543621065, 280 },
      { 827698488, 769 },
      { 1951208733, 72 },
      { kMarkvNoneOfTheAbove, 57 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeFunction, 8), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 827698488, 769 },
      { 1951208733, 70 },
      { 3095831808, 280 },
      { kMarkvNoneOfTheAbove, 57 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeFunction, 9), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 1296054774, 768 },
      { kMarkvNoneOfTheAbove, 39 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeFunction, 10), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 1296054774, 768 },
      { kMarkvNoneOfTheAbove, 39 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeFunction, 11), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 789872778, 2 },
      { 1951208733, 1 },
      { kMarkvNoneOfTheAbove, 1 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeFunction, 12), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 789872778, 1 },
      { 1951208733, 1 },
      { kMarkvNoneOfTheAbove, 1 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeFunction, 13), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 1951208733, 1 },
      { kMarkvNoneOfTheAbove, 1 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeFunction, 14), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 1951208733, 1 },
      { kMarkvNoneOfTheAbove, 1 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpTypeFunction, 15), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 789872778, 131377 },
      { 1951208733, 80649 },
      { 2430404313, 26170 },
      { kMarkvNoneOfTheAbove, 11910 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpConstant, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 51041423, 3448 },
      { 142465290, 5528 },
      { 144116905, 6299 },
      { 210116709, 2952 },
      { 290391815, 2553 },
      { 296981500, 6322 },
      { 438318340, 1966 },
      { 529742207, 4930 },
      { 677668732, 3862 },
      { 825595257, 1839 },
      { 917019124, 2289 },
      { 959681532, 5820 },
      { 1064945649, 2835 },
      { 1092948665, 1302 },
      { 1156369516, 5573 },
      { 1158021131, 5843 },
      { 1304296041, 4447 },
      { 1452222566, 2858 },
      { 1543646433, 3827 },
      { 1691572958, 3611 },
      { 1782996825, 5546 },
      { 1784648440, 6297 },
      { 1930923350, 3002 },
      { 2078849875, 1492 },
      { 2170273742, 5556 },
      { 2318200267, 3772 },
      { 2321729979, 1788 },
      { 2444465148, 2736 },
      { 2466126792, 2308 },
      { 2524697596, 1550 },
      { 2557550659, 4897 },
      { 2705477184, 3775 },
      { 2715370488, 1481 },
      { 2796901051, 5729 },
      { 2798552666, 6191 },
      { 2855506940, 6239 },
      { 2944827576, 2681 },
      { 3092754101, 1691 },
      { 3184177968, 4428 },
      { 3266548732, 1338 },
      { 3332104493, 3784 },
      { 3571454885, 3796 },
      { 3719381410, 1260 },
      { 3810805277, 5917 },
      { 3929248764, 5071 },
      { 3958731802, 4263 },
      { 4106658327, 2374 },
      { 4198082194, 4030 },
      { kMarkvNoneOfTheAbove, 11910 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpConstant, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 18068 },
      { 1247793383, 6298 },
      { 2320303498, 14361 },
      { 3334207724, 14469 },
      { kMarkvNoneOfTheAbove, 2661 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpConstantComposite, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 15502752, 1215 },
      { 46736908, 284 },
      { 139011596, 4612 },
      { 149720480, 280 },
      { 249378857, 1370 },
      { 251209228, 529 },
      { 503145996, 782 },
      { 836581417, 281 },
      { 882718761, 560 },
      { 1289566249, 293 },
      { 1325348861, 1056 },
      { 1558001705, 828 },
      { 1646147798, 1056 },
      { 1679946323, 1056 },
      { 1766401548, 1560 },
      { 1992893964, 953 },
      { 2123388694, 1056 },
      { 2162986400, 771 },
      { 2580096524, 873 },
      { 2598189097, 828 },
      { 2683080096, 2603 },
      { 2698156268, 1056 },
      { 2763960513, 407 },
      { 3015046341, 407 },
      { 3133016299, 1056 },
      { 3251128023, 2952 },
      { 3504158761, 3903 },
      { 3533637837, 280 },
      { 3535289452, 564 },
      { 3536941067, 564 },
      { 3538592682, 1628 },
      { 3540244297, 1631 },
      { 3541895912, 1631 },
      { 3570219049, 828 },
      { 3653838348, 2070 },
      { 3764205609, 2982 },
      { 3882634684, 1056 },
      { 3913885196, 2575 },
      { 3982047273, 428 },
      { 4024252457, 801 },
      { 4243119782, 1056 },
      { 4255182614, 384 },
      { kMarkvNoneOfTheAbove, 2661 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpConstantComposite, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 142465290, 1631 },
      { 158160339, 1056 },
      { 169135842, 1056 },
      { 210116709, 2952 },
      { 296981500, 7913 },
      { 529742207, 280 },
      { 615748604, 408 },
      { 910398460, 535 },
      { 959681532, 1327 },
      { 1039111164, 830 },
      { 1087394637, 1056 },
      { 1156369516, 1628 },
      { 1450415100, 289 },
      { 1543672828, 428 },
      { 2100532220, 428 },
      { 2170273742, 564 },
      { 2285081596, 560 },
      { 2326636627, 407 },
      { 2444465148, 1887 },
      { 2732195517, 1056 },
      { 2763232252, 428 },
      { 2796901051, 1631 },
      { 2855506940, 14214 },
      { 2922615804, 280 },
      { 2937761472, 384 },
      { 3202349435, 1056 },
      { 3362723943, 1056 },
      { 3712763835, 1056 },
      { 3810805277, 564 },
      { 3929248764, 2673 },
      { 4016096296, 1056 },
      { 4172568578, 407 },
      { 4248015868, 687 },
      { kMarkvNoneOfTheAbove, 2661 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpConstantComposite, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 52882140, 1056 },
      { 210116709, 2952 },
      { 296981500, 7857 },
      { 385229009, 1056 },
      { 615748604, 407 },
      { 910398460, 531 },
      { 959681532, 1306 },
      { 1031290113, 1056 },
      { 1039111164, 830 },
      { 1172110445, 1056 },
      { 1450415100, 289 },
      { 1543672828, 428 },
      { 1622381564, 593 },
      { 1782996825, 6298 },
      { 1971252067, 1056 },
      { 2100532220, 428 },
      { 2268204687, 384 },
      { 2326636627, 407 },
      { 2444465148, 1862 },
      { 2490492987, 1056 },
      { 2678954464, 1056 },
      { 2763232252, 428 },
      { 2855506940, 14149 },
      { 2922615804, 280 },
      { 3912967080, 1056 },
      { 3929248764, 2721 },
      { 4172568578, 407 },
      { 4248015868, 691 },
      { kMarkvNoneOfTheAbove, 2661 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpConstantComposite, 3), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 210116709, 2952 },
      { 296981500, 10208 },
      { 615748604, 400 },
      { 910398460, 529 },
      { 959681532, 1249 },
      { 1039111164, 829 },
      { 1092948665, 235 },
      { 1450415100, 286 },
      { 1543672828, 428 },
      { 1612361408, 384 },
      { 2100532220, 428 },
      { 2326636627, 407 },
      { 2444465148, 2112 },
      { 2524697596, 782 },
      { 2763232252, 428 },
      { 2855506940, 7695 },
      { 3929248764, 1455 },
      { 4172568578, 407 },
      { 4248015868, 684 },
      { kMarkvNoneOfTheAbove, 1628 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpConstantComposite, 4), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 296981500, 2428 },
      { 615748604, 400 },
      { 959681532, 287 },
      { 1039111164, 828 },
      { 1450415100, 282 },
      { 1543672828, 428 },
      { 1939359710, 384 },
      { 2100532220, 428 },
      { 2113115132, 563 },
      { 2326636627, 407 },
      { 2444465148, 346 },
      { 2763232252, 428 },
      { 2855506940, 5746 },
      { 3929248764, 592 },
      { 4172568578, 407 },
      { 4248015868, 400 },
      { kMarkvNoneOfTheAbove, 724 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpConstantComposite, 5), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 545678922, 768 },
      { 679771963, 21369 },
      { 929101967, 414 },
      { 1951208733, 4389 },
      { 2320303498, 4649 },
      { 3056042030, 9553 },
      { 3334207724, 5962 },
      { 3357250579, 280 },
      { 3705139860, 483 },
      { 3800912395, 871 },
      { kMarkvNoneOfTheAbove, 2489 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFunction, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 35240468, 360 },
      { 123060826, 400 },
      { 184634770, 312 },
      { 359054425, 445 },
      { 459968607, 280 },
      { 945128292, 267 },
      { 1058429216, 312 },
      { 1182296898, 528 },
      { 1238120570, 768 },
      { 1429389803, 1866 },
      { 1717510093, 768 },
      { 1766422419, 768 },
      { 1775308984, 1604 },
      { 1776629361, 269 },
      { 1824526196, 536 },
      { 1957265068, 1834 },
      { 1998433745, 800 },
      { 2055664760, 3192 },
      { 2303184249, 5830 },
      { 2451531615, 816 },
      { 2507457870, 807 },
      { 2550501832, 768 },
      { 2649103430, 2132 },
      { 2831059514, 1596 },
      { 3269075805, 2239 },
      { 3323202731, 796 },
      { 3361419439, 1277 },
      { 3464197236, 308 },
      { 3472029049, 530 },
      { 3518630848, 312 },
      { 3604842236, 6402 },
      { 3653985133, 921 },
      { 4121643374, 324 },
      { 4185590212, 338 },
      { 4233562270, 841 },
      { 4235213885, 264 },
      { kMarkvNoneOfTheAbove, 2489 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFunction, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 75986790, 6402 },
      { 95470391, 768 },
      { 170378107, 304 },
      { 172029722, 921 },
      { 204234270, 368 },
      { 205885885, 312 },
      { 244668133, 768 },
      { 265778447, 360 },
      { 753954113, 267 },
      { 1000070091, 800 },
      { 1671139745, 338 },
      { 1776526161, 308 },
      { 1887808856, 1834 },
      { 1889460471, 1596 },
      { 1917966999, 530 },
      { 2044728014, 528 },
      { 2192810893, 312 },
      { 2293247016, 2239 },
      { 2503194620, 807 },
      { 2608484640, 768 },
      { 2615111110, 445 },
      { 2668769415, 768 },
      { 2759951687, 1866 },
      { 2761603302, 5830 },
      { 2856623532, 280 },
      { 2956189845, 3192 },
      { 3447882276, 400 },
      { 3633746133, 1604 },
      { 3635397748, 2132 },
      { 3712296962, 536 },
      { 3727494858, 816 },
      { 3747079365, 796 },
      { 3750617468, 269 },
      { 4024173916, 324 },
      { 4215670524, 1277 },
      { 4217322139, 1105 },
      { kMarkvNoneOfTheAbove, 2489 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFunction, 3), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 3251 },
      { 827698488, 11319 },
      { 1294403159, 8654 },
      { 1296054774, 39042 },
      { 1297706389, 7083 },
      { 1951208733, 1554 },
      { 2194691858, 863 },
      { 2320303498, 3673 },
      { 2926633629, 672 },
      { 3334207724, 2614 },
      { 4141567741, 936 },
      { kMarkvNoneOfTheAbove, 4222 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFunctionParameter, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 522971108, 39042 },
      { 1395113939, 7083 },
      { 1510333659, 1554 },
      { 1957218950, 11319 },
      { 1977038330, 2614 },
      { 1978689945, 3251 },
      { 1980341560, 3673 },
      { 2262220987, 936 },
      { 2674422363, 672 },
      { 3465954368, 863 },
      { 3945795573, 8654 },
      { kMarkvNoneOfTheAbove, 4222 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFunctionParameter, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 545678922, 768 },
      { 679771963, 22667 },
      { 929101967, 414 },
      { 1951208733, 4839 },
      { 2320303498, 5238 },
      { 3056042030, 3521 },
      { 3334207724, 6505 },
      { 3357250579, 280 },
      { 3705139860, 483 },
      { 3800912395, 883 },
      { kMarkvNoneOfTheAbove, 2335 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFunctionCall, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 57149555, 549 },
      { 86116519, 768 },
      { 168339452, 240 },
      { 181902171, 5622 },
      { 284226441, 240 },
      { 314809953, 470 },
      { 330249537, 2312 },
      { 545363837, 800 },
      { 707478563, 240 },
      { 740921498, 536 },
      { 807276090, 604 },
      { 824323032, 471 },
      { 1162127370, 927 },
      { 1245448751, 768 },
      { 1375043498, 312 },
      { 1603937321, 1603 },
      { 1708264968, 528 },
      { 1717555224, 400 },
      { 1838993983, 242 },
      { 1949856502, 2133 },
      { 2108571893, 235 },
      { 2110223508, 255 },
      { 2293637521, 796 },
      { 2378763734, 560 },
      { 2516325050, 280 },
      { 2645135839, 339 },
      { 2708915136, 1769 },
      { 2894979602, 768 },
      { 2903897222, 344 },
      { 3054834317, 1596 },
      { 3266028549, 796 },
      { 3299488628, 499 },
      { 3345707173, 421 },
      { 3536390697, 1098 },
      { 3647606635, 768 },
      { 3760372982, 234 },
      { 3823959661, 312 },
      { 3839389658, 324 },
      { 4130950286, 1632 },
      { 4169878842, 312 },
      { 4174489262, 3192 },
      { 4237497041, 308 },
      { kMarkvNoneOfTheAbove, 2335 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFunctionCall, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 35240468, 344 },
      { 123060826, 400 },
      { 184634770, 312 },
      { 459968607, 280 },
      { 619875033, 198 },
      { 1058429216, 312 },
      { 1182296898, 528 },
      { 1238120570, 768 },
      { 1429389803, 1632 },
      { 1717510093, 768 },
      { 1766422419, 768 },
      { 1775308984, 1603 },
      { 1917336504, 1098 },
      { 1957265068, 1769 },
      { 1998433745, 800 },
      { 2055664760, 3192 },
      { 2303184249, 5622 },
      { 2451531615, 421 },
      { 2507457870, 796 },
      { 2550501832, 768 },
      { 2590402790, 224 },
      { 2649103430, 2133 },
      { 2831059514, 1596 },
      { 3269075805, 2312 },
      { 3361419439, 658 },
      { 3472029049, 549 },
      { 3518630848, 312 },
      { 3653985133, 927 },
      { 4185590212, 339 },
      { kMarkvNoneOfTheAbove, 1785 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFunctionCall, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 37459569, 714 },
      { 162167595, 240 },
      { 701281393, 240 },
      { 1630583316, 672 },
      { 1913735398, 260 },
      { 1918481917, 6069 },
      { 1955871800, 312 },
      { 1977038330, 278 },
      { 2053214130, 338 },
      { 2443959748, 796 },
      { 2564745684, 240 },
      { 2622612602, 628 },
      { 2677252364, 308 },
      { 2736026107, 234 },
      { 2790624748, 17566 },
      { 2882994691, 324 },
      { 2970183398, 703 },
      { 3253403867, 1246 },
      { 3570411982, 817 },
      { 3662767579, 6992 },
      { 3884846406, 312 },
      { 3910458990, 312 },
      { 3927915220, 936 },
      { 4224872590, 1985 },
      { kMarkvNoneOfTheAbove, 2292 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFunctionCall, 3), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 744062262, 168 },
      { 1280126114, 174 },
      { 1367301635, 726 },
      { 1684282922, 104 },
      { 1918481917, 1536 },
      { 1978689945, 271 },
      { 1980341560, 798 },
      { 2443959748, 811 },
      { 2629265310, 104 },
      { 2790624748, 7942 },
      { 2970183398, 120 },
      { 3044188332, 168 },
      { 3662767579, 866 },
      { 3887377256, 499 },
      { 3971481069, 88 },
      { 4224872590, 2344 },
      { kMarkvNoneOfTheAbove, 863 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFunctionCall, 4), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 37459569, 156 },
      { 837715723, 120 },
      { 1352628475, 140 },
      { 1918481917, 889 },
      { 1978689945, 388 },
      { 1980341560, 171 },
      { 2096388952, 71 },
      { 2622612602, 83 },
      { 2790624748, 6955 },
      { 2970183398, 814 },
      { 3783543823, 120 },
      { 4224872590, 1235 },
      { kMarkvNoneOfTheAbove, 570 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFunctionCall, 5), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 1510333659, 69 },
      { 1684282922, 184 },
      { 1918481917, 800 },
      { 2790624748, 4578 },
      { 3662767579, 130 },
      { 4224872590, 2033 },
      { kMarkvNoneOfTheAbove, 403 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFunctionCall, 6), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 161668409, 19 },
      { 976111724, 64 },
      { 1510333659, 69 },
      { 1918481917, 73 },
      { 2790624748, 796 },
      { 3525913657, 16 },
      { 3570411982, 156 },
      { 4224872590, 1601 },
      { kMarkvNoneOfTheAbove, 151 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFunctionCall, 7), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 226836633, 10 },
      { 296981500, 12 },
      { 1510333659, 138 },
      { 1684282922, 64 },
      { 3495546641, 156 },
      { 3564402361, 9 },
      { 4224872590, 1052 },
      { kMarkvNoneOfTheAbove, 79 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFunctionCall, 8), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 161668409, 15 },
      { 188347929, 10 },
      { 215027449, 7 },
      { 296981500, 7 },
      { 653708953, 10 },
      { 1510333659, 69 },
      { 1584774136, 280 },
      { 2444465148, 7 },
      { 3033873113, 10 },
      { 3499234137, 8 },
      { 3525913657, 11 },
      { 3552593177, 7 },
      { 4224872590, 769 },
      { kMarkvNoneOfTheAbove, 64 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFunctionCall, 9), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 825595257, 37 },
      { 1510333659, 69 },
      { 2096388952, 10 },
      { 2713718873, 7 },
      { 4064212479, 280 },
      { 4163160985, 17 },
      { 4224872590, 770 },
      { kMarkvNoneOfTheAbove, 64 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFunctionCall, 10), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 123108003, 7 },
      { 296981500, 19 },
      { 595410904, 9 },
      { 1466938734, 11 },
      { 1503477720, 6 },
      { 1816558243, 11 },
      { 1990431740, 13 },
      { 2724625059, 7 },
      { 2790624748, 768 },
      { 2812498065, 22 },
      { 3160388974, 8 },
      { 3745223676, 8 },
      { kMarkvNoneOfTheAbove, 46 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFunctionCall, 11), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 1054641568, 18 },
      { 1269075360, 15 },
      { 1675922848, 8 },
      { 2038205856, 22 },
      { 2433519008, 5 },
      { 2790624748, 768 },
      { 2840366496, 32 },
      { 2851900832, 6 },
      { 2964622752, 12 },
      { 3654061472, 5 },
      { kMarkvNoneOfTheAbove, 45 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFunctionCall, 12), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 107544081, 2 },
      { 125015036, 1 },
      { 586244865, 10 },
      { 1033081852, 36 },
      { 1064945649, 2 },
      { 1155765244, 2 },
      { 1304296041, 1 },
      { 1543646433, 7 },
      { 1782996825, 7 },
      { 1941148668, 4 },
      { 2002490364, 8 },
      { 2022347217, 3 },
      { 2063832060, 1 },
      { 2487708241, 1 },
      { 2726532092, 2 },
      { 2849215484, 2 },
      { 2966409025, 2 },
      { 3445109809, 3 },
      { 3458449569, 8 },
      { 3634598908, 20 },
      { 3695940604, 5 },
      { 3923810593, 3 },
      { kMarkvNoneOfTheAbove, 7 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFunctionCall, 13), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 247698428, 1 },
      { 309040124, 12 },
      { 333554713, 6 },
      { 572905105, 8 },
      { 1033081852, 2 },
      { 2002490364, 1 },
      { 2009007457, 1 },
      { 2487708241, 3 },
      { 3634598908, 1 },
      { 3695940604, 8 },
      { 3923810593, 6 },
      { kMarkvNoneOfTheAbove, 3 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFunctionCall, 14), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 247698428, 9 },
      { 1033081852, 5 },
      { 2002490364, 2 },
      { 2910557180, 1 },
      { 3757282300, 7 },
      { kMarkvNoneOfTheAbove, 2 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFunctionCall, 15), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 1033081852, 11 },
      { 1094423548, 3 },
      { 2002490364, 2 },
      { 3757282300, 8 },
      { kMarkvNoneOfTheAbove, 2 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFunctionCall, 16), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 543621065, 14838 },
      { 827698488, 31357 },
      { 1294403159, 39679 },
      { 1296054774, 73059 },
      { 1297706389, 23158 },
      { 1322549027, 2234 },
      { 2194691858, 16586 },
      { 2468230023, 25074 },
      { 2547657777, 20644 },
      { 2549309392, 4100 },
      { 2550961007, 6551 },
      { 2934934694, 26138 },
      { 2936586309, 8297 },
      { 2938237924, 7156 },
      { 3095831808, 12726 },
      { 3183924418, 3026 },
      { 3561562003, 17353 },
      { 4028622909, 4163 },
      { 4050155669, 2039 },
      { kMarkvNoneOfTheAbove, 18578 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpVariable, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 37459569, 16586 },
      { 137840602, 8297 },
      { 565334834, 2234 },
      { 625975427, 3026 },
      { 630964591, 6551 },
      { 680016782, 4163 },
      { 1009983433, 7156 },
      { 1572088444, 25074 },
      { 1584774136, 14838 },
      { 1918481917, 39679 },
      { 2790624748, 73059 },
      { 3181646225, 20644 },
      { 3253403867, 2039 },
      { 3560665067, 26138 },
      { 3662767579, 23158 },
      { 4053789056, 4100 },
      { 4064212479, 12726 },
      { 4192247221, 17353 },
      { 4224872590, 31357 },
      { kMarkvNoneOfTheAbove, 18578 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpVariable, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 162255877, 29330 },
      { 679771963, 184683 },
      { 789872778, 5526 },
      { 1951208733, 312816 },
      { 2320303498, 67529 },
      { 2984325996, 26930 },
      { 3334207724, 235873 },
      { 3868239231, 5490 },
      { 3869890846, 29529 },
      { kMarkvNoneOfTheAbove, 45884 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpLoad, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 269823086, 11528 },
      { 543558236, 26650 },
      { 810488476, 37277 },
      { 870594305, 18502 },
      { 883854656, 28660 },
      { 1141965917, 9956 },
      { 1570165302, 20878 },
      { 1684282922, 11978 },
      { 1742737136, 9134 },
      { 1901166356, 15145 },
      { 1949759310, 29050 },
      { 2087004702, 9538 },
      { 2096388952, 88141 },
      { 2157103435, 9408 },
      { 2219733501, 12365 },
      { 2356768706, 5490 },
      { 2517964682, 12667 },
      { 2614879967, 8357 },
      { 2622612602, 60397 },
      { 2660843182, 8754 },
      { 2970183398, 10697 },
      { 3091876332, 14791 },
      { 3187066832, 46642 },
      { 3244209297, 5903 },
      { 3487022798, 7415 },
      { 3496407048, 24777 },
      { 3570411982, 24027 },
      { 3692647551, 12609 },
      { 3713290482, 11513 },
      { 3831290364, 4827 },
      { kMarkvNoneOfTheAbove, 45884 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpLoad, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 37459569, 28660 },
      { 137840602, 10697 },
      { 522971108, 46642 },
      { 565334834, 5490 },
      { 630964591, 12667 },
      { 1079999262, 9408 },
      { 1395113939, 8754 },
      { 1572088444, 11978 },
      { 1584774136, 29050 },
      { 1649426421, 5903 },
      { 1918481917, 60397 },
      { 1957218950, 15145 },
      { 2311941439, 12365 },
      { 2313593054, 14791 },
      { 2790624748, 88141 },
      { 2838165089, 12609 },
      { 2839816704, 11528 },
      { 2841468319, 9956 },
      { 3181646225, 24027 },
      { 3364388739, 18502 },
      { 3366040354, 9134 },
      { 3367691969, 8357 },
      { 3369343584, 7415 },
      { 3560665067, 24777 },
      { 3662767579, 20878 },
      { 3945795573, 11513 },
      { 4064212479, 26650 },
      { 4224872590, 37277 },
      { 4239834800, 9538 },
      { 4243138030, 4827 },
      { kMarkvNoneOfTheAbove, 45884 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpLoad, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 137840602, 10645 },
      { 522971108, 4262 },
      { 1009983433, 7155 },
      { 1079999262, 5472 },
      { 1572088444, 25652 },
      { 1918481917, 57341 },
      { 2311941439, 3048 },
      { 2313593054, 2662 },
      { 2790624748, 109737 },
      { 2838165089, 3002 },
      { 2994529201, 2631 },
      { 2996180816, 2631 },
      { 2997832431, 2631 },
      { 3253403867, 2970 },
      { 3364388739, 4407 },
      { 3560665067, 29652 },
      { 3662767579, 26995 },
      { 3945795573, 3265 },
      { 4192247221, 17947 },
      { 4224872590, 49052 },
      { kMarkvNoneOfTheAbove, 22868 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpStore, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 177111659, 3073 },
      { 296981500, 4883 },
      { 408465899, 4042 },
      { 810488476, 5485 },
      { 870594305, 3826 },
      { 1901166356, 2504 },
      { 2055836767, 4342 },
      { 2087004702, 7962 },
      { 2096388952, 13554 },
      { 2204920111, 2607 },
      { 2517964682, 2855 },
      { 2622612602, 8721 },
      { 2660843182, 2945 },
      { 2855506940, 7887 },
      { 2959147533, 4106 },
      { 3187066832, 14420 },
      { 3504158761, 4496 },
      { 3570411982, 7499 },
      { 3764205609, 3112 },
      { 3831290364, 4042 },
      { kMarkvNoneOfTheAbove, 22868 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpStore, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 440421571, 81479 },
      { 827698488, 168666 },
      { 907126242, 49525 },
      { 908777857, 28009 },
      { 910429472, 10962 },
      { 1294403159, 90396 },
      { 1296054774, 11390 },
      { 1297706389, 16495 },
      { 2080953106, 6685 },
      { 2468230023, 28811 },
      { 2547657777, 3168 },
      { 3561562003, 5486 },
      { kMarkvNoneOfTheAbove, 25425 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpAccessChain, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 155458798, 2764 },
      { 213642219, 3261 },
      { 221900294, 3725 },
      { 1079999262, 10176 },
      { 1302400505, 3016 },
      { 1649426421, 5903 },
      { 2311941439, 15413 },
      { 2313593054, 17453 },
      { 2838165089, 15611 },
      { 2839816704, 13213 },
      { 2841468319, 10035 },
      { 2994529201, 3875 },
      { 2996180816, 3875 },
      { 2997832431, 3875 },
      { 3364388739, 22909 },
      { 3366040354, 10479 },
      { 3367691969, 9241 },
      { 3369343584, 8477 },
      { 3928842969, 3965 },
      { 3930494584, 2604 },
      { 4239834800, 9855 },
      { 4241486415, 4579 },
      { 4243138030, 4858 },
      { 4244789645, 4478 },
      { kMarkvNoneOfTheAbove, 25425 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpAccessChain, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 116376005, 11317 },
      { 400248103, 2592 },
      { 406044930, 14594 },
      { 522971108, 9029 },
      { 615341051, 4736 },
      { 625975427, 28360 },
      { 680016782, 27906 },
      { 763027711, 2560 },
      { 977312655, 5120 },
      { 1009983433, 3392 },
      { 1062250709, 9534 },
      { 1410849099, 2880 },
      { 1642805350, 2544 },
      { 1692932387, 3680 },
      { 1698730948, 3360 },
      { 1827244161, 3880 },
      { 1918481917, 51106 },
      { 2096472894, 2800 },
      { 2190437442, 23694 },
      { 2785441472, 4968 },
      { 2790624748, 38859 },
      { 2879917723, 13248 },
      { 2882994691, 24268 },
      { 3181646225, 5415 },
      { 3255947500, 2560 },
      { 3263901372, 11625 },
      { 3268751013, 3105 },
      { 3390051757, 24308 },
      { 3560665067, 23770 },
      { 3662767579, 32866 },
      { 3717523241, 8504 },
      { 3941049054, 2928 },
      { 3945795573, 6731 },
      { 4080527786, 4088 },
      { 4290024976, 33634 },
      { kMarkvNoneOfTheAbove, 25425 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpAccessChain, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 51041423, 5073 },
      { 142465290, 23895 },
      { 144116905, 52934 },
      { 290391815, 4836 },
      { 529742207, 10987 },
      { 677668732, 3519 },
      { 917019124, 2965 },
      { 1064945649, 5290 },
      { 1156369516, 21757 },
      { 1158021131, 35366 },
      { 1304296041, 5209 },
      { 1543646433, 12248 },
      { 1691572958, 3742 },
      { 1782996825, 88059 },
      { 1784648440, 50244 },
      { 1930923350, 5932 },
      { 2170273742, 21328 },
      { 2318200267, 3425 },
      { 2557550659, 7737 },
      { 2705477184, 14450 },
      { 2796901051, 20697 },
      { 2798552666, 38359 },
      { 2944827576, 2670 },
      { 3184177968, 11727 },
      { 3332104493, 4702 },
      { 3571454885, 5081 },
      { 3810805277, 15923 },
      { 3958731802, 4802 },
      { 4198082194, 10470 },
      { kMarkvNoneOfTheAbove, 25425 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpAccessChain, 3), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 142465290, 4929 },
      { 144116905, 5415 },
      { 198967948, 870 },
      { 529742207, 959 },
      { 586244865, 790 },
      { 825595257, 970 },
      { 917019124, 725 },
      { 973521782, 790 },
      { 1156369516, 3794 },
      { 1158021131, 14513 },
      { 1212872174, 809 },
      { 1452222566, 1025 },
      { 1543646433, 624 },
      { 1600149091, 790 },
      { 1782996825, 6267 },
      { 1784648440, 7937 },
      { 1839499483, 970 },
      { 1930923350, 602 },
      { 2170273742, 1121 },
      { 2226776400, 729 },
      { 2466126792, 961 },
      { 2557550659, 741 },
      { 2614053317, 790 },
      { 2796901051, 4386 },
      { 2798552666, 5645 },
      { 2853403709, 809 },
      { 3184177968, 940 },
      { 3240680626, 790 },
      { 3480031018, 970 },
      { 3571454885, 614 },
      { 3810805277, 1221 },
      { 3867307935, 809 },
      { 4106658327, 1025 },
      { 4198082194, 840 },
      { 4254584852, 790 },
      { kMarkvNoneOfTheAbove, 4046 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpAccessChain, 4), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 144116905, 1628 },
      { 1158021131, 1628 },
      { 1784648440, 1640 },
      { 2798552666, 1628 },
      { kMarkvNoneOfTheAbove, 329 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpAccessChain, 5), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 88112 },
      { 2320303498, 65099 },
      { 3334207724, 42240 },
      { kMarkvNoneOfTheAbove, 9773 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpVectorShuffle, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 177111659, 3512 },
      { 325481827, 1596 },
      { 397240040, 1006 },
      { 425242871, 1536 },
      { 640865899, 1356 },
      { 669323523, 1332 },
      { 837715723, 7999 },
      { 921246433, 1226 },
      { 1022544883, 1913 },
      { 1054461787, 1479 },
      { 1203545131, 3778 },
      { 1352628475, 2331 },
      { 1367301635, 20492 },
      { 1516384979, 1640 },
      { 1548491889, 1056 },
      { 2055836767, 7635 },
      { 2204920111, 3546 },
      { 2572638469, 3936 },
      { 2597020383, 1824 },
      { 2604242419, 1056 },
      { 2812159762, 1596 },
      { 2842919847, 1020 },
      { 3156916514, 1294 },
      { 3619787319, 5968 },
      { 3701632935, 1073 },
      { 3783543823, 3137 },
      { 3783756895, 1304 },
      { kMarkvNoneOfTheAbove, 9773 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpVectorShuffle, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 18791930, 1046 },
      { 495107308, 1283 },
      { 504514034, 1056 },
      { 1071164424, 1784 },
      { 1136911283, 1922 },
      { 1376656865, 1056 },
      { 1570165302, 2402 },
      { 1822823090, 1314 },
      { 1977038330, 2615 },
      { 2096388952, 10291 },
      { 2157103435, 8640 },
      { 2231688008, 1824 },
      { 2622612602, 44313 },
      { 2815450086, 1104 },
      { 3112016555, 1172 },
      { 3187066832, 2135 },
      { 3296722158, 1356 },
      { 3496407048, 21164 },
      { 3570411982, 14972 },
      { 3577050706, 1002 },
      { 3713290482, 5700 },
      { 3973051545, 1536 },
      { kMarkvNoneOfTheAbove, 9773 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpVectorShuffle, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 18791930, 1046 },
      { 495107308, 1283 },
      { 504514034, 1056 },
      { 1071164424, 1604 },
      { 1136911283, 1562 },
      { 1376656865, 1056 },
      { 1570165302, 3206 },
      { 1949856502, 1602 },
      { 1977038330, 2615 },
      { 2096388952, 10511 },
      { 2157103435, 3936 },
      { 2231688008, 1824 },
      { 2622612602, 29141 },
      { 2656211099, 1536 },
      { 3112016555, 1172 },
      { 3187066832, 1216 },
      { 3296722158, 1356 },
      { 3496407048, 17939 },
      { 3570411982, 14972 },
      { 3713290482, 3967 },
      { 3973051545, 1536 },
      { kMarkvNoneOfTheAbove, 9773 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpVectorShuffle, 3), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 61690 },
      { 1146476634, 766 },
      { 2320303498, 25693 },
      { 3334207724, 57419 },
      { 3800912395, 842 },
      { 3802564010, 1307 },
      { kMarkvNoneOfTheAbove, 7423 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpCompositeConstruct, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 429277936, 863 },
      { 449954059, 1356 },
      { 508217552, 2861 },
      { 552163695, 790 },
      { 565959741, 914 },
      { 597509298, 914 },
      { 655451355, 790 },
      { 689091011, 1121 },
      { 708937084, 768 },
      { 886972033, 814 },
      { 1032593647, 1341 },
      { 1209418480, 830 },
      { 1319785741, 2428 },
      { 1342456627, 790 },
      { 1445744287, 790 },
      { 1479383943, 949 },
      { 1483007500, 1184 },
      { 1565244936, 768 },
      { 1692546388, 791 },
      { 1753105144, 1241 },
      { 1822886579, 1128 },
      { 2119793999, 921 },
      { 2132749559, 890 },
      { 2236037219, 790 },
      { 2269676875, 936 },
      { 2476252195, 925 },
      { 2579539855, 790 },
      { 2613179511, 1251 },
      { 2621255555, 1872 },
      { 2758064548, 814 },
      { 2817335337, 856 },
      { 2881302403, 1843 },
      { 2883413496, 768 },
      { 2923042491, 790 },
      { 3063300848, 943 },
      { 3151638847, 3256 },
      { 3169223085, 773 },
      { 3232730629, 800 },
      { 3233393284, 1110 },
      { 3266545127, 845 },
      { 3297003816, 814 },
      { 3323682385, 1644 },
      { 3369832787, 790 },
      { 3403472443, 1181 },
      { 3547456240, 891 },
      { 3732082272, 782 },
      { 3753486980, 3563 },
      { 3816961131, 828 },
      { 3850262739, 825 },
      { 3931641900, 1550 },
      { 4056838059, 781 },
      { 4160125719, 790 },
      { 4193765375, 913 },
      { kMarkvNoneOfTheAbove, 7423 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpCompositeConstruct, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 25054578, 914 },
      { 135761240, 1121 },
      { 296981500, 6809 },
      { 412284019, 811 },
      { 447740977, 1025 },
      { 663636505, 955 },
      { 810488476, 3290 },
      { 870594305, 1068 },
      { 959681532, 1791 },
      { 975616242, 961 },
      { 1141965917, 7552 },
      { 1191511770, 938 },
      { 1204787336, 791 },
      { 1377226159, 1198 },
      { 1400260920, 1000 },
      { 1431526331, 790 },
      { 1447712361, 814 },
      { 1503491507, 970 },
      { 1749741911, 768 },
      { 1791352211, 1471 },
      { 1959401596, 790 },
      { 2011183308, 782 },
      { 2031366772, 970 },
      { 2207819821, 1221 },
      { 2247262300, 825 },
      { 2319227476, 1402 },
      { 2487276861, 790 },
      { 2559242037, 970 },
      { 2730802997, 828 },
      { 2789375411, 1356 },
      { 2817579280, 1550 },
      { 2847102741, 1392 },
      { 2855506940, 3667 },
      { 2860348412, 1147 },
      { 3015152126, 790 },
      { 3087117302, 870 },
      { 3091876332, 5819 },
      { 3168953855, 856 },
      { 3260844985, 768 },
      { 3374978006, 1331 },
      { 3375366192, 1184 },
      { 3543027391, 790 },
      { 3579593979, 1644 },
      { 3902853271, 1221 },
      { 4070902656, 790 },
      { 4088223711, 768 },
      { 4185661467, 914 },
      { 4214833008, 1025 },
      { kMarkvNoneOfTheAbove, 7423 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpCompositeConstruct, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 25054578, 914 },
      { 109210376, 768 },
      { 150820676, 1471 },
      { 269823086, 7477 },
      { 296981500, 6826 },
      { 318870061, 790 },
      { 390835237, 970 },
      { 606730765, 825 },
      { 678695941, 1402 },
      { 810488476, 3257 },
      { 846745326, 790 },
      { 870594305, 1467 },
      { 918710502, 970 },
      { 959681532, 4352 },
      { 1090271462, 828 },
      { 1204787336, 791 },
      { 1206571206, 1392 },
      { 1323407757, 1154 },
      { 1374620591, 790 },
      { 1446585767, 870 },
      { 1447712361, 814 },
      { 1620313450, 768 },
      { 1734446471, 1331 },
      { 1734834657, 1184 },
      { 1902495856, 790 },
      { 2011183308, 782 },
      { 2207819821, 814 },
      { 2219733501, 5689 },
      { 2262321736, 1221 },
      { 2430371121, 790 },
      { 2447692176, 768 },
      { 2574301473, 1025 },
      { 2790197001, 1121 },
      { 2817579280, 1550 },
      { 2855506940, 7393 },
      { 2860348412, 1147 },
      { 3066719780, 811 },
      { 3102176738, 1025 },
      { 3168953855, 856 },
      { 3318072266, 955 },
      { 3630052003, 961 },
      { 3845947531, 938 },
      { 4054696681, 1000 },
      { 4085962092, 790 },
      { 4157927268, 970 },
      { 4182141402, 1644 },
      { 4185661467, 928 },
      { kMarkvNoneOfTheAbove, 7423 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpCompositeConstruct, 3), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 21949760, 723 },
      { 25054578, 928 },
      { 93914936, 1298 },
      { 94303122, 1184 },
      { 261964321, 790 },
      { 296981500, 3290 },
      { 451264926, 994 },
      { 549825025, 614 },
      { 621790201, 918 },
      { 789839586, 790 },
      { 810488476, 2934 },
      { 870594305, 925 },
      { 933769938, 1025 },
      { 959681532, 1485 },
      { 1149665466, 1121 },
      { 1204787336, 791 },
      { 1227221002, 3663 },
      { 1426188245, 811 },
      { 1447712361, 814 },
      { 1461645203, 1025 },
      { 1561718045, 3456 },
      { 1575323572, 672 },
      { 1677540731, 955 },
      { 1989520468, 961 },
      { 2011183308, 782 },
      { 2205415996, 936 },
      { 2414165146, 1000 },
      { 2445430557, 790 },
      { 2517395733, 970 },
      { 2805256437, 1414 },
      { 2855506940, 4276 },
      { 2860348412, 1147 },
      { 2921996537, 695 },
      { 2973305822, 790 },
      { 3045270998, 970 },
      { 3168953855, 856 },
      { 3261166526, 825 },
      { 3333131702, 1339 },
      { 3501181087, 790 },
      { 3573146263, 970 },
      { 3692647551, 7139 },
      { 3744707223, 828 },
      { 3789041791, 719 },
      { 3861006967, 1287 },
      { 4029056352, 790 },
      { 4101021528, 870 },
      { 4185661467, 914 },
      { kMarkvNoneOfTheAbove, 6100 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpCompositeConstruct, 4), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 37009196, 949 },
      { 277023757, 489 },
      { 296981500, 8344 },
      { 348988933, 781 },
      { 451264926, 333 },
      { 492919285, 326 },
      { 564884461, 936 },
      { 804899022, 790 },
      { 810488476, 832 },
      { 870594305, 1250 },
      { 876864198, 790 },
      { 928261291, 415 },
      { 959681532, 544 },
      { 1020794550, 322 },
      { 1092759726, 294 },
      { 1164724902, 1341 },
      { 1323407757, 493 },
      { 1332774287, 790 },
      { 1377226159, 384 },
      { 1404739463, 790 },
      { 1447712361, 814 },
      { 1450415100, 347 },
      { 1548669815, 322 },
      { 1620634991, 825 },
      { 1692600167, 1128 },
      { 1860649552, 790 },
      { 1932614728, 890 },
      { 2076545080, 318 },
      { 2087004702, 468 },
      { 2148510256, 719 },
      { 2219733501, 425 },
      { 2220475432, 1251 },
      { 2388524817, 790 },
      { 2460489993, 790 },
      { 2604420345, 324 },
      { 2676385521, 723 },
      { 2748350697, 1181 },
      { 2855506940, 6433 },
      { 2916400082, 790 },
      { 2988365258, 489 },
      { 3204260786, 614 },
      { 3276225962, 913 },
      { 3403472443, 314 },
      { 3444275347, 790 },
      { 3516240523, 489 },
      { 3588205699, 925 },
      { 3732136051, 602 },
      { 3756631791, 384 },
      { 3804101227, 1121 },
      { 3874089391, 592 },
      { 4044115788, 569 },
      { 4116080964, 845 },
      { 4260011316, 342 },
      { kMarkvNoneOfTheAbove, 2905 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpCompositeConstruct, 5), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 789872778, 1644 },
      { 1951208733, 176575 },
      { 2320303498, 2520 },
      { 3334207724, 1480 },
      { kMarkvNoneOfTheAbove, 9178 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpCompositeExtract, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 37009196, 949 },
      { 93914936, 1298 },
      { 94303122, 1184 },
      { 135761240, 1121 },
      { 150820676, 1471 },
      { 390835237, 970 },
      { 447740977, 1025 },
      { 461040879, 936 },
      { 564884461, 936 },
      { 621790201, 918 },
      { 663636505, 957 },
      { 678695941, 1402 },
      { 898191441, 3256 },
      { 918710502, 970 },
      { 933769938, 1025 },
      { 975616242, 961 },
      { 1075177279, 936 },
      { 1149665466, 1121 },
      { 1164724902, 1341 },
      { 1191511770, 938 },
      { 1206571206, 1392 },
      { 1400260920, 1000 },
      { 1461645203, 1025 },
      { 1503491507, 970 },
      { 1677540731, 955 },
      { 1692600167, 1128 },
      { 1734446471, 1331 },
      { 1734834657, 1184 },
      { 1791352211, 1472 },
      { 1989520468, 961 },
      { 2031366772, 970 },
      { 2205415996, 936 },
      { 2220475432, 1251 },
      { 2262321736, 1221 },
      { 2319227476, 1402 },
      { 2414165146, 1000 },
      { 2517395733, 970 },
      { 2559242037, 970 },
      { 2574301473, 1025 },
      { 2748350697, 1181 },
      { 2790197001, 1121 },
      { 2805256437, 1414 },
      { 2847102741, 1394 },
      { 3045270998, 970 },
      { 3102176738, 1025 },
      { 3318072266, 955 },
      { 3333131702, 1339 },
      { 3374978006, 1331 },
      { 3375366192, 1184 },
      { 3573146263, 970 },
      { 3588205699, 925 },
      { 3630052003, 961 },
      { 3732916270, 936 },
      { 3804101227, 1121 },
      { 3845947531, 938 },
      { 3861006967, 1287 },
      { 3902853271, 1221 },
      { 4054696681, 1000 },
      { 4157927268, 970 },
      { 4214833008, 1025 },
      { kMarkvNoneOfTheAbove, 9178 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpCompositeExtract, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 13107491, 5698 },
      { 68162857, 1536 },
      { 293528591, 3160 },
      { 359275450, 2196 },
      { 425022309, 3800 },
      { 435174951, 1536 },
      { 490769168, 4000 },
      { 556516027, 1016 },
      { 622262886, 1344 },
      { 688009745, 1368 },
      { 753756604, 3300 },
      { 819503463, 4273 },
      { 1165671422, 3160 },
      { 1231418281, 2276 },
      { 1297165140, 3700 },
      { 1308743936, 936 },
      { 1367301635, 2433 },
      { 1494405717, 1334 },
      { 1548491889, 1584 },
      { 1560152576, 2408 },
      { 1623041839, 1536 },
      { 1625899435, 1803 },
      { 1691646294, 5141 },
      { 1978689945, 1046 },
      { 2037710159, 1644 },
      { 2037814253, 3160 },
      { 2096388952, 3000 },
      { 2103561112, 2196 },
      { 2169307971, 3700 },
      { 2262220987, 3432 },
      { 2366548548, 1342 },
      { 2432295407, 2456 },
      { 2498042266, 3748 },
      { 2563789125, 5324 },
      { 2604242419, 1584 },
      { 2674422363, 2016 },
      { 2909957084, 3160 },
      { 2975703943, 2196 },
      { 3041450802, 3664 },
      { 3107197661, 976 },
      { 3151638847, 3256 },
      { 3187066832, 3552 },
      { 3238691379, 1342 },
      { 3304438238, 2896 },
      { 3370185097, 3816 },
      { 3435931956, 5271 },
      { 3659992949, 1584 },
      { 3716353056, 3160 },
      { 3782099915, 3160 },
      { 3805291365, 2484 },
      { 3847846774, 3400 },
      { 3913593633, 3920 },
      { 3979340492, 980 },
      { 3997038726, 2928 },
      { 4110834210, 1352 },
      { 4176581069, 2892 },
      { 4242327928, 4484 },
      { kMarkvNoneOfTheAbove, 9178 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpCompositeExtract, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 3866587616, 576 },
      { 3868239231, 5280 },
      { 3869890846, 23474 },
      { kMarkvNoneOfTheAbove, 1467 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpSampledImage, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 1164218401, 3456 },
      { 2036361232, 23194 },
      { 2366506734, 280 },
      { 2637132451, 1824 },
      { 3237903670, 576 },
      { kMarkvNoneOfTheAbove, 1467 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpSampledImage, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 543558236, 26650 },
      { 1069781886, 1824 },
      { 1596005536, 576 },
      { 1622674811, 280 },
      { kMarkvNoneOfTheAbove, 1467 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpSampledImage, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 1949759310, 29050 },
      { 3028875885, 280 },
      { kMarkvNoneOfTheAbove, 1467 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpSampledImage, 3), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 3334207724, 39107 },
      { kMarkvNoneOfTheAbove, 1956 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpImageSampleImplicitLod, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 49925462, 256 },
      { 232480090, 240 },
      { 236660303, 1204 },
      { 258702501, 240 },
      { 488500848, 1836 },
      { 495107308, 3749 },
      { 534980544, 200 },
      { 603948232, 245 },
      { 610554692, 303 },
      { 727870679, 240 },
      { 893813591, 196 },
      { 938959680, 240 },
      { 1129305538, 300 },
      { 1265998516, 388 },
      { 1365842164, 388 },
      { 1435557674, 240 },
      { 1508074873, 768 },
      { 1562748773, 300 },
      { 1642818143, 494 },
      { 1766155648, 240 },
      { 1802975153, 240 },
      { 1898331277, 240 },
      { 2095565698, 314 },
      { 2157241872, 240 },
      { 2160683793, 338 },
      { 2214176073, 240 },
      { 2231688008, 1824 },
      { 2232491275, 528 },
      { 2571973814, 240 },
      { 2659248097, 240 },
      { 2677252364, 308 },
      { 2683344251, 240 },
      { 2833158656, 204 },
      { 2839765116, 644 },
      { 2855895374, 390 },
      { 2894433252, 240 },
      { 2913136690, 388 },
      { 3012980338, 380 },
      { 3321164184, 248 },
      { 3327770644, 798 },
      { 3457985288, 768 },
      { 3474362419, 300 },
      { 3651028922, 196 },
      { 3657635382, 976 },
      { 3680828077, 300 },
      { 3716743507, 199 },
      { 3721629220, 240 },
      { 3750872570, 196 },
      { 3757479030, 976 },
      { 3775051398, 240 },
      { 3797204453, 576 },
      { 3816274764, 300 },
      { 3853804849, 240 },
      { 4112715444, 240 },
      { kMarkvNoneOfTheAbove, 1956 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpImageSampleImplicitLod, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 883854656, 18903 },
      { 1962971231, 629 },
      { 2036361232, 14638 },
      { 2356768706, 1827 },
      { 2637132451, 1824 },
      { 3237903670, 576 },
      { 3829682756, 710 },
      { kMarkvNoneOfTheAbove, 1956 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpImageSampleImplicitLod, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 13349969, 240 },
      { 71871541, 240 },
      { 255302575, 388 },
      { 274356208, 240 },
      { 332877780, 240 },
      { 681879640, 240 },
      { 740401212, 240 },
      { 740921498, 494 },
      { 778976588, 199 },
      { 883177998, 196 },
      { 921246433, 976 },
      { 1142801403, 240 },
      { 1162127370, 576 },
      { 1187426424, 300 },
      { 1201322975, 240 },
      { 1329499601, 976 },
      { 1352628475, 1836 },
      { 1360580283, 204 },
      { 1516384979, 245 },
      { 1544541995, 314 },
      { 1570165302, 714 },
      { 1600392975, 798 },
      { 1723565506, 240 },
      { 1782087078, 240 },
      { 1980341560, 356 },
      { 2276405827, 768 },
      { 2323422236, 338 },
      { 2389169095, 300 },
      { 2397813436, 240 },
      { 2414368715, 240 },
      { 2472890287, 240 },
      { 2517964682, 1204 },
      { 2532518896, 1824 },
      { 2689233019, 200 },
      { 2754074729, 390 },
      { 2775037933, 300 },
      { 2804281092, 768 },
      { 2841008029, 380 },
      { 2949054885, 196 },
      { 3209518833, 256 },
      { 3249261197, 388 },
      { 3302913198, 300 },
      { 3328526828, 240 },
      { 3357308053, 196 },
      { 3507288448, 300 },
      { 3619787319, 3749 },
      { 3627739127, 644 },
      { 3628201427, 248 },
      { 3758028577, 240 },
      { 3783543823, 303 },
      { 3816550149, 240 },
      { 3908450875, 240 },
      { 3966972447, 240 },
      { 4142016703, 388 },
      { kMarkvNoneOfTheAbove, 1956 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpImageSampleImplicitLod, 3), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 2855506940, 1 },
      { 3266548732, 1 },
      { 3732640764, 33 },
      { kMarkvNoneOfTheAbove, 2 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpImageSampleImplicitLod, 5), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 3334207724, 19083 },
      { kMarkvNoneOfTheAbove, 955 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpImageSampleExplicitLod, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 27177503, 173 },
      { 162608772, 1644 },
      { 180913835, 240 },
      { 414444763, 247 },
      { 443347828, 280 },
      { 587888644, 284 },
      { 750870327, 168 },
      { 875212982, 1644 },
      { 899320334, 173 },
      { 989813600, 284 },
      { 1057606514, 180 },
      { 1310404265, 1680 },
      { 1543935193, 351 },
      { 1623013158, 168 },
      { 1705716306, 284 },
      { 1747355813, 1644 },
      { 1755165354, 240 },
      { 1781864804, 312 },
      { 1941403425, 312 },
      { 2023008475, 1120 },
      { 2157463947, 104 },
      { 2213902613, 104 },
      { 2317002568, 104 },
      { 2373441234, 104 },
      { 2495155989, 168 },
      { 2577859137, 284 },
      { 2752450145, 120 },
      { 2857814560, 267 },
      { 2895151306, 1120 },
      { 3029606778, 104 },
      { 3296722158, 1356 },
      { 3367298820, 168 },
      { 3450001968, 828 },
      { 3526837441, 168 },
      { 3609540589, 835 },
      { 3862396471, 126 },
      { 3901749609, 104 },
      { kMarkvNoneOfTheAbove, 955 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpImageSampleExplicitLod, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 883854656, 9757 },
      { 1962971231, 240 },
      { 2036361232, 8556 },
      { 2366506734, 280 },
      { 3829682756, 250 },
      { kMarkvNoneOfTheAbove, 955 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpImageSampleExplicitLod, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 449954059, 1356 },
      { 1022544883, 126 },
      { 1570165302, 4275 },
      { 1894133125, 416 },
      { 1943501114, 120 },
      { 1980341560, 240 },
      { 2517964682, 2577 },
      { 2616085763, 392 },
      { 2710583246, 840 },
      { 2924263085, 267 },
      { 3202324433, 284 },
      { 3289213933, 284 },
      { 3323682385, 4932 },
      { 3366848728, 487 },
      { 3417583519, 180 },
      { 3732916270, 624 },
      { 3877813395, 280 },
      { 4028028350, 280 },
      { 4178218543, 208 },
      { kMarkvNoneOfTheAbove, 955 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpImageSampleExplicitLod, 3), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 2845699577, 126 },
      { 2855506940, 18938 },
      { kMarkvNoneOfTheAbove, 955 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpImageSampleExplicitLod, 5), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 3533637837, 280 },
      { 3535289452, 564 },
      { 3536941067, 564 },
      { 3538592682, 1628 },
      { 3540244297, 3097 },
      { 3541895912, 3097 },
      { kMarkvNoneOfTheAbove, 462 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpImageSampleExplicitLod, 6), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 1951208733, 7119 },
      { kMarkvNoneOfTheAbove, 356 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpImageSampleDrefExplicitLod, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 98822682, 768 },
      { 1210863750, 384 },
      { 1447712361, 3256 },
      { 1588591290, 384 },
      { 1929656674, 768 },
      { 2648449598, 384 },
      { 3007564557, 407 },
      { 3717888826, 768 },
      { kMarkvNoneOfTheAbove, 356 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpImageSampleDrefExplicitLod, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 1164218401, 3456 },
      { 2356768706, 3663 },
      { kMarkvNoneOfTheAbove, 356 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpImageSampleDrefExplicitLod, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 219787208, 384 },
      { 708937084, 768 },
      { 1565244936, 768 },
      { 1764687614, 384 },
      { 2037699457, 407 },
      { 2883413496, 768 },
      { 3151638847, 3256 },
      { 3948153850, 384 },
      { kMarkvNoneOfTheAbove, 356 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpImageSampleDrefExplicitLod, 3), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 156823196, 384 },
      { 551969662, 768 },
      { 581799434, 384 },
      { 762000698, 768 },
      { 898191441, 3256 },
      { 947116128, 384 },
      { 3452282954, 768 },
      { 4233277939, 407 },
      { kMarkvNoneOfTheAbove, 356 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpImageSampleDrefExplicitLod, 4), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 2855506940, 7119 },
      { kMarkvNoneOfTheAbove, 356 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpImageSampleDrefExplicitLod, 6), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 5202 },
      { 1951208733, 4280 },
      { 2320303498, 63 },
      { 3334207724, 156 },
      { kMarkvNoneOfTheAbove, 486 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFNegate, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 30458606, 120 },
      { 615331870, 1604 },
      { 840892127, 156 },
      { 887470203, 51 },
      { 1341943567, 768 },
      { 2564745684, 553 },
      { 2577387676, 768 },
      { 2724166585, 1596 },
      { 2733721088, 386 },
      { 2794551319, 240 },
      { 2972515618, 69 },
      { 3068671770, 59 },
      { 3303441573, 52 },
      { 3462411709, 240 },
      { 3664980682, 54 },
      { 3756631791, 768 },
      { 4118875338, 1536 },
      { 4216545943, 442 },
      { kMarkvNoneOfTheAbove, 486 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFNegate, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 498575, 768 },
      { 94149821, 240 },
      { 181902171, 1596 },
      { 269823086, 1604 },
      { 546323384, 52 },
      { 877895868, 1536 },
      { 1377226159, 768 },
      { 1510333659, 51 },
      { 1525586608, 156 },
      { 1651580604, 54 },
      { 1684282922, 59 },
      { 1742737136, 386 },
      { 1799299383, 240 },
      { 1978689945, 442 },
      { 1990382740, 69 },
      { 2533958702, 768 },
      { 2575883808, 120 },
      { 2970183398, 553 },
      { kMarkvNoneOfTheAbove, 486 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFNegate, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 37076 },
      { 1951208733, 30588 },
      { 2320303498, 30190 },
      { 3334207724, 8587 },
      { kMarkvNoneOfTheAbove, 5323 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFAdd, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 68162857, 768 },
      { 435174951, 768 },
      { 613951792, 814 },
      { 657322125, 960 },
      { 698256316, 768 },
      { 707080663, 560 },
      { 920941800, 1596 },
      { 997105648, 828 },
      { 1623041839, 768 },
      { 1669930486, 1536 },
      { 1708225821, 650 },
      { 1834885790, 658 },
      { 2276405827, 768 },
      { 2278571792, 658 },
      { 2656211099, 1536 },
      { 2804281092, 768 },
      { 2808437352, 552 },
      { 2870852215, 1596 },
      { 3001815609, 661 },
      { 3325419312, 834 },
      { 3543304670, 821 },
      { 3805291365, 828 },
      { 4182141402, 1644 },
      { kMarkvNoneOfTheAbove, 5323 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFAdd, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 294390719, 1536 },
      { 296981500, 1562 },
      { 402732606, 552 },
      { 810488476, 2822 },
      { 837715723, 1235 },
      { 847697527, 658 },
      { 959681532, 2101 },
      { 963902061, 1644 },
      { 1026574472, 782 },
      { 1367301635, 1400 },
      { 1570165302, 6765 },
      { 1730674532, 827 },
      { 1742737136, 782 },
      { 1898717303, 782 },
      { 1992893964, 843 },
      { 2096388952, 14068 },
      { 2129301998, 622 },
      { 2219733501, 1503 },
      { 2572638469, 1536 },
      { 2762094724, 658 },
      { 3001444829, 622 },
      { 3091876332, 800 },
      { 3187066832, 1685 },
      { 3641833815, 829 },
      { 3783543823, 3293 },
      { 3873587660, 622 },
      { kMarkvNoneOfTheAbove, 5323 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFAdd, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 503145996, 782 },
      { 782333456, 960 },
      { 837715723, 2045 },
      { 870594305, 1439 },
      { 952536201, 600 },
      { 959681532, 3754 },
      { 1026574472, 768 },
      { 1092948665, 1058 },
      { 1367301635, 1461 },
      { 1445617921, 560 },
      { 1516384979, 877 },
      { 1898717303, 768 },
      { 2096388952, 3106 },
      { 2219733501, 1907 },
      { 2355487981, 768 },
      { 2359973133, 542 },
      { 2444465148, 768 },
      { 2517964682, 1655 },
      { 2683080096, 1209 },
      { 3032677281, 1596 },
      { 3063300848, 834 },
      { 3317923659, 768 },
      { 3401762422, 834 },
      { 3436143898, 867 },
      { 3439607047, 674 },
      { 3560552546, 1596 },
      { kMarkvNoneOfTheAbove, 5323 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFAdd, 3), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 15233 },
      { 1951208733, 28267 },
      { 2320303498, 4451 },
      { 3334207724, 6727 },
      { kMarkvNoneOfTheAbove, 2734 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFSub, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 50385656, 792 },
      { 92552159, 331 },
      { 195244192, 600 },
      { 260283527, 280 },
      { 298622829, 400 },
      { 333855951, 782 },
      { 355531721, 296 },
      { 373366972, 280 },
      { 437867560, 344 },
      { 664302909, 324 },
      { 860536596, 401 },
      { 1000715980, 312 },
      { 1123617794, 528 },
      { 1170252460, 360 },
      { 1190147516, 800 },
      { 1215030156, 576 },
      { 1220749418, 384 },
      { 1335532923, 400 },
      { 1388411861, 360 },
      { 1461398554, 453 },
      { 1740864598, 344 },
      { 1763758554, 422 },
      { 1797960910, 960 },
      { 1903659435, 400 },
      { 2009083389, 411 },
      { 2094550054, 400 },
      { 2124837447, 1596 },
      { 2175136253, 428 },
      { 2190283997, 398 },
      { 2207819821, 407 },
      { 2269114589, 931 },
      { 2269130237, 768 },
      { 2275654429, 360 },
      { 2330636993, 2924 },
      { 2481746922, 528 },
      { 2500819054, 400 },
      { 2503770904, 672 },
      { 2585116089, 360 },
      { 2603020391, 768 },
      { 2604576561, 422 },
      { 2795773560, 800 },
      { 2803893660, 428 },
      { 2847824651, 319 },
      { 2885926612, 280 },
      { 2890638791, 1597 },
      { 2987773500, 288 },
      { 3088785099, 528 },
      { 3175802740, 288 },
      { 3181956744, 300 },
      { 3249278823, 400 },
      { 3256304404, 312 },
      { 3335250889, 584 },
      { 3510242586, 422 },
      { 3536449078, 288 },
      { 3536471583, 769 },
      { 3547074732, 372 },
      { 3564989939, 400 },
      { 3579593979, 1644 },
      { 3626033249, 422 },
      { 3727034815, 768 },
      { 3850327306, 422 },
      { 3859644190, 423 },
      { 3950980241, 422 },
      { 4136183787, 365 },
      { 4167600590, 312 },
      { 4169938071, 300 },
      { 4185661467, 928 },
      { kMarkvNoneOfTheAbove, 2734 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFSub, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 249378857, 850 },
      { 295017943, 360 },
      { 296981500, 13508 },
      { 669323523, 480 },
      { 777316081, 324 },
      { 777606168, 400 },
      { 836581417, 280 },
      { 860536596, 400 },
      { 1027242654, 428 },
      { 1167160774, 724 },
      { 1191015885, 342 },
      { 1203545131, 500 },
      { 1319785741, 801 },
      { 1367301635, 401 },
      { 1547894601, 280 },
      { 1684282922, 575 },
      { 1752686878, 576 },
      { 1797397012, 400 },
      { 1901166356, 691 },
      { 1906988301, 344 },
      { 2055836767, 564 },
      { 2096388952, 1164 },
      { 2217833278, 422 },
      { 2500819054, 800 },
      { 2575525651, 426 },
      { 2660843182, 545 },
      { 2745576697, 280 },
      { 2803893660, 428 },
      { 2855506940, 2603 },
      { 3088785099, 288 },
      { 3091876332, 2915 },
      { 3187066832, 4156 },
      { 3244209297, 1063 },
      { 3461159629, 365 },
      { 3564989939, 400 },
      { 3653838348, 2512 },
      { 3656700104, 344 },
      { 3692647551, 773 },
      { 3713290482, 800 },
      { 3753486980, 290 },
      { 3783756895, 600 },
      { 3797961332, 1597 },
      { 3859644190, 422 },
      { kMarkvNoneOfTheAbove, 2734 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFSub, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 49456560, 960 },
      { 170690025, 1597 },
      { 256142724, 312 },
      { 295017943, 760 },
      { 296981500, 1753 },
      { 330249537, 784 },
      { 374446446, 288 },
      { 426869087, 398 },
      { 435256475, 422 },
      { 470277359, 768 },
      { 589814822, 288 },
      { 592180731, 422 },
      { 663258455, 422 },
      { 670060216, 300 },
      { 808861080, 312 },
      { 810488476, 564 },
      { 870594305, 754 },
      { 877895868, 792 },
      { 1167160774, 401 },
      { 1203545131, 571 },
      { 1278818058, 672 },
      { 1319785741, 1064 },
      { 1377226159, 407 },
      { 1514228000, 360 },
      { 1530183840, 312 },
      { 1558001705, 800 },
      { 1558990974, 576 },
      { 1633850097, 782 },
      { 1684282922, 283 },
      { 1742737136, 424 },
      { 1862531541, 360 },
      { 1898577041, 400 },
      { 1901166356, 446 },
      { 1939930453, 400 },
      { 1941148668, 614 },
      { 1955104493, 1644 },
      { 2162274327, 768 },
      { 2219733501, 1052 },
      { 2244928358, 416 },
      { 2272221101, 528 },
      { 2445518906, 344 },
      { 2446960972, 300 },
      { 2566666743, 422 },
      { 2598189097, 800 },
      { 2614879967, 439 },
      { 2775815164, 531 },
      { 2970183398, 511 },
      { 3117071189, 400 },
      { 3187066832, 634 },
      { 3244209297, 4163 },
      { 3351672873, 400 },
      { 3392887901, 1056 },
      { 3487022798, 433 },
      { 3628821857, 280 },
      { 3692647551, 416 },
      { 3716588461, 400 },
      { 3753486980, 1892 },
      { 3765247327, 931 },
      { 3783756895, 300 },
      { 3817149113, 288 },
      { kMarkvNoneOfTheAbove, 2734 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFSub, 3), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 46680 },
      { 1951208733, 58202 },
      { 2320303498, 14317 },
      { 3334207724, 13013 },
      { kMarkvNoneOfTheAbove, 6611 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFMul, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 49456560, 960 },
      { 716110421, 929 },
      { 734656765, 984 },
      { 782333456, 960 },
      { 799719863, 807 },
      { 878980100, 779 },
      { 963902061, 1644 },
      { 1633850097, 782 },
      { 1722633666, 838 },
      { 1730674532, 836 },
      { 2680283743, 1596 },
      { 3293372874, 768 },
      { 3401762422, 834 },
      { 3697738938, 1537 },
      { kMarkvNoneOfTheAbove, 6611 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFMul, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 200922300, 1920 },
      { 397240040, 914 },
      { 565959741, 914 },
      { 597509298, 914 },
      { 698256316, 768 },
      { 810488476, 7577 },
      { 837715723, 858 },
      { 870594305, 1504 },
      { 959681532, 867 },
      { 1054461787, 1479 },
      { 1158929937, 1479 },
      { 1168927492, 1920 },
      { 1203545131, 1070 },
      { 1324351672, 844 },
      { 1367301635, 1410 },
      { 1400986378, 918 },
      { 1570165302, 696 },
      { 1618544981, 1456 },
      { 1684282922, 1059 },
      { 1692546388, 691 },
      { 1830851200, 1920 },
      { 1901166356, 3108 },
      { 1955104493, 1644 },
      { 1993928944, 782 },
      { 2096388952, 8850 },
      { 2161102232, 834 },
      { 2273129209, 918 },
      { 2517964682, 860 },
      { 2622612602, 1337 },
      { 2660843182, 1472 },
      { 3265340343, 768 },
      { 3270430997, 999 },
      { 3487022798, 888 },
      { 3496407048, 677 },
      { 3504158761, 1123 },
      { 3692647551, 996 },
      { 3732082272, 782 },
      { 3929248764, 1290 },
      { kMarkvNoneOfTheAbove, 6611 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFMul, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 5753037, 1087 },
      { 13319433, 867 },
      { 15502752, 1211 },
      { 50385656, 782 },
      { 162608772, 1479 },
      { 810488476, 2913 },
      { 870594305, 1237 },
      { 875212982, 1479 },
      { 959681532, 2042 },
      { 970218682, 768 },
      { 1019457583, 710 },
      { 1367301635, 1112 },
      { 1618544981, 792 },
      { 1684282922, 3499 },
      { 1742737136, 916 },
      { 1747355813, 1479 },
      { 1755648697, 781 },
      { 1901166356, 1383 },
      { 1980341560, 860 },
      { 1992893964, 847 },
      { 2096388952, 1698 },
      { 2130747644, 2764 },
      { 2135340676, 768 },
      { 2219733501, 947 },
      { 2269130237, 768 },
      { 2567464216, 680 },
      { 2580096524, 872 },
      { 2614879967, 3106 },
      { 2622612602, 770 },
      { 2970183398, 1048 },
      { 3002890475, 1804 },
      { 3051544163, 900 },
      { 3091876332, 770 },
      { 3187066832, 1658 },
      { 3266548732, 1078 },
      { 3487022798, 880 },
      { 3496407048, 722 },
      { 3713290482, 906 },
      { 3923686994, 672 },
      { 3929248764, 1950 },
      { 4169226615, 792 },
      { 4223980846, 816 },
      { kMarkvNoneOfTheAbove, 6611 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFMul, 3), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 2894 },
      { 1951208733, 7965 },
      { 2320303498, 2677 },
      { 3334207724, 1102 },
      { kMarkvNoneOfTheAbove, 732 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFDiv, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 10142671, 162 },
      { 41739659, 360 },
      { 171334650, 216 },
      { 200553094, 240 },
      { 257136089, 422 },
      { 294390719, 768 },
      { 462664429, 528 },
      { 514399844, 120 },
      { 626480004, 284 },
      { 651464351, 235 },
      { 701281393, 240 },
      { 783918780, 768 },
      { 862784766, 216 },
      { 914431979, 101 },
      { 930804377, 168 },
      { 952536201, 300 },
      { 1043738701, 168 },
      { 1161646834, 77 },
      { 1434223270, 300 },
      { 1440646342, 236 },
      { 1508570930, 174 },
      { 1761469971, 196 },
      { 1799299383, 240 },
      { 1910240213, 284 },
      { 1945006185, 174 },
      { 1998444837, 400 },
      { 2045285083, 564 },
      { 2180835121, 120 },
      { 2279273489, 126 },
      { 2289803479, 384 },
      { 2348676810, 168 },
      { 2573160348, 564 },
      { 2639720559, 960 },
      { 2890570341, 150 },
      { 3023287679, 471 },
      { 3069748713, 120 },
      { 3118548424, 284 },
      { 3274369826, 120 },
      { 3354166585, 120 },
      { 3877813395, 280 },
      { 4117704995, 126 },
      { kMarkvNoneOfTheAbove, 732 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFDiv, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 139011596, 126 },
      { 296981500, 553 },
      { 370232173, 216 },
      { 492958971, 283 },
      { 604894932, 150 },
      { 810488476, 523 },
      { 959681532, 1128 },
      { 1013756921, 284 },
      { 1209418480, 262 },
      { 1318081294, 168 },
      { 1367301635, 456 },
      { 1625742020, 174 },
      { 1684282922, 242 },
      { 1746004874, 168 },
      { 1758287856, 284 },
      { 1825629063, 101 },
      { 2094550054, 400 },
      { 2096388952, 610 },
      { 2389274518, 120 },
      { 2636946065, 216 },
      { 2709694527, 768 },
      { 2715304020, 360 },
      { 2806296851, 236 },
      { 2864543087, 168 },
      { 2912189913, 120 },
      { 3091876332, 1180 },
      { 3138977758, 126 },
      { 3198541202, 768 },
      { 3260579369, 300 },
      { 3263841912, 300 },
      { 3335250889, 240 },
      { 3345856521, 174 },
      { 3510242586, 422 },
      { 3783543823, 120 },
      { 3820814597, 284 },
      { 3925550473, 120 },
      { 4282128820, 120 },
      { kMarkvNoneOfTheAbove, 732 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFDiv, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 116093251, 240 },
      { 149720480, 280 },
      { 183103444, 240 },
      { 251209228, 528 },
      { 296981500, 167 },
      { 357505993, 216 },
      { 410274915, 280 },
      { 452208841, 1128 },
      { 810488476, 987 },
      { 1144188012, 150 },
      { 1220127364, 280 },
      { 1321616112, 528 },
      { 1324351672, 1128 },
      { 1497092462, 120 },
      { 1531216990, 196 },
      { 1543672828, 476 },
      { 1904128160, 126 },
      { 2071351379, 768 },
      { 2119793999, 384 },
      { 2274779301, 240 },
      { 2357410109, 284 },
      { 2364922143, 101 },
      { 3047649911, 782 },
      { 3331487616, 126 },
      { 3398029189, 77 },
      { 3570219049, 828 },
      { 3585315836, 237 },
      { 3724004880, 360 },
      { 3931641900, 1550 },
      { 3955205564, 235 },
      { 4127308103, 216 },
      { kMarkvNoneOfTheAbove, 732 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpFDiv, 3), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 36016 },
      { 2320303498, 9812 },
      { 3334207724, 7076 },
      { kMarkvNoneOfTheAbove, 2646 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpVectorTimesScalar, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 49925544, 336 },
      { 256142724, 312 },
      { 364249359, 315 },
      { 386525753, 368 },
      { 417507386, 400 },
      { 470277359, 829 },
      { 497658126, 768 },
      { 559246409, 312 },
      { 679061455, 365 },
      { 815034111, 1364 },
      { 849867303, 768 },
      { 926966938, 300 },
      { 1004117804, 384 },
      { 1159301677, 511 },
      { 1177204104, 312 },
      { 1547894601, 280 },
      { 1558990974, 576 },
      { 1618754372, 528 },
      { 1752686878, 576 },
      { 1832728204, 384 },
      { 1837107871, 280 },
      { 1881707511, 312 },
      { 2004567202, 312 },
      { 2157491424, 280 },
      { 2158581495, 344 },
      { 2162274327, 768 },
      { 2179819210, 336 },
      { 2286296546, 384 },
      { 2306141594, 735 },
      { 2409582776, 312 },
      { 2445518906, 344 },
      { 2473053808, 530 },
      { 2504802016, 1596 },
      { 2568506610, 272 },
      { 2663821806, 336 },
      { 2745576697, 280 },
      { 2801333547, 768 },
      { 2937458041, 312 },
      { 3032677281, 1596 },
      { 3262572726, 468 },
      { 3275917447, 312 },
      { 3294118495, 346 },
      { 3307100165, 468 },
      { 3333014979, 336 },
      { 3351672873, 400 },
      { 3451465268, 336 },
      { 3462666495, 288 },
      { 3560552546, 1596 },
      { 3628821857, 280 },
      { 3641833815, 830 },
      { 3656700104, 344 },
      { 3715344675, 384 },
      { 3716588461, 400 },
      { 3748372419, 384 },
      { 3763445005, 336 },
      { 3797961332, 1605 },
      { 3803792712, 312 },
      { 3851276475, 336 },
      { 4248577718, 312 },
      { kMarkvNoneOfTheAbove, 2646 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpVectorTimesScalar, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 314809953, 468 },
      { 497658126, 768 },
      { 669323523, 365 },
      { 882718761, 280 },
      { 952793297, 300 },
      { 1004117804, 384 },
      { 1096668307, 280 },
      { 1191015885, 342 },
      { 1310404265, 480 },
      { 1325348861, 528 },
      { 1367301635, 2118 },
      { 1570165302, 967 },
      { 1608650199, 312 },
      { 1618544981, 768 },
      { 1646147798, 528 },
      { 1679946323, 528 },
      { 1740610993, 280 },
      { 1832728204, 384 },
      { 1898577041, 400 },
      { 1939930453, 400 },
      { 2023008475, 480 },
      { 2055836767, 550 },
      { 2096388952, 8700 },
      { 2123388694, 528 },
      { 2129301998, 480 },
      { 2221815776, 344 },
      { 2310211686, 300 },
      { 2378763734, 560 },
      { 2504743441, 280 },
      { 2622612602, 1110 },
      { 2660843182, 1737 },
      { 2698156268, 528 },
      { 2801333547, 768 },
      { 2895151306, 480 },
      { 2970183398, 1286 },
      { 3001444829, 480 },
      { 3133016299, 528 },
      { 3187066832, 8684 },
      { 3496407048, 351 },
      { 3536390697, 976 },
      { 3713290482, 816 },
      { 3715344675, 384 },
      { 3733149123, 280 },
      { 3748372419, 384 },
      { 3873587660, 480 },
      { 3882634684, 528 },
      { 4169226615, 768 },
      { 4243119782, 528 },
      { kMarkvNoneOfTheAbove, 2646 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpVectorTimesScalar, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 151810803, 576 },
      { 253329281, 936 },
      { 374446446, 336 },
      { 461040879, 936 },
      { 783918780, 768 },
      { 810488476, 2982 },
      { 824323032, 468 },
      { 870594305, 987 },
      { 959681532, 1100 },
      { 1128498881, 365 },
      { 1141965917, 727 },
      { 1204787336, 768 },
      { 1278818058, 336 },
      { 1372785527, 1718 },
      { 1428131230, 280 },
      { 1543672828, 400 },
      { 1603824970, 300 },
      { 1607552353, 336 },
      { 1635292159, 576 },
      { 1641070431, 672 },
      { 1684282922, 1477 },
      { 1781765116, 280 },
      { 1838763297, 504 },
      { 1901166356, 1676 },
      { 2011183308, 768 },
      { 2032069771, 768 },
      { 2071351379, 576 },
      { 2087004702, 720 },
      { 2244928358, 1718 },
      { 2332185394, 312 },
      { 2374216296, 368 },
      { 2443610186, 520 },
      { 2503770904, 336 },
      { 2526961521, 528 },
      { 2548929978, 384 },
      { 2805567641, 300 },
      { 2903897222, 344 },
      { 2990874939, 336 },
      { 3046948325, 384 },
      { 3117071189, 1721 },
      { 3188115516, 1920 },
      { 3417584874, 672 },
      { 3561482820, 768 },
      { 3733675151, 504 },
      { 3831290364, 457 },
      { 3895246551, 336 },
      { 3929248764, 5273 },
      { 4066056986, 384 },
      { 4092487128, 1920 },
      { 4167600590, 312 },
      { kMarkvNoneOfTheAbove, 2646 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpVectorTimesScalar, 3), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 679771963, 3873 },
      { 2320303498, 676 },
      { 3334207724, 1237 },
      { kMarkvNoneOfTheAbove, 290 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpVectorTimesMatrix, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 48436644, 36 },
      { 141989664, 56 },
      { 175557883, 56 },
      { 220963215, 56 },
      { 260854789, 83 },
      { 366042640, 120 },
      { 555268144, 528 },
      { 715031696, 58 },
      { 866465932, 168 },
      { 965107091, 36 },
      { 1008886329, 800 },
      { 1220568834, 104 },
      { 1309374657, 108 },
      { 1310074086, 104 },
      { 1697592748, 312 },
      { 1943501114, 120 },
      { 1970030385, 56 },
      { 2171063603, 96 },
      { 2429837795, 56 },
      { 2687889790, 56 },
      { 2737723643, 64 },
      { 2740962442, 604 },
      { 2780637092, 153 },
      { 2888846867, 312 },
      { 3109534682, 104 },
      { 3138114807, 104 },
      { 3186798019, 36 },
      { 3189747501, 104 },
      { 3251376223, 168 },
      { 3333392507, 168 },
      { 3406379873, 104 },
      { 3538736557, 161 },
      { 3687495409, 104 },
      { 3778669608, 59 },
      { 3898133019, 100 },
      { 4133079518, 49 },
      { 4164519323, 312 },
      { kMarkvNoneOfTheAbove, 290 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpVectorTimesMatrix, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 773136177, 56 },
      { 807276090, 604 },
      { 1299197852, 312 },
      { 1977038330, 307 },
      { 1978689945, 120 },
      { 2096388952, 106 },
      { 2519373000, 168 },
      { 2622612602, 358 },
      { 2758064548, 800 },
      { 3187066832, 624 },
      { 3232730629, 800 },
      { 3291262551, 56 },
      { 3296691317, 59 },
      { 3354880177, 40 },
      { 3570411982, 318 },
      { 3662491924, 104 },
      { 3713290482, 800 },
      { 3981986008, 104 },
      { kMarkvNoneOfTheAbove, 290 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpVectorTimesMatrix, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 228295080, 332 },
      { 527665290, 504 },
      { 633814697, 800 },
      { 1199433675, 40 },
      { 1320485014, 96 },
      { 1375043498, 936 },
      { 1650330230, 195 },
      { 1958114751, 168 },
      { 2024203744, 312 },
      { 2512104293, 312 },
      { 2524124676, 137 },
      { 3191000483, 528 },
      { 3409337475, 64 },
      { 3470815076, 674 },
      { 3471003517, 168 },
      { 3997038726, 382 },
      { 4071043660, 40 },
      { 4113156318, 83 },
      { kMarkvNoneOfTheAbove, 290 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpVectorTimesMatrix, 3), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 1951208733, 48636 },
      { kMarkvNoneOfTheAbove, 2432 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpDot, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 170690025, 1598 },
      { 192597017, 407 },
      { 206688607, 768 },
      { 276931161, 280 },
      { 461476226, 720 },
      { 790502615, 1019 },
      { 804806426, 280 },
      { 805072272, 528 },
      { 953089104, 384 },
      { 1044760482, 432 },
      { 1192184805, 247 },
      { 1331586698, 340 },
      { 1428131230, 280 },
      { 1664796690, 344 },
      { 2071351379, 2690 },
      { 2215759158, 280 },
      { 2222365618, 280 },
      { 2274779301, 344 },
      { 2332185394, 312 },
      { 2416108131, 916 },
      { 2525771779, 251 },
      { 2652773591, 280 },
      { 2919796598, 1356 },
      { 3029342296, 359 },
      { 3047649911, 782 },
      { 3104643263, 963 },
      { 3196248405, 407 },
      { 3198541202, 768 },
      { 3263841912, 300 },
      { 3334421268, 384 },
      { 3516147927, 280 },
      { 3622349409, 800 },
      { 3656012911, 294 },
      { 3714664910, 1019 },
      { 3765247327, 941 },
      { 3857562147, 268 },
      { 4044023192, 280 },
      { kMarkvNoneOfTheAbove, 2432 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpDot, 1), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 7139729, 320 },
      { 50385656, 1574 },
      { 151070081, 320 },
      { 181902171, 1356 },
      { 461861047, 400 },
      { 837715723, 1598 },
      { 886972033, 814 },
      { 983299427, 361 },
      { 1341913581, 320 },
      { 1367301635, 682 },
      { 1499923635, 1638 },
      { 1516384979, 280 },
      { 1570165302, 959 },
      { 1735295265, 526 },
      { 1766401548, 782 },
      { 1796311149, 351 },
      { 1980341560, 269 },
      { 2012838864, 1980 },
      { 2024071551, 1040 },
      { 2034948181, 320 },
      { 2096388952, 3875 },
      { 2178878533, 320 },
      { 2289183712, 468 },
      { 2477434291, 834 },
      { 2621255555, 852 },
      { 2622612602, 854 },
      { 2673174969, 432 },
      { 2872580757, 1266 },
      { 2881302403, 3451 },
      { 2970183398, 1784 },
      { 3187066832, 4802 },
      { 3224952074, 488 },
      { 3244383472, 521 },
      { 3256304404, 267 },
      { 3335250889, 344 },
      { 3362830643, 1560 },
      { 3609072425, 320 },
      { 3682213068, 468 },
      { 3783543823, 437 },
      { kMarkvNoneOfTheAbove, 2432 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpDot, 2), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 27177503, 511 },
      { 50385656, 784 },
      { 129748122, 518 },
      { 139011596, 695 },
      { 162608772, 492 },
      { 181902171, 800 },
      { 355531721, 287 },
      { 429023543, 384 },
      { 504514034, 832 },
      { 837715723, 1598 },
      { 861753115, 480 },
      { 875212982, 492 },
      { 899320334, 511 },
      { 1218218291, 384 },
      { 1347339159, 464 },
      { 1376656865, 831 },
      { 1522979646, 464 },
      { 1548491889, 528 },
      { 1570165302, 941 },
      { 1608650199, 312 },
      { 1735295265, 494 },
      { 1747355813, 492 },
      { 1766401548, 787 },
      { 1871105284, 518 },
      { 1978689945, 841 },
      { 1980341560, 269 },
      { 2024071551, 520 },
      { 2096388952, 3615 },
      { 2181030375, 480 },
      { 2395612122, 247 },
      { 2446960972, 300 },
      { 2517964682, 3486 },
      { 2564745684, 250 },
      { 2577387676, 768 },
      { 2604242419, 528 },
      { 2683080096, 1120 },
      { 2763960513, 407 },
      { 2817823941, 672 },
      { 2970183398, 1472 },
      { 2984459037, 528 },
      { 3015046341, 407 },
      { 3055195668, 832 },
      { 3187066832, 2702 },
      { 3193597927, 518 },
      { 3224258475, 480 },
      { 3335250889, 344 },
      { 3609540589, 511 },
      { 3619404941, 480 },
      { 3727034815, 768 },
      { 3742724777, 480 },
      { 3836179806, 811 },
      { 3862763081, 300 },
      { 3913885196, 1554 },
      { 3927338499, 832 },
      { 4032662899, 480 },
      { 4228303141, 1392 },
      { kMarkvNoneOfTheAbove, 2432 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpDot, 3), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 1036475267, 77328 },
      { kMarkvNoneOfTheAbove, 3867 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpLabel, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 1036475267, 839 },
      { kMarkvNoneOfTheAbove, 42 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpBranch, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 151106310, 264 },
      { 225077829, 170 },
      { 226729444, 171 },
      { 228381059, 167 },
      { 318865353, 216 },
      { 396722247, 319 },
      { 712410678, 396 },
      { 714062293, 198 },
      { 715713908, 198 },
      { 716745003, 163 },
      { 731672929, 179 },
      { 780132540, 144 },
      { 846740618, 216 },
      { 1264135136, 96 },
      { 1265790880, 83 },
      { 1627462327, 60 },
      { 1886643897, 96 },
      { 2213590260, 470 },
      { 2260180705, 144 },
      { 2362510635, 69 },
      { 2415213815, 52 },
      { 2797372854, 384 },
      { 2845438658, 346 },
      { 2998831773, 477 },
      { 3272058566, 96 },
      { 3376008946, 240 },
      { 3726639245, 373 },
      { 3809250715, 240 },
      { 3825297415, 360 },
      { 3995335475, 59 },
      { 4085957384, 216 },
      { 4129952079, 552 },
      { kMarkvNoneOfTheAbove, 500 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpBranchConditional, 0), std::move(codec));
  }

  {
    std::unique_ptr<HuffmanCodec<uint64_t>> codec(new HuffmanCodec<uint64_t>({
      { 57149555, 528 },
      { 255835594, 414 },
      { 330249537, 720 },
      { 388686774, 768 },
      { 508217552, 534 },
      { 559246409, 312 },
      { 599185303, 1596 },
      { 649208064, 414 },
      { 679061455, 365 },
      { 810488476, 1580 },
      { 951841533, 268 },
      { 1008886329, 800 },
      { 1022544883, 267 },
      { 1215030156, 576 },
      { 1453447304, 528 },
      { 1603937321, 1596 },
      { 1617826947, 336 },
      { 1643868273, 240 },
      { 1672607981, 264 },
      { 1755165354, 240 },
      { 1781864804, 312 },
      { 2096388952, 5643 },
      { 2204920111, 536 },
      { 2244470522, 1596 },
      { 2330636993, 2396 },
      { 2424848261, 336 },
      { 2603020391, 768 },
      { 2622612602, 767 },
      { 2645135839, 336 },
      { 2660843182, 1009 },
      { 2708915136, 1144 },
      { 2724166585, 1596 },
      { 2728667725, 528 },
      { 2890638791, 1596 },
      { 2941648648, 280 },
      { 2970183398, 440 },
      { 2998120306, 216 },
      { 3123244280, 871 },
      { 3209399506, 468 },
      { 3230260738, 344 },
      { 3345707173, 337 },
      { 3397078357, 242 },
      { 3569736966, 224 },
      { 3816961131, 828 },
      { 4091670162, 344 },
      { 4237497041, 308 },
      { kMarkvNoneOfTheAbove, 2014 },
    }));

    codecs.emplace(std::pair<uint32_t, uint32_t>(SpvOpReturnValue, 0), std::move(codec));
  }

  return codecs;
}
