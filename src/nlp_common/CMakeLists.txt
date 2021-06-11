add_library(nlp_common STATIC
    AwkInputStream.h AwkInputStream.cc
    BaseIncrNgramLM.h
    BaseNgramLM.h
    BidTrie.h
    Bitset.h
    Count.h
    ctimer.h ctimer.c
    ErrorDefs.h
    getdelim.h getdelim.c
    getline.h getline.c
    ins_op_pair.h
    LM_Defs.h
    LogCount.h
    lt_op_vec.h
    MathDefs.h
    MathFuncs.h MathFuncs.cc
    mem_alloc_utils.h mem_alloc_utils.cc
    NbestTableNode.h
    NbestTransTable.h
    options.h options.cc
    OrderedVector.h
    PositionIndex.h
    printAligFuncs.h printAligFuncs.cc
    Prob.h Prob.cc
    Score.h
    SingleWordVocab.h SingleWordVocab.cc
    SmtDefs.h
    StatModelDefs.h
    StrProcUtils.h StrProcUtils.cc
    TranslationData.h
    Trie.h
    TrieVecs.h
    uiHashF.h
    uiPairHashF.h
    WordAligMatrix.h WordAligMatrix.cc
    WordIndex.h
)
