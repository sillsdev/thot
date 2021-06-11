add_library(stack_dec STATIC
    _assistedTrans.h
    _nbUncoupledAssistedTrans.h
    _pbTransModel.h
    _phraseBasedTransModel.h
    _phraseHypothesis.h
    _phraseHypothesisRec.h
    _phrSwTransModel.h
    _smtModel.h
    _smtMultiStack.h
    _smtStack.h
    _stack_decoder_statistics.h
    _stackDecoder.h
    _stackDecoderRec.h
    BaseAssistedTrans.h
    BaseHypothesis.h
    BaseHypothesisRec.h
    BaseHypState.h BaseHypState.cc
    BaseLogLinWeightUpdater.h
    BaseMiraScorer.h
    BasePbTransModel.h
    BasePbTransModelFeature.h
    BasePbTransModelStats.h
    BasePhraseHypothesis.h
    BasePhraseHypothesisRec.h
    BaseScorer.h
    BaseSmtModel.h
    BaseSmtMultiStack.h
    BaseSmtStack.h
    BaseStackDecoder.h
    BaseTranslationMetadata.h
    bleu.h bleu.cc
    chrf.h chrf.cc
    DictFeat.h DictFeat.cc
    DirectPhraseModelFeat.h DirectPhraseModelFeat.cc
    FeaturesInfo.h
    HypDebugData.h
    HypSortCriterion.h
    HypStateDict.h
    HypStateDictData.h
    InversePhraseModelFeat.h InversePhraseModelFeat.cc
    KbMiraLlWu.h KbMiraLlWu.cc
    LangModelFeat.h LangModelFeat.cc
    LangModelInfo.h
    LangModelPars.h
    LangModelsInfo.h
    LM_State.h
    MiraBleu.h MiraBleu.cc
    MiraChrF.h MiraChrF.cc
    MiraGtm.h MiraGtm.cc
    MiraWer.h MiraWer.cc
    multi_stack_decoder_rec.h
    NbestTransCacheData.h
    NgramCacheTable.h
    OnlineTrainingPars.h
    OnTheFlyDictFeat.h OnTheFlyDictFeat.cc
    PbTransModel.h
    PbTransModelInputVars.h
    PbTransModelPars.h
    PhraseBasedTmHyp.h
    PhraseBasedTmHypRec.h
    PhraseCacheTable.h
    PhraseModelInfo.h
    PhraseModelPars.h
    PhrasePairCacheTable.h
    PhrHypData.h
    PhrHypDataStr.h
    PhrHypEqClassF.h PhrHypEqClassF.cc
    PhrHypNumcovJumps01EqClassF.h PhrHypNumcovJumps01EqClassF.cc
    PhrHypNumcovJumpsEqClassF.h PhrHypNumcovJumpsEqClassF.cc
    PhrHypState.h PhrHypState.cc
    PhrLocalSwLiTm.h PhrLocalSwLiTm.cc
    PhrLocalSwLiTmHypRec.h
    PhrNbestTransTable.h
    PhrNbestTransTablePref.h
    PhrNbestTransTablePrefKey.h PhrNbestTransTablePrefKey.cc
    PhrNbestTransTableRef.h
    PhrNbestTransTableRefKey.h PhrNbestTransTableRefKey.cc
    PhrScoreInfo.h PhrScoreInfo.cc
    ScoreCompDefs.h
    SmtModelUtils.h SmtModelUtils.cc
    SmtMultiStackRec.h
    SmtStack.h
    SourceSegmentation.h
    SrcPhraseLenFeat.h SrcPhraseLenFeat.cc
    SrcPosJumpFeat.h SrcPosJumpFeat.cc
    SwModelInfo.h
    SwModelPars.h
    SwModelsInfo.h
    TranslationMetadata.h
    TrgPhraseLenFeat.h TrgPhraseLenFeat.cc
    WeightUpdateUtils.h WeightUpdateUtils.cc
    WgUncoupledAssistedTrans.h
    WordPenaltyFeat.h WordPenaltyFeat.cc
    WpModelInfo.h
)

target_include_directories(stack_dec PUBLIC
    "${PROJECT_SOURCE_DIR}/src/downhill_simplex"
    "${PROJECT_SOURCE_DIR}/src/error_correction"
    "${PROJECT_SOURCE_DIR}/src/incr_models"
    "${PROJECT_SOURCE_DIR}/src/nlp_common"
    "${PROJECT_SOURCE_DIR}/src/phrase_models"
    "${PROJECT_SOURCE_DIR}/src/sw_models"
)
