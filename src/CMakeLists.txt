add_library(thot_lib OBJECT
    downhill_simplex/step_by_step_dhs.c
    downhill_simplex/step_by_step_dhs.h
    error_correction/_editDist.h
    error_correction/_editDistBasedEcm.cc
    error_correction/_editDistBasedEcm.h
    error_correction/BaseEcmForWg.h
    error_correction/BaseEcModelForNbUcat.h
    error_correction/BaseEditDist.h
    error_correction/BaseErrorCorrectionModel.cc
    error_correction/BaseErrorCorrectionModel.h
    error_correction/BaseWgProcessorForAnlp.h
    error_correction/EditDistForStr.cc
    error_correction/EditDistForStr.h
    error_correction/EditDistForVec.h
    error_correction/EditDistForVecString.cc
    error_correction/EditDistForVecString.h
    error_correction/HypStateIndex.h
    error_correction/NbestCorrections.h
    error_correction/NbSearchHighLevelHyp.h
    error_correction/NbSearchHyp.h
    error_correction/NbSearchStack.cc
    error_correction/NbSearchStack.h
    error_correction/NonPbEcModelForNbUcat.cc
    error_correction/NonPbEcModelForNbUcat.h
    error_correction/PfsmEcm.cc
    error_correction/PfsmEcm.h
    error_correction/PfsmEcmForWg.cc
    error_correction/PfsmEcmForWg.h
    error_correction/PfsmEcmForWgEsi.h
    error_correction/PrefAlignInfo.h
    error_correction/RejectedWordsSet.h
    error_correction/WgHandler.cc
    error_correction/WgHandler.h
    error_correction/WgProcessorForAnlp.h
    error_correction/WordAndCharLevelOps.h
    error_correction/WordGraph.cc
    error_correction/WordGraph.h
    error_correction/WordGraphArc.h
    error_correction/WordGraphArcId.h
    error_correction/WordGraphStateData.h
    incr_models/_incrEncCondProbModel.h
    incr_models/_incrJelMerNgramLM.h
    incr_models/_incrNgramLM.h
    incr_models/BaseIncrCondProbModel.h
    incr_models/BaseIncrCondProbTable.h
    incr_models/BaseIncrEncCondProbModel.h
    incr_models/BaseIncrEncoder.h
    incr_models/BaseWordPenaltyModel.h
    incr_models/im_pair.h
    incr_models/IncrCondProbTable.h
    incr_models/IncrEncoder.h
    incr_models/IncrJelMerNgramLM.cc
    incr_models/IncrJelMerNgramLM.h
    incr_models/IncrNgramLM.cc
    incr_models/IncrNgramLM.h
    incr_models/lm_ienc.cc
    incr_models/lm_ienc.h
    incr_models/vecx_x_incr_cptable.h
    incr_models/vecx_x_incr_ecpm.h
    incr_models/vecx_x_incr_enc.h
    incr_models/WordPenaltyModel.cc
    incr_models/WordPenaltyModel.h
    incr_models/WordPredictor.cc
    incr_models/WordPredictor.h
    nlp_common/AwkInputStream.cc
    nlp_common/AwkInputStream.h
    nlp_common/BaseIncrNgramLM.h
    nlp_common/BaseNgramLM.h
    nlp_common/BidTrie.h
    nlp_common/Bitset.h
    nlp_common/Count.h
    nlp_common/ctimer.c
    nlp_common/ctimer.h
    nlp_common/ErrorDefs.h
    nlp_common/Exceptions.h
    nlp_common/getdelim.c
    nlp_common/getdelim.h
    nlp_common/getline.c
    nlp_common/getline.h
    nlp_common/ins_op_pair.h
    nlp_common/LM_Defs.h
    nlp_common/LogCount.h
    nlp_common/lt_op_vec.h
    nlp_common/MathDefs.h
    nlp_common/MathFuncs.cc
    nlp_common/MathFuncs.h
    nlp_common/Matrix.h
    nlp_common/NbestTableNode.h
    nlp_common/NbestTransTable.h
    nlp_common/OrderedVector.h
    nlp_common/PositionIndex.h
    nlp_common/printAligFuncs.cc
    nlp_common/printAligFuncs.h
    nlp_common/Prob.cc
    nlp_common/Prob.h
    nlp_common/Score.h
    nlp_common/SingleWordVocab.cc
    nlp_common/SingleWordVocab.h
    nlp_common/SmtDefs.h
    nlp_common/StrProcUtils.cc
    nlp_common/StrProcUtils.h
    nlp_common/TranslationData.h
    nlp_common/Trie.h
    nlp_common/TrieVecs.h
    nlp_common/uiHashF.h
    nlp_common/uiPairHashF.h
    nlp_common/WordAlignmentMatrix.cc
    nlp_common/WordAlignmentMatrix.h
    nlp_common/WordClasses.cc
    nlp_common/WordClasses.h
    nlp_common/WordIndex.h
    phrase_models/_incrPhraseModel.cc
    phrase_models/_incrPhraseModel.h
    phrase_models/_wbaIncrPhraseModel.cc
    phrase_models/_wbaIncrPhraseModel.h
    phrase_models/AligInfo.h
    phrase_models/AlignmentContainer.cc
    phrase_models/AlignmentContainer.h
    phrase_models/AlignmentExtractor.cc
    phrase_models/AlignmentExtractor.h
    phrase_models/BaseCountPhraseModel.h
    phrase_models/BaseIncrPhraseModel.cc
    phrase_models/BaseIncrPhraseModel.h
    phrase_models/BasePhraseModel.cc
    phrase_models/BasePhraseModel.h
    phrase_models/BasePhrasePairFilter.h
    phrase_models/BasePhraseTable.h
    phrase_models/BpSet.cc
    phrase_models/BpSet.h
    phrase_models/BpSetInfo.h
    phrase_models/CategPhrasePairFilter.cc
    phrase_models/CategPhrasePairFilter.h
    phrase_models/CellAlignment.h
    phrase_models/CellID.h
    phrase_models/HatTriePhraseTable.cc
    phrase_models/HatTriePhraseTable.h
    phrase_models/IncrPhraseModel.cc
    phrase_models/IncrPhraseModel.h
    phrase_models/PhraseDefs.h
    phrase_models/PhraseExtractionCell.h
    phrase_models/PhraseExtractionTable.cc
    phrase_models/PhraseExtractionTable.h
    phrase_models/PhraseExtractParameters.h
    phrase_models/PhraseExtractUtils.cc
    phrase_models/PhraseExtractUtils.h
    phrase_models/PhraseId.h
    phrase_models/PhrasePair.h
    phrase_models/PhrasePairInfo.h
    phrase_models/PhraseSortCriterion.h
    phrase_models/PhraseTransTableNodeData.h
    phrase_models/SegLenTable.cc
    phrase_models/SegLenTable.h
    phrase_models/SentSegmentation.h
    phrase_models/SrcSegmLenTable.cc
    phrase_models/SrcSegmLenTable.h
    phrase_models/SrfBisegm.h
    phrase_models/SrfNodeInfo.h
    phrase_models/SrfNodeInfoMap.cc
    phrase_models/SrfNodeInfoMap.h
    phrase_models/SrfNodeKey.cc
    phrase_models/SrfNodeKey.h
    phrase_models/StlPhraseTable.cc
    phrase_models/StlPhraseTable.h
    phrase_models/StrictCategPhrasePairFilter.cc
    phrase_models/StrictCategPhrasePairFilter.h
    phrase_models/TrgCutsTable.cc
    phrase_models/TrgCutsTable.h
    phrase_models/TrgSegmLenTable.cc
    phrase_models/TrgSegmLenTable.h
    phrase_models/VecUnsignedIntSortCriterion.h
    phrase_models/WbaIncrPhraseModel.cc
    phrase_models/WbaIncrPhraseModel.h
    stack_dec/_assistedTrans.h
    stack_dec/_nbUncoupledAssistedTrans.h
    stack_dec/_pbTransModel.h
    stack_dec/_phraseBasedTransModel.h
    stack_dec/_phraseHypothesis.h
    stack_dec/_phraseHypothesisRec.h
    stack_dec/_phrSwTransModel.h
    stack_dec/_smtModel.h
    stack_dec/_smtMultiStack.h
    stack_dec/_smtStack.h
    stack_dec/_stack_decoder_statistics.h
    stack_dec/_stackDecoder.h
    stack_dec/_stackDecoderRec.h
    stack_dec/BaseAssistedTrans.h
    stack_dec/BaseHypothesis.h
    stack_dec/BaseHypothesisRec.h
    stack_dec/BaseHypState.cc
    stack_dec/BaseHypState.h
    stack_dec/BaseLogLinWeightUpdater.h
    stack_dec/BaseMiraScorer.h
    stack_dec/BasePbTransModel.h
    stack_dec/BasePbTransModelFeature.h
    stack_dec/BasePbTransModelStats.h
    stack_dec/BasePhraseHypothesis.h
    stack_dec/BasePhraseHypothesisRec.h
    stack_dec/BaseScorer.h
    stack_dec/BaseSmtModel.h
    stack_dec/BaseSmtMultiStack.h
    stack_dec/BaseSmtStack.h
    stack_dec/BaseStackDecoder.h
    stack_dec/BaseTranslationMetadata.h
    stack_dec/bleu.cc
    stack_dec/bleu.h
    stack_dec/chrf.cc
    stack_dec/chrf.h
    stack_dec/DictFeat.cc
    stack_dec/DictFeat.h
    stack_dec/DirectPhraseModelFeat.cc
    stack_dec/DirectPhraseModelFeat.h
    stack_dec/FeaturesInfo.h
    stack_dec/HypSortCriterion.h
    stack_dec/HypStateDict.h
    stack_dec/HypStateDictData.h
    stack_dec/InversePhraseModelFeat.cc
    stack_dec/InversePhraseModelFeat.h
    stack_dec/KbMiraLlWu.cc
    stack_dec/KbMiraLlWu.h
    stack_dec/LangModelFeat.cc
    stack_dec/LangModelFeat.h
    stack_dec/LangModelInfo.h
    stack_dec/LangModelPars.h
    stack_dec/LM_State.h
    stack_dec/MiraBleu.cc
    stack_dec/MiraBleu.h
    stack_dec/MiraChrF.cc
    stack_dec/MiraChrF.h
    stack_dec/MiraGtm.cc
    stack_dec/MiraGtm.h
    stack_dec/MiraWer.cc
    stack_dec/MiraWer.h
    stack_dec/multi_stack_decoder_rec.h
    stack_dec/NbestTransCacheData.h
    stack_dec/OnlineTrainingPars.h
    stack_dec/OnTheFlyDictFeat.cc
    stack_dec/OnTheFlyDictFeat.h
    stack_dec/PbTransModel.h
    stack_dec/PbTransModelInputVars.h
    stack_dec/PbTransModelPars.h
    stack_dec/PhraseBasedTmHyp.h
    stack_dec/PhraseBasedTmHypRec.h
    stack_dec/PhraseCacheTable.h
    stack_dec/PhraseModelInfo.h
    stack_dec/PhraseModelPars.h
    stack_dec/PhrasePairCacheTable.h
    stack_dec/PhrHypData.h
    stack_dec/PhrHypDataStr.h
    stack_dec/PhrHypEqClassF.cc
    stack_dec/PhrHypEqClassF.h
    stack_dec/PhrHypNumcovJumps01EqClassF.cc
    stack_dec/PhrHypNumcovJumps01EqClassF.h
    stack_dec/PhrHypNumcovJumpsEqClassF.cc
    stack_dec/PhrHypNumcovJumpsEqClassF.h
    stack_dec/PhrHypState.cc
    stack_dec/PhrHypState.h
    stack_dec/PhrLocalSwLiTm.cc
    stack_dec/PhrLocalSwLiTm.h
    stack_dec/PhrLocalSwLiTmHypRec.h
    stack_dec/PhrNbestTransTable.h
    stack_dec/PhrNbestTransTablePref.h
    stack_dec/PhrNbestTransTablePrefKey.cc
    stack_dec/PhrNbestTransTablePrefKey.h
    stack_dec/PhrNbestTransTableRef.h
    stack_dec/PhrNbestTransTableRefKey.cc
    stack_dec/PhrNbestTransTableRefKey.h
    stack_dec/PhrScoreInfo.cc
    stack_dec/PhrScoreInfo.h
    stack_dec/ScoreCompDefs.h
    stack_dec/SmtModelUtils.cc
    stack_dec/SmtModelUtils.h
    stack_dec/SmtMultiStackRec.h
    stack_dec/SmtStack.h
    stack_dec/SourceSegmentation.h
    stack_dec/SrcPhraseLenFeat.cc
    stack_dec/SrcPhraseLenFeat.h
    stack_dec/SrcPosJumpFeat.cc
    stack_dec/SrcPosJumpFeat.h
    stack_dec/SwModelInfo.h
    stack_dec/SwModelPars.h
    stack_dec/TranslationMetadata.h
    stack_dec/TrgPhraseLenFeat.cc
    stack_dec/TrgPhraseLenFeat.h
    stack_dec/WeightUpdateUtils.cc
    stack_dec/WeightUpdateUtils.h
    stack_dec/WgUncoupledAssistedTrans.h
    stack_dec/WordPenaltyFeat.cc
    stack_dec/WordPenaltyFeat.h
    sw_models/Aligner.h
    sw_models/AlignmentInfo.h
    sw_models/AlignmentModel.h
    sw_models/AlignmentModelBase.cc
    sw_models/AlignmentModelBase.h
    sw_models/AlignmentTable.cc
    sw_models/AlignmentTable.h
    sw_models/anjiMatrix.cc
    sw_models/anjiMatrix.h
    sw_models/anjm1ip_anjiMatrix.cc
    sw_models/anjm1ip_anjiMatrix.h
    sw_models/CachedHmmAligLgProb.cc
    sw_models/CachedHmmAligLgProb.h
    sw_models/DistortionTable.cc
    sw_models/DistortionTable.h
    sw_models/DoubleMatrix.cc
    sw_models/DoubleMatrix.h
    sw_models/FastAlignModel.cc
    sw_models/FastAlignModel.h
    sw_models/FertilityTable.cc
    sw_models/FertilityTable.h
    sw_models/HeadDistortionTable.cc
    sw_models/HeadDistortionTable.h
    sw_models/HmmAlignmentModel.cc
    sw_models/HmmAlignmentModel.h
    sw_models/HmmAlignmentTable.cc
    sw_models/HmmAlignmentTable.h
    sw_models/Ibm1AlignmentModel.cc
    sw_models/Ibm1AlignmentModel.h
    sw_models/Ibm2AlignmentModel.cc
    sw_models/Ibm2AlignmentModel.h
    sw_models/Ibm3AlignmentModel.cc
    sw_models/Ibm3AlignmentModel.h
    sw_models/Ibm4AlignmentModel.cc
    sw_models/Ibm4AlignmentModel.h
    sw_models/IncrAlignmentModel.h
    sw_models/IncrHmmAlignmentModel.cc
    sw_models/IncrHmmAlignmentModel.h
    sw_models/IncrHmmAlignmentTrainer.cc
    sw_models/IncrHmmAlignmentTrainer.h
    sw_models/IncrIbm1AlignmentModel.cc
    sw_models/IncrIbm1AlignmentModel.h
    sw_models/IncrIbm1AlignmentTrainer.cc
    sw_models/IncrIbm1AlignmentTrainer.h
    sw_models/IncrIbm2AlignmentModel.cc
    sw_models/IncrIbm2AlignmentModel.h
    sw_models/IncrIbm2AlignmentTrainer.cc
    sw_models/IncrIbm2AlignmentTrainer.h
    sw_models/LexCounts.h
    sw_models/LexTable.h
    sw_models/LightSentenceHandler.cc
    sw_models/LightSentenceHandler.h
    sw_models/MemoryLexTable.cc
    sw_models/MemoryLexTable.h
    sw_models/NonheadDistortionTable.cc
    sw_models/NonheadDistortionTable.h
    sw_models/NormalSentenceLengthModel.cc
    sw_models/NormalSentenceLengthModel.h
    sw_models/SentenceHandler.h
    sw_models/SentenceLengthModel.h
    sw_models/SentenceLengthModelBase.cc
    sw_models/SentenceLengthModelBase.h
    sw_models/StepwiseAlignmentModel.h
    sw_models/SwDefs.h
    sw_models/SymmetrizedAligner.cc
    sw_models/SymmetrizedAligner.h
  )

target_include_directories(thot_lib PUBLIC
    "${PROJECT_SOURCE_DIR}/include"
    "${PROJECT_SOURCE_DIR}/src"
)

target_compile_definitions(thot_lib PUBLIC
    THOT_USE_HAT_TRIE_PHRASE_TABLE
)

target_link_libraries(thot_lib PUBLIC
    OpenMP::OpenMP_CXX
    ${GMP_LIBRARIES}
    yaml-cpp
)

if(BUILD_PYTHON_MODULE)
    add_subdirectory(python_module)
endif()

if(BUILD_SHARED_LIBRARY)
    add_subdirectory(shared_library)
endif()
