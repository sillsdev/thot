#include "incr_models/IncrJelMerNgramLM.h"
#include "incr_models/WordPenaltyModel.h"
#include "nlp_common/ErrorDefs.h"
#include "stack_dec/PhrLocalSwLiTm.h"
#include "stack_dec/TranslationMetadata.h"
#include "stack_dec/multi_stack_decoder_rec.h"
#include "sw_models/Aligner.h"
#include "sw_models/AlignmentModel.h"
#include "sw_models/FastAlignModel.h"
#include "sw_models/HmmAlignmentModel.h"
#include "sw_models/Ibm1AlignmentModel.h"
#include "sw_models/Ibm2AlignmentModel.h"
#include "sw_models/Ibm3AlignmentModel.h"
#include "sw_models/Ibm4AlignmentModel.h"
#include "sw_models/IncrAlignmentModel.h"
#include "sw_models/IncrHmmAlignmentModel.h"
#include "sw_models/IncrIbm1AlignmentModel.h"
#include "sw_models/IncrIbm2AlignmentModel.h"
#include "sw_models/NormalSentenceLengthModel.h"
#include "sw_models/SentenceLengthModel.h"
#include "sw_models/SymmetrizedAligner.h"

#include <memory>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

std::vector<WordIndex> getSrcWordIndices(Aligner& aligner, const char* srcSentence)
{
  std::vector<WordIndex> wordIndices;
  size_t i = 0;
  std::string s;
  while (srcSentence[i] != 0)
  {
    s = "";
    while (srcSentence[i] == ' ' && srcSentence[i] != 0)
    {
      ++i;
    }
    while (srcSentence[i] != ' ' && srcSentence[i] != 0)
    {
      s = s + srcSentence[i];
      ++i;
    }
    if (s != "")
    {
      WordIndex wordIndex = aligner.stringToSrcWordIndex(s);
      wordIndices.push_back(wordIndex);
    }
  }
  return wordIndices;
}

std::vector<WordIndex> getSrcWordIndices(Aligner& aligner, const std::vector<std::string>& srcSentence)
{
  std::vector<WordIndex> wordIndices;
  for (auto& w : srcSentence)
    wordIndices.push_back(aligner.stringToSrcWordIndex(w));
  return wordIndices;
}

std::vector<WordIndex> getTrgWordIndices(Aligner& aligner, const char* trgSentence)
{
  std::vector<WordIndex> wordIndices;
  size_t i = 0;
  std::string s;
  while (trgSentence[i] != 0)
  {
    s = "";
    while (trgSentence[i] == ' ' && trgSentence[i] != 0)
    {
      ++i;
    }
    while (trgSentence[i] != ' ' && trgSentence[i] != 0)
    {
      s = s + trgSentence[i];
      ++i;
    }
    if (s != "")
    {
      WordIndex wordIndex = aligner.stringToTrgWordIndex(s);
      wordIndices.push_back(wordIndex);
    }
  }
  return wordIndices;
}

std::vector<WordIndex> getTrgWordIndices(Aligner& aligner, const std::vector<std::string>& trgSentence)
{
  std::vector<WordIndex> wordIndices;
  for (auto& w : trgSentence)
    wordIndices.push_back(aligner.stringToTrgWordIndex(w));
  return wordIndices;
}

AlignmentModel* createAlignmentModel(AlignmentModelType type)
{
  switch (type)
  {
  case AlignmentModelType::Ibm1:
    return new Ibm1AlignmentModel();
  case AlignmentModelType::Ibm2:
    return new Ibm2AlignmentModel();
  case AlignmentModelType::Hmm:
    return new HmmAlignmentModel();
  case AlignmentModelType::Ibm3:
    return new Ibm3AlignmentModel();
  case AlignmentModelType::Ibm4:
    return new Ibm4AlignmentModel();
  case AlignmentModelType::IncrIbm1:
    return new IncrIbm1AlignmentModel();
  case AlignmentModelType::IncrIbm2:
    return new IncrIbm2AlignmentModel();
  case AlignmentModelType::IncrHmm:
    return new IncrHmmAlignmentModel();
  case AlignmentModelType::FastAlign:
    return new FastAlignModel();
  }
  return nullptr;
}

TranslationData translate(multi_stack_decoder_rec<PhrLocalSwLiTm>& decoder, const std::string& sentence)
{
  PhrLocalSwLiTm::Hypothesis hyp = decoder.translate(sentence);

  TranslationData result;
  std::vector<std::pair<PositionIndex, PositionIndex>> amatrix;
  decoder.getSmtModel()->aligMatrix(hyp, amatrix);
  decoder.getSmtModel()->getPhraseAlignment(amatrix, result.sourceSegmentation, result.targetSegmentCuts);
  result.target = decoder.getSmtModel()->getTransInPlainTextVec(hyp, result.targetUnknownWords);
  result.score = decoder.getSmtModel()->getScoreForHyp(hyp);
  result.scoreComponents = decoder.getSmtModel()->scoreCompsForHyp(hyp);
  return result;
}

std::vector<TranslationData> translateN(multi_stack_decoder_rec<PhrLocalSwLiTm>& decoder, const std::string& sentence,
                                        int n)
{
  decoder.enableWordGraph();

  // Use translator
  decoder.translate(sentence);
  WordGraph* wg = decoder.getWordGraphPtr();

  decoder.disableWordGraph();

  std::vector<TranslationData> translations;
  wg->obtainNbestList(n, translations);
  return translations;
}

PYBIND11_MODULE(thot, m)
{
  py::module common = m.def_submodule("common");

  py::class_<WordAlignmentMatrix>(common, "WordAlignmentMatrix")
      .def(py::init())
      .def(py::init<unsigned int, unsigned int>(), py::arg("row_length"), py::arg("column_length"))
      .def_property_readonly("row_length", &WordAlignmentMatrix::get_I)
      .def_property_readonly("column_length", &WordAlignmentMatrix::get_J)
      .def("init", &WordAlignmentMatrix::init, py::arg("row_length"), py::arg("column_length"))
      .def(
          "__getitem__",
          [](const WordAlignmentMatrix& matrix, std::tuple<unsigned int, unsigned int> key) {
            return matrix.getValue(std::get<0>(key), std::get<1>(key));
          },
          py::arg("key"))
      .def(
          "__setitem__",
          [](WordAlignmentMatrix& matrix, std::tuple<unsigned int, unsigned int> key, bool value) {
            matrix.setValue(std::get<0>(key), std::get<1>(key), value);
          },
          py::arg("key"), py::arg("value"))
      .def("clear", &WordAlignmentMatrix::clear)
      .def("reset_all", &WordAlignmentMatrix::reset)
      .def("set_all", static_cast<void (WordAlignmentMatrix::*)()>(&WordAlignmentMatrix::set))
      .def("set", static_cast<void (WordAlignmentMatrix::*)(unsigned int, unsigned int)>(&WordAlignmentMatrix::set),
           py::arg("i"), py::arg("j"))
      .def("put_list", &WordAlignmentMatrix::putAligVec, py::arg("list"))
      .def("to_list",
           [](const WordAlignmentMatrix& matrix) {
             std::vector<PositionIndex> alignment;
             matrix.getAligVec(alignment);
             return alignment;
           })
      .def("transpose", &WordAlignmentMatrix::transpose)
      .def("flip", &WordAlignmentMatrix::flip)
      .def(
          "intersect", [](WordAlignmentMatrix& matrix, const WordAlignmentMatrix& other) { return matrix &= other; },
          py::arg("other"))
      .def(
          "union", [](WordAlignmentMatrix& matrix, const WordAlignmentMatrix& other) { return matrix |= other; },
          py::arg("other"))
      .def(
          "xor", [](WordAlignmentMatrix& matrix, const WordAlignmentMatrix& other) { return matrix ^= other; },
          py::arg("other"))
      .def(
          "add", [](WordAlignmentMatrix& matrix, const WordAlignmentMatrix& other) { return matrix += other; },
          py::arg("other"))
      .def(
          "subtract", [](WordAlignmentMatrix& matrix, const WordAlignmentMatrix& other) { return matrix -= other; },
          py::arg("other"))
      .def("och_symmetrize", &WordAlignmentMatrix::symmetr1, py::arg("other"))
      .def("priority_symmetrize", &WordAlignmentMatrix::symmetr2, py::arg("other"))
      .def("grow", &WordAlignmentMatrix::grow, py::arg("other"))
      .def("grow_diag", &WordAlignmentMatrix::growDiag, py::arg("other"))
      .def("grow_diag_final", &WordAlignmentMatrix::growDiagFinal, py::arg("other"))
      .def("grow_diag_final_and", &WordAlignmentMatrix::growDiagFinalAnd, py::arg("other"))
      .def(
          "__eq__", [](const WordAlignmentMatrix& matrix, const WordAlignmentMatrix& other) { return matrix == other; },
          py::arg("other"))
      .def("to_numpy", [](WordAlignmentMatrix& matrix) {
        return py::array_t<bool>({matrix.get_I(), matrix.get_J()}, matrix.ptr() == nullptr ? nullptr : matrix.ptr()[0]);
      });

  py::class_<IncrJelMerNgramLM>(common, "NGramLanguageModel")
      .def(py::init())
      .def(
          "load", [](IncrJelMerNgramLM& model, const char* filename) { return model.load(filename) == THOT_OK; },
          py::arg("filename"))
      .def(
          "get_sentence_log_probability",
          [](IncrJelMerNgramLM& model, const std::vector<std::string>& sentence) {
            return (double)model.getSentenceLog10ProbStr(sentence);
          },
          py::arg("sentence"))
      .def("clear", &IncrJelMerNgramLM::clear);

  py::module alignment = m.def_submodule("alignment");

  py::class_<Aligner, std::shared_ptr<Aligner>>(alignment, "Aligner")
      .def("get_src_word_index", &Aligner::stringToSrcWordIndex, py::arg("word"))
      .def("get_trg_word_index", &Aligner::stringToTrgWordIndex, py::arg("word"))
      .def(
          "get_best_alignment",
          [](Aligner& aligner, const char* srcSentence, const char* trgSentence) {
            WordAlignmentMatrix waMatrix;
            LgProb logProb = aligner.getBestAlignment(getSrcWordIndices(aligner, srcSentence),
                                                      getTrgWordIndices(aligner, trgSentence), waMatrix);
            return std::make_tuple((double)logProb, std::move(waMatrix));
          },
          py::arg("src_sentence"), py::arg("trg_sentence"))
      .def(
          "get_best_alignment",
          [](Aligner& aligner, const std::vector<std::string>& srcSentence,
             const std::vector<std::string>& trgSentence) {
            WordAlignmentMatrix waMatrix;
            LgProb logProb = aligner.getBestAlignment(getSrcWordIndices(aligner, srcSentence),
                                                      getTrgWordIndices(aligner, trgSentence), waMatrix);
            return std::make_tuple((double)logProb, std::move(waMatrix));
          },
          py::arg("src_sentence"), py::arg("trg_sentence"))
      .def(
          "get_best_alignment",
          [](Aligner& aligner, const std::vector<WordIndex>& srcSentence, const std::vector<WordIndex>& trgSentence) {
            WordAlignmentMatrix waMatrix;
            LgProb logProb = aligner.getBestAlignment(srcSentence, trgSentence, waMatrix);
            return std::make_tuple((double)logProb, std::move(waMatrix));
          },
          py::arg("src_sentence"), py::arg("trg_sentence"))
      .def(
          "get_best_alignments",
          [](Aligner& aligner, const std::vector<std::vector<std::string>>& srcSentences,
             const std::vector<std::vector<std::string>>& trgSentences) {
            std::vector<std::tuple<double, WordAlignmentMatrix>> alignments(srcSentences.size());
#pragma omp parallel for schedule(dynamic)
            for (int i = 0; i < (int)srcSentences.size(); ++i)
            {
              WordAlignmentMatrix waMatrix;
              LgProb logProb = aligner.getBestAlignment(getSrcWordIndices(aligner, srcSentences[i]),
                                                        getTrgWordIndices(aligner, trgSentences[i]), waMatrix);
              alignments[i] = std::make_tuple((double)logProb, std::move(waMatrix));
            }
            return alignments;
          },
          py::arg("src_sentences"), py::arg("trg_sentences"))
      .def(
          "get_best_alignments",
          [](Aligner& aligner, const std::vector<std::vector<WordIndex>>& srcSentences,
             const std::vector<std::vector<WordIndex>>& trgSentences) {
            std::vector<std::tuple<double, WordAlignmentMatrix>> alignments(srcSentences.size());
#pragma omp parallel for schedule(dynamic)
            for (int i = 0; i < (int)srcSentences.size(); ++i)
            {
              WordAlignmentMatrix waMatrix;
              LgProb logProb = aligner.getBestAlignment(srcSentences[i], trgSentences[i], waMatrix);
              alignments[i] = std::make_tuple((double)logProb, std::move(waMatrix));
            }
            return alignments;
          },
          py::arg("src_sentences"), py::arg("trg_sentences"));

  py::enum_<SymmetrizationHeuristic>(alignment, "SymmetrizationHeuristic")
      .value("NONE", SymmetrizationHeuristic::None)
      .value("UNION", SymmetrizationHeuristic::Union)
      .value("INTERSECTION", SymmetrizationHeuristic::Intersection)
      .value("OCH", SymmetrizationHeuristic::Och)
      .value("GROW", SymmetrizationHeuristic::Grow)
      .value("GROW_DIAG", SymmetrizationHeuristic::GrowDiag)
      .value("GROW_DIAG_FINAL", SymmetrizationHeuristic::GrowDiagFinal)
      .value("GROW_DIAG_FINAL_AND", SymmetrizationHeuristic::GrowDiagFinalAnd);

  py::class_<SymmetrizedAligner, Aligner, std::shared_ptr<SymmetrizedAligner>>(alignment, "SymmetrizedAligner")
      .def(py::init<std::shared_ptr<Aligner>, std::shared_ptr<Aligner>>(), py::arg("direct_aligner"),
           py::arg("inverse_aligner"))
      .def_property("heuristic", &SymmetrizedAligner::getHeuristic, &SymmetrizedAligner::setHeuristic);

  py::enum_<AlignmentModelType>(alignment, "AlignmentModelType")
      .value("IBM1", AlignmentModelType::Ibm1)
      .value("IBM2", AlignmentModelType::Ibm2)
      .value("HMM", AlignmentModelType::Hmm)
      .value("IBM3", AlignmentModelType::Ibm3)
      .value("IBM4", AlignmentModelType::Ibm4)
      .value("FAST_ALIGN", AlignmentModelType::FastAlign)
      .value("INCR_IBM1", AlignmentModelType::IncrIbm1)
      .value("INCR_IBM2", AlignmentModelType::IncrIbm2)
      .value("INCR_HMM", AlignmentModelType::IncrHmm);

  py::class_<AlignmentModel, Aligner, std::shared_ptr<AlignmentModel>>(alignment, "AlignmentModel")
      .def_property_readonly("model_type", &AlignmentModel::getModelType)
      .def_property("variational_bayes", &AlignmentModel::getVariationalBayes, &AlignmentModel::setVariationalBayes)
      .def(
          "read_sentence_pairs",
          [](AlignmentModel& model, const char* srcFileName, const char* trgFileName, const char* sentCountsFile) {
            std::pair<unsigned int, unsigned int> sentRange;
            model.readSentencePairs(srcFileName, trgFileName, sentCountsFile == nullptr ? "" : sentCountsFile,
                                    sentRange);
          },
          py::arg("src_filename"), py::arg("trg_filename"), py::arg("counts_filename") = nullptr)
      .def(
          "add_sentence_pair",
          [](AlignmentModel& model, const std::vector<std::string>& srcSentence,
             const std::vector<std::string>& trgSentence,
             float c) { return model.addSentencePair(srcSentence, trgSentence, c); },
          py::arg("src_sentence"), py::arg("trg_sentence"), py::arg("count") = 1)
      .def_property_readonly("num_sentence_pairs", &AlignmentModel::numSentencePairs)
      .def(
          "get_sentence_pair",
          [](AlignmentModel& model, unsigned int n) {
            std::vector<std::string> srcSentence, trgSentence;
            Count c;
            if (model.getSentencePair(n, srcSentence, trgSentence, c) == THOT_ERROR)
              throw std::out_of_range("The sentence pair index is out of range.");
            return std::make_tuple(srcSentence, trgSentence, (float)c);
          },
          py::arg("n"))
      .def_property_readonly("max_sentence_length", &AlignmentModel::getMaxSentenceLength)
      .def("start_training", [](AlignmentModel& model) { return model.startTraining(); })
      .def("train", [](AlignmentModel& model) { model.train(); })
      .def("end_training", &AlignmentModel::endTraining)
      .def(
          "sentence_length_prob",
          [](AlignmentModel& model, unsigned int slen, unsigned int tlen) {
            return (double)model.sentenceLengthProb(slen, tlen);
          },
          py::arg("src_length"), py::arg("trg_length"))
      .def(
          "sentence_length_log_prob",
          [](AlignmentModel& model, unsigned int slen, unsigned int tlen) {
            return (double)model.sentenceLengthLogProb(slen, tlen);
          },
          py::arg("src_length"), py::arg("trg_length"))
      .def(
          "load", [](AlignmentModel& model, const char* prefFileName) { return model.load(prefFileName) == THOT_OK; },
          py::arg("prefix_filename"))
      .def(
          "print", [](AlignmentModel& model, const char* prefFileName) { return model.print(prefFileName) == THOT_OK; },
          py::arg("prefix_filename"))
      .def_property_readonly("src_vocab_size", &AlignmentModel::getSrcVocabSize)
      .def("get_src_word", &AlignmentModel::wordIndexToSrcString, py::arg("word_index"))
      .def("src_word_exists", &AlignmentModel::existSrcSymbol, py::arg("word"))
      .def("add_src_word", &AlignmentModel::addSrcSymbol, py::arg("word"))
      .def_property_readonly("trg_vocab_size", &AlignmentModel::getTrgVocabSize)
      .def("get_trg_word", &AlignmentModel::wordIndexToTrgString, py::arg("word_index"))
      .def("trg_word_exists", &AlignmentModel::existTrgSymbol, py::arg("word"))
      .def("add_trg_word", &AlignmentModel::addTrgSymbol, py::arg("word"))
      .def(
          "get_translations",
          [](AlignmentModel& model, WordIndex s, double threshold) {
            NbestTableNode<WordIndex> targetWords;
            model.getEntriesForSource(s, targetWords);
            std::vector<std::tuple<WordIndex, double>> targetWordsVec;
            for (NbestTableNode<WordIndex>::iterator iter = targetWords.begin(); iter != targetWords.end(); ++iter)
              if (iter->first >= threshold)
                targetWordsVec.push_back(std::make_tuple(iter->second, iter->first));
            return targetWordsVec;
          },
          py::arg("s"), py::arg("threshold") = 0)
      .def("clear", &AlignmentModel::clear)
      .def(
          "translation_prob",
          [](AlignmentModel& model, WordIndex s, WordIndex t) { return (double)model.translationProb(s, t); },
          py::arg("src_word_index"), py::arg("trg_word_index"))
      .def(
          "translation_log_prob",
          [](AlignmentModel& model, WordIndex s, WordIndex t) { return (double)model.translationLogProb(s, t); },
          py::arg("src_word_index"), py::arg("trg_word_index"))
      .def(
          "map_src_word_to_word_class",
          [](AlignmentModel& model, const std::string& word, const std::string& wordClass) {
            model.mapSrcWordToWordClass(model.addSrcSymbol(word), wordClass);
          },
          py::arg("word"), py::arg("word_class"))
      .def(
          "map_trg_word_to_word_class",
          [](AlignmentModel& model, const std::string& word, const std::string& wordClass) {
            model.mapTrgWordToWordClass(model.addTrgSymbol(word), wordClass);
          },
          py::arg("word"), py::arg("word_class"));

  py::class_<IncrAlignmentModel, AlignmentModel, std::shared_ptr<IncrAlignmentModel>>(alignment, "IncrAlignmentModel")
      .def(
          "start_incr_training",
          [](IncrAlignmentModel& model, std::pair<unsigned int, unsigned int> sentPairRange) {
            model.startIncrTraining(sentPairRange);
          },
          py::arg("sentence_pair_range"))
      .def(
          "incr_train",
          [](IncrAlignmentModel& model, std::pair<unsigned int, unsigned int> sentPairRange) {
            model.incrTrain(sentPairRange);
          },
          py::arg("sentence_pair_range"))
      .def("end_incr_training", &IncrAlignmentModel::endIncrTraining);

  py::class_<Ibm1AlignmentModel, AlignmentModel, std::shared_ptr<Ibm1AlignmentModel>>(alignment, "Ibm1AlignmentModel")
      .def(py::init());

  py::class_<IncrIbm1AlignmentModel, Ibm1AlignmentModel, IncrAlignmentModel, std::shared_ptr<IncrIbm1AlignmentModel>>(
      alignment, "IncrIbm1AlignmentModel")
      .def(py::init());

  py::class_<Ibm2AlignmentModel, Ibm1AlignmentModel, std::shared_ptr<Ibm2AlignmentModel>>(alignment,
                                                                                          "Ibm2AlignmentModel")
      .def(py::init())
      .def(py::init<Ibm1AlignmentModel&>(), py::arg("model"))
      .def_property("compact_alignment_table", &Ibm2AlignmentModel::getCompactAlignmentTable,
                    &Ibm2AlignmentModel::setCompactAlignmentTable)
      .def(
          "alignment_prob",
          [](Ibm2AlignmentModel& model, PositionIndex j, PositionIndex slen, PositionIndex tlen, PositionIndex i) {
            return (double)model.alignmentProb(j, slen, tlen, i);
          },
          py::arg("j"), py::arg("src_length"), py::arg("trg_length"), py::arg("i"))
      .def(
          "alignment_log_prob",
          [](Ibm2AlignmentModel& model, PositionIndex j, PositionIndex slen, PositionIndex tlen, PositionIndex i) {
            return (double)model.alignmentLogProb(j, slen, tlen, i);
          },
          py::arg("j"), py::arg("src_length"), py::arg("trg_length"), py::arg("i"));

  py::class_<IncrIbm2AlignmentModel, Ibm2AlignmentModel, IncrAlignmentModel, std::shared_ptr<IncrIbm2AlignmentModel>>(
      alignment, "IncrIbm2AlignmentModel")
      .def(py::init());

  py::class_<HmmAlignmentModel, Ibm2AlignmentModel, std::shared_ptr<HmmAlignmentModel>>(alignment, "HmmAlignmentModel")
      .def(py::init())
      .def(py::init<Ibm1AlignmentModel&>(), py::arg("model"))
      .def_property(
          "hmm_p0", [](HmmAlignmentModel& model) { return double{model.getHmmP0()}; },
          [](HmmAlignmentModel& model, double p0) { model.setHmmP0(p0); })
      .def_property("lexical_smoothing_factor", &HmmAlignmentModel::getLexicalSmoothFactor,
                    &HmmAlignmentModel::setLexicalSmoothFactor)
      .def_property("hmm_alignment_smoothing_factor", &HmmAlignmentModel::getHmmAlignmentSmoothFactor,
                    &HmmAlignmentModel::setHmmAlignmentSmoothFactor)
      .def(
          "hmm_alignment_prob",
          [](HmmAlignmentModel& model, PositionIndex prev_i, PositionIndex slen, PositionIndex i) {
            return (double)model.hmmAlignmentProb(prev_i, slen, i);
          },
          py::arg("prev_i"), py::arg("src_length"), py::arg("i"))
      .def(
          "hmm_alignment_log_prob",
          [](HmmAlignmentModel& model, PositionIndex prev_i, PositionIndex slen, PositionIndex i) {
            return (double)model.hmmAlignmentLogProb(prev_i, slen, i);
          },
          py::arg("prev_i"), py::arg("src_length"), py::arg("i"));

  py::class_<IncrHmmAlignmentModel, HmmAlignmentModel, IncrAlignmentModel, std::shared_ptr<IncrHmmAlignmentModel>>(
      alignment, "IncrHmmAlignmentModel")
      .def(py::init());

  py::class_<FastAlignModel, IncrAlignmentModel, std::shared_ptr<FastAlignModel>>(alignment, "FastAlignModel",
                                                                                  py::multiple_inheritance())
      .def(py::init())
      .def_property(
          "fast_align_p0", [](FastAlignModel& model) { return double{model.getFastAlignP0()}; },
          [](FastAlignModel& model, double p0) { model.setFastAlignP0(p0); })
      .def(
          "alignment_prob",
          [](FastAlignModel& model, PositionIndex j, PositionIndex slen, PositionIndex tlen, PositionIndex i) {
            return (double)model.alignmentProb(j, slen, tlen, i);
          },
          py::arg("j"), py::arg("src_length"), py::arg("trg_length"), py::arg("i"))
      .def(
          "alignment_log_prob",
          [](FastAlignModel& model, PositionIndex j, PositionIndex slen, PositionIndex tlen, PositionIndex i) {
            return (double)model.alignmentLogProb(j, slen, tlen, i);
          },
          py::arg("j"), py::arg("src_length"), py::arg("trg_length"), py::arg("i"));

  py::class_<Ibm3AlignmentModel, Ibm2AlignmentModel, std::shared_ptr<Ibm3AlignmentModel>>(alignment,
                                                                                          "Ibm3AlignmentModel")
      .def(py::init())
      .def(py::init<Ibm2AlignmentModel&>(), py::arg("model"))
      .def(py::init<HmmAlignmentModel&>(), py::arg("model"))
      .def_property("fertility_smoothing_factor", &Ibm3AlignmentModel::getFertilitySmoothFactor,
                    &Ibm3AlignmentModel::setFertilitySmoothFactor)
      .def_property("count_threshold", &Ibm3AlignmentModel::getCountThreshold, &Ibm3AlignmentModel::setCountThreshold)
      .def(
          "distortion_prob",
          [](Ibm3AlignmentModel& model, PositionIndex i, PositionIndex slen, PositionIndex tlen, PositionIndex j) {
            return (double)model.distortionProb(i, slen, tlen, j);
          },
          py::arg("i"), py::arg("src_length"), py::arg("trg_length"), py::arg("j"))
      .def(
          "distortion_log_prob",
          [](Ibm3AlignmentModel& model, PositionIndex i, PositionIndex slen, PositionIndex tlen, PositionIndex j) {
            return (double)model.distortionLogProb(i, slen, tlen, j);
          },
          py::arg("i"), py::arg("src_length"), py::arg("trg_length"), py::arg("j"))
      .def(
          "fertility_prob",
          [](Ibm3AlignmentModel& model, WordIndex s, PositionIndex phi) { return (double)model.fertilityProb(s, phi); },
          py::arg("src_word_index"), py::arg("fertility"))
      .def(
          "fertility_log_prob",
          [](Ibm3AlignmentModel& model, WordIndex s, PositionIndex phi) {
            return (double)model.fertilityLogProb(s, phi);
          },
          py::arg("src_word_index"), py::arg("fertility"));

  py::class_<Ibm4AlignmentModel, Ibm3AlignmentModel, std::shared_ptr<Ibm4AlignmentModel>>(alignment,
                                                                                          "Ibm4AlignmentModel")
      .def(py::init())
      .def(py::init<HmmAlignmentModel&>(), py::arg("model"))
      .def(py::init<Ibm3AlignmentModel&>(), py::arg("model"))
      .def_property("distortion_smoothing_factor", &Ibm4AlignmentModel::getDistortionSmoothFactor,
                    &Ibm4AlignmentModel::setDistortionSmoothFactor)
      .def(
          "head_distortion_prob",
          [](Ibm4AlignmentModel& model, WordClassIndex srcWordClass, WordClassIndex trgWordClass, PositionIndex tlen,
             int dj) { return (double)model.headDistortionProb(srcWordClass, trgWordClass, tlen, dj); },
          py::arg("src_word_class"), py::arg("trg_word_class"), py::arg("trg_length"), py::arg("dj"))
      .def(
          "head_distortion_log_prob",
          [](Ibm4AlignmentModel& model, WordClassIndex src_word_class, WordClassIndex trg_word_class,
             PositionIndex tlen,
             int dj) { return (double)model.headDistortionLogProb(src_word_class, trg_word_class, tlen, dj); },
          py::arg("src_word_class"), py::arg("trg_word_class"), py::arg("trg_length"), py::arg("dj"))
      .def(
          "nonhead_distortion_prob",
          [](Ibm4AlignmentModel& model, WordClassIndex trg_word_class, PositionIndex tlen, int dj) {
            return (double)model.nonheadDistortionProb(trg_word_class, tlen, dj);
          },
          py::arg("trg_word_class"), py::arg("trg_length"), py::arg("dj"))
      .def(
          "nonhead_distortion_log_prob",
          [](Ibm4AlignmentModel& model, WordClassIndex trg_word_class, PositionIndex tlen, int dj) {
            return (double)model.nonheadDistortionLogProb(trg_word_class, tlen, dj);
          },
          py::arg("trg_word_class"), py::arg("trg_length"), py::arg("dj"));

  py::class_<SentenceLengthModel, std::shared_ptr<SentenceLengthModel>>(alignment, "SentenceLengthModel")
      .def(
          "sentence_length_prob",
          [](SentenceLengthModel& model, unsigned int slen, unsigned int tlen) {
            return (double)model.sentenceLengthProb(slen, tlen);
          },
          py::arg("src_length"), py::arg("trg_length"))
      .def(
          "sentence_length_log_prob",
          [](SentenceLengthModel& model, unsigned int slen, unsigned int tlen) {
            return (double)model.sentenceLengthLogProb(slen, tlen);
          },
          py::arg("src_length"), py::arg("trg_length"))
      .def(
          "train_sentence_pair",
          [](SentenceLengthModel& model, const std::vector<std::string>& srcSentence,
             const std::vector<std::string>& trgSentence,
             float c) { model.trainSentencePair(srcSentence, trgSentence, c); },
          py::arg("src_sentence"), py::arg("trg_sentence"), py::arg("count") = 1)
      .def(
          "train_sentence_pair",
          [](SentenceLengthModel& model, unsigned int slen, unsigned int tlen, float c) {
            model.trainSentencePair(slen, tlen, c);
          },
          py::arg("src_length"), py::arg("trg_length"), py::arg("count") = 1)
      .def(
          "load", [](SentenceLengthModel& model, const char* fileName) { return model.load(fileName) == THOT_OK; },
          py::arg("filename"))
      .def(
          "print", [](SentenceLengthModel& model, const char* fileName) { return model.print(fileName) == THOT_OK; },
          py::arg("filename"))
      .def("clear", &SentenceLengthModel::clear);

  py::class_<NormalSentenceLengthModel, SentenceLengthModel, std::shared_ptr<NormalSentenceLengthModel>>(
      alignment, "NormalSentenceLengthModel")
      .def(py::init());

  py::module translation = m.def_submodule("translation");

  py::class_<OnlineTrainingPars>(translation, "OnlineTrainingParameters")
      .def(py::init())
      .def_readwrite("algorithm", &OnlineTrainingPars::onlineLearningAlgorithm)
      .def_readwrite("learning_rate_policy", &OnlineTrainingPars::learningRatePolicy)
      .def_readwrite("learn_step_size", &OnlineTrainingPars::learnStepSize)
      .def_readwrite("em_iters", &OnlineTrainingPars::emIters)
      .def_readwrite("e", &OnlineTrainingPars::E_par)
      .def_readwrite("r", &OnlineTrainingPars::R_par);

  py::class_<PhrLocalSwLiTm>(translation, "SmtModel")
      .def(py::init([](AlignmentModelType modelType) {
             auto model = new PhrLocalSwLiTm;

             auto langModelInfo = new LangModelInfo;
             auto phrModelInfo = new PhraseModelInfo;
             auto swModelInfo = new SwModelInfo;

             phrModelInfo->phraseModelPars.ptsWeightVec.push_back(DEFAULT_PTS_WEIGHT);
             phrModelInfo->phraseModelPars.pstWeightVec.push_back(DEFAULT_PST_WEIGHT);

             langModelInfo->wpModel.reset(new WordPenaltyModel);
             langModelInfo->langModel.reset(new IncrJelMerNgramLM);

             phrModelInfo->invPhraseModel.reset(new WbaIncrPhraseModel);

             swModelInfo->swAligModels.push_back(std::shared_ptr<AlignmentModel>(createAlignmentModel(modelType)));
             swModelInfo->invSwAligModels.push_back(std::shared_ptr<AlignmentModel>(createAlignmentModel(modelType)));

             model->setLangModelInfo(langModelInfo);
             model->setPhraseModelInfo(phrModelInfo);
             model->setSwModelInfo(swModelInfo);
             model->setTranslationMetadata(new TranslationMetadata<PhrScoreInfo>);
             return model;
           }),
           py::arg("model_type"))
      .def(
          "load_translation_model",
          [](PhrLocalSwLiTm& model, const char* prefFileName) { return model.loadAligModel(prefFileName) == THOT_OK; },
          py::arg("prefix_filename"))
      .def(
          "load_language_model",
          [](PhrLocalSwLiTm& model, const char* prefFileName) { return model.loadLangModel(prefFileName) == THOT_OK; },
          py::arg("prefix_filename"))
      .def("clear", &PhrLocalSwLiTm::clear)
      .def_property("non_monotonicity", &PhrLocalSwLiTm::get_U_par, &PhrLocalSwLiTm::set_U_par)
      .def_property("w", &PhrLocalSwLiTm::get_W_par, &PhrLocalSwLiTm::set_W_par)
      .def_property("a", &PhrLocalSwLiTm::get_A_par, &PhrLocalSwLiTm::set_A_par)
      .def_property("e", &PhrLocalSwLiTm::get_E_par, &PhrLocalSwLiTm::set_E_par)
      .def_property("heuristic", &PhrLocalSwLiTm::getHeuristic, &PhrLocalSwLiTm::setHeuristic)
      .def_property(
          "online_training_parameters", &PhrLocalSwLiTm::getOnlineTrainingPars,
          [](PhrLocalSwLiTm& model, const OnlineTrainingPars& params) { model.setOnlineTrainingPars(params); })
      .def_property(
          "weights",
          [](PhrLocalSwLiTm& model) {
            std::vector<std::pair<std::string, float>> compWeights;
            model.getWeights(compWeights);

            std::vector<float> weights;
            for (auto& weight : compWeights)
              weights.push_back(weight.second);
            return weights;
          },
          &PhrLocalSwLiTm::setWeights)
      .def_property_readonly("direct_word_alignment_model",
                             [](PhrLocalSwLiTm& model) { return model.getSwModelInfo()->swAligModels[0]; })
      .def_property_readonly("inverse_word_alignment_model",
                             [](PhrLocalSwLiTm& model) { return model.getSwModelInfo()->invSwAligModels[0]; })
      .def(
          "print_translation_model",
          [](PhrLocalSwLiTm& model, const std::string& prefFileName) {
            return model.printAligModel(prefFileName) == THOT_OK;
          },
          py::arg("prefix_filename"))
      .def(
          "print_language_model",
          [](PhrLocalSwLiTm& model, const std::string& prefFileName) {
            return model.printLangModel(prefFileName) == THOT_OK;
          },
          py::arg("prefix_filename"));

  py::class_<TranslationData>(translation, "TranslationData")
      .def_readonly("target", &TranslationData::target)
      .def_readonly("source_segmentation", &TranslationData::sourceSegmentation)
      .def_readonly("target_segment_cuts", &TranslationData::targetSegmentCuts)
      .def_readonly("target_unknown_words", &TranslationData::targetUnknownWords)
      .def_readonly("score", &TranslationData::score)
      .def_readonly("score_components", &TranslationData::scoreComponents);

  py::class_<WordGraphArc>(translation, "WordGraphArc")
      .def_readonly("in_state", &WordGraphArc::predStateIndex)
      .def_readonly("out_state", &WordGraphArc::succStateIndex)
      .def_readonly("score", &WordGraphArc::arcScore)
      .def_readonly("words", &WordGraphArc::words)
      .def_readonly("source_start_index", &WordGraphArc::srcStartIndex)
      .def_readonly("source_end_index", &WordGraphArc::srcEndIndex)
      .def_readonly("is_unknown", &WordGraphArc::unknown);

  py::class_<WordGraphStateData>(translation, "WordGraphState")
      .def_readonly("in_arc_ids", &WordGraphStateData::arcsToPredStates)
      .def_readonly("out_arc_ids", &WordGraphStateData::arcsToSuccStates);

  py::class_<WordGraph>(translation, "WordGraph")
      .def_property_readonly("empty", &WordGraph::empty)
      .def_property_readonly("num_arcs", &WordGraph::numArcs)
      .def_property_readonly("num_states", &WordGraph::numStates)
      .def_property_readonly("initial_state_score", &WordGraph::getInitialStateScore)
      .def_property_readonly("final_states", &WordGraph::getFinalStateSet)
      .def(
          "get_in_arc_ids",
          [](WordGraph& wordGraph, HypStateIndex stateIndex) {
            std::vector<WordGraphArcId> arcIds;
            wordGraph.getArcIdsToPredStates(stateIndex, arcIds);
            return arcIds;
          },
          py::arg("state_id"))
      .def(
          "get_out_arc_ids",
          [](WordGraph& wordGraph, HypStateIndex stateIndex) {
            std::vector<WordGraphArcId> arcIds;
            wordGraph.getArcIdsToSuccStates(stateIndex, arcIds);
            return arcIds;
          },
          py::arg("state_id"))
      .def("get_arc", &WordGraph::wordGraphArcId2WordGraphArc, py::arg("arc_id"))
      .def("get_state", &WordGraph::getWordGraphStateData, py::arg("state_id"))
      .def("is_final_state", &WordGraph::stateIsFinal, py::arg("state_id"));

  py::class_<multi_stack_decoder_rec<PhrLocalSwLiTm>>(translation, "SmtDecoder")
      .def(py::init([](PhrLocalSwLiTm& model) {
             auto stackDecoder = new multi_stack_decoder_rec<PhrLocalSwLiTm>;
             stackDecoder->setParentSmtModel(&model);
             auto smtModel = dynamic_cast<PhrLocalSwLiTm*>(model.clone());
             smtModel->setTranslationMetadata(new TranslationMetadata<PhrScoreInfo>);
             stackDecoder->setSmtModel(smtModel);
             stackDecoder->useBestScorePruning(true);
             return stackDecoder;
           }),
           py::arg("model"))
      .def_property("i", &multi_stack_decoder_rec<PhrLocalSwLiTm>::get_I_par,
                    &multi_stack_decoder_rec<PhrLocalSwLiTm>::set_I_par)
      .def_property("s", &multi_stack_decoder_rec<PhrLocalSwLiTm>::get_S_par,
                    &multi_stack_decoder_rec<PhrLocalSwLiTm>::set_S_par)
      .def_property("is_breadth_first", &multi_stack_decoder_rec<PhrLocalSwLiTm>::get_breadthFirst,
                    &multi_stack_decoder_rec<PhrLocalSwLiTm>::set_breadthFirst)
      .def(
          "translate",
          [](multi_stack_decoder_rec<PhrLocalSwLiTm>& decoder, const std::string& sentence) {
            return translate(decoder, sentence);
          },
          py::arg("sentence"))
      .def(
          "translate_batch",
          [](multi_stack_decoder_rec<PhrLocalSwLiTm>& decoder, const std::vector<std::string>& sentences) {
            std::vector<TranslationData> results(sentences.size());
#pragma omp parallel
            {
              multi_stack_decoder_rec<PhrLocalSwLiTm> threadDecoder;
              threadDecoder.setParentSmtModel(decoder.getParentSmtModel());
              auto smtModel = dynamic_cast<PhrLocalSwLiTm*>(decoder.getParentSmtModel()->clone());
              smtModel->setTranslationMetadata(new TranslationMetadata<PhrScoreInfo>);
              threadDecoder.setSmtModel(smtModel);
              threadDecoder.useBestScorePruning(true);
              threadDecoder.set_I_par(decoder.get_I_par());
              threadDecoder.set_S_par(decoder.get_S_par());
              threadDecoder.set_breadthFirst(decoder.get_breadthFirst());

#pragma omp for schedule(dynamic)
              for (int i = 0; i < (int)sentences.size(); i++)
              {
                results[i] = translate(threadDecoder, sentences[i]);
              }
            }
            return results;
          },
          py::arg("sentences"))
      .def(
          "translate_n",
          [](multi_stack_decoder_rec<PhrLocalSwLiTm>& decoder, const std::string& sentence, int n) {
            return translateN(decoder, sentence, n);
          },
          py::arg("sentence"), py::arg("n"))
      .def(
          "translate_n_batch",
          [](multi_stack_decoder_rec<PhrLocalSwLiTm>& decoder, const std::vector<std::string>& sentences, int n) {
            std::vector<std::vector<TranslationData>> results(sentences.size());
#pragma omp parallel
            {
              multi_stack_decoder_rec<PhrLocalSwLiTm> threadDecoder;
              threadDecoder.setParentSmtModel(decoder.getParentSmtModel());
              auto smtModel = dynamic_cast<PhrLocalSwLiTm*>(decoder.getParentSmtModel()->clone());
              smtModel->setTranslationMetadata(new TranslationMetadata<PhrScoreInfo>);
              threadDecoder.setSmtModel(smtModel);
              threadDecoder.useBestScorePruning(true);
              threadDecoder.set_I_par(decoder.get_I_par());
              threadDecoder.set_S_par(decoder.get_S_par());
              threadDecoder.set_breadthFirst(decoder.get_breadthFirst());

#pragma omp for schedule(dynamic)
              for (int i = 0; i < (int)sentences.size(); i++)
              {
                results[i] = translateN(threadDecoder, sentences[i], n);
              }
            }
            return results;
          },
          py::arg("sentences"), py::arg("n"))
      .def(
          "get_word_graph",
          [](multi_stack_decoder_rec<PhrLocalSwLiTm>& decoder, const std::string& sentence) {
            decoder.useBestScorePruning(false);

            decoder.enableWordGraph();

            PhrLocalSwLiTm::Hypothesis hyp = decoder.translate(sentence);
            WordGraph* wg = decoder.getWordGraphPtr();

            decoder.disableWordGraph();

            decoder.useBestScorePruning(true);

            if (decoder.getSmtModel()->isComplete(hyp))
            {
              // Remove non-useful states from word-graph
              wg->obtainWgComposedOfUsefulStates();
              wg->orderArcsTopol();

              return new WordGraph(*wg);
            }

            return new WordGraph;
          },
          py::arg("sentence"))
      .def(
          "train_sentence_pair",
          [](multi_stack_decoder_rec<PhrLocalSwLiTm>& decoder, const std::string& sourceSentence,
             const std::string& targetSentence) {
  // Obtain system translation
#ifdef THOT_ENABLE_UPDATE_LLWEIGHTS
            decoder.enableWordGraph();
#endif

            PhrLocalSwLiTm::Hypothesis hyp = decoder.translate(sourceSentence);
            std::string sysSent = decoder.getSmtModel()->getTransInPlainText(hyp);

            // Add sentence to word-predictor
            decoder.getParentSmtModel()->addSentenceToWordPred(StrProcUtils::stringToStringVector(targetSentence));

#ifdef THOT_ENABLE_UPDATE_LLWEIGHTS
            // Train log-linear weights

            // Retrieve pointer to wordgraph
            WordGraph* wgPtr = decoder.getWordGraphPtr();
            decoder.getParentSmtModel()->updateLogLinearWeights(targetSentence, wgPtr);

            decoder.disableWordGraph();
#endif

            // Train generative models
            return decoder.getParentSmtModel()->onlineTrainFeatsSentPair(sourceSentence.c_str(), targetSentence.c_str(),
                                                                         sysSent.c_str())
                == THOT_OK;
          },
          py::arg("source_sentence"), py::arg("target_sentence"))
      .def("clear", &multi_stack_decoder_rec<PhrLocalSwLiTm>::clear);

  py::class_<PhraseExtractParameters>(translation, "PhraseExtractParameters")
      .def(py::init())
      .def_readwrite("monotone", &PhraseExtractParameters::monotone)
      .def_readwrite("max_target_phrase_length", &PhraseExtractParameters::maxTrgPhraseLength)
      .def_readwrite("constrain_source_length", &PhraseExtractParameters::constraintSrcLen)
      .def_readwrite("count_spurious", &PhraseExtractParameters::countSpurious)
      .def_readwrite("max_combs_in_table", &PhraseExtractParameters::maxNumbOfCombsInTable);

  py::class_<WbaIncrPhraseModel>(translation, "PhraseModel")
      .def(py::init())
      .def(
          "build",
          [](WbaIncrPhraseModel& model, const char* aligFileName, PhraseExtractParameters phePars, bool pseudoML) {
            return model.generateWbaIncrPhraseModel(aligFileName, phePars, pseudoML) == THOT_OK;
          },
          py::arg("alignment_filename"), py::arg("parameters"), py::arg("pseudo_ml"))
      .def(
          "print_phrase_table",
          [](WbaIncrPhraseModel& model, const char* fileName, int n) {
            return model.printPhraseTable(fileName, n) == THOT_OK;
          },
          py::arg("filename"), py::arg("n") = -1)
      .def("clear", &WbaIncrPhraseModel::clear);

  py::class_<AlignmentExtractor>(translation, "AlignmentExtractor")
      .def(py::init())
      .def(
          "open",
          [](AlignmentExtractor& extractor, const char* fileName, unsigned int fileFormat) {
            return extractor.open(fileName, fileFormat) == THOT_OK;
          },
          py::arg("filename"), py::arg("file_format") = GIZA_ALIG_FILE_FORMAT)
      .def(
          "intersect",
          [](AlignmentExtractor& extractor, const char* gizaAligFileName, const char* outFileName, bool transpose) {
            return extractor.intersect(gizaAligFileName, outFileName, transpose) == THOT_OK;
          },
          py::arg("alignment_filename"), py::arg("output_filename"), py::arg("transpose") = false)
      .def(
          "sum",
          [](AlignmentExtractor& extractor, const char* gizaAligFileName, const char* outFileName, bool transpose) {
            return extractor.sum(gizaAligFileName, outFileName, transpose) == THOT_OK;
          },
          py::arg("alignment_filename"), py::arg("output_filename"), py::arg("transpose") = false)
      .def(
          "symmetrize1",
          [](AlignmentExtractor& extractor, const char* gizaAligFileName, const char* outFileName, bool transpose) {
            return extractor.symmetr1(gizaAligFileName, outFileName, transpose) == THOT_OK;
          },
          py::arg("alignment_filename"), py::arg("output_filename"), py::arg("transpose") = false)
      .def(
          "symmetrize2",
          [](AlignmentExtractor& extractor, const char* gizaAligFileName, const char* outFileName, bool transpose) {
            return extractor.symmetr2(gizaAligFileName, outFileName, transpose) == THOT_OK;
          },
          py::arg("alignment_filename"), py::arg("output_filename"), py::arg("transpose") = false)
      .def("close", &AlignmentExtractor::close);
  ;
}
