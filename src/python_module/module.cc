#include "nlp_common/ErrorDefs.h"
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
        return py::array_t<bool>({matrix.get_I(), matrix.get_J()}, matrix.ptr()[0]);
      });

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

  py::class_<AlignmentModel, Aligner, std::shared_ptr<AlignmentModel>>(alignment, "AlignmentModel")
      .def_property("variational_bayes", &AlignmentModel::getVariationalBayes, &AlignmentModel::setVariationalBayes)
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
      .def("start_training", [](AlignmentModel& model) { model.startTraining(); })
      .def("train", [](AlignmentModel& model) { model.train(); })
      .def("end_training", &AlignmentModel::endTraining)
      .def(
          "get_sentence_length_prob",
          [](AlignmentModel& model, unsigned int slen, unsigned int tlen) {
            return (double)model.getSentenceLengthProb(slen, tlen);
          },
          py::arg("src_length"), py::arg("trg_length"))
      .def(
          "get_sentence_length_log_prob",
          [](AlignmentModel& model, unsigned int slen, unsigned int tlen) {
            return (double)model.getSentenceLengthLgProb(slen, tlen);
          },
          py::arg("src_length"), py::arg("trg_length"))
      .def(
          "load", [](AlignmentModel& model, const char* prefFileName) { model.load(prefFileName); },
          py::arg("prefix_filename"))
      .def(
          "print", [](AlignmentModel& model, const char* prefFileName) { model.print(prefFileName); },
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
          "get_translation_prob",
          [](AlignmentModel& model, WordIndex s, WordIndex t) { return (double)model.pts(s, t); },
          py::arg("src_word_index"), py::arg("trg_word_index"))
      .def(
          "get_translation_log_prob",
          [](AlignmentModel& model, WordIndex s, WordIndex t) { return (double)model.logpts(s, t); },
          py::arg("src_word_index"), py::arg("trg_word_index"));

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
      .def(
          "get_alignment_prob",
          [](Ibm2AlignmentModel& model, PositionIndex j, PositionIndex slen, PositionIndex tlen, PositionIndex i) {
            return (double)model.aProb(j, slen, tlen, i);
          },
          py::arg("j"), py::arg("src_length"), py::arg("trg_length"), py::arg("i"))
      .def(
          "get_alignment_log_prob",
          [](Ibm2AlignmentModel& model, PositionIndex j, PositionIndex slen, PositionIndex tlen, PositionIndex i) {
            return (double)model.logaProb(j, slen, tlen, i);
          },
          py::arg("j"), py::arg("src_length"), py::arg("trg_length"), py::arg("i"));

  py::class_<IncrIbm2AlignmentModel, Ibm2AlignmentModel, IncrAlignmentModel, std::shared_ptr<IncrIbm2AlignmentModel>>(
      alignment, "IncrIbm2AlignmentModel")
      .def(py::init());

  py::class_<HmmAlignmentModel, Ibm1AlignmentModel, std::shared_ptr<HmmAlignmentModel>>(alignment, "HmmAlignmentModel")
      .def(py::init())
      .def(py::init<Ibm1AlignmentModel&>(), py::arg("model"))
      .def(
          "get_alignment_prob",
          [](HmmAlignmentModel& model, PositionIndex prev_i, PositionIndex slen, PositionIndex i) {
            return (double)model.aProb(prev_i, slen, i);
          },
          py::arg("prev_i"), py::arg("src_length"), py::arg("i"))
      .def(
          "get_alignment_log_prob",
          [](HmmAlignmentModel& model, PositionIndex prev_i, PositionIndex slen, PositionIndex i) {
            return (double)model.logaProb(prev_i, slen, i);
          },
          py::arg("prev_i"), py::arg("src_length"), py::arg("i"));

  py::class_<IncrHmmAlignmentModel, HmmAlignmentModel, IncrAlignmentModel, std::shared_ptr<IncrHmmAlignmentModel>>(
      alignment, "IncrHmmAlignmentModel")
      .def(py::init());

  py::class_<FastAlignModel, IncrAlignmentModel, std::shared_ptr<FastAlignModel>>(alignment, "FastAlignModel",
                                                                                  py::multiple_inheritance())
      .def(py::init())
      .def(
          "get_alignment_prob",
          [](FastAlignModel& model, PositionIndex j, PositionIndex slen, PositionIndex tlen, PositionIndex i) {
            return (double)model.aProb(j, slen, tlen, i);
          },
          py::arg("j"), py::arg("src_length"), py::arg("trg_length"), py::arg("i"))
      .def(
          "get_alignment_log_prob",
          [](FastAlignModel& model, PositionIndex j, PositionIndex slen, PositionIndex tlen, PositionIndex i) {
            return (double)model.logaProb(j, slen, tlen, i);
          },
          py::arg("j"), py::arg("src_length"), py::arg("trg_length"), py::arg("i"));

  py::class_<Ibm3AlignmentModel, Ibm2AlignmentModel, std::shared_ptr<Ibm3AlignmentModel>>(alignment,
                                                                                          "Ibm3AlignmentModel")
      .def(py::init())
      .def(py::init<Ibm2AlignmentModel&>(), py::arg("model"))
      .def(py::init<HmmAlignmentModel&>(), py::arg("model"))
      .def(
          "get_distortion_prob",
          [](Ibm3AlignmentModel& model, PositionIndex i, PositionIndex slen, PositionIndex tlen, PositionIndex j) {
            return (double)model.distortionProb(i, slen, tlen, j);
          },
          py::arg("i"), py::arg("src_length"), py::arg("trg_length"), py::arg("j"))
      .def(
          "get_distortion_log_prob",
          [](Ibm3AlignmentModel& model, PositionIndex i, PositionIndex slen, PositionIndex tlen, PositionIndex j) {
            return (double)model.logDistortionProb(i, slen, tlen, j);
          },
          py::arg("i"), py::arg("src_length"), py::arg("trg_length"), py::arg("j"))
      .def(
          "get_fertility_prob",
          [](Ibm3AlignmentModel& model, WordIndex s, PositionIndex phi) { return (double)model.fertilityProb(s, phi); },
          py::arg("src_word_index"), py::arg("fertility"))
      .def(
          "get_fertility_log_prob",
          [](Ibm3AlignmentModel& model, WordIndex s, PositionIndex phi) {
            return (double)model.logFertilityProb(s, phi);
          },
          py::arg("src_word_index"), py::arg("fertility"));

  py::class_<Ibm4AlignmentModel, Ibm3AlignmentModel, std::shared_ptr<Ibm4AlignmentModel>>(alignment,
                                                                                          "Ibm4AlignmentModel")
      .def(py::init())
      .def(py::init<Ibm3AlignmentModel&>(), py::arg("model"))
      .def(
          "get_head_distortion_prob",
          [](Ibm4AlignmentModel& model, WordClassIndex srcWordClass, WordClassIndex trgWordClass, PositionIndex tlen,
             int dj) { return (double)model.headDistortionProb(srcWordClass, trgWordClass, tlen, dj); },
          py::arg("src_word_class"), py::arg("trg_word_class"), py::arg("trg_length"), py::arg("dj"))
      .def(
          "get_head_distortion_log_prob",
          [](Ibm4AlignmentModel& model, WordClassIndex src_word_class, WordClassIndex trg_word_class,
             PositionIndex tlen,
             int dj) { return (double)model.logHeadDistortionProb(src_word_class, trg_word_class, tlen, dj); },
          py::arg("src_word_class"), py::arg("trg_word_class"), py::arg("trg_length"), py::arg("dj"))
      .def(
          "get_nonhead_distortion_prob",
          [](Ibm4AlignmentModel& model, WordClassIndex trg_word_class, PositionIndex tlen, int dj) {
            return (double)model.nonheadDistortionProb(trg_word_class, tlen, dj);
          },
          py::arg("trg_word_class"), py::arg("trg_length"), py::arg("dj"))
      .def(
          "get_nonhead_distortion_log_prob",
          [](Ibm4AlignmentModel& model, WordClassIndex trg_word_class, PositionIndex tlen, int dj) {
            return (double)model.logNonheadDistortionProb(trg_word_class, tlen, dj);
          },
          py::arg("trg_word_class"), py::arg("trg_length"), py::arg("dj"));
}
