#include "nlp_common/ErrorDefs.h"
#include "nlp_common/WordAlignmentMatrix.h"
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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

using namespace std;

PYBIND11_MODULE(thot, m)
{
  py::module common = m.def_submodule("common");

  py::class_<WordAlignmentMatrix>(common, "WordAlignmentMatrix")
      .def(py::init<unsigned int, unsigned int>(), py::arg("row_length"), py::arg("column_length"))
      .def_property_readonly("row_length", &WordAlignmentMatrix::get_I)
      .def_property_readonly("column_length", &WordAlignmentMatrix::get_J)
      .def("get_value", &WordAlignmentMatrix::getValue, py::arg("i"), py::arg("j"))
      .def("set_value", &WordAlignmentMatrix::setValue, py::arg("i"), py::arg("j"), py::arg("value"))
      .def("get_asymmetric_alignment", [](WordAlignmentMatrix& matrix) {
        vector<PositionIndex> alignment;
        bool result = matrix.getAligVec(alignment);
        return make_tuple(result, alignment);
      });

  py::module alignment = m.def_submodule("alignment");

  py::class_<AlignmentModel>(alignment, "AlignmentModel")
      .def_property("variational_bayes", &AlignmentModel::getVariationalBayes, &AlignmentModel::setVariationalBayes)
      .def(
          "add_sentence_pair",
          [](AlignmentModel& model, const std::vector<std::string>& srcSentence,
             const std::vector<std::string>& trgSentence,
             float c) { return model.addSentencePair(srcSentence, trgSentence, c); },
          py::arg("src_sentence"), py::arg("trg_sentence"), py::arg("count"))
      .def_property_readonly("num_sentence_pairs", &AlignmentModel::numSentencePairs)
      .def(
          "get_sentence_pair",
          [](AlignmentModel& model, unsigned int n) {
            vector<string> srcSentence, trgSentence;
            Count c;
            if (model.getSentencePair(n, srcSentence, trgSentence, c) == THOT_ERROR)
              throw out_of_range("The sentence pair index is out of range.");
            return make_tuple(srcSentence, trgSentence, (float)c);
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
          "get_best_alignment",
          [](AlignmentModel& model, const char* srcSentence, const char* trgSentence) {
            WordAlignmentMatrix waMatrix;
            LgProb logProb = model.getBestAlignment(srcSentence, trgSentence, waMatrix);
            return make_tuple((double)logProb, waMatrix);
          },
          py::arg("src_sentence"), py::arg("trg_sentence"))
      .def(
          "get_best_alignment",
          [](AlignmentModel& model, const std::vector<std::string>& srcSentence,
             const std::vector<std::string>& trgSentence) {
            WordAlignmentMatrix waMatrix;
            LgProb logProb = model.getBestAlignment(srcSentence, trgSentence, waMatrix);
            return make_tuple((double)logProb, waMatrix);
          },
          py::arg("src_sentence"), py::arg("trg_sentence"))
      .def(
          "get_best_alignment",
          [](AlignmentModel& model, const std::vector<WordIndex>& srcSentence,
             const std::vector<WordIndex>& trgSentence) {
            WordAlignmentMatrix waMatrix;
            LgProb logProb = model.getBestAlignment(srcSentence, trgSentence, waMatrix);
            return make_tuple((double)logProb, waMatrix);
          },
          py::arg("src_sentence"), py::arg("trg_sentence"))
      .def(
          "load", [](AlignmentModel& model, const char* prefFileName) { model.load(prefFileName); },
          py::arg("prefix_filename"))
      .def(
          "print", [](AlignmentModel& model, const char* prefFileName) { model.print(prefFileName); },
          py::arg("prefix_filename"))
      .def_property_readonly("src_vocab_size", &AlignmentModel::getSrcVocabSize)
      .def("get_src_word_index", &AlignmentModel::stringToSrcWordIndex, py::arg("word"))
      .def("get_src_word", &AlignmentModel::wordIndexToSrcString, py::arg("word_index"))
      .def("src_word_exists", &AlignmentModel::existSrcSymbol, py::arg("word"))
      .def("add_src_word", &AlignmentModel::addSrcSymbol, py::arg("word"))
      .def_property_readonly("trg_vocab_size", &AlignmentModel::getTrgVocabSize)
      .def("get_trg_word_index", &AlignmentModel::stringToTrgWordIndex, py::arg("word"))
      .def("get_trg_word", &AlignmentModel::wordIndexToTrgString, py::arg("word_index"))
      .def("trg_word_exists", &AlignmentModel::existTrgSymbol, py::arg("word"))
      .def("add_trg_word", &AlignmentModel::addTrgSymbol, py::arg("word"))
      .def(
          "get_translations",
          [](AlignmentModel& model, WordIndex s, double threshold) {
            NbestTableNode<WordIndex> targetWords;
            model.getEntriesForSource(s, targetWords);
            vector<tuple<WordIndex, double>> targetWordsVec;
            for (NbestTableNode<WordIndex>::iterator iter = targetWords.begin(); iter != targetWords.end(); ++iter)
              if (iter->first >= threshold)
                targetWordsVec.push_back(make_tuple(iter->second, iter->first));
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

  py::class_<IncrAlignmentModel, AlignmentModel>(alignment, "IncrAlignmentModel")
      .def(
          "start_incr_training",
          [](IncrAlignmentModel& model, pair<unsigned int, unsigned int> sentPairRange) {
            model.startIncrTraining(sentPairRange);
          },
          py::arg("sentence_pair_range"))
      .def(
          "incr_train",
          [](IncrAlignmentModel& model, pair<unsigned int, unsigned int> sentPairRange) {
            model.incrTrain(sentPairRange);
          },
          py::arg("sentence_pair_range"))
      .def("end_incr_training", &IncrAlignmentModel::endIncrTraining);

  py::class_<Ibm1AlignmentModel, AlignmentModel>(alignment, "Ibm1AlignmentModel").def(py::init());

  py::class_<IncrIbm1AlignmentModel, Ibm1AlignmentModel, IncrAlignmentModel>(alignment, "IncrIbm1AlignmentModel")
      .def(py::init());

  py::class_<Ibm2AlignmentModel, Ibm1AlignmentModel>(alignment, "Ibm2AlignmentModel")
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

  py::class_<IncrIbm2AlignmentModel, Ibm2AlignmentModel, IncrAlignmentModel>(alignment, "IncrIbm2AlignmentModel")
      .def(py::init());

  py::class_<HmmAlignmentModel, Ibm1AlignmentModel>(alignment, "HmmAlignmentModel")
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

  py::class_<IncrHmmAlignmentModel, HmmAlignmentModel, IncrAlignmentModel>(alignment, "IncrHmmAlignmentModel")
      .def(py::init());

  py::class_<FastAlignModel, IncrAlignmentModel>(alignment, "FastAlignModel", py::multiple_inheritance())
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

  py::class_<Ibm3AlignmentModel, Ibm2AlignmentModel>(alignment, "Ibm3AlignmentModel")
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

  py::class_<Ibm4AlignmentModel, Ibm3AlignmentModel>(alignment, "Ibm4AlignmentModel")
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
