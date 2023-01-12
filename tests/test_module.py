from typing import List

from pytest import approx
import numpy as np
from thot.alignment import (
    AlignmentModelType,
    AlignmentModel,
    HmmAlignmentModel,
    Ibm1AlignmentModel,
    NormalSentenceLengthModel,
    SymmetrizationHeuristic,
    SymmetrizedAligner,
)
from thot.translation import SmtModel, SmtDecoder


def test_alignment_model() -> None:
    train_src_sentences = [
        "isthay isyay ayay esttay-N .",
        "ouyay ouldshay esttay-V oftenyay .",
        "isyay isthay orkingway ?",
        "isthay ouldshay orkway-V .",
        "ityay isyay orkingway .",
        "orkway-N ancay ebay ardhay !",
        "ayay esttay-N ancay ebay ardhay .",
        "isthay isyay ayay ordway !",
    ]
    train_trg_sentences = [
        "this is a test N .",
        "you should test V often .",
        "is this working ?",
        "this should work V .",
        "it is working .",
        "work N can be hard !",
        "a test N can be hard .",
        "this is a word !",
    ]

    direct_ibm1_model = Ibm1AlignmentModel()
    _add_sentence_pairs(direct_ibm1_model, train_src_sentences, train_trg_sentences)
    _train_model(direct_ibm1_model, 2)

    direct_hmm_model = HmmAlignmentModel(direct_ibm1_model)
    direct_hmm_model.hmm_p0 = 0.1
    _train_model(direct_hmm_model, 2)

    inverse_ibm1_model = Ibm1AlignmentModel()
    _add_sentence_pairs(inverse_ibm1_model, train_trg_sentences, train_src_sentences)
    _train_model(inverse_ibm1_model, 2)

    inverse_hmm_model = HmmAlignmentModel(inverse_ibm1_model)
    inverse_hmm_model.hmm_p0 = 0.1
    _train_model(inverse_hmm_model, 2)

    aligner = SymmetrizedAligner(direct_hmm_model, inverse_hmm_model)
    aligner.heuristic = SymmetrizationHeuristic.NONE

    align_src_sentences = [
        "isthay isyay ayay esttay-N .".split(),
        "isthay isyay otnay ayay esttay-N .".split(),
        "isthay isyay ayay esttay-N ardhay .".split(),
        "".split(),
    ]
    align_trg_sentences = [
        "this is a test N .".split(),
        "this is not a test N .".split(),
        "this is a hard test N .".split(),
        "".split(),
    ]
    alignments = aligner.get_best_alignments(align_src_sentences, align_trg_sentences)
    assert len(alignments) == 4
    assert np.array_equal(alignments[0][1].to_numpy(), _create_matrix(5, [1, 2, 3, 4, 4, 5]))
    assert np.array_equal(alignments[1][1].to_numpy(), _create_matrix(6, [1, 2, 3, 4, 5, 5, 6]))
    assert np.array_equal(alignments[2][1].to_numpy(), _create_matrix(6, [1, 2, 3, 5, 4, 4, 4]))
    assert np.array_equal(alignments[3][1].to_numpy(), _create_matrix(0, []))


def test_sentence_length_model() -> None:
    model = NormalSentenceLengthModel()
    model.train_sentence_pair(10, 20)
    model.train_sentence_pair(5, 10)
    model.train_sentence_pair(7, 14)
    model.train_sentence_pair(9, 18)
    model.train_sentence_pair(11, 22)

    assert model.sentence_length_prob(8, 16) == approx(0.0966, abs=0.0001)
    assert model.sentence_length_prob(10, 20) == approx(0.0815, abs=0.0001)
    assert model.sentence_length_prob(7, 10) == approx(0.0389, abs=0.0001)


def test_smt_model() -> None:
    model = SmtModel(AlignmentModelType.FAST_ALIGN)
    params = model.online_training_parameters
    assert params.em_iters == 5
    weights = model.weights
    assert len(weights) == 8
    direct_model = model.direct_word_alignment_model
    assert direct_model.model_type == AlignmentModelType.FAST_ALIGN
    decoder = SmtDecoder(model)
    result = decoder.translate("this is a test")
    assert len(result.target) == 4


def _add_sentence_pairs(model: AlignmentModel, src_sentences: List[str], trg_sentences: List[str]) -> None:
    for src_sentence, trg_sentence in zip(src_sentences, trg_sentences):
        model.add_sentence_pair(src_sentence.split(), trg_sentence.split())


def _train_model(model: AlignmentModel, iters: int) -> None:
    model.start_training()
    for _ in range(iters):
        model.train()
    model.end_training()


def _create_matrix(src_len: int, alignment: List[int]) -> np.ndarray:
    matrix = np.full((src_len, len(alignment)), False)
    for j, i in enumerate(alignment):
        i -= 1
        if i >= 0:
            matrix[i, j] = True
    return matrix
