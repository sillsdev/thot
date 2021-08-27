from typing import List

import numpy as np
from thot.alignment import (
    AlignmentModel,
    HmmAlignmentModel,
    Ibm1AlignmentModel,
    SymmetrizationHeuristic,
    SymmetrizedAligner,
)


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
    _train_model(direct_ibm1_model, 1)

    direct_hmm_model = HmmAlignmentModel(direct_ibm1_model)
    _train_model(direct_hmm_model, 2)

    inverse_ibm1_model = Ibm1AlignmentModel()
    _add_sentence_pairs(inverse_ibm1_model, train_trg_sentences, train_src_sentences)
    _train_model(inverse_ibm1_model, 1)

    inverse_hmm_model = HmmAlignmentModel(inverse_ibm1_model)
    _train_model(inverse_hmm_model, 2)

    aligner = SymmetrizedAligner(direct_hmm_model, inverse_hmm_model)
    aligner.heuristic = SymmetrizationHeuristic.NONE

    align_src_sentences = [
        "isthay isyay ayay esttay-N .".split(),
        "isthay isyay otnay ayay esttay-N .".split(),
        "isthay isyay ayay esttay-N ardhay .".split(),
    ]
    align_trg_sentences = [
        "this is a test N .".split(),
        "this is not a test N .".split(),
        "this is a hard test N .".split(),
    ]
    alignments = aligner.get_best_alignments(align_src_sentences, align_trg_sentences)
    assert len(alignments) == 3
    assert np.array_equal(alignments[0][1].to_numpy(), _create_matrix(5, [1, 2, 3, 4, 4, 5]))
    assert np.array_equal(alignments[1][1].to_numpy(), _create_matrix(6, [1, 2, 3, 4, 5, 5, 6]))
    assert np.array_equal(alignments[2][1].to_numpy(), _create_matrix(6, [1, 2, 3, 5, 4, 4, 6]))


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
