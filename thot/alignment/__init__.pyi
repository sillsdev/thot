from enum import Enum
from typing import Optional, Sequence, Tuple, overload

from ..common import WordAlignmentMatrix

class Aligner:
    def get_src_word_index(self, word: str) -> int: ...
    def get_trg_word_index(self, word: str) -> int: ...
    @overload
    def get_best_alignment(self, src_sentence: str, trg_sentence: str) -> Tuple[float, WordAlignmentMatrix]: ...
    @overload
    def get_best_alignment(
        self, src_sentence: Sequence[int], trg_sentence: Sequence[int]
    ) -> Tuple[float, WordAlignmentMatrix]: ...
    @overload
    def get_best_alignment(
        self, src_sentence: Sequence[str], trg_sentence: Sequence[str]
    ) -> Tuple[float, WordAlignmentMatrix]: ...
    @overload
    def get_best_alignments(
        self, src_sentences: Sequence[Sequence[int]], trg_sentences: Sequence[Sequence[int]]
    ) -> Sequence[Tuple[float, WordAlignmentMatrix]]: ...
    @overload
    def get_best_alignments(
        self, src_sentences: Sequence[Sequence[str]], trg_sentences: Sequence[Sequence[str]]
    ) -> Sequence[Tuple[float, WordAlignmentMatrix]]: ...

class SymmetrizationHeuristic(Enum):
    NONE = ...
    UNION = ...
    INTERSECTION = ...
    OCH = ...
    GROW = ...
    GROW_DIAG = ...
    GROW_DIAG_FINAL = ...
    GROW_DIAG_FINAL_AND = ...

class SymmetrizedAligner(Aligner):
    def __init__(self, direct_aligner: Aligner, inverse_aligner: Aligner) -> None: ...
    @property
    def heuristic(self) -> SymmetrizationHeuristic: ...
    @heuristic.setter
    def heuristic(self, value: SymmetrizationHeuristic) -> None: ...

class AlignmentModelType(Enum):
    IBM1 = ...
    IBM2 = ...
    HMM = ...
    IBM3 = ...
    IBM4 = ...
    FAST_ALIGN = ...
    INCR_IBM1 = ...
    INCR_IBM2 = ...
    INCR_HMM = ...

class AlignmentModel(Aligner):
    @property
    def model_type(self) -> AlignmentModelType: ...
    @property
    def num_sentence_pairs(self) -> int: ...
    @property
    def src_vocab_size(self) -> int: ...
    @property
    def trg_vocab_size(self) -> int: ...
    @property
    def variational_bayes(self) -> bool: ...
    @variational_bayes.setter
    def variational_bayes(self, value: bool) -> None: ...
    @property
    def max_sentence_length(self) -> int: ...
    def read_sentence_pairs(
        self, src_filename: str, trg_filename: str, counts_filename: Optional[str] = None
    ) -> None: ...
    def add_sentence_pair(
        self, src_sentence: Sequence[str], trg_sentence: Sequence[str], count: float = 1
    ) -> Tuple[int, int]: ...
    def get_sentence_pair(self, n: int) -> Tuple[Sequence[str], Sequence[str], float]: ...
    def add_src_word(self, word: str) -> int: ...
    def get_src_word(self, word_index: int) -> str: ...
    def src_word_exists(self, word: str) -> bool: ...
    def add_trg_word(self, word: str) -> int: ...
    def get_trg_word(self, word_index: int) -> str: ...
    def trg_word_exists(self, word: str) -> bool: ...
    def start_training(self) -> int: ...
    def train(self) -> None: ...
    def end_training(self) -> None: ...
    def sentence_length_log_prob(self, src_length: int, trg_length: int) -> float: ...
    def sentence_length_prob(self, src_length: int, trg_length: int) -> float: ...
    def translation_log_prob(self, src_word_index: int, trg_word_index: int) -> float: ...
    def translation_prob(self, src_word_index: int, trg_word_index: int) -> float: ...
    def get_translations(self, s: int, threshold: float = 0) -> Sequence[Tuple[int, float]]: ...
    def map_src_word_to_word_class(self, word: str, word_class: str) -> None: ...
    def map_trg_word_to_word_class(self, word: str, word_class: str) -> None: ...
    def load(self, prefix_filename: str) -> bool: ...
    def print(self, prefix_filename: str) -> bool: ...
    def clear(self) -> None: ...

class IncrAlignmentModel(AlignmentModel):
    def start_incr_training(self, sentence_pair_range: Tuple[int, int]) -> None: ...
    def incr_train(self, sentence_pair_range: Tuple[int, int]) -> None: ...
    def end_incr_training(self) -> None: ...

class Ibm1AlignmentModel(AlignmentModel):
    def __init__(self) -> None: ...

class IncrIbm1AlignmentModel(Ibm1AlignmentModel, IncrAlignmentModel):
    def __init__(self) -> None: ...

class Ibm2AlignmentModel(Ibm1AlignmentModel, AlignmentModel):
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, model: Ibm1AlignmentModel) -> None: ...
    @property
    def compact_alignment_table(self) -> bool: ...
    @compact_alignment_table.setter
    def compact_alignment_table(self, value: bool) -> None: ...
    def alignment_log_prob(self, j: int, src_length: int, trg_length: int, i: int) -> float: ...
    def alignment_prob(self, j: int, src_length: int, trg_length: int, i: int) -> float: ...

class IncrIbm2AlignmentModel(Ibm2AlignmentModel, IncrAlignmentModel):
    def __init__(self) -> None: ...

class HmmAlignmentModel(Ibm1AlignmentModel, AlignmentModel):
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, model: Ibm1AlignmentModel) -> None: ...
    @property
    def hmm_p0(self) -> float: ...
    @hmm_p0.setter
    def hmm_p0(self, value: float) -> None: ...
    @property
    def lexical_smoothing_factor(self) -> float: ...
    @lexical_smoothing_factor.setter
    def lexical_smoothing_factor(self, value: float) -> None: ...
    @property
    def hmm_alignment_smoothing_factor(self) -> float: ...
    @hmm_alignment_smoothing_factor.setter
    def hmm_alignment_smoothing_factor(self, value: float) -> None: ...
    def hmm_alignment_log_prob(self, prev_i: int, src_length: int, i: int) -> float: ...
    def hmm_alignment_prob(self, prev_i: int, src_length: int, i: int) -> float: ...

class IncrHmmAlignmentModel(HmmAlignmentModel, IncrAlignmentModel):
    def __init__(self) -> None: ...

class FastAlignModel(IncrAlignmentModel):
    def __init__(self) -> None: ...
    @property
    def fast_align_p0(self) -> float: ...
    @fast_align_p0.setter
    def fast_align_p0(self, value: float) -> None: ...
    def alignment_log_prob(self, j: int, src_length: int, trg_length: int, i: int) -> float: ...
    def alignment_prob(self, j: int, src_length: int, trg_length: int, i: int) -> float: ...

class Ibm3AlignmentModel(Ibm2AlignmentModel, Ibm1AlignmentModel, AlignmentModel):
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, model: HmmAlignmentModel) -> None: ...
    @overload
    def __init__(self, model: Ibm2AlignmentModel) -> None: ...
    @property
    def fertility_smoothing_factor(self) -> float: ...
    @fertility_smoothing_factor.setter
    def fertility_smoothing_factor(self, value: float) -> None: ...
    @property
    def count_threshold(self) -> float: ...
    @count_threshold.setter
    def count_threshold(self, value: float) -> None: ...
    def distortion_log_prob(self, i: int, src_length: int, trg_length: int, j: int) -> float: ...
    def distortion_prob(self, i: int, src_length: int, trg_length: int, j: int) -> float: ...
    def fertility_log_prob(self, src_word_index: int, fertility: int) -> float: ...
    def fertility_prob(self, src_word_index: int, fertility: int) -> float: ...

class Ibm4AlignmentModel(Ibm3AlignmentModel, Ibm2AlignmentModel, Ibm1AlignmentModel, AlignmentModel):
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, model: HmmAlignmentModel) -> None: ...
    @overload
    def __init__(self, model: Ibm3AlignmentModel) -> None: ...
    @property
    def distortion_smoothing_factor(self) -> float: ...
    @distortion_smoothing_factor.setter
    def distortion_smoothing_factor(self, value: float) -> None: ...
    def head_distortion_log_prob(self, src_word_class: int, trg_word_class: int, trg_length: int, dj: int) -> float: ...
    def head_distortion_prob(self, src_word_class: int, trg_word_class: int, trg_length: int, dj: int) -> float: ...
    def nonhead_distortion_log_prob(self, trg_word_class: int, trg_length: int, dj: int) -> float: ...
    def nonhead_distortion_prob(self, trg_word_class: int, trg_length: int, dj: int) -> float: ...

class SentenceLengthModel:
    def sentence_length_prob(self, src_length: int, trg_length: int) -> float: ...
    def sentence_length_log_prob(self, src_length: int, trg_length: int) -> float: ...
    @overload
    def train_sentence_pair(
        self, src_sentence: Sequence[str], trg_sentence: Sequence[str], count: float = 1
    ) -> None: ...
    @overload
    def train_sentence_pair(self, src_length: int, trg_length: int, count: float = 1) -> None: ...
    def load(self, filename: str) -> None: ...
    def print(self, filename: str) -> None: ...
    def clear(self) -> None: ...

class NormalSentenceLengthModel(SentenceLengthModel):
    def __init__(self) -> None: ...

__all__ = [
    "Aligner",
    "AlignmentModel",
    "AlignmentModelType",
    "FastAlignModel",
    "HmmAlignmentModel",
    "Ibm1AlignmentModel",
    "Ibm2AlignmentModel",
    "Ibm3AlignmentModel",
    "Ibm4AlignmentModel",
    "IncrAlignmentModel",
    "IncrHmmAlignmentModel",
    "IncrIbm1AlignmentModel",
    "IncrIbm2AlignmentModel",
    "NormalSentenceLengthModel",
    "SentenceLengthModel",
    "SymmetrizationHeuristic",
    "SymmetrizedAligner",
]
