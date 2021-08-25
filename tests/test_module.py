from thot.alignment import HmmAlignmentModel

def test_alignment_model() -> None:
    model = HmmAlignmentModel()
    model.add_sentence_pair("isthay isyay ayay esttay-N .".split(), "this is a test N .".split())
    model.add_sentence_pair("ouyay ouldshay esttay-V oftenyay .".split(), "you should test V often .".split())
    model.add_sentence_pair("isyay isthay orkingway ?".split(), "is this working ?".split())
    model.add_sentence_pair("isthay ouldshay orkway-V .".split(), "this should work V .".split())
    model.add_sentence_pair("ityay isyay orkingway .".split(), "it is working .".split())
    model.add_sentence_pair("orkway-N ancay ebay ardhay !".split(), "work N can be hard !".split())
    model.add_sentence_pair("ayay esttay-N ancay ebay ardhay .".split(), "a test N can be hard .".split())
    model.add_sentence_pair("isthay isyay ayay ordway !".split(), "this is a word !".split())

    model.start_training()
    for _ in range(2):
        model.train()
    model.end_training()

    src_sentences = [
        "isthay isyay ayay esttay-N .".split(),
        "isthay isyay otnay ayay esttay-N .".split(),
        "isthay isyay ayay esttay-N ardhay .".split()
    ]
    trg_sentences = [
        "this is a test N .".split(),
        "this is not a test N .".split(),
        "this is a hard test N .".split()
    ]
    alignments = model.get_best_alignments(src_sentences, trg_sentences)
    assert len(alignments) == 3
    assert alignments[0][1] == [1, 2, 3, 4, 4, 5]
    assert alignments[1][1] == [1, 2, 3, 4, 5, 5, 6]
    assert alignments[2][1] == [1, 2, 3, 5, 4, 4, 6]
