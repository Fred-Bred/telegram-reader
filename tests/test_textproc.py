from reader.textproc import (
    chunk_text,
    fix_hyphenation,
    group_chunks,
    guess_title,
    merge_pages,
    sanitize_filename,
)


def test_fix_hyphenation_joins_soft_hyphens():
    assert fix_hyphenation("an expe-\nrience of") == "an experience of"
    assert fix_hyphenation("Das Nor-\nwegen") == "Das Norwegen"


def test_fix_hyphenation_keeps_common_compound_hyphens():
    assert fix_hyphenation("a well-\nknown author") == "a well-known author"


def test_merge_pages_joins_word_across_page_break():
    merged = merge_pages(["text that ends with expe-", "rience of a lifetime"])
    assert merged == "text that ends with experience of a lifetime"


def test_merge_pages_keeps_compound_hyphen_across_page_break():
    merged = merge_pages(["the well-", "known answer."])
    assert merged == "the well-known answer."


def test_merge_pages_keeps_paragraph_gap_between_pages():
    merged = merge_pages(["first page end.", "Second page start."])
    assert merged == "first page end.\n\nSecond page start."


def test_merge_pages_skips_empty_pages():
    assert merge_pages(["a", "", "   ", "b"]) == "a\n\nb"


def test_guess_title_prefers_first_clean_line():
    assert guess_title("Chapter 3: Methods\nThe study...\n") == "Chapter 3: Methods"


def test_guess_title_skips_short_and_sentence_lines():
    text = "This sentence is long and ends with a period.\nTitle Of Paper\n"
    assert guess_title(text) == "Title Of Paper"


def test_guess_title_none():
    assert guess_title("x") is None


def test_sanitize_filename():
    assert sanitize_filename("What? is/this: a*title?.pdf") == "What is this a title.pdf"
    assert sanitize_filename("") == "document"
    assert sanitize_filename("x" * 200) == "x" * 60


def test_chunk_text_respects_limit_and_paragraphs():
    text = ("para one. " * 30).strip() + "\n\n" + ("para two. " * 30)
    chunks = chunk_text(text, max_chars=200)
    assert all(len(chunk) <= 200 for chunk in chunks)
    assert "".join(chunks).startswith("para one.")
    assert "para two." in chunks[-1]


def test_chunk_text_splits_oversized_paragraph_by_sentence():
    paragraph = "Sentence one. " * 50
    chunks = chunk_text(paragraph, max_chars=100)
    assert all(len(chunk) <= 100 for chunk in chunks)
    assert " ".join(chunks) == paragraph.strip()


def test_group_chunks_splits_by_part_size():
    chunks = [f"chunk-{i} " * 10 for i in range(10)]
    groups = group_chunks(chunks, part_chars=120)
    assert len(groups) >= 2
    for group in groups:
        assert sum(len(chunk) for chunk in group) <= 120 + max(len(c) for c in chunks)
    flat = [chunk for group in groups for chunk in group]
    assert flat == chunks
