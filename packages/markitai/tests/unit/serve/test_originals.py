def test_outlook_and_notebook_originals_have_their_own_media_types() -> None:
    from pathlib import Path

    from markitai.serve.originals import media_type_of

    assert media_type_of(Path("mail.MSG")) == "application/vnd.ms-outlook"
    assert media_type_of(Path("nb.ipynb")) == "application/x-ipynb+json"
    assert media_type_of(Path("blob.unknownsuffix")) == "application/octet-stream"
