"""Prompt packs that come from outside this codebase.

`worker/longform/` holds the prompt text this platform authored and measured;
this package holds text a vendor or the client supplied, kept as data and read
at runtime. The split matters when the two disagree, which they do — see
`worker/prompt/ltx25/__init__.py`.
"""
