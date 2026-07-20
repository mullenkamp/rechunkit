# Changelog

## Unreleased

### Fixed
- `guess_chunk_shape` now raises `ValueError` when any shape dim is <= 0 (previously a zero-length dim passed straight through, producing an invalid chunk shape with a zero dim that breaks any downstream chunk arithmetic). The empty shape `()` still returns `()`.
- `guess_chunk_shape` now accepts numpy integer shape values (`np.int64` etc.) instead of raising `TypeError`; returned chunk dims are always plain Python ints.

### Documentation
- `rechunker()` docstring and the how-it-works concept page now document the yield-lifetime contract: yielded arrays may be views into the internal buffer, must be consumed or copied before advancing the generator, and must be treated as read-only.

## 0.5.1

Prior releases have no changelog; see the git history.
