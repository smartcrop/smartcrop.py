# Changelog

## v0.5.0 (2026-03-02)

Performance update

### Major compatibility breaks

- Remove `thirds()` and `saturation()` functions
- Remove different scores for `skin`, `detail`, and `saturation` from `score`
dict in resulting crops in favor of __only one__ total score __float__ value 

### Minor compatibility breaks

- Introduce `num_scale_steps` argument for `analyze()`, `crop()` and `crops()`
 methods, that determines the total number of different scales and replaces
 `scale_step`. 
- Change signatures of `score()`, `detect_skin()` and `detect_saturation()` methods
- Rename `imortance()` method

### Fix and enhancements

- Performance improvements: 
  - Vectorize importance calculation and cache them per crop dimension #49
  - Simplification of `score()` method through features precalculation #49, #51, #53
  - Use float32 instead of float64 for better performance with larger arrays #53
- Improve creating crop candidates and prevent crop duplicates #54
- Fix the #50 issue with bad scale ranges in `crops()` #54
- Better handling of bad arguments through propper error messages in `crops()` #54
- Extract `prepare_features_image()` method from `analyse()` #47
- Simplify top_crop lookup in `analyse()` method #47
- Vectorize `debug_crop()` method #49
- Add missing docstrings and type hints #47, #55

## v0.4.2 (2026-01-12)

### Features

* Resize image using `Image.resize` instead of `Image.thumbnail` #45 by @PeterL1n

### Fix and enhancements

* CI: Add GitHub Workflow for Python linting & testing
* CI: Drop Travis CI

## v0.4.1 (2024-10-17)

A big Thank You to @derVedro for the many contributions!

### Features

* Add `setup.py`to be able to release with twine
* Handle image orientation with `pillow` instead of `GExiv2` #35
* Process `RGBA`, `P` or `L` mode images properly #40

### Fix and enhancements

* Make `detect_skin()` cleaner and faster #42
* Use dataclasses to reduce boilerplate code #43
* Use slots to optimize memory usage #43
* Use `np.asarray` instead of `np.array` #36
* Drop `PyGObject` and `pytoolbox` dependency
* Refactor `thirds()`
* Fix `debug_crop()` (importance mask scaling & co) #37 #38
* Fix `saturation()` #41
* Update `flake8` configuration

## v0.4.0 (2023-07-05)

### Minor compatibility breaks

- Requires OS packages for working on EXIF metadata

### Features

- Apply orientation from EXIF metadata to fix #24 reported by @vlasvlasvlas

### Fix and enhancements

- Add type hints
- Modernize packaging & testing to fix #28

## v0.3.4 (2023-07-04)

### Fix and enhancements

- Fix Image.ANTIALIAS deprecation @GjjvdBurg
- CI: Update test matrix for Python 3.9 to 3.11
- CI: Update test matrix for Pillow 8 to 10
- CI: Toggle cache

## v0.3.3 (2021-07-05)

### Fix and enhancements

- Fix example code, thanks to @kaypon
- Make code compatible with Pillow 8.3.0, thanks to @GjjvdBurg

## v0.3.2 (2020-04-16)

### Fix and enhancements

- Use numpy to optimize (vectorize) detect methods
  - Inspired by the work done by @jrast
  - Keeping the exact same results
  - Tested on 100+ pics from [smartcrop.js example images](https://github.com/jwagner/smartcrop.js/tree/master/examples/images)
- Add changelog

## v0.3.1 (2020-04-13)

### Fix and enhancements

- Revert to previous version, see [#12](https://github.com/smartcrop/smartcrop.py/issues/12)
- Fix README, thanks to @nhnminh
- Refresh README

## v0.3.0

### Major compatibility breaks

- Break python2 compatibility

### Fix and enhancements

- Require at least Pillow 4.0
- Test against more Pillow versions
- Update Travis CI config
- Update README
- Add examples

## v0.2.1

### Fix and enhancements

- Add LICENSE
- Add Travis CI config
- Add tests

## v0.2

### Minor compatibility breaks

- Use default Python
- Fix wrong CIE conversion values (see [smartcrop.js #86](https://github.com/jwagner/smartcrop.js/issues/86))

### Fix and enhancements

- Add AUTHORS
- Add a test script (will be renamed examples/test_bed.py)
- Use PIL to perform edge detection and CIE image
- Create CIE image instead of repeated function call

## v0.1

- Initial tag
