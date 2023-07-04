# Changelog

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
