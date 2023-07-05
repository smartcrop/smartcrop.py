#!/usr/bin/env python

from __future__ import annotations

import argparse, collections, os, subprocess, sys, time

from pytoolbox import filesystem
from pytoolbox.argparse import is_dir, is_file, FullPaths


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     epilog='smartcrop.py test bed')
    parser.add_argument('scripts', action=FullPaths, nargs='+', type=is_file)
    parser.add_argument('-s', '--source', action=FullPaths, type=is_dir)
    parser.add_argument('-t', '--target', action=FullPaths, type=is_dir)
    parser.add_argument('-p', '--passes', default=3, type=int)
    parser.add_argument('-w', '--width', default=200, type=int)
    parser.add_argument('-x', '--height', default=200, type=int)
    args = parser.parse_args()

    scripts = [args.scripts] if isinstance(args.scripts, str) else args.scripts
    if len(set(os.path.basename(s) for s in scripts)) < len(scripts):
        sys.exit('Please make sure scripts names are unique.')

    source_filenames = sorted(filesystem.find_recursive(args.source, '*.jpg'))
    timing_by_script = collections.defaultdict(list)
    for script in scripts:
        name = os.path.basename(script)
        source_directory = args.source + os.path.sep
        target_directory = os.path.join(args.target, name) + os.path.sep
        for i in range(1, args.passes + 1):
            print('Script {0} round {1} of 3'.format(name, i))
            start_time = time.time()
            for source_filename in source_filenames:
                target_filename = source_filename.replace(source_directory, target_directory)
                filesystem.makedirs(target_filename, parent=True)
                print(source_filename, target_filename)
                assert source_filename != target_filename
                subprocess.check_call([
                    'python', script,
                    '--width', str(args.width),
                    '--height', str(args.height),
                    source_filename, target_filename,
                ])
            timing_by_script[name].append(time.time() - start_time)

    for name, timings in sorted(timing_by_script.items()):
        print(name, *('{0:.2f}'.format(t) for t in timings))


if __name__ == '__main__':
    main()
