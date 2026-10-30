#!/usr/bin/env python3
"""Build-time syntax adapters for the vendored AMD FSR1 FP32 functions.

Only four named functions are selected. Mathematics stays byte-for-byte except
out/inout reference syntax and explicit Metal input-resource arguments. The CPU
reference receives only C++ reference syntax, independently of the Metal adapter.
Outputs belong to the existing external build tree, never the source or audit.
"""
import argparse
from pathlib import Path
import re


def functions(source):
    masked = re.sub(r'/\*.*?\*/|//[^\n]*', lambda m: ''.join('\n' if c == '\n' else ' ' for c in m[0]), source, flags=re.S)
    result = []
    for name in ('FsrEasuTapF', 'FsrEasuSetF', 'FsrEasuF', 'FsrRcasF'):
        starts = list(re.finditer(r'\bvoid\s+' + name + r'\s*\(', masked))
        if len(starts) != 1:
            raise ValueError('Expected one AMD definition: ' + name)
        start = starts[0].start()
        body = masked.index('{', start)
        depth, end = 1, body + 1
        while depth:
            depth += (masked[end] == '{') - (masked[end] == '}')
            end += 1
        result.append((name, source[start:end]))
    return result


def adapt(name, text, metal):
    qualifier = 'thread ' if metal else ''
    text = re.sub(r'\b(?:inout|out)\s+(AF[1234])\s+(\w+)', lambda m: qualifier + m[1] + '& ' + m[2], text)
    if metal and name in ('FsrEasuF', 'FsrRcasF'):
        body = text.index('{')
        end = text.rindex(')', 0, body)
        text = text[:end] + ', texture2d<float> source, constant fsr_params& params' + text[end:]
        callbacks = ('FsrEasuRF', 'FsrEasuGF', 'FsrEasuBF') if name == 'FsrEasuF' else ('FsrRcasLoadF',)
        for callback in callbacks:
            for match in reversed(list(re.finditer(r'\b' + callback + r'\(', text))):
                depth, close = 1, match.end()
                while depth:
                    depth += (text[close] == '(') - (text[close] == ')')
                    close += 1
                text = text[:close - 1] + ', source, params' + text[close - 1:]
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('amd', type=Path)
    parser.add_argument('template', type=Path)
    parser.add_argument('metal_output', type=Path)
    parser.add_argument('reference_output', type=Path)
    args = parser.parse_args()
    source = args.amd.read_text()
    begin = source.index('// Copyright (c) 2021 Advanced Micro Devices')
    end = source.index('// THE SOFTWARE.', begin) + len('// THE SOFTWARE.')
    license_text = source[begin:end] + '\n'
    selected = functions(source)
    metal = '\n\n'.join(adapt(name, text, True) for name, text in selected)
    reference = '\n\n'.join(adapt(name, text, False) for name, text in selected)
    template = args.template.read_text()
    if template.count('// AMD_FSR_FUNCTIONS') != 1 or template.count('// AMD_FSR_LICENSE') != 1:
        raise ValueError('Missing unique native FSR insertion markers')
    args.metal_output.write_text(template.replace('// AMD_FSR_FUNCTIONS', metal).replace('// AMD_FSR_LICENSE', license_text))
    args.reference_output.write_text(license_text + reference + '\n')


if __name__ == '__main__':
    main()
