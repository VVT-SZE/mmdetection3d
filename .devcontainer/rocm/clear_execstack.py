"""Clear legacy executable-stack flags in local ROCm wheel libraries.

ROCm 5.6 wheels predate glibc 2.41, which rejects loading these libraries.
Only ELF64 little-endian PT_GNU_STACK flags are changed; code is untouched.
"""

import struct
import sysconfig
from pathlib import Path


def clear_execstack(path):
    with path.open('r+b') as library:
        header = library.read(64)
        if header[:6] != b'\x7fELF\x02\x01':
            return
        offset = struct.unpack_from('<Q', header, 32)[0]
        entry_size, count = struct.unpack_from('<HH', header, 54)
        for index in range(count):
            position = offset + index * entry_size
            library.seek(position)
            kind, flags = struct.unpack('<II', library.read(8))
            if kind == 0x6474E551 and flags & 1:
                library.seek(position + 4)
                library.write(struct.pack('<I', flags & ~1))
                print(f'Cleared executable-stack flag: {path.name}')


if __name__ == '__main__':
    library_dir = Path(sysconfig.get_path('purelib')) / 'torch' / 'lib'
    for path in sorted(library_dir.glob('*.so*')):
        if path.is_file() and not path.is_symlink():
            clear_execstack(path)
