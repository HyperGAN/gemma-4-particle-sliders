"""Read selected public dataset ZIP members without downloading repository images.

Range responses are checked exactly. A server ignoring Range is an error, never
permission to download the entire 22 GB archive.
"""
from __future__ import annotations

import io
import requests


class RangeFile(io.RawIOBase):
    def __init__(self, url: str, size: int, *, block_size=1024 * 1024, budget=128 * 1024**2):
        self.url, self.size, self.block_size, self.budget = url, size, block_size, budget
        self.position = self.transferred = 0
        self.cache: dict[int, bytes] = {}
        self.session = requests.Session()

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        target = offset + (0 if whence == io.SEEK_SET else self.position if whence == io.SEEK_CUR else self.size)
        if target < 0:
            raise ValueError("negative seek")
        self.position = target
        return target

    def _block(self, start):
        if start in self.cache:
            return self.cache[start]
        end = min(start + self.block_size, self.size) - 1
        count = end - start + 1
        if self.transferred + count > self.budget:
            raise ValueError("remote ZIP transfer budget exceeded")
        # Give intermediary caches a distinct key for each Range.
        with self.session.get(self.url, params={"range_start": start},
                              headers={"Range": f"bytes={start}-{end}"}, stream=True, timeout=60) as response:
            expected = f"bytes {start}-{end}/{self.size}"
            if response.status_code != 206 or response.headers.get("Content-Range") != expected:
                raise IOError(f"server did not honor exact Range: {response.status_code}, {response.headers.get('Content-Range')}")
            payload = response.raw.read(count + 1)
            if len(payload) != count:
                raise IOError("truncated or oversized range response")
        self.transferred += count
        # A bounded cache keeps even a long member stream small.
        if len(self.cache) >= 8:
            self.cache.pop(next(iter(self.cache)))
        self.cache[start] = payload
        return payload

    def read(self, size=-1):
        count = max(0, self.size - self.position) if size < 0 else max(0, min(size, self.size - self.position))
        if count > self.budget:
            raise ValueError("requested read exceeds transfer budget")
        result = []
        while count:
            start = self.position // self.block_size * self.block_size
            block = self._block(start)
            offset = self.position - start
            chunk = block[offset:offset + count]
            result.append(chunk)
            self.position += len(chunk)
            count -= len(chunk)
        return b"".join(result)

    def close(self):
        self.session.close()
        super().close()
