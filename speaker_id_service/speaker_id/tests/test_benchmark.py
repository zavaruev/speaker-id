import os
import tempfile
from benchmark import create_large_file

def test_create_large_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "dummy.txt")
        create_large_file(path, size_mb=1)
        assert os.path.exists(path)
        assert os.stat(path).st_size == 1 * 1024 * 1024
