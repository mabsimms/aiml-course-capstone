
from capstone.utils import get_git_info

def test_get_git_info():
    info = get_git_info()
    assert info is not None