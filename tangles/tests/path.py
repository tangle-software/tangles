import os

test_data_path = os.path.dirname(os.path.realpath(__file__))

def get_test_data_path(rel_path: str) -> str:
  return os.path.join(test_data_path, "_test_data", rel_path)