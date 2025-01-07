"""tests for utils.py"""

import os
import tempfile

import pytest
from atlantes.utils import (
    find_most_recent_checkpoint,
    get_commit_hash,
    is_directory_empty,
    process_additional_args,
)


def test_unit_find_most_recent_checkpoint() -> None:
    """Test for find_most_recent_checkpoint function."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a few checkpoint files such that the str ordering indicates chronological order
        checkpoint_files = [
            "model_2024-06-20-20-02_epcoh0.pt",
            "model_2024-06-20-20-02_epoch1.pt",
            "model_2024-06-20-20-02_epoch2.pt",
            "model_2024-06-20-28-02_epoch1.pt",
            "model_2024-06-20-20-02_epoch3.pt",
            "model_2024-06-20-28-02_epoch0.pt",
        ]
        for file in checkpoint_files:
            with open(os.path.join(temp_dir, file), "w") as f:
                f.write("Test")

        # Assert that the most recent checkpoint is model.ckpt-4000
        most_recent_checkpoint = find_most_recent_checkpoint(temp_dir)
        assert most_recent_checkpoint == "model_2024-06-20-28-02_epoch1.pt"


class TestUnitProcessAdditionalArgs:
    """Tests for the process_additional_args function."""

    def test_no_additional_args(self) -> None:
        """Test for no additional args."""
        args = ()
        config = {"key1": "value1", "key2": "value2"}
        processed_config = process_additional_args(args, config)
        assert processed_config == config

    def test_additional_args(self) -> None:
        """Test for additional args."""
        args = ("key1=value3", "key2=value4")
        config = {"key1": "value1", "key2": "value2"}
        processed_config = process_additional_args(args, config)
        assert processed_config == {"key1": "value3", "key2": "value4"}

    def test_process_additional_args_with_none_type_config(self) -> None:
        """Test for additional args with None type config."""
        args = ("key1=['value1']",)
        config = {"key1": None}
        # Does not convert types if None in config
        processed_config = process_additional_args(args, config)
        assert processed_config == {"key1": "['value1']"}

    def test_process_additional_args_with_bool_type_config(self) -> None:
        """Test for additional args with bool type config."""
        args = ("key1=False",)
        config = {"key1": True}
        # Does not convert types if bool in config
        processed_config = process_additional_args(args, config)
        assert processed_config == {"key1": False}

    def test_process_additional_args_with_list_type_config(self) -> None:
        """Test for additional args with list type config."""
        args = ("key1=['value1']",)
        config = {"key1": ["value0"]}
        processed_config = process_additional_args(args, config)
        assert processed_config == {"key1": ["value1"]}

    def test_process_additional_args_with_null_type_config(self) -> None:
        """Test for additional args with null type config."""
        args = ("key1=null",)
        config_1 = {"key1": "value1"}
        config_2 = {"key1": ["value1"]}
        processed_config_1 = process_additional_args(args, config_1)
        processed_config_2 = process_additional_args(args, config_2)
        assert processed_config_1 == {"key1": None} and processed_config_2 == {
            "key1": None
        }


class TestIsDirectoryEmpty:
    """Tests for the is_directory_empty function."""

    def test_empty_directory(self) -> None:
        """Test for empty directory."""
        with tempfile.TemporaryDirectory() as empty_dir:
            # Assert that the directory is empty
            assert is_directory_empty(empty_dir)

    def test_non_empty_directory_1(self) -> None:
        """Test for non-empty directory."""
        with tempfile.TemporaryDirectory() as non_empty_dir:
            with open(os.path.join(non_empty_dir, "test.txt"), "w") as file:
                file.write("Test")
            assert not is_directory_empty(non_empty_dir)

    def test_non_empty_directory_2(self) -> None:
        """Test for non-empty directory."""
        with tempfile.TemporaryDirectory() as non_empty_dir:
            # make a diectory within the temporary directory
            os.mkdir(os.path.join(non_empty_dir, "test_dirr"))
            assert not is_directory_empty(non_empty_dir)

    def test_non_empty_directory_3(self) -> None:
        """Test for non-empty directory."""
        with tempfile.TemporaryDirectory() as non_empty_dir:
            # make a hidden file within the temporary directory
            with open(os.path.join(non_empty_dir, ".test.txt"), "w") as file:
                file.write("Test")
            assert not is_directory_empty(non_empty_dir)

    def test_directory_does_not_exist(self) -> None:
        """Test for directory that does not exist."""
        with pytest.raises(FileNotFoundError):
            is_directory_empty("fake_dir3198339")

    def test_get_commit_hash(self) -> None:
        """Testst that the GIT_COMMIT_HASH environment variable is set."""
        git_hash = get_commit_hash()
        assert git_hash != "no-hash"
