import pytest

from colens.configuration import (
    UnknownSubsectionError,
    _check_unknown_entries,
    _construct_subsection_dict,
)


def test_construct_subsection_dict(capsys):
    expected = {"arg2": "value2"}
    assert (
        _construct_subsection_dict(
            match_args=("arg1", "arg2", "arg3"), obj={"arg2": "value2"}
        )
        == expected
    )
    out, err = capsys.readouterr()
    assert out == "Leaving arg1, arg3 with their default values.\n"

    # don't print warning if no argument is left with default value
    expected = {"arg2": "value2"}
    assert (
        _construct_subsection_dict(match_args=("arg2",), obj={"arg2": "value2"})
        == expected
    )
    out, err = capsys.readouterr()
    assert out == ""


def test_check_unknown_entries_subsection_dict_unknown_subsection():
    expected = ("subsection1", "subsection2")
    given = (
        "subsection_not_exist1",
        "subsection1",
        "subsection2",
    )
    with pytest.raises(
        UnknownSubsectionError,
        match="The following subsections are not known: subsection_not_exist1",
    ):
        _check_unknown_entries(
            expected=expected, given=given, error=UnknownSubsectionError
        )


def test_check_unknown_entries_subsection_dict_unknown_subsections():
    expected = ("subsection1", "subsection2")
    given = (
        "subsection_not_exist1",
        "subsection_not_exist2",
        "subsection1",
        "subsection2",
    )
    with pytest.raises(
        UnknownSubsectionError,
        match="The following subsections are not known: subsection_not_exist1, subsection_not_exist2",
    ):
        _check_unknown_entries(
            expected=expected, given=given, error=UnknownSubsectionError
        )
