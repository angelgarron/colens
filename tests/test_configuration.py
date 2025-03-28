import pytest

from colens.configuration import UnknownSubsectionError, _check_unknown_entries


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
