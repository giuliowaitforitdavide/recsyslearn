"""Base class of every test case, with assertions on the exact message of errors and warnings."""

from __future__ import annotations

import re
import unittest


def _exactly(message: str) -> str:
    """
    Build a regular expression that matches exactly the given message.

    :param message: The expected message.
    :type message: str
    :return: The escaped message, anchored at both ends.
    :rtype: str
    """

    return rf"\A{re.escape(message)}\Z"


class TestCase(unittest.TestCase):
    """
    unittest.TestCase with assertions that check both the type and the exact message.

    The names follow Django's ``assertRaisesMessage``, but the check is stricter: Django only
    requires the message to contain the expected text, here it must be equal to it (see
    AGENTS.md, section 4).
    """

    def assertRaisesMessage(
        self, expected_exception: type[BaseException], expected_message: str
    ) -> unittest.case._AssertRaisesContext:
        """
        Assert that the block raises expected_exception with exactly expected_message.

        Use it as a context manager: ``with self.assertRaisesMessage(ValueError, "…"): …``.

        :param expected_exception: The exception type that must be raised.
        :type expected_exception: type[BaseException]
        :param expected_message: The exact message, i.e. ``str(exception)``.
        :type expected_message: str
        :return: The context manager of assertRaisesRegex.
        :rtype: unittest.case._AssertRaisesContext
        """

        return self.assertRaisesRegex(expected_exception, _exactly(expected_message))

    def assertWarnsMessage(
        self, expected_warning: type[Warning], expected_message: str
    ) -> unittest.case._AssertWarnsContext:
        """
        Assert that the block emits expected_warning with exactly expected_message.

        Use it as a context manager: ``with self.assertWarnsMessage(UserWarning, "…"): …``.

        :param expected_warning: The warning category that must be emitted.
        :type expected_warning: type[Warning]
        :param expected_message: The exact message of the warning.
        :type expected_message: str
        :return: The context manager of assertWarnsRegex.
        :rtype: unittest.case._AssertWarnsContext
        """

        return self.assertWarnsRegex(expected_warning, _exactly(expected_message))
