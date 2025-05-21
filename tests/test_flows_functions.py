import unittest

from pipecat_flows.types import FlowsFunction

# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for FlowsFunction class."""


class TestFlowsFunction(unittest.TestCase):
    def test_name_is_set_from_function(self):
        """Test that FlowsFunction extracts the name from the function."""

        def test_function(args):
            return {}

        func = FlowsFunction(function=test_function)
        self.assertEqual(func.name, "test_function")

    def test_description_is_set_from_function(self):
        """Test that FlowsFunction extracts the description from the function."""

        def test_function(args):
            """This is a test function."""
            return {}

        func = FlowsFunction(function=test_function)
        self.assertEqual(func.description, "This is a test function.")


if __name__ == "__main__":
    unittest.main()
