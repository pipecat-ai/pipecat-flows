from typing import Optional, Union
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

        def my_function():
            return {}

        func = FlowsFunction(function=my_function)
        self.assertEqual(func.name, "my_function")

    def test_description_is_set_from_function(self):
        """Test that FlowsFunction extracts the description from the function."""

        def my_function():
            """This is a test function."""
            return {}

        func = FlowsFunction(function=my_function)
        self.assertEqual(func.description, "This is a test function.")

    def test_properties_are_set_from_function(self):
        """Test that FlowsFunction extracts the properties from the function."""

        def my_function_no_params():
            return {}

        func = FlowsFunction(function=my_function_no_params)
        self.assertEqual(func.properties, {})

        def my_function_simple_params(name: str, age: int, height: Union[float, None]):
            return {}

        func = FlowsFunction(function=my_function_simple_params)
        self.assertEqual(
            func.properties,
            {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "height": {"type": "number", "nullable": True},
            },
        )

        def my_function_complex_params(
            address_lines: list[str], nickname: Union[str, int], extra: Optional[dict[str, str]]
        ):
            return {}

        func = FlowsFunction(function=my_function_complex_params)
        self.assertEqual(
            func.properties,
            {
                "address_lines": {"type": "array", "items": {"type": "string"}},
                "nickname": {"anyOf": [{"type": "string"}, {"type": "integer"}]},
                "extra": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                    "nullable": True,
                },
            },
        )


if __name__ == "__main__":
    unittest.main()
