import unittest
from typing import Optional, Union

from pipecat_flows.exceptions import InvalidFunctionError
from pipecat_flows.manager import FlowManager
from pipecat_flows.types import FlowsDirectFunction

# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for FlowsDirectFunction class."""


class TestFlowsDirectFunction(unittest.TestCase):
    def test_name_is_set_from_function(self):
        """Test that FlowsDirectFunction extracts the name from the function."""

        async def my_function(flow_manager: FlowManager):
            return {}, None

        func = FlowsDirectFunction(function=my_function)
        self.assertEqual(func.name, "my_function")

    def test_description_is_set_from_function(self):
        """Test that FlowsDirectFunction extracts the description from the function."""

        async def my_function_short_description(flow_manager: FlowManager):
            """This is a test function."""
            return {}, None

        func = FlowsDirectFunction(function=my_function_short_description)
        self.assertEqual(func.description, "This is a test function.")

        async def my_function_long_description(flow_manager: FlowManager):
            """
            This is a test function.

            It does some really cool stuff.

            Trust me, you'll want to use it.
            """
            return {}, None

        func = FlowsDirectFunction(function=my_function_long_description)
        self.assertEqual(
            func.description,
            "This is a test function.\n\nIt does some really cool stuff.\n\nTrust me, you'll want to use it.",
        )

    def test_properties_are_set_from_function(self):
        """Test that FlowsDirectFunction extracts the properties from the function."""

        async def my_function_no_params(flow_manager: FlowManager):
            return {}, None

        func = FlowsDirectFunction(function=my_function_no_params)
        self.assertEqual(func.properties, {})

        async def my_function_simple_params(
            name: str, age: int, height: Union[float, None], flow_manager: FlowManager
        ):
            return {}, None

        func = FlowsDirectFunction(function=my_function_simple_params)
        self.assertEqual(
            func.properties,
            {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "height": {"anyOf": [{"type": "number"}, {"type": "null"}]},
            },
        )

        async def my_function_complex_params(
            address_lines: list[str],
            nickname: str | int | float,
            extra: Optional[dict[str, str]],
            flow_manager: FlowManager,
        ):
            return {}, None

        func = FlowsDirectFunction(function=my_function_complex_params)
        self.assertEqual(
            func.properties,
            {
                "address_lines": {"type": "array", "items": {"type": "string"}},
                "nickname": {
                    "anyOf": [{"type": "string"}, {"type": "integer"}, {"type": "number"}]
                },
                "extra": {
                    "anyOf": [
                        {"type": "object", "additionalProperties": {"type": "string"}},
                        {"type": "null"},
                    ]
                },
            },
        )

    def test_required_is_set_from_function(self):
        """Test that FlowsDirectFunction extracts the required properties from the function."""

        async def my_function_no_params(flow_manager: FlowManager):
            return {}, None

        func = FlowsDirectFunction(function=my_function_no_params)
        self.assertEqual(func.required, [])

        async def my_function_simple_params(
            name: str, age: int, height: Union[float, None] = None, flow_manager: FlowManager = None
        ):
            return {}, None

        func = FlowsDirectFunction(function=my_function_simple_params)
        self.assertEqual(func.required, ["name", "age"])

        async def my_function_complex_params(
            address_lines: list[str],
            nickname: str | int | None = "Bud",
            extra: Optional[dict[str, str]] = None,
            flow_manager: FlowManager = None,
        ):
            return {}, None

        func = FlowsDirectFunction(function=my_function_complex_params)
        self.assertEqual(func.required, ["address_lines"])

    def test_property_descriptions_are_set_from_function(self):
        """Test that FlowsDirectFunction extracts the property descriptions from the function."""

        async def my_function(
            name: str, age: int, height: Union[float, None], flow_manager: FlowManager
        ):
            """
            This is a test function.

            Args:
                name (str): The name of the person.
                age (int): The age of the person.
                height (float | None): The height of the person in meters. Defaults to None.
            """
            return {}, None

        func = FlowsDirectFunction(function=my_function)

        # Validate that the function description is still set correctly even with the longer docstring
        self.assertEqual(func.description, "This is a test function.")

        # Validate that the property descriptions are set correctly
        self.assertEqual(
            func.properties,
            {
                "name": {"type": "string", "description": "The name of the person."},
                "age": {"type": "integer", "description": "The age of the person."},
                "height": {
                    "anyOf": [{"type": "number"}, {"type": "null"}],
                    "description": "The height of the person in meters. Defaults to None.",
                },
            },
        )

    def test_invalid_functions_fail_validation(self):
        """Test that invalid functions fail FlowsDirectFunction validation."""

        def my_function_non_async(flow_manager: FlowManager):
            return {}, None

        with self.assertRaises(InvalidFunctionError):
            FlowsDirectFunction.validate_function(my_function_non_async)

        async def my_function_missing_flow_manager():
            return {}, None

        with self.assertRaises(InvalidFunctionError):
            FlowsDirectFunction.validate_function(my_function_missing_flow_manager)


if __name__ == "__main__":
    unittest.main()
