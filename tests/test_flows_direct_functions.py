import asyncio
import unittest
from typing import Optional, TypedDict, Union

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
            return {"status": "success"}, None

        self.assertIsNone(FlowsDirectFunction.validate_function(my_function))
        func = FlowsDirectFunction(function=my_function)
        self.assertEqual(func.name, "my_function")

    def test_description_is_set_from_function(self):
        """Test that FlowsDirectFunction extracts the description from the function."""

        async def my_function_short_description(flow_manager: FlowManager):
            """This is a test function."""
            return {"status": "success"}, None

        self.assertIsNone(FlowsDirectFunction.validate_function(my_function_short_description))
        func = FlowsDirectFunction(function=my_function_short_description)
        self.assertEqual(func.description, "This is a test function.")

        async def my_function_long_description(flow_manager: FlowManager):
            """
            This is a test function.

            It does some really cool stuff.

            Trust me, you'll want to use it.
            """
            return {"status": "success"}, None

        self.assertIsNone(FlowsDirectFunction.validate_function(my_function_long_description))
        func = FlowsDirectFunction(function=my_function_long_description)
        self.assertEqual(
            func.description,
            "This is a test function.\n\nIt does some really cool stuff.\n\nTrust me, you'll want to use it.",
        )

    def test_properties_are_set_from_function(self):
        """Test that FlowsDirectFunction extracts the properties from the function."""

        async def my_function_no_params(flow_manager: FlowManager):
            return {"status": "success"}, None

        self.assertIsNone(FlowsDirectFunction.validate_function(my_function_no_params))
        func = FlowsDirectFunction(function=my_function_no_params)
        self.assertEqual(func.properties, {})

        async def my_function_simple_params(
            flow_manager: FlowManager, name: str, age: int, height: Union[float, None]
        ):
            return {"status": "success"}, None

        self.assertIsNone(FlowsDirectFunction.validate_function(my_function_simple_params))
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
            flow_manager: FlowManager,
            address_lines: list[str],
            nickname: str | int | float,
            extra: Optional[dict[str, str]],
        ):
            return {"status": "success"}, None

        self.assertIsNone(FlowsDirectFunction.validate_function(my_function_complex_params))
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

        class MyInfo1(TypedDict):
            name: str
            age: int

        class MyInfo2(TypedDict, total=False):
            name: str
            age: int

        async def my_function_complex_type_params(
            flow_manager: FlowManager, info1: MyInfo1, info2: MyInfo2
        ):
            return {"status": "success"}, None

        self.assertIsNone(FlowsDirectFunction.validate_function(my_function_complex_type_params))
        func = FlowsDirectFunction(function=my_function_complex_type_params)
        self.assertEqual(
            func.properties,
            {
                "info1": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"},
                    },
                    "required": ["name", "age"],
                },
                "info2": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"},
                    },
                },
            },
        )

    def test_required_is_set_from_function(self):
        """Test that FlowsDirectFunction extracts the required properties from the function."""

        async def my_function_no_params(flow_manager: FlowManager):
            return {"status": "success"}, None

        self.assertIsNone(FlowsDirectFunction.validate_function(my_function_no_params))
        func = FlowsDirectFunction(function=my_function_no_params)
        self.assertEqual(func.required, [])

        async def my_function_simple_params(
            flow_manager: FlowManager, name: str, age: int, height: Union[float, None] = None
        ):
            return {"status": "success"}, None

        self.assertIsNone(FlowsDirectFunction.validate_function(my_function_simple_params))
        func = FlowsDirectFunction(function=my_function_simple_params)
        self.assertEqual(func.required, ["name", "age"])

        async def my_function_complex_params(
            flow_manager: FlowManager,
            address_lines: Optional[list[str]],
            nickname: str | int = "Bud",
            extra: Optional[dict[str, str]] = None,
        ):
            return {"status": "success"}, None

        self.assertIsNone(FlowsDirectFunction.validate_function(my_function_complex_params))
        func = FlowsDirectFunction(function=my_function_complex_params)
        self.assertEqual(func.required, ["address_lines"])

    def test_property_descriptions_are_set_from_function(self):
        """Test that FlowsDirectFunction extracts the property descriptions from the function."""

        async def my_function(
            flow_manager: FlowManager, name: str, age: int, height: Union[float, None]
        ):
            """
            This is a test function.

            Args:
                name (str): The name of the person.
                age (int): The age of the person.
                height (float | None): The height of the person in meters. Defaults to None.
            """
            return {"status": "success"}, None

        self.assertIsNone(FlowsDirectFunction.validate_function(my_function))
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
            return {"status": "success"}, None

        with self.assertRaises(InvalidFunctionError):
            FlowsDirectFunction.validate_function(my_function_non_async)

        async def my_function_missing_flow_manager():
            return {"status": "success"}, None

        with self.assertRaises(InvalidFunctionError):
            FlowsDirectFunction.validate_function(my_function_missing_flow_manager)

        async def my_function_misplaced_flow_manager(foo: str, flow_manager: FlowManager):
            return {"status": "success"}, None

        with self.assertRaises(InvalidFunctionError):
            FlowsDirectFunction.validate_function(my_function_misplaced_flow_manager)

    def test_invoke_calls_function_with_args_and_flow_manager(self):
        """Test that FlowsDirectFunction.invoke calls the function with correct args and flow_manager."""

        called = {}

        class DummyFlowManager:
            pass

        async def my_function(flow_manager: DummyFlowManager, name: str, age: int):
            called["flow_manager"] = flow_manager
            called["name"] = name
            called["age"] = age
            return {"status": "success"}, None

        func = FlowsDirectFunction(function=my_function)
        flow_manager = DummyFlowManager()
        args = {"name": "Alice", "age": 30}

        result = asyncio.run(func.invoke(args=args, flow_manager=flow_manager))
        self.assertEqual(result, ({"status": "success"}, None))
        self.assertIs(called["flow_manager"], flow_manager)
        self.assertEqual(called["name"], "Alice")
        self.assertEqual(called["age"], 30)


if __name__ == "__main__":
    unittest.main()
