from langchain_core.messages import AIMessage
from langchain_core.messages.utils import merge_message_runs


def test_merge_message_runs_preserves_invalid_tool_calls() -> None:
    invalid_call = {
        "name": "search_notes",
        "args": "{bad json",
        "id": "bad-1",
        "error": "parse error",
    }
    first = AIMessage(content="calling", invalid_tool_calls=[invalid_call])
    second = AIMessage(content="anything")

    merged = merge_message_runs([first, second])[0]

    assert merged.invalid_tool_calls == [invalid_call]
    assert merged.tool_calls == []


def test_merge_message_runs_keeps_invalid_calls_from_both_messages() -> None:
    first_call = {
        "name": "search_notes",
        "args": "{bad json",
        "id": "bad-1",
        "error": "parse error",
    }
    second_call = {
        "name": "lookup",
        "args": "{also bad",
        "id": "bad-2",
        "error": "parse error",
    }

    merged = merge_message_runs(
        [
            AIMessage(content="first", invalid_tool_calls=[first_call]),
            AIMessage(content="second", invalid_tool_calls=[second_call]),
        ]
    )[0]

    assert merged.invalid_tool_calls == [first_call, second_call]
    assert merged.tool_calls == []
