from types import SimpleNamespace

import pytest

from ssvep_fbcca.llm import ChatBackendError, ChatMessage, OpenAIChatBackend


class FakeResponses:
    def __init__(self, answer="好的"):
        self.answer = answer
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(output_text=self.answer)


class FakeClient:
    def __init__(self, responses):
        self.responses = responses


def test_send_uses_responses_api_and_preserves_history():
    responses = FakeResponses()
    backend = OpenAIChatBackend(client=FakeClient(responses), model="test-model")

    answer = backend.send("你好", history=[ChatMessage("user", "之前的问题")])

    assert answer == "好的"
    assert responses.calls == [{
        "model": "test-model",
        "input": [
            {"role": "user", "content": "之前的问题"},
            {"role": "user", "content": "你好"},
        ],
    }]


def test_empty_text_is_rejected_before_network_call():
    responses = FakeResponses()
    backend = OpenAIChatBackend(client=FakeClient(responses))

    with pytest.raises(ValueError):
        backend.send("  ")
    assert responses.calls == []


def test_empty_response_is_reported():
    backend = OpenAIChatBackend(client=FakeClient(FakeResponses(answer=" ")))

    with pytest.raises(ChatBackendError, match="空文本"):
        backend.send("你好")
