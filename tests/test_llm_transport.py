from des_multi_agent.llm.transport import RequestTransport


def test_transport_includes_api_key_header_by_default():
    captured = {}

    def fake_request(url, payload, api_key=None, timeout_seconds=None):
        captured["url"] = url
        captured["payload"] = payload
        captured["api_key"] = api_key
        captured["timeout_seconds"] = timeout_seconds
        return "{}"

    transport = RequestTransport(request_fn=fake_request, timeout_seconds=12.5)
    transport.post_json("https://example.com", {"hello": "world"}, api_key="secret")

    assert captured["api_key"] == "secret"
    assert captured["timeout_seconds"] == 12.5


def test_transport_can_omit_api_key_header():
    captured = {}

    def fake_request(url, payload, api_key=None, timeout_seconds=None):
        captured["api_key"] = api_key
        return "{}"

    transport = RequestTransport(request_fn=fake_request, timeout_seconds=3.0)
    transport.post_json("https://example.com", {"hello": "world"}, api_key="secret", include_api_key_in_header=False)

    assert captured["api_key"] is None
