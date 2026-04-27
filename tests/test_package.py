import pii_service


def test_package_imports():
    assert pii_service.__doc__ == "Offline PII redaction service."
