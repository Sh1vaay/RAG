from src.category_detector import detect_category


def test_detects_cctv():
    assert detect_category("What's included in the CCTV AMC plan?") == "CCTV"


def test_detects_ftth():
    assert detect_category("How long does FTTH installation take?") == "FTTH"


def test_detects_electrical():
    assert detect_category("Do you handle electrical wiring for offices?") == "Electrical"


def test_detects_wifi_variants():
    assert detect_category("Do you provide Wi-Fi solutions for warehouses?") == "WiFi"
    assert detect_category("Can you set up a wifi mesh network?") == "WiFi"


def test_returns_none_for_ambiguous_query():
    # No category keyword present - this is the known limitation documented
    # in the project write-up: a genuinely ambiguous query gets no filter.
    result = detect_category("How many business days does installation take?")
    assert result is None


def test_case_insensitive():
    assert detect_category("CCTV CAMERA REPAIR TIME") == "CCTV"
