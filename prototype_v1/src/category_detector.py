"""
Simple keyword-based category detector.

This exists specifically because of a real bug: pure semantic similarity
search confused "How long does FTTH installation take?" with electrical
installation content, since both documents share heavy vocabulary overlap
("installation", "business days", "technician"). Detecting the likely
category from the query up front lets the retriever filter to the right
document category before running similarity search, instead of relying on
similarity alone to sort it out.
"""

from typing import Optional

CATEGORY_KEYWORDS = {
    "CCTV": ["cctv", "camera", "surveillance", "footage"],
    "FTTH": ["ftth", "fiber", "fibre", "broadband"],
    "Electrical": ["electrical", "wiring", "panel", "electrician"],
    "Solar": ["solar", "panel efficiency", "inverter"],
    "WiFi": ["wifi", "wi-fi", "wireless", "router", "access point", "mesh"],
    "BPO": ["bpo", "support desk", "ticket", "escalat"],
    "DataCenter": ["data center", "datacenter", "co-location", "colocation", "uptime sla"],
    "SmartCity": ["smart city", "streetlight", "traffic sensor"],
    "StructuredCabling": ["structured cabling", "tia/eia", "cabling certification"],
    "Cloud": ["finops", "cloud spend", "aws", "azure", "cost visibility"],
    "ITInfra": ["it infrastructure", "server hardware", "preventive maintenance"],
    "AccessControl": [
        "access control",
        "biometric",
        "access card",
        "fingerprint",
        "facial recognition",
    ],
}


def detect_category(query: str) -> Optional[str]:
    """Return the best-matching category for a query, or None if no strong match."""
    q = query.lower()
    scores = {}
    for category, keywords in CATEGORY_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in q)
        if score > 0:
            scores[category] = score
    if not scores:
        return None
    return max(scores, key=scores.get)
