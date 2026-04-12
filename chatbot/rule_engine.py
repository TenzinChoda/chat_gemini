"""Layer 1: keyword-based FAQ rules for Bhutan Telecom."""

from __future__ import annotations

import re

FAQ_RULES: list[dict] = [
    {
        "intent": "sim_activation",
        "keywords": ["sim", "activate", "activation", "new sim"],
        "response": (
            "For SIM activation or a new B-Mobile SIM, visit any Bhutan Telecom service center "
            "with your original CID (or valid e-copy). You can also check SIM and eSIM options on "
            "the BT website under Mobile → SIM. For tourist SIMs, passport copy is required."
        ),
    },
    {
        "intent": "data_package",
        "keywords": ["data", "package", "4g", "5g", "internet pack"],
        "response": (
            "You can buy prepaid data packs via E-Load, B-Wallet, B-Ngul, Chharo, MBoB, mPAY, or TPAY; "
            "paper vouchers (e.g. 49, 99) may also be available. Dial *170# to check balances. "
            "Student and special plans may apply—see bt.bt FAQ for current packages."
        ),
    },
    {
        "intent": "balance_deduction",
        "keywords": ["balance", "deduction", "deducted", "money gone"],
        "response": (
            "Data usage includes all internet activity. Check usage on your phone under Settings → Data usage, "
            "and ensure background data and auto-updates are limited. For billing disputes, call 1600 (toll-free) "
            "or visit a BT counter with your number and bill details."
        ),
    },
    {
        "intent": "network_issue",
        "keywords": ["network", "signal", "call drop", "slow internet", "speed"],
        "response": (
            "BT covers all 20 districts; 4G uses band 3 (1800 MHz). Try toggling airplane mode, "
            "check APN (Name: B-Mobile, APN: internet), and test in another location. If issues persist, "
            "call 1600 or visit a service center."
        ),
    },
    {
        "intent": "sim_barred",
        "keywords": ["barred", "bar", "unbar", "blocked", "suspended"],
        "response": (
            "If your SIM is barred or suspended, visit a BT service center with your CID or call "
            "the contact center (1600) with your mobile number ready. Do not share OTPs or PINs with anyone."
        ),
    },
    {
        "intent": "bill_enquiry",
        "keywords": ["bill", "billing", "invoice", "payment", "outstanding"],
        "response": (
            "Postpaid bills show usage and charges on your monthly invoice (often emailed). "
            "For leased line or corporate billing, contact BT with your account ID. GST may apply as per invoice notes."
        ),
    },
    {
        "intent": "puk_code",
        "keywords": ["puk", "puk code", "sim locked", "pin locked"],
        "response": (
            "If your SIM is PIN/PUK locked, do not guess the codes—visit a BT counter with CID proof "
            "or call customer care (1600) for safe unlocking steps."
        ),
    },
    {
        "intent": "apn_setting",
        "keywords": ["apn", "setting", "configure internet"],
        "response": (
            "Set APN to Name: B-Mobile (or BTL), APN: internet, authentication PAP/CHAP if prompted. "
            "Path: Settings → Mobile networks → Access Point Names. Save and select this APN, then restart mobile data."
        ),
    },
    {
        "intent": "eload_voucher",
        "keywords": ["eload", "voucher", "recharge", "topup"],
        "response": (
            "E-Load lets retailers and customers recharge talk-time and data via SMS; minimum amounts start from Nu. 5. "
            "Dial *170# or call 123 after recharge to verify balance. Dealers can use *130# menus as per BT instructions."
        ),
    },
    {
        "intent": "fwa",
        "keywords": ["fwa", "home wifi", "home internet", "broadband"],
        "response": (
            "For home internet and broadband (including AirFiber and leased options), see bt.bt under Internet. "
            "Plans vary by speed and quota—visit a counter or call 1600 for coverage and application steps."
        ),
    },
    {
        "intent": "esim",
        "keywords": ["esim", "e-sim", "digital sim"],
        "response": (
            "eSIM is supported on compatible handsets. Check device settings for eSIM or dual SIM options, "
            "or ask at a BT counter whether your model supports BTL eSIM provisioning."
        ),
    },
    {
        "intent": "greeting",
        "keywords": ["hello", "hi", "hey", "good morning", "good afternoon"],
        "response": (
            "Hello! I am the BT Virtual Assistant. Ask me about mobile, data, billing, SIM, or network—"
            "or say what you need help with."
        ),
    },
]


def _keyword_match(text: str, kw: str) -> bool:
    """Avoid false positives (e.g. 'sim' matching inside 'esim')."""
    kw = kw.lower().strip()
    if not kw:
        return False
    if " " in kw:
        return kw in text
    return bool(re.search(rf"(?<!\w){re.escape(kw)}(?!\w)", text))


def get_rule_response(user_message: str) -> tuple[str | None, str | None]:
    """
    Match FAQ rules by counting keyword hits in the lowercased message.
    Returns (response_text, intent_name) or (None, None).
    """
    text = (user_message or "").lower()
    best_intent: str | None = None
    best_score = 0
    best_response: str | None = None
    for rule in FAQ_RULES:
        score = sum(1 for kw in rule["keywords"] if _keyword_match(text, kw))
        # Prefer later rules on equal score so more specific intents (e.g. sim_barred) can win over generic "sim".
        if score >= best_score:
            best_score = score
            best_intent = rule["intent"]
            best_response = rule["response"]
    if best_score < 1:
        return None, None
    return best_response, best_intent
