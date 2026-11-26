"""Entrypoints for native trajectory readers."""

from atm.parsers import mini_swe_agent, openhands, swe_agent
from atm.parsers.base import Parser

PARSERS: dict[str, Parser] = {
    "SWE-agent": swe_agent.parse,
    "OpenHands": openhands.parse,
    "mini-SWE-agent": mini_swe_agent.parse,
}
