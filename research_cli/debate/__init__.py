"""A bounded multi-agent debate over one EvidenceBundle.

Agents argue from numbered facts and may not fetch, invent or advise; code
checks every sentence they produce, ranks the cases with a mostly
deterministic rubric, and supplies every likelihood figure itself.
"""

from .models import AgentCase, DebateResult
from .protocol import run_debate

__all__ = ["AgentCase", "DebateResult", "run_debate"]
