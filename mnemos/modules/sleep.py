"""
mnemos/modules/sleep.py — Sleep Daemon (Hippocampal-Neocortical Memory Consolidation).

Neuroscience Basis:
    During sleep — particularly slow-wave (deep) sleep — the hippocampus
    'replays' the day's episodic memories. This replay allows the neocortex
    to gradually extract generalizable semantic knowledge from specific
    episodic events (Squire & Alvarez, 1995; McClelland et al., 1995).

    The process is two-stage:
    1. Hippocampus (fast learner): stores raw, detailed episodic memories
       immediately during waking experience
    2. Neocortex (slow learner): extracts statistical regularities and
       semantic structure during sleep consolidation, then stores
       abstracted knowledge long-term

    The hippocampal representation is pruned after successful transfer —
    the detailed episodic trace is 'forgotten' in favor of the generalized
    semantic representation. This is why we remember *what happened* but
    forget the exact words spoken.

    The optional 'proceduralization' step mirrors the hippocampus-to-basal-
    ganglia transfer of declarative knowledge into procedural skills:
    explicit step-by-step reasoning becomes automatic reflexes.

Mnemos Implementation:
    An episodic buffer holds raw Interactions from the current session.
    During consolidation (triggered by inactivity or schedule):
    - The LLM is asked to extract permanent facts and user preferences
    - Semantic MemoryChunks are created from extracted facts
    - Chunks are stored in long-term store
    - The episodic buffer is cleared (garbage collection)
    - Optionally, repeated reasoning patterns are crystallized into Python tools
"""

from __future__ import annotations

import asyncio
import re
import time
from datetime import datetime, timezone
from typing import Any

from ..config import SleepConfig
from ..memory_safety import MemoryWriteFirewall
from ..types import ConsolidationResult, Interaction, MemoryChunk
from ..utils.embeddings import EmbeddingProvider
from ..utils.llm import LLMProvider
from ..utils.storage import MemoryStore


class SleepDaemon:
    """
    Asynchronous sleep consolidation daemon for episodic-to-semantic memory transfer.

    Maintains an episodic buffer of raw interactions (hippocampal store) and
    periodically consolidates them into semantic MemoryChunks in long-term storage
    (neocortical store), then clears the raw episodes.

    Args:
        store: Long-term memory store for consolidated facts.
        config: SleepConfig controlling consolidation schedule and behavior.
    """

    def __init__(
        self,
        store: MemoryStore,
        config: SleepConfig | None = None,
        write_firewall: MemoryWriteFirewall | None = None,
    ) -> None:
        self._store = store
        self._config = config or SleepConfig()
        self._write_firewall = write_firewall
        # Episodic buffer: fast, in-memory list of raw interactions
        # Analogous to hippocampal short-term episodic storage
        self._episodic_buffer: list[Interaction] = []
        # Timestamp of last consolidation
        self._last_consolidation: float = time.time()
        # Stats
        self._total_consolidations: int = 0
        self._total_facts_extracted: int = 0
        self._total_chunks_pruned: int = 0
        # Daemon task reference
        self._daemon_task: asyncio.Task[Any] | None = None

    def add_episode(self, interaction: Interaction) -> None:
        """
        Add a raw interaction to the episodic buffer.

        Called after each conversation turn to accumulate episodes for
        eventual consolidation. Analogous to hippocampal encoding during
        waking experience.

        Args:
            interaction: The interaction to buffer.
        """
        self._episodic_buffer.append(interaction)

    def get_episode_count(self) -> int:
        """Return the number of interactions currently in the episodic buffer."""
        return len(self._episodic_buffer)

    def should_consolidate(self) -> bool:
        """
        Determine whether conditions are met to trigger a consolidation cycle.

        Consolidation is appropriate when:
        1. Enough episodes have accumulated (min_episodes_before_consolidation)
        2. Enough time has passed since last consolidation (consolidation_interval_seconds)

        Returns:
            True if consolidation should run now.
        """
        episode_threshold = (
            len(self._episodic_buffer) >= self._config.min_episodes_before_consolidation
        )
        time_threshold = (
            time.time() - self._last_consolidation >= self._config.consolidation_interval_seconds
        )
        return episode_threshold and time_threshold

    def _format_episodes_for_prompt(self) -> str:
        """
        Format the episodic buffer as a readable transcript for the LLM.

        Takes up to max_episodes_per_consolidation episodes to avoid
        exceeding context windows on very large buffers.

        Returns:
            Formatted transcript string.
        """
        episodes = self._episodic_buffer[-self._config.max_episodes_per_consolidation :]
        lines = []
        for i, ep in enumerate(episodes, 1):
            ts = ep.timestamp.strftime("%H:%M:%S")
            lines.append(f"[{i}] {ts} {ep.role.upper()}: {ep.content}")
        return "\n".join(lines)

    async def consolidate(
        self,
        llm_provider: LLMProvider,
        embedder: EmbeddingProvider,
    ) -> ConsolidationResult:
        """
        Run a full consolidation cycle: extract facts, store as semantic chunks,
        prune raw episodes.

        This is the 'sleep' cycle:
        1. Format episodic buffer as a transcript
        2. Ask LLM to extract permanent facts, preferences, patterns
        3. Create a MemoryChunk for each extracted fact (with embedding)
        4. Store all chunks in long-term store
        5. Clear the episodic buffer
        6. Optionally generate procedural tool code

        Args:
            llm_provider: LLM for fact extraction and proceduralization.
            embedder: Embedding provider for new semantic chunks.

        Returns:
            ConsolidationResult with extracted facts, pruned count, tools generated.
        """
        start_time = time.time()

        if not self._episodic_buffer:
            return ConsolidationResult(
                facts_extracted=[],
                chunks_pruned=0,
                tools_generated=[],
                duration_seconds=0.0,
            )

        episodes_text = self._format_episodes_for_prompt()
        episodes_to_prune = len(self._episodic_buffer)

        # Step 1: Extract permanent facts via LLM
        consolidation_prompt = self._config.consolidation_prompt.format(episodes=episodes_text)
        try:
            raw_facts = await llm_provider.predict(consolidation_prompt)
        except Exception as e:
            # Consolidation failure: don't prune episodes
            return ConsolidationResult(
                facts_extracted=[],
                chunks_pruned=0,
                tools_generated=[],
                duration_seconds=time.time() - start_time,
            )

        # Step 2: Parse extracted facts (numbered list format)
        facts = self._parse_fact_list(raw_facts)

        # Step 3: Create semantic MemoryChunks from extracted facts
        semantic_chunks: list[MemoryChunk] = []
        for fact in facts:
            if not fact.strip():
                continue
            safe_fact = fact
            redactions: list[str] = []
            if self._write_firewall is not None:
                safety = self._write_firewall.apply(fact)
                if not safety.allowed:
                    continue
                safe_fact = safety.content
                redactions = [match.label for match in safety.matches]

            embedding = embedder.embed(safe_fact)
            chunk = MemoryChunk(
                content=safe_fact,
                embedding=embedding,
                metadata={
                    "source": "sleep_consolidation",
                    "consolidation_time": datetime.now(timezone.utc).isoformat(),
                    "episode_count": episodes_to_prune,
                    "safety_redactions": redactions,
                },
                salience=0.7,  # Consolidated facts have high salience
            )
            semantic_chunks.append(chunk)

        # Step 4: Store in long-term memory
        for chunk in semantic_chunks:
            self._store.store(chunk)

        # Step 5: Clear episodic buffer (active forgetting of raw episodes)
        self._episodic_buffer.clear()

        # Step 6: Optional proceduralization
        tools_generated: list[str] = []
        if self._config.enable_proceduralization and facts:
            tool_code = await self.proceduralize(llm_provider, episodes_text)
            if tool_code:
                tools_generated.append(tool_code)

        # Update state
        self._last_consolidation = time.time()
        self._total_consolidations += 1
        self._total_facts_extracted += len(facts)
        self._total_chunks_pruned += episodes_to_prune

        return ConsolidationResult(
            facts_extracted=facts,
            chunks_pruned=episodes_to_prune,
            tools_generated=tools_generated,
            duration_seconds=time.time() - start_time,
        )

    def _parse_fact_list(self, raw_text: str) -> list[str]:
        """
        Parse a numbered list from LLM output into individual fact strings.

        Handles formats like:
        - "1. fact one"
        - "1) fact one"
        - "- fact one"
        - Bare lines without numbers

        Args:
            raw_text: Raw LLM response containing a fact list.

        Returns:
            List of stripped fact strings.
        """
        lines = raw_text.strip().split("\n")
        facts = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Remove leading number/bullet markers
            cleaned = re.sub(r"^(\d+[\.\)]\s*|-\s*|\*\s*)", "", line)
            cleaned = cleaned.strip()
            if cleaned:
                facts.append(cleaned)
        return facts

    async def proceduralize(
        self,
        llm_provider: LLMProvider,
        episodes_text: str | None = None,
    ) -> str | None:
        """
        Identify repeated reasoning patterns and generate Python tool code.

        This mirrors the hippocampus-to-basal-ganglia transfer where
        repeated explicit reasoning becomes implicit procedural skill.
        The generated code is a Python function that automates the pattern.

        WARNING: The returned code is UNTRUSTED LLM output. It has NOT been
        validated, sandboxed, or audited for correctness or safety. Callers
        MUST NEVER execute the returned code without thorough human review.
        Treat it as a *suggestion* that requires manual vetting before use
        in any runtime environment.

        Args:
            llm_provider: LLM for code generation.
            episodes_text: Pre-formatted transcript (optional — uses buffer if None).

        Returns:
            Python code string if a pattern was found, None otherwise.
        """
        if episodes_text is None:
            if not self._episodic_buffer:
                return None
            episodes_text = self._format_episodes_for_prompt()

        prompt = self._config.proceduralization_prompt.format(episodes=episodes_text)

        try:
            response = await llm_provider.predict(prompt)
        except Exception:
            return None

        response = response.strip()

        if response.upper() == "NO_PATTERN" or not response:
            return None

        # Validate it looks like Python code
        if "def " in response or "import " in response or "return " in response:
            return response

        return None

    async def run_daemon(
        self,
        llm_provider: LLMProvider,
        embedder: EmbeddingProvider,
        interval: float | None = None,
        stop_event: asyncio.Event | None = None,
    ) -> None:
        """
        Async loop that periodically checks for consolidation conditions and runs.

        Designed to run as a background task:
            task = asyncio.create_task(daemon.run_daemon(llm, embedder))

        Args:
            llm_provider: LLM for consolidation.
            embedder: Embedding provider for new chunks.
            interval: Check interval in seconds (default: config.consolidation_interval_seconds / 10).
            stop_event: Optional asyncio.Event to signal graceful shutdown.
        """
        check_interval = interval or (self._config.consolidation_interval_seconds / 10)
        self._daemon_task = asyncio.current_task()

        while True:
            if stop_event and stop_event.is_set():
                break

            try:
                if self.should_consolidate():
                    await self.consolidate(llm_provider, embedder)
            except Exception:
                pass  # Daemon must not crash

            try:
                await asyncio.sleep(check_interval)
            except asyncio.CancelledError:
                break

    def stop_daemon(self) -> None:
        """Cancel the background daemon task if running."""
        if self._daemon_task and not self._daemon_task.done():
            self._daemon_task.cancel()

    def get_stats(self) -> dict[str, Any]:
        """Return statistics about the SleepDaemon module."""
        time_since_last = time.time() - self._last_consolidation
        return {
            "module": "SleepDaemon",
            "episodic_buffer_size": len(self._episodic_buffer),
            "total_consolidations": self._total_consolidations,
            "total_facts_extracted": self._total_facts_extracted,
            "total_chunks_pruned": self._total_chunks_pruned,
            "seconds_since_last_consolidation": round(time_since_last, 1),
            "should_consolidate_now": self.should_consolidate(),
            "proceduralization_enabled": self._config.enable_proceduralization,
        }
