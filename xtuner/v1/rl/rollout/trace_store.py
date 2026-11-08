import time
import asyncio
import gc
from functools import wraps
from uuid import uuid4
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import ray
from pydantic import BaseModel, ConfigDict, Field

from xtuner.v1.data_proto.rl_data import RolloutState, discard_rollout_state
from xtuner.v1.utils import get_logger


_STORE_NAME = "rollout_trace_store"
_STORE_NAMESPACE = "xtuner_rollout"
_handle_cache: Any = None


def _free_ray_refs(obj: Any):
    """Recursively free ray.ObjectRef instances trapped inside an object.

    Args:
        obj (Any): The object that may contain ray.ObjectRef references (e.g., dict, list, tuple).
    """
    if isinstance(obj, ray.ObjectRef):
        try:
            ray.internal.free([obj], local_only=False)
        except Exception as e:
            get_logger().error(f"Failed to free Ray ObjectRef {obj}: {e}")
    elif isinstance(obj, TokenizedSegment):
        _free_ray_refs(obj.expert_key)
    elif isinstance(obj, dict):
        for v in obj.values():
            _free_ray_refs(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _free_ray_refs(v)
    elif hasattr(obj, "model_dump"):  # Pydantic v2
        for v in obj.model_dump().values() if hasattr(obj.model_dump, "__call__") else {}.values():
            _free_ray_refs(v)
    elif hasattr(obj, "dict") and callable(getattr(obj, "dict")):  # Pydantic v1
        for v in obj.dict().values():
            _free_ray_refs(v)
    elif hasattr(obj, "__dict__"):
        for v in vars(obj).values():
            _free_ray_refs(v)


def _common_prefix_len(left: str, right: str) -> int:
    for idx, (left_ch, right_ch) in enumerate(zip(left, right)):
        if left_ch != right_ch:
            return idx
    return min(len(left), len(right))


def _compact_text(text: str, limit: int = 256) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return f"{text[:half]} ... {text[-half:]}"


def _mismatch_window(left: str, right: str, *, radius: int = 160) -> dict[str, Any]:
    prefix_len = _common_prefix_len(left, right)
    start = max(prefix_len - radius, 0)
    end = prefix_len + radius
    return {
        "common_prefix_len": prefix_len,
        "prompt_window": left[start:end],
        "key_window": right[start:end],
    }


def _resolve_routed_experts(expert_keys: List[Any], session_id: str) -> List[Any] | None:
    """Collapse per-node ``expert_key`` values into the exported
    ``routed_experts``.

    The SessionServer's single-phase write guarantees that, within one session,
    either every trace node carries routed experts (MoE) or none of them do
    (dense). This helper enforces that invariant at export time.

    Args:
        expert_keys (List[Any]): One ``expert_key`` per trace node, in path order.
        session_id (str): The session identifier, used only for error messages.

    Returns:
        List[Any] | None: The per-node expert refs for a MoE trace, or ``None``
        when the trace carries no routed experts at all (dense model), so the
        trainer skips routed-expert consumption entirely.
    """
    present = [expert_key is not None for expert_key in expert_keys]
    if not any(present):
        return None
    if all(present):
        return expert_keys
    raise ValueError(
        f"Inconsistent routed_experts in session {session_id!r}: {sum(present)}/{len(present)} trace nodes carry "
        "routed experts; the trace mixes MoE turns with missing-expert turns and cannot be exported."
    )


class TokenizedSegment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    token_ids: List[int]
    labels: List[int] | None = Field(default=None, repr=False)
    logprobs: List[float] | None = Field(default=None, repr=False)
    expert_key: Any = Field(default=None, repr=False)
    length: int | None = None
    model_step: int | None = None
    policy_version: int | None = None
    request_id: str | None = None

    def model_post_init(self, _):
        if self.labels is None:
            self.labels = [-100] * len(self.token_ids)
        if self.logprobs is None:
            self.logprobs = [0.0] * len(self.token_ids)
        if self.length is None:
            self.length = len(self.token_ids)
        assert len(self.token_ids) == len(self.labels) == len(self.logprobs)


@dataclass
class TreeNode:
    """A node in the prefix tree (Trie).

    Attributes:
        value (Optional[Any]): The value stored in the node.
        parent (Optional["TreeNode"]): The parent node.
        children (dict[str, "TreeNode"]): The child nodes, mapping string tokens to TreeNodes.
        created_at (float): The timestamp when the node was created.
    """

    value: Optional[Any]
    parent: Optional["TreeNode"]
    children: dict[str, "TreeNode"] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


class Trie:
    def __init__(self):
        """Initialize the prefix tree (Trie)."""
        self.root = TreeNode(value=None, parent=None)

    def keys(self) -> List[str]:
        """Get all keys (i.e., strings) stored in the Trie."""
        result = []

        def _collect(node: TreeNode, path: List[str]):
            if node.value is not None:
                result.append("".join(path))
            for token, child in node.children.items():
                _collect(child, path + [token])

        _collect(self.root, [])
        return result

    def insert(self, key: str, value: Any) -> None:
        """Insert a (key, value) pair into the Trie.

        Args:
            key (str): The key string to insert.
            value (Any): The value to store at the destination node.
        """
        node = self.root
        while key:
            best_prefix = ""
            for token in node.children:
                if key.startswith(token) and len(token) > len(best_prefix):
                    best_prefix = token

            if best_prefix:
                node = node.children[best_prefix]
                key = key[len(best_prefix) :]
            else:
                # create a new child node for the remaining key
                node.children[key] = TreeNode(value=None, parent=node)
                node = node.children[key]
                break

        node.value = value

    def search(self, text: str, filter_none: bool = False) -> Tuple[str, List["TreeNode"]]:
        """Search for the longest prefix matching the input text.

        Args:
            text (str): The input string to search for.
            filter_none (bool): If True, only returns nodes whose value is not None
                (i.e., nodes where a value was explicitly inserted).

        Returns:
            Tuple[str, List["TreeNode"]]: A tuple containing the matched longest prefix string
            and a list of nodes along the matched path (excluding the root).
        """
        node = self.root
        key_path, matched_nodes = [], []

        while text:
            best_prefix = ""
            for token in node.children:
                if text.startswith(token) and len(token) > len(best_prefix):
                    best_prefix = token

            if best_prefix:
                node = node.children[best_prefix]
                matched_nodes.append(node)
                key_path.append(best_prefix)
                text = text[len(best_prefix) :]
            else:
                break

        if filter_none:
            matched_nodes = [n for n in matched_nodes if n.value is not None]

        return "".join(key_path), matched_nodes

    def release(self, key: str | None = None):
        """Release nodes and free associated resources (e.g. Ray objects) for a
        given key.

        If the key is None, the entire Trie is released. Prunes empty branches bottom-up.

        Args:
            key (Optional[str]): The key string whose associated nodes should be released.
                If None, releases the entire tree.
        """

        def _free_subtree(node: TreeNode):
            for child in node.children.values():
                _free_subtree(child)
            if node.value is not None:
                _free_ray_refs(node.value)
                node.value = None
            node.children.clear()

        if key is None:
            _free_subtree(self.root)
            return

        node = self.root
        path = []
        while key:
            best_prefix = ""
            for token in node.children:
                if key.startswith(token) and len(token) > len(best_prefix):
                    best_prefix = token

            if best_prefix:
                path.append((node, best_prefix))
                node = node.children[best_prefix]
                key = key[len(best_prefix) :]
            else:
                return

        if node.value is not None:
            _free_ray_refs(node.value)
            node.value = None

        for parent, token in reversed(path):
            child_node = parent.children[token]
            if child_node.value is None and not child_node.children:
                del parent.children[token]
            else:
                break


@dataclass(frozen=True)
class SegmentPath:
    """Actor-local immutable history; never serialize parents across Ray."""
    parent: "SegmentPath | None"
    segment: TokenizedSegment


def path_segments(path):
    result = []
    while path is not None:
        result.append(path.segment)
        path = path.parent
    return reversed(result)


def _trace_operation(method):
    @wraps(method)
    def measured(self, *args, **kwargs):
        if not self._metrics_enabled:
            return method(self, *args, **kwargs)
        started = time.perf_counter()
        try:
            return method(self, *args, **kwargs)
        finally:
            elapsed = time.perf_counter() - started
            stat = self._operation_metrics.setdefault(method.__name__, {"calls": 0, "seconds": 0.0, "max_seconds": 0.0})
            stat["calls"] += 1
            stat["seconds"] += elapsed
            stat["max_seconds"] = max(stat["max_seconds"], elapsed)
    return measured


@ray.remote(num_cpus=0)
class RolloutTraceStore:
    """Actor for managing trace stores (Tries) across different rollout
    sessions."""

    def __init__(self, byte_limit: int | None = None, topology: tuple[int, int] | None = None):
        """Initialize the rollout trace store actor."""
        self._metrics_enabled = os.environ.get("XTUNER_TRACE_METRICS", "0") == "1"
        self._operation_metrics = {}
        self._gc_metrics = {"calls": 0, "seconds": 0.0, "max_seconds": 0.0}
        if self._metrics_enabled:
            gc_stat = self._gc_metrics
            gc_started = {}
            def record_gc(phase, info):
                generation = info["generation"]
                if phase == "start":
                    gc_started[generation] = time.perf_counter()
                elif generation in gc_started:
                    duration = time.perf_counter() - gc_started.pop(generation)
                    gc_stat["calls"] += 1
                    gc_stat["seconds"] += duration
                    gc_stat["max_seconds"] = max(gc_stat["max_seconds"], duration)
            gc.callbacks.append(record_gc)
        self.sessions: Dict[str, Trie] = {}
        self.objects: Dict[str, ray.ObjectRef] = {}
        self.updated_at: Dict[str, float] = {}
        # Full routing belongs to an endpoint request, not to the reused text
        # nodes. The nodes keep the original generation probabilities/versions.
        self.turn_traces: dict[str, dict[str, dict]] = {}
        self.turn_reservations: dict[str, int] = {}
        self.pending_prefixes = {}
        self.session_pending = {}
        self.session_epochs = {}
        self.closed_sessions = set()
        self.cleanup_jobs = {}
        self._request_count = 0
        self._routing_bytes = 0
        self._reserved_bytes = 0
        self._segment_count = 0
        self._token_count = 0
        self.session_segments = {}
        self.smoke_held_sessions: set[str] = set()
        self.turn_owners: dict[str, str] = {}
        self.cancelled_turn_sessions: dict[str, str] = {}
        total_limit = int(os.environ.get("XTUNER_TRACE_HARD_BYTES", str(128 * 1024**3)))
        self.turn_byte_limit = total_limit if byte_limit is None else byte_limit
        self._topology = topology or (1, total_limit)

    def trace_topology(self):
        return self._topology

    def reserve_turn(self, request_id: str, byte_count: int, session_id: str | None = None):
        if session_id is not None:
            pending = self.pending_prefixes.get(request_id)
            if (session_id in self.closed_sessions or session_id in self.cancelled_turn_sessions
                    or pending is None or pending[0] != session_id):
                raise ValueError("reservation has no active prepared turn")
        if byte_count < 0:
            raise ValueError("negative reservation")
        memory = self.turn_memory()
        if memory["routing_bytes"] + memory["reserved_bytes"] - self.turn_reservations.get(request_id, 0) + byte_count > self.turn_byte_limit:
            raise MemoryError("turn routing storage budget reached")
        self._reserved_bytes += byte_count - self.turn_reservations.get(request_id, 0)
        self.turn_reservations[request_id] = byte_count

    def claim_turn_session(self, session_id: str, owner: str):
        reason = "session_closed" if session_id in self.closed_sessions else self.cancelled_turn_sessions.get(session_id)
        if reason:
            return {"acquired": False, "cancel_reason": reason}
        current = self.turn_owners.setdefault(session_id, owner)
        return {"acquired": current == owner, "cancel_reason": None}

    def release_turn_session(self, session_id: str, owner: str):
        if self.turn_owners.get(session_id) == owner:
            del self.turn_owners[session_id]

    def cancel_turn_session(self, session_id: str, reason: str):
        self.cancelled_turn_sessions[session_id] = reason

    def smoke_hold_session(self, session_id: str):
        self.smoke_held_sessions.add(session_id)

    def smoke_required_version(self, session_id: str):
        if session_id not in self.smoke_held_sessions:
            return -1
        versions = [r["version"] for k, r in self.turn_traces.get(session_id, {}).items()
                    if not k.startswith("key:")]
        return min(max(versions, default=-1) + 1, 2)

    def oldest_session_versions(self, session_ids: list[str]) -> dict[str, int]:
        """Conservative live-Trial bound before harness endpoint selection."""
        result = {}
        for session_id in session_ids:
            versions = [record["version"] for key, record in self.turn_traces.get(session_id, {}).items()
                        if not key.startswith("key:")]
            if versions:
                result[session_id] = min(versions)
        return result

    def release_turn_reservation(self, request_id: str, session_id: str | None = None):
        self._reserved_bytes -= self.turn_reservations.pop(request_id, 0)
        pending = self.pending_prefixes.pop(request_id, None)
        if pending is not None:
            requests = self.session_pending.get(pending[0])
            if requests is not None:
                requests.discard(request_id)
                if not requests:
                    self.session_pending.pop(pending[0], None)

    def _prefix_path(self, session_id, text):
        trie = self.get_or_create(session_id)
        prefix, nodes = trie.search(text, filter_none=True)
        records = self.turn_traces.get(session_id, {})
        path = None
        start = 0
        for index in range(len(nodes) - 1, -1, -1):
            request_id = getattr(nodes[index].value, "request_id", None)
            if request_id in records:
                path = records[request_id]["path"]
                start = index + 1
                break
        for node in nodes[start:]:
            path = SegmentPath(path, node.value)
        return prefix, path

    @_trace_operation
    def prepare_turn(self, session_id: str, prompt: str, request_id: str) -> dict:
        """Return tokens and a request-scoped capability, never historical segments."""
        if session_id in self.cancelled_turn_sessions:
            raise ValueError("session cancelled")
        if request_id in self.pending_prefixes or request_id in self.turn_traces.get(session_id, {}):
            raise ValueError("duplicate turn request preparation")
        prefix, path = self._prefix_path(session_id, prompt)
        handle = uuid4().hex
        self.pending_prefixes[request_id] = (session_id, self.session_epochs[session_id], handle, prompt, path, prefix)
        self.session_pending.setdefault(session_id, set()).add(request_id)
        return {"prefix": prefix, "input_ids": [token for seg in path_segments(path) for token in seg.token_ids],
                "handle": {"request_id": request_id, "capability": handle}}

    @_trace_operation
    def commit_turn(self, session_id: str, prompt_key: str, key: str,
                    prefix_segments: list[TokenizedSegment] | dict, delta: TokenizedSegment | None,
                    response: TokenizedSegment, routed_experts: Any, metadata: dict) -> None:
        # Lists support old direct callers; production uses request-scoped handles.
        request_key = metadata["request_id"]
        if not isinstance(request_key, str) or not request_key or request_key.startswith("key:"):
            raise ValueError("invalid turn request identity")
        if session_id in self.closed_sessions or session_id in self.cancelled_turn_sessions:
            raise ValueError("session closed or cancelled")
        records = self.turn_traces.get(session_id, {})
        if request_key in records:
            raise ValueError("duplicate turn request commit")
        additions = [*([delta] if delta is not None else []), response]
        if any(seg.model_step is None and any(t != -100 for t in seg.labels) for seg in additions):
            raise ValueError("trainable turn tokens must have a generation version")
        if not key.startswith(prompt_key) or key != prompt_key + response.text:
            raise ValueError("response endpoint does not match prompt")
        routing_bytes = metadata.get("routing_bytes", 0)
        reserved = self.turn_reservations.get(request_key)
        if routing_bytes < 0 or (reserved is not None and routing_bytes > reserved):
            raise MemoryError("full routing exceeded its request reservation")
        if isinstance(prefix_segments, dict):
            pending = self.pending_prefixes.get(request_key)
            if (pending is None or pending[:2] != (session_id, self.session_epochs.get(session_id))
                    or prefix_segments != {"request_id": request_key, "capability": pending[2]}
                    or pending[3] != prompt_key):
                raise ValueError("invalid or stale prefix handle")
            path = pending[4]
            prefix = pending[5]
        else:
            prefix, path = self._prefix_path(session_id, prompt_key)
            if list(path_segments(path)) != prefix_segments:
                raise ValueError("prefix segments differ from stored history")
        if (delta.text if delta is not None else "") != prompt_key[len(prefix):]:
            raise ValueError("delta does not match prepared prefix")
        if response.request_id != request_key:
            raise ValueError("response request identity mismatch")
        # Own copies of the new data; the actor's historical segments are never mutated.
        additions = [seg.model_copy(deep=True) for seg in additions]
        for seg in additions:
            path = SegmentPath(path, seg)
        record = {**metadata, "path": path, "key": key, "routed_experts": routed_experts[0]}
        trie = self.get_or_create(session_id)
        if delta is not None:
            trie.insert(prompt_key, additions[0])
        trie.insert(key, additions[-1])
        records = self.turn_traces.setdefault(session_id, {})
        records[request_key] = record
        records["key:" + key] = record
        owned = self.session_segments.setdefault(session_id, [])
        owned.extend(additions)
        self._segment_count += len(additions)
        self._token_count += sum(len(seg.token_ids) for seg in additions)
        self._request_count += 1
        self._routing_bytes += routing_bytes
        self.release_turn_reservation(request_key)

    @_trace_operation
    def turn_memory(self) -> dict:
        return {"sessions": len(self.turn_traces), "requests": self._request_count,
                "routing_bytes": self._routing_bytes, "reserved_bytes": self._reserved_bytes,
                "hard_limit": self.turn_byte_limit, "segments": self._segment_count,
                "tokens": self._token_count, "cleanup_sessions": len(self.cleanup_jobs)}

    def trace_metrics(self) -> dict:
        # Logical counters do not approximate Python object overhead.
        rss = None
        try:
            with open("/proc/self/status") as status:
                rss = next(int(line.split()[1]) * 1024 for line in status if line.startswith("VmRSS:"))
        except (OSError, StopIteration):
            pass
        return {"operations": self._operation_metrics, "gc": self._gc_metrics,
                "rss_bytes": rss, "logical": self.turn_memory()}

    def audit_turn_memory(self) -> dict:
        """Expensive independent counters for tests/diagnostics only."""
        records = [r for turns in self.turn_traces.values() for k, r in turns.items() if not k.startswith("key:")]
        segments = [seg for values in self.session_segments.values() for seg in values]
        return {"sessions": len(self.turn_traces), "requests": len(records),
                "routing_bytes": sum(r.get("routing_bytes", 0) for r in records),
                "reserved_bytes": sum(self.turn_reservations.values()), "hard_limit": self.turn_byte_limit,
                "segments": len(segments), "tokens": sum(len(seg.token_ids) for seg in segments),
                "cleanup_sessions": len(self.cleanup_jobs)}

    def acknowledge_turn_routes(self, session_id: str, ack: dict) -> dict:
        """Release superseded routes after the recorder confirms receipt.

        Called under the cross-replica session owner, before admitting the next
        turn. Unknown/unacknowledged responses and exported consumers survive.
        Token segments, logprobs and policy versions are never pruned here.
        """
        if not isinstance(ack, dict):
            raise ValueError("invalid route acknowledgement")
        observed, retained = ack.get("observed"), ack.get("retained")
        if (not isinstance(observed, list) or not isinstance(retained, list)
                or not observed or not retained
                or any(not isinstance(x, str) or not x or x.startswith("key:")
                       for x in observed + retained)):
            raise ValueError("invalid route acknowledgement IDs")
        observed, retained = set(observed), set(retained)
        records = self.turn_traces.get(session_id, {})
        if not retained <= observed or any(x not in records for x in observed):
            raise ValueError("route acknowledgement does not match session records")
        if any(records[x].get("routing_released") for x in retained):
            # A concurrently issued request may carry an older recorder snapshot.
            # Its already-superseded endpoints must not resurrect or free routes.
            return {"released_routes": 0, "released_bytes": 0}
        unused = [records[x] for x in observed - retained
                  if records[x]["routed_experts"] is not None and not records[x].get("exported")]
        freed_bytes = sum(r.get("routing_bytes", 0) for r in unused)
        if unused:
            ray.internal.free([r["routed_experts"] for r in unused], local_only=False)
        for record in unused:
            self._routing_bytes -= record.get("routing_bytes", 0)
            record["routed_experts"] = None
            record["routing_bytes"] = 0
            record["routing_released"] = True
        return {"released_routes": len(unused), "released_bytes": freed_bytes}

    def retain_turn_routes(self, session_id: str, request_ids: list[str]):
        """After a Trial ends, keep only routes selected for training.

        Token/logprob/version snapshots remain available. No live Trial may
        call this: its harness has not yet frozen the selected endpoints.
        """
        records = self.turn_traces[session_id]
        selected = set(request_ids)
        if not selected or any(key not in records for key in selected):
            raise ValueError("selected training request has no turn record")
        unused = [record for key, record in records.items()
                  if not key.startswith("key:") and key not in selected
                  and record["routed_experts"] is not None]
        if unused:
            ray.internal.free([record["routed_experts"] for record in unused], local_only=False)
        for record in unused:
            self._routing_bytes -= record.get("routing_bytes", 0)
            record["routed_experts"] = None
            record["routing_bytes"] = 0
            record["routing_released"] = True

    def get_or_create(self, session_id: str) -> Trie:
        """Get the Trie for a session, or create one if it doesn't exist.

        Args:
            session_id (str): The session identifier.

        Returns:
            Trie: The Trie instance associated with the session.
        """
        if session_id in self.closed_sessions:
            raise ValueError("session closed")
        if session_id not in self.sessions:
            self.sessions[session_id] = Trie()
            self.session_epochs[session_id] = uuid4().hex
        return self.sessions[session_id]

    def list_sessions(self) -> List[str]:
        """List all session ids currently stored in the actor."""
        return list(self.sessions.keys())

    def keys(self, session_id: str) -> List[str]:
        """Get all keys (i.e., strings) stored in a session's Trie.

        Args:
            session_id (str): The session identifier.

        Returns:
            List[str]: A list of all keys in the session's Trie.
        """
        trie = self.get_or_create(session_id)
        return trie.keys()

    def insert(self, session_id: str, key: str, value: Any):
        """Insert a (key, value) pair into a session's Trie.

        Args:
            session_id (str): The session identifier.
            key (str): The key string.
            value (Any): The trace segment/value to store.
        """
        trie = self.get_or_create(session_id)
        return trie.insert(key, value)

    @_trace_operation
    def search(self, session_id: str, text: str, filter_none: bool = False):
        """Search the longest prefix in a session's Trie.

        Args:
            session_id (str): The session identifier.
            text (str): The input text to search.
            filter_none (bool): Whether to filter out nodes with no value.

        Returns:
            Tuple[str, List["TreeNode"]]: The matched prefix and matched nodes.
        """
        if not self.turn_traces.get(session_id):
            return self.get_or_create(session_id).search(text, filter_none)
        prefix, path = self._prefix_path(session_id, text)
        return prefix, [TreeNode(value=seg, parent=None) for seg in path_segments(path)]

    @_trace_operation
    def begin_release_sessions(self, session_ids: list[str]) -> list[str]:
        found = []
        for sid in dict.fromkeys(session_ids):
            if sid not in self.sessions:
                continue
            found.append(sid)
            if sid in self.cleanup_jobs:
                continue
            self.closed_sessions.add(sid)
            self.cleanup_jobs[sid] = {"stack": [self.sessions[sid].root]}
            for request in list(self.session_pending.get(sid, ())):
                self.release_turn_reservation(request)
        return found

    @_trace_operation
    def continue_release_sessions(self, session_ids: list[str], max_items: int = 64) -> dict:
        """One bounded actor task. Caller yields between invocations."""
        if max_items < 1:
            raise ValueError("cleanup batch must be positive")
        deadline = time.monotonic() + 0.02
        refs = []
        processed = 0
        for sid in dict.fromkeys(session_ids):
            job = self.cleanup_jobs.get(sid)
            if job is None:
                continue
            records = self.turn_traces.get(sid, {})
            stack = job["stack"]
            owned = self.session_segments.get(sid, [])
            while processed < max_items and time.monotonic() < deadline:
                if records:
                    name, record = records.popitem()
                    if not name.startswith("key:"):
                        if record["routed_experts"] is not None:
                            refs.append(record["routed_experts"])
                        self._routing_bytes -= record.get("routing_bytes", 0)
                        self._request_count -= 1
                        record["routed_experts"] = None
                    # Reverse insertion order drops descendants before parents.
                    # Parent chains remain anchored by their own earlier records.
                    del record
                elif stack:
                    node = stack[-1]
                    if node.children:
                        _, child = node.children.popitem()
                        child.parent = None
                        stack.append(child)
                    else:
                        if node.value is not None:
                            _free_ray_refs(node.value)
                            node.value = None
                        node.parent = None
                        stack.pop()
                elif owned:
                    seg = owned.pop()
                    self._segment_count -= 1
                    self._token_count -= len(seg.token_ids)
                    del seg
                else:
                    self.sessions.pop(sid, None)
                    self.turn_traces.pop(sid, None)
                    self.session_segments.pop(sid, None)
                    self.session_epochs.pop(sid, None)
                    self.smoke_held_sessions.discard(sid)
                    self.turn_owners.pop(sid, None)
                    self.cleanup_jobs.pop(sid, None)
                    break
                processed += 1
            if processed >= max_items or time.monotonic() >= deadline:
                break
        if refs:
            ray.internal.free(refs, local_only=False)
        return {"pending": [sid for sid in session_ids if sid in self.cleanup_jobs], "processed": processed}

    def release(self, session_id: str, key: str | None = None):
        if key is not None:
            self.get_or_create(session_id).release(key)
            return
        assert session_id in self.sessions, f"Session ID '{session_id}' not found for release."
        self.release_sessions([session_id])

    def release_sessions(self, session_ids: list[str]) -> list[str]:
        """Synchronous compatibility API; async consumers use bounded RPCs."""
        found = self.begin_release_sessions(session_ids)
        pending = found
        while pending:
            pending = self.continue_release_sessions(pending)["pending"]
        return found

    def release_all(self):
        """Release all sessions and free associated resources."""
        self.release_sessions(list(self.sessions))
        self.objects.clear()
        self.updated_at.clear()

    @_trace_operation
    def export_training_trace(self, session_id: str, prompt_text: str, request_id: str | None = None) -> dict:
        """Export the stored training trace given a complete prompt text.

        Args:
            session_id (str): The session identifier.
            prompt_text (str): The complete assembled prompt string to look up.

        Returns:
            dict: The trace dictionary containing `input_ids`, `labels`, `logprobs`,
                and `routed_experts`. `routed_experts` is a per-node list of expert-id
                refs for a MoE trace, or `None` when the trace carries no routed
                experts at all (dense model).

        Raises:
            ValueError: If the prompt_text does not completely match the trace keys in the session.
        """
        if session_id in self.closed_sessions:
            raise ValueError("session closed")
        turns = self.turn_traces.get(session_id)
        if turns is not None:
            record = turns.get(request_id if request_id is not None else "key:" + prompt_text)
            if record is None:
                raise ValueError("selected turn endpoint has no complete routing record")
            if record["key"] != prompt_text:
                raise ValueError("selected request identity does not match its rendered endpoint")
            if record.get("routing_released"):
                raise ValueError("routing for this unselected terminal endpoint was released")
            trace = {"input_ids": [], "labels": [], "logprobs": [], "model_steps": [], "policy_versions": []}
            for segment in path_segments(record["path"]):
                trace["input_ids"].extend(segment.token_ids)
                trace["labels"].extend(segment.labels)
                trace["logprobs"].extend(segment.logprobs)
                trace["model_steps"].extend([segment.model_step] * len(segment.token_ids))
                trace["policy_versions"].extend([segment.policy_version] * len(segment.token_ids))
            trace["routed_experts"] = [record["routed_experts"]] if record["routed_experts"] is not None else None
            trace["routing_source_request_id"] = record["request_id"]
            trace["routing_source_version"] = record["version"]
            record["exported"] = True  # Pin until terminal session cleanup.
            return trace
        trie = self.get_or_create(session_id)
        key, nodes = trie.search(prompt_text, filter_none=True)
        if prompt_text != key:
            session_keys = trie.keys()
            key_summaries = [
                {
                    "idx": idx,
                    "len": len(session_key),
                    "text": _compact_text(session_key),
                    "mismatch": _mismatch_window(prompt_text, session_key),
                }
                for idx, session_key in enumerate(session_keys[:8])
            ]
            debug_message = (
                "[TraceStore] prompt mismatch "
                f"session={session_id!r} prompt_len={len(prompt_text)} "
                f"matched_len={len(key)} key_count={len(session_keys)} "
                f"prompt={_compact_text(prompt_text)!r} key_summaries={key_summaries!r}"
            )
            get_logger().error(debug_message)
            raise ValueError(
                f"Prompt text does not match any trace key in session {session_id!r}: "
                f"prompt_len={len(prompt_text)} matched_len={len(key)} key_count={len(session_keys)}. "
                "See the logged '[TraceStore] prompt mismatch' report for the full diff."
            )
        trace: dict[str, Any] = {"input_ids": [], "labels": [], "logprobs": []}
        expert_keys: list[Any] = []
        for node in nodes:
            node_val = node.value
            if not isinstance(node_val, TokenizedSegment):
                raise TypeError(f"Unexpected trace node value type: {type(node_val)!r}")
            assert node_val.labels is not None
            assert node_val.logprobs is not None
            trace["input_ids"].extend(node_val.token_ids)
            trace["labels"].extend(node_val.labels)
            trace["logprobs"].extend(node_val.logprobs)
            expert_keys.append(node_val.expert_key)
        trace["routed_experts"] = _resolve_routed_experts(expert_keys, session_id)
        return trace

    def get_objects(self, keys: list[str]) -> list[ray.ObjectRef]:
        """Fetch ray.ObjectRef elements by their keys.

        Args:
            keys (list[str]): The list of object keys to retrieve.

        Returns:
            list[ray.ObjectRef]: The mapped ray.ObjectRefs.
        """
        return [self.objects[key] for key in keys if key in self.objects]


def _acquire_store(create):
    global _handle_cache, _handle_topology
    from .trace_shards import shard_count, trace_budget, acquire_shards
    topology = (shard_count(), trace_budget())
    if _handle_cache is not None:
        if globals().get('_handle_topology') != topology:
            raise RuntimeError("trace shard topology or budget changed within this process")
        return _handle_cache
    handle = acquire_shards(RolloutTraceStore, _STORE_NAME, _STORE_NAMESPACE, create=create)
    if handle is not None:
        _handle_cache, _handle_topology = handle, topology
    return handle


def get_store():
    """Acquire session-affine stores; validate topology even for one shard."""
    return _acquire_store(create=True)


def get_existing_store():
    """Find existing trace owners without starting actors during cleanup."""
    global _handle_cache, _handle_topology
    if not ray.is_initialized():
        _handle_cache = None
        _handle_topology = None
        return None
    return _acquire_store(create=False)


async def release_existing_sessions(session_ids: list[str]) -> set[str]:
    """Release trace sessions that exist without creating the singleton store.

    Args:
        session_ids (list[str]): Candidate trace session identifiers.

    Returns:
        set[str]: Session identifiers that existed and were released.
    """
    session_ids = list(dict.fromkeys(str(session_id) for session_id in session_ids))
    if not session_ids:
        return set()

    store = get_existing_store()
    if store is None:
        return set()

    found = await store.begin_release_sessions.remote(session_ids)
    pending = found
    while pending:
        pending = (await store.continue_release_sessions.remote(pending))["pending"]
        await asyncio.sleep(0)
    return set(found)


async def release_and_discard_rollout_groups(groups: list[list[RolloutState]]) -> None:
    """Release trace-owned resources before discarding terminal rollouts.

    Sessions released by the trace store have already freed their routed-expert references. Detach those references
    before the generic rollout-state cleanup so it does not explicitly free them a second time. Rollouts whose sessions
    are absent from the store retain their references for the generic cleanup path.
    """
    released_session_ids = await release_existing_sessions(
        [str(item.session_id) for group in groups for item in group if item.session_id is not None]
    )
    for group in groups:
        for item in group:
            if item.session_id is not None and str(item.session_id) in released_session_ids:
                item.routed_experts = None
            discard_rollout_state(item)


if __name__ == "__main__":
    print("=== 评估使用 Trie 加速 tokenize.py 避免多轮对话重复 tokenization ===")

    # 1. 初始化
    trie = Trie()

    # 2. 模拟增量 Tokenize 输出
    def mock_segment_tokenize(segment_text: str, is_assistant_response: bool):
        tok_len = max(1, len(segment_text) // 4)
        if is_assistant_response:
            return {
                "segment_text": segment_text,
                "token_ids": [1000] * tok_len,
                "labels": [1000] * tok_len,  # 模型真实生成内容，参与 Loss 计算
                "logprobs": [-0.1] * tok_len,  # 拥有真实的概率分布
            }
        else:
            return {
                "segment_text": segment_text,
                "token_ids": [1000] * tok_len,
                "labels": [-100] * tok_len,  # 模板/用户文本被掩蔽 (Mask)
                "logprobs": [0.0] * tok_len,
            }

    # 3. 定义两轮对话的四个增量片段，明确分离“非模型生成段”与“模型生成段”
    non_asst_str1 = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n"
    asst_str1 = "Hi there!<|im_end|>"
    non_asst_str2 = "\n<|im_start|>user\nIntroduce yourself.<|im_end|>\n<|im_start|>assistant\n"
    asst_str2 = "I am an AI assistant created to help you.<|im_end|>"

    # 4. 模拟运行流：我们在每收到一个新的增量片段时，将“至今为止的整个 Prefix String”作为 key，
    #    且把“最新计算好的增量 Token 状态”作为 value 存入 Trie。

    print("\n[Step 1] 处理第一轮 non_assistant 提示...")
    current_prefix = non_asst_str1
    val1 = mock_segment_tokenize(non_asst_str1, is_assistant_response=False)
    trie.insert(current_prefix, val1)

    print("[Step 2] 模型生成了第一轮回复，将其追加...")
    current_prefix += asst_str1
    val2 = mock_segment_tokenize(asst_str1, is_assistant_response=True)
    trie.insert(current_prefix, val2)

    print("[Step 3] 追加第二轮 non_assistant 提示...")
    current_prefix += non_asst_str2
    val3 = mock_segment_tokenize(non_asst_str2, is_assistant_response=False)
    trie.insert(current_prefix, val3)

    # 5. 模拟一次线上推理 / 强化学习推演的过程
    # 假设我们现在需要对一段更长的请求（第二轮模型开始生成前）的完整 prompt 进行处理：
    turn2_prompt = non_asst_str1 + asst_str1 + non_asst_str2

    print(f"\n[Search] 收到需要 tokenizer 计算的长 Prompt (总长 {len(turn2_prompt)} 字符)")
    prefix, nodes = trie.search(turn2_prompt)

    # 6. 从前缀树中收集沿路径出现的所有 value
    cached_length = 0
    reconstructed_labels = []
    reconstructed_string_parts = []

    for node in nodes:
        if node.value is not None:
            val = node.value
            cached_length += len(val["segment_text"])
            reconstructed_labels.extend(val["labels"])
            reconstructed_string_parts.append(val["segment_text"])
            print(
                f"   -> 命中缓存片段: {repr(val['segment_text'][:30])}..., 长度: {len(val['segment_text'])}, 掩蔽标记({val['labels'][0]}): {len(val['labels'])} tokens"
            )

    reconstructed_string = "".join(reconstructed_string_parts)
    print(f"\n   -> 测试字符串重建: {reconstructed_string == turn2_prompt[:cached_length]}")

    print("\n[Result] 统计验证：")
    print(f"  -> Cache 命中的总字符长度: {cached_length} / {len(turn2_prompt)}")

    if cached_length == len(turn2_prompt):
        print("  -> 🥳 完美！整个 Prompt 都命中了 Trie 缓存内容，完全不需要调用 tokenizer.encode 进行重复处理！")
    else:
        unmatched_text = turn2_prompt[cached_length:]
        print(f"  -> 剩下实际需要交给 tokenizer 的未命中部分: {unmatched_text!r}")
    trie.release()
