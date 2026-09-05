#!/usr/bin/env python3
"""ouroboros.

  tier 0  REFERENCE_DIGESTS     one digest per order, the oracle every other
          tier is differentially tested against.

  tier 1  GSL                   a declarative glyph language.  The glyph is no
          longer drawn; it is the result of evaluating a program, whose strokes
          are the cells at which a projection over a finite group does not
          vanish - the group solved for rather than written down, and audited
          against the group axioms once it is.  Nineteen
          layers stand between that source text and the output:

              layer -3  metaprogramming ... contracts, aspects, registries
              layer -2  observability ..... spans, metrics, event bus
              layer -1  diagnostics ....... severities, catalogues, i18n
              layer  0  functional core ... Result/Maybe monads, laziness
              layer  1  configuration ..... layered providers, feature flags
              layer  2  algebra ........... lattice, operators, symmetry groups
              layer  3  preprocessing ..... macros, pragmas, source maps
              layer  4  lexical analysis .. table-driven transducer
              layer  5  syntax ............ grammar metatheory, FIRST/FOLLOW
              layer  6  semantics ......... scopes, unification-based inference
              layer  7  lowering .......... symbolic IR, basic blocks
              layer  8  optimisation ...... fixed-point pass manager
              layer  9  object format ..... container, constant pool, CRC-32
              layer 10  execution ......... stack VM plus threaded-code tier
              layer 11  persistence ....... event-sourced aggregate, unit of work
              layer 12  presentation ...... concurrent rasteriser, middleware
              layer 13  composition ....... autowiring container, lifetimes
              layer 14  orchestration ..... pipeline state machine, motifs
              layer 15  assurance ......... invariant self-test battery

  tier 2  LLVM                  layer 16 lowers that bytecode to LLVM IR: the
          operand stack becomes memory, every instruction address becomes a
          basic block, and the machine's jumps become branches between them.
          The result verifies, JITs, and links to a native ELF binary.

  tier 3  GSL-2                 a second language, general enough to write a
          compiler in, plus a compiler for it written in itself.  A Python seed
          compiles it once; from then on it compiles its own source, and the
          build is proven correct by a three-generation fixpoint: the
          compiler's output on its own source stops changing.

  tier 4  the closed loop       tier 1 and tier 2 again, written in the
          language of tier 3 and compiled by the compiler that compiles
          itself: the same preprocessor, transducer, parser, analyser, pass
          manager and assembler, and the same lowering after them, in one
          GSL-2 translation unit whose sections carry the layer numbers they
          answer to.  The seed turns the crank once, on the GSL-2 compiler
          alone, and after that no interpreter is left anywhere in the chain.
          The claim is not that the two front ends agree but that they are
          the same compiler: for any program either accepts, both emit the
          same bytes.

  tier 5  machine code          layer 18 drops the toolchain: no llvmlite, no
          assembler, no linker, no libc.  The object module is encoded as
          x86-64, aarch64 or riscv64 directly - the operand stack is the
          hardware stack, the canvas is .bss - and wrapped in a static ELF64
          executable by hand.  A little over a kilobyte, and all it asks of
          the world is write and exit_group.  Three machines behind one ELF
          writer, which is what obliges the writer to say which it means.

  tier 6  WebAssembly           layer 19 encodes the same object module as a
          wasm reactor, by hand and with nothing installed.  It imports
          nothing at all: the module exports its memory, the offset of its
          text and a render that answers a length, so the host is five lines
          and any engine will do.  Inside a basic block the lowering is one
          instruction to one, since both machines are stack machines; the
          branches are where they differ, and the instruction stream becomes
          a table over a program counter.  Layer 20 then reads that module
          back and runs it here, so the file executes its own output and the
          tier stops needing an engine at all.

  tier 7  no kernel             layer 22 asks for nothing underneath at all.
          The instruction stream and the whole runtime are layer 18's,
          unchanged; only the three things a kernel was being asked for
          differ - who zeroes the data, what a finished program does, and
          where the octets go when there is no descriptor to write them to.
          In front sits one sector of sixteen-bit real mode walking the
          machine up to the sixty-four the encoder emits: a page table, a
          descriptor table, and three bits in three registers.  The figure
          arrives in the text buffer and out of the serial port, and the
          program halts rather than exits, there being nothing to exit to.

    python3 ouroboros.py                 render via the virtual machine
    python3 ouroboros.py --emit-llvm     lower the program to LLVM IR
    python3 ouroboros.py --jit           JIT-execute that IR
    python3 ouroboros.py --bootstrap     run the full self-hosting bootstrap
    python3 ouroboros.py --close-the-loop    compile the glyph with the
                                             front end written in GSL-2
    python3 ouroboros.py --emit-elf PATH  write a static executable, no toolchain
    python3 ouroboros.py --machine aarch64 --emit-elf PATH   for the other one
    python3 ouroboros.py --emit-wasm PATH write a wasm reactor, no toolchain
    python3 ouroboros.py --emit-boot PATH write a disk that needs no kernel
    python3 ouroboros.py --run-wasm PATH  run one back, with no engine either
    python3 ouroboros.py --selftest      differential-test every tier
    python3 ouroboros.py --emit-everything   dump all of it at once

  A PATH of - is standard input or standard output, so the tiers pipe:

    python3 ouroboros.py --emit-wasm - | python3 ouroboros.py --run-wasm -

"""

from __future__ import annotations

import abc
import argparse
import binascii
import collections
import enum
import functools
import hashlib
import inspect
import io
import itertools
import logging
import operator
import os
import random
import re
import shutil
import string
import struct
import subprocess
import sys
import tempfile
import threading
import time
import typing
import weakref
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    ClassVar,
    Final,
    FrozenSet,
    Generic,
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    Protocol,
    Sequence,
    TypeVar,
    runtime_checkable,
)


# ======================================================================
# Tier 0: the oracle
# ======================================================================


REFERENCE_DIGESTS: Final[Mapping[int, str]] = {
    3: "5096ea608a8cdbf06a4e52b768c5b6d52e3c6a7a2cf70c357da47b1a9ff67e0d",
    5: "1ff20b9f118d749e5ca94589f0a09120348a4f9623a8b3e02fdf21942e83ec20",
    7: "c5d6eaa07b200c3e85045d8caee9885fcd5d68b83da18382989a66f350c16445",
    9: "e09740ece6247f128af5701f0e20e1bd18b705cd311583137b3b9811903f30a7",
    11: "64de5f8b35fefc481705759b69b94cc6dbda8cab697f02d5fa0359ce27f43600",
    15: "a269a97f986aed1981dc6fb04c3dde0cd38416e53a0177ed8619efa8b0036f79",
    21: "ebf1e7ccc342f88963e7d5846195403c70f18095c57add6f4681518b35065216",
}


def reference_digest(rendering: str) -> str:
    """The witness for one rendering."""
    return hashlib.sha256(rendering.encode()).hexdigest()


__all__ = [
    "GlyphPlatformError",
    "ContractViolation",
    "PreprocessorError",
    "LexicalError",
    "SyntaxError_",
    "SemanticError",
    "TypeInferenceError",
    "CodeGenerationError",
    "ObjectFormatError",
    "ExecutionFault",
    "GroupAxiomViolation",
    "ResolutionFailure",
    "Result",
    "Ok",
    "Err",
    "Maybe",
    "Coordinate",
    "LinearOperator",
    "SymmetryGroup",
    "GlyphVirtualMachine",
    "ServiceContainer",
    "SynthesisOrchestrator",
    "CompilationArtifacts",
    "synthesize",
    "reference_digest",
    "LlvmLoweringBackend",
    "LlvmToolchainService",
    "bootstrap",
    "main",
]

PLATFORM_VERSION: Final[str] = "2.0.0"
OBJECT_MAGIC: Final[bytes] = b"GVM\x02"
OBJECT_FORMAT_VERSION: Final[int] = 2
DEFAULT_LATTICE_ORDER: Final[int] = 7
DEFAULT_MOTIF: Final[str] = "primary"
JIT_TIER_UP_THRESHOLD: Final[int] = 64
RASTERIZER_WORKER_COUNT: Final[int] = 4

_LOGGER: Final[logging.Logger] = logging.getLogger("glyph.hyperplatform")
_LOGGER.addHandler(logging.NullHandler())

_T = TypeVar("_T")
_U = TypeVar("_U")
_NodeT = TypeVar("_NodeT", bound="AstNode")


# ======================================================================
# Layer -3: metaprogramming substrate
# ======================================================================


class Sentinel:
    """Interned, falsey, self-describing marker objects."""

    _interned: ClassVar[MutableMapping[str, "Sentinel"]] = {}

    def __new__(cls, name: str) -> "Sentinel":
        existing = cls._interned.get(name)
        if existing is not None:
            return existing
        instance = super().__new__(cls)
        instance._name = name  # type: ignore[attr-defined]
        cls._interned[name] = instance
        return instance

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"<{self._name}>"  # type: ignore[attr-defined]

    def __bool__(self) -> bool:
        return False


MISSING: Final[Sentinel] = Sentinel("MISSING")
UNSET: Final[Sentinel] = Sentinel("UNSET")


class GlyphPlatformError(Exception):
    """Root of the platform's exception hierarchy."""


class ContractViolation(GlyphPlatformError):
    """A precondition, postcondition or class invariant was falsified."""


CONTRACTS_ENABLED: bool = True


def _project_arguments(
    predicate: Callable[..., bool], arguments: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Restricts a bound argument mapping to the predicate's own parameters."""
    parameters = inspect.signature(predicate).parameters
    return {key: value for key, value in arguments.items() if key in parameters}


def requires(predicate: Callable[..., bool], description: str) -> Callable[[Callable[..., _T]], Callable[..., _T]]:
    """Design-by-contract precondition decorator."""

    def decorator(function: Callable[..., _T]) -> Callable[..., _T]:
        if not CONTRACTS_ENABLED:
            return function
        signature = inspect.signature(function)

        @functools.wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> _T:
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            if not predicate(**_project_arguments(predicate, bound.arguments)):
                raise ContractViolation(
                    f"precondition violated in {function.__qualname__}: {description}"
                )
            return function(*args, **kwargs)

        return wrapper

    return decorator


def ensures(predicate: Callable[..., bool], description: str) -> Callable[[Callable[..., _T]], Callable[..., _T]]:
    """Design-by-contract postcondition decorator; predicate sees ``result``."""

    def decorator(function: Callable[..., _T]) -> Callable[..., _T]:
        if not CONTRACTS_ENABLED:
            return function
        signature = inspect.signature(function)

        @functools.wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> _T:
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            result = function(*args, **kwargs)
            projected = dict(_project_arguments(predicate, bound.arguments))
            projected["result"] = result
            if not predicate(**projected):
                raise ContractViolation(
                    f"postcondition violated in {function.__qualname__}: {description}"
                )
            return result

        return wrapper

    return decorator


def invariant(cls: type) -> type:
    """Class decorator weaving ``__invariant__`` around every public method."""
    if not CONTRACTS_ENABLED:
        return cls
    checker = getattr(cls, "__invariant__", None)
    if checker is None:  # pragma: no cover - defensive
        raise ContractViolation(f"{cls.__name__} declares no __invariant__")

    def weave(name: str, method: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(method)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            result = method(self, *args, **kwargs)
            if not checker(self):
                raise ContractViolation(
                    f"invariant of {cls.__name__} violated after {name}"
                )
            return result

        return wrapper

    for name, member in list(vars(cls).items()):
        if name.startswith("_") or not inspect.isfunction(member):
            continue
        setattr(cls, name, weave(name, member))
    return cls


class PluginRegistryMeta(abc.ABCMeta):
    """Metaclass auto-registering concrete subclasses of a registry root."""

    def __init__(cls, name: str, bases: tuple[type, ...], namespace: dict[str, Any], **kwargs: Any) -> None:
        super().__init__(name, bases, namespace, **kwargs)
        if namespace.get("__registry_root__", False):
            cls._registry: MutableMapping[str, type] = {}
            return
        registry = getattr(cls, "_registry", None)
        if registry is None or inspect.isabstract(cls):
            return
        key = namespace.get("__registry_key__") or _kebab(name)
        registry[key] = cls

    def lookup(cls, key: str) -> type:
        registry = getattr(cls, "_registry", {})
        try:
            return registry[key]
        except KeyError as exc:
            catalogue = ", ".join(sorted(registry))
            raise ResolutionFailure(
                f"no plugin registered under {key!r}; known plugins: {catalogue}"
            ) from exc

    def catalogue(cls) -> tuple[str, ...]:
        return tuple(sorted(getattr(cls, "_registry", {})))


def _kebab(identifier: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "-", identifier).lower()


@runtime_checkable
class Aspect(Protocol):
    """A cross-cutting concern woven around a join point."""

    def before(self, join_point: str, arguments: Mapping[str, Any]) -> None: ...

    def after(self, join_point: str, result: Any, elapsed: float) -> None: ...

    def on_error(self, join_point: str, error: BaseException) -> None: ...


_ASPECTS: list[Aspect] = []


def register_aspect(aspect: Aspect | type) -> Any:
    """Registers an aspect; classes are instantiated on the caller's behalf."""
    _ASPECTS.append(aspect() if inspect.isclass(aspect) else aspect)
    return aspect


def woven(function: Callable[..., _T]) -> Callable[..., _T]:
    """Weaves every globally registered aspect around ``function``."""
    join_point = function.__qualname__

    @functools.wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> _T:
        for aspect in _ASPECTS:
            aspect.before(join_point, kwargs)
        started = time.perf_counter()
        try:
            result = function(*args, **kwargs)
        except BaseException as error:
            for aspect in _ASPECTS:
                aspect.on_error(join_point, error)
            raise
        elapsed = time.perf_counter() - started
        for aspect in _ASPECTS:
            aspect.after(join_point, result, elapsed)
        return result

    return wrapper


# ======================================================================
# Layer -2: observability
# ======================================================================


@dataclass
class Span:
    """A node in the hierarchical execution trace."""

    name: str
    attributes: MutableMapping[str, Any] = field(default_factory=dict)
    started: float = field(default_factory=time.perf_counter)
    finished: float | None = None
    children: list["Span"] = field(default_factory=list)

    @property
    def elapsed_ms(self) -> float:
        end = self.finished if self.finished is not None else time.perf_counter()
        return (end - self.started) * 1000.0

    def render(self, depth: int = 0) -> str:
        rail = "  " * depth + ("└─ " if depth else "")
        annotations = "".join(f" {k}={v!r}" for k, v in self.attributes.items())
        lines = [f"{rail}{self.name} [{self.elapsed_ms:7.3f} ms]{annotations}"]
        lines.extend(child.render(depth + 1) for child in self.children)
        return "\n".join(lines)


class Tracer:
    """A thread-aware hierarchical span recorder."""

    def __init__(self, name: str) -> None:
        self._root = Span(name)
        self._local = threading.local()
        self._lock = threading.RLock()

    def _stack(self) -> list[Span]:
        stack = getattr(self._local, "stack", None)
        if stack is None:
            stack = [self._root]
            self._local.stack = stack
        return stack

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[Span]:
        stack = self._stack()
        node = Span(name, dict(attributes))
        with self._lock:
            stack[-1].children.append(node)
        stack.append(node)
        try:
            yield node
        finally:
            node.finished = time.perf_counter()
            stack.pop()

    @property
    def root(self) -> Span:
        return self._root

    def render(self) -> str:
        self._root.finished = time.perf_counter()
        return self._root.render()


class MetricRegistry:
    """A minimal counter-and-histogram registry with textual exposition."""

    def __init__(self) -> None:
        self._counters: MutableMapping[str, int] = collections.Counter()
        self._histograms: MutableMapping[str, list[float]] = collections.defaultdict(list)
        self._lock = threading.Lock()

    def increment(self, name: str, amount: int = 1) -> None:
        with self._lock:
            self._counters[name] += amount

    def observe(self, name: str, value: float) -> None:
        with self._lock:
            self._histograms[name].append(value)

    def render(self) -> str:
        with self._lock:
            lines = [f"counter {name} = {value}" for name, value in sorted(self._counters.items())]
            for name, samples in sorted(self._histograms.items()):
                ordered = sorted(samples)
                midpoint = ordered[len(ordered) // 2]
                lines.append(
                    f"histogram {name} n={len(ordered)} "
                    f"min={ordered[0]:.6f} p50={midpoint:.6f} max={ordered[-1]:.6f}"
                )
        return "\n".join(lines) or "(no metrics recorded)"


TRACER: Final[Tracer] = Tracer("synthesis")
METRICS: Final[MetricRegistry] = MetricRegistry()


@register_aspect
class TelemetryAspect:
    """Cross-cutting aspect funnelling join points into the metric registry."""

    def before(self, join_point: str, arguments: Mapping[str, Any]) -> None:
        METRICS.increment(f"invocations.{join_point}")

    def after(self, join_point: str, result: Any, elapsed: float) -> None:
        METRICS.observe(f"latency_seconds.{join_point}", elapsed)

    def on_error(self, join_point: str, error: BaseException) -> None:
        METRICS.increment(f"faults.{join_point}.{type(error).__name__}")


@dataclass(frozen=True, slots=True)
class PlatformEvent:
    """Base class for everything published on the bus."""

    stage: str
    detail: str


@dataclass(frozen=True, slots=True)
class StageEntered(PlatformEvent):
    pass


@dataclass(frozen=True, slots=True)
class StageCompleted(PlatformEvent):
    artefact_size: int = 0


@dataclass(frozen=True, slots=True)
class DiagnosticRaised(PlatformEvent):
    severity: str = "error"


class EventBus:
    """Priority-ordered synchronous publish/subscribe fabric."""

    def __init__(self) -> None:
        self._subscribers: MutableMapping[type, list[tuple[int, Callable[[Any], None]]]] = (
            collections.defaultdict(list)
        )
        self._lock = threading.RLock()
        self._journal: list[PlatformEvent] = []

    def subscribe(
        self, event_type: type, handler: Callable[[Any], None], priority: int = 0
    ) -> Callable[[Any], None]:
        with self._lock:
            self._subscribers[event_type].append((priority, handler))
            self._subscribers[event_type].sort(key=operator.itemgetter(0), reverse=True)
        return handler

    def publish(self, event: PlatformEvent) -> None:
        with self._lock:
            self._journal.append(event)
            handlers = [
                handler
                for event_type, bucket in self._subscribers.items()
                if isinstance(event, event_type)
                for _, handler in bucket
            ]
        for handler in handlers:
            handler(event)

    @property
    def journal(self) -> tuple[PlatformEvent, ...]:
        with self._lock:
            return tuple(self._journal)


EVENT_BUS: Final[EventBus] = EventBus()


# ======================================================================
# Layer -1: diagnostics and internationalisation
# ======================================================================


MESSAGE_CATALOGUE: Final[Mapping[str, Mapping[str, str]]] = {
    "en": {
        "stage.enter": "entering stage {stage}",
        "stage.leave": "stage {stage} produced {size} artefact unit(s)",
        "diag.unknown_symbol": "unknown symbol {name!r}",
        "diag.duplicate_symbol": "symbol {name!r} is already bound in this scope",
        "diag.unexpected_token": "expected {expected}, found {found}",
        "diag.bad_order": "lattice order must be an odd positive integer, got {order}",
        "diag.clipped": "{count} emitted cell(s) fell outside the lattice and were clipped",
        "diag.macro_recursion": "macro {name!r} expands into itself",
        "report.ok": "synthesis completed in {ms:.3f} ms",
        "report.fail": "synthesis aborted: {error}",
    },
    "fr": {
        "stage.enter": "entrée dans l'étape {stage}",
        "stage.leave": "l'étape {stage} a produit {size} unité(s) d'artefact",
        "diag.unknown_symbol": "symbole inconnu {name!r}",
        "diag.duplicate_symbol": "le symbole {name!r} est déjà lié dans cette portée",
        "diag.unexpected_token": "attendu {expected}, trouvé {found}",
        "diag.bad_order": "l'ordre du treillis doit être un entier positif impair, reçu {order}",
        "diag.clipped": "{count} cellule(s) émise(s) hors treillis ont été rognées",
        "diag.macro_recursion": "la macro {name!r} se développe en elle-même",
        "report.ok": "synthèse terminée en {ms:.3f} ms",
        "report.fail": "synthèse interrompue : {error}",
    },
}


class MessageCatalog:
    """Localised message formatter with graceful fallback to English."""

    def __init__(self, language: str = "en") -> None:
        self._language = language if language in MESSAGE_CATALOGUE else "en"

    @property
    def language(self) -> str:
        return self._language

    def __call__(self, key: str, **arguments: Any) -> str:
        table = MESSAGE_CATALOGUE[self._language]
        template = table.get(key) or MESSAGE_CATALOGUE["en"].get(key, key)
        try:
            return template.format(**arguments)
        except (KeyError, IndexError):  # pragma: no cover - defensive
            return template


CATALOG: MessageCatalog = MessageCatalog("en")


class Severity(enum.IntEnum):
    HINT = 10
    WARNING = 20
    ERROR = 30
    FATAL = 40


@dataclass(frozen=True, slots=True)
class SourcePosition:
    """A one-based line/column pair, plus the absolute offset."""

    line: int
    column: int
    offset: int = 0

    def __str__(self) -> str:
        return f"{self.line}:{self.column}"


ORIGIN: Final[SourcePosition] = SourcePosition(1, 1, 0)


@dataclass(frozen=True, slots=True)
class Diagnostic:
    """A single machine-readable compiler remark."""

    severity: Severity
    code: str
    message: str
    position: SourcePosition

    def render(self, source: str | None = None) -> str:
        head = f"{self.position} {self.severity.name.lower()}[{self.code}]: {self.message}"
        if source is None:
            return head
        lines = source.splitlines()
        if not 1 <= self.position.line <= len(lines):
            return head
        excerpt = lines[self.position.line - 1]
        caret = " " * max(self.position.column - 1, 0) + "^"
        return f"{head}\n    {excerpt}\n    {caret}"


class DiagnosticEngine:
    """Collects diagnostics, escalating the first fatal one into an exception."""

    def __init__(self, catalog: MessageCatalog | None = None) -> None:
        self._catalog = catalog or CATALOG
        self._records: list[Diagnostic] = []

    def emit(
        self,
        severity: Severity,
        code: str,
        key: str,
        position: SourcePosition = ORIGIN,
        **arguments: Any,
    ) -> Diagnostic:
        diagnostic = Diagnostic(severity, code, self._catalog(key, **arguments), position)
        self._records.append(diagnostic)
        EVENT_BUS.publish(
            DiagnosticRaised("diagnostics", diagnostic.message, severity.name.lower())
        )
        return diagnostic

    @property
    def records(self) -> tuple[Diagnostic, ...]:
        return tuple(self._records)

    @property
    def worst(self) -> Severity:
        return max((record.severity for record in self._records), default=Severity.HINT)

    def render(self, source: str | None = None) -> str:
        return "\n".join(record.render(source) for record in self._records)


DIAGNOSTICS: DiagnosticEngine = DiagnosticEngine()


class PreprocessorError(GlyphPlatformError):
    """Raised when macro expansion or pragma handling fails."""


class LexicalError(GlyphPlatformError):
    """Raised when the transducer reaches a rejecting configuration."""


class SyntaxError_(GlyphPlatformError):
    """Raised when the token stream violates the grammar."""


class SemanticError(GlyphPlatformError):
    """Raised when a well-formed tree is nonetheless meaningless."""


class TypeInferenceError(SemanticError):
    """Raised when unification of two type terms fails."""


class CodeGenerationError(GlyphPlatformError):
    """Raised when lowering or assembly cannot proceed."""


class ObjectFormatError(GlyphPlatformError):
    """Raised when a serialised object module fails validation."""


class ExecutionFault(GlyphPlatformError):
    """Raised when the virtual machine enters an illegal configuration."""


class GroupAxiomViolation(GlyphPlatformError):
    """Raised when a purported symmetry group is not in fact a group."""


class ResolutionFailure(GlyphPlatformError):
    """Raised when the composition root cannot satisfy a dependency."""


# ======================================================================
# Layer 0: functional core
# ======================================================================


class Result(Generic[_T], abc.ABC):
    """A right-biased disjunction between a value and a failure."""

    __slots__ = ()

    @property
    @abc.abstractmethod
    def is_ok(self) -> bool: ...

    @abc.abstractmethod
    def map(self, function: Callable[[_T], _U]) -> "Result[_U]": ...

    @abc.abstractmethod
    def bind(self, function: Callable[[_T], "Result[_U]"]) -> "Result[_U]": ...

    @abc.abstractmethod
    def unwrap_or_raise(self) -> _T: ...

    @abc.abstractmethod
    def unwrap_or(self, fallback: _T) -> _T: ...

    @staticmethod
    def attempt(thunk: Callable[[], _T]) -> "Result[_T]":
        try:
            return Ok(thunk())
        except GlyphPlatformError as error:
            return Err(error)

    @staticmethod
    def traverse(items: Iterable["Result[_T]"]) -> "Result[tuple[_T, ...]]":
        collected: list[_T] = []
        for item in items:
            if not item.is_ok:
                return typing.cast("Result[tuple[_T, ...]]", item)
            collected.append(item.unwrap_or_raise())
        return Ok(tuple(collected))


@dataclass(frozen=True, slots=True)
class Ok(Result[_T]):
    value: _T

    @property
    def is_ok(self) -> bool:
        return True

    def map(self, function: Callable[[_T], _U]) -> Result[_U]:
        return Ok(function(self.value))

    def bind(self, function: Callable[[_T], Result[_U]]) -> Result[_U]:
        return function(self.value)

    def unwrap_or_raise(self) -> _T:
        return self.value

    def unwrap_or(self, fallback: _T) -> _T:
        return self.value


@dataclass(frozen=True, slots=True)
class Err(Result[_T]):
    failure: GlyphPlatformError

    @property
    def is_ok(self) -> bool:
        return False

    def map(self, function: Callable[[_T], _U]) -> Result[_U]:
        return typing.cast(Result[_U], self)

    def bind(self, function: Callable[[_T], Result[_U]]) -> Result[_U]:
        return typing.cast(Result[_U], self)

    def unwrap_or_raise(self) -> _T:
        raise self.failure

    def unwrap_or(self, fallback: _T) -> _T:
        return fallback


@dataclass(frozen=True, slots=True)
class Maybe(Generic[_T]):
    """An option type used where ``None`` would have been perfectly adequate."""

    _value: _T | Sentinel = MISSING

    @classmethod
    def some(cls, value: _T) -> "Maybe[_T]":
        return cls(value)

    @classmethod
    def nothing(cls) -> "Maybe[_T]":
        return cls(MISSING)

    @property
    def is_present(self) -> bool:
        return self._value is not MISSING

    def map(self, function: Callable[[_T], _U]) -> "Maybe[_U]":
        if not self.is_present:
            return Maybe.nothing()
        return Maybe.some(function(typing.cast(_T, self._value)))

    def or_else(self, fallback: _T) -> _T:
        return typing.cast(_T, self._value) if self.is_present else fallback


class Lazy(Generic[_T]):
    """A thread-safe, memoising thunk."""

    __slots__ = ("_thunk", "_value", "_lock")

    def __init__(self, thunk: Callable[[], _T]) -> None:
        self._thunk = thunk
        self._value: _T | Sentinel = UNSET
        self._lock = threading.Lock()

    def force(self) -> _T:
        if self._value is UNSET:
            with self._lock:
                if self._value is UNSET:
                    self._value = self._thunk()
        return typing.cast(_T, self._value)


# ======================================================================
# Layer 1: layered configuration
# ======================================================================


@dataclass(frozen=True, slots=True)
class ConfigurationKey(Generic[_T]):
    """A typed, validated, self-documenting configuration key."""

    name: str
    default: _T
    coercer: Callable[[str], _T]
    validator: Callable[[_T], bool] = lambda _value: True
    documentation: str = ""


@runtime_checkable
class ConfigurationProvider(Protocol):
    """A source of raw string-valued configuration entries."""

    @property
    def origin(self) -> str: ...

    def lookup(self, name: str) -> str | None: ...


@dataclass(frozen=True, slots=True)
class MappingProvider:
    origin: str
    entries: Mapping[str, str]

    def lookup(self, name: str) -> str | None:
        return self.entries.get(name)


@dataclass(frozen=True, slots=True)
class EnvironmentProvider:
    prefix: str = "GLYPH_"

    @property
    def origin(self) -> str:
        return f"environment({self.prefix}*)"

    def lookup(self, name: str) -> str | None:
        return os.environ.get(self.prefix + name.upper().replace(".", "_"))


class Configuration:
    """The resolved, immutable view over an ordered provider chain."""

    def __init__(
        self,
        keys: Sequence[ConfigurationKey[Any]],
        providers: Sequence[ConfigurationProvider],
    ) -> None:
        self._resolved: dict[str, Any] = {}
        self._provenance: dict[str, str] = {}
        for key in keys:
            value, origin = key.default, "default"
            for provider in providers:
                raw = provider.lookup(key.name)
                if raw is None:
                    continue
                try:
                    value = key.coercer(raw)
                except (TypeError, ValueError) as error:
                    raise GlyphPlatformError(
                        f"configuration key {key.name!r} rejected {raw!r}: {error}"
                    ) from error
                origin = provider.origin
            if not key.validator(value):
                raise GlyphPlatformError(
                    f"configuration key {key.name!r} failed validation with {value!r}"
                )
            self._resolved[key.name] = value
            self._provenance[key.name] = origin

    def __getitem__(self, name: str) -> Any:
        try:
            return self._resolved[name]
        except KeyError as exc:
            raise ResolutionFailure(f"undeclared configuration key {name!r}") from exc

    def provenance(self, name: str) -> str:
        return self._provenance.get(name, "unknown")

    def render(self) -> str:
        return "\n".join(
            f"{name} = {value!r}  ({self._provenance[name]})"
            for name, value in sorted(self._resolved.items())
        )


def _parse_boolean(raw: str) -> bool:
    normalised = raw.strip().lower()
    if normalised in {"1", "true", "yes", "on"}:
        return True
    if normalised in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"not a boolean: {raw!r}")


CONFIGURATION_KEYS: Final[tuple[ConfigurationKey[Any], ...]] = (
    ConfigurationKey("lattice.order", DEFAULT_LATTICE_ORDER, int, lambda v: v > 0 and v % 2 == 1,
                     "edge length of the square lattice; must be odd"),
    ConfigurationKey("motif", DEFAULT_MOTIF, str, lambda v: bool(v), "registered motif key"),
    ConfigurationKey("language", "en", str, lambda v: v in MESSAGE_CATALOGUE, "diagnostic language"),
    ConfigurationKey("optimise", True, _parse_boolean, documentation="run the optimisation pipeline"),
    ConfigurationKey("jit", True, _parse_boolean, documentation="enable the threaded-code tier"),
    ConfigurationKey("roundtrip", True, _parse_boolean,
                     documentation="serialise and reload the object module before execution"),
    ConfigurationKey("theme", "asterisk", str, lambda v: bool(v), "registered rendering theme"),
    ConfigurationKey("workers", RASTERIZER_WORKER_COUNT, int, lambda v: v >= 1,
                     "rasteriser worker count"),
)


class ConfigurationBuilder:
    """Fluent builder assembling the provider chain in precedence order."""

    def __init__(self) -> None:
        self._providers: list[ConfigurationProvider] = []

    def with_defaults(self) -> "ConfigurationBuilder":
        return self

    def with_environment(self, prefix: str = "GLYPH_") -> "ConfigurationBuilder":
        self._providers.append(EnvironmentProvider(prefix))
        return self

    def with_mapping(self, origin: str, entries: Mapping[str, str]) -> "ConfigurationBuilder":
        self._providers.append(MappingProvider(origin, dict(entries)))
        return self

    def build(self) -> Configuration:
        return Configuration(CONFIGURATION_KEYS, tuple(self._providers))


class FeatureFlags:
    """A read-through façade turning boolean configuration keys into flags."""

    def __init__(self, configuration: Configuration) -> None:
        self._configuration = configuration

    def __getattr__(self, name: str) -> bool:
        return bool(self._configuration[name])


# ======================================================================
# Layer 2: discrete algebra and symmetry
# ======================================================================


class Coordinate:
    """An interned, immutable lattice cell address (flyweight pattern)."""

    __slots__ = ("row", "column", "__weakref__")

    _pool: ClassVar["weakref.WeakValueDictionary[tuple[int, int], Coordinate]"] = (
        weakref.WeakValueDictionary()
    )
    _pool_lock: ClassVar[threading.Lock] = threading.Lock()

    def __new__(cls, row: int, column: int) -> "Coordinate":
        key = (row, column)
        with cls._pool_lock:
            existing = cls._pool.get(key)
            if existing is not None:
                METRICS.increment("flyweight.coordinate.hits")
                return existing
            instance = super().__new__(cls)
            object.__setattr__(instance, "row", row)
            object.__setattr__(instance, "column", column)
            cls._pool[key] = instance
            METRICS.increment("flyweight.coordinate.misses")
            return instance

    def __setattr__(self, name: str, value: Any) -> None:  # pragma: no cover - guard
        raise ContractViolation("Coordinate instances are immutable")

    def __repr__(self) -> str:
        return f"Coordinate({self.row}, {self.column})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Coordinate):
            return NotImplemented
        return (self.row, self.column) == (other.row, other.column)

    def __hash__(self) -> int:
        return hash((Coordinate, self.row, self.column))

    def __iter__(self) -> Iterator[int]:
        yield self.row
        yield self.column

    def translated(self, delta_row: int, delta_column: int) -> "Coordinate":
        return Coordinate(self.row + delta_row, self.column + delta_column)

    def within(self, order: int) -> bool:
        return 0 <= self.row < order and 0 <= self.column < order


@dataclass(frozen=True, slots=True)
class LinearOperator:
    """An integer 2x2 endomorphism of the coordinate module."""

    a: int
    b: int
    c: int
    d: int

    IDENTITY: ClassVar["LinearOperator"]

    def __matmul__(self, other: "LinearOperator") -> "LinearOperator":
        return LinearOperator(
            self.a * other.a + self.b * other.c,
            self.a * other.b + self.b * other.d,
            self.c * other.a + self.d * other.c,
            self.c * other.b + self.d * other.d,
        )

    @property
    def determinant(self) -> int:
        return self.a * self.d - self.b * self.c

    def apply(self, row: int, column: int) -> tuple[int, int]:
        return (self.a * row + self.b * column, self.c * row + self.d * column)

    @property
    def transpose(self) -> "LinearOperator":
        return LinearOperator(self.a, self.c, self.b, self.d)

    @property
    def trace(self) -> int:
        return self.a + self.d

    def __str__(self) -> str:
        return f"[[{self.a} {self.b}] [{self.c} {self.d}]]"


LinearOperator.IDENTITY = LinearOperator(1, 0, 0, 1)

OPERATOR_BOUND: Final[int] = 1

# Class functions of an operator, each the character of a representation of
# the group solved for below: the one-dimensional trivial and sign
# representations, and the defining two-dimensional one.
CHARACTERS: Final[Mapping[str, Callable[[LinearOperator], int]]] = {
    "unit": lambda operator: 1,
    "sign": lambda operator: operator.determinant,
    "standard": lambda operator: operator.trace,
}


@functools.lru_cache(maxsize=None)
def solve_operators(admitted: FrozenSet[int]) -> tuple[LinearOperator, ...]:
    """Every integer operator satisfying the relations, over a bounded box.

    Four unknowns and two relations: the operator composed with its own
    transpose is the identity, and the sign character takes at it one of the
    ``admitted`` values.  Nothing is tabulated - the entries are searched for
    at the moment they are wanted, and what is written down is only what they
    have to satisfy.

    The first relation is what confines the search to a box at all: it forces
    every row to have unit norm, so no entry can exceed one however wide the
    span is opened.
    """
    span = range(-OPERATOR_BOUND, OPERATOR_BOUND + 1)
    sign = CHARACTERS["sign"]
    solutions = tuple(
        candidate
        for a, b, c, d in itertools.product(span, repeat=4)
        if sign(candidate := LinearOperator(a, b, c, d)) in admitted
        and candidate.transpose @ candidate == LinearOperator.IDENTITY
    )
    if not solutions:
        raise GroupAxiomViolation(
            f"the relations admit no operator of sign {sorted(admitted)}"
        )
    return solutions


@dataclass(frozen=True, slots=True)
class AffineTransform:
    """A linear operator conjugated by a translation to the lattice centroid."""

    linear: LinearOperator
    centre: Coordinate

    def __call__(self, point: Coordinate) -> Coordinate:
        row, column = self.linear.apply(
            point.row - self.centre.row, point.column - self.centre.column
        )
        return Coordinate(row + self.centre.row, column + self.centre.column)

    def compose(self, other: "AffineTransform") -> "AffineTransform":
        if other.centre != self.centre:  # pragma: no cover - defensive
            raise GroupAxiomViolation("cannot compose transforms about distinct centroids")
        return AffineTransform(self.linear @ other.linear, self.centre)

    def __str__(self) -> str:
        return str(self.linear)


@dataclass(frozen=True, slots=True)
class SymmetryGroup:
    """A finite transformation group presented by generators and relations."""

    elements: tuple[AffineTransform, ...]
    centre: Coordinate
    presentation: str

    @classmethod
    @woven
    def generated_by(
        cls,
        generators: Sequence[LinearOperator],
        centre: Coordinate,
        expected_order: int,
        presentation: str = "<generators>",
    ) -> "SymmetryGroup":
        """Closes the generator set under composition and audits the axioms."""
        identity = AffineTransform(LinearOperator.IDENTITY, centre)
        frontier = [identity, *(AffineTransform(g, centre) for g in generators)]
        seen: dict[LinearOperator, AffineTransform] = {}
        while frontier:
            candidate = frontier.pop()
            if candidate.linear in seen:
                continue
            seen[candidate.linear] = candidate
            for known in list(seen.values()):
                for product in (candidate.compose(known), known.compose(candidate)):
                    if product.linear not in seen:
                        frontier.append(product)
        elements = tuple(sorted(seen.values(), key=lambda t: (t.linear.a, t.linear.b, t.linear.c, t.linear.d)))
        group = cls(elements, centre, presentation)
        group.audit(expected_order)
        return group

    @property
    def order(self) -> int:
        return len(self.elements)

    @property
    def identity(self) -> AffineTransform:
        return AffineTransform(LinearOperator.IDENTITY, self.centre)

    def audit(self, expected_order: int | None = None) -> "SymmetryGroup":
        """Verifies closure, identity, inverses, associativity and cardinality."""
        linears = {element.linear for element in self.elements}
        if LinearOperator.IDENTITY not in linears:
            raise GroupAxiomViolation("the identity is absent from the generated set")
        for left in self.elements:
            for right in self.elements:
                if left.compose(right).linear not in linears:
                    raise GroupAxiomViolation(f"closure fails for {left} ∘ {right}")
        for element in self.elements:
            if not any(
                element.compose(other).linear == LinearOperator.IDENTITY
                for other in self.elements
            ):
                raise GroupAxiomViolation(f"element {element} has no inverse")
        sample = self.elements[: min(4, len(self.elements))]
        for x, y, z in itertools.product(sample, repeat=3):
            if x.compose(y).compose(z).linear != x.compose(y.compose(z)).linear:
                raise GroupAxiomViolation("composition is not associative")
        if expected_order is not None and self.order != expected_order:
            raise GroupAxiomViolation(
                f"generated group has order {self.order}, expected {expected_order}"
            )
        METRICS.increment("group.audits")
        return self

    def cayley_table(self) -> tuple[tuple[int, ...], ...]:
        index = {element.linear: position for position, element in enumerate(self.elements)}
        return tuple(
            tuple(index[left.compose(right).linear] for right in self.elements)
            for left in self.elements
        )

    def element_order(self, element: AffineTransform) -> int:
        current, count = element, 1
        while current.linear != LinearOperator.IDENTITY:
            current = current.compose(element)
            count += 1
            if count > self.order:  # pragma: no cover - impossible in a finite group
                raise GroupAxiomViolation("element order exceeds group order")
        return count

    def cyclic_subgroups(self) -> tuple[frozenset[LinearOperator], ...]:
        subgroups: set[frozenset[LinearOperator]] = set()
        for element in self.elements:
            members, current = set(), element
            while True:
                members.add(current.linear)
                if current.linear == LinearOperator.IDENTITY:
                    break
                current = current.compose(element)
            subgroups.add(frozenset(members))
        return tuple(sorted(subgroups, key=len))

    def satisfies_lagrange(self) -> bool:
        return all(self.order % len(subgroup) == 0 for subgroup in self.cyclic_subgroups())

    def orbit(self, point: Coordinate) -> frozenset[Coordinate]:
        return frozenset(element(point) for element in self.elements)

    def stabiliser(self, point: Coordinate) -> tuple[AffineTransform, ...]:
        return tuple(element for element in self.elements if element(point) == point)

    def satisfies_orbit_stabiliser(self, point: Coordinate) -> bool:
        return len(self.orbit(point)) * len(self.stabiliser(point)) == self.order

    def project(
        self, support: Iterable[Coordinate], character: Callable[[LinearOperator], int]
    ) -> Mapping[Coordinate, int]:
        """The indicator of ``support`` under the operator this character names.

        Each element contributes the character's value at it to every cell the
        element carries the support onto.  The scaling by the group's order is
        dropped, because the only question ever put to a coefficient is
        whether it vanishes.
        """
        material = tuple(support)
        coefficients: dict[Coordinate, int] = collections.defaultdict(int)
        for element in self.elements:
            value = character(element.linear)
            if not value:
                continue
            for point in material:
                coefficients[element(point)] += value
        return coefficients

    @woven
    def close(self, support: Iterable[Coordinate]) -> frozenset[Coordinate]:
        """The cells where the projection for the unit character does not vanish.

        It has to be that character and no other: an indicator is nowhere
        negative, so under the unit character nothing in a sum can cancel
        anything else, and a cell survives exactly when some element carries
        the support onto it.  Under a character that takes more than one
        value, cells reachable by elements of opposing sign fall out.
        """
        return frozenset(
            point
            for point, value in self.project(support, CHARACTERS["unit"]).items()
            if value
        )

    def is_invariant(self, support: FrozenSet[Coordinate]) -> bool:
        return all(
            frozenset(element(point) for point in support) == support
            for element in self.elements
        )

    def render(self) -> str:
        rows = [
            f"presentation: {self.presentation}",
            f"order: {self.order}",
            f"lagrange: {'satisfied' if self.satisfies_lagrange() else 'VIOLATED'}",
            "elements:",
        ]
        rows.extend(
            f"  g{index} = {element}  (order {self.element_order(element)})"
            for index, element in enumerate(self.elements)
        )
        rows.append("cayley table:")
        for index, row in enumerate(self.cayley_table()):
            rows.append(f"  g{index} | " + " ".join(f"g{cell}" for cell in row))
        rows.append("characters:")
        width = max(len(name) for name in CHARACTERS)
        for name, character in CHARACTERS.items():
            values = " ".join(f"{character(e.linear):>3}" for e in self.elements)
            rows.append(f"  {name:<{width}} | {values}")
        return "\n".join(rows)


# ======================================================================
# Layer 3: preprocessing
# ======================================================================


@dataclass(frozen=True, slots=True)
class MacroDefinition:
    name: str
    replacement: str
    defined_at: int


@dataclass(frozen=True, slots=True)
class PreprocessedUnit:
    """Expanded text plus the line map back to the original translation unit."""

    text: str
    line_map: tuple[int, ...]
    macros: Mapping[str, MacroDefinition]
    pragmas: Mapping[str, str]

    def original_line(self, line: int) -> int:
        if 1 <= line <= len(self.line_map):
            return self.line_map[line - 1]
        return line


class Preprocessor:
    """Handles line splicing, ``#define`` macros and ``#pragma`` directives."""

    _DIRECTIVE = re.compile(r"^\s*#\s*(?P<directive>[a-z]+)\s*(?P<body>.*)$")
    _DEFINE = re.compile(r"^(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s+(?P<value>.*)$")
    _EXPANSION_LIMIT: Final[int] = 32

    def __init__(self, source: str, diagnostics: DiagnosticEngine | None = None) -> None:
        self._source = source
        self._diagnostics = diagnostics or DIAGNOSTICS
        self._macros: dict[str, MacroDefinition] = {}
        self._pragmas: dict[str, str] = {}

    @woven
    def run(self) -> PreprocessedUnit:
        spliced = self._splice(self._source.splitlines())
        emitted: list[str] = []
        line_map: list[int] = []
        for original_line, text in spliced:
            directive = self._DIRECTIVE.match(text)
            if directive is not None:
                self._handle(directive.group("directive"), directive.group("body"), original_line)
                continue
            emitted.append(self._expand(text, original_line))
            line_map.append(original_line)
        METRICS.increment("preprocessor.lines", len(emitted))
        return PreprocessedUnit(
            "\n".join(emitted), tuple(line_map), dict(self._macros), dict(self._pragmas)
        )

    @staticmethod
    def _splice(lines: Sequence[str]) -> list[tuple[int, str]]:
        spliced: list[tuple[int, str]] = []
        buffer, anchor = "", 1
        for number, line in enumerate(lines, start=1):
            if not buffer:
                anchor = number
            if line.rstrip().endswith("\\"):
                buffer += line.rstrip()[:-1]
                continue
            spliced.append((anchor, buffer + line))
            buffer = ""
        if buffer:
            spliced.append((anchor, buffer))
        return spliced

    def _handle(self, directive: str, body: str, line: int) -> None:
        if directive == "define":
            match = self._DEFINE.match(body.strip())
            if match is None:
                raise PreprocessorError(f"malformed #define on line {line}: {body!r}")
            name = match.group("name")
            self._macros[name] = MacroDefinition(name, match.group("value").strip(), line)
        elif directive == "pragma":
            key, _, value = body.strip().partition(" ")
            self._pragmas[key] = value.strip()
        elif directive == "undef":
            self._macros.pop(body.strip(), None)
        else:
            raise PreprocessorError(f"unknown directive #{directive} on line {line}")

    def _expand(self, text: str, line: int) -> str:
        for _ in range(self._EXPANSION_LIMIT):
            replaced = re.sub(
                r"\b[A-Za-z_][A-Za-z0-9_]*\b",
                lambda match: self._macros[match.group(0)].replacement
                if match.group(0) in self._macros
                else match.group(0),
                text,
            )
            if replaced == text:
                return text
            text = replaced
        self._diagnostics.emit(
            Severity.FATAL, "PP0001", "diag.macro_recursion", SourcePosition(line, 1), name=text
        )
        raise PreprocessorError(f"macro expansion did not converge on line {line}")


# ======================================================================
# Layer 4: lexical analysis
# ======================================================================


# ----------------------------------------------------------------------
# The surface of the language, as one image compiled at import
# ----------------------------------------------------------------------
#
# Every word the tables stand on is interned once and addressed by an offset
# in base thirty-six thereafter, so what is written down is a vocabulary and
# runs of offsets into it rather than the tables themselves.  The five
# sections are the vocabulary, the reserved words, the punctuation, the
# productions, and the infix operators with their binding powers.

FRONT_END_IMAGE: Final[str] = (
    "about antidiagonal at centroid column cyclic diagonal dihedral emit for "
    "in lattice let order paint row span stroke symmetry ( LEFT_PAREN ) "
    "RIGHT_PAREN * STAR + PLUS - MINUS / SLASH ; SEMICOLON = EQUALS { "
    "LEFT_BRACE } RIGHT_BRACE program body 'lattice' 'order' expression ';' "
    "'symmetry' family INTEGER 'about' 'centroid' 'cyclic' 'dihedral' "
    "statement ε binding emission painting iteration 'let' IDENTIFIER '=' "
    "'stroke' run 'emit' 'paint' 'for' 'in' '..' '{' '}' orientation 'at' "
    "'span' 'row' 'column' 'diagonal' 'antidiagonal' term expression_tail '+'"
    " '-' factor term_tail '*' '/' '(' ')'~0.1.2.3.4.5.6.7.8.9.a.b.c.d.e.f.g."
    "h.i~j.k,l.m,n.o,p.q,r.s,t.u,v.w,x.y,z.10,11.12~13:b.i.14,b:15.16.17.18,i"
    ":19.1a.1b.1c.1d.18,1a:1e|1f,14:1g.14|1h,1g:1i|h|1j|1k|1l,1i:1m.1n.1o.17."
    "18,h:1p.1n.1o.1q.18,1j:1r.1n.18,1k:1s.1q.18,1l:1t.1n.1u.17.1v.17.1w.14.1"
    "x,1q:1y.1z.17.20.17.1v.17,1y:21|22|23|24,17:25.26,26:27.25.26|28.25.26|1"
    "h,25:29.2a,2a:2b.29.2a|2c.29.2a|1h,29:1b|1n|28.29|2d.17.2e~q.p.a,s.r.a,o"
    ".n.k,u.t.k"
)


@dataclass(frozen=True, slots=True)
class FrontEndImage:
    """What the image stands for: the tables layers 4 and 5 are driven by."""

    keywords: FrozenSet[str]
    punctuation: Mapping[str, str]
    grammar: Mapping[str, tuple[tuple[str, ...], ...]]
    infix: tuple[tuple[str, str, int], ...]


def compile_front_end(image: str) -> FrontEndImage:
    """Reads an image back into the tables it stands for."""
    try:
        vocabulary, reserved, marks, rules, operators = image.split("~")
    except ValueError as exc:
        raise LexicalError("the image does not have five sections") from exc
    words = vocabulary.split(" ")

    def at(code: str) -> str:
        try:
            return words[int(code, 36)]
        except (ValueError, IndexError) as exc:
            raise LexicalError(f"{code!r} addresses no word") from exc

    def fields(record: str, count: int) -> list[str]:
        parts = record.split(".")
        if len(parts) != count:
            raise LexicalError(f"{record!r} is not {count} fields")
        return parts

    return FrontEndImage(
        frozenset(at(code) for code in reserved.split(".")),
        {at(a): at(b) for a, b in (fields(r, 2) for r in marks.split(","))},
        {
            at(head): tuple(
                tuple(at(symbol) for symbol in alternative.split("."))
                for alternative in body.split("|")
            )
            for head, _, body in (r.partition(":") for r in rules.split(","))
        },
        tuple(
            (at(a), at(b), int(c, 36))
            for a, b, c in (fields(r, 3) for r in operators.split(","))
        ),
    )


FRONT_END: Final[FrontEndImage] = compile_front_end(FRONT_END_IMAGE)


class TokenKind(enum.Enum):
    KEYWORD = enum.auto()
    IDENTIFIER = enum.auto()
    INTEGER = enum.auto()
    PLUS = enum.auto()
    MINUS = enum.auto()
    STAR = enum.auto()
    SLASH = enum.auto()
    EQUALS = enum.auto()
    RANGE = enum.auto()
    SEMICOLON = enum.auto()
    LEFT_PAREN = enum.auto()
    RIGHT_PAREN = enum.auto()
    LEFT_BRACE = enum.auto()
    RIGHT_BRACE = enum.auto()
    END_OF_INPUT = enum.auto()


KEYWORDS: Final[FrozenSet[str]] = FRONT_END.keywords

PUNCTUATION: Final[Mapping[str, TokenKind]] = {
    mark: TokenKind[name] for mark, name in FRONT_END.punctuation.items()
}


# ----------------------------------------------------------------------
# The scanner, as one image compiled at import
# ----------------------------------------------------------------------
#
# Three tables: which class a character falls into, tried in the order
# listed; what a state accepts as when a lexeme ends in it; and where a state
# goes on a class, with what to do on the way.  The driver below consults
# them and holds no knowledge of the language itself.

SCANNER_IMAGE: Final[str] = (
    "alpha LETTER digit DIGIT newline NEWLINE blank SPACE dot DOT hash "
    "COMMENT punct PUNCTUATION any OTHER IDENTIFIER reserved NUMERAL INTEGER "
    "RANGE_PENDING RANGE GROUND ACCUMULATE SKIP EMIT_PUNCTUATION REJECT FAIL~"
    "0.1,2.3,4.5,6.7,8.9,a.b,c.d,e.f~g.h,i.j,k.l~m.1.g.n,m.3.i.n,m.9.k.n,m.7."
    "m.o,m.5.m.o,m.b.b.o,m.d.m.p,m.f.q.r,g.1.g.n,g.3.g.n,i.3.i.n,k.9.k.n,b.5."
    "m.o"
)


@dataclass(frozen=True, slots=True)
class ScannerImage:
    """What the image stands for: the three tables layer 4 is driven by."""

    classifiers: tuple[tuple[str, str], ...]
    accepting: Mapping[str, str]
    transitions: Mapping[tuple[str, str], tuple[str, str]]


def compile_scanner(image: str) -> ScannerImage:
    """Reads an image back into the tables the transducer consults."""
    try:
        vocabulary, classes, accepting, transitions = image.split("~")
    except ValueError as exc:
        raise LexicalError("the image does not have four sections") from exc
    words = vocabulary.split(" ")

    def at(code: str) -> str:
        try:
            return words[int(code, 36)]
        except (ValueError, IndexError) as exc:
            raise LexicalError(f"{code!r} addresses no word") from exc

    return ScannerImage(
        tuple((at(a), at(b)) for a, b in (r.split(".") for r in classes.split(","))),
        {at(a): at(b) for a, b in (r.split(".") for r in accepting.split(","))},
        {
            (at(a), at(b)): (at(c), at(d))
            for a, b, c, d in (r.split(".") for r in transitions.split(","))
        },
    )


SCANNER: Final[ScannerImage] = compile_scanner(SCANNER_IMAGE)

CHARACTER_SOURCES: Final[Mapping[str, Callable[[str], bool]]] = {
    "alpha": lambda character: character in string.ascii_letters or character == "_",
    "digit": lambda character: character in string.digits,
    "newline": lambda character: character == "\n",
    "blank": lambda character: character in " \t\r",
    "dot": lambda character: character == ".",
    "hash": lambda character: character == "#",
    "punct": lambda character: character in PUNCTUATION,
    "any": lambda character: True,
}


class CharacterClass(enum.Enum):
    LETTER = enum.auto()
    DIGIT = enum.auto()
    SPACE = enum.auto()
    DOT = enum.auto()
    COMMENT = enum.auto()
    PUNCTUATION = enum.auto()
    NEWLINE = enum.auto()
    OTHER = enum.auto()

    @classmethod
    def of(cls, character: str) -> "CharacterClass":
        for source, name in SCANNER.classifiers:
            if CHARACTER_SOURCES[source](character):
                return cls[name]
        raise LexicalError(f"{character!r} falls into no class")  # pragma: no cover


class TransducerState(enum.Enum):
    GROUND = enum.auto()
    IDENTIFIER = enum.auto()
    NUMERAL = enum.auto()
    RANGE_PENDING = enum.auto()
    COMMENT = enum.auto()
    REJECT = enum.auto()


class TransducerAction(enum.Enum):
    SKIP = enum.auto()
    ACCUMULATE = enum.auto()
    EMIT_THEN_RESCAN = enum.auto()
    EMIT_PUNCTUATION = enum.auto()
    FAIL = enum.auto()


TRANSITION_TABLE: Final[
    Mapping[tuple[TransducerState, CharacterClass], tuple[TransducerState, TransducerAction]]
] = {
    (TransducerState[state], CharacterClass[characters]): (
        TransducerState[target],
        TransducerAction[action],
    )
    for (state, characters), (target, action) in SCANNER.transitions.items()
}


def _accepting_table() -> Mapping["TransducerState", Callable[[str], "TokenKind"]]:
    """What each accepting state answers with, the reserved words apart."""

    def constant(name: str) -> Callable[[str], TokenKind]:
        return lambda lexeme: TokenKind[name]

    def reserved(lexeme: str) -> TokenKind:
        return TokenKind.KEYWORD if lexeme in KEYWORDS else TokenKind.IDENTIFIER

    return {
        TransducerState[state]: reserved if kind == "reserved" else constant(kind)
        for state, kind in SCANNER.accepting.items()
    }


@dataclass(frozen=True, slots=True)
class Token:
    """A lexeme, its classification, its provenance and its leading trivia."""

    kind: TokenKind
    lexeme: str
    position: SourcePosition
    trivia: str = ""

    @property
    def value(self) -> int:
        return int(self.lexeme)

    def __str__(self) -> str:
        return f"{self.kind.name}({self.lexeme!r})@{self.position}"


class FiniteStateTransducer:
    """A table-driven maximal-munch scanner over the preprocessed unit."""

    _ACCEPTING: Final[Mapping[TransducerState, Callable[[str], TokenKind]]] = (
        _accepting_table()
    )

    def __init__(self, unit: PreprocessedUnit | str) -> None:
        self._unit = unit if isinstance(unit, PreprocessedUnit) else PreprocessedUnit(unit, (), {}, {})
        self._text = self._unit.text
        self._state = TransducerState.GROUND
        self._buffer: list[str] = []
        self._anchor = ORIGIN
        self._tokens: list[Token] = []
        self._trivia: list[str] = []

    @woven
    def tokenize(self) -> tuple[Token, ...]:
        line, column, index = 1, 1, 0
        while index <= len(self._text):
            character = self._text[index] if index < len(self._text) else "\0"
            klass = CharacterClass.of(character) if index < len(self._text) else CharacterClass.OTHER
            at_end = index >= len(self._text)
            position = SourcePosition(line, column, index)
            if at_end:
                self._flush(position)
                break
            key = (self._state, klass)
            transition = TRANSITION_TABLE.get(key)
            if transition is None:
                if self._state is TransducerState.COMMENT:
                    self._trivia.append(character)
                    index, line, column = self._advance(character, index, line, column)
                    continue
                if self._state in self._ACCEPTING:
                    self._flush(position)
                    continue
                raise LexicalError(
                    f"transducer rejected {character!r} in state {self._state.name} at {position}"
                )
            next_state, action = transition
            if action is TransducerAction.FAIL:
                raise LexicalError(f"illegal character {character!r} at {position}")
            if action is TransducerAction.SKIP:
                self._trivia.append(character)
            elif action is TransducerAction.ACCUMULATE:
                if not self._buffer:
                    self._anchor = position
                self._buffer.append(character)
            elif action is TransducerAction.EMIT_PUNCTUATION:
                self._tokens.append(
                    Token(PUNCTUATION[character], character, position, self._drain_trivia())
                )
            self._state = next_state
            index, line, column = self._advance(character, index, line, column)
        self._tokens.append(
            Token(TokenKind.END_OF_INPUT, "", SourcePosition(line, column, len(self._text)), self._drain_trivia())
        )
        METRICS.increment("lexer.tokens", len(self._tokens))
        return tuple(self._tokens)

    @staticmethod
    def _advance(character: str, index: int, line: int, column: int) -> tuple[int, int, int]:
        if character == "\n":
            return index + 1, line + 1, 1
        return index + 1, line, column + 1

    def _drain_trivia(self) -> str:
        trivia = "".join(self._trivia)
        self._trivia.clear()
        return trivia

    def _flush(self, position: SourcePosition) -> None:
        if not self._buffer:
            self._state = TransducerState.GROUND
            return
        lexeme = "".join(self._buffer)
        classifier = self._ACCEPTING.get(self._state)
        if classifier is None:  # pragma: no cover - defensive
            raise LexicalError(f"non-accepting state {self._state.name} at {position}")
        kind = classifier(lexeme)
        if kind is TokenKind.RANGE and lexeme != "..":
            raise LexicalError(f"malformed range operator {lexeme!r} at {self._anchor}")
        self._tokens.append(Token(kind, lexeme, self._anchor, self._drain_trivia()))
        self._buffer.clear()
        self._state = TransducerState.GROUND


class TokenStream:
    """A cursor over the token tuple offering lookahead and recovery."""

    def __init__(self, tokens: Sequence[Token]) -> None:
        self._tokens = tuple(tokens)
        self._cursor = 0

    def peek(self, distance: int = 0) -> Token:
        index = min(self._cursor + distance, len(self._tokens) - 1)
        return self._tokens[index]

    def advance(self) -> Token:
        token = self.peek()
        if token.kind is not TokenKind.END_OF_INPUT:
            self._cursor += 1
        return token

    def check(self, kind: TokenKind, lexeme: str | None = None) -> bool:
        token = self.peek()
        return token.kind is kind and (lexeme is None or token.lexeme == lexeme)

    def match(self, kind: TokenKind, lexeme: str | None = None) -> bool:
        if self.check(kind, lexeme):
            self.advance()
            return True
        return False

    def expect(self, kind: TokenKind, lexeme: str | None = None) -> Token:
        if self.check(kind, lexeme):
            return self.advance()
        expected = lexeme if lexeme is not None else kind.name
        found = self.peek()
        DIAGNOSTICS.emit(
            Severity.FATAL, "SY0001", "diag.unexpected_token", found.position,
            expected=expected, found=str(found),
        )
        raise SyntaxError_(
            f"{found.position}: expected {expected}, found {found}"
        )

    def synchronise(self, anchors: FrozenSet[TokenKind]) -> None:
        while self.peek().kind not in anchors and self.peek().kind is not TokenKind.END_OF_INPUT:
            self.advance()


# ======================================================================
# Layer 5: grammar metatheory and syntax analysis
# ======================================================================


EPSILON: Final[str] = "ε"

GSL_GRAMMAR: Final[Mapping[str, tuple[tuple[str, ...], ...]]] = FRONT_END.grammar


class GrammarAnalyzer:
    """Computes FIRST and FOLLOW sets so that error recovery looks principled."""

    def __init__(self, productions: Mapping[str, tuple[tuple[str, ...], ...]], start: str) -> None:
        self._productions = productions
        self._start = start
        self._first = Lazy(self._compute_first)
        self._follow = Lazy(self._compute_follow)

    def is_nonterminal(self, symbol: str) -> bool:
        return symbol in self._productions

    def _compute_first(self) -> Mapping[str, frozenset[str]]:
        first: dict[str, set[str]] = {name: set() for name in self._productions}
        changed = True
        while changed:
            changed = False
            for head, alternatives in self._productions.items():
                for alternative in alternatives:
                    for symbol in alternative:
                        if symbol == EPSILON:
                            addition = {EPSILON}
                        elif self.is_nonterminal(symbol):
                            addition = first[symbol] - {EPSILON}
                        else:
                            addition = {symbol}
                        if not addition <= first[head]:
                            first[head] |= addition
                            changed = True
                        if symbol == EPSILON or (
                            self.is_nonterminal(symbol) and EPSILON in first[symbol]
                        ):
                            continue
                        break
                    else:
                        if EPSILON not in first[head]:
                            first[head].add(EPSILON)
                            changed = True
        return {head: frozenset(members) for head, members in first.items()}

    def _compute_follow(self) -> Mapping[str, frozenset[str]]:
        first = self._first.force()
        follow: dict[str, set[str]] = {name: set() for name in self._productions}
        follow[self._start].add("$")
        changed = True
        while changed:
            changed = False
            for head, alternatives in self._productions.items():
                for alternative in alternatives:
                    for index, symbol in enumerate(alternative):
                        if not self.is_nonterminal(symbol):
                            continue
                        trailer: set[str] = set()
                        nullable_tail = True
                        for successor in alternative[index + 1:]:
                            if self.is_nonterminal(successor):
                                trailer |= first[successor] - {EPSILON}
                                if EPSILON not in first[successor]:
                                    nullable_tail = False
                                    break
                            else:
                                trailer.add(successor)
                                nullable_tail = False
                                break
                        if nullable_tail:
                            trailer |= follow[head]
                        if not trailer <= follow[symbol]:
                            follow[symbol] |= trailer
                            changed = True
        return {head: frozenset(members) for head, members in follow.items()}

    @property
    def first(self) -> Mapping[str, frozenset[str]]:
        return self._first.force()

    @property
    def follow(self) -> Mapping[str, frozenset[str]]:
        return self._follow.force()

    def render(self) -> str:
        lines = ["FIRST sets:"]
        lines.extend(
            f"  FIRST({head}) = {{ {', '.join(sorted(members))} }}"
            for head, members in sorted(self.first.items())
        )
        lines.append("FOLLOW sets:")
        lines.extend(
            f"  FOLLOW({head}) = {{ {', '.join(sorted(members))} }}"
            for head, members in sorted(self.follow.items())
        )
        return "\n".join(lines)


GRAMMAR: Final[GrammarAnalyzer] = GrammarAnalyzer(GSL_GRAMMAR, "program")

STATEMENT_ANCHORS: Final[FrozenSet[TokenKind]] = frozenset(
    {TokenKind.SEMICOLON, TokenKind.RIGHT_BRACE, TokenKind.END_OF_INPUT}
)


class Orientation(enum.Enum):
    ROW = "row"
    COLUMN = "column"
    DIAGONAL = "diagonal"
    ANTIDIAGONAL = "antidiagonal"


class SymmetryFamily(enum.Enum):
    CYCLIC = "cyclic"
    DIHEDRAL = "dihedral"

    @property
    def generators(self) -> tuple[LinearOperator, ...]:
        return solve_operators(self.admitted)

    @property
    def admitted(self) -> FrozenSet[int]:
        """The values the sign character may take across this family.

        The cyclic family is where that character is trivial, which is to say
        its kernel; the dihedral one is everything the relations allow.
        """
        return frozenset({1}) if self is SymmetryFamily.CYCLIC else frozenset({1, -1})

    def expected_order(self, cardinality: int) -> int:
        return cardinality if self is SymmetryFamily.CYCLIC else 2 * cardinality


class AstVisitor(Generic[_T], abc.ABC):
    """Double-dispatch visitor over the abstract syntax tree."""

    def visit(self, node: "AstNode") -> _T:
        return node.accept(self)

    @abc.abstractmethod
    def visit_integer_literal(self, node: "IntegerLiteral") -> _T: ...
    @abc.abstractmethod
    def visit_symbol_reference(self, node: "SymbolReference") -> _T: ...
    @abc.abstractmethod
    def visit_unary_operation(self, node: "UnaryOperation") -> _T: ...
    @abc.abstractmethod
    def visit_binary_operation(self, node: "BinaryOperation") -> _T: ...
    @abc.abstractmethod
    def visit_run(self, node: "Run") -> _T: ...
    @abc.abstractmethod
    def visit_binding(self, node: "Binding") -> _T: ...
    @abc.abstractmethod
    def visit_stroke_declaration(self, node: "StrokeDeclaration") -> _T: ...
    @abc.abstractmethod
    def visit_emission(self, node: "Emission") -> _T: ...
    @abc.abstractmethod
    def visit_painting(self, node: "Painting") -> _T: ...
    @abc.abstractmethod
    def visit_iteration(self, node: "Iteration") -> _T: ...
    @abc.abstractmethod
    def visit_lattice_declaration(self, node: "LatticeDeclaration") -> _T: ...
    @abc.abstractmethod
    def visit_symmetry_declaration(self, node: "SymmetryDeclaration") -> _T: ...
    @abc.abstractmethod
    def visit_program(self, node: "Program") -> _T: ...


class AstNode(abc.ABC):
    """Base class of every syntax tree node."""

    __slots__ = ()

    @abc.abstractmethod
    def accept(self, visitor: AstVisitor[_T]) -> _T: ...

    @property
    def children(self) -> tuple["AstNode", ...]:
        collected: list[AstNode] = []
        for name in getattr(self, "__dataclass_fields__", {}):
            value = getattr(self, name)
            if isinstance(value, AstNode):
                collected.append(value)
            elif isinstance(value, tuple):
                collected.extend(item for item in value if isinstance(item, AstNode))
        return tuple(collected)

    def walk(self) -> Iterator["AstNode"]:
        yield self
        for child in self.children:
            yield from child.walk()

    def render(self, depth: int = 0) -> str:
        fields = {
            name: value
            for name, value in ((n, getattr(self, n)) for n in getattr(self, "__dataclass_fields__", {}))
            if not isinstance(value, (AstNode, tuple))
        }
        annotation = " ".join(f"{k}={v!r}" for k, v in fields.items())
        lines = ["  " * depth + f"{type(self).__name__} {annotation}".rstrip()]
        lines.extend(child.render(depth + 1) for child in self.children)
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class IntegerLiteral(AstNode):
    value: int
    position: SourcePosition = ORIGIN

    def accept(self, visitor: AstVisitor[_T]) -> _T:
        return visitor.visit_integer_literal(self)


@dataclass(frozen=True, slots=True)
class SymbolReference(AstNode):
    name: str
    position: SourcePosition = ORIGIN

    def accept(self, visitor: AstVisitor[_T]) -> _T:
        return visitor.visit_symbol_reference(self)


@dataclass(frozen=True, slots=True)
class UnaryOperation(AstNode):
    operator_symbol: str
    operand: AstNode
    position: SourcePosition = ORIGIN

    def accept(self, visitor: AstVisitor[_T]) -> _T:
        return visitor.visit_unary_operation(self)


@dataclass(frozen=True, slots=True)
class BinaryOperation(AstNode):
    operator_symbol: str
    left: AstNode
    right: AstNode
    position: SourcePosition = ORIGIN

    def accept(self, visitor: AstVisitor[_T]) -> _T:
        return visitor.visit_binary_operation(self)


@dataclass(frozen=True, slots=True)
class Run(AstNode):
    orientation: Orientation
    index: AstNode
    lower: AstNode
    upper: AstNode
    position: SourcePosition = ORIGIN

    def accept(self, visitor: AstVisitor[_T]) -> _T:
        return visitor.visit_run(self)


@dataclass(frozen=True, slots=True)
class Binding(AstNode):
    name: str
    value: AstNode
    position: SourcePosition = ORIGIN

    def accept(self, visitor: AstVisitor[_T]) -> _T:
        return visitor.visit_binding(self)


@dataclass(frozen=True, slots=True)
class StrokeDeclaration(AstNode):
    name: str
    run: Run
    position: SourcePosition = ORIGIN

    def accept(self, visitor: AstVisitor[_T]) -> _T:
        return visitor.visit_stroke_declaration(self)


@dataclass(frozen=True, slots=True)
class Emission(AstNode):
    name: str
    position: SourcePosition = ORIGIN

    def accept(self, visitor: AstVisitor[_T]) -> _T:
        return visitor.visit_emission(self)


@dataclass(frozen=True, slots=True)
class Painting(AstNode):
    run: Run
    position: SourcePosition = ORIGIN

    def accept(self, visitor: AstVisitor[_T]) -> _T:
        return visitor.visit_painting(self)


@dataclass(frozen=True, slots=True)
class Iteration(AstNode):
    variable: str
    lower: AstNode
    upper: AstNode
    body: tuple[AstNode, ...]
    position: SourcePosition = ORIGIN

    def accept(self, visitor: AstVisitor[_T]) -> _T:
        return visitor.visit_iteration(self)


@dataclass(frozen=True, slots=True)
class LatticeDeclaration(AstNode):
    order: AstNode
    position: SourcePosition = ORIGIN

    def accept(self, visitor: AstVisitor[_T]) -> _T:
        return visitor.visit_lattice_declaration(self)


@dataclass(frozen=True, slots=True)
class SymmetryDeclaration(AstNode):
    family: SymmetryFamily
    cardinality: int
    position: SourcePosition = ORIGIN

    def accept(self, visitor: AstVisitor[_T]) -> _T:
        return visitor.visit_symmetry_declaration(self)


@dataclass(frozen=True, slots=True)
class Program(AstNode):
    lattice: LatticeDeclaration
    symmetry: SymmetryDeclaration
    body: tuple[AstNode, ...]

    def accept(self, visitor: AstVisitor[_T]) -> _T:
        return visitor.visit_program(self)


@dataclass(frozen=True, slots=True)
class InfixOperatorSpec:
    """Precedence-climbing metadata for a binary operator."""

    symbol: str
    precedence: int
    right_associative: bool = False


INFIX_TABLE: Final[Mapping[TokenKind, InfixOperatorSpec]] = {
    TokenKind[kind]: InfixOperatorSpec(symbol, precedence)
    for kind, symbol, precedence in FRONT_END.infix
}


# ----------------------------------------------------------------------
# The statement forms, derived from the productions
# ----------------------------------------------------------------------
#
# The productions used to describe the syntax and something else used to
# parse it, which left two descriptions free to drift apart.  The forms are
# read off the productions now, so there is one description and the grammar
# is the one that speaks.
#
# What cannot come off a production stays here: which node a nonterminal
# builds, and which enumeration a symbol names along with what to call it
# when the word is not one of its members.  Expressions are still climbed by
# the binding powers the surface image carries, not descended by the
# expression productions, because those are right recursive where the
# operators are left associative.

FORM_BUILDS: Final[Mapping[str, str]] = {
    "lattice": "LatticeDeclaration",
    "symmetry": "SymmetryDeclaration",
    "binding": "Binding",
    "stroke": "StrokeDeclaration",
    "emission": "Emission",
    "painting": "Painting",
    "iteration": "Iteration",
    "run": "Run",
}

ENUM_FORMS: Final[Mapping[str, tuple[str, str]]] = {
    "family": ("SymmetryFamily", "symmetry family"),
    "orientation": ("Orientation", "orientation"),
}

DESCENDING_STEPS: Final[Mapping[str, str]] = {"expression": "expr", "run": "run"}

TERMINAL_KINDS: Final[Mapping[str, str]] = {
    **{mark: kind.name for mark, kind in PUNCTUATION.items()},
    "..": "RANGE",
}


def _step_for(symbol: str, production: Sequence[str], position: int) -> tuple[str, ...]:
    """The step a production symbol asks the parser to take."""
    if symbol.startswith("'"):
        literal = symbol[1:-1]
        if literal in KEYWORDS:
            return ("kw", literal)
        return ("tok", TERMINAL_KINDS[literal])
    if symbol == "IDENTIFIER":
        return ("id",)
    if symbol == "INTEGER":
        return ("int",)
    if symbol in ENUM_FORMS:
        name, phrase = ENUM_FORMS[symbol]
        return ("enum", name, *phrase.split(" "))
    if symbol == "body":
        closing = production[position + 1]
        return ("body", TERMINAL_KINDS[closing[1:-1]])
    try:
        return (DESCENDING_STEPS[symbol],)
    except KeyError as exc:
        raise SyntaxError_(f"no step descends for {symbol!r}") from exc


def derive_forms() -> Mapping[str, tuple[tuple[str, ...], ...]]:
    """Each form, read off the single production of the nonterminal it builds."""
    forms: dict[str, tuple[tuple[str, ...], ...]] = {}
    for name, builds in FORM_BUILDS.items():
        alternatives = GSL_GRAMMAR.get(name, ())
        if len(alternatives) != 1:
            raise SyntaxError_(
                f"{name!r} has {len(alternatives)} production(s) and a form wants one"
            )
        production = alternatives[0]
        steps = [
            _step_for(symbol, production, position)
            for position, symbol in enumerate(production)
        ]
        steps.append(("make", builds))
        forms[name] = tuple(steps)
    return forms


def derive_statements() -> Mapping[str, str]:
    """Which keyword opens which form, from what a statement may be."""
    opening: dict[str, str] = {}
    for alternative in GSL_GRAMMAR["statement"]:
        form = alternative[0]
        if form not in FORM_BUILDS:
            raise SyntaxError_(f"the statement {form!r} builds nothing")
        opening[GSL_GRAMMAR[form][0][0][1:-1]] = form
    return opening


FORMS: Final[Mapping[str, tuple[tuple[str, ...], ...]]] = derive_forms()
STATEMENT_FORMS: Final[Mapping[str, str]] = derive_statements()

NODE_BY_NAME: Final[Mapping[str, type]] = {
    node.__name__: node
    for node in (
        LatticeDeclaration, SymmetryDeclaration, Binding, StrokeDeclaration,
        Emission, Painting, Iteration, Run,
    )
}

ENUM_BY_NAME: Final[Mapping[str, type[enum.Enum]]] = {
    "SymmetryFamily": SymmetryFamily,
    "Orientation": Orientation,
}


def grammar_disagreements() -> tuple[str, ...]:
    """Where the grammar cannot be taken at its word.

    The forms are read off the productions, so the two can no longer say
    different things.  What is left to check is that the reading had one
    answer to take: a nonterminal a form is built from must have exactly one
    production, or the derivation quietly takes the first alternative and the
    rest are never parsed.  ``derive_forms`` refuses that outright, so this
    only reports it where a report is wanted rather than an exception.
    """
    found: list[str] = []
    for name in FORM_BUILDS:
        alternatives = GSL_GRAMMAR.get(name, ())
        if len(alternatives) != 1:
            found.append(f"{name!r} has {len(alternatives)} production(s), not one")
    for alternative in GSL_GRAMMAR["statement"]:
        if alternative[0] not in FORM_BUILDS:
            found.append(f"the statement {alternative[0]!r} builds nothing")
    return tuple(found)


class RecursiveDescentParser:
    """Recursive descent for statements, precedence climbing for expressions."""

    def __init__(self, tokens: Sequence[Token], diagnostics: DiagnosticEngine | None = None) -> None:
        self._stream = TokenStream(tokens)
        self._diagnostics = diagnostics or DIAGNOSTICS

    @woven
    def parse(self) -> Program:
        with TRACER.span("parse"):
            lattice = self._parse_form("lattice")
            symmetry = self._parse_form("symmetry")
            body = self._parse_body(terminator=TokenKind.END_OF_INPUT)
            self._stream.expect(TokenKind.END_OF_INPUT)
            return Program(lattice, symmetry, body)

    def _member(self, names: Sequence[str], token: Token) -> enum.Enum:
        try:
            return ENUM_BY_NAME[names[0]](token.lexeme)
        except ValueError as exc:
            what = " ".join(names[1:])
            raise SyntaxError_(
                f"{token.position}: unknown {what} {token.lexeme!r}"
            ) from exc

    def _parse_form(self, form: str) -> AstNode:
        """Runs one form, and answers the node its last step builds."""
        taken: list[Any] = []
        position: SourcePosition | None = None
        for code, *arguments in FORMS[form]:
            token: Token | None = None
            if code == "kw":
                token = self._stream.expect(TokenKind.KEYWORD, arguments[0])
            elif code == "tok":
                token = self._stream.expect(TokenKind[arguments[0]])
            elif code == "id":
                token = self._stream.expect(TokenKind.IDENTIFIER)
                taken.append(token.lexeme)
            elif code == "int":
                token = self._stream.expect(TokenKind.INTEGER)
                taken.append(token.value)
            elif code == "enum":
                token = self._stream.expect(TokenKind.KEYWORD)
                taken.append(self._member(arguments, token))
            elif code == "expr":
                taken.append(self._parse_expression())
            elif code == "run":
                taken.append(self._parse_form("run"))
            elif code == "body":
                taken.append(self._parse_body(terminator=TokenKind[arguments[0]]))
            elif code == "make":
                return NODE_BY_NAME[arguments[0]](*taken, position)
            else:  # pragma: no cover - defensive
                raise SyntaxError_(f"{code!r} is not a step this takes")
            if position is None and token is not None:
                position = token.position
        raise SyntaxError_(f"the form {form!r} builds nothing")  # pragma: no cover

    def _parse_body(self, terminator: TokenKind) -> tuple[AstNode, ...]:
        statements: list[AstNode] = []
        while not self._stream.check(terminator):
            statements.append(self._parse_statement())
        return tuple(statements)

    def _parse_statement(self) -> AstNode:
        token = self._stream.peek()
        form = (
            STATEMENT_FORMS.get(token.lexeme)
            if token.kind is TokenKind.KEYWORD
            else None
        )
        if form is None:
            self._diagnostics.emit(
                Severity.FATAL, "SY0002", "diag.unexpected_token", token.position,
                expected="a statement keyword", found=str(token),
            )
            self._stream.synchronise(STATEMENT_ANCHORS)
            raise SyntaxError_(f"{token.position}: {token} cannot begin a statement")
        return self._parse_form(form)

    def _parse_expression(self, minimum_precedence: int = 0) -> AstNode:
        left = self._parse_prefix()
        while True:
            spec = INFIX_TABLE.get(self._stream.peek().kind)
            if spec is None or spec.precedence < minimum_precedence:
                return left
            token = self._stream.advance()
            next_precedence = spec.precedence + (0 if spec.right_associative else 1)
            right = self._parse_expression(next_precedence)
            left = BinaryOperation(spec.symbol, left, right, token.position)

    def _parse_prefix(self) -> AstNode:
        token = self._stream.peek()
        if token.kind is TokenKind.INTEGER:
            self._stream.advance()
            return IntegerLiteral(token.value, token.position)
        if token.kind is TokenKind.IDENTIFIER:
            self._stream.advance()
            return SymbolReference(token.lexeme, token.position)
        if token.kind is TokenKind.MINUS:
            self._stream.advance()
            return UnaryOperation("-", self._parse_prefix(), token.position)
        if token.kind is TokenKind.LEFT_PAREN:
            self._stream.advance()
            inner = self._parse_expression()
            self._stream.expect(TokenKind.RIGHT_PAREN)
            return inner
        self._diagnostics.emit(
            Severity.FATAL, "SY0003", "diag.unexpected_token", token.position,
            expected="an expression", found=str(token),
        )
        raise SyntaxError_(f"{token.position}: {token} cannot begin an expression")


# ======================================================================
# Layer 6: semantic analysis and type inference
# ======================================================================


class TypeTerm(abc.ABC):
    """A term in the (extremely small) type language of GSL/2."""

    __slots__ = ()

    @abc.abstractmethod
    def __str__(self) -> str: ...


@dataclass(frozen=True, slots=True)
class TypeConstructor(TypeTerm):
    name: str
    arguments: tuple[TypeTerm, ...] = ()

    def __str__(self) -> str:
        if not self.arguments:
            return self.name
        return f"{self.name}<{', '.join(map(str, self.arguments))}>"


@dataclass(frozen=True, slots=True)
class TypeVariable(TypeTerm):
    identifier: int

    def __str__(self) -> str:
        return f"τ{self.identifier}"


INT_TYPE: Final[TypeConstructor] = TypeConstructor("Int")
INTERVAL_TYPE: Final[TypeConstructor] = TypeConstructor("Interval", (INT_TYPE,))
STROKE_TYPE: Final[TypeConstructor] = TypeConstructor("Stroke")
UNIT_TYPE: Final[TypeConstructor] = TypeConstructor("Unit")


class Unifier:
    """Robinson unification over a union-find substitution."""

    def __init__(self) -> None:
        self._substitution: dict[int, TypeTerm] = {}
        self._counter = itertools.count()

    def fresh(self) -> TypeVariable:
        return TypeVariable(next(self._counter))

    def resolve(self, term: TypeTerm) -> TypeTerm:
        while isinstance(term, TypeVariable) and term.identifier in self._substitution:
            term = self._substitution[term.identifier]
        return term

    def unify(self, left: TypeTerm, right: TypeTerm) -> TypeTerm:
        left, right = self.resolve(left), self.resolve(right)
        if left == right:
            return left
        if isinstance(left, TypeVariable):
            self._substitution[left.identifier] = right
            return right
        if isinstance(right, TypeVariable):
            self._substitution[right.identifier] = left
            return left
        assert isinstance(left, TypeConstructor) and isinstance(right, TypeConstructor)
        if left.name != right.name or len(left.arguments) != len(right.arguments):
            raise TypeInferenceError(f"cannot unify {left} with {right}")
        for a, b in zip(left.arguments, right.arguments):
            self.unify(a, b)
        return left


class SymbolKind(enum.Enum):
    INTRINSIC = enum.auto()
    BINDING = enum.auto()
    STROKE = enum.auto()
    INDUCTION = enum.auto()


@dataclass(frozen=True, slots=True)
class Symbol:
    name: str
    kind: SymbolKind
    type_term: TypeTerm
    slot: int = -1
    run: "Run | None" = None


class LexicalScope:
    """A chained, insertion-ordered symbol table."""

    def __init__(self, parent: "LexicalScope | None" = None, label: str = "global") -> None:
        self._parent = parent
        self._label = label
        self._symbols: dict[str, Symbol] = {}

    def declare(self, symbol: Symbol, position: SourcePosition = ORIGIN) -> Symbol:
        if symbol.name in self._symbols:
            DIAGNOSTICS.emit(
                Severity.FATAL, "SE0001", "diag.duplicate_symbol", position, name=symbol.name
            )
            raise SemanticError(f"{position}: symbol {symbol.name!r} already declared")
        self._symbols[symbol.name] = symbol
        return symbol

    def resolve(self, name: str, position: SourcePosition = ORIGIN) -> Symbol:
        scope: LexicalScope | None = self
        while scope is not None:
            found = scope._symbols.get(name)
            if found is not None:
                return found
            scope = scope._parent
        DIAGNOSTICS.emit(Severity.FATAL, "SE0002", "diag.unknown_symbol", position, name=name)
        raise SemanticError(f"{position}: unknown symbol {name!r}")

    def child(self, label: str) -> "LexicalScope":
        return LexicalScope(self, label)

    def render(self, depth: int = 0) -> str:
        rail = "  " * depth
        lines = [f"{rail}scope {self._label}:"]
        lines.extend(
            f"{rail}  {symbol.name}: {symbol.type_term} "
            f"({symbol.kind.name.lower()}, slot={symbol.slot})"
            for symbol in self._symbols.values()
        )
        return "\n".join(lines)


INTRINSIC_NAMES: Final[tuple[str, ...]] = ("zero", "apothem", "extremum", "magnitude")


@dataclass(slots=True)
class SemanticModel:
    """The analyser's output: everything the back end needs to know."""

    order: int = DEFAULT_LATTICE_ORDER
    family: SymmetryFamily = SymmetryFamily.CYCLIC
    cardinality: int = 4
    frame_size: int = 0
    scope: LexicalScope = field(default_factory=LexicalScope)
    resolutions: dict[int, Symbol] = field(default_factory=dict)
    strokes: dict[str, Run] = field(default_factory=dict)
    types: dict[int, str] = field(default_factory=dict)

    @property
    def apothem(self) -> int:
        return self.order // 2

    @property
    def centre(self) -> Coordinate:
        return Coordinate(self.apothem, self.apothem)


class ConstantEvaluator(AstVisitor[int]):
    """Evaluates the closed subset of the expression language at compile time."""

    def __init__(self, environment: Mapping[str, int]) -> None:
        self._environment = environment

    def visit_integer_literal(self, node: IntegerLiteral) -> int:
        return node.value

    def visit_symbol_reference(self, node: SymbolReference) -> int:
        try:
            return self._environment[node.name]
        except KeyError as exc:
            raise SemanticError(
                f"{node.position}: {node.name!r} is not a compile-time constant"
            ) from exc

    def visit_unary_operation(self, node: UnaryOperation) -> int:
        return -self.visit(node.operand)

    def visit_binary_operation(self, node: BinaryOperation) -> int:
        table: Mapping[str, Callable[[int, int], int]] = {
            "+": operator.add, "-": operator.sub,
            "*": operator.mul, "/": operator.floordiv,
        }
        return table[node.operator_symbol](self.visit(node.left), self.visit(node.right))

    def _reject(self, node: AstNode) -> int:
        raise SemanticError(f"{type(node).__name__} is not a constant expression")

    visit_run = _reject
    visit_binding = _reject
    visit_stroke_declaration = _reject
    visit_emission = _reject
    visit_painting = _reject
    visit_iteration = _reject
    visit_lattice_declaration = _reject
    visit_symmetry_declaration = _reject
    visit_program = _reject


# ----------------------------------------------------------------------
# Analysis, as one image compiled at import
# ----------------------------------------------------------------------
#
# A program per kind of node again, though the opcodes here earn their keep
# less evenly than the ones lowering uses: half are read by several nodes and
# half by exactly one, because what a declaration has to check is particular
# to that declaration.  The sequence is the part that is data.

ANALYSIS_IMAGE: Final[str] = (
    "Program visit lattice symmetry intrinsics each body answer UNIT "
    "LatticeDeclaration order SymmetryDeclaration cardinality Binding fresh "
    "value bind BINDING name StrokeDeclaration stroke run strokes Emission "
    "use STROKE Painting Iteration int lower upper push for- variable "
    "INDUCTION Run index IntegerLiteral INT SymbolReference deny answer-of "
    "UnaryOperation operand BinaryOperation left right~0:1.2|1.3|4|5.6|7.8,9:"
    "a|7.8,b:c|7.8,d:e.f|g.h.i|7.8,j:k.l|m|7.8,n:o.p|7.8,q:k.l|7.8,r:s.t|s.u|"
    "v.w.x|g.y.x|5.6|7.8,z:s.10|s.t|s.u|7.p,11:7.12,13:14.p|15,16:s.17|7.12,1"
    "8:s.19|s.1a|7.12"
)


def compile_analysis(image: str) -> Mapping[str, tuple[tuple[str, ...], ...]]:
    """Reads an image back into the programs layer 6 runs."""
    try:
        vocabulary, programs = image.split("~")
    except ValueError as exc:
        raise SemanticError("the image does not have two sections") from exc
    words = vocabulary.split(" ")

    def at(code: str) -> str:
        try:
            return words[int(code, 36)]
        except (ValueError, IndexError) as exc:
            raise SemanticError(f"{code!r} addresses no word") from exc

    return {
        at(head): tuple(
            tuple(at(t) for t in step.split(".")) for step in body.split("|")
        )
        for head, _, body in (r.partition(":") for r in programs.split(","))
    }


ANALYSIS: Final[Mapping[str, tuple[tuple[str, ...], ...]]] = compile_analysis(ANALYSIS_IMAGE)

TYPE_BY_NAME: Final[Mapping[str, TypeTerm]] = {
    "INT": INT_TYPE, "STROKE": STROKE_TYPE, "UNIT": UNIT_TYPE,
}


class SemanticAnalyzer(AstVisitor[TypeTerm]):
    """Resolves names, allocates frame slots and infers (all four) types."""

    def __init__(self, diagnostics: DiagnosticEngine | None = None) -> None:
        self._diagnostics = diagnostics or DIAGNOSTICS
        self._unifier = Unifier()
        self._model = SemanticModel()
        self._scope = self._model.scope
        self._next_slot = 0

    @woven
    def analyze(self, program: Program) -> SemanticModel:
        with TRACER.span("semantic-analysis"):
            self.visit(program)
            self._model.frame_size = self._next_slot
            return self._model

    def _allocate(self) -> int:
        slot = self._next_slot
        self._next_slot += 1
        return slot

    def _annotate(self, node: AstNode, term: TypeTerm) -> TypeTerm:
        self._model.types[id(node)] = str(self._unifier.resolve(term))
        return term

    def _dispatch(self, node: AstNode) -> TypeTerm:
        resolved: Symbol | None = None
        outer: list[LexicalScope] = []
        try:
            for code, *arguments in ANALYSIS.get(type(node).__name__, ()):
                if code == "visit":
                    self.visit(getattr(node, arguments[0]))
                elif code == "int":
                    self._unifier.unify(self.visit(getattr(node, arguments[0])), INT_TYPE)
                elif code == "stroke":
                    self._unifier.unify(self.visit(getattr(node, arguments[0])), STROKE_TYPE)
                elif code == "each":
                    for statement in getattr(node, arguments[0]):
                        self.visit(statement)
                elif code == "push":
                    outer.append(self._scope)
                    self._scope = self._scope.child(
                        arguments[0] + getattr(node, arguments[1])
                    )
                elif code == "fresh":
                    value_type = self.visit(getattr(node, arguments[0]))
                    variable = self._unifier.fresh()
                    self._unifier.unify(variable, value_type)
                    self._unifier.unify(variable, INT_TYPE)
                elif code == "bind":
                    symbol = Symbol(
                        getattr(node, arguments[1]),
                        SymbolKind[arguments[0]],
                        INT_TYPE,
                        self._allocate(),
                    )
                    self._scope.declare(symbol, node.position)
                    self._model.resolutions[id(node)] = symbol
                elif code == "intrinsics":
                    for name in INTRINSIC_NAMES:
                        self._scope.declare(Symbol(name, SymbolKind.INTRINSIC, INT_TYPE))
                elif code == "order":
                    order = ConstantEvaluator({}).visit(node.order)
                    if order <= 0 or order % 2 == 0:
                        self._diagnostics.emit(
                            Severity.FATAL, "SE0003", "diag.bad_order",
                            node.position, order=order,
                        )
                        raise SemanticError(
                            f"{node.position}: illegal lattice order {order}"
                        )
                    self._model.order = order
                elif code == "cardinality":
                    if node.cardinality != 4:
                        raise SemanticError(
                            f"{node.position}: this platform only implements 4-fold "
                            f"symmetry, got {node.cardinality}"
                        )
                    self._model.family = node.family
                    self._model.cardinality = node.cardinality
                elif code == "strokes":
                    self._scope.declare(
                        Symbol(node.name, SymbolKind.STROKE, STROKE_TYPE, run=node.run),
                        node.position,
                    )
                    self._model.strokes[node.name] = node.run
                elif code == "use":
                    resolved = self._scope.resolve(node.name, node.position)
                    if resolved.kind is not SymbolKind[arguments[0]]:
                        raise SemanticError(
                            f"{node.position}: {node.name!r} is not a stroke"
                        )
                    self._model.resolutions[id(node)] = resolved
                elif code == "deny":
                    resolved = self._scope.resolve(node.name, node.position)
                    if resolved.kind is SymbolKind[arguments[0]]:
                        raise SemanticError(
                            f"{node.position}: strokes are not first-class values"
                        )
                    self._model.resolutions[id(node)] = resolved
                elif code == "answer":
                    return self._annotate(node, TYPE_BY_NAME[arguments[0]])
                elif code == "answer-of":
                    return self._annotate(node, typing.cast(Symbol, resolved).type_term)
                else:  # pragma: no cover - defensive
                    raise SemanticError(f"{code!r} is not an opcode this runs")
        finally:
            if outer:
                self._scope = outer[0]
        raise SemanticError(  # pragma: no cover - defensive
            f"the program for {type(node).__name__} answers nothing"
        )

    visit_program = _dispatch
    visit_lattice_declaration = _dispatch
    visit_symmetry_declaration = _dispatch
    visit_binding = _dispatch
    visit_stroke_declaration = _dispatch
    visit_emission = _dispatch
    visit_painting = _dispatch
    visit_iteration = _dispatch
    visit_run = _dispatch
    visit_integer_literal = _dispatch
    visit_symbol_reference = _dispatch
    visit_unary_operation = _dispatch
    visit_binary_operation = _dispatch


# ======================================================================
# Layer 7: intermediate representation and lowering
# ======================================================================


class OperandKind(enum.Enum):
    NONE = "none"
    CONSTANT = "const"
    NAME = "name"
    SLOT = "slot"
    TARGET = "target"


class Instruction(abc.ABC, metaclass=PluginRegistryMeta):
    """A GVM instruction; concrete subclasses self-register by opcode."""

    __registry_root__ = True
    __slots__ = ()

    opcode: ClassVar[int] = -1
    operand_kind: ClassVar[OperandKind] = OperandKind.NONE
    stack_delta: ClassVar[int] = 0

    @property
    def operand(self) -> Any:
        return None

    def mnemonic(self) -> str:
        return _kebab(type(self).__name__).replace("-", ".")

    def __str__(self) -> str:
        operand = self.operand
        return self.mnemonic() if operand is None else f"{self.mnemonic()} {operand!r}"


_OPCODES: dict[int, type[Instruction]] = {}


def opcode(number: int, kind: OperandKind = OperandKind.NONE, delta: int = 0):
    """Class decorator assigning and registering a numeric opcode."""

    def decorator(cls: type[Instruction]) -> type[Instruction]:
        if number in _OPCODES:  # pragma: no cover - defensive
            raise CodeGenerationError(f"opcode {number} is already taken by {_OPCODES[number]}")
        cls.opcode = number
        cls.operand_kind = kind
        cls.stack_delta = delta
        _OPCODES[number] = cls
        return cls

    return decorator


@opcode(0x01, OperandKind.CONSTANT, delta=+1)
@dataclass(frozen=True, slots=True)
class PushConstant(Instruction):
    value: int

    @property
    def operand(self) -> Any:
        return self.value


@opcode(0x02, OperandKind.NAME, delta=+1)
@dataclass(frozen=True, slots=True)
class LoadIntrinsic(Instruction):
    name: str

    @property
    def operand(self) -> Any:
        return self.name


@opcode(0x03, OperandKind.SLOT, delta=+1)
@dataclass(frozen=True, slots=True)
class LoadLocal(Instruction):
    slot: int

    @property
    def operand(self) -> Any:
        return self.slot


@opcode(0x04, OperandKind.SLOT, delta=-1)
@dataclass(frozen=True, slots=True)
class StoreLocal(Instruction):
    slot: int

    @property
    def operand(self) -> Any:
        return self.slot


@opcode(0x10, delta=-1)
@dataclass(frozen=True, slots=True)
class BinaryAdd(Instruction):
    pass


@opcode(0x11, delta=-1)
@dataclass(frozen=True, slots=True)
class BinarySubtract(Instruction):
    pass


@opcode(0x12, delta=-1)
@dataclass(frozen=True, slots=True)
class BinaryMultiply(Instruction):
    pass


@opcode(0x13, delta=-1)
@dataclass(frozen=True, slots=True)
class BinaryDivide(Instruction):
    pass


@opcode(0x14)
@dataclass(frozen=True, slots=True)
class Negate(Instruction):
    pass


@opcode(0x15, delta=-1)
@dataclass(frozen=True, slots=True)
class CompareLessEqual(Instruction):
    pass


@opcode(0x20, delta=-1)
@dataclass(frozen=True, slots=True)
class MakeInterval(Instruction):
    pass


@opcode(0x21, OperandKind.NAME, delta=-2)
@dataclass(frozen=True, slots=True)
class EmitOrientedRun(Instruction):
    orientation: Orientation

    @property
    def operand(self) -> Any:
        return self.orientation.value


@opcode(0x30, OperandKind.TARGET)
@dataclass(frozen=True, slots=True)
class Jump(Instruction):
    target: Any

    @property
    def operand(self) -> Any:
        return self.target


@opcode(0x31, OperandKind.TARGET, delta=-1)
@dataclass(frozen=True, slots=True)
class JumpIfFalse(Instruction):
    target: Any

    @property
    def operand(self) -> Any:
        return self.target


@opcode(0x40)
@dataclass(frozen=True, slots=True)
class CloseUnderGroup(Instruction):
    pass


@opcode(0xFF)
@dataclass(frozen=True, slots=True)
class Halt(Instruction):
    pass


@dataclass(frozen=True, slots=True)
class Label:
    """A symbolic branch target; erased by the assembler."""

    name: str

    def __str__(self) -> str:
        return f"{self.name}:"


IrItem = typing.Union[Instruction, Label]


# ----------------------------------------------------------------------
# Lowering, as one image compiled at import
# ----------------------------------------------------------------------
#
# A program per kind of node, in the same vocabulary-and-offsets form the
# surface of the language is written in.  An opcode either walks into the
# tree, emits an instruction, or binds and places a label; what each kind of
# node lowers to is the run of opcodes listed against it and nothing else.

LOWERING_IMAGE: Final[str] = (
    "+ BinaryAdd - BinarySubtract * BinaryMultiply / BinaryDivide Program "
    "each body emit CloseUnderGroup Halt LatticeDeclaration "
    "SymmetryDeclaration Binding visit value sym StoreLocal slot "
    "StrokeDeclaration Emission lowered Painting run Run index lower upper "
    "MakeInterval node EmitOrientedRun orientation Iteration mark head "
    "loop.head. exit loop.exit. place LoadLocal CompareLessEqual goto "
    "JumpIfFalse lit PushConstant 1 Jump IntegerLiteral SymbolReference "
    "bykind LoadIntrinsic name UnaryOperation operand Negate BinaryOperation "
    "left right pick operator_symbol~0.1,2.3,4.5,6.7~8:9.a|b.c|b.d,e:,f:,g:h."
    "i|j.k.l,m:,n:o,p:h.q,r:h.s|h.t|h.u|b.v|w.x.y,z:10.11.12|10.13.14|h.t|j.k"
    ".l|15.11|j.16.l|h.u|b.17|18.19.13|9.a|j.16.l|1a.1b.1c|b.1|j.k.l|18.1d.11"
    "|15.13,1e:w.1b.i,1f:1g.1h.1i.16.l,1j:h.1k|b.1l,1m:h.1n|h.1o|1p.1q"
)


@dataclass(frozen=True, slots=True)
class LoweringImage:
    """What the image stands for: an operator table and a program per node."""

    operators: Mapping[str, str]
    programs: Mapping[str, tuple[tuple[str, ...], ...]]


def compile_lowering(image: str) -> LoweringImage:
    """Reads an image back into the programs layer 7 runs."""
    try:
        vocabulary, operators, programs = image.split("~")
    except ValueError as exc:
        raise CodeGenerationError("the image does not have three sections") from exc
    words = vocabulary.split(" ")

    def at(code: str) -> str:
        try:
            return words[int(code, 36)]
        except (ValueError, IndexError) as exc:
            raise CodeGenerationError(f"{code!r} addresses no word") from exc

    return LoweringImage(
        {at(a): at(b) for a, b in (r.split(".") for r in operators.split(","))},
        {
            at(head): (
                tuple(tuple(at(t) for t in op.split(".")) for op in body.split("|"))
                if body
                else ()
            )
            for head, _, body in (r.partition(":") for r in programs.split(","))
        },
    )


LOWERING: Final[LoweringImage] = compile_lowering(LOWERING_IMAGE)

INSTRUCTION_BY_NAME: Final[Mapping[str, type[Instruction]]] = {
    instruction.__name__: instruction for instruction in _OPCODES.values()
}


class BytecodeEmitter(AstVisitor[None]):
    """Lowers the typed tree by running the program the image holds for a node.

    Every one of the visitor's thirteen methods is the same method.  Which
    program runs is settled by the node the double dispatch arrives at, and
    that program is read out of the image rather than written here.
    """

    def __init__(self, model: SemanticModel) -> None:
        self._model = model
        self._items: list[IrItem] = []
        self._labels = itertools.count()

    @woven
    def lower(self, program: Program) -> tuple[IrItem, ...]:
        with TRACER.span("lowering"):
            self.visit(program)
            return tuple(self._items)

    def _emit(self, item: IrItem) -> None:
        self._items.append(item)

    def _label(self, stem: str) -> Label:
        return Label(f".{stem}{next(self._labels)}")

    @staticmethod
    def _instruction(name: str) -> type[Instruction]:
        try:
            return INSTRUCTION_BY_NAME[name]
        except KeyError as exc:
            raise CodeGenerationError(f"{name!r} names no instruction") from exc

    def _resolve(self, node: AstNode) -> Symbol:
        symbol = self._model.resolutions.get(id(node))
        if symbol is None:  # pragma: no cover - defensive
            raise CodeGenerationError(f"{type(node).__name__} was never resolved")
        return symbol

    def _dispatch(self, node: AstNode) -> None:
        marks: dict[str, Label] = {}
        for code, *arguments in LOWERING.programs.get(type(node).__name__, ()):
            if code == "visit":
                self.visit(getattr(node, arguments[0]))
            elif code == "each":
                for item in getattr(node, arguments[0]):
                    self.visit(item)
            elif code == "emit":
                self._emit(self._instruction(arguments[0])())
            elif code == "node":
                self._emit(self._instruction(arguments[0])(getattr(node, arguments[1])))
            elif code == "sym":
                symbol = self._resolve(node)
                self._emit(self._instruction(arguments[0])(getattr(symbol, arguments[1])))
            elif code == "lit":
                self._emit(self._instruction(arguments[0])(int(arguments[1])))
            elif code == "mark":
                marks[arguments[0]] = self._label(arguments[1])
            elif code == "place":
                self._emit(marks[arguments[0]])
            elif code == "goto":
                self._emit(self._instruction(arguments[0])(marks[arguments[1]].name))
            elif code == "pick":
                chosen = LOWERING.operators[getattr(node, arguments[0])]
                self._emit(self._instruction(chosen)())
            elif code == "bykind":
                symbol = self._resolve(node)
                head, attribute = (
                    arguments[:2] if symbol.kind is SymbolKind.INTRINSIC else arguments[2:]
                )
                self._emit(self._instruction(head)(getattr(symbol, attribute)))
            elif code == "lowered":
                resolved = self._model.resolutions.get(id(node))
                run = (
                    resolved.run if resolved is not None
                    else self._model.strokes.get(node.name)
                )
                if run is None:  # pragma: no cover - defensive
                    raise CodeGenerationError(
                        f"stroke {node.name!r} has no lowered definition"
                    )
                self.visit(run)
            else:  # pragma: no cover - defensive
                raise CodeGenerationError(f"{code!r} is not an opcode this runs")

    visit_program = _dispatch
    visit_lattice_declaration = _dispatch
    visit_symmetry_declaration = _dispatch
    visit_binding = _dispatch
    visit_stroke_declaration = _dispatch
    visit_emission = _dispatch
    visit_painting = _dispatch
    visit_run = _dispatch
    visit_iteration = _dispatch
    visit_integer_literal = _dispatch
    visit_symbol_reference = _dispatch
    visit_unary_operation = _dispatch
    visit_binary_operation = _dispatch


# ======================================================================
# Layer 8: control flow analysis and optimisation
# ======================================================================


@dataclass(slots=True)
class BasicBlock:
    """A maximal straight-line run of instructions with a single entry."""

    index: int
    label: str | None
    instructions: list[Instruction] = field(default_factory=list)
    successors: list[str | int] = field(default_factory=list)

    @property
    def terminator(self) -> Instruction | None:
        return self.instructions[-1] if self.instructions else None


class ControlFlowGraph:
    """Partitions a symbolic listing into basic blocks and computes reachability."""

    def __init__(self, items: Sequence[IrItem]) -> None:
        self.blocks: list[BasicBlock] = []
        self._by_label: dict[str, int] = {}
        self._build(items)
        self._link()

    def _build(self, items: Sequence[IrItem]) -> None:
        current = BasicBlock(0, None)
        self.blocks.append(current)
        for item in items:
            if isinstance(item, Label):
                if current.instructions or current.label is not None:
                    current = BasicBlock(len(self.blocks), item.name)
                    self.blocks.append(current)
                else:
                    current.label = item.name
                self._by_label[item.name] = current.index
                continue
            current.instructions.append(item)
            if isinstance(item, (Jump, JumpIfFalse, Halt)):
                current = BasicBlock(len(self.blocks), None)
                self.blocks.append(current)
        if len(self.blocks) > 1 and not current.instructions and current.label is None:
            self.blocks.pop()

    def _link(self) -> None:
        for block in self.blocks:
            terminator = block.terminator
            if isinstance(terminator, Jump):
                block.successors = [terminator.target]
            elif isinstance(terminator, Halt):
                block.successors = []
            else:
                successors: list[str | int] = []
                if isinstance(terminator, JumpIfFalse):
                    successors.append(terminator.target)
                if block.index + 1 < len(self.blocks):
                    successors.append(block.index + 1)
                block.successors = successors

    def reachable(self) -> frozenset[int]:
        seen: set[int] = set()
        frontier = [0]
        while frontier:
            index = frontier.pop()
            if index in seen or index >= len(self.blocks):
                continue
            seen.add(index)
            for successor in self.blocks[index].successors:
                target = self._by_label.get(successor) if isinstance(successor, str) else successor
                if target is not None:
                    frontier.append(target)
        return frozenset(seen)

    def render(self) -> str:
        live = self.reachable()
        lines = []
        for block in self.blocks:
            marker = "" if block.index in live else "  ; UNREACHABLE"
            lines.append(f"block {block.index} ({block.label or 'fallthrough'}){marker}")
            lines.extend(f"    {instruction}" for instruction in block.instructions)
            lines.append(f"    -> {block.successors or 'exit'}")
        return "\n".join(lines)


class OptimizationPass(abc.ABC, metaclass=PluginRegistryMeta):
    """A semantics-preserving rewrite of the symbolic listing."""

    __registry_root__ = True

    @property
    def name(self) -> str:
        return _kebab(type(self).__name__)

    @abc.abstractmethod
    def apply(self, items: list[IrItem]) -> list[IrItem]: ...


# ----------------------------------------------------------------------
# The peephole, as one image compiled at import
# ----------------------------------------------------------------------
#
# A rule is a window of matchers, how far to advance when it fires, a guard,
# and what to leave in the window's place.  Which rules answer to which name
# is the whole of what separates one of the three passes below from another:
# each is a docstring and a line saying that its name is where to look.

PEEPHOLE_IMAGE: Final[str] = (
    "BinaryAdd add BinarySubtract sub BinaryMultiply mul BinaryDivide "
    "floordiv 0 1 constant-folding k f 3 div fold n 2 - neg "
    "algebraic-identities * neutral redundant-branch-elimination j l same~0.1"
    ",2.3,4.5,6.7~8.0,8.2,9.4,9.6~a:b.b.c>d>e>f|b.g>h>i>j,k:b.l>h>m>i,n:o.p>9"
    ">q>i"
)


@dataclass(frozen=True, slots=True)
class PeepholeImage:
    """What the image stands for: two tables and the rules of each pass."""

    foldable: Mapping[str, str]
    neutral: tuple[tuple[int, str], ...]
    rules: Mapping[str, tuple[tuple[tuple[str, ...], int, str, str], ...]]


def compile_peephole(image: str) -> PeepholeImage:
    """Reads an image back into the rules layer 8 rewrites under."""
    try:
        vocabulary, foldable, neutral, rules = image.split("~")
    except ValueError as exc:
        raise CodeGenerationError("the image does not have four sections") from exc
    words = vocabulary.split(" ")

    def at(code: str) -> str:
        try:
            return words[int(code, 36)]
        except (ValueError, IndexError) as exc:
            raise CodeGenerationError(f"{code!r} addresses no word") from exc

    return PeepholeImage(
        {at(a): at(b) for a, b in (r.split(".") for r in foldable.split(","))},
        tuple((int(at(a)), at(b)) for a, b in (r.split(".") for r in neutral.split(","))),
        {
            at(head): tuple(
                (tuple(at(m) for m in f.split(".")), int(at(c)), at(g), at(e))
                for f, c, g, e in (rule.split(">") for rule in body.split("|"))
            )
            for head, _, body in (r.partition(":") for r in rules.split(","))
        },
    )


PEEPHOLE: Final[PeepholeImage] = compile_peephole(PEEPHOLE_IMAGE)

FOLDABLE: Final[Mapping[type, Callable[[int, int], int]]] = {
    INSTRUCTION_BY_NAME[name]: getattr(operator, function)
    for name, function in PEEPHOLE.foldable.items()
}

NEUTRAL: Final[tuple[tuple[int, type], ...]] = tuple(
    (value, INSTRUCTION_BY_NAME[name]) for value, name in PEEPHOLE.neutral
)

MATCHERS: Final[Mapping[str, Callable[[Any], bool]]] = {
    "k": lambda item: isinstance(item, PushConstant),
    "f": lambda item: type(item) in FOLDABLE,
    "n": lambda item: isinstance(item, Negate),
    "j": lambda item: isinstance(item, Jump),
    "l": lambda item: isinstance(item, Label),
    "*": lambda item: True,
}

GUARDS: Final[Mapping[str, Callable[[Sequence[Any]], bool]]] = {
    "-": lambda window: True,
    "div": lambda window: not (
        FOLDABLE[type(window[2])] is operator.floordiv and window[1].value == 0
    ),
    "neutral": lambda window: any(
        window[0].value == value and isinstance(window[1], operation)
        for value, operation in NEUTRAL
    ),
    "same": lambda window: window[0].target == window[1].name,
}

EMISSIONS: Final[Mapping[str, Callable[[Sequence[Any]], tuple[IrItem, ...]]]] = {
    "-": lambda window: (),
    "fold": lambda window: (
        PushConstant(FOLDABLE[type(window[2])](window[0].value, window[1].value)),
    ),
    "neg": lambda window: (PushConstant(-window[0].value),),
}


def run_rules(name: str, items: list[IrItem]) -> list[IrItem]:
    """Rewrites a listing under whichever rules the image lists against a name."""
    rules = PEEPHOLE.rules.get(name, ())
    output: list[IrItem] = []
    index = 0
    while index < len(items):
        for matchers, consume, guard, emission in rules:
            window = items[index : index + len(matchers)]
            if len(window) != len(matchers):
                continue
            if not all(MATCHERS[code](item) for code, item in zip(matchers, window)):
                continue
            if not GUARDS[guard](window):
                continue
            output.extend(EMISSIONS[emission](window))
            index += consume
            break
        else:
            output.append(items[index])
            index += 1
    return output


class ConstantFolding(OptimizationPass):
    """Evaluates closed integer subexpressions at compile time."""

    def apply(self, items: list[IrItem]) -> list[IrItem]:
        return run_rules(self.name, items)


class AlgebraicIdentities(OptimizationPass):
    """Deletes operations that are neutral elements of their operator."""

    def apply(self, items: list[IrItem]) -> list[IrItem]:
        return run_rules(self.name, items)


class RedundantBranchElimination(OptimizationPass):
    """Removes jumps whose target is the immediately following label."""

    def apply(self, items: list[IrItem]) -> list[IrItem]:
        return run_rules(self.name, items)


class UnreachableCodeElimination(OptimizationPass):
    """Drops instructions that no control flow path can reach."""

    def apply(self, items: list[IrItem]) -> list[IrItem]:
        output: list[IrItem] = []
        live = True
        for item in items:
            if isinstance(item, Label):
                live = True
                output.append(item)
                continue
            if live:
                output.append(item)
                if isinstance(item, (Jump, Halt)):
                    live = False
        return output


class DeadLabelElimination(OptimizationPass):
    """Erases labels that no branch instruction references."""

    def apply(self, items: list[IrItem]) -> list[IrItem]:
        referenced = {
            item.target
            for item in items
            if isinstance(item, (Jump, JumpIfFalse)) and isinstance(item.target, str)
        }
        return [
            item
            for item in items
            if not (isinstance(item, Label) and item.name not in referenced)
        ]


@dataclass(frozen=True, slots=True)
class PassStatistics:
    pass_name: str
    rounds: int
    removed: int


class PassManager:
    """Runs the pass pipeline to a fixed point, or until patience runs out."""

    _PATIENCE: Final[int] = 8

    def __init__(self, passes: Sequence[OptimizationPass] | None = None) -> None:
        self._passes = tuple(passes) if passes is not None else tuple(
            cls() for cls in (
                ConstantFolding, AlgebraicIdentities, RedundantBranchElimination,
                UnreachableCodeElimination, DeadLabelElimination,
            )
        )
        self.statistics: list[PassStatistics] = []

    @woven
    def optimise(self, items: Sequence[IrItem]) -> tuple[IrItem, ...]:
        with TRACER.span("optimise", passes=len(self._passes)):
            working = list(items)
            tally: MutableMapping[str, tuple[int, int]] = collections.defaultdict(lambda: (0, 0))
            for round_number in range(self._PATIENCE):
                before = len(working)
                for optimisation in self._passes:
                    previous = len(working)
                    working = optimisation.apply(working)
                    rounds, removed = tally[optimisation.name]
                    if len(working) != previous:
                        tally[optimisation.name] = (rounds + 1, removed + previous - len(working))
                if len(working) == before:
                    break
            self.statistics = [
                PassStatistics(name, rounds, removed)
                for name, (rounds, removed) in sorted(tally.items())
            ]
            METRICS.increment("optimiser.instructions_removed", len(items) - len(working))
            return tuple(working)

    def render(self) -> str:
        if not self.statistics:
            return "(the optimiser found nothing to do)"
        return "\n".join(
            f"{entry.pass_name}: fired in {entry.rounds} round(s), "
            f"{entry.removed} item(s) removed"
            for entry in self.statistics
        )


# ======================================================================
# Layer 9: object module and container format
# ======================================================================


@dataclass(frozen=True, slots=True)
class ObjectModule:
    """A linked, label-free instruction listing plus its execution metadata."""

    order: int
    family: SymmetryFamily
    cardinality: int
    frame_size: int
    instructions: tuple[Instruction, ...]

    @property
    def symmetry_order(self) -> int:
        return self.family.expected_order(self.cardinality)

    def disassembly(self) -> str:
        lines = [
            f"; module order={self.order} symmetry={self.family.value}{self.cardinality} "
            f"frame={self.frame_size} length={len(self.instructions)}"
        ]
        depth = 0
        for address, instruction in enumerate(self.instructions):
            depth += instruction.stack_delta
            lines.append(f"{address:04d}  {depth:+3d}  {instruction}")
        return "\n".join(lines)


class Assembler:
    """Resolves symbolic labels into absolute addresses."""

    @woven
    def assemble(self, items: Sequence[IrItem], model: SemanticModel) -> ObjectModule:
        addresses: dict[str, int] = {}
        address = 0
        for item in items:
            if isinstance(item, Label):
                addresses[item.name] = address
            else:
                address += 1
        instructions: list[Instruction] = []
        for item in items:
            if isinstance(item, Label):
                continue
            if isinstance(item, (Jump, JumpIfFalse)) and isinstance(item.target, str):
                try:
                    resolved = addresses[item.target]
                except KeyError as exc:
                    raise CodeGenerationError(f"unresolved branch target {item.target!r}") from exc
                instructions.append(type(item)(resolved))
            else:
                instructions.append(item)
        if not instructions or not isinstance(instructions[-1], Halt):
            raise CodeGenerationError("object module does not terminate in a halt")
        return ObjectModule(
            model.order, model.family, model.cardinality, model.frame_size, tuple(instructions)
        )


def _packed(layout: str, value: int, field: str) -> bytes:
    """``struct.pack``, but a value the field cannot hold is a diagnostic.

    The container's integers are fixed width, so a program can be perfectly
    well typed, lower cleanly and still not fit in the object format.  Naming
    the field it overflowed is more use than a ``struct.error`` raised four
    frames further down.
    """
    try:
        return struct.pack(layout, value)
    except struct.error as exc:
        raise ObjectFormatError(
            f"{field} {value} does not fit the container's {layout} field"
        ) from exc


# ----------------------------------------------------------------------
# The container's header, as one image compiled at import
# ----------------------------------------------------------------------
#
# The layout was stated twice: once as the fields the encoder wrote in order
# and once as the format the decoder unpacked them with.  It is stated here
# instead, and both directions are read off it - the format string is the
# codes joined, and the names are what the decoder unpacks into.

CONTAINER_IMAGE: Final[str] = (
    "4s magic B version container H order lattice family symmetry cardinality"
    " frame size~0.1.1,2.3.4.3,5.6.7.6,2.8.9.8,2.a.9.a,5.b.b.c"
)


def compile_container(image: str) -> tuple[tuple[str, str, str], ...]:
    """Reads an image back into the fields the container's header carries."""
    try:
        vocabulary, fields = image.split("~")
    except ValueError as exc:
        raise ObjectFormatError("the image does not have two sections") from exc
    words = vocabulary.split(" ")

    def at(code: str) -> str:
        try:
            return words[int(code, 36)]
        except (ValueError, IndexError) as exc:
            raise ObjectFormatError(f"{code!r} addresses no word") from exc

    return tuple(
        (at(parts[0]), at(parts[1]), " ".join(at(part) for part in parts[2:]))
        for parts in (record.split(".") for record in fields.split(","))
    )


CONTAINER_HEADER: Final[tuple[tuple[str, str, str], ...]] = compile_container(CONTAINER_IMAGE)

HEADER_SOURCES: Final[Mapping[str, Callable[["ObjectModule"], Any]]] = {
    "magic": lambda module: OBJECT_MAGIC,
    "version": lambda module: OBJECT_FORMAT_VERSION,
    "order": lambda module: module.order,
    "family": lambda module: list(SymmetryFamily).index(module.family),
    "cardinality": lambda module: module.cardinality,
    "frame": lambda module: module.frame_size,
}


class ObjectCodec:
    """Serialises and reloads modules through a checksummed binary container.

    Layout::

        magic(4) version(1) order(H) family(1) cardinality(1) frame(H)
        constants(H) [i4 ...] names(H) [len(1) utf8 ...] code(I) [op(1) arg(i4) ...]
        crc32(I)
    """

    _HEADER = struct.Struct("<" + "".join(code for code, _, _ in CONTAINER_HEADER))

    @woven
    def encode(self, module: ObjectModule) -> bytes:
        constants: list[int] = []
        names: list[str] = []

        def intern(pool: list[Any], value: Any) -> int:
            if value not in pool:
                pool.append(value)
            return pool.index(value)

        encoded: list[tuple[int, int]] = []
        for instruction in module.instructions:
            kind = instruction.operand_kind
            if kind is OperandKind.CONSTANT:
                argument = intern(constants, instruction.operand)
            elif kind is OperandKind.NAME:
                argument = intern(names, instruction.operand)
            elif kind in (OperandKind.SLOT, OperandKind.TARGET):
                argument = int(instruction.operand)
            else:
                argument = 0
            encoded.append((instruction.opcode, argument))

        buffer = io.BytesIO()
        for code, source, label in CONTAINER_HEADER:
            buffer.write(_packed("<" + code, HEADER_SOURCES[source](module), label))
        buffer.write(_packed("<H", len(constants), "constant pool size"))
        for value in constants:
            buffer.write(_packed("<i", value, "constant"))
        buffer.write(_packed("<H", len(names), "name pool size"))
        for name in names:
            payload = name.encode("utf-8")
            buffer.write(_packed("<B", len(payload), f"length of name {name!r}"))
            buffer.write(payload)
        buffer.write(_packed("<I", len(encoded), "instruction count"))
        for op, argument in encoded:
            buffer.write(_packed("<B", op, "opcode") + _packed("<i", argument, "operand"))
        body = buffer.getvalue()
        return body + struct.pack("<I", binascii.crc32(body) & 0xFFFFFFFF)

    @woven
    def decode(self, blob: bytes) -> ObjectModule:
        if len(blob) < self._HEADER.size + 4:
            raise ObjectFormatError("object blob is too short to contain a header")
        body, checksum = blob[:-4], struct.unpack("<I", blob[-4:])[0]
        if binascii.crc32(body) & 0xFFFFFFFF != checksum:
            raise ObjectFormatError("object blob failed its CRC-32 integrity check")
        header = dict(
            zip(
                (source for _, source, _ in CONTAINER_HEADER),
                self._HEADER.unpack(body[: self._HEADER.size]),
            )
        )
        magic, version = header["magic"], header["version"]
        order, family_index = header["order"], header["family"]
        cardinality, frame_size = header["cardinality"], header["frame"]
        if magic != OBJECT_MAGIC:
            raise ObjectFormatError(f"bad magic {magic!r}, expected {OBJECT_MAGIC!r}")
        if version != OBJECT_FORMAT_VERSION:
            raise ObjectFormatError(f"unsupported container version {version}")
        cursor = self._HEADER.size
        constant_count = struct.unpack_from("<H", body, cursor)[0]
        cursor += 2
        constants = list(struct.unpack_from(f"<{constant_count}i", body, cursor))
        cursor += 4 * constant_count
        name_count = struct.unpack_from("<H", body, cursor)[0]
        cursor += 2
        names: list[str] = []
        for _ in range(name_count):
            length = struct.unpack_from("<B", body, cursor)[0]
            cursor += 1
            names.append(body[cursor : cursor + length].decode("utf-8"))
            cursor += length
        code_length = struct.unpack_from("<I", body, cursor)[0]
        cursor += 4
        instructions: list[Instruction] = []
        for _ in range(code_length):
            op, argument = struct.unpack_from("<Bi", body, cursor)
            cursor += 5
            cls = _OPCODES.get(op)
            if cls is None:
                raise ObjectFormatError(f"unknown opcode 0x{op:02X}")
            kind = cls.operand_kind
            if kind is OperandKind.NONE:
                instructions.append(cls())  # type: ignore[call-arg]
            elif kind is OperandKind.CONSTANT:
                instructions.append(cls(constants[argument]))  # type: ignore[call-arg]
            elif kind is OperandKind.NAME:
                raw = names[argument]
                value = Orientation(raw) if cls is EmitOrientedRun else raw
                instructions.append(cls(value))  # type: ignore[call-arg]
            else:
                instructions.append(cls(argument))  # type: ignore[call-arg]
        return ObjectModule(
            order, list(SymmetryFamily)[family_index], cardinality, frame_size, tuple(instructions)
        )


# ======================================================================
# Layer 11: the event-sourced canvas aggregate
# ======================================================================


@dataclass(frozen=True, slots=True)
class CanvasCommand(abc.ABC):
    """The write side of the canvas CQRS split."""


@dataclass(frozen=True, slots=True)
class PaintCells(CanvasCommand):
    cells: FrozenSet[Coordinate]
    provenance: str = "paint"


@dataclass(frozen=True, slots=True)
class ApplyGroupClosure(CanvasCommand):
    group: SymmetryGroup


@dataclass(frozen=True, slots=True)
class CanvasEvent(abc.ABC):
    """An immutable fact recorded in the canvas event stream."""

    sequence: int


@dataclass(frozen=True, slots=True)
class CellsPainted(CanvasEvent):
    cells: FrozenSet[Coordinate]
    provenance: str


@dataclass(frozen=True, slots=True)
class ClosureApplied(CanvasEvent):
    cells: FrozenSet[Coordinate]
    group_order: int


@dataclass(frozen=True, slots=True)
class CanvasMemento:
    """A point-in-time snapshot supporting transactional rollback."""

    version: int
    cells: FrozenSet[Coordinate]


@invariant
class CanvasAggregate:
    """A set of lit cells reconstructible from its own event stream."""

    def __init__(self, order: int) -> None:
        self._order = order
        self._cells: set[Coordinate] = set()
        self._events: list[CanvasEvent] = []
        self._clipped = 0

    def __invariant__(self) -> bool:
        return all(cell.within(self._order) for cell in self._cells)

    @property
    def order(self) -> int:
        return self._order

    @property
    def clipped(self) -> int:
        return self._clipped

    @property
    def version(self) -> int:
        return len(self._events)

    @property
    def support(self) -> FrozenSet[Coordinate]:
        return frozenset(self._cells)

    @property
    def events(self) -> tuple[CanvasEvent, ...]:
        return tuple(self._events)

    def handle(self, command: CanvasCommand) -> CanvasEvent:
        if isinstance(command, PaintCells):
            retained = frozenset(cell for cell in command.cells if cell.within(self._order))
            self._clipped += len(command.cells) - len(retained)
            event: CanvasEvent = CellsPainted(self.version, retained, command.provenance)
        elif isinstance(command, ApplyGroupClosure):
            closure = command.group.close(self._cells)
            retained = frozenset(cell for cell in closure if cell.within(self._order))
            self._clipped += len(closure) - len(retained)
            event = ClosureApplied(self.version, retained, command.group.order)
        else:  # pragma: no cover - defensive
            raise ExecutionFault(f"unroutable canvas command {command!r}")
        self._apply(event)
        self._events.append(event)
        return event

    def _apply(self, event: CanvasEvent) -> None:
        if isinstance(event, CellsPainted):
            self._cells |= event.cells
        elif isinstance(event, ClosureApplied):
            self._cells |= event.cells

    def snapshot(self) -> CanvasMemento:
        return CanvasMemento(self.version, frozenset(self._cells))

    def restore(self, memento: CanvasMemento) -> None:
        self._cells = set(memento.cells)
        del self._events[memento.version:]

    @classmethod
    def replay(cls, order: int, events: Sequence[CanvasEvent]) -> "CanvasAggregate":
        aggregate = cls(order)
        for event in events:
            aggregate._apply(event)
            aggregate._events.append(event)
        return aggregate


class CanvasRepository:
    """An in-memory aggregate store, because a database would be excessive."""

    def __init__(self) -> None:
        self._storage: dict[str, CanvasAggregate] = {}

    def create(self, identity: str, order: int) -> CanvasAggregate:
        aggregate = CanvasAggregate(order)
        self._storage[identity] = aggregate
        return aggregate

    def load(self, identity: str) -> CanvasAggregate:
        try:
            return self._storage[identity]
        except KeyError as exc:
            raise ResolutionFailure(f"no canvas aggregate named {identity!r}") from exc


@contextmanager
def unit_of_work(aggregate: CanvasAggregate) -> Iterator[CanvasAggregate]:
    """Rolls the aggregate back to its entry snapshot if the body raises."""
    memento = aggregate.snapshot()
    try:
        yield aggregate
    except Exception:
        aggregate.restore(memento)
        METRICS.increment("canvas.rollbacks")
        raise


# ======================================================================
# Layer 10: the Glyph Virtual Machine
# ======================================================================


@dataclass(slots=True)
class MachineState:
    """Everything the interpreter and the threaded-code tier both mutate."""

    stack: list[Any] = field(default_factory=list)
    locals: list[int] = field(default_factory=list)
    program_counter: int = 0
    steps: int = 0


class GlyphVirtualMachine:
    """A stack machine with an interpreter tier and a threaded-code tier."""

    def __init__(
        self,
        module: ObjectModule,
        repository: CanvasRepository | None = None,
        jit_enabled: bool = True,
    ) -> None:
        self._module = module
        self._state = MachineState(locals=[0] * max(module.frame_size, 1))
        self._repository = repository or CanvasRepository()
        self._canvas = self._repository.create("primary", module.order)
        self._jit_enabled = jit_enabled
        self._tiered_up = False
        self._threaded: tuple[Callable[[int], int], ...] | None = None
        apothem = module.order // 2
        self._group = SymmetryGroup.generated_by(
            module.family.generators,
            Coordinate(apothem, apothem),
            module.symmetry_order,
            presentation=f"{module.family.value}-{module.cardinality} about centroid",
        )
        self._intrinsics: Mapping[str, int] = {
            "zero": 0,
            "apothem": apothem,
            "extremum": module.order - 1,
            "magnitude": module.order,
        }

    @property
    def group(self) -> SymmetryGroup:
        return self._group

    @property
    def canvas(self) -> CanvasAggregate:
        return self._canvas

    @property
    def tiered_up(self) -> bool:
        return self._tiered_up

    def _pop(self) -> Any:
        try:
            return self._state.stack.pop()
        except IndexError as exc:
            raise ExecutionFault("operand stack underflow") from exc

    @woven
    def run(self) -> FrozenSet[Coordinate]:
        with TRACER.span("execute", jit=self._jit_enabled), unit_of_work(self._canvas):
            state = self._state
            code = self._module.instructions
            while True:
                if not 0 <= state.program_counter < len(code):
                    raise ExecutionFault(
                        f"program counter {state.program_counter} left the module"
                    )
                instruction = code[state.program_counter]
                if isinstance(instruction, Halt):
                    break
                state.steps += 1
                if (
                    self._jit_enabled
                    and not self._tiered_up
                    and state.steps > JIT_TIER_UP_THRESHOLD
                ):
                    self._tier_up()
                if self._threaded is not None:
                    state.program_counter = self._threaded[state.program_counter](
                        state.program_counter
                    )
                else:
                    state.program_counter = self._step(instruction, state.program_counter)
            if state.stack:
                raise ExecutionFault(f"operand stack not drained: {state.stack!r}")
        if self._canvas.clipped:
            DIAGNOSTICS.emit(
                Severity.WARNING, "VM0001", "diag.clipped", count=self._canvas.clipped
            )
        METRICS.increment("vm.steps", state.steps)
        return self._canvas.support

    # -- tier 0: the switch interpreter ---------------------------------

    @functools.singledispatchmethod
    def _step(self, instruction: Instruction, address: int) -> int:
        raise ExecutionFault(f"illegal instruction {instruction!r} at {address}")

    @_step.register
    def _(self, instruction: PushConstant, address: int) -> int:
        self._state.stack.append(instruction.value)
        return address + 1

    @_step.register
    def _(self, instruction: LoadIntrinsic, address: int) -> int:
        try:
            self._state.stack.append(self._intrinsics[instruction.name])
        except KeyError as exc:
            raise ExecutionFault(f"unbound intrinsic {instruction.name!r}") from exc
        return address + 1

    @_step.register
    def _(self, instruction: LoadLocal, address: int) -> int:
        self._state.stack.append(self._state.locals[instruction.slot])
        return address + 1

    @_step.register
    def _(self, instruction: StoreLocal, address: int) -> int:
        self._state.locals[instruction.slot] = self._pop()
        return address + 1

    @_step.register
    def _(self, instruction: BinaryAdd, address: int) -> int:
        right, left = self._pop(), self._pop()
        self._state.stack.append(left + right)
        return address + 1

    @_step.register
    def _(self, instruction: BinarySubtract, address: int) -> int:
        right, left = self._pop(), self._pop()
        self._state.stack.append(left - right)
        return address + 1

    @_step.register
    def _(self, instruction: BinaryMultiply, address: int) -> int:
        right, left = self._pop(), self._pop()
        self._state.stack.append(left * right)
        return address + 1

    @_step.register
    def _(self, instruction: BinaryDivide, address: int) -> int:
        right, left = self._pop(), self._pop()
        if right == 0:
            raise ExecutionFault("division by zero")
        self._state.stack.append(left // right)
        return address + 1

    @_step.register
    def _(self, instruction: Negate, address: int) -> int:
        self._state.stack.append(-self._pop())
        return address + 1

    @_step.register
    def _(self, instruction: CompareLessEqual, address: int) -> int:
        right, left = self._pop(), self._pop()
        self._state.stack.append(1 if left <= right else 0)
        return address + 1

    @_step.register
    def _(self, instruction: MakeInterval, address: int) -> int:
        upper, lower = self._pop(), self._pop()
        if lower > upper:
            raise ExecutionFault(f"degenerate interval [{lower}, {upper}]")
        self._state.stack.append(range(lower, upper + 1))
        return address + 1

    @_step.register
    def _(self, instruction: EmitOrientedRun, address: int) -> int:
        interval, index = self._pop(), self._pop()
        self._canvas.handle(
            PaintCells(
                frozenset(map(_cell_factory(instruction.orientation, index), interval)),
                instruction.orientation.value,
            )
        )
        return address + 1

    @_step.register
    def _(self, instruction: Jump, address: int) -> int:
        return int(instruction.target)

    @_step.register
    def _(self, instruction: JumpIfFalse, address: int) -> int:
        return address + 1 if self._pop() else int(instruction.target)

    @_step.register
    def _(self, instruction: CloseUnderGroup, address: int) -> int:
        self._canvas.handle(ApplyGroupClosure(self._group))
        return address + 1

    # -- tier 1: threaded code ------------------------------------------

    def _tier_up(self) -> None:
        """Replaces dynamic dispatch with a vector of pre-bound closures."""
        with TRACER.span("jit-tier-up", instructions=len(self._module.instructions)):
            self._threaded = tuple(
                self._compile_one(instruction) for instruction in self._module.instructions
            )
            self._tiered_up = True
            METRICS.increment("jit.tier_ups")

    def _compile_one(self, instruction: Instruction) -> Callable[[int], int]:
        step = self._step
        return lambda address, _instruction=instruction: step(_instruction, address)


@functools.lru_cache(maxsize=None)
def _cell_factory(orientation: Orientation, index: int) -> Callable[[int], Coordinate]:
    """Memoised closure turning a scalar into a lattice cell."""
    if orientation is Orientation.ROW:
        return lambda scalar: Coordinate(index, scalar)
    if orientation is Orientation.COLUMN:
        return lambda scalar: Coordinate(scalar, index)
    if orientation is Orientation.DIAGONAL:
        return lambda scalar: Coordinate(scalar, scalar + index)
    return lambda scalar: Coordinate(scalar, index - scalar)


# ======================================================================
# Layer 12: presentation
# ======================================================================


class Ink(enum.Enum):
    OCCUPIED = "occupied"
    VACANT = "vacant"


class Theme(abc.ABC, metaclass=PluginRegistryMeta):
    """A registered mapping from ink to characters."""

    __registry_root__ = True

    separator: ClassVar[str] = " "

    @abc.abstractmethod
    def glyph(self, ink: Ink) -> str: ...

    def join(self, cells: Sequence[str]) -> str:
        return self.separator.join(cells)


class AsteriskTheme(Theme):
    __registry_key__ = "asterisk"

    def glyph(self, ink: Ink) -> str:
        return "*" if ink is Ink.OCCUPIED else " "


class BlockTheme(Theme):
    __registry_key__ = "block"
    separator = ""

    def glyph(self, ink: Ink) -> str:
        return "██" if ink is Ink.OCCUPIED else "  "


class ShadeTheme(Theme):
    __registry_key__ = "shade"

    def glyph(self, ink: Ink) -> str:
        return "#" if ink is Ink.OCCUPIED else "."


@dataclass(frozen=True, slots=True)
class RenderContext:
    """The immutable input to the rasterisation pipeline."""

    order: int
    support: FrozenSet[Coordinate]
    group: SymmetryGroup
    theme: Theme
    workers: int = RASTERIZER_WORKER_COUNT


@runtime_checkable
class RasterizationStrategy(Protocol):
    """Turns a set of lit cells into a character mosaic."""

    def rasterize(self, context: RenderContext) -> str: ...


@dataclass(frozen=True, slots=True)
class ConcurrentRasterizer:
    """Materialises row bands in parallel and reassembles them in order."""

    def rasterize(self, context: RenderContext) -> str:
        order, theme, support = context.order, context.theme, context.support

        def band(bounds: tuple[int, int]) -> tuple[int, list[str]]:
            start, stop = bounds
            rendered = [
                theme.join(
                    [
                        theme.glyph(
                            Ink.OCCUPIED if Coordinate(row, column) in support else Ink.VACANT
                        )
                        for column in range(order)
                    ]
                )
                for row in range(start, stop)
            ]
            METRICS.increment("rasteriser.bands")
            return start, rendered

        stride = max(1, -(-order // max(context.workers, 1)))
        partitions = [(start, min(start + stride, order)) for start in range(0, order, stride)]
        with ThreadPoolExecutor(max_workers=context.workers) as pool:
            bands = sorted(pool.map(band, partitions), key=operator.itemgetter(0))
        return "\n".join(line for _, lines in bands for line in lines)


@dataclass(frozen=True, slots=True)
class SequentialRasterizer:
    """A single-threaded oracle used by the assurance layer for cross-checking."""

    def rasterize(self, context: RenderContext) -> str:
        theme = context.theme
        return "\n".join(
            theme.join(
                [
                    theme.glyph(
                        Ink.OCCUPIED
                        if Coordinate(row, column) in context.support
                        else Ink.VACANT
                    )
                    for column in range(context.order)
                ]
            )
            for row in range(context.order)
        )


@runtime_checkable
class RenderMiddleware(Protocol):
    """A link in the rendering chain of responsibility."""

    def __call__(
        self, context: RenderContext, nxt: Callable[[RenderContext], str]
    ) -> str: ...


@dataclass(frozen=True, slots=True)
class TracingMiddleware:
    def __call__(self, context: RenderContext, nxt: Callable[[RenderContext], str]) -> str:
        with TRACER.span("render", cells=len(context.support), order=context.order):
            return nxt(context)


@dataclass(frozen=True, slots=True)
class BoundsCheckingMiddleware:
    def __call__(self, context: RenderContext, nxt: Callable[[RenderContext], str]) -> str:
        strays = [cell for cell in context.support if not cell.within(context.order)]
        if strays:
            raise ExecutionFault(f"{len(strays)} cell(s) escaped the lattice: {strays[:3]}")
        return nxt(context)


@dataclass(frozen=True, slots=True)
class SymmetryInvariantMiddleware:
    def __call__(self, context: RenderContext, nxt: Callable[[RenderContext], str]) -> str:
        if not context.group.is_invariant(context.support):
            raise GroupAxiomViolation("the support is not invariant under the declared group")
        METRICS.increment("render.invariant_checks")
        return nxt(context)


@dataclass(frozen=True, slots=True)
class TrailingSpaceTrimmingMiddleware:
    def __call__(self, context: RenderContext, nxt: Callable[[RenderContext], str]) -> str:
        return "\n".join(line.rstrip() for line in nxt(context).splitlines())


class RenderPipeline:
    """Folds the middleware sequence around the terminal strategy."""

    def __init__(
        self, strategy: RasterizationStrategy, middleware: Sequence[RenderMiddleware] = ()
    ) -> None:
        self._strategy = strategy
        self._middleware = tuple(middleware)

    @woven
    def __call__(self, context: RenderContext) -> str:
        chain: Callable[[RenderContext], str] = self._strategy.rasterize
        for layer in reversed(self._middleware):
            chain = functools.partial(self._link, layer, chain)
        return chain(context)

    @staticmethod
    def _link(
        layer: RenderMiddleware, nxt: Callable[[RenderContext], str], context: RenderContext
    ) -> str:
        return layer(context, nxt)


# ======================================================================
# Layer 13: the composition root
# ======================================================================


class Lifetime(enum.Enum):
    TRANSIENT = enum.auto()
    SINGLETON = enum.auto()


@dataclass(frozen=True, slots=True)
class Registration:
    key: Any
    factory: Callable[..., Any]
    lifetime: Lifetime


class ServiceContainer:
    """An autowiring container resolving constructor parameters by annotation."""

    def __init__(self) -> None:
        self._registrations: dict[Any, Registration] = {}
        self._singletons: dict[Any, Any] = {}
        self._resolving: list[Any] = []

    def register_instance(self, key: Any, instance: Any) -> "ServiceContainer":
        self._registrations[key] = Registration(key, lambda: instance, Lifetime.SINGLETON)
        self._singletons[key] = instance
        return self

    def register(
        self, key: Any, factory: Callable[..., Any], lifetime: Lifetime = Lifetime.TRANSIENT
    ) -> "ServiceContainer":
        self._registrations[key] = Registration(key, factory, lifetime)
        return self

    def resolve(self, key: Any) -> Any:
        if key in self._resolving:
            cycle = " -> ".join(getattr(item, "__name__", str(item)) for item in [*self._resolving, key])
            raise ResolutionFailure(f"circular dependency: {cycle}")
        registration = self._registrations.get(key)
        if registration is None:
            if inspect.isclass(key):
                return self._autowire(key)
            raise ResolutionFailure(f"nothing registered for {key!r}")
        if registration.lifetime is Lifetime.SINGLETON and key in self._singletons:
            return self._singletons[key]
        self._resolving.append(key)
        try:
            instance = self._invoke(registration.factory)
        finally:
            self._resolving.pop()
        if registration.lifetime is Lifetime.SINGLETON:
            self._singletons[key] = instance
        return instance

    def _invoke(self, factory: Callable[..., Any]) -> Any:
        signature = inspect.signature(factory)
        if not signature.parameters:
            return factory()
        hints = typing.get_type_hints(factory)
        arguments = {
            name: self.resolve(hints[name])
            for name in signature.parameters
            if name in hints
        }
        return factory(**arguments)

    def _autowire(self, cls: type) -> Any:
        self._resolving.append(cls)
        try:
            return self._invoke(cls)
        finally:
            self._resolving.pop()

    def render(self) -> str:
        return "\n".join(
            f"{getattr(key, '__name__', key)} -> "
            f"{getattr(registration.factory, '__qualname__', registration.factory)} "
            f"[{registration.lifetime.name.lower()}]"
            for key, registration in sorted(
                self._registrations.items(), key=lambda item: str(item[0])
            )
        )


# ======================================================================
# Layer 14: motif catalogue and orchestration
# ======================================================================


class RewriteError(GlyphPlatformError):
    """A term did not reach a normal form."""


REWRITE_LIMIT: Final[int] = 512

ORIENTATION_ATOMS: Final[Mapping[str, Orientation]] = {
    "r": Orientation.ROW,
    "c": Orientation.COLUMN,
    "d": Orientation.DIAGONAL,
    "a": Orientation.ANTIDIAGONAL,
}

DUAL_ATOMS: Final[Mapping[str, str]] = {"c": "r", "r": "c", "d": "a", "a": "d"}


def applied(*symbols: Any) -> Any:
    """Left-associated application, which is the only way a term is built."""
    head, *rest = symbols
    for symbol in rest:
        head = (head, symbol)
    return head


def _spine(term: Any) -> tuple[Any, list[Any]]:
    """Unwinds an application into what is being applied and to what."""
    arguments: list[Any] = []
    while isinstance(term, tuple):
        term, argument = term
        arguments.append(argument)
    arguments.reverse()
    return term, arguments


def _pair(term: Any) -> tuple[int, int]:
    head, arguments = _spine(term)
    if head != "P" or len(arguments) != 2:
        raise RewriteError(f"expected a pair, found {term!r}")
    return typing.cast(tuple[int, int], tuple(arguments))


def _entry(term: Any) -> list[Any]:
    head, arguments = _spine(term)
    if head != "R" or len(arguments) != 4:
        raise RewriteError(f"expected an entry, found {term!r}")
    return arguments


def _dual(atom: Any) -> str:
    try:
        return DUAL_ATOMS[atom]
    except (KeyError, TypeError) as exc:
        raise RewriteError(f"{atom!r} has no dual") from exc


def _successor(term: Any) -> Any:
    scale, offset = _pair(term)
    return applied("P", scale + 1, offset)


def _increment(term: Any) -> Any:
    scale, offset = _pair(term)
    return applied("P", scale, offset + 1)


def _advance(term: Any) -> Any:
    orientation, index, lower, upper = _entry(term)
    return applied("R", _dual(orientation), lower, _increment(index), upper)


def _widen(term: Any) -> Any:
    orientation, index, lower, upper = _entry(term)
    return applied("R", orientation, _increment(index), lower, upper)


REWRITE_RULES: Final[Mapping[str, tuple[int, Callable[..., Any]]]] = {
    "I": (1, lambda x: x),
    "K": (2, lambda x, y: x),
    "S": (3, lambda x, y, z: applied(x, z, (y, z))),
    "B": (3, lambda x, y, z: applied(x, (y, z))),
    "C": (3, lambda x, y, z: applied(x, z, y)),
    "W": (2, lambda x, y: applied(x, y, y)),
    "O": (0, lambda: applied("P", 0, 0)),
    "N": (1, _successor),
    "U": (1, _increment),
    "A": (1, _advance),
    "H": (1, _widen),
}

# The rules that read their argument rather than merely moving it, and so
# cannot fire until it has stopped changing.
STRICT_RULES: Final[FrozenSet[str]] = frozenset({"N", "U", "A", "H"})


def _rebuild(head: Any, arguments: Sequence[Any]) -> Any:
    for argument in arguments:
        head = (head, argument)
    return head


def _rewrite_once(term: Any) -> Any | None:
    """One step, leftmost and outermost, or ``None`` at a normal form."""
    head, arguments = _spine(term)
    rule = REWRITE_RULES.get(head) if isinstance(head, str) else None
    if rule is not None:
        arity, contract = rule
        if len(arguments) >= arity:
            if head in STRICT_RULES:
                for position in range(arity):
                    reduced = _rewrite_once(arguments[position])
                    if reduced is not None:
                        arguments[position] = reduced
                        return _rebuild(head, arguments)
            return _rebuild(contract(*arguments[:arity]), arguments[arity:])
    for position, argument in enumerate(arguments):
        reduced = _rewrite_once(argument)
        if reduced is not None:
            arguments[position] = reduced
            return _rebuild(head, arguments)
    return None


def normal_form(term: Any, limit: int = REWRITE_LIMIT) -> Any:
    """Rewrites until the term stops changing, and answers what it settled on."""
    for _ in range(limit):
        reduced = _rewrite_once(term)
        if reduced is None:
            return term
        term = reduced
    raise RewriteError(f"no normal form within {limit} rewrites")


@functools.lru_cache(maxsize=None)
def seed_of(term: Any) -> tuple["SeedRun", ...]:
    """The entries a term settles on, read off its normal form."""
    entries: list[SeedRun] = []
    cursor = normal_form(term)
    while True:
        head, arguments = _spine(cursor)
        if head == "E":
            return tuple(entries)
        if head != "L" or len(arguments) != 2:
            raise RewriteError(f"expected a list, found {cursor!r}")
        orientation, index, lower, upper = _entry(arguments[0])
        try:
            oriented = ORIENTATION_ATOMS[orientation]
        except (KeyError, TypeError) as exc:
            raise RewriteError(f"{orientation!r} names no orientation") from exc
        entries.append(SeedRun(oriented, _pair(index), _pair(lower), _pair(upper)))
        cursor = arguments[1]


@dataclass(frozen=True, slots=True)
class SeedRun:
    """One entry of a seed: an oriented run, sited relative to the centre.

    Each of the three bounds is a pair, read as that many apothems plus that
    much again, so an entry says the same thing at every order.
    """

    orientation: Orientation
    index: tuple[int, int]
    lower: tuple[int, int]
    upper: tuple[int, int]


def _seed_term(coefficients: tuple[int, int]) -> str:
    """One bound of an entry, as an expression in the bound variable."""
    scale, offset = coefficients
    magnitude = "UNITY" if abs(offset) == 1 else str(abs(offset))
    if not scale:
        if not offset:
            return "zero"
        return magnitude if offset > 0 else f"( zero - {magnitude} )"
    base = "u" if scale == 1 else f"( u * {scale} )"
    if not offset:
        return base
    return f"( {base} + {magnitude} )" if offset > 0 else f"( {base} - {magnitude} )"


class Motif(abc.ABC, metaclass=PluginRegistryMeta):
    """A registered seed, and the group whose orbit of it is taken.

    A subclass carries data and no text: the entries of its seed, the family
    it is closed under, and that family's cardinality.  The translation unit
    is derived from those three, so what distinguishes one entry of the
    catalogue from another is a handful of integers.
    """

    __registry_root__ = True

    description: ClassVar[str] = ""
    family: ClassVar[SymmetryFamily] = SymmetryFamily.CYCLIC
    cardinality: ClassVar[int] = 4
    term: ClassVar[Any] = "E"
    declared: ClassVar[bool] = False

    @property
    def seed(self) -> tuple[SeedRun, ...]:
        return seed_of(self.term)

    def source(self, order: int) -> str:
        lines = [
            "#pragma gsl 2",
            "#pragma platform hyperenterprise",
            f"#define CARDINALITY {self.cardinality}",
            "#define UNITY ( 2 - 1 )",
            "",
            f"lattice order {order} ;",
            f"symmetry {self.family.value} CARDINALITY about centroid ;",
            "",
            "let u = apothem ;",
            "",
        ]
        for position, entry in enumerate(self.seed):
            index = _seed_term(entry.index)
            lower, upper = _seed_term(entry.lower), _seed_term(entry.upper)
            run = f"{entry.orientation.value} at {index}"
            if self.declared:
                lines.append(f"stroke s{position} = {run} span {lower} .. {upper} ;")
                lines.append(f"emit s{position} ;")
            else:
                bound = f"k{position}"
                lines.append(f"for {bound} in {lower} .. {upper} {{")
                lines.append(f"    paint {run} span {bound} .. {bound} ;")
                lines.append("}")
        return "\n".join(lines) + "\n"


ROOT: Final[Any] = applied(
    "R", "c", applied("N", "O"), "O", applied("N", applied("N", "O"))
)

# The tail of a list, from whatever the head settles on.
TAIL: Final[Any] = applied("B", applied("C", "L", "E"), "A")
LONGER: Final[Any] = applied(
    "S", applied("B", "L", "A"), applied("B", applied("C", "L", "E"), applied("B", "H", "A"))
)


class PrimaryMotif(Motif):
    """The root and what follows from it, under a group of order four."""

    __registry_key__ = "primary"
    description = "the root and what follows from it, under a group of order four"
    term = applied("S", "L", TAIL, ROOT)


class DoubledMotif(Motif):
    """The same, carried one step further."""

    __registry_key__ = "doubled"
    description = "the same, carried one step further"
    term = applied("S", "L", LONGER, ROOT)


class MinimalMotif(Motif):
    """The root alone."""

    __registry_key__ = "minimal"
    description = "the root alone"
    declared = True
    term = applied("C", "L", "E", ROOT)


# The two entries no operator relates, so the only seed written out in full.
SLANT: Final[Any] = applied(
    "L",
    applied("R", "d", applied("U", "O"), "O", applied("N", applied("N", "O"))),
    applied(
        "L",
        applied("R", "a", applied("N", applied("N", "O")), "O",
                applied("N", applied("N", "O"))),
        "E",
    ),
)


class StatedMotif(Motif):
    """Two entries neither of which follows from the other."""

    __registry_key__ = "stated"
    description = "two entries stated rather than derived, one of each remaining orientation"
    term = SLANT


class FoldedMotif(Motif):
    """The primary term under a group of twice the cardinality."""

    __registry_key__ = "folded"
    description = "the primary term, closed under a group of order eight"
    family = SymmetryFamily.DIHEDRAL
    term = PrimaryMotif.term


class PipelineStage(enum.Enum):
    PENDING = enum.auto()
    PREPROCESSED = enum.auto()
    TOKENIZED = enum.auto()
    PARSED = enum.auto()
    ANALYZED = enum.auto()
    LOWERED = enum.auto()
    OPTIMISED = enum.auto()
    ASSEMBLED = enum.auto()
    SERIALISED = enum.auto()
    EXECUTED = enum.auto()
    RENDERED = enum.auto()
    COMPLETE = enum.auto()


LEGAL_TRANSITIONS: Final[Mapping[PipelineStage, FrozenSet[PipelineStage]]] = {
    PipelineStage.PENDING: frozenset({PipelineStage.PREPROCESSED}),
    PipelineStage.PREPROCESSED: frozenset({PipelineStage.TOKENIZED}),
    PipelineStage.TOKENIZED: frozenset({PipelineStage.PARSED}),
    PipelineStage.PARSED: frozenset({PipelineStage.ANALYZED}),
    PipelineStage.ANALYZED: frozenset({PipelineStage.LOWERED}),
    PipelineStage.LOWERED: frozenset({PipelineStage.OPTIMISED, PipelineStage.ASSEMBLED}),
    PipelineStage.OPTIMISED: frozenset({PipelineStage.ASSEMBLED}),
    PipelineStage.ASSEMBLED: frozenset({PipelineStage.SERIALISED, PipelineStage.EXECUTED}),
    PipelineStage.SERIALISED: frozenset({PipelineStage.EXECUTED}),
    PipelineStage.EXECUTED: frozenset({PipelineStage.RENDERED}),
    PipelineStage.RENDERED: frozenset({PipelineStage.COMPLETE}),
    PipelineStage.COMPLETE: frozenset(),
}


class PipelineStateMachine:
    """Enforces the declared stage ordering of the synthesis pipeline."""

    def __init__(self) -> None:
        self._stage = PipelineStage.PENDING
        self._history: list[PipelineStage] = [PipelineStage.PENDING]

    @property
    def stage(self) -> PipelineStage:
        return self._stage

    @property
    def history(self) -> tuple[PipelineStage, ...]:
        return tuple(self._history)

    @contextmanager
    def transition(self, target: PipelineStage) -> Iterator[None]:
        if target not in LEGAL_TRANSITIONS[self._stage]:
            raise GlyphPlatformError(
                f"illegal pipeline transition {self._stage.name} -> {target.name}"
            )
        EVENT_BUS.publish(StageEntered(target.name, CATALOG("stage.enter", stage=target.name)))
        yield
        self._stage = target
        self._history.append(target)


@dataclass(frozen=True, slots=True)
class CompilationArtifacts:
    """Every intermediate the toolchain produced for one synthesis request."""

    configuration: Configuration
    motif: str
    source: str
    unit: PreprocessedUnit
    tokens: tuple[Token, ...]
    tree: Program
    model: SemanticModel
    listing: tuple[IrItem, ...]
    optimised: tuple[IrItem, ...]
    graph: ControlFlowGraph
    passes: PassManager
    module: ObjectModule
    blob: bytes
    machine: GlyphVirtualMachine
    support: FrozenSet[Coordinate]
    rendering: str
    history: tuple[PipelineStage, ...]
    elapsed_ms: float


class SynthesisOrchestrator:
    """Drives every layer in order and collects the artefacts along the way."""

    def __init__(
        self,
        configuration: Configuration,
        container: "ServiceContainer | None" = None,
        source: str | None = None,
    ) -> None:
        self._configuration = configuration
        self._flags = FeatureFlags(configuration)
        self._machine = PipelineStateMachine()
        self._container = container or self._compose()
        self._source = source

    def _compose(self) -> ServiceContainer:
        container = ServiceContainer()
        theme_cls = typing.cast(type, Theme.lookup(self._configuration["theme"]))
        container.register_instance(Theme, theme_cls())
        container.register(
            RasterizationStrategy, ConcurrentRasterizer, Lifetime.SINGLETON
        )
        container.register(Assembler, Assembler, Lifetime.SINGLETON)
        container.register(ObjectCodec, ObjectCodec, Lifetime.SINGLETON)
        container.register(CanvasRepository, CanvasRepository, Lifetime.SINGLETON)
        container.register(PassManager, PassManager, Lifetime.SINGLETON)
        return container

    @property
    def container(self) -> ServiceContainer:
        return self._container

    def _announce(self, stage: PipelineStage, size: int) -> None:
        EVENT_BUS.publish(
            StageCompleted(stage.name, CATALOG("stage.leave", stage=stage.name, size=size), size)
        )

    @woven
    def run(self) -> CompilationArtifacts:
        started = time.perf_counter()
        order = int(self._configuration["lattice.order"])
        motif_key = str(self._configuration["motif"])
        if self._source is not None:
            source = self._source
        else:
            source = typing.cast(type, Motif.lookup(motif_key))().source(order)

        with self._machine.transition(PipelineStage.PREPROCESSED):
            unit = Preprocessor(source).run()
        self._announce(PipelineStage.PREPROCESSED, len(unit.text))

        with self._machine.transition(PipelineStage.TOKENIZED):
            tokens = FiniteStateTransducer(unit).tokenize()
        self._announce(PipelineStage.TOKENIZED, len(tokens))

        with self._machine.transition(PipelineStage.PARSED):
            tree = RecursiveDescentParser(tokens).parse()
        self._announce(PipelineStage.PARSED, len(tuple(tree.walk())))

        with self._machine.transition(PipelineStage.ANALYZED):
            model = SemanticAnalyzer().analyze(tree)
        self._announce(PipelineStage.ANALYZED, model.frame_size)

        with self._machine.transition(PipelineStage.LOWERED):
            listing = BytecodeEmitter(model).lower(tree)
        self._announce(PipelineStage.LOWERED, len(listing))

        passes = typing.cast(PassManager, self._container.resolve(PassManager))
        if self._flags.optimise:
            with self._machine.transition(PipelineStage.OPTIMISED):
                optimised = passes.optimise(listing)
            self._announce(PipelineStage.OPTIMISED, len(optimised))
        else:
            optimised = tuple(listing)

        graph = ControlFlowGraph(optimised)

        with self._machine.transition(PipelineStage.ASSEMBLED):
            assembler = typing.cast(Assembler, self._container.resolve(Assembler))
            module = assembler.assemble(optimised, model)
        self._announce(PipelineStage.ASSEMBLED, len(module.instructions))

        codec = typing.cast(ObjectCodec, self._container.resolve(ObjectCodec))
        blob = codec.encode(module)
        if self._flags.roundtrip:
            with self._machine.transition(PipelineStage.SERIALISED):
                module = codec.decode(blob)
            self._announce(PipelineStage.SERIALISED, len(blob))

        with self._machine.transition(PipelineStage.EXECUTED):
            vm = GlyphVirtualMachine(
                module,
                typing.cast(CanvasRepository, self._container.resolve(CanvasRepository)),
                jit_enabled=self._flags.jit,
            )
            support = vm.run()
        self._announce(PipelineStage.EXECUTED, len(support))

        with self._machine.transition(PipelineStage.RENDERED):
            pipeline = RenderPipeline(
                strategy=typing.cast(
                    RasterizationStrategy, self._container.resolve(RasterizationStrategy)
                ),
                middleware=(
                    TracingMiddleware(),
                    BoundsCheckingMiddleware(),
                    SymmetryInvariantMiddleware(),
                    TrailingSpaceTrimmingMiddleware(),
                ),
            )
            context = RenderContext(
                order=module.order,
                support=support,
                group=vm.group,
                theme=typing.cast(Theme, self._container.resolve(Theme)),
                workers=int(self._configuration["workers"]),
            )
            rendering = pipeline(context)
        self._announce(PipelineStage.RENDERED, len(rendering))

        with self._machine.transition(PipelineStage.COMPLETE):
            pass

        return CompilationArtifacts(
            configuration=self._configuration,
            motif=motif_key,
            source=source,
            unit=unit,
            tokens=tokens,
            tree=tree,
            model=model,
            listing=tuple(listing),
            optimised=tuple(optimised),
            graph=graph,
            passes=passes,
            module=module,
            blob=blob,
            machine=vm,
            support=support,
            rendering=rendering,
            history=self._machine.history,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )


# ======================================================================
# Layer 15: assurance
# ======================================================================


@dataclass(frozen=True, slots=True)
class CheckResult:
    name: str
    passed: bool
    detail: str = ""

    def render(self) -> str:
        mark = "PASS" if self.passed else "FAIL"
        return f"[{mark}] {self.name}" + (f" — {self.detail}" if self.detail else "")


class AssuranceSuite:
    """A battery of invariants asserted against a completed synthesis."""

    def run(self, artifacts: CompilationArtifacts) -> tuple[CheckResult, ...]:
        return tuple(
            check(artifacts)
            for check in (
                self._group_axioms,
                self._lagrange,
                self._orbit_stabiliser,
                self._codec_roundtrip,
                self._optimiser_preserves_semantics,
                self._closure_idempotent,
                self._support_invariant,
                self._rasterisers_agree,
                self._event_replay,
                self._lexer_covers_source,
                self._forms_answer_to_grammar,
                self._coordinate_flyweight,
                self._coordinate_immutable,
            )
        )

    @staticmethod
    def _group_axioms(artifacts: CompilationArtifacts) -> CheckResult:
        try:
            artifacts.machine.group.audit(artifacts.module.symmetry_order)
        except GroupAxiomViolation as error:
            return CheckResult("group axioms", False, str(error))
        return CheckResult("group axioms", True, f"order {artifacts.machine.group.order}")

    @staticmethod
    def _lagrange(artifacts: CompilationArtifacts) -> CheckResult:
        group = artifacts.machine.group
        return CheckResult(
            "Lagrange's theorem",
            group.satisfies_lagrange(),
            f"{len(group.cyclic_subgroups())} cyclic subgroup(s)",
        )

    @staticmethod
    def _orbit_stabiliser(artifacts: CompilationArtifacts) -> CheckResult:
        group = artifacts.machine.group
        sample = sorted(artifacts.support, key=lambda c: (c.row, c.column))[:8]
        ok = all(group.satisfies_orbit_stabiliser(point) for point in sample)
        return CheckResult("orbit-stabiliser theorem", ok, f"{len(sample)} sample point(s)")

    @staticmethod
    def _codec_roundtrip(artifacts: CompilationArtifacts) -> CheckResult:
        codec = ObjectCodec()
        again = codec.encode(codec.decode(artifacts.blob))
        return CheckResult(
            "object codec round-trip", again == artifacts.blob, f"{len(artifacts.blob)} bytes"
        )

    @staticmethod
    def _optimiser_preserves_semantics(artifacts: CompilationArtifacts) -> CheckResult:
        baseline = Assembler().assemble(artifacts.listing, artifacts.model)
        support = GlyphVirtualMachine(baseline, jit_enabled=False).run()
        removed = len(artifacts.listing) - len(artifacts.optimised)
        return CheckResult(
            "optimiser preserves semantics",
            support == artifacts.support,
            f"{removed} item(s) removed with no observable difference",
        )

    @staticmethod
    def _closure_idempotent(artifacts: CompilationArtifacts) -> CheckResult:
        group = artifacts.machine.group
        once = group.close(artifacts.support)
        return CheckResult("closure is idempotent", group.close(once) == once)

    @staticmethod
    def _support_invariant(artifacts: CompilationArtifacts) -> CheckResult:
        return CheckResult(
            "support is group-invariant", artifacts.machine.group.is_invariant(artifacts.support)
        )

    @staticmethod
    def _rasterisers_agree(artifacts: CompilationArtifacts) -> CheckResult:
        context = RenderContext(
            artifacts.module.order, artifacts.support, artifacts.machine.group, AsteriskTheme()
        )
        concurrent = ConcurrentRasterizer().rasterize(context)
        sequential = SequentialRasterizer().rasterize(context)
        return CheckResult("rasteriser tiers agree", concurrent == sequential)

    @staticmethod
    def _event_replay(artifacts: CompilationArtifacts) -> CheckResult:
        canvas = artifacts.machine.canvas
        replayed = CanvasAggregate.replay(canvas.order, canvas.events)
        return CheckResult(
            "event replay reconstructs the canvas",
            replayed.support == artifacts.support,
            f"{len(canvas.events)} event(s)",
        )

    @staticmethod
    def _lexer_covers_source(artifacts: CompilationArtifacts) -> CheckResult:
        reconstructed = "".join(token.trivia + token.lexeme for token in artifacts.tokens)
        return CheckResult(
            "lexer partitions the whole unit", reconstructed == artifacts.unit.text
        )

    @staticmethod
    def _forms_answer_to_grammar(artifacts: CompilationArtifacts) -> CheckResult:
        disagreements = grammar_disagreements()
        return CheckResult(
            "the grammar answers for every form",
            not disagreements,
            f"{len(FORMS)} form(s), each with one production to read"
            if not disagreements
            else "; ".join(disagreements),
        )

    @staticmethod
    def _coordinate_flyweight(artifacts: CompilationArtifacts) -> CheckResult:
        return CheckResult(
            "coordinates are interned", Coordinate(2, 3) is Coordinate(2, 3)
        )

    @staticmethod
    def _coordinate_immutable(artifacts: CompilationArtifacts) -> CheckResult:
        try:
            Coordinate(1, 1).row = 9  # type: ignore[misc]
        except ContractViolation:
            return CheckResult("coordinates reject mutation", True)
        return CheckResult("coordinates reject mutation", False, "assignment succeeded")


# ======================================================================
# The façade and the driver
# ======================================================================


def synthesize(
    order: int = DEFAULT_LATTICE_ORDER, **overrides: Any
) -> Result[CompilationArtifacts]:
    """Compiles and renders one motif, wrapping any platform error in ``Err``."""
    entries = {"lattice.order": str(order)}
    entries.update({key: str(value) for key, value in overrides.items()})
    configuration = (
        ConfigurationBuilder()
        .with_defaults()
        .with_environment()
        .with_mapping("api-call", entries)
        .build()
    )
    return Result.attempt(lambda: SynthesisOrchestrator(configuration).run())


def synthesize_source(
    source: str, order: int = DEFAULT_LATTICE_ORDER, **overrides: Any
) -> Result[CompilationArtifacts]:
    """Compiles and renders arbitrary GSL source rather than a registered motif."""
    entries = {"lattice.order": str(order), "motif": "<source>"}
    entries.update({key: str(value) for key, value in overrides.items()})
    configuration = (
        ConfigurationBuilder()
        .with_defaults()
        .with_environment()
        .with_mapping("api-call", entries)
        .build()
    )
    return Result.attempt(
        lambda: SynthesisOrchestrator(configuration, source=source).run()
    )


# ======================================================================
# Layer 16: the LLVM lowering backend
# ======================================================================
#
# The virtual machine is a stack machine with arbitrary control flow, so the
# lowering keeps the operand stack in memory - an alloca'd array plus a stack
# pointer - and gives every instruction address its own basic block.  Jumps
# become branches between those blocks.  This is deliberately naive: SROA and
# mem2reg promote the stack away at -O1 and above, and what survives to the
# object file bears no resemblance to what is emitted here.


class LlvmToolchainUnavailable(GlyphPlatformError):
    """The optional llvmlite binding is not importable in this interpreter."""


class LlvmLoweringError(GlyphPlatformError):
    """The object module could not be lowered to the LLVM IR dialect."""


@dataclass(frozen=True, slots=True)
class TargetProfile:
    """Static description of the machine the IR is being lowered for."""

    triple: str = "x86_64-unknown-linux-gnu"
    identifier: str = "glyph.canonical"
    source_filename: str = "canonical.gsl"


@dataclass(frozen=True, slots=True)
class LlvmModule:
    """A textual LLVM IR translation unit plus its provenance metadata."""

    text: str
    profile: TargetProfile
    order: int

    def __str__(self) -> str:
        return self.text


OPERAND_STACK_DEPTH: Final[int] = 256

_ORIENTATION_ENCODING: Final[Mapping[Orientation, int]] = {
    Orientation.ROW: 0,
    Orientation.COLUMN: 1,
    Orientation.DIAGONAL: 2,
    Orientation.ANTIDIAGONAL: 3,
}

_LLVM_RUNTIME_TEMPLATE: Final[string.Template] = string.Template(
    """
@canvas = internal global [$CELLS x i8] zeroinitializer
@snapshot = internal global [$CELLS x i8] zeroinitializer

declare i32 @putchar(i32)
declare i32 @fflush(ptr)

define internal void @gvm.paint(i32 %row, i32 %col) {
entry:
  %r0 = icmp sge i32 %row, 0
  %r1 = icmp slt i32 %row, $ORDER
  %rok = and i1 %r0, %r1
  %c0 = icmp sge i32 %col, 0
  %c1 = icmp slt i32 %col, $ORDER
  %cok = and i1 %c0, %c1
  %ok = and i1 %rok, %cok
  br i1 %ok, label %store, label %done

store:
  %o0 = mul nsw i32 %row, $ORDER
  %o1 = add nsw i32 %o0, %col
  %o2 = sext i32 %o1 to i64
  %slot = getelementptr inbounds [$CELLS x i8], ptr @canvas, i64 0, i64 %o2
  store i8 1, ptr %slot, align 1
  br label %done

done:
  ret void
}

define internal void @gvm.emit_run(i32 %orient, i32 %index, i32 %lo, i32 %hi) {
entry:
  %cursor = alloca i32, align 4
  store i32 %lo, ptr %cursor, align 4
  br label %head

head:
  %cur = load i32, ptr %cursor, align 4
  %more = icmp sle i32 %cur, %hi
  br i1 %more, label %body, label %exit

body:
  %scalar = load i32, ptr %cursor, align 4
  switch i32 %orient, label %o.row [ i32 1, label %o.column
                                     i32 2, label %o.diagonal
                                     i32 3, label %o.antidiagonal ]

o.row:
  call void @gvm.paint(i32 %index, i32 %scalar)
  br label %step

o.column:
  call void @gvm.paint(i32 %scalar, i32 %index)
  br label %step

o.diagonal:
  %d.col = add nsw i32 %scalar, %index
  call void @gvm.paint(i32 %scalar, i32 %d.col)
  br label %step

o.antidiagonal:
  %a.col = sub nsw i32 %index, %scalar
  call void @gvm.paint(i32 %scalar, i32 %a.col)
  br label %step

step:
  %now = load i32, ptr %cursor, align 4
  %next = add nsw i32 %now, 1
  store i32 %next, ptr %cursor, align 4
  br label %head

exit:
  ret void
}

define internal void @gvm.apply(i32 %a, i32 %b, i32 %c, i32 %d) {
entry:
  %row = alloca i32, align 4
  %col = alloca i32, align 4
  store i32 0, ptr %row, align 4
  br label %row.head

row.head:
  %r = load i32, ptr %row, align 4
  %rmore = icmp slt i32 %r, $ORDER
  br i1 %rmore, label %row.body, label %row.done

row.body:
  store i32 0, ptr %col, align 4
  br label %col.head

col.head:
  %cv = load i32, ptr %col, align 4
  %cmore = icmp slt i32 %cv, $ORDER
  br i1 %cmore, label %col.body, label %col.done

col.body:
  %br0 = load i32, ptr %row, align 4
  %bc0 = load i32, ptr %col, align 4
  %q0 = mul nsw i32 %br0, $ORDER
  %q1 = add nsw i32 %q0, %bc0
  %q2 = sext i32 %q1 to i64
  %src = getelementptr inbounds [$CELLS x i8], ptr @snapshot, i64 0, i64 %q2
  %ink = load i8, ptr %src, align 1
  %lit = icmp ne i8 %ink, 0
  br i1 %lit, label %col.mark, label %col.step

col.mark:
  %mr = load i32, ptr %row, align 4
  %mc = load i32, ptr %col, align 4
  %dr = sub nsw i32 %mr, $APOTHEM
  %dc = sub nsw i32 %mc, $APOTHEM
  %t0 = mul nsw i32 %a, %dr
  %t1 = mul nsw i32 %b, %dc
  %t2 = add nsw i32 %t0, %t1
  %nr = add nsw i32 %t2, $APOTHEM
  %t3 = mul nsw i32 %c, %dr
  %t4 = mul nsw i32 %d, %dc
  %t5 = add nsw i32 %t3, %t4
  %nc = add nsw i32 %t5, $APOTHEM
  call void @gvm.paint(i32 %nr, i32 %nc)
  br label %col.step

col.step:
  %cnow = load i32, ptr %col, align 4
  %cnext = add nsw i32 %cnow, 1
  store i32 %cnext, ptr %col, align 4
  br label %col.head

col.done:
  %rnow = load i32, ptr %row, align 4
  %rnext = add nsw i32 %rnow, 1
  store i32 %rnext, ptr %row, align 4
  br label %row.head

row.done:
  ret void
}

define internal void @gvm.snapshot() {
entry:
  %i = alloca i32, align 4
  store i32 0, ptr %i, align 4
  br label %head

head:
  %c = load i32, ptr %i, align 4
  %more = icmp slt i32 %c, $CELLS
  br i1 %more, label %body, label %done

body:
  %ix = load i32, ptr %i, align 4
  %ix64 = sext i32 %ix to i64
  %sp = getelementptr inbounds [$CELLS x i8], ptr @canvas, i64 0, i64 %ix64
  %sv = load i8, ptr %sp, align 1
  %dp = getelementptr inbounds [$CELLS x i8], ptr @snapshot, i64 0, i64 %ix64
  store i8 %sv, ptr %dp, align 1
  %nx = add nsw i32 %ix, 1
  store i32 %nx, ptr %i, align 4
  br label %head

done:
  ret void
}

define internal void @gvm.render() {
entry:
  %row = alloca i32, align 4
  %col = alloca i32, align 4
  %last = alloca i32, align 4
  store i32 0, ptr %row, align 4
  br label %row.head

row.head:
  %r = load i32, ptr %row, align 4
  %rmore = icmp slt i32 %r, $ORDER
  br i1 %rmore, label %scan.init, label %row.done

scan.init:
  store i32 -1, ptr %last, align 4
  store i32 0, ptr %col, align 4
  br label %scan.head

scan.head:
  %sc = load i32, ptr %col, align 4
  %smore = icmp slt i32 %sc, $ORDER
  br i1 %smore, label %scan.body, label %scan.exit

scan.body:
  %sr = load i32, ptr %row, align 4
  %sc2 = load i32, ptr %col, align 4
  %so0 = mul nsw i32 %sr, $ORDER
  %so1 = add nsw i32 %so0, %sc2
  %so2 = sext i32 %so1 to i64
  %sp = getelementptr inbounds [$CELLS x i8], ptr @canvas, i64 0, i64 %so2
  %sink = load i8, ptr %sp, align 1
  %slit = icmp ne i8 %sink, 0
  br i1 %slit, label %scan.mark, label %scan.step

scan.mark:
  %sc3 = load i32, ptr %col, align 4
  store i32 %sc3, ptr %last, align 4
  br label %scan.step

scan.step:
  %sc4 = load i32, ptr %col, align 4
  %sc5 = add nsw i32 %sc4, 1
  store i32 %sc5, ptr %col, align 4
  br label %scan.head

scan.exit:
  store i32 0, ptr %col, align 4
  br label %print.head

print.head:
  %pc = load i32, ptr %col, align 4
  %lastv = load i32, ptr %last, align 4
  %pmore = icmp sle i32 %pc, %lastv
  br i1 %pmore, label %print.body, label %print.newline

print.body:
  %pc2 = load i32, ptr %col, align 4
  %needsep = icmp sgt i32 %pc2, 0
  br i1 %needsep, label %print.separator, label %print.cell

print.separator:
  %sepres = call i32 @putchar(i32 32)
  br label %print.cell

print.cell:
  %pr = load i32, ptr %row, align 4
  %pc3 = load i32, ptr %col, align 4
  %po0 = mul nsw i32 %pr, $ORDER
  %po1 = add nsw i32 %po0, %pc3
  %po2 = sext i32 %po1 to i64
  %pp = getelementptr inbounds [$CELLS x i8], ptr @canvas, i64 0, i64 %po2
  %pink = load i8, ptr %pp, align 1
  %plit = icmp ne i8 %pink, 0
  %glyph = select i1 %plit, i32 42, i32 32
  %cellres = call i32 @putchar(i32 %glyph)
  %pc4 = load i32, ptr %col, align 4
  %pc5 = add nsw i32 %pc4, 1
  store i32 %pc5, ptr %col, align 4
  br label %print.head

print.newline:
  %nlres = call i32 @putchar(i32 10)
  %rn = load i32, ptr %row, align 4
  %rn2 = add nsw i32 %rn, 1
  store i32 %rn2, ptr %row, align 4
  br label %row.head

row.done:
  ret void
}
"""
)


class _VirtualRegisterAllocator:
    """Monotonic namer guaranteeing uniqueness of SSA value identifiers."""

    def __init__(self, prefix: str = "g") -> None:
        self._prefix = prefix
        self._counter = itertools.count()

    def fresh(self) -> str:
        return f"%{self._prefix}{next(self._counter)}"


class LlvmLoweringBackend:
    """Translates a linked object module into an LLVM IR translation unit.

    The operand stack lives in an ``alloca``'d array indexed by a stack
    pointer, and every instruction address becomes its own basic block, so
    the machine's jumps translate directly into branches.  Intervals are
    flattened: ``MakeInterval`` is erased and its bounds are left on the stack
    as two ordinary entries, which is exactly what ``EmitOrientedRun`` pops.
    """

    def __init__(self, profile: TargetProfile | None = None) -> None:
        self._profile = profile or TargetProfile()

    @woven
    def lower(self, module: ObjectModule) -> LlvmModule:
        apothem = module.order // 2
        runtime = _LLVM_RUNTIME_TEMPLATE.substitute(
            ORDER=module.order,
            CELLS=module.order * module.order,
            APOTHEM=apothem,
        )
        header = "\n".join(
            (
                f"; ModuleID = '{self._profile.identifier}'",
                f'source_filename = "{self._profile.source_filename}"',
                f'target triple = "{self._profile.triple}"',
            )
        )
        return LlvmModule(
            text="\n".join((header, runtime, self._lower_entry_point(module), "")),
            profile=self._profile,
            order=module.order,
        )

    # -- operand stack helpers ------------------------------------------

    def _push(self, body: list[str], registers: _VirtualRegisterAllocator, value: str) -> None:
        top = registers.fresh()
        slot = registers.fresh()
        bumped = registers.fresh()
        body.append(f"  {top} = load i64, ptr %sp, align 8")
        body.append(
            f"  {slot} = getelementptr inbounds [{OPERAND_STACK_DEPTH} x i64], "
            f"ptr %stack, i64 0, i64 {top}"
        )
        body.append(f"  store i64 {value}, ptr {slot}, align 8")
        body.append(f"  {bumped} = add nsw i64 {top}, 1")
        body.append(f"  store i64 {bumped}, ptr %sp, align 8")

    def _pop(self, body: list[str], registers: _VirtualRegisterAllocator) -> str:
        top = registers.fresh()
        lowered = registers.fresh()
        slot = registers.fresh()
        value = registers.fresh()
        body.append(f"  {top} = load i64, ptr %sp, align 8")
        body.append(f"  {lowered} = sub nsw i64 {top}, 1")
        body.append(f"  store i64 {lowered}, ptr %sp, align 8")
        body.append(
            f"  {slot} = getelementptr inbounds [{OPERAND_STACK_DEPTH} x i64], "
            f"ptr %stack, i64 0, i64 {lowered}"
        )
        body.append(f"  {value} = load i64, ptr {slot}, align 8")
        return value

    def _frame_slot(
        self, body: list[str], registers: _VirtualRegisterAllocator, slot: int, frame: int
    ) -> str:
        address = registers.fresh()
        body.append(
            f"  {address} = getelementptr inbounds [{frame} x i64], "
            f"ptr %frame, i64 0, i64 {slot}"
        )
        return address

    def _floor_divide(
        self, body: list[str], registers: _VirtualRegisterAllocator, left: str, right: str
    ) -> str:
        """Emits flooring division, because the machine's oracle is Python.

        ``sdiv`` truncates towards zero and ``//`` floors, so they part company
        whenever exactly one operand is negative: -1 // 4 is -1, but sdiv gives
        0.  Decrementing the quotient when the remainder is non-zero and its
        sign differs from the divisor's recovers the interpreter's answer.
        """
        quotient = registers.fresh()
        remainder = registers.fresh()
        inexact = registers.fresh()
        remainder_negative = registers.fresh()
        divisor_negative = registers.fresh()
        signs_differ = registers.fresh()
        correct = registers.fresh()
        decremented = registers.fresh()
        result = registers.fresh()
        body.append(f"  {quotient} = sdiv i64 {left}, {right}")
        body.append(f"  {remainder} = srem i64 {left}, {right}")
        body.append(f"  {inexact} = icmp ne i64 {remainder}, 0")
        body.append(f"  {remainder_negative} = icmp slt i64 {remainder}, 0")
        body.append(f"  {divisor_negative} = icmp slt i64 {right}, 0")
        body.append(f"  {signs_differ} = xor i1 {remainder_negative}, {divisor_negative}")
        body.append(f"  {correct} = and i1 {inexact}, {signs_differ}")
        body.append(f"  {decremented} = sub nsw i64 {quotient}, 1")
        body.append(
            f"  {result} = select i1 {correct}, i64 {decremented}, i64 {quotient}"
        )
        return result

    def _truncate(self, body: list[str], registers: _VirtualRegisterAllocator, value: str) -> str:
        narrowed = registers.fresh()
        body.append(f"  {narrowed} = trunc i64 {value} to i32")
        return narrowed

    # -- entry point -----------------------------------------------------

    def _lower_entry_point(self, module: ObjectModule) -> str:
        registers = _VirtualRegisterAllocator()
        frame = max(module.frame_size, 1)
        apothem = module.order // 2
        intrinsics = {
            "zero": 0,
            "apothem": apothem,
            "extremum": module.order - 1,
            "magnitude": module.order,
        }
        group = SymmetryGroup.generated_by(
            module.family.generators,
            Coordinate(apothem, apothem),
            module.symmetry_order,
            presentation=f"{module.family.value}-{module.cardinality} about centroid",
        )

        body: list[str] = [
            "entry:",
            f"  %stack = alloca [{OPERAND_STACK_DEPTH} x i64], align 8",
            "  %sp = alloca i64, align 8",
            f"  %frame = alloca [{frame} x i64], align 8",
            "  store i64 0, ptr %sp, align 8",
        ]
        for slot in range(frame):
            address = self._frame_slot(body, registers, slot, frame)
            body.append(f"  store i64 0, ptr {address}, align 8")
        body.append("  br label %A0")

        for address_index, instruction in enumerate(module.instructions):
            body.append("")
            body.append(f"A{address_index}:")
            self._lower_instruction(
                body, registers, instruction, address_index, module, frame, intrinsics, group
            )

        body.append("")
        body.append("exit:")
        body.append("  call void @gvm.render()")
        body.append(f"  {registers.fresh()} = call i32 @fflush(ptr null)")
        body.append("  ret i32 0")
        return "\n".join(("define i32 @main() {", *body, "}"))

    def _lower_instruction(
        self,
        body: list[str],
        registers: _VirtualRegisterAllocator,
        instruction: Instruction,
        address: int,
        module: ObjectModule,
        frame: int,
        intrinsics: Mapping[str, int],
        group: SymmetryGroup,
    ) -> None:
        fallthrough = f"  br label %A{address + 1}"

        match instruction:
            case Halt():
                body.append("  br label %exit")
                return
            case Jump(target=target):
                body.append(f"  br label %A{int(target)}")
                return
            case JumpIfFalse(target=target):
                condition = self._pop(body, registers)
                test = registers.fresh()
                body.append(f"  {test} = icmp ne i64 {condition}, 0")
                body.append(
                    f"  br i1 {test}, label %A{address + 1}, label %A{int(target)}"
                )
                return
            case PushConstant(value=value):
                self._push(body, registers, str(value))
            case LoadIntrinsic(name=name):
                try:
                    self._push(body, registers, str(intrinsics[name]))
                except KeyError as exc:
                    raise LlvmLoweringError(f"unbound intrinsic {name!r}") from exc
            case LoadLocal(slot=slot):
                address_register = self._frame_slot(body, registers, slot, frame)
                value = registers.fresh()
                body.append(f"  {value} = load i64, ptr {address_register}, align 8")
                self._push(body, registers, value)
            case StoreLocal(slot=slot):
                value = self._pop(body, registers)
                address_register = self._frame_slot(body, registers, slot, frame)
                body.append(f"  store i64 {value}, ptr {address_register}, align 8")
            case BinaryAdd() | BinarySubtract() | BinaryMultiply():
                mnemonic = {
                    BinaryAdd: "add nsw",
                    BinarySubtract: "sub nsw",
                    BinaryMultiply: "mul nsw",
                }[type(instruction)]
                right = self._pop(body, registers)
                left = self._pop(body, registers)
                result = registers.fresh()
                body.append(f"  {result} = {mnemonic} i64 {left}, {right}")
                self._push(body, registers, result)
            case BinaryDivide():
                right = self._pop(body, registers)
                left = self._pop(body, registers)
                self._push(body, registers, self._floor_divide(body, registers, left, right))
            case Negate():
                operand = self._pop(body, registers)
                result = registers.fresh()
                body.append(f"  {result} = sub nsw i64 0, {operand}")
                self._push(body, registers, result)
            case CompareLessEqual():
                right = self._pop(body, registers)
                left = self._pop(body, registers)
                flag = registers.fresh()
                result = registers.fresh()
                body.append(f"  {flag} = icmp sle i64 {left}, {right}")
                body.append(f"  {result} = zext i1 {flag} to i64")
                self._push(body, registers, result)
            case MakeInterval():
                body.append("  ; make.interval is erased: bounds stay on the stack")
            case EmitOrientedRun(orientation=orientation):
                upper = self._truncate(body, registers, self._pop(body, registers))
                lower = self._truncate(body, registers, self._pop(body, registers))
                index = self._truncate(body, registers, self._pop(body, registers))
                body.append(
                    f"  call void @gvm.emit_run(i32 {_ORIENTATION_ENCODING[orientation]}, "
                    f"i32 {index}, i32 {lower}, i32 {upper})"
                )
            case CloseUnderGroup():
                body.append("  call void @gvm.snapshot()")
                for element in group.elements:
                    linear = element.linear
                    body.append(
                        f"  call void @gvm.apply(i32 {linear.a}, i32 {linear.b}, "
                        f"i32 {linear.c}, i32 {linear.d})"
                    )
            case _:
                raise LlvmLoweringError(f"unlowerable instruction {instruction!r}")

        body.append(fallthrough)


class LlvmToolchainService:
    """Adapter over the optional llvmlite binding: verify, optimise, JIT, link."""

    @functools.cached_property
    def _binding(self) -> Any:
        try:
            import llvmlite.binding as binding
        except ImportError as exc:
            raise LlvmToolchainUnavailable(
                "llvmlite is not installed; --emit-llvm still works, but "
                "verification, optimisation, JIT and object emission do not"
            ) from exc
        binding.initialize_native_target()
        binding.initialize_native_asmprinter()
        return binding

    @property
    def version(self) -> str:
        return ".".join(str(part) for part in self._binding.llvm_version_info)

    def _parse(self, module: LlvmModule) -> Any:
        parsed = self._binding.parse_assembly(module.text)
        parsed.verify()
        return parsed

    def _target_machine(self, *, jit: bool = False) -> Any:
        """The machine to lower for, which is not the same one twice.

        Objects go to a linker whose default is a position-independent
        executable, so they are emitted small and PIC.  llvmlite's default code
        model is the large one: it puts the text in ``.ltext`` with absolute
        relocations, and the linker then warns its way around them and marks
        the result DT_TEXTREL.  The JIT keeps that default, since it resolves
        the addresses itself and never goes near a linker.
        """
        target = self._binding.Target.from_default_triple()
        if jit:
            return target.create_target_machine()
        return target.create_target_machine(reloc="pic", codemodel="small")

    @woven
    def verify(self, module: LlvmModule) -> str:
        self._parse(module)
        return f"LLVM {self.version}: module verified"

    @woven
    def optimize(self, module: LlvmModule, level: int = 2) -> LlvmModule:
        binding = self._binding
        parsed = self._parse(module)
        tuning = binding.PipelineTuningOptions(speed_level=level)
        builder = binding.PassBuilder(self._target_machine(), tuning)
        builder.getModulePassManager().run(parsed, builder)
        return LlvmModule(str(parsed), module.profile, module.order)

    @woven
    def object_code(self, module: LlvmModule) -> bytes:
        return self._target_machine().emit_object(self._parse(module))

    @woven
    def emit_object(self, module: LlvmModule, path: str) -> str:
        Path(path).write_bytes(self.object_code(module))
        return path

    @woven
    def emit_assembly(self, module: LlvmModule) -> str:
        return self._target_machine().emit_assembly(self._parse(module))

    @woven
    def jit_execute(self, module: LlvmModule) -> int:
        import ctypes

        binding = self._binding
        machine = self._target_machine(jit=True)
        parsed = self._parse(module)
        with binding.create_mcjit_compiler(parsed, machine) as engine:
            engine.finalize_object()
            engine.run_static_constructors()
            entry = engine.get_function_address("main")
            if not entry:
                raise LlvmToolchainUnavailable("the JIT could not resolve @main")
            return ctypes.CFUNCTYPE(ctypes.c_int)(entry)()


# ======================================================================
# Layer 17: differential fuzzing
# ======================================================================
#
# Every tier in this file claims to compute the same thing.  The assurance
# suite checks that on one hand-written motif, which is the weakest possible
# evidence.  This layer generates random glyph programs instead and asserts
# that the interpreter, the threaded-code tier, the optimiser, the object
# codec, the LLVM JIT and a linked native binary all agree on every one of
# them.  Programs are generated valid by construction: intervals cannot be
# degenerate, divisors cannot be zero, and every value stays far inside the
# range where Python's arbitrary-precision integers and LLVM's wrapping
# 64-bit arithmetic are obliged to agree.


GENERATED_ORDERS: Final[tuple[int, ...]] = (3, 5, 7, 9, 11, 15)
MAXIMUM_EXPRESSION_DEPTH: Final[int] = 3
MAXIMUM_NESTING_DEPTH: Final[int] = 2


class GslProgramGenerator:
    """Emits random but always well-formed GSL translation units."""

    def __init__(self, entropy: random.Random) -> None:
        self._entropy = entropy
        self._counter = itertools.count()
        self._scopes: list[list[str]] = [[]]
        self._strokes: list[list[str]] = [[]]

    # -- naming and scope ----------------------------------------------

    def _name(self, stem: str) -> str:
        return f"{stem}{next(self._counter)}"

    def _visible(self) -> tuple[str, ...]:
        return tuple(name for scope in self._scopes for name in scope)

    def _visible_strokes(self) -> tuple[str, ...]:
        return tuple(name for scope in self._strokes for name in scope)

    @contextmanager
    def _nested(self) -> Iterator[None]:
        self._scopes.append([])
        self._strokes.append([])
        try:
            yield
        finally:
            self._scopes.pop()
            self._strokes.pop()

    # -- expressions ----------------------------------------------------

    def _atom(self) -> str:
        choices = list(INTRINSIC_NAMES) + [*self._visible()]
        if self._entropy.random() < 0.4 or not choices:
            return str(self._entropy.randint(0, 9))
        return self._entropy.choice(choices)

    def _expression(self, depth: int = 0) -> str:
        if depth >= MAXIMUM_EXPRESSION_DEPTH or self._entropy.random() < 0.45:
            return self._atom()
        roll = self._entropy.random()
        if roll < 0.12:
            return f"- {self._expression(depth + 1)}"
        if roll < 0.22:
            # A literal divisor keeps the quotient defined for every input.
            return f"( {self._expression(depth + 1)} / {self._entropy.randint(1, 4)} )"
        symbol = self._entropy.choice(("+", "-", "*"))
        left, right = self._expression(depth + 1), self._expression(depth + 1)
        return f"( {left} {symbol} {right} )"

    def _run(self) -> str:
        orientation = self._entropy.choice([member.value for member in Orientation])
        index = self._expression()
        lower = self._expression()
        width = self._entropy.randint(0, 4)
        upper = lower if width == 0 else f"( {lower} + {width} )"
        return f"{orientation} at {index} span {lower} .. {upper}"

    # -- statements -----------------------------------------------------

    def _statements(self, depth: int, budget: int) -> list[str]:
        emitted: list[str] = []
        for _ in range(budget):
            emitted.extend(self._statement(depth))
        return emitted

    def _statement(self, depth: int) -> list[str]:
        roll = self._entropy.random()
        if roll < 0.22:
            name = self._name("v")
            line = f"let {name} = {self._expression()} ;"
            self._scopes[-1].append(name)
            return [line]
        if roll < 0.50:
            name = self._name("s")
            declaration = f"stroke {name} = {self._run()} ;"
            self._strokes[-1].append(name)
            return [declaration, f"emit {name} ;"]
        if roll < 0.60 and self._visible_strokes():
            return [f"emit {self._entropy.choice(self._visible_strokes())} ;"]
        if roll < 0.85 or depth >= MAXIMUM_NESTING_DEPTH:
            return [f"paint {self._run()} ;"]
        variable = self._name("i")
        lower = self._entropy.randint(0, 3)
        upper = lower + self._entropy.randint(0, 3)
        with self._nested():
            self._scopes[-1].append(variable)
            body = self._statements(depth + 1, self._entropy.randint(1, 2))
        return [
            f"for {variable} in {lower} .. {upper} {{",
            *(f"    {line}" for line in body),
            "}",
        ]

    # -- translation units ----------------------------------------------

    def generate(self) -> tuple[str, int]:
        self._scopes, self._strokes = [[]], [[]]
        order = self._entropy.choice(GENERATED_ORDERS)
        family = self._entropy.choice([member.value for member in SymmetryFamily])
        directives = []
        if self._entropy.random() < 0.5:
            directives = ["#define SPAN 4", "#pragma gsl 2"]
        body = self._statements(0, self._entropy.randint(1, 4))
        if not any(line.startswith(("emit", "paint")) for line in body):
            body.append(f"paint {self._run()} ;")
        lines = [
            *directives,
            f"lattice order {order} ;",
            f"symmetry {family} 4 about centroid ;",
            *body,
        ]
        return "\n".join(lines) + "\n", order


def _capture_native_stdout(action: Callable[[], Any]) -> str:
    """Runs ``action`` with file descriptor 1 redirected into a temporary file.

    The JIT writes through C stdio, which Python's ``sys.stdout`` cannot see,
    so the descriptor itself has to be rebound for the duration of the call.
    """
    sys.stdout.flush()
    saved = os.dup(1)
    with tempfile.TemporaryFile(mode="w+b") as sink:
        os.dup2(sink.fileno(), 1)
        try:
            action()
        finally:
            sys.stdout.flush()
            os.dup2(saved, 1)
            os.close(saved)
        sink.seek(0)
        return sink.read().decode()


@dataclass(frozen=True, slots=True)
class FuzzFinding:
    """One generated program on which two tiers disagreed."""

    case: int
    tier: str
    source: str
    expected: str
    produced: str
    baseline: str = "interpreter"

    def render(self) -> str:
        return "\n".join(
            (
                f"case {self.case}: tier {self.tier!r} disagreed with {self.baseline}",
                "--- source ---",
                self.source.rstrip(),
                f"--- {self.baseline} ---",
                self.expected,
                f"--- {self.tier} ---",
                self.produced,
            )
        )


def _first_difference(expected: str, produced: str) -> tuple[str, str]:
    """The first line on which two listings part company, with its number."""
    left, right = expected.splitlines(), produced.splitlines()
    for number in range(max(len(left), len(right))):
        one = left[number] if number < len(left) else "(end of listing)"
        other = right[number] if number < len(right) else "(end of listing)"
        if one != other:
            return f"line {number + 1}: {one}", f"line {number + 1}: {other}"
    return "(identical)", "(identical)"


@dataclass(frozen=True, slots=True)
class FuzzReport:
    executed: int
    tiers: tuple[str, ...]
    comparisons: int
    findings: tuple[FuzzFinding, ...]
    skipped: tuple[str, ...]

    @property
    def clean(self) -> bool:
        return not self.findings

    def render(self) -> str:
        lines = [
            f"{self.executed} program(s), {self.comparisons} cross-tier comparison(s) "
            f"over {', '.join(self.tiers)}"
        ]
        for skip in self.skipped:
            lines.append(f"  skipped: {skip}")
        if self.clean:
            lines.append("  [ok]   every tier agreed on every generated program")
        else:
            lines.append(f"  [FAIL] {len(self.findings)} disagreement(s)")
            lines.extend(finding.render() for finding in self.findings)
        return "\n".join(lines)


class DifferentialFuzzer:
    """Cross-checks every execution tier against the plain interpreter."""

    def __init__(
        self, seed: int = 0, native: bool = False, front_end: bool = False
    ) -> None:
        self._entropy = random.Random(seed)
        self._generator = GslProgramGenerator(self._entropy)
        self._native = native
        self._front_end = front_end

    def _render(self, source: str, order: int, **flags: Any) -> str:
        return synthesize_source(source, order, **flags).unwrap_or_raise().rendering

    @woven
    def run(self, iterations: int = 100) -> FuzzReport:
        with tempfile.TemporaryDirectory(prefix="ouroboros-fuzz-") as scratch:
            return self._sweep(iterations, Path(scratch))

    def _sweep(self, iterations: int, scratch: Path) -> FuzzReport:
        findings: list[FuzzFinding] = []
        skipped: list[str] = []
        tiers = ["interpreter", "threaded", "optimised", "round-tripped"]
        toolchain: LlvmToolchainService | None = None
        try:
            toolchain = LlvmToolchainService()
            toolchain.version
            tiers.append("jit")
            if self._native:
                tiers.append("native")
        except LlvmToolchainUnavailable as exc:
            toolchain = None
            skipped.append(f"llvm tiers ({exc})")

        machines: list[str] = []
        for architecture in MACHINES:
            if not machine_code_runnable(architecture):
                skipped.append(f"elf({architecture}) tier (this host cannot run it)")
                continue
            try:
                self._through_machine(
                    synthesize(3).unwrap_or_raise().module, scratch, architecture
                )
                tiers.append(f"elf({architecture})")
                machines.append(architecture)
            except (GlyphPlatformError, OSError) as exc:
                skipped.append(f"elf({architecture}) tier ({exc})")

        wasm = False
        if wasm_host() is not None:
            try:
                self._through_wasm(synthesize(3).unwrap_or_raise().module, scratch)
                tiers.append("wasm")
                wasm = True
            except (GlyphPlatformError, OSError, subprocess.CalledProcessError) as exc:
                skipped.append(f"wasm tier ({exc})")
        else:
            skipped.append("wasm tier (no WebAssembly host is installed)")
        tiers.append("read-back")

        front_end: Path | None = None
        if self._front_end and toolchain is not None:
            try:
                front_end, _ = build_front_end(scratch / "loop")
                tiers.extend(("front-end", "front-end-ir"))
            except (GlyphPlatformError, subprocess.CalledProcessError, OSError) as exc:
                skipped.append(f"front-end tiers ({exc})")

        comparisons = 0
        executed = 0
        for case in range(iterations):
            source, order = self._generator.generate()
            baseline = self._render(
                source, order, optimise="false", jit="false", roundtrip="false"
            )
            executed += 1
            variants: list[tuple[str, Callable[[], str]]] = [
                ("threaded", lambda: self._render(
                    source, order, optimise="false", jit="true", roundtrip="false")),
                ("optimised", lambda: self._render(
                    source, order, optimise="true", jit="false", roundtrip="false")),
                ("round-tripped", lambda: self._render(
                    source, order, optimise="true", jit="true", roundtrip="true")),
            ]
            if toolchain is not None:
                variants.append(("jit", lambda: self._through_llvm(source, order, toolchain)))
                if self._native:
                    variants.append(("native", lambda: self._through_native(source, order)))
            for architecture in machines:
                variants.append(
                    (f"elf({architecture})", lambda a=architecture: self._through_machine(
                        self._object_for(source, order), scratch, a))
                )
            if wasm:
                variants.append(
                    ("wasm", lambda: self._through_wasm(
                        self._object_for(source, order), scratch))
                )
            variants.append(
                ("read-back", lambda: self._through_read_back(
                    self._object_for(source, order)))
            )
            if front_end is not None:
                # The front end's claim is byte equality with layer 16, which is
                # stronger than agreeing on the glyph, so both are checked.
                produced_ir = _run(front_end, source)
                reference_ir = self._module_for(source, order).text
                comparisons += 1
                if produced_ir != reference_ir:
                    expected, got = _first_difference(reference_ir, produced_ir)
                    findings.append(
                        FuzzFinding(case, "front-end-ir", source, expected, got, "layer 16")
                    )
                variants.append(
                    ("front-end", lambda: self._through_front_end(produced_ir, scratch))
                )

            for label, produce in variants:
                comparisons += 1
                produced = produce()
                if produced != baseline:
                    findings.append(
                        FuzzFinding(case, label, source, baseline, produced)
                    )
        return FuzzReport(
            executed, tuple(tiers), comparisons, tuple(findings), tuple(skipped)
        )

    def _object_for(self, source: str, order: int) -> ObjectModule:
        return synthesize_source(source, order).unwrap_or_raise().module

    def _module_for(self, source: str, order: int) -> LlvmModule:
        return LlvmLoweringBackend().lower(self._object_for(source, order))

    def _through_machine(
        self, module: ObjectModule, scratch: Path, architecture: str | None = None
    ) -> str:
        binary = write_executable(module, scratch / "machine", architecture)
        return _run(binary, "").removesuffix("\n")

    def _through_wasm(self, module: ObjectModule, scratch: Path) -> str:
        return run_wasm(write_wasm(module, scratch / "case.wasm")).removesuffix("\n")

    def _through_read_back(self, module: ObjectModule) -> str:
        return execute_wasm(wasm_module(module)).removesuffix("\n")

    def _through_llvm(
        self, source: str, order: int, toolchain: LlvmToolchainService
    ) -> str:
        module = self._module_for(source, order)
        return _capture_native_stdout(
            lambda: toolchain.jit_execute(module)
        ).removesuffix("\n")

    def _through_native(self, source: str, order: int) -> str:
        module = self._module_for(source, order)
        with tempfile.TemporaryDirectory(prefix="ouroboros-fuzz-") as scratch:
            binary = link_executable(module.text, Path(scratch) / "case", 2)
            return _run(binary, "").removesuffix("\n")

    def _through_front_end(self, ir: str, scratch: Path) -> str:
        return _run(link_executable(ir, scratch / "case", 2), "").removesuffix("\n")


def fuzz(
    iterations: int = 100,
    seed: int = 0,
    native: bool = False,
    front_end: bool = False,
) -> FuzzReport:
    """Generates ``iterations`` random programs and cross-checks every tier."""
    return DifferentialFuzzer(seed, native, front_end).run(iterations)


# ======================================================================
# Tier 3: the GSL-2 self-hosting bootstrap
# ======================================================================
#
# The glyph language cannot express a compiler, so self-hosting needs a
# second, general language.  GSL-2 is that language: one integer type, one
# flat memory, and just enough control flow to write a compiler in.  The
# compiler for it is written in itself and embedded below verbatim; the
# Python seed that follows exists only to turn the crank the first time.


class Gsl2Error(GlyphPlatformError):
    """The GSL-2 seed compiler rejected a translation unit."""


GSLC_GSL2: Final[str] = r'''# gslc.gsl2 - the GSL-2 compiler, written in GSL-2.
#
# Reads GSL-2 source on stdin, writes LLVM IR on stdout.  Semantically
# identical to the stage-0 compiler gsl2c.py: for any accepted input both
# emit the same bytes, which is what makes the bootstrap fixpoint checkable.

var MEMSIZE = 2000000;
var STRBASE = 1500000;
var STRLIMIT = 490000;
var FRAME = 64;
var MAXGLOBALS = 256;

var SRC = 0;
var STRBUF = 500000;
var LOCNAME = 900000;
var LOCLEN = 900100;
var GLBNAME = 900200;
var GLBLEN = 900600;
var TOKBUF = 950000;
var ARGS = 960000;

var T_EOF = 0;
var T_NUM = 1;
var T_IDENT = 2;
var T_STR = 3;
var T_LPAREN = 4;
var T_RPAREN = 5;
var T_LBRACE = 6;
var T_RBRACE = 7;
var T_LBRACK = 8;
var T_RBRACK = 9;
var T_COMMA = 10;
var T_SEMI = 11;
var T_ASSIGN = 12;
var T_EQ = 13;
var T_NE = 14;
var T_LT = 15;
var T_LE = 16;
var T_GT = 17;
var T_GE = 18;
var T_PLUS = 19;
var T_MINUS = 20;
var T_STAR = 21;
var T_SLASH = 22;
var T_PERCENT = 23;
var T_NOT = 24;
var T_ANDAND = 25;
var T_OROR = 26;
var T_FN = 27;
var T_VAR = 28;
var T_IF = 29;
var T_ELSE = 30;
var T_WHILE = 31;
var T_RETURN = 32;
var T_MEM = 33;

var pos = 0;
var srclen = 0;
var strtop = 1;
var nlocals = 0;
var nglobals = 0;
var regcnt = 0;
var labelcnt = 0;
var line = 1;
var argsp = 0;

fn strlen(s) {
  var i = 0;
  while (mem[s + i] != 0) {
    i = i + 1;
  }
  return i;
}

fn es(s) {
  var i = 0;
  while (mem[s + i] != 0) {
    putchar(mem[s + i]);
    i = i + 1;
  }
  return 0;
}

fn en(n) {
  if (n < 0) {
    putchar('-');
    n = 0 - n;
  }
  if (n >= 10) {
    en(n / 10);
  }
  putchar(48 + n % 10);
  return 0;
}

fn ec(c) {
  putchar(c);
  return 0;
}

fn fail(msg) {
  es("; error: ");
  es(msg);
  es(" near line ");
  en(line);
  es("\n");
  exit(1);
  return 0;
}

fn kw_is(start, length, word) {
  if (length != strlen(word)) {
    return 0;
  }
  var i = 0;
  while (i < length) {
    if (mem[start + i] != mem[word + i]) {
      return 0;
    }
    i = i + 1;
  }
  return 1;
}

fn name_eq(a, alen, b, blen) {
  if (alen != blen) {
    return 0;
  }
  var i = 0;
  while (i < alen) {
    if (mem[a + i] != mem[b + i]) {
      return 0;
    }
    i = i + 1;
  }
  return 1;
}

fn emit_name(start, length) {
  var i = 0;
  while (i < length) {
    ec(mem[start + i]);
    i = i + 1;
  }
  return 0;
}

# ----------------------------------------------------------------------
# lexer
# ----------------------------------------------------------------------

fn is_digit(c) {
  if (c >= 48 && c <= 57) {
    return 1;
  }
  return 0;
}

fn is_alpha(c) {
  if (c >= 97 && c <= 122) {
    return 1;
  }
  if (c >= 65 && c <= 90) {
    return 1;
  }
  if (c == 95) {
    return 1;
  }
  return 0;
}

fn is_alnum(c) {
  if (is_alpha(c)) {
    return 1;
  }
  return is_digit(c);
}

fn skip_ws() {
  while (1) {
    var c = mem[SRC + pos];
    if (c == 35) {
      while (mem[SRC + pos] != 10 && mem[SRC + pos] != 0) {
        pos = pos + 1;
      }
    } else {
      if (c == 10) {
        line = line + 1;
        pos = pos + 1;
      } else {
        if (c == 32 || c == 9 || c == 13) {
          pos = pos + 1;
        } else {
          return 0;
        }
      }
    }
  }
  return 0;
}

fn escape_of(c) {
  if (c == 110) {
    return 10;
  }
  if (c == 116) {
    return 9;
  }
  if (c == 48) {
    return 0;
  }
  if (c == 92) {
    return 92;
  }
  if (c == 34) {
    return 34;
  }
  if (c == 39) {
    return 39;
  }
  if (c == 114) {
    return 13;
  }
  fail("bad escape");
  return 0;
}

fn scan(slot) {
  var base = TOKBUF + slot * 4;
  skip_ws();
  var c = mem[SRC + pos];
  mem[base + 1] = 0;
  mem[base + 2] = SRC + pos;
  mem[base + 3] = 0;
  if (c == 0) {
    mem[base] = T_EOF;
    return 0;
  }
  if (is_digit(c)) {
    var v = 0;
    while (is_digit(mem[SRC + pos])) {
      v = v * 10 + mem[SRC + pos] - 48;
      pos = pos + 1;
    }
    mem[base] = T_NUM;
    mem[base + 1] = v;
    return 0;
  }
  if (is_alpha(c)) {
    var start = SRC + pos;
    while (is_alnum(mem[SRC + pos])) {
      pos = pos + 1;
    }
    var length = SRC + pos - start;
    mem[base + 2] = start;
    mem[base + 3] = length;
    if (kw_is(start, length, "fn")) {
      mem[base] = T_FN;
      return 0;
    }
    if (kw_is(start, length, "var")) {
      mem[base] = T_VAR;
      return 0;
    }
    if (kw_is(start, length, "if")) {
      mem[base] = T_IF;
      return 0;
    }
    if (kw_is(start, length, "else")) {
      mem[base] = T_ELSE;
      return 0;
    }
    if (kw_is(start, length, "while")) {
      mem[base] = T_WHILE;
      return 0;
    }
    if (kw_is(start, length, "return")) {
      mem[base] = T_RETURN;
      return 0;
    }
    if (kw_is(start, length, "mem")) {
      mem[base] = T_MEM;
      return 0;
    }
    mem[base] = T_IDENT;
    return 0;
  }
  if (c == 39) {
    pos = pos + 1;
    var w = mem[SRC + pos];
    if (w == 92) {
      pos = pos + 1;
      w = escape_of(mem[SRC + pos]);
    }
    pos = pos + 1;
    if (mem[SRC + pos] != 39) {
      fail("unterminated char literal");
    }
    pos = pos + 1;
    mem[base] = T_NUM;
    mem[base + 1] = w;
    return 0;
  }
  if (c == 34) {
    pos = pos + 1;
    var off = STRBASE + strtop;
    while (mem[SRC + pos] != 34) {
      if (mem[SRC + pos] == 0) {
        fail("unterminated string literal");
      }
      var u = mem[SRC + pos];
      if (u == 92) {
        pos = pos + 1;
        u = escape_of(mem[SRC + pos]);
      }
      mem[STRBUF + strtop] = u;
      strtop = strtop + 1;
      pos = pos + 1;
    }
    pos = pos + 1;
    mem[STRBUF + strtop] = 0;
    strtop = strtop + 1;
    if (strtop > STRLIMIT) {
      fail("string pool overflow");
    }
    mem[base] = T_STR;
    mem[base + 1] = off;
    return 0;
  }
  pos = pos + 1;
  var d = mem[SRC + pos];
  if (c == 61) {
    if (d == 61) {
      pos = pos + 1;
      mem[base] = T_EQ;
      return 0;
    }
    mem[base] = T_ASSIGN;
    return 0;
  }
  if (c == 33) {
    if (d == 61) {
      pos = pos + 1;
      mem[base] = T_NE;
      return 0;
    }
    mem[base] = T_NOT;
    return 0;
  }
  if (c == 60) {
    if (d == 61) {
      pos = pos + 1;
      mem[base] = T_LE;
      return 0;
    }
    mem[base] = T_LT;
    return 0;
  }
  if (c == 62) {
    if (d == 61) {
      pos = pos + 1;
      mem[base] = T_GE;
      return 0;
    }
    mem[base] = T_GT;
    return 0;
  }
  if (c == 38) {
    if (d == 38) {
      pos = pos + 1;
      mem[base] = T_ANDAND;
      return 0;
    }
    fail("single & is not an operator");
  }
  if (c == 124) {
    if (d == 124) {
      pos = pos + 1;
      mem[base] = T_OROR;
      return 0;
    }
    fail("single | is not an operator");
  }
  if (c == 40) {
    mem[base] = T_LPAREN;
    return 0;
  }
  if (c == 41) {
    mem[base] = T_RPAREN;
    return 0;
  }
  if (c == 123) {
    mem[base] = T_LBRACE;
    return 0;
  }
  if (c == 125) {
    mem[base] = T_RBRACE;
    return 0;
  }
  if (c == 91) {
    mem[base] = T_LBRACK;
    return 0;
  }
  if (c == 93) {
    mem[base] = T_RBRACK;
    return 0;
  }
  if (c == 44) {
    mem[base] = T_COMMA;
    return 0;
  }
  if (c == 59) {
    mem[base] = T_SEMI;
    return 0;
  }
  if (c == 43) {
    mem[base] = T_PLUS;
    return 0;
  }
  if (c == 45) {
    mem[base] = T_MINUS;
    return 0;
  }
  if (c == 42) {
    mem[base] = T_STAR;
    return 0;
  }
  if (c == 47) {
    mem[base] = T_SLASH;
    return 0;
  }
  if (c == 37) {
    mem[base] = T_PERCENT;
    return 0;
  }
  fail("unexpected character");
  return 0;
}

fn tk() {
  return mem[TOKBUF];
}

fn tnum() {
  return mem[TOKBUF + 1];
}

fn tstart() {
  return mem[TOKBUF + 2];
}

fn tlen() {
  return mem[TOKBUF + 3];
}

fn tk2() {
  return mem[TOKBUF + 4];
}

fn advance() {
  mem[TOKBUF] = mem[TOKBUF + 4];
  mem[TOKBUF + 1] = mem[TOKBUF + 5];
  mem[TOKBUF + 2] = mem[TOKBUF + 6];
  mem[TOKBUF + 3] = mem[TOKBUF + 7];
  scan(1);
  return 0;
}

fn expect(kind, what) {
  if (tk() != kind) {
    fail(what);
  }
  advance();
  return 0;
}

# ----------------------------------------------------------------------
# symbol tables
# ----------------------------------------------------------------------

fn declare_local(start, length) {
  if (nlocals >= FRAME) {
    fail("too many locals in one function");
  }
  mem[LOCNAME + nlocals] = start;
  mem[LOCLEN + nlocals] = length;
  nlocals = nlocals + 1;
  return nlocals - 1;
}

fn find_local(start, length) {
  var i = nlocals - 1;
  while (i >= 0) {
    if (name_eq(mem[LOCNAME + i], mem[LOCLEN + i], start, length)) {
      return i;
    }
    i = i - 1;
  }
  return 0 - 1;
}

fn declare_global(start, length) {
  if (nglobals >= MAXGLOBALS) {
    fail("too many globals");
  }
  mem[GLBNAME + nglobals] = start;
  mem[GLBLEN + nglobals] = length;
  nglobals = nglobals + 1;
  return nglobals - 1;
}

fn find_global(start, length) {
  var i = nglobals - 1;
  while (i >= 0) {
    if (name_eq(mem[GLBNAME + i], mem[GLBLEN + i], start, length)) {
      return i;
    }
    i = i - 1;
  }
  return 0 - 1;
}

fn alloc_temp_slot() {
  if (nlocals >= FRAME) {
    fail("frame overflow");
  }
  mem[LOCNAME + nlocals] = 0;
  mem[LOCLEN + nlocals] = 0;
  nlocals = nlocals + 1;
  return nlocals - 1;
}

# ----------------------------------------------------------------------
# emission primitives
# ----------------------------------------------------------------------

fn new_reg() {
  regcnt = regcnt + 1;
  return regcnt;
}

fn new_label() {
  labelcnt = labelcnt + 1;
  return labelcnt;
}

fn er(r) {
  es("%t");
  en(r);
  return 0;
}

fn elabel(l) {
  es("L");
  en(l);
  return 0;
}

fn emit_label(l) {
  elabel(l);
  es(":\n");
  return 0;
}

fn emit_br(l) {
  es("  br label %");
  elabel(l);
  es("\n");
  return 0;
}

fn emit_cond_br(r, a, b) {
  var t = new_reg();
  es("  ");
  er(t);
  es(" = icmp ne i64 ");
  er(r);
  es(", 0\n");
  es("  br i1 ");
  er(t);
  es(", label %");
  elabel(a);
  es(", label %");
  elabel(b);
  es("\n");
  return 0;
}

fn gen_const(v) {
  var r = new_reg();
  es("  ");
  er(r);
  es(" = add i64 0, ");
  en(v);
  es("\n");
  return r;
}

fn gen_slot_addr(slot) {
  var r = new_reg();
  es("  ");
  er(r);
  es(" = getelementptr inbounds [");
  en(FRAME);
  es(" x i64], ptr %frame, i64 0, i64 ");
  en(slot);
  es("\n");
  return r;
}

fn gen_mem_addr(index_reg) {
  var r = new_reg();
  es("  ");
  er(r);
  es(" = getelementptr inbounds [");
  en(MEMSIZE);
  es(" x i64], ptr @memory, i64 0, i64 ");
  er(index_reg);
  es("\n");
  return r;
}

fn gen_load(addr_reg) {
  var r = new_reg();
  es("  ");
  er(r);
  es(" = load i64, ptr ");
  er(addr_reg);
  es(", align 8\n");
  return r;
}

fn gen_store(value_reg, addr_reg) {
  es("  store i64 ");
  er(value_reg);
  es(", ptr ");
  er(addr_reg);
  es(", align 8\n");
  return 0;
}

fn gen_global_load(start, length) {
  var r = new_reg();
  es("  ");
  er(r);
  es(" = load i64, ptr @g_");
  emit_name(start, length);
  es(", align 8\n");
  return r;
}

fn gen_global_store(value_reg, start, length) {
  es("  store i64 ");
  er(value_reg);
  es(", ptr @g_");
  emit_name(start, length);
  es(", align 8\n");
  return 0;
}

fn gen_binary(op, a, b) {
  var r = new_reg();
  es("  ");
  er(r);
  if (op == T_PLUS) {
    es(" = add nsw i64 ");
  }
  if (op == T_MINUS) {
    es(" = sub nsw i64 ");
  }
  if (op == T_STAR) {
    es(" = mul nsw i64 ");
  }
  if (op == T_SLASH) {
    es(" = sdiv i64 ");
  }
  if (op == T_PERCENT) {
    es(" = srem i64 ");
  }
  er(a);
  es(", ");
  er(b);
  es("\n");
  return r;
}

fn gen_compare(op, a, b) {
  var c = new_reg();
  es("  ");
  er(c);
  if (op == T_EQ) {
    es(" = icmp eq i64 ");
  }
  if (op == T_NE) {
    es(" = icmp ne i64 ");
  }
  if (op == T_LT) {
    es(" = icmp slt i64 ");
  }
  if (op == T_LE) {
    es(" = icmp sle i64 ");
  }
  if (op == T_GT) {
    es(" = icmp sgt i64 ");
  }
  if (op == T_GE) {
    es(" = icmp sge i64 ");
  }
  er(a);
  es(", ");
  er(b);
  es("\n");
  var r = new_reg();
  es("  ");
  er(r);
  es(" = zext i1 ");
  er(c);
  es(" to i64\n");
  return r;
}

# ----------------------------------------------------------------------
# expressions
# ----------------------------------------------------------------------

fn parse_call_builtin(kind, first_arg) {
  if (kind == 1) {
    var t = new_reg();
    es("  ");
    er(t);
    es(" = trunc i64 ");
    er(first_arg);
    es(" to i32\n");
    var u = new_reg();
    es("  ");
    er(u);
    es(" = call i32 @putchar(i32 ");
    er(t);
    es(")\n");
    var r = new_reg();
    es("  ");
    er(r);
    es(" = sext i32 ");
    er(u);
    es(" to i64\n");
    return r;
  }
  if (kind == 2) {
    var u2 = new_reg();
    es("  ");
    er(u2);
    es(" = call i32 @getchar()\n");
    var r2 = new_reg();
    es("  ");
    er(r2);
    es(" = sext i32 ");
    er(u2);
    es(" to i64\n");
    return r2;
  }
  var t3 = new_reg();
  es("  ");
  er(t3);
  es(" = trunc i64 ");
  er(first_arg);
  es(" to i32\n");
  es("  call void @exit(i32 ");
  er(t3);
  es(")\n");
  return gen_const(0);
}

fn parse_primary() {
  if (tk() == T_NUM) {
    var v = tnum();
    advance();
    return gen_const(v);
  }
  if (tk() == T_STR) {
    var v2 = tnum();
    advance();
    return gen_const(v2);
  }
  if (tk() == T_LPAREN) {
    advance();
    var r = parse_expr();
    expect(T_RPAREN, "expected )");
    return r;
  }
  if (tk() == T_MEM) {
    advance();
    expect(T_LBRACK, "expected [ after mem");
    var i0 = parse_expr();
    expect(T_RBRACK, "expected ]");
    return gen_load(gen_mem_addr(i0));
  }
  if (tk() == T_IDENT) {
    var start = tstart();
    var length = tlen();
    advance();
    if (tk() != T_LPAREN) {
      var slot = find_local(start, length);
      if (slot >= 0) {
        return gen_load(gen_slot_addr(slot));
      }
      if (find_global(start, length) >= 0) {
        return gen_global_load(start, length);
      }
      fail("unknown identifier");
    }
    advance();
    if (kw_is(start, length, "putchar")) {
      var a1 = parse_expr();
      expect(T_RPAREN, "expected )");
      return parse_call_builtin(1, a1);
    }
    if (kw_is(start, length, "getchar")) {
      expect(T_RPAREN, "expected )");
      return parse_call_builtin(2, 0);
    }
    if (kw_is(start, length, "exit")) {
      var a2 = parse_expr();
      expect(T_RPAREN, "expected )");
      return parse_call_builtin(3, a2);
    }
    var base = argsp;
    argsp = argsp + 9;
    var nargs = 0;
    while (tk() != T_RPAREN) {
      if (nargs > 0) {
        expect(T_COMMA, "expected , between arguments");
      }
      mem[ARGS + base + nargs] = parse_expr();
      nargs = nargs + 1;
      if (nargs > 8) {
        fail("too many arguments");
      }
    }
    advance();
    var r2 = new_reg();
    es("  ");
    er(r2);
    es(" = call i64 @f_");
    emit_name(start, length);
    es("(");
    var i = 0;
    while (i < nargs) {
      if (i > 0) {
        es(", ");
      }
      es("i64 ");
      er(mem[ARGS + base + i]);
      i = i + 1;
    }
    es(")\n");
    argsp = base;
    return r2;
  }
  fail("expected an expression");
  return 0;
}

fn parse_unary() {
  if (tk() == T_MINUS) {
    advance();
    var a = parse_unary();
    return gen_binary(T_MINUS, gen_const(0), a);
  }
  if (tk() == T_NOT) {
    advance();
    var b = parse_unary();
    return gen_compare(T_EQ, b, gen_const(0));
  }
  return parse_primary();
}

fn parse_mul() {
  var a = parse_unary();
  while (tk() == T_STAR || tk() == T_SLASH || tk() == T_PERCENT) {
    var op = tk();
    advance();
    var b = parse_unary();
    a = gen_binary(op, a, b);
  }
  return a;
}

fn parse_add() {
  var a = parse_mul();
  while (tk() == T_PLUS || tk() == T_MINUS) {
    var op = tk();
    advance();
    var b = parse_mul();
    a = gen_binary(op, a, b);
  }
  return a;
}

fn parse_rel() {
  var a = parse_add();
  while (tk() == T_LT || tk() == T_LE || tk() == T_GT || tk() == T_GE) {
    var op = tk();
    advance();
    var b = parse_add();
    a = gen_compare(op, a, b);
  }
  return a;
}

fn parse_eq() {
  var a = parse_rel();
  while (tk() == T_EQ || tk() == T_NE) {
    var op = tk();
    advance();
    var b = parse_rel();
    a = gen_compare(op, a, b);
  }
  return a;
}

fn parse_and() {
  var a = parse_eq();
  while (tk() == T_ANDAND) {
    advance();
    var slot = alloc_temp_slot();
    var s1 = gen_slot_addr(slot);
    es("  store i64 0, ptr ");
    er(s1);
    es(", align 8\n");
    var lrhs = new_label();
    var lend = new_label();
    emit_cond_br(a, lrhs, lend);
    emit_label(lrhs);
    var b = parse_eq();
    var v = gen_compare(T_NE, b, gen_const(0));
    gen_store(v, gen_slot_addr(slot));
    emit_br(lend);
    emit_label(lend);
    a = gen_load(gen_slot_addr(slot));
  }
  return a;
}

fn parse_or() {
  var a = parse_and();
  while (tk() == T_OROR) {
    advance();
    var slot = alloc_temp_slot();
    var s1 = gen_slot_addr(slot);
    es("  store i64 1, ptr ");
    er(s1);
    es(", align 8\n");
    var lrhs = new_label();
    var lend = new_label();
    emit_cond_br(a, lend, lrhs);
    emit_label(lrhs);
    var b = parse_and();
    var v = gen_compare(T_NE, b, gen_const(0));
    gen_store(v, gen_slot_addr(slot));
    emit_br(lend);
    emit_label(lend);
    a = gen_load(gen_slot_addr(slot));
  }
  return a;
}

fn parse_expr() {
  return parse_or();
}

# ----------------------------------------------------------------------
# statements
# ----------------------------------------------------------------------

fn parse_block() {
  expect(T_LBRACE, "expected {");
  while (tk() != T_RBRACE) {
    if (tk() == T_EOF) {
      fail("unterminated block");
    }
    parse_stmt();
  }
  advance();
  return 0;
}

fn parse_stmt() {
  if (tk() == T_VAR) {
    advance();
    if (tk() != T_IDENT) {
      fail("expected a name after var");
    }
    var start = tstart();
    var length = tlen();
    advance();
    expect(T_ASSIGN, "expected = in var declaration");
    var r = parse_expr();
    expect(T_SEMI, "expected ;");
    var slot = declare_local(start, length);
    gen_store(r, gen_slot_addr(slot));
    return 0;
  }
  if (tk() == T_IF) {
    advance();
    expect(T_LPAREN, "expected ( after if");
    var c = parse_expr();
    expect(T_RPAREN, "expected )");
    var lthen = new_label();
    var lelse = new_label();
    var lend = new_label();
    emit_cond_br(c, lthen, lelse);
    emit_label(lthen);
    parse_block();
    emit_br(lend);
    emit_label(lelse);
    if (tk() == T_ELSE) {
      advance();
      parse_block();
    }
    emit_br(lend);
    emit_label(lend);
    return 0;
  }
  if (tk() == T_WHILE) {
    advance();
    var lhead = new_label();
    var lbody = new_label();
    var lend2 = new_label();
    emit_br(lhead);
    emit_label(lhead);
    expect(T_LPAREN, "expected ( after while");
    var c2 = parse_expr();
    expect(T_RPAREN, "expected )");
    emit_cond_br(c2, lbody, lend2);
    emit_label(lbody);
    parse_block();
    emit_br(lhead);
    emit_label(lend2);
    return 0;
  }
  if (tk() == T_RETURN) {
    advance();
    var r2 = parse_expr();
    expect(T_SEMI, "expected ;");
    es("  ret i64 ");
    er(r2);
    es("\n");
    emit_label(new_label());
    return 0;
  }
  if (tk() == T_MEM) {
    advance();
    expect(T_LBRACK, "expected [ after mem");
    var i0 = parse_expr();
    expect(T_RBRACK, "expected ]");
    expect(T_ASSIGN, "expected = in mem assignment");
    var v0 = parse_expr();
    expect(T_SEMI, "expected ;");
    gen_store(v0, gen_mem_addr(i0));
    return 0;
  }
  if (tk() == T_IDENT && tk2() == T_ASSIGN) {
    var start2 = tstart();
    var length2 = tlen();
    advance();
    advance();
    var r3 = parse_expr();
    expect(T_SEMI, "expected ;");
    var slot2 = find_local(start2, length2);
    if (slot2 >= 0) {
      gen_store(r3, gen_slot_addr(slot2));
      return 0;
    }
    if (find_global(start2, length2) >= 0) {
      gen_global_store(r3, start2, length2);
      return 0;
    }
    fail("assignment to unknown identifier");
  }
  parse_expr();
  expect(T_SEMI, "expected ;");
  return 0;
}

# ----------------------------------------------------------------------
# declarations
# ----------------------------------------------------------------------

fn parse_global_decl() {
  advance();
  if (tk() != T_IDENT) {
    fail("expected a name after var");
  }
  var start = tstart();
  var length = tlen();
  advance();
  expect(T_ASSIGN, "expected = in global declaration");
  var neg = 0;
  if (tk() == T_MINUS) {
    neg = 1;
    advance();
  }
  if (tk() != T_NUM) {
    fail("global initializers must be integer literals");
  }
  var v = tnum();
  advance();
  expect(T_SEMI, "expected ;");
  declare_global(start, length);
  es("@g_");
  emit_name(start, length);
  es(" = internal global i64 ");
  if (neg == 1) {
    es("-");
  }
  en(v);
  es("\n");
  return 0;
}

fn parse_function() {
  advance();
  if (tk() != T_IDENT) {
    fail("expected a function name");
  }
  var start = tstart();
  var length = tlen();
  advance();
  expect(T_LPAREN, "expected ( after function name");
  nlocals = 0;
  regcnt = 0;
  labelcnt = 0;
  var nparams = 0;
  while (tk() != T_RPAREN) {
    if (nparams > 0) {
      expect(T_COMMA, "expected , between parameters");
    }
    if (tk() != T_IDENT) {
      fail("expected a parameter name");
    }
    declare_local(tstart(), tlen());
    nparams = nparams + 1;
    advance();
  }
  advance();
  es("\ndefine i64 @f_");
  emit_name(start, length);
  es("(");
  var i = 0;
  while (i < nparams) {
    if (i > 0) {
      es(", ");
    }
    es("i64 %p");
    en(i);
    i = i + 1;
  }
  es(") {\nentry:\n");
  es("  %frame = alloca [");
  en(FRAME);
  es(" x i64], align 8\n");
  i = 0;
  while (i < nparams) {
    es("  %a");
    en(i);
    es(" = getelementptr inbounds [");
    en(FRAME);
    es(" x i64], ptr %frame, i64 0, i64 ");
    en(i);
    es("\n");
    es("  store i64 %p");
    en(i);
    es(", ptr %a");
    en(i);
    es(", align 8\n");
    i = i + 1;
  }
  parse_block();
  es("  ret i64 0\n}\n");
  return 0;
}

fn emit_header() {
  es("target triple = \"x86_64-unknown-linux-gnu\"\n\n");
  es("@memory = internal global [");
  en(MEMSIZE);
  es(" x i64] zeroinitializer\n\n");
  es("declare i32 @putchar(i32)\n");
  es("declare i32 @getchar()\n");
  es("declare void @exit(i32)\n\n");
  return 0;
}

fn emit_trailer() {
  es("\n@strdata = internal constant [");
  en(strtop);
  es(" x i64] [");
  var i = 0;
  while (i < strtop) {
    if (i > 0) {
      es(", ");
    }
    es("i64 ");
    en(mem[STRBUF + i]);
    i = i + 1;
  }
  es("]\n\n");
  es("define internal void @__init_strings() {\nentry:\n");
  es("  %i = alloca i64, align 8\n");
  es("  store i64 0, ptr %i, align 8\n");
  es("  br label %head\n\nhead:\n");
  es("  %c = load i64, ptr %i, align 8\n");
  es("  %m = icmp slt i64 %c, ");
  en(strtop);
  es("\n  br i1 %m, label %body, label %done\n\nbody:\n");
  es("  %si = load i64, ptr %i, align 8\n");
  es("  %sp = getelementptr inbounds [");
  en(strtop);
  es(" x i64], ptr @strdata, i64 0, i64 %si\n");
  es("  %sv = load i64, ptr %sp, align 8\n");
  es("  %dj = add nsw i64 %si, ");
  en(STRBASE);
  es("\n");
  es("  %dp = getelementptr inbounds [");
  en(MEMSIZE);
  es(" x i64], ptr @memory, i64 0, i64 %dj\n");
  es("  store i64 %sv, ptr %dp, align 8\n");
  es("  %ni = add nsw i64 %si, 1\n");
  es("  store i64 %ni, ptr %i, align 8\n");
  es("  br label %head\n\ndone:\n  ret void\n}\n\n");
  es("define i32 @main() {\nentry:\n");
  es("  call void @__init_strings()\n");
  es("  %r = call i64 @f_main()\n");
  es("  %t = trunc i64 %r to i32\n");
  es("  ret i32 %t\n}\n");
  return 0;
}

fn read_source() {
  var i = 0;
  var c = getchar();
  while (c != 0 - 1) {
    mem[SRC + i] = c;
    i = i + 1;
    c = getchar();
  }
  mem[SRC + i] = 0;
  srclen = i;
  return 0;
}

fn compile_unit() {
  emit_header();
  scan(0);
  scan(1);
  while (tk() != T_EOF) {
    if (tk() == T_VAR) {
      parse_global_decl();
    } else {
      if (tk() == T_FN) {
        parse_function();
      } else {
        fail("expected fn or var at top level");
      }
    }
  }
  emit_trailer();
  return 0;
}

fn main() {
  read_source();
  compile_unit();
  return 0;
}
'''

GLYPH_GSL2: Final[str] = r'''# The canonical glyph, expressed in GSL-2.
# Reads an optional odd lattice order from stdin; defaults to 7.
# Two canonical strokes are emitted, then closed under the cyclic group C4.

var order = 7;
var apothem = 3;

fn cell(row, col) {
  return row * order + col;
}

fn emit_run(index, lo, hi, orient) {
  var cursor = lo;
  while (cursor <= hi) {
    if (orient == 0) {
      mem[cell(index, cursor)] = 1;
    } else {
      mem[cell(cursor, index)] = 1;
    }
    cursor = cursor + 1;
  }
  return 0;
}

fn close_group(passes) {
  var pass = 0;
  while (pass < passes) {
    var row = 0;
    while (row < order) {
      var col = 0;
      while (col < order) {
        if (mem[cell(row, col)] != 0) {
          mem[cell(col, 2 * apothem - row)] = 1;
        }
        col = col + 1;
      }
      row = row + 1;
    }
    pass = pass + 1;
  }
  return 0;
}

fn render() {
  var row = 0;
  while (row < order) {
    var last = 0 - 1;
    var col = 0;
    while (col < order) {
      if (mem[cell(row, col)] != 0) {
        last = col;
      }
      col = col + 1;
    }
    col = 0;
    while (col <= last) {
      if (col > 0) {
        putchar(' ');
      }
      if (mem[cell(row, col)] != 0) {
        putchar('*');
      } else {
        putchar(' ');
      }
      col = col + 1;
    }
    putchar('\n');
    row = row + 1;
  }
  return 0;
}

fn read_order() {
  var value = 0;
  var seen = 0;
  var c = getchar();
  while (c >= '0' && c <= '9') {
    value = value * 10 + c - '0';
    seen = 1;
    c = getchar();
  }
  if (seen == 0) {
    return 7;
  }
  return value;
}

fn main() {
  order = read_order();
  if (order < 3 || order % 2 == 0) {
    return 2;
  }
  apothem = order / 2;
  emit_run(apothem, 0, order - 1, 1);
  emit_run(0, apothem + 1, order - 1, 0);
  close_group(3);
  render();
  return 0;
}
'''

# ----------------------------------------------------------------------
# tier 4: the front end, written in the language tier 3 compiles
# ----------------------------------------------------------------------
#
# Everything above is a circle with one end loose: tiers 1 and 2 are
# Python, tier 3 is native but only compiles itself.  glyphc.gsl2 joins
# the ends.  It is the whole of tier 1 - preprocessor, transducer, parser,
# analyser, pass manager, assembler - and the tier 2 lowering after it,
# written a second time in GSL-2 and compiled by the self-hosted compiler
# above.  The claim it makes is not that it agrees but that it is the
# same compiler: for any program either accepts, both emit the same bytes.


GLYPHC_GSL2: Final[str] = r'''# glyphc.gsl2 - the GSL front end, written in GSL-2.
#
# Reads a GSL program on stdin, writes LLVM IR on stdout.  Byte-identical to
# what layer 16 emits for the same program: the whole of tier 1 - preprocessor,
# transducer, parser, analyser, emitter, pass manager, assembler - and then the
# tier 2 lowering, all of it a second time, in a language that compiles itself.

var MEM_STRINGS = 1500000;

var SRC = 0;
var PPS = 300000;
var MACT = 600000;
var MACNM = 660000;
var MACNL = 660300;
var MACVO = 660600;
var MACVL = 660900;
var LINEB = 670000;
var EXPB = 680000;
var TOKK = 700000;
var TOKS = 730000;
var TOKL = 760000;
var TOKV = 790000;
var SYMN = 820000;
var SYML = 822000;
var SYMK = 824000;
var SYMSL = 826000;
var SYMD = 828000;
var SYMR0 = 830000;
var SYMR1 = 832000;
var ITK = 840000;
var ITA = 900000;
var OTK = 960000;
var OTA = 1020000;
var LBLA = 1080000;
var GRPA = 1090000;
var GRPB = 1090100;
var GRPC = 1090200;
var GRPD = 1090300;
var STK = 1100000;
var STA = 1150000;
var REFL = 1200000;
var CODEK = 1210000;
var CODEA = 1270000;

var MAXMACRO = 256;
var MAXTOK = 30000;
var MAXSYM = 2000;
var MAXITEM = 60000;
var MAXLABEL = 8192;

var TK_EOI = 0;
var TK_KW = 1;
var TK_IDENT = 2;
var TK_INT = 3;
var TK_PLUS = 4;
var TK_MINUS = 5;
var TK_STAR = 6;
var TK_SLASH = 7;
var TK_EQUALS = 8;
var TK_RANGE = 9;
var TK_SEMI = 10;
var TK_LPAREN = 11;
var TK_RPAREN = 12;
var TK_LBRACE = 13;
var TK_RBRACE = 14;

var OP_PUSH = 1;
var OP_INTR = 2;
var OP_LOADL = 3;
var OP_STOREL = 4;
var OP_ADD = 5;
var OP_SUB = 6;
var OP_MUL = 7;
var OP_DIV = 8;
var OP_NEG = 9;
var OP_CMPLE = 10;
var OP_MKIV = 11;
var OP_EMIT = 12;
var OP_JMP = 13;
var OP_JF = 14;
var OP_CLOSE = 15;
var OP_HALT = 16;
var OP_LABEL = 17;

var SK_INTRINSIC = 0;
var SK_BINDING = 1;
var SK_STROKE = 2;
var SK_INDUCTION = 3;

var srclen = 0;
var ppslen = 0;
var nmacro = 0;
var mactop = 0;
var ntok = 0;
var tp = 0;
var nsym = 0;
var nitem = 0;
var nslot = 0;
var nlabel = 0;
var depth = 0;
var ORDER = 7;
var CELLS = 49;
var APOTHEM = 3;
var EXTREMUM = 6;
var family = 0;
var cardinality = 4;
var ngroup = 0;
var reg = 0;
var frame = 1;
var stroketop = 0;
var ncode = 0;

# ----------------------------------------------------------------------
# output and diagnostics
# ----------------------------------------------------------------------

fn es(s) {
  var i = 0;
  while (mem[s + i] != 0) {
    putchar(mem[s + i]);
    i = i + 1;
  }
  return 0;
}

fn en(n) {
  if (n < 0) {
    putchar('-');
    n = 0 - n;
  }
  if (n >= 10) {
    en(n / 10);
  }
  putchar(48 + n % 10);
  return 0;
}

fn fail(msg) {
  es("; error: ");
  es(msg);
  es("\n");
  exit(1);
  return 0;
}

fn strlen(s) {
  var i = 0;
  while (mem[s + i] != 0) {
    i = i + 1;
  }
  return i;
}

# Python's // floors; GSL-2's / truncates.  The whole platform's arithmetic
# oracle is the interpreter, so every division here has to floor.
fn fdiv(a, b) {
  var q = a / b;
  var r = a % b;
  if (r != 0) {
    if ((r < 0) != (b < 0)) {
      q = q - 1;
    }
  }
  return q;
}

fn text_eq(a, alen, b, blen) {
  if (alen != blen) {
    return 0;
  }
  var i = 0;
  while (i < alen) {
    if (mem[a + i] != mem[b + i]) {
      return 0;
    }
    i = i + 1;
  }
  return 1;
}

fn lex_is(start, length, word) {
  return text_eq(start, length, word, strlen(word));
}

fn is_alpha(c) {
  if (c >= 97 && c <= 122) {
    return 1;
  }
  if (c >= 65 && c <= 90) {
    return 1;
  }
  if (c == 95) {
    return 1;
  }
  return 0;
}

fn is_digit(c) {
  if (c >= 48 && c <= 57) {
    return 1;
  }
  return 0;
}

fn is_alnum(c) {
  if (is_alpha(c)) {
    return 1;
  }
  return is_digit(c);
}

fn is_space(c) {
  if (c == 32 || c == 9 || c == 13) {
    return 1;
  }
  return 0;
}

fn read_stdin() {
  var i = 0;
  var c = getchar();
  while (c >= 0) {
    mem[SRC + i] = c;
    i = i + 1;
    c = getchar();
  }
  mem[SRC + i] = 0;
  srclen = i;
  return 0;
}

# ----------------------------------------------------------------------
# layer 3: preprocessing
# ----------------------------------------------------------------------

fn macro_find(start, length) {
  var i = 0;
  while (i < nmacro) {
    if (mem[MACNL + i] == length) {
      if (text_eq(mem[MACNM + i], length, start, length)) {
        return i;
      }
    }
    i = i + 1;
  }
  return 0 - 1;
}

fn mac_intern(start, length) {
  var at = mactop;
  var i = 0;
  while (i < length) {
    mem[MACT + at + i] = mem[start + i];
    i = i + 1;
  }
  mactop = mactop + length;
  return at + MACT;
}

fn macro_define(nstart, nlen, vstart, vlen) {
  var slot = macro_find(nstart, nlen);
  if (slot < 0) {
    if (nmacro >= MAXMACRO) {
      fail("too many macros");
    }
    slot = nmacro;
    nmacro = nmacro + 1;
  }
  mem[MACNM + slot] = mac_intern(nstart, nlen);
  mem[MACNL + slot] = nlen;
  mem[MACVO + slot] = mac_intern(vstart, vlen);
  mem[MACVL + slot] = vlen;
  return 0;
}

# The span [start, start + length) with leading and trailing whitespace
# removed; the trimmed start is returned and the length written back through
# the one-cell cell at TRIM.
var TRIM = 660950;

fn trim(start, length) {
  var b = start;
  var e = start + length;
  while (b < e && is_space(mem[b])) {
    b = b + 1;
  }
  while (e > b && is_space(mem[e - 1])) {
    e = e - 1;
  }
  mem[TRIM] = e - b;
  return b;
}

fn directive_define(bstart, blen) {
  var s = trim(bstart, blen);
  var n = mem[TRIM];
  if (n == 0 || is_alpha(mem[s]) == 0) {
    fail("malformed #define");
  }
  var k = 0;
  while (k < n && is_alnum(mem[s + k])) {
    k = k + 1;
  }
  if (k >= n || is_space(mem[s + k]) == 0) {
    fail("malformed #define");
  }
  var v = trim(s + k, n - k);
  macro_define(s, k, v, mem[TRIM]);
  return 0;
}

fn directive_undef(bstart, blen) {
  var s = trim(bstart, blen);
  var slot = macro_find(s, mem[TRIM]);
  if (slot >= 0) {
    mem[MACNL + slot] = 0;
  }
  return 0;
}

# Recognises ^\s*#\s*([a-z]+)\s*(.*)$ over the logical line held in LINEB.
# Returns 0 when the line is not a directive, 1 when it was one and handled.
fn directive(length) {
  var j = 0;
  while (j < length && is_space(mem[LINEB + j])) {
    j = j + 1;
  }
  if (j >= length || mem[LINEB + j] != 35) {
    return 0;
  }
  j = j + 1;
  while (j < length && is_space(mem[LINEB + j])) {
    j = j + 1;
  }
  var k = j;
  while (k < length && mem[LINEB + k] >= 97 && mem[LINEB + k] <= 122) {
    k = k + 1;
  }
  if (k == j) {
    return 0;
  }
  var dstart = LINEB + j;
  var dlen = k - j;
  while (k < length && is_space(mem[LINEB + k])) {
    k = k + 1;
  }
  if (lex_is(dstart, dlen, "define")) {
    directive_define(LINEB + k, length - k);
    return 1;
  }
  if (lex_is(dstart, dlen, "pragma")) {
    return 1;
  }
  if (lex_is(dstart, dlen, "undef")) {
    directive_undef(LINEB + k, length - k);
    return 1;
  }
  fail("unknown preprocessor directive");
  return 1;
}

# One substitution sweep of LINEB into EXPB: every maximal run of word
# characters that begins with a letter and names a macro is replaced by that
# macro's replacement list.  Returns the length written.
fn expand_once(length) {
  var i = 0;
  var out = 0;
  while (i < length) {
    var c = mem[LINEB + i];
    if (is_alnum(c)) {
      var j = i;
      while (j < length && is_alnum(mem[LINEB + j])) {
        j = j + 1;
      }
      var slot = 0 - 1;
      if (is_alpha(c)) {
        slot = macro_find(LINEB + i, j - i);
      }
      if (slot >= 0) {
        var k = 0;
        while (k < mem[MACVL + slot]) {
          mem[EXPB + out] = mem[mem[MACVO + slot] + k];
          out = out + 1;
          k = k + 1;
        }
      } else {
        var k2 = i;
        while (k2 < j) {
          mem[EXPB + out] = mem[LINEB + k2];
          out = out + 1;
          k2 = k2 + 1;
        }
      }
      i = j;
    } else {
      mem[EXPB + out] = c;
      out = out + 1;
      i = i + 1;
    }
  }
  return out;
}

fn expand(length) {
  var round = 0;
  while (round < 32) {
    var out = expand_once(length);
    if (out == length && text_eq(LINEB, length, EXPB, out)) {
      return length;
    }
    var i = 0;
    while (i < out) {
      mem[LINEB + i] = mem[EXPB + i];
      i = i + 1;
    }
    length = out;
    round = round + 1;
  }
  fail("macro expansion did not converge");
  return 0;
}

fn pps_append(length) {
  if (ppslen > 0) {
    mem[PPS + ppslen] = 10;
    ppslen = ppslen + 1;
  }
  var i = 0;
  while (i < length) {
    mem[PPS + ppslen] = mem[LINEB + i];
    ppslen = ppslen + 1;
    i = i + 1;
  }
  return 0;
}

fn emit_logical(length) {
  if (directive(length) == 1) {
    return 0;
  }
  pps_append(expand(length));
  return 0;
}

# Line splicing, then directives, then expansion.  A line whose last
# non-blank character is a backslash is joined with the one after it, the
# backslash and the trailing blanks going away with it.
fn preprocess() {
  var i = 0;
  var held = 0;
  while (i < srclen) {
    var b = i;
    while (i < srclen && mem[SRC + i] != 10) {
      i = i + 1;
    }
    var e = i;
    if (i < srclen) {
      i = i + 1;
    }
    var r = e;
    while (r > b && is_space(mem[SRC + r - 1])) {
      r = r - 1;
    }
    var spliced = 0;
    if (r > b && mem[SRC + r - 1] == 92) {
      spliced = 1;
      e = r - 1;
    }
    var k = b;
    while (k < e) {
      mem[LINEB + held] = mem[SRC + k];
      held = held + 1;
      k = k + 1;
    }
    if (spliced == 0) {
      emit_logical(held);
      held = 0;
    }
  }
  if (held > 0) {
    emit_logical(held);
  }
  mem[PPS + ppslen] = 0;
  return 0;
}

# ----------------------------------------------------------------------
# layer 4: lexical analysis
# ----------------------------------------------------------------------
#
# The same table-driven transducer, unrolled into control flow: five states,
# maximal munch, and a rescan whenever an accepting state meets a character it
# cannot extend with.

var ST_GROUND = 0;
var ST_IDENT = 1;
var ST_NUM = 2;
var ST_RANGE = 3;
var ST_COMMENT = 4;

fn is_keyword(start, length) {
  if (lex_is(start, length, "lattice")) {
    return 1;
  }
  if (lex_is(start, length, "order")) {
    return 1;
  }
  if (lex_is(start, length, "symmetry")) {
    return 1;
  }
  if (lex_is(start, length, "cyclic")) {
    return 1;
  }
  if (lex_is(start, length, "dihedral")) {
    return 1;
  }
  if (lex_is(start, length, "about")) {
    return 1;
  }
  if (lex_is(start, length, "centroid")) {
    return 1;
  }
  if (lex_is(start, length, "stroke")) {
    return 1;
  }
  if (lex_is(start, length, "emit")) {
    return 1;
  }
  if (lex_is(start, length, "paint")) {
    return 1;
  }
  if (lex_is(start, length, "let")) {
    return 1;
  }
  if (lex_is(start, length, "for")) {
    return 1;
  }
  if (lex_is(start, length, "in")) {
    return 1;
  }
  if (lex_is(start, length, "at")) {
    return 1;
  }
  if (lex_is(start, length, "span")) {
    return 1;
  }
  if (lex_is(start, length, "row")) {
    return 1;
  }
  if (lex_is(start, length, "column")) {
    return 1;
  }
  if (lex_is(start, length, "diagonal")) {
    return 1;
  }
  if (lex_is(start, length, "antidiagonal")) {
    return 1;
  }
  return 0;
}

fn punctuation_kind(c) {
  if (c == 43) {
    return TK_PLUS;
  }
  if (c == 45) {
    return TK_MINUS;
  }
  if (c == 42) {
    return TK_STAR;
  }
  if (c == 47) {
    return TK_SLASH;
  }
  if (c == 61) {
    return TK_EQUALS;
  }
  if (c == 59) {
    return TK_SEMI;
  }
  if (c == 40) {
    return TK_LPAREN;
  }
  if (c == 41) {
    return TK_RPAREN;
  }
  if (c == 123) {
    return TK_LBRACE;
  }
  if (c == 125) {
    return TK_RBRACE;
  }
  return 0 - 1;
}

fn push_token(kind, start, length, value) {
  if (ntok >= MAXTOK) {
    fail("token buffer overflow");
  }
  mem[TOKK + ntok] = kind;
  mem[TOKS + ntok] = start;
  mem[TOKL + ntok] = length;
  mem[TOKV + ntok] = value;
  ntok = ntok + 1;
  return 0;
}

fn digits_value(start, length) {
  var v = 0;
  var i = 0;
  while (i < length) {
    v = v * 10 + mem[start + i] - 48;
    i = i + 1;
  }
  return v;
}

# The accepting states and their classifications.  A run of dots is only a
# range operator when it is exactly two long.
fn flush(state, start, length) {
  if (length == 0) {
    return 0;
  }
  if (state == ST_IDENT) {
    if (is_keyword(start, length)) {
      push_token(TK_KW, start, length, 0);
    } else {
      push_token(TK_IDENT, start, length, 0);
    }
    return 0;
  }
  if (state == ST_NUM) {
    push_token(TK_INT, start, length, digits_value(start, length));
    return 0;
  }
  if (state == ST_RANGE) {
    if (length != 2) {
      fail("malformed range operator");
    }
    push_token(TK_RANGE, start, length, 0);
    return 0;
  }
  fail("non-accepting transducer state");
  return 0;
}

fn tokenize() {
  var state = ST_GROUND;
  var start = 0;
  var length = 0;
  var i = 0;
  ntok = 0;
  while (i < ppslen) {
    var c = mem[PPS + i];
    if (state == ST_COMMENT) {
      if (c == 10) {
        state = ST_GROUND;
      }
      i = i + 1;
    } else {
      if (state == ST_IDENT && is_alnum(c)) {
        length = length + 1;
        i = i + 1;
      } else {
        if (state == ST_NUM && is_digit(c)) {
          length = length + 1;
          i = i + 1;
        } else {
          if (state == ST_RANGE && c == 46) {
            length = length + 1;
            i = i + 1;
          } else {
            if (state != ST_GROUND) {
              flush(state, start, length);
              length = 0;
              state = ST_GROUND;
            } else {
              if (is_alpha(c)) {
                state = ST_IDENT;
                start = PPS + i;
                length = 1;
                i = i + 1;
              } else {
                if (is_digit(c)) {
                  state = ST_NUM;
                  start = PPS + i;
                  length = 1;
                  i = i + 1;
                } else {
                  if (c == 46) {
                    state = ST_RANGE;
                    start = PPS + i;
                    length = 1;
                    i = i + 1;
                  } else {
                    if (c == 35) {
                      state = ST_COMMENT;
                      i = i + 1;
                    } else {
                      if (c == 10 || is_space(c)) {
                        i = i + 1;
                      } else {
                        var kind = punctuation_kind(c);
                        if (kind < 0) {
                          fail("illegal character in the source text");
                        }
                        push_token(kind, PPS + i, 1, 0);
                        i = i + 1;
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  flush(state, start, length);
  push_token(TK_EOI, PPS + ppslen, 0, 0);
  return 0;
}

# ----------------------------------------------------------------------
# layer 6: scopes and symbols
# ----------------------------------------------------------------------

fn sym_declare(start, length, kind, slot) {
  var i = nsym;
  while (i > 0 && mem[SYMD + i - 1] == depth) {
    i = i - 1;
    if (text_eq(mem[SYMN + i], mem[SYML + i], start, length)) {
      fail("symbol already declared in this scope");
    }
  }
  if (nsym >= MAXSYM) {
    fail("symbol table overflow");
  }
  mem[SYMN + nsym] = start;
  mem[SYML + nsym] = length;
  mem[SYMK + nsym] = kind;
  mem[SYMSL + nsym] = slot;
  mem[SYMD + nsym] = depth;
  mem[SYMR0 + nsym] = 0;
  mem[SYMR1 + nsym] = 0;
  nsym = nsym + 1;
  return nsym - 1;
}

fn sym_resolve(start, length) {
  var i = nsym;
  while (i > 0) {
    i = i - 1;
    if (text_eq(mem[SYMN + i], mem[SYML + i], start, length)) {
      return i;
    }
  }
  return 0 - 1;
}

fn scope_leave(mark) {
  nsym = mark;
  depth = depth - 1;
  return 0;
}

fn alloc_slot() {
  nslot = nslot + 1;
  return nslot - 1;
}

fn new_label() {
  nlabel = nlabel + 1;
  return nlabel - 1;
}

# ----------------------------------------------------------------------
# layer 7: the symbolic listing
# ----------------------------------------------------------------------

fn emit_item(kind, arg) {
  if (nitem >= MAXITEM) {
    fail("listing overflow");
  }
  mem[ITK + nitem] = kind;
  mem[ITA + nitem] = arg;
  nitem = nitem + 1;
  return 0;
}

# Strokes are lowered where they are declared, so that the names inside them
# resolve in the scope that declared them, and the resulting run of items is
# lifted out of the listing and replayed at every emission.  The same device
# holds a loop's upper bound, which is parsed before the induction variable
# exists but emitted after it.
fn capture_take(mark) {
  var at = stroketop;
  var i = mark;
  while (i < nitem) {
    mem[STK + stroketop] = mem[ITK + i];
    mem[STA + stroketop] = mem[ITA + i];
    stroketop = stroketop + 1;
    i = i + 1;
  }
  nitem = mark;
  return at;
}

fn splice(from, to) {
  var i = from;
  while (i < to) {
    emit_item(mem[STK + i], mem[STA + i]);
    i = i + 1;
  }
  return 0;
}

# ----------------------------------------------------------------------
# layers 5 and 6: recursive descent, precedence climbing, and lowering
# ----------------------------------------------------------------------
#
# One pass does the work of three.  The analyser and the emitter never
# disagree about a name because they are the same traversal, and the only
# places where the two orders differ - a stroke's run, a loop's upper bound -
# are captured and replayed rather than revisited.

fn tkind() {
  return mem[TOKK + tp];
}

fn tstart() {
  return mem[TOKS + tp];
}

fn tlen() {
  return mem[TOKL + tp];
}

fn tval() {
  return mem[TOKV + tp];
}

fn advance() {
  if (mem[TOKK + tp] != TK_EOI) {
    tp = tp + 1;
  }
  return 0;
}

fn at_kw(word) {
  if (mem[TOKK + tp] != TK_KW) {
    return 0;
  }
  return lex_is(mem[TOKS + tp], mem[TOKL + tp], word);
}

fn expect(kind, what) {
  if (mem[TOKK + tp] != kind) {
    fail(what);
  }
  advance();
  return 0;
}

fn expect_kw(word, what) {
  if (at_kw(word) == 0) {
    fail(what);
  }
  advance();
  return 0;
}

fn infix_precedence(kind) {
  if (kind == TK_PLUS || kind == TK_MINUS) {
    return 10;
  }
  if (kind == TK_STAR || kind == TK_SLASH) {
    return 20;
  }
  return 0 - 1;
}

fn infix_opcode(kind) {
  if (kind == TK_PLUS) {
    return OP_ADD;
  }
  if (kind == TK_MINUS) {
    return OP_SUB;
  }
  if (kind == TK_STAR) {
    return OP_MUL;
  }
  return OP_DIV;
}

fn intrinsic_index(start, length) {
  if (lex_is(start, length, "zero")) {
    return 0;
  }
  if (lex_is(start, length, "apothem")) {
    return 1;
  }
  if (lex_is(start, length, "extremum")) {
    return 2;
  }
  return 3;
}

# ---- the closed expression language, evaluated rather than lowered -----

fn const_prefix() {
  if (tkind() == TK_INT) {
    var v = tval();
    advance();
    return v;
  }
  if (tkind() == TK_MINUS) {
    advance();
    return 0 - const_prefix();
  }
  if (tkind() == TK_LPAREN) {
    advance();
    var inner = const_expr(0);
    expect(TK_RPAREN, "expected )");
    return inner;
  }
  fail("this is not a constant expression");
  return 0;
}

fn const_expr(minimum) {
  var left = const_prefix();
  while (1) {
    var precedence = infix_precedence(tkind());
    if (precedence < 0 || precedence < minimum) {
      return left;
    }
    var kind = tkind();
    advance();
    var right = const_expr(precedence + 1);
    if (kind == TK_PLUS) {
      left = left + right;
    } else {
      if (kind == TK_MINUS) {
        left = left - right;
      } else {
        if (kind == TK_STAR) {
          left = left * right;
        } else {
          left = fdiv(left, right);
        }
      }
    }
  }
  return left;
}

# ---- the open expression language, lowered to the operand stack --------

fn parse_prefix() {
  if (tkind() == TK_INT) {
    emit_item(OP_PUSH, tval());
    advance();
    return 0;
  }
  if (tkind() == TK_IDENT) {
    var slot = sym_resolve(tstart(), tlen());
    if (slot < 0) {
      fail("unknown symbol");
    }
    if (mem[SYMK + slot] == SK_STROKE) {
      fail("strokes are not first-class values");
    }
    if (mem[SYMK + slot] == SK_INTRINSIC) {
      emit_item(OP_INTR, mem[SYMSL + slot]);
    } else {
      emit_item(OP_LOADL, mem[SYMSL + slot]);
    }
    advance();
    return 0;
  }
  if (tkind() == TK_MINUS) {
    advance();
    parse_prefix();
    emit_item(OP_NEG, 0);
    return 0;
  }
  if (tkind() == TK_LPAREN) {
    advance();
    parse_expr(0);
    expect(TK_RPAREN, "expected )");
    return 0;
  }
  fail("expected an expression");
  return 0;
}

fn parse_expr(minimum) {
  parse_prefix();
  while (1) {
    var precedence = infix_precedence(tkind());
    if (precedence < 0 || precedence < minimum) {
      return 0;
    }
    var kind = tkind();
    advance();
    parse_expr(precedence + 1);
    emit_item(infix_opcode(kind), 0);
  }
  return 0;
}

# ---- runs, statements, declarations ------------------------------------

fn orientation_code() {
  if (at_kw("row")) {
    return 0;
  }
  if (at_kw("column")) {
    return 1;
  }
  if (at_kw("diagonal")) {
    return 2;
  }
  if (at_kw("antidiagonal")) {
    return 3;
  }
  return 0 - 1;
}

fn parse_run() {
  var orientation = orientation_code();
  if (orientation < 0) {
    fail("unknown orientation");
  }
  advance();
  expect_kw("at", "expected at");
  parse_expr(0);
  expect_kw("span", "expected span");
  parse_expr(0);
  expect(TK_RANGE, "expected ..");
  parse_expr(0);
  emit_item(OP_MKIV, 0);
  emit_item(OP_EMIT, orientation);
  return 0;
}

fn parse_binding() {
  advance();
  if (tkind() != TK_IDENT) {
    fail("expected a name after let");
  }
  var start = tstart();
  var length = tlen();
  advance();
  expect(TK_EQUALS, "expected =");
  parse_expr(0);
  expect(TK_SEMI, "expected ;");
  var slot = alloc_slot();
  sym_declare(start, length, SK_BINDING, slot);
  emit_item(OP_STOREL, slot);
  return 0;
}

fn parse_stroke() {
  advance();
  if (tkind() != TK_IDENT) {
    fail("expected a name after stroke");
  }
  var start = tstart();
  var length = tlen();
  advance();
  expect(TK_EQUALS, "expected =");
  var mark = nitem;
  parse_run();
  expect(TK_SEMI, "expected ;");
  var at = capture_take(mark);
  var slot = sym_declare(start, length, SK_STROKE, 0 - 1);
  mem[SYMR0 + slot] = at;
  mem[SYMR1 + slot] = stroketop;
  return 0;
}

fn parse_emission() {
  advance();
  if (tkind() != TK_IDENT) {
    fail("expected a stroke name after emit");
  }
  var slot = sym_resolve(tstart(), tlen());
  if (slot < 0) {
    fail("unknown symbol");
  }
  if (mem[SYMK + slot] != SK_STROKE) {
    fail("that name is not a stroke");
  }
  advance();
  expect(TK_SEMI, "expected ;");
  splice(mem[SYMR0 + slot], mem[SYMR1 + slot]);
  return 0;
}

fn parse_painting() {
  advance();
  parse_run();
  expect(TK_SEMI, "expected ;");
  return 0;
}

fn parse_iteration() {
  advance();
  if (tkind() != TK_IDENT) {
    fail("expected a name after for");
  }
  var start = tstart();
  var length = tlen();
  advance();
  var head = new_label();
  var done = new_label();
  expect_kw("in", "expected in");
  parse_expr(0);
  expect(TK_RANGE, "expected ..");
  var mark = nitem;
  parse_expr(0);
  var upper = capture_take(mark);
  var top = stroketop;
  expect(TK_LBRACE, "expected {");
  var slot = alloc_slot();
  depth = depth + 1;
  var outer = nsym;
  sym_declare(start, length, SK_INDUCTION, slot);
  emit_item(OP_STOREL, slot);
  emit_item(OP_LABEL, head);
  emit_item(OP_LOADL, slot);
  splice(upper, top);
  emit_item(OP_CMPLE, 0);
  emit_item(OP_JF, done);
  parse_body(TK_RBRACE);
  expect(TK_RBRACE, "expected }");
  emit_item(OP_LOADL, slot);
  emit_item(OP_PUSH, 1);
  emit_item(OP_ADD, 0);
  emit_item(OP_STOREL, slot);
  emit_item(OP_JMP, head);
  emit_item(OP_LABEL, done);
  scope_leave(outer);
  return 0;
}

fn parse_statement() {
  if (at_kw("let")) {
    return parse_binding();
  }
  if (at_kw("stroke")) {
    return parse_stroke();
  }
  if (at_kw("emit")) {
    return parse_emission();
  }
  if (at_kw("paint")) {
    return parse_painting();
  }
  if (at_kw("for")) {
    return parse_iteration();
  }
  fail("that token cannot begin a statement");
  return 0;
}

fn parse_body(terminator) {
  while (tkind() != terminator) {
    parse_statement();
  }
  return 0;
}

fn parse_lattice() {
  expect_kw("lattice", "expected lattice");
  expect_kw("order", "expected order");
  ORDER = const_expr(0);
  expect(TK_SEMI, "expected ;");
  if (ORDER <= 0 || ORDER % 2 == 0) {
    fail("illegal lattice order");
  }
  APOTHEM = ORDER / 2;
  EXTREMUM = ORDER - 1;
  CELLS = ORDER * ORDER;
  return 0;
}

fn parse_symmetry() {
  expect_kw("symmetry", "expected symmetry");
  if (at_kw("cyclic")) {
    family = 0;
  } else {
    if (at_kw("dihedral")) {
      family = 1;
    } else {
      fail("unknown symmetry family");
    }
  }
  advance();
  if (tkind() != TK_INT) {
    fail("expected the symmetry cardinality");
  }
  cardinality = tval();
  advance();
  expect_kw("about", "expected about");
  expect_kw("centroid", "expected centroid");
  expect(TK_SEMI, "expected ;");
  if (cardinality != 4) {
    fail("this platform only implements 4-fold symmetry");
  }
  return 0;
}

fn declare_one_intrinsic(name, index) {
  sym_declare(name, strlen(name), SK_INTRINSIC, index);
  return 0;
}

fn declare_intrinsics() {
  declare_one_intrinsic("zero", 0);
  declare_one_intrinsic("apothem", 1);
  declare_one_intrinsic("extremum", 2);
  declare_one_intrinsic("magnitude", 3);
  return 0;
}

fn parse_program() {
  tp = 0;
  parse_lattice();
  parse_symmetry();
  declare_intrinsics();
  parse_body(TK_EOI);
  expect(TK_EOI, "expected the end of the program");
  emit_item(OP_CLOSE, 0);
  emit_item(OP_HALT, 0);
  frame = nslot;
  if (frame < 1) {
    frame = 1;
  }
  return 0;
}

# ----------------------------------------------------------------------
# layer 8: the pass pipeline
# ----------------------------------------------------------------------
#
# Five peephole rewrites, run round after round until a round changes
# nothing, or until patience runs out at eight.

fn is_foldable(kind) {
  if (kind == OP_ADD || kind == OP_SUB || kind == OP_MUL || kind == OP_DIV) {
    return 1;
  }
  return 0;
}

fn fold(kind, left, right) {
  if (kind == OP_ADD) {
    return left + right;
  }
  if (kind == OP_SUB) {
    return left - right;
  }
  if (kind == OP_MUL) {
    return left * right;
  }
  return fdiv(left, right);
}

fn keep(out, index) {
  mem[OTK + out] = mem[ITK + index];
  mem[OTA + out] = mem[ITA + index];
  return out + 1;
}

fn commit(out) {
  var i = 0;
  while (i < out) {
    mem[ITK + i] = mem[OTK + i];
    mem[ITA + i] = mem[OTA + i];
    i = i + 1;
  }
  nitem = out;
  return 0;
}

fn pass_constant_folding() {
  var i = 0;
  var out = 0;
  while (i < nitem) {
    var folded = 0;
    if (i + 2 < nitem && mem[ITK + i] == OP_PUSH && mem[ITK + i + 1] == OP_PUSH) {
      if (is_foldable(mem[ITK + i + 2])) {
        if (mem[ITK + i + 2] != OP_DIV || mem[ITA + i + 1] != 0) {
          mem[OTK + out] = OP_PUSH;
          mem[OTA + out] = fold(mem[ITK + i + 2], mem[ITA + i], mem[ITA + i + 1]);
          out = out + 1;
          i = i + 3;
          folded = 1;
        }
      }
    }
    if (folded == 0) {
      if (i + 1 < nitem && mem[ITK + i] == OP_PUSH && mem[ITK + i + 1] == OP_NEG) {
        mem[OTK + out] = OP_PUSH;
        mem[OTA + out] = 0 - mem[ITA + i];
        out = out + 1;
        i = i + 2;
      } else {
        out = keep(out, i);
        i = i + 1;
      }
    }
  }
  return commit(out);
}

fn is_neutral(value, kind) {
  if (value == 0 && (kind == OP_ADD || kind == OP_SUB)) {
    return 1;
  }
  if (value == 1 && (kind == OP_MUL || kind == OP_DIV)) {
    return 1;
  }
  return 0;
}

fn pass_algebraic_identities() {
  var i = 0;
  var out = 0;
  while (i < nitem) {
    if (i + 1 < nitem && mem[ITK + i] == OP_PUSH
        && is_neutral(mem[ITA + i], mem[ITK + i + 1])) {
      i = i + 2;
    } else {
      out = keep(out, i);
      i = i + 1;
    }
  }
  return commit(out);
}

fn pass_redundant_branch_elimination() {
  var i = 0;
  var out = 0;
  while (i < nitem) {
    if (mem[ITK + i] == OP_JMP && i + 1 < nitem && mem[ITK + i + 1] == OP_LABEL
        && mem[ITA + i + 1] == mem[ITA + i]) {
      i = i + 1;
    } else {
      out = keep(out, i);
      i = i + 1;
    }
  }
  return commit(out);
}

fn pass_unreachable_code_elimination() {
  var i = 0;
  var out = 0;
  var live = 1;
  while (i < nitem) {
    if (mem[ITK + i] == OP_LABEL) {
      live = 1;
      out = keep(out, i);
    } else {
      if (live == 1) {
        out = keep(out, i);
        if (mem[ITK + i] == OP_JMP || mem[ITK + i] == OP_HALT) {
          live = 0;
        }
      }
    }
    i = i + 1;
  }
  return commit(out);
}

fn pass_dead_label_elimination() {
  var i = 0;
  while (i < nlabel) {
    mem[REFL + i] = 0;
    i = i + 1;
  }
  i = 0;
  while (i < nitem) {
    if (mem[ITK + i] == OP_JMP || mem[ITK + i] == OP_JF) {
      mem[REFL + mem[ITA + i]] = 1;
    }
    i = i + 1;
  }
  i = 0;
  var out = 0;
  while (i < nitem) {
    if (mem[ITK + i] == OP_LABEL && mem[REFL + mem[ITA + i]] == 0) {
      i = i + 1;
    } else {
      out = keep(out, i);
      i = i + 1;
    }
  }
  return commit(out);
}

fn optimise() {
  var round = 0;
  while (round < 8) {
    var before = nitem;
    pass_constant_folding();
    pass_algebraic_identities();
    pass_redundant_branch_elimination();
    pass_unreachable_code_elimination();
    pass_dead_label_elimination();
    if (nitem == before) {
      return 0;
    }
    round = round + 1;
  }
  return 0;
}

# ----------------------------------------------------------------------
# layer 9: assembly
# ----------------------------------------------------------------------

fn assemble() {
  var address = 0;
  var i = 0;
  while (i < nitem) {
    if (mem[ITK + i] == OP_LABEL) {
      mem[LBLA + mem[ITA + i]] = address;
    } else {
      address = address + 1;
    }
    i = i + 1;
  }
  ncode = 0;
  i = 0;
  while (i < nitem) {
    if (mem[ITK + i] != OP_LABEL) {
      mem[CODEK + ncode] = mem[ITK + i];
      if (mem[ITK + i] == OP_JMP || mem[ITK + i] == OP_JF) {
        mem[CODEA + ncode] = mem[LBLA + mem[ITA + i]];
      } else {
        mem[CODEA + ncode] = mem[ITA + i];
      }
      ncode = ncode + 1;
    }
    i = i + 1;
  }
  if (ncode == 0 || mem[CODEK + ncode - 1] != OP_HALT) {
    fail("object module does not terminate in a halt");
  }
  return 0;
}

# ----------------------------------------------------------------------
# layer 2: the symmetry group
# ----------------------------------------------------------------------

fn group_has(a, b, c, d) {
  var i = 0;
  while (i < ngroup) {
    if (mem[GRPA + i] == a && mem[GRPB + i] == b) {
      if (mem[GRPC + i] == c && mem[GRPD + i] == d) {
        return 1;
      }
    }
    i = i + 1;
  }
  return 0;
}

fn group_add(a, b, c, d) {
  if (group_has(a, b, c, d)) {
    return 0;
  }
  mem[GRPA + ngroup] = a;
  mem[GRPB + ngroup] = b;
  mem[GRPC + ngroup] = c;
  mem[GRPD + ngroup] = d;
  ngroup = ngroup + 1;
  return 1;
}

fn group_product(i, j) {
  group_add(mem[GRPA + i] * mem[GRPA + j] + mem[GRPB + i] * mem[GRPC + j],
            mem[GRPA + i] * mem[GRPB + j] + mem[GRPB + i] * mem[GRPD + j],
            mem[GRPC + i] * mem[GRPA + j] + mem[GRPD + i] * mem[GRPC + j],
            mem[GRPC + i] * mem[GRPB + j] + mem[GRPD + i] * mem[GRPD + j]);
  return 0;
}

fn group_precedes(i, j) {
  if (mem[GRPA + i] != mem[GRPA + j]) {
    return mem[GRPA + i] < mem[GRPA + j];
  }
  if (mem[GRPB + i] != mem[GRPB + j]) {
    return mem[GRPB + i] < mem[GRPB + j];
  }
  if (mem[GRPC + i] != mem[GRPC + j]) {
    return mem[GRPC + i] < mem[GRPC + j];
  }
  return mem[GRPD + i] < mem[GRPD + j];
}

fn group_swap(i, j) {
  var t = mem[GRPA + i];
  mem[GRPA + i] = mem[GRPA + j];
  mem[GRPA + j] = t;
  t = mem[GRPB + i];
  mem[GRPB + i] = mem[GRPB + j];
  mem[GRPB + j] = t;
  t = mem[GRPC + i];
  mem[GRPC + i] = mem[GRPC + j];
  mem[GRPC + j] = t;
  t = mem[GRPD + i];
  mem[GRPD + i] = mem[GRPD + j];
  mem[GRPD + j] = t;
  return 0;
}

# Closure of the generators under composition, then the total order the
# backend iterates the elements in.  The quarter turn generates C4; adding the
# mirror generates the dihedral group of twice the order.
fn group_build() {
  ngroup = 0;
  group_add(1, 0, 0, 1);
  group_add(0, 0 - 1, 1, 0);
  if (family == 1) {
    group_add(1, 0, 0, 0 - 1);
  }
  var changed = 1;
  while (changed == 1) {
    changed = 0;
    var i = 0;
    while (i < ngroup) {
      var j = 0;
      while (j < ngroup) {
        var before = ngroup;
        group_product(i, j);
        if (ngroup != before) {
          changed = 1;
        }
        j = j + 1;
      }
      i = i + 1;
    }
  }
  var expected = cardinality;
  if (family == 1) {
    expected = 2 * cardinality;
  }
  if (ngroup != expected) {
    fail("the generated group has the wrong order");
  }
  var p = 1;
  while (p < ngroup) {
    var q = p;
    while (q > 0 && group_precedes(q, q - 1)) {
      group_swap(q, q - 1);
      q = q - 1;
    }
    p = p + 1;
  }
  return 0;
}

# ----------------------------------------------------------------------
# the runtime the lowered machine calls into, emitted verbatim
# ----------------------------------------------------------------------

fn emit_runtime_0() {
  es("\n");
  es("@canvas = internal global [");
  en(CELLS);
  es(" x i8] zeroinitializer\n");
  es("@snapshot = internal global [");
  en(CELLS);
  es(" x i8] zeroinitializer\n");
  es("\n");
  es("declare i32 @putchar(i32)\n");
  es("declare i32 @fflush(ptr)\n");
  es("\n");
  es("define internal void @gvm.paint(i32 %row, i32 %col) {\n");
  es("entry:\n");
  es("  %r0 = icmp sge i32 %row, 0\n");
  es("  %r1 = icmp slt i32 %row, ");
  en(ORDER);
  es("\n");
  es("  %rok = and i1 %r0, %r1\n");
  es("  %c0 = icmp sge i32 %col, 0\n");
  es("  %c1 = icmp slt i32 %col, ");
  en(ORDER);
  es("\n");
  es("  %cok = and i1 %c0, %c1\n");
  es("  %ok = and i1 %rok, %cok\n");
  es("  br i1 %ok, label %store, label %done\n");
  es("\n");
  es("store:\n");
  es("  %o0 = mul nsw i32 %row, ");
  en(ORDER);
  es("\n");
  return 0;
}

fn emit_runtime_1() {
  es("  %o1 = add nsw i32 %o0, %col\n");
  es("  %o2 = sext i32 %o1 to i64\n");
  es("  %slot = getelementptr inbounds [");
  en(CELLS);
  es(" x i8], ptr @canvas, i64 0, i64 %o2\n");
  es("  store i8 1, ptr %slot, align 1\n");
  es("  br label %done\n");
  es("\n");
  es("done:\n");
  es("  ret void\n");
  es("}\n");
  es("\n");
  es("define internal void @gvm.emit_run(i32 %orient, i32 %index, i32 %lo, i32 %hi) {\n");
  es("entry:\n");
  es("  %cursor = alloca i32, align 4\n");
  es("  store i32 %lo, ptr %cursor, align 4\n");
  es("  br label %head\n");
  es("\n");
  es("head:\n");
  es("  %cur = load i32, ptr %cursor, align 4\n");
  es("  %more = icmp sle i32 %cur, %hi\n");
  es("  br i1 %more, label %body, label %exit\n");
  es("\n");
  es("body:\n");
  es("  %scalar = load i32, ptr %cursor, align 4\n");
  es("  switch i32 %orient, label %o.row [ i32 1, label %o.column\n");
  es("                                     i32 2, label %o.diagonal\n");
  es("                                     i32 3, label %o.antidiagonal ]\n");
  es("\n");
  es("o.row:\n");
  return 0;
}

fn emit_runtime_2() {
  es("  call void @gvm.paint(i32 %index, i32 %scalar)\n");
  es("  br label %step\n");
  es("\n");
  es("o.column:\n");
  es("  call void @gvm.paint(i32 %scalar, i32 %index)\n");
  es("  br label %step\n");
  es("\n");
  es("o.diagonal:\n");
  es("  %d.col = add nsw i32 %scalar, %index\n");
  es("  call void @gvm.paint(i32 %scalar, i32 %d.col)\n");
  es("  br label %step\n");
  es("\n");
  es("o.antidiagonal:\n");
  es("  %a.col = sub nsw i32 %index, %scalar\n");
  es("  call void @gvm.paint(i32 %scalar, i32 %a.col)\n");
  es("  br label %step\n");
  es("\n");
  es("step:\n");
  es("  %now = load i32, ptr %cursor, align 4\n");
  es("  %next = add nsw i32 %now, 1\n");
  es("  store i32 %next, ptr %cursor, align 4\n");
  es("  br label %head\n");
  es("\n");
  es("exit:\n");
  es("  ret void\n");
  es("}\n");
  es("\n");
  es("define internal void @gvm.apply(i32 %a, i32 %b, i32 %c, i32 %d) {\n");
  es("entry:\n");
  es("  %row = alloca i32, align 4\n");
  return 0;
}

fn emit_runtime_3() {
  es("  %col = alloca i32, align 4\n");
  es("  store i32 0, ptr %row, align 4\n");
  es("  br label %row.head\n");
  es("\n");
  es("row.head:\n");
  es("  %r = load i32, ptr %row, align 4\n");
  es("  %rmore = icmp slt i32 %r, ");
  en(ORDER);
  es("\n");
  es("  br i1 %rmore, label %row.body, label %row.done\n");
  es("\n");
  es("row.body:\n");
  es("  store i32 0, ptr %col, align 4\n");
  es("  br label %col.head\n");
  es("\n");
  es("col.head:\n");
  es("  %cv = load i32, ptr %col, align 4\n");
  es("  %cmore = icmp slt i32 %cv, ");
  en(ORDER);
  es("\n");
  es("  br i1 %cmore, label %col.body, label %col.done\n");
  es("\n");
  es("col.body:\n");
  es("  %br0 = load i32, ptr %row, align 4\n");
  es("  %bc0 = load i32, ptr %col, align 4\n");
  es("  %q0 = mul nsw i32 %br0, ");
  en(ORDER);
  es("\n");
  es("  %q1 = add nsw i32 %q0, %bc0\n");
  es("  %q2 = sext i32 %q1 to i64\n");
  return 0;
}

fn emit_runtime_4() {
  es("  %src = getelementptr inbounds [");
  en(CELLS);
  es(" x i8], ptr @snapshot, i64 0, i64 %q2\n");
  es("  %ink = load i8, ptr %src, align 1\n");
  es("  %lit = icmp ne i8 %ink, 0\n");
  es("  br i1 %lit, label %col.mark, label %col.step\n");
  es("\n");
  es("col.mark:\n");
  es("  %mr = load i32, ptr %row, align 4\n");
  es("  %mc = load i32, ptr %col, align 4\n");
  es("  %dr = sub nsw i32 %mr, ");
  en(APOTHEM);
  es("\n");
  es("  %dc = sub nsw i32 %mc, ");
  en(APOTHEM);
  es("\n");
  es("  %t0 = mul nsw i32 %a, %dr\n");
  es("  %t1 = mul nsw i32 %b, %dc\n");
  es("  %t2 = add nsw i32 %t0, %t1\n");
  es("  %nr = add nsw i32 %t2, ");
  en(APOTHEM);
  es("\n");
  es("  %t3 = mul nsw i32 %c, %dr\n");
  es("  %t4 = mul nsw i32 %d, %dc\n");
  es("  %t5 = add nsw i32 %t3, %t4\n");
  es("  %nc = add nsw i32 %t5, ");
  en(APOTHEM);
  es("\n");
  es("  call void @gvm.paint(i32 %nr, i32 %nc)\n");
  es("  br label %col.step\n");
  return 0;
}

fn emit_runtime_5() {
  es("\n");
  es("col.step:\n");
  es("  %cnow = load i32, ptr %col, align 4\n");
  es("  %cnext = add nsw i32 %cnow, 1\n");
  es("  store i32 %cnext, ptr %col, align 4\n");
  es("  br label %col.head\n");
  es("\n");
  es("col.done:\n");
  es("  %rnow = load i32, ptr %row, align 4\n");
  es("  %rnext = add nsw i32 %rnow, 1\n");
  es("  store i32 %rnext, ptr %row, align 4\n");
  es("  br label %row.head\n");
  es("\n");
  es("row.done:\n");
  es("  ret void\n");
  es("}\n");
  es("\n");
  es("define internal void @gvm.snapshot() {\n");
  es("entry:\n");
  es("  %i = alloca i32, align 4\n");
  es("  store i32 0, ptr %i, align 4\n");
  es("  br label %head\n");
  es("\n");
  es("head:\n");
  es("  %c = load i32, ptr %i, align 4\n");
  es("  %more = icmp slt i32 %c, ");
  en(CELLS);
  es("\n");
  es("  br i1 %more, label %body, label %done\n");
  es("\n");
  return 0;
}

fn emit_runtime_6() {
  es("body:\n");
  es("  %ix = load i32, ptr %i, align 4\n");
  es("  %ix64 = sext i32 %ix to i64\n");
  es("  %sp = getelementptr inbounds [");
  en(CELLS);
  es(" x i8], ptr @canvas, i64 0, i64 %ix64\n");
  es("  %sv = load i8, ptr %sp, align 1\n");
  es("  %dp = getelementptr inbounds [");
  en(CELLS);
  es(" x i8], ptr @snapshot, i64 0, i64 %ix64\n");
  es("  store i8 %sv, ptr %dp, align 1\n");
  es("  %nx = add nsw i32 %ix, 1\n");
  es("  store i32 %nx, ptr %i, align 4\n");
  es("  br label %head\n");
  es("\n");
  es("done:\n");
  es("  ret void\n");
  es("}\n");
  es("\n");
  es("define internal void @gvm.render() {\n");
  es("entry:\n");
  es("  %row = alloca i32, align 4\n");
  es("  %col = alloca i32, align 4\n");
  es("  %last = alloca i32, align 4\n");
  es("  store i32 0, ptr %row, align 4\n");
  es("  br label %row.head\n");
  es("\n");
  es("row.head:\n");
  es("  %r = load i32, ptr %row, align 4\n");
  es("  %rmore = icmp slt i32 %r, ");
  en(ORDER);
  es("\n");
  return 0;
}

fn emit_runtime_7() {
  es("  br i1 %rmore, label %scan.init, label %row.done\n");
  es("\n");
  es("scan.init:\n");
  es("  store i32 -1, ptr %last, align 4\n");
  es("  store i32 0, ptr %col, align 4\n");
  es("  br label %scan.head\n");
  es("\n");
  es("scan.head:\n");
  es("  %sc = load i32, ptr %col, align 4\n");
  es("  %smore = icmp slt i32 %sc, ");
  en(ORDER);
  es("\n");
  es("  br i1 %smore, label %scan.body, label %scan.exit\n");
  es("\n");
  es("scan.body:\n");
  es("  %sr = load i32, ptr %row, align 4\n");
  es("  %sc2 = load i32, ptr %col, align 4\n");
  es("  %so0 = mul nsw i32 %sr, ");
  en(ORDER);
  es("\n");
  es("  %so1 = add nsw i32 %so0, %sc2\n");
  es("  %so2 = sext i32 %so1 to i64\n");
  es("  %sp = getelementptr inbounds [");
  en(CELLS);
  es(" x i8], ptr @canvas, i64 0, i64 %so2\n");
  es("  %sink = load i8, ptr %sp, align 1\n");
  es("  %slit = icmp ne i8 %sink, 0\n");
  es("  br i1 %slit, label %scan.mark, label %scan.step\n");
  es("\n");
  es("scan.mark:\n");
  return 0;
}

fn emit_runtime_8() {
  es("  %sc3 = load i32, ptr %col, align 4\n");
  es("  store i32 %sc3, ptr %last, align 4\n");
  es("  br label %scan.step\n");
  es("\n");
  es("scan.step:\n");
  es("  %sc4 = load i32, ptr %col, align 4\n");
  es("  %sc5 = add nsw i32 %sc4, 1\n");
  es("  store i32 %sc5, ptr %col, align 4\n");
  es("  br label %scan.head\n");
  es("\n");
  es("scan.exit:\n");
  es("  store i32 0, ptr %col, align 4\n");
  es("  br label %print.head\n");
  es("\n");
  es("print.head:\n");
  es("  %pc = load i32, ptr %col, align 4\n");
  es("  %lastv = load i32, ptr %last, align 4\n");
  es("  %pmore = icmp sle i32 %pc, %lastv\n");
  es("  br i1 %pmore, label %print.body, label %print.newline\n");
  es("\n");
  es("print.body:\n");
  es("  %pc2 = load i32, ptr %col, align 4\n");
  es("  %needsep = icmp sgt i32 %pc2, 0\n");
  es("  br i1 %needsep, label %print.separator, label %print.cell\n");
  es("\n");
  es("print.separator:\n");
  es("  %sepres = call i32 @putchar(i32 32)\n");
  es("  br label %print.cell\n");
  es("\n");
  es("print.cell:\n");
  return 0;
}

fn emit_runtime_9() {
  es("  %pr = load i32, ptr %row, align 4\n");
  es("  %pc3 = load i32, ptr %col, align 4\n");
  es("  %po0 = mul nsw i32 %pr, ");
  en(ORDER);
  es("\n");
  es("  %po1 = add nsw i32 %po0, %pc3\n");
  es("  %po2 = sext i32 %po1 to i64\n");
  es("  %pp = getelementptr inbounds [");
  en(CELLS);
  es(" x i8], ptr @canvas, i64 0, i64 %po2\n");
  es("  %pink = load i8, ptr %pp, align 1\n");
  es("  %plit = icmp ne i8 %pink, 0\n");
  es("  %glyph = select i1 %plit, i32 42, i32 32\n");
  es("  %cellres = call i32 @putchar(i32 %glyph)\n");
  es("  %pc4 = load i32, ptr %col, align 4\n");
  es("  %pc5 = add nsw i32 %pc4, 1\n");
  es("  store i32 %pc5, ptr %col, align 4\n");
  es("  br label %print.head\n");
  es("\n");
  es("print.newline:\n");
  es("  %nlres = call i32 @putchar(i32 10)\n");
  es("  %rn = load i32, ptr %row, align 4\n");
  es("  %rn2 = add nsw i32 %rn, 1\n");
  es("  store i32 %rn2, ptr %row, align 4\n");
  es("  br label %row.head\n");
  es("\n");
  es("row.done:\n");
  es("  ret void\n");
  es("}\n");
  return 0;
}

fn emit_runtime() {
  emit_runtime_0();
  emit_runtime_1();
  emit_runtime_2();
  emit_runtime_3();
  emit_runtime_4();
  emit_runtime_5();
  emit_runtime_6();
  emit_runtime_7();
  emit_runtime_8();
  emit_runtime_9();
  return 0;
}

# ----------------------------------------------------------------------
# layer 16: lowering the machine to LLVM IR
# ----------------------------------------------------------------------
#
# The operand stack becomes an alloca'd array plus a stack pointer, every
# instruction address becomes a basic block, and the machine's jumps become
# branches between them.

var STACK_DEPTH = 256;

fn new_reg() {
  reg = reg + 1;
  return reg - 1;
}

fn er(r) {
  putchar('%');
  putchar('g');
  en(r);
  return 0;
}

# An operand is either a virtual register or a literal folded in at compile
# time; the two are printed the same way everywhere they appear.
fn eop(is_register, value) {
  if (is_register == 1) {
    er(value);
  } else {
    en(value);
  }
  return 0;
}

fn push_operand(is_register, value) {
  var top = new_reg();
  var slot = new_reg();
  var bumped = new_reg();
  es("  ");
  er(top);
  es(" = load i64, ptr %sp, align 8\n");
  es("  ");
  er(slot);
  es(" = getelementptr inbounds [");
  en(STACK_DEPTH);
  es(" x i64], ptr %stack, i64 0, i64 ");
  er(top);
  es("\n");
  es("  store i64 ");
  eop(is_register, value);
  es(", ptr ");
  er(slot);
  es(", align 8\n");
  es("  ");
  er(bumped);
  es(" = add nsw i64 ");
  er(top);
  es(", 1\n");
  es("  store i64 ");
  er(bumped);
  es(", ptr %sp, align 8\n");
  return 0;
}

fn pop_operand() {
  var top = new_reg();
  var lowered = new_reg();
  var slot = new_reg();
  var value = new_reg();
  es("  ");
  er(top);
  es(" = load i64, ptr %sp, align 8\n");
  es("  ");
  er(lowered);
  es(" = sub nsw i64 ");
  er(top);
  es(", 1\n");
  es("  store i64 ");
  er(lowered);
  es(", ptr %sp, align 8\n");
  es("  ");
  er(slot);
  es(" = getelementptr inbounds [");
  en(STACK_DEPTH);
  es(" x i64], ptr %stack, i64 0, i64 ");
  er(lowered);
  es("\n");
  es("  ");
  er(value);
  es(" = load i64, ptr ");
  er(slot);
  es(", align 8\n");
  return value;
}

fn frame_slot(slot) {
  var address = new_reg();
  es("  ");
  er(address);
  es(" = getelementptr inbounds [");
  en(frame);
  es(" x i64], ptr %frame, i64 0, i64 ");
  en(slot);
  es("\n");
  return address;
}

# sdiv truncates towards zero where the interpreter floors, so the quotient
# is decremented whenever the division was inexact and the remainder and the
# divisor disagree about sign.
fn floor_divide(left, right) {
  var quotient = new_reg();
  var remainder = new_reg();
  var inexact = new_reg();
  var remainder_negative = new_reg();
  var divisor_negative = new_reg();
  var signs_differ = new_reg();
  var correct = new_reg();
  var decremented = new_reg();
  var result = new_reg();
  es("  ");
  er(quotient);
  es(" = sdiv i64 ");
  er(left);
  es(", ");
  er(right);
  es("\n");
  es("  ");
  er(remainder);
  es(" = srem i64 ");
  er(left);
  es(", ");
  er(right);
  es("\n");
  es("  ");
  er(inexact);
  es(" = icmp ne i64 ");
  er(remainder);
  es(", 0\n");
  es("  ");
  er(remainder_negative);
  es(" = icmp slt i64 ");
  er(remainder);
  es(", 0\n");
  es("  ");
  er(divisor_negative);
  es(" = icmp slt i64 ");
  er(right);
  es(", 0\n");
  es("  ");
  er(signs_differ);
  es(" = xor i1 ");
  er(remainder_negative);
  es(", ");
  er(divisor_negative);
  es("\n");
  es("  ");
  er(correct);
  es(" = and i1 ");
  er(inexact);
  es(", ");
  er(signs_differ);
  es("\n");
  es("  ");
  er(decremented);
  es(" = sub nsw i64 ");
  er(quotient);
  es(", 1\n");
  es("  ");
  er(result);
  es(" = select i1 ");
  er(correct);
  es(", i64 ");
  er(decremented);
  es(", i64 ");
  er(quotient);
  es("\n");
  return result;
}

fn truncate(value) {
  var narrowed = new_reg();
  es("  ");
  er(narrowed);
  es(" = trunc i64 ");
  er(value);
  es(" to i32\n");
  return narrowed;
}

fn intrinsic_value(index) {
  if (index == 0) {
    return 0;
  }
  if (index == 1) {
    return APOTHEM;
  }
  if (index == 2) {
    return EXTREMUM;
  }
  return ORDER;
}

fn lower_arithmetic(kind) {
  var right = pop_operand();
  var left = pop_operand();
  var result = new_reg();
  es("  ");
  er(result);
  if (kind == OP_ADD) {
    es(" = add nsw i64 ");
  }
  if (kind == OP_SUB) {
    es(" = sub nsw i64 ");
  }
  if (kind == OP_MUL) {
    es(" = mul nsw i64 ");
  }
  er(left);
  es(", ");
  er(right);
  es("\n");
  push_operand(1, result);
  return 0;
}

fn lower_emit_run(orientation) {
  var upper = truncate(pop_operand());
  var lower = truncate(pop_operand());
  var index = truncate(pop_operand());
  es("  call void @gvm.emit_run(i32 ");
  en(orientation);
  es(", i32 ");
  er(index);
  es(", i32 ");
  er(lower);
  es(", i32 ");
  er(upper);
  es(")\n");
  return 0;
}

fn lower_close() {
  es("  call void @gvm.snapshot()\n");
  var i = 0;
  while (i < ngroup) {
    es("  call void @gvm.apply(i32 ");
    en(mem[GRPA + i]);
    es(", i32 ");
    en(mem[GRPB + i]);
    es(", i32 ");
    en(mem[GRPC + i]);
    es(", i32 ");
    en(mem[GRPD + i]);
    es(")\n");
    i = i + 1;
  }
  return 0;
}

fn lower_branch(address, target) {
  var condition = pop_operand();
  var test = new_reg();
  es("  ");
  er(test);
  es(" = icmp ne i64 ");
  er(condition);
  es(", 0\n");
  es("  br i1 ");
  er(test);
  es(", label %A");
  en(address + 1);
  es(", label %A");
  en(target);
  es("\n");
  return 0;
}

fn lower_stack_op(kind) {
  if (kind == OP_DIV) {
    var right = pop_operand();
    var left = pop_operand();
    push_operand(1, floor_divide(left, right));
    return 0;
  }
  if (kind == OP_NEG) {
    var operand = pop_operand();
    var negated = new_reg();
    es("  ");
    er(negated);
    es(" = sub nsw i64 0, ");
    er(operand);
    es("\n");
    push_operand(1, negated);
    return 0;
  }
  if (kind == OP_CMPLE) {
    var upper = pop_operand();
    var lower = pop_operand();
    var flag = new_reg();
    var result = new_reg();
    es("  ");
    er(flag);
    es(" = icmp sle i64 ");
    er(lower);
    es(", ");
    er(upper);
    es("\n");
    es("  ");
    er(result);
    es(" = zext i1 ");
    er(flag);
    es(" to i64\n");
    push_operand(1, result);
    return 0;
  }
  return lower_arithmetic(kind);
}

fn lower_instruction(address) {
  var kind = mem[CODEK + address];
  var argument = mem[CODEA + address];
  if (kind == OP_HALT) {
    es("  br label %exit\n");
    return 0;
  }
  if (kind == OP_JMP) {
    es("  br label %A");
    en(argument);
    es("\n");
    return 0;
  }
  if (kind == OP_JF) {
    lower_branch(address, argument);
    return 0;
  }
  if (kind == OP_PUSH) {
    push_operand(0, argument);
  }
  if (kind == OP_INTR) {
    push_operand(0, intrinsic_value(argument));
  }
  if (kind == OP_LOADL) {
    var source = frame_slot(argument);
    var value = new_reg();
    es("  ");
    er(value);
    es(" = load i64, ptr ");
    er(source);
    es(", align 8\n");
    push_operand(1, value);
  }
  if (kind == OP_STOREL) {
    var stored = pop_operand();
    var target = frame_slot(argument);
    es("  store i64 ");
    er(stored);
    es(", ptr ");
    er(target);
    es(", align 8\n");
  }
  if (kind == OP_ADD || kind == OP_SUB || kind == OP_MUL || kind == OP_DIV
      || kind == OP_NEG || kind == OP_CMPLE) {
    lower_stack_op(kind);
  }
  if (kind == OP_MKIV) {
    es("  ; make.interval is erased: bounds stay on the stack\n");
  }
  if (kind == OP_EMIT) {
    lower_emit_run(argument);
  }
  if (kind == OP_CLOSE) {
    lower_close();
  }
  es("  br label %A");
  en(address + 1);
  es("\n");
  return 0;
}

fn emit_entry_point() {
  es("define i32 @main() {\n");
  es("entry:\n");
  es("  %stack = alloca [");
  en(STACK_DEPTH);
  es(" x i64], align 8\n");
  es("  %sp = alloca i64, align 8\n");
  es("  %frame = alloca [");
  en(frame);
  es(" x i64], align 8\n");
  es("  store i64 0, ptr %sp, align 8\n");
  var slot = 0;
  while (slot < frame) {
    var address = frame_slot(slot);
    es("  store i64 0, ptr ");
    er(address);
    es(", align 8\n");
    slot = slot + 1;
  }
  es("  br label %A0\n");
  var i = 0;
  while (i < ncode) {
    es("\nA");
    en(i);
    es(":\n");
    lower_instruction(i);
    i = i + 1;
  }
  es("\nexit:\n");
  es("  call void @gvm.render()\n");
  es("  ");
  er(new_reg());
  es(" = call i32 @fflush(ptr null)\n");
  es("  ret i32 0\n");
  es("}\n");
  return 0;
}

fn emit_module() {
  es("; ModuleID = 'glyph.canonical'\n");
  es("source_filename = \"canonical.gsl\"\n");
  es("target triple = \"x86_64-unknown-linux-gnu\"\n");
  emit_runtime();
  es("\n");
  emit_entry_point();
  return 0;
}

# ----------------------------------------------------------------------
# the driver
# ----------------------------------------------------------------------

fn main() {
  read_stdin();
  preprocess();
  tokenize();
  parse_program();
  optimise();
  assemble();
  group_build();
  emit_module();
  return 0;
}
'''

_S0_OUT: list[str] = []
_S0_SOURCE: str = ""


# ----------------------------------------------------------------------
# stage 0: the seed compiler
# ----------------------------------------------------------------------


MEMSIZE = 2000000
STRBASE = 1500000
STRLIMIT = 490000
FRAME = 64
MAXGLOBALS = 256

SRC = 0
STRBUF = 500000
LOCNAME = 900000
LOCLEN = 900100
GLBNAME = 900200
GLBLEN = 900600
TOKBUF = 950000
ARGS = 960000

T_EOF = 0
T_NUM = 1
T_IDENT = 2
T_STR = 3
T_LPAREN = 4
T_RPAREN = 5
T_LBRACE = 6
T_RBRACE = 7
T_LBRACK = 8
T_RBRACK = 9
T_COMMA = 10
T_SEMI = 11
T_ASSIGN = 12
T_EQ = 13
T_NE = 14
T_LT = 15
T_LE = 16
T_GT = 17
T_GE = 18
T_PLUS = 19
T_MINUS = 20
T_STAR = 21
T_SLASH = 22
T_PERCENT = 23
T_NOT = 24
T_ANDAND = 25
T_OROR = 26
T_FN = 27
T_VAR = 28
T_IF = 29
T_ELSE = 30
T_WHILE = 31
T_RETURN = 32
T_MEM = 33

mem = [0] * 1200000
pos = 0
srclen = 0
strtop = 1
nlocals = 0
nglobals = 0
regcnt = 0
labelcnt = 0
line = 1
argsp = 0


def es(s):
    _S0_OUT.append(s)


def en(n):
    _S0_OUT.append(str(n))


def ec(c):
    _S0_OUT.append(chr(c))


def fail(msg):
    raise Gsl2Error(f"{msg} near line {line}")


def kw_is(start, length, word):
    if length != len(word):
        return 0
    i = 0
    while i < length:
        if mem[start + i] != ord(word[i]):
            return 0
        i = i + 1
    return 1


def name_eq(a, alen, b, blen):
    if alen != blen:
        return 0
    i = 0
    while i < alen:
        if mem[a + i] != mem[b + i]:
            return 0
        i = i + 1
    return 1


def emit_name(start, length):
    i = 0
    while i < length:
        ec(mem[start + i])
        i = i + 1


# ----------------------------------------------------------------------
# lexer
# ----------------------------------------------------------------------


def is_digit(c):
    return 1 if 48 <= c <= 57 else 0


def is_alpha(c):
    if 97 <= c <= 122:
        return 1
    if 65 <= c <= 90:
        return 1
    if c == 95:
        return 1
    return 0


def is_alnum(c):
    if is_alpha(c):
        return 1
    return is_digit(c)


def skip_ws():
    global pos, line
    while 1:
        c = mem[SRC + pos]
        if c == 35:
            while mem[SRC + pos] != 10 and mem[SRC + pos] != 0:
                pos = pos + 1
        else:
            if c == 10:
                line = line + 1
                pos = pos + 1
            else:
                if c == 32 or c == 9 or c == 13:
                    pos = pos + 1
                else:
                    return 0
    return 0


def escape_of(c):
    if c == 110:
        return 10
    if c == 116:
        return 9
    if c == 48:
        return 0
    if c == 92:
        return 92
    if c == 34:
        return 34
    if c == 39:
        return 39
    if c == 114:
        return 13
    fail("bad escape")
    return 0


def scan(slot):
    global pos, strtop
    base = TOKBUF + slot * 4
    skip_ws()
    c = mem[SRC + pos]
    mem[base + 1] = 0
    mem[base + 2] = SRC + pos
    mem[base + 3] = 0
    if c == 0:
        mem[base] = T_EOF
        return 0
    if is_digit(c):
        v = 0
        while is_digit(mem[SRC + pos]):
            v = v * 10 + mem[SRC + pos] - 48
            pos = pos + 1
        mem[base] = T_NUM
        mem[base + 1] = v
        return 0
    if is_alpha(c):
        start = SRC + pos
        while is_alnum(mem[SRC + pos]):
            pos = pos + 1
        length = SRC + pos - start
        mem[base + 2] = start
        mem[base + 3] = length
        if kw_is(start, length, "fn"):
            mem[base] = T_FN
            return 0
        if kw_is(start, length, "var"):
            mem[base] = T_VAR
            return 0
        if kw_is(start, length, "if"):
            mem[base] = T_IF
            return 0
        if kw_is(start, length, "else"):
            mem[base] = T_ELSE
            return 0
        if kw_is(start, length, "while"):
            mem[base] = T_WHILE
            return 0
        if kw_is(start, length, "return"):
            mem[base] = T_RETURN
            return 0
        if kw_is(start, length, "mem"):
            mem[base] = T_MEM
            return 0
        mem[base] = T_IDENT
        return 0
    if c == 39:
        pos = pos + 1
        v = mem[SRC + pos]
        if v == 92:
            pos = pos + 1
            v = escape_of(mem[SRC + pos])
        pos = pos + 1
        if mem[SRC + pos] != 39:
            fail("unterminated char literal")
        pos = pos + 1
        mem[base] = T_NUM
        mem[base + 1] = v
        return 0
    if c == 34:
        pos = pos + 1
        off = STRBASE + strtop
        while mem[SRC + pos] != 34:
            if mem[SRC + pos] == 0:
                fail("unterminated string literal")
            v = mem[SRC + pos]
            if v == 92:
                pos = pos + 1
                v = escape_of(mem[SRC + pos])
            mem[STRBUF + strtop] = v
            strtop = strtop + 1
            pos = pos + 1
        pos = pos + 1
        mem[STRBUF + strtop] = 0
        strtop = strtop + 1
        if strtop > STRLIMIT:
            fail("string pool overflow")
        mem[base] = T_STR
        mem[base + 1] = off
        return 0
    pos = pos + 1
    d = mem[SRC + pos]
    if c == 61:
        if d == 61:
            pos = pos + 1
            mem[base] = T_EQ
            return 0
        mem[base] = T_ASSIGN
        return 0
    if c == 33:
        if d == 61:
            pos = pos + 1
            mem[base] = T_NE
            return 0
        mem[base] = T_NOT
        return 0
    if c == 60:
        if d == 61:
            pos = pos + 1
            mem[base] = T_LE
            return 0
        mem[base] = T_LT
        return 0
    if c == 62:
        if d == 61:
            pos = pos + 1
            mem[base] = T_GE
            return 0
        mem[base] = T_GT
        return 0
    if c == 38:
        if d == 38:
            pos = pos + 1
            mem[base] = T_ANDAND
            return 0
        fail("single & is not an operator")
    if c == 124:
        if d == 124:
            pos = pos + 1
            mem[base] = T_OROR
            return 0
        fail("single | is not an operator")
    if c == 40:
        mem[base] = T_LPAREN
        return 0
    if c == 41:
        mem[base] = T_RPAREN
        return 0
    if c == 123:
        mem[base] = T_LBRACE
        return 0
    if c == 125:
        mem[base] = T_RBRACE
        return 0
    if c == 91:
        mem[base] = T_LBRACK
        return 0
    if c == 93:
        mem[base] = T_RBRACK
        return 0
    if c == 44:
        mem[base] = T_COMMA
        return 0
    if c == 59:
        mem[base] = T_SEMI
        return 0
    if c == 43:
        mem[base] = T_PLUS
        return 0
    if c == 45:
        mem[base] = T_MINUS
        return 0
    if c == 42:
        mem[base] = T_STAR
        return 0
    if c == 47:
        mem[base] = T_SLASH
        return 0
    if c == 37:
        mem[base] = T_PERCENT
        return 0
    fail("unexpected character")
    return 0


def tk():
    return mem[TOKBUF]


def tnum():
    return mem[TOKBUF + 1]


def tstart():
    return mem[TOKBUF + 2]


def tlen():
    return mem[TOKBUF + 3]


def tk2():
    return mem[TOKBUF + 4]


def advance():
    mem[TOKBUF] = mem[TOKBUF + 4]
    mem[TOKBUF + 1] = mem[TOKBUF + 5]
    mem[TOKBUF + 2] = mem[TOKBUF + 6]
    mem[TOKBUF + 3] = mem[TOKBUF + 7]
    scan(1)
    return 0


def expect(kind, what):
    if tk() != kind:
        fail(what)
    advance()
    return 0


# ----------------------------------------------------------------------
# symbol tables
# ----------------------------------------------------------------------


def declare_local(start, length):
    global nlocals
    if nlocals >= FRAME:
        fail("too many locals in one function")
    mem[LOCNAME + nlocals] = start
    mem[LOCLEN + nlocals] = length
    nlocals = nlocals + 1
    return nlocals - 1


def find_local(start, length):
    i = nlocals - 1
    while i >= 0:
        if name_eq(mem[LOCNAME + i], mem[LOCLEN + i], start, length):
            return i
        i = i - 1
    return 0 - 1


def declare_global(start, length):
    global nglobals
    if nglobals >= MAXGLOBALS:
        fail("too many globals")
    mem[GLBNAME + nglobals] = start
    mem[GLBLEN + nglobals] = length
    nglobals = nglobals + 1
    return nglobals - 1


def find_global(start, length):
    i = nglobals - 1
    while i >= 0:
        if name_eq(mem[GLBNAME + i], mem[GLBLEN + i], start, length):
            return i
        i = i - 1
    return 0 - 1


def alloc_temp_slot():
    global nlocals
    if nlocals >= FRAME:
        fail("frame overflow")
    mem[LOCNAME + nlocals] = 0
    mem[LOCLEN + nlocals] = 0
    nlocals = nlocals + 1
    return nlocals - 1


# ----------------------------------------------------------------------
# emission primitives
# ----------------------------------------------------------------------


def new_reg():
    global regcnt
    regcnt = regcnt + 1
    return regcnt


def new_label():
    global labelcnt
    labelcnt = labelcnt + 1
    return labelcnt


def er(r):
    es("%t")
    en(r)
    return 0


def elabel(l):
    es("L")
    en(l)
    return 0


def emit_label(l):
    elabel(l)
    es(":\n")
    return 0


def emit_br(l):
    es("  br label %")
    elabel(l)
    es("\n")
    return 0


def emit_cond_br(r, a, b):
    t = new_reg()
    es("  ")
    er(t)
    es(" = icmp ne i64 ")
    er(r)
    es(", 0\n")
    es("  br i1 ")
    er(t)
    es(", label %")
    elabel(a)
    es(", label %")
    elabel(b)
    es("\n")
    return 0


def gen_const(v):
    r = new_reg()
    es("  ")
    er(r)
    es(" = add i64 0, ")
    en(v)
    es("\n")
    return r


def gen_slot_addr(slot):
    r = new_reg()
    es("  ")
    er(r)
    es(" = getelementptr inbounds [")
    en(FRAME)
    es(" x i64], ptr %frame, i64 0, i64 ")
    en(slot)
    es("\n")
    return r


def gen_mem_addr(index_reg):
    r = new_reg()
    es("  ")
    er(r)
    es(" = getelementptr inbounds [")
    en(MEMSIZE)
    es(" x i64], ptr @memory, i64 0, i64 ")
    er(index_reg)
    es("\n")
    return r


def gen_load(addr_reg):
    r = new_reg()
    es("  ")
    er(r)
    es(" = load i64, ptr ")
    er(addr_reg)
    es(", align 8\n")
    return r


def gen_store(value_reg, addr_reg):
    es("  store i64 ")
    er(value_reg)
    es(", ptr ")
    er(addr_reg)
    es(", align 8\n")
    return 0


def gen_global_load(start, length):
    r = new_reg()
    es("  ")
    er(r)
    es(" = load i64, ptr @g_")
    emit_name(start, length)
    es(", align 8\n")
    return r


def gen_global_store(value_reg, start, length):
    es("  store i64 ")
    er(value_reg)
    es(", ptr @g_")
    emit_name(start, length)
    es(", align 8\n")
    return 0


def gen_binary(op, a, b):
    r = new_reg()
    es("  ")
    er(r)
    if op == T_PLUS:
        es(" = add nsw i64 ")
    if op == T_MINUS:
        es(" = sub nsw i64 ")
    if op == T_STAR:
        es(" = mul nsw i64 ")
    if op == T_SLASH:
        es(" = sdiv i64 ")
    if op == T_PERCENT:
        es(" = srem i64 ")
    er(a)
    es(", ")
    er(b)
    es("\n")
    return r


def gen_compare(op, a, b):
    c = new_reg()
    es("  ")
    er(c)
    if op == T_EQ:
        es(" = icmp eq i64 ")
    if op == T_NE:
        es(" = icmp ne i64 ")
    if op == T_LT:
        es(" = icmp slt i64 ")
    if op == T_LE:
        es(" = icmp sle i64 ")
    if op == T_GT:
        es(" = icmp sgt i64 ")
    if op == T_GE:
        es(" = icmp sge i64 ")
    er(a)
    es(", ")
    er(b)
    es("\n")
    r = new_reg()
    es("  ")
    er(r)
    es(" = zext i1 ")
    er(c)
    es(" to i64\n")
    return r


# ----------------------------------------------------------------------
# expressions
# ----------------------------------------------------------------------


def parse_call_builtin(kind, first_arg):
    if kind == 1:
        t = new_reg()
        es("  ")
        er(t)
        es(" = trunc i64 ")
        er(first_arg)
        es(" to i32\n")
        u = new_reg()
        es("  ")
        er(u)
        es(" = call i32 @putchar(i32 ")
        er(t)
        es(")\n")
        r = new_reg()
        es("  ")
        er(r)
        es(" = sext i32 ")
        er(u)
        es(" to i64\n")
        return r
    if kind == 2:
        u = new_reg()
        es("  ")
        er(u)
        es(" = call i32 @getchar()\n")
        r = new_reg()
        es("  ")
        er(r)
        es(" = sext i32 ")
        er(u)
        es(" to i64\n")
        return r
    t = new_reg()
    es("  ")
    er(t)
    es(" = trunc i64 ")
    er(first_arg)
    es(" to i32\n")
    es("  call void @exit(i32 ")
    er(t)
    es(")\n")
    return gen_const(0)


def parse_primary():
    global argsp
    if tk() == T_NUM:
        v = tnum()
        advance()
        return gen_const(v)
    if tk() == T_STR:
        v = tnum()
        advance()
        return gen_const(v)
    if tk() == T_LPAREN:
        advance()
        r = parse_expr()
        expect(T_RPAREN, "expected )")
        return r
    if tk() == T_MEM:
        advance()
        expect(T_LBRACK, "expected [ after mem")
        i = parse_expr()
        expect(T_RBRACK, "expected ]")
        return gen_load(gen_mem_addr(i))
    if tk() == T_IDENT:
        start = tstart()
        length = tlen()
        advance()
        if tk() != T_LPAREN:
            slot = find_local(start, length)
            if slot >= 0:
                return gen_load(gen_slot_addr(slot))
            if find_global(start, length) >= 0:
                return gen_global_load(start, length)
            fail("unknown identifier")
        advance()
        if kw_is(start, length, "putchar"):
            a = parse_expr()
            expect(T_RPAREN, "expected )")
            return parse_call_builtin(1, a)
        if kw_is(start, length, "getchar"):
            expect(T_RPAREN, "expected )")
            return parse_call_builtin(2, 0)
        if kw_is(start, length, "exit"):
            a = parse_expr()
            expect(T_RPAREN, "expected )")
            return parse_call_builtin(3, a)
        base = argsp
        argsp = argsp + 9
        nargs = 0
        while tk() != T_RPAREN:
            if nargs > 0:
                expect(T_COMMA, "expected , between arguments")
            mem[ARGS + base + nargs] = parse_expr()
            nargs = nargs + 1
            if nargs > 8:
                fail("too many arguments")
        advance()
        r = new_reg()
        es("  ")
        er(r)
        es(" = call i64 @f_")
        emit_name(start, length)
        es("(")
        i = 0
        while i < nargs:
            if i > 0:
                es(", ")
            es("i64 ")
            er(mem[ARGS + base + i])
            i = i + 1
        es(")\n")
        argsp = base
        return r
    fail("expected an expression")
    return 0


def parse_unary():
    if tk() == T_MINUS:
        advance()
        a = parse_unary()
        return gen_binary(T_MINUS, gen_const(0), a)
    if tk() == T_NOT:
        advance()
        a = parse_unary()
        return gen_compare(T_EQ, a, gen_const(0))
    return parse_primary()


def parse_mul():
    a = parse_unary()
    while tk() == T_STAR or tk() == T_SLASH or tk() == T_PERCENT:
        op = tk()
        advance()
        b = parse_unary()
        a = gen_binary(op, a, b)
    return a


def parse_add():
    a = parse_mul()
    while tk() == T_PLUS or tk() == T_MINUS:
        op = tk()
        advance()
        b = parse_mul()
        a = gen_binary(op, a, b)
    return a


def parse_rel():
    a = parse_add()
    while tk() == T_LT or tk() == T_LE or tk() == T_GT or tk() == T_GE:
        op = tk()
        advance()
        b = parse_add()
        a = gen_compare(op, a, b)
    return a


def parse_eq():
    a = parse_rel()
    while tk() == T_EQ or tk() == T_NE:
        op = tk()
        advance()
        b = parse_rel()
        a = gen_compare(op, a, b)
    return a


def parse_and():
    a = parse_eq()
    while tk() == T_ANDAND:
        advance()
        slot = alloc_temp_slot()
        s1 = gen_slot_addr(slot)
        es("  store i64 0, ptr ")
        er(s1)
        es(", align 8\n")
        lrhs = new_label()
        lend = new_label()
        emit_cond_br(a, lrhs, lend)
        emit_label(lrhs)
        b = parse_eq()
        v = gen_compare(T_NE, b, gen_const(0))
        gen_store(v, gen_slot_addr(slot))
        emit_br(lend)
        emit_label(lend)
        a = gen_load(gen_slot_addr(slot))
    return a


def parse_or():
    a = parse_and()
    while tk() == T_OROR:
        advance()
        slot = alloc_temp_slot()
        s1 = gen_slot_addr(slot)
        es("  store i64 1, ptr ")
        er(s1)
        es(", align 8\n")
        lrhs = new_label()
        lend = new_label()
        emit_cond_br(a, lend, lrhs)
        emit_label(lrhs)
        b = parse_and()
        v = gen_compare(T_NE, b, gen_const(0))
        gen_store(v, gen_slot_addr(slot))
        emit_br(lend)
        emit_label(lend)
        a = gen_load(gen_slot_addr(slot))
    return a


def parse_expr():
    return parse_or()


# ----------------------------------------------------------------------
# statements
# ----------------------------------------------------------------------


def parse_block():
    expect(T_LBRACE, "expected {")
    while tk() != T_RBRACE:
        if tk() == T_EOF:
            fail("unterminated block")
        parse_stmt()
    advance()
    return 0


def parse_stmt():
    if tk() == T_VAR:
        advance()
        if tk() != T_IDENT:
            fail("expected a name after var")
        start = tstart()
        length = tlen()
        advance()
        expect(T_ASSIGN, "expected = in var declaration")
        r = parse_expr()
        expect(T_SEMI, "expected ;")
        slot = declare_local(start, length)
        gen_store(r, gen_slot_addr(slot))
        return 0
    if tk() == T_IF:
        advance()
        expect(T_LPAREN, "expected ( after if")
        c = parse_expr()
        expect(T_RPAREN, "expected )")
        lthen = new_label()
        lelse = new_label()
        lend = new_label()
        emit_cond_br(c, lthen, lelse)
        emit_label(lthen)
        parse_block()
        emit_br(lend)
        emit_label(lelse)
        if tk() == T_ELSE:
            advance()
            parse_block()
        emit_br(lend)
        emit_label(lend)
        return 0
    if tk() == T_WHILE:
        advance()
        lhead = new_label()
        lbody = new_label()
        lend = new_label()
        emit_br(lhead)
        emit_label(lhead)
        expect(T_LPAREN, "expected ( after while")
        c = parse_expr()
        expect(T_RPAREN, "expected )")
        emit_cond_br(c, lbody, lend)
        emit_label(lbody)
        parse_block()
        emit_br(lhead)
        emit_label(lend)
        return 0
    if tk() == T_RETURN:
        advance()
        r = parse_expr()
        expect(T_SEMI, "expected ;")
        es("  ret i64 ")
        er(r)
        es("\n")
        emit_label(new_label())
        return 0
    if tk() == T_MEM:
        advance()
        expect(T_LBRACK, "expected [ after mem")
        i = parse_expr()
        expect(T_RBRACK, "expected ]")
        expect(T_ASSIGN, "expected = in mem assignment")
        v = parse_expr()
        expect(T_SEMI, "expected ;")
        gen_store(v, gen_mem_addr(i))
        return 0
    if tk() == T_IDENT and tk2() == T_ASSIGN:
        start = tstart()
        length = tlen()
        advance()
        advance()
        r = parse_expr()
        expect(T_SEMI, "expected ;")
        slot = find_local(start, length)
        if slot >= 0:
            gen_store(r, gen_slot_addr(slot))
            return 0
        if find_global(start, length) >= 0:
            gen_global_store(r, start, length)
            return 0
        fail("assignment to unknown identifier")
    parse_expr()
    expect(T_SEMI, "expected ;")
    return 0


# ----------------------------------------------------------------------
# declarations
# ----------------------------------------------------------------------


def parse_global_decl():
    advance()
    if tk() != T_IDENT:
        fail("expected a name after var")
    start = tstart()
    length = tlen()
    advance()
    expect(T_ASSIGN, "expected = in global declaration")
    neg = 0
    if tk() == T_MINUS:
        neg = 1
        advance()
    if tk() != T_NUM:
        fail("global initializers must be integer literals")
    v = tnum()
    advance()
    expect(T_SEMI, "expected ;")
    declare_global(start, length)
    es("@g_")
    emit_name(start, length)
    es(" = internal global i64 ")
    if neg == 1:
        es("-")
    en(v)
    es("\n")
    return 0


def parse_function():
    global nlocals, regcnt, labelcnt
    advance()
    if tk() != T_IDENT:
        fail("expected a function name")
    start = tstart()
    length = tlen()
    advance()
    expect(T_LPAREN, "expected ( after function name")
    nlocals = 0
    regcnt = 0
    labelcnt = 0
    nparams = 0
    while tk() != T_RPAREN:
        if nparams > 0:
            expect(T_COMMA, "expected , between parameters")
        if tk() != T_IDENT:
            fail("expected a parameter name")
        declare_local(tstart(), tlen())
        nparams = nparams + 1
        advance()
    advance()
    es("\ndefine i64 @f_")
    emit_name(start, length)
    es("(")
    i = 0
    while i < nparams:
        if i > 0:
            es(", ")
        es("i64 %p")
        en(i)
        i = i + 1
    es(") {\nentry:\n")
    es("  %frame = alloca [")
    en(FRAME)
    es(" x i64], align 8\n")
    i = 0
    while i < nparams:
        es("  %a")
        en(i)
        es(" = getelementptr inbounds [")
        en(FRAME)
        es(" x i64], ptr %frame, i64 0, i64 ")
        en(i)
        es("\n")
        es("  store i64 %p")
        en(i)
        es(", ptr %a")
        en(i)
        es(", align 8\n")
        i = i + 1
    parse_block()
    es("  ret i64 0\n}\n")
    return 0


def emit_header():
    es("target triple = \"x86_64-unknown-linux-gnu\"\n\n")
    es("@memory = internal global [")
    en(MEMSIZE)
    es(" x i64] zeroinitializer\n\n")
    es("declare i32 @putchar(i32)\n")
    es("declare i32 @getchar()\n")
    es("declare void @exit(i32)\n\n")
    return 0


def emit_trailer():
    es("\n@strdata = internal constant [")
    en(strtop)
    es(" x i64] [")
    i = 0
    while i < strtop:
        if i > 0:
            es(", ")
        es("i64 ")
        en(mem[STRBUF + i])
        i = i + 1
    es("]\n\n")
    es("define internal void @__init_strings() {\nentry:\n")
    es("  %i = alloca i64, align 8\n")
    es("  store i64 0, ptr %i, align 8\n")
    es("  br label %head\n\nhead:\n")
    es("  %c = load i64, ptr %i, align 8\n")
    es("  %m = icmp slt i64 %c, ")
    en(strtop)
    es("\n  br i1 %m, label %body, label %done\n\nbody:\n")
    es("  %si = load i64, ptr %i, align 8\n")
    es("  %sp = getelementptr inbounds [")
    en(strtop)
    es(" x i64], ptr @strdata, i64 0, i64 %si\n")
    es("  %sv = load i64, ptr %sp, align 8\n")
    es("  %dj = add nsw i64 %si, ")
    en(STRBASE)
    es("\n")
    es("  %dp = getelementptr inbounds [")
    en(MEMSIZE)
    es(" x i64], ptr @memory, i64 0, i64 %dj\n")
    es("  store i64 %sv, ptr %dp, align 8\n")
    es("  %ni = add nsw i64 %si, 1\n")
    es("  store i64 %ni, ptr %i, align 8\n")
    es("  br label %head\n\ndone:\n  ret void\n}\n\n")
    es("define i32 @main() {\nentry:\n")
    es("  call void @__init_strings()\n")
    es("  %r = call i64 @f_main()\n")
    es("  %t = trunc i64 %r to i32\n")
    es("  ret i32 %t\n}\n")
    return 0


def read_source():
    global srclen
    data = _S0_SOURCE.encode()
    i = 0
    while i < len(data):
        mem[SRC + i] = data[i]
        i = i + 1
    mem[SRC + i] = 0
    srclen = i
    return 0


def compile_unit():
    emit_header()
    scan(0)
    scan(1)
    while tk() != T_EOF:
        if tk() == T_VAR:
            parse_global_decl()
        else:
            if tk() == T_FN:
                parse_function()
            else:
                fail("expected fn or var at top level")
    emit_trailer()
    return 0


def _stage0_run():
    read_source()
    compile_unit()
    return 0


def _stage0_reset() -> None:
    """Return the seed compiler to a pristine state between translation units."""
    global mem, pos, srclen, strtop, nlocals, nglobals, regcnt, labelcnt
    global line, argsp, _S0_OUT
    mem = [0] * 1200000
    pos = 0
    srclen = 0
    strtop = 1
    nlocals = 0
    nglobals = 0
    regcnt = 0
    labelcnt = 0
    line = 1
    argsp = 0
    _S0_OUT = []


def gsl2_compile(source: str) -> str:
    """Compile GSL-2 source to LLVM IR using the Python seed compiler."""
    global _S0_SOURCE
    _stage0_reset()
    _S0_SOURCE = source
    _stage0_run()
    return "".join(_S0_OUT)


# ----------------------------------------------------------------------
# native toolchain
# ----------------------------------------------------------------------


def link_executable(ir_text: str, exe_path: Path, opt_level: int = 2) -> Path:
    """Assemble LLVM IR to an object file and link it into a native binary."""
    if shutil.which("gcc") is None and shutil.which("cc") is None:
        raise LlvmToolchainUnavailable("no system C linker (gcc/cc) on PATH")
    toolchain = LlvmToolchainService()
    module = LlvmModule(text=ir_text, profile=TargetProfile(), order=0)
    if opt_level:
        module = toolchain.optimize(module, opt_level)
    object_path = exe_path.with_suffix(".o")
    toolchain.emit_object(module, str(object_path))
    linker = shutil.which("gcc") or shutil.which("cc")
    subprocess.run([linker, str(object_path), "-o", str(exe_path)], check=True)
    return exe_path


def _run(executable: Path, stdin_text: str) -> str:
    completed = subprocess.run(
        [str(executable)],
        input=stdin_text.encode(),
        stdout=subprocess.PIPE,
        check=True,
    )
    return completed.stdout.decode()


@dataclass(frozen=True, slots=True)
class BootstrapReport:
    """The outcome of a full three-generation self-hosting bootstrap."""

    workdir: Path
    stage1_ir: str
    stage2_ir: str
    stage3_ir: str
    glyph: str

    @property
    def seed_agrees(self) -> bool:
        return self.stage1_ir == self.stage2_ir

    @property
    def fixpoint(self) -> bool:
        return self.stage2_ir == self.stage3_ir


def bootstrap(workdir: Path | None = None, opt_level: int = 2) -> BootstrapReport:
    """Compile the GSL-2 compiler with itself until its output stops changing.

    stage0 is the Python seed above.  stage1 is what the seed makes of
    gslc.gsl2.  stage2 is what stage1 makes of the very same source, and
    stage3 what stage2 makes of it.  stage2 == stage3 is the fixpoint that
    proves the compiler reproduces itself; stage1 == stage2 additionally
    proves the seed was never needed for anything but the first turn of the
    crank.
    """
    directory = Path(workdir or tempfile.mkdtemp(prefix="ouroboros-"))
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "gslc.gsl2").write_text(GSLC_GSL2)
    (directory / "glyph.gsl2").write_text(GLYPH_GSL2)

    stage1_ir = gsl2_compile(GSLC_GSL2)
    (directory / "stage1.ll").write_text(stage1_ir)
    stage1 = link_executable(stage1_ir, directory / "stage1", opt_level)

    stage2_ir = _run(stage1, GSLC_GSL2)
    (directory / "stage2.ll").write_text(stage2_ir)
    stage2 = link_executable(stage2_ir, directory / "stage2", opt_level)

    stage3_ir = _run(stage2, GSLC_GSL2)
    (directory / "stage3.ll").write_text(stage3_ir)
    link_executable(stage3_ir, directory / "stage3", opt_level)

    glyph_ir = _run(stage2, GLYPH_GSL2)
    (directory / "glyph.ll").write_text(glyph_ir)
    glyph = link_executable(glyph_ir, directory / "glyph", opt_level)

    return BootstrapReport(
        workdir=directory,
        stage1_ir=stage1_ir,
        stage2_ir=stage2_ir,
        stage3_ir=stage3_ir,
        glyph=_run(glyph, "").removesuffix("\n"),
    )


# ----------------------------------------------------------------------
# closing the loop
# ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LoopCase:
    """One motif compiled the long way round and checked against the short."""

    motif: str
    order: int
    identical: bool
    glyph: str
    expected: str

    @property
    def renders(self) -> bool:
        return self.glyph == self.expected

    @property
    def passed(self) -> bool:
        return self.identical and self.renders


@dataclass(frozen=True, slots=True)
class LoopReport:
    """The outcome of driving a motif through the native front end."""

    workdir: Path
    glyphc_ir: str
    cases: tuple[LoopCase, ...]

    @property
    def clean(self) -> bool:
        return all(case.passed for case in self.cases)


def build_front_end(directory: Path, opt_level: int = 2) -> tuple[Path, str]:
    """Builds glyphc without Python compiling anything but the seed's one turn.

    The seed compiles gslc.gsl2; that binary compiles its own source, which is
    where Python stops; and the compiler that comes out of it compiles the GSL
    front end.  Returns the front end and the IR it was built from.
    """
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "gslc.gsl2").write_text(GSLC_GSL2)
    (directory / "glyphc.gsl2").write_text(GLYPHC_GSL2)

    seeded = link_executable(gsl2_compile(GSLC_GSL2), directory / "stage1", opt_level)
    gslc_ir = _run(seeded, GSLC_GSL2)
    (directory / "gslc.ll").write_text(gslc_ir)
    gslc = link_executable(gslc_ir, directory / "gslc", opt_level)

    glyphc_ir = _run(gslc, GLYPHC_GSL2)
    (directory / "glyphc.ll").write_text(glyphc_ir)
    return link_executable(glyphc_ir, directory / "glyphc", opt_level), glyphc_ir


def close_the_loop(
    workdir: Path | None = None,
    opt_level: int = 2,
    orders: Sequence[int] = (3, 7, 11, 21),
    motifs: Sequence[str] | None = None,
) -> LoopReport:
    """Compiles a motif with a front end that Python had no hand in running.

    The seed turns the crank once, on gslc.gsl2 alone.  From there GSL-2
    compiles itself, the compiler it produces compiles glyphc.gsl2, and that
    binary compiles the motif: text in, LLVM IR out, no interpreter anywhere
    in the chain.  The result is compared byte for byte against what layer 16
    emits for the same source, and the linked glyph against what the virtual
    machine renders.
    """
    directory = Path(workdir or tempfile.mkdtemp(prefix="ouroboros-loop-"))
    glyphc, glyphc_ir = build_front_end(directory, opt_level)

    cases: list[LoopCase] = []
    for motif in motifs if motifs is not None else Motif.catalogue():
        for order in orders:
            source = typing.cast(type, Motif.lookup(motif))().source(order)
            stem = f"{motif}-{order}"
            (directory / f"{stem}.gsl").write_text(source)
            produced = _run(glyphc, source)
            (directory / f"{stem}.ll").write_text(produced)
            artifacts = synthesize_source(source, order).unwrap_or_raise()
            reference = LlvmLoweringBackend().lower(artifacts.module).text
            binary = link_executable(produced, directory / stem, opt_level)
            cases.append(
                LoopCase(
                    motif=motif,
                    order=order,
                    identical=produced == reference,
                    glyph=_run(binary, "").removesuffix("\n"),
                    expected=artifacts.rendering,
                )
            )
    return LoopReport(directory, glyphc_ir, tuple(cases))


# ======================================================================
# Tier 5: straight to the machine
# ======================================================================
#
# Layer 18 drops the toolchain.  No llvmlite, no assembler, no linker, no
# libc: the object module is encoded as x86-64 machine code and wrapped in an
# ELF64 executable here, by hand, and the only thing the result asks of the
# world is two system calls.  A few kilobytes of the standard library stand
# between the glyph program and a file the kernel will run.


class MachineCodeError(GlyphPlatformError):
    """The object module could not be encoded for this machine."""


class Register(enum.IntEnum):
    """The sixteen general-purpose registers, in their encoding order."""

    RAX = 0
    RCX = 1
    RDX = 2
    RBX = 3
    RSP = 4
    RBP = 5
    RSI = 6
    RDI = 7
    R8 = 8
    R9 = 9
    R10 = 10
    R11 = 11
    R12 = 12
    R13 = 13
    R14 = 14
    R15 = 15


@dataclass(frozen=True, slots=True)
class MemoryOperand:
    """base + index * scale + displacement, always encoded through a SIB byte."""

    base: Register
    index: Register | None = None
    scale: int = 1
    displacement: int = 0


CONDITION_CODES: Final[Mapping[str, int]] = {
    "o": 0x0, "no": 0x1, "b": 0x2, "ae": 0x3, "e": 0x4, "ne": 0x5,
    "be": 0x6, "a": 0x7, "s": 0x8, "ns": 0x9, "p": 0xA, "np": 0xB,
    "l": 0xC, "ge": 0xD, "le": 0xE, "g": 0xF,
}

SCALE_ENCODING: Final[Mapping[int, int]] = {1: 0, 2: 1, 4: 2, 8: 3}


class X86Assembler:
    """An x86-64 encoder: the instructions the machine needs, and no more.

    Not to be confused with layer 9's :class:`Assembler`, which resolves the
    virtual machine's labels.  This one resolves x86 branch displacements and
    emits octets.  Memory operands always take the SIB form with a 32-bit
    displacement, which costs a few bytes per instruction and removes every
    special case from the ModRM encoding.
    """

    def __init__(self) -> None:
        self._code = bytearray()
        self._labels: dict[str, int] = {}
        self._fixups: list[tuple[int, str]] = []

    def __len__(self) -> int:
        return len(self._code)

    # -- labels and relocation -------------------------------------------

    def label(self, name: str) -> None:
        if name in self._labels:
            raise MachineCodeError(f"duplicate label {name!r}")
        self._labels[name] = len(self._code)

    def link(self) -> bytes:
        """Patches every forward and backward branch, then freezes the text."""
        for site, name in self._fixups:
            try:
                target = self._labels[name]
            except KeyError as exc:
                raise MachineCodeError(f"unresolved branch target {name!r}") from exc
            struct.pack_into("<i", self._code, site, target - (site + 4))
        return bytes(self._code)

    def _displacement(self, name: str) -> None:
        self._fixups.append((len(self._code), name))
        self._code.extend(b"\0\0\0\0")

    # -- encoding ---------------------------------------------------------

    def _emit(self, *octets: int) -> None:
        self._code.extend(octets)

    def _modrm(self, reg: int, rm: Register | MemoryOperand) -> tuple[int, bytes]:
        """Returns the REX.RXB bits and the ModRM, SIB and displacement bytes."""
        rex = 0x4 if reg >= 8 else 0
        if isinstance(rm, Register):
            if rm >= 8:
                rex |= 0x1
            return rex, bytes((0xC0 | ((reg & 7) << 3) | (rm & 7),))
        if rm.base >= 8:
            rex |= 0x1
        index = 0b100
        if rm.index is not None:
            if rm.index is Register.RSP:
                raise MachineCodeError("rsp cannot be an index register")
            if rm.index >= 8:
                rex |= 0x2
            index = rm.index & 7
        try:
            displacement = struct.pack("<i", rm.displacement)
        except struct.error as exc:
            raise MachineCodeError(f"displacement {rm.displacement} exceeds 32 bits") from exc
        return rex, bytes((
            0x80 | ((reg & 7) << 3) | 0b100,
            (SCALE_ENCODING[rm.scale] << 6) | (index << 3) | (rm.base & 7),
        )) + displacement

    def _quad(self, opcode: Sequence[int], reg: int, rm: Register | MemoryOperand) -> None:
        """A 64-bit operation: REX.W is always present."""
        rex, tail = self._modrm(reg, rm)
        self._emit(0x48 | rex, *opcode)
        self._code.extend(tail)

    def _octet(self, opcode: Sequence[int], reg: int, rm: Register | MemoryOperand) -> None:
        """An 8-bit operation: REX appears only when an operand demands it."""
        rex, tail = self._modrm(reg, rm)
        if rex:
            self._emit(0x40 | rex)
        self._emit(*opcode)
        self._code.extend(tail)

    def _signed(self, value: int, width: str) -> bytes:
        try:
            return struct.pack(width, value)
        except struct.error as exc:
            raise MachineCodeError(f"literal {value} does not fit the operand") from exc

    # -- data movement ----------------------------------------------------

    def load(self, destination: Register, source: Register | MemoryOperand) -> None:
        self._quad((0x8B,), destination, source)

    def store(self, destination: Register | MemoryOperand, source: Register) -> None:
        self._quad((0x89,), source, destination)

    def immediate(self, destination: Register, value: int) -> None:
        self._emit(0x48 | (0x1 if destination >= 8 else 0), 0xB8 | (destination & 7))
        self._code.extend(self._signed(value, "<q"))

    def load_octet(self, destination: Register, source: MemoryOperand) -> None:
        self._octet((0x8A,), destination, source)

    def store_octet(self, destination: MemoryOperand, source: Register) -> None:
        self._octet((0x88,), source, destination)

    def store_octet_immediate(self, destination: MemoryOperand, value: int) -> None:
        self._octet((0xC6,), 0, destination)
        self._emit(value & 0xFF)

    def widen_octet(self, destination: Register, source: Register | MemoryOperand) -> None:
        self._quad((0x0F, 0xB6), destination, source)

    def address_of(self, destination: Register, source: MemoryOperand) -> None:
        self._quad((0x8D,), destination, source)

    def push(self, register: Register) -> None:
        if register >= 8:
            self._emit(0x41)
        self._emit(0x50 | (register & 7))

    def pop(self, register: Register) -> None:
        if register >= 8:
            self._emit(0x41)
        self._emit(0x58 | (register & 7))

    # -- arithmetic -------------------------------------------------------

    _DIRECT: Final[Mapping[str, int]] = {
        "add": 0x03, "or": 0x0B, "and": 0x23, "sub": 0x2B, "xor": 0x33, "cmp": 0x3B,
    }
    _EXTENSION: Final[Mapping[str, int]] = {
        "add": 0, "or": 1, "and": 4, "sub": 5, "xor": 6, "cmp": 7,
    }

    def arithmetic(
        self, operation: str, destination: Register, source: Register | MemoryOperand
    ) -> None:
        self._quad((self._DIRECT[operation],), destination, source)

    def arithmetic_immediate(
        self, operation: str, destination: Register | MemoryOperand, value: int
    ) -> None:
        self._quad((0x81,), self._EXTENSION[operation], destination)
        self._code.extend(self._signed(value, "<i"))

    def compare_octet_immediate(self, destination: MemoryOperand, value: int) -> None:
        self._octet((0x80,), 7, destination)
        self._emit(value & 0xFF)

    def multiply(self, destination: Register, source: Register | MemoryOperand) -> None:
        self._quad((0x0F, 0xAF), destination, source)

    def multiply_immediate(
        self, destination: Register, source: Register | MemoryOperand, value: int
    ) -> None:
        self._quad((0x69,), destination, source)
        self._code.extend(self._signed(value, "<i"))

    def divide(self, divisor: Register | MemoryOperand) -> None:
        self._quad((0xF7,), 7, divisor)

    def sign_extend(self) -> None:
        self._emit(0x48, 0x99)

    def negate(self, destination: Register | MemoryOperand) -> None:
        self._quad((0xF7,), 3, destination)

    def increment(self, destination: Register | MemoryOperand) -> None:
        self._quad((0xFF,), 0, destination)

    def decrement(self, destination: Register | MemoryOperand) -> None:
        self._quad((0xFF,), 1, destination)

    def test(self, left: Register, right: Register | MemoryOperand) -> None:
        self._quad((0x85,), left, right)

    def set_if(self, condition: str, destination: Register | MemoryOperand) -> None:
        self._octet((0x0F, 0x90 | CONDITION_CODES[condition]), 0, destination)

    # -- control flow -----------------------------------------------------

    def jump(self, name: str) -> None:
        self._emit(0xE9)
        self._displacement(name)

    def jump_if(self, condition: str, name: str) -> None:
        self._emit(0x0F, 0x80 | CONDITION_CODES[condition])
        self._displacement(name)

    def call(self, name: str) -> None:
        self._emit(0xE8)
        self._displacement(name)

    def ret(self) -> None:
        self._emit(0xC3)

    def syscall(self) -> None:
        self._emit(0x0F, 0x05)

    def halt(self) -> None:
        self._emit(0xF4)

    def out(self) -> None:
        """out dx, al: the one way out of a machine with no kernel."""
        self._emit(0xEE)


IMAGE_BASE: Final[int] = 0x400000
DATA_BASE: Final[int] = 0x600000
PAGE_SIZE: Final[int] = 0x1000
SYS_WRITE: Final[int] = 1
SYS_EXIT_GROUP: Final[int] = 231
DATA_REGISTER: Final[Register] = Register.R15


def _align_up(value: int, boundary: int) -> int:
    return (value + boundary - 1) // boundary * boundary


@dataclass(frozen=True, slots=True)
class DataLayout:
    """Where the canvas, the frame and the output buffer sit inside .bss."""

    canvas: int
    snapshot: int
    frame: int
    output: int
    size: int

    @classmethod
    def of(cls, order: int, frame_size: int) -> "DataLayout":
        cells = order * order
        frame = _align_up(2 * cells, 8)
        output = frame + frame_size * 8
        return cls(0, cells, frame, output, output + 2 * cells + order + 16)


class NativeCodeBackend:
    """Encodes a linked object module as x86-64 machine code.

    The operand stack is the hardware stack, so a push is a push and the
    register allocation problem never arises.  The runtime routines are
    reached by ``call``, whose return address lands on that same stack and is
    taken off again by ``ret``, leaving it balanced across the call.  Every
    other piece of state - canvas, snapshot, frame, output buffer - lives in
    .bss at a fixed address the prologue puts in r15, so the text needs no
    relocation beyond its own branches.
    """

    ORIENTATION_ENCODING: ClassVar[Mapping[Orientation, int]] = {
        Orientation.ROW: 0,
        Orientation.COLUMN: 1,
        Orientation.DIAGONAL: 2,
        Orientation.ANTIDIAGONAL: 3,
    }

    def __init__(self, module: ObjectModule) -> None:
        self._module = module
        self._order = module.order
        self._cells = module.order * module.order
        self._apothem = module.order // 2
        self._frame = max(module.frame_size, 1)
        self._layout = DataLayout.of(self._order, self._frame)
        self._group = SymmetryGroup.generated_by(
            module.family.generators,
            Coordinate(self._apothem, self._apothem),
            module.symmetry_order,
            presentation=f"{module.family.value}-{module.cardinality} about centroid",
        )

    @property
    def layout(self) -> DataLayout:
        return self._layout

    def _canvas(self, index: Register) -> MemoryOperand:
        return MemoryOperand(DATA_REGISTER, index, 1, self._layout.canvas)

    def _snapshot(self, index: Register) -> MemoryOperand:
        return MemoryOperand(DATA_REGISTER, index, 1, self._layout.snapshot)

    def _slot(self, slot: int) -> MemoryOperand:
        return MemoryOperand(DATA_REGISTER, None, 1, self._layout.frame + slot * 8)

    @woven
    def encode(self) -> bytes:
        """The whole text section: entry point, instruction stream, runtime."""
        with TRACER.span("encode", order=self._order):
            text = X86Assembler()
            text.label("_start")
            self._prologue(text)
            for address, instruction in enumerate(self._module.instructions):
                text.label(f"A{address}")
                self._instruction(text, instruction, address)
            text.label("exit")
            text.call("render")
            self._epilogue(text)
            self._paint(text)
            self._emit_run(text)
            self._snapshot_routine(text)
            self._apply(text)
            self._render(text)
            self._appendix(text)
            METRICS.increment("machine.instructions", len(self._module.instructions))
            return text.link()

    # -- the instruction stream -------------------------------------------
    #
    # Blocks are laid out in address order, so an instruction that falls
    # through to the next one needs no branch at all.

    def _instruction(
        self, text: X86Assembler, instruction: Instruction, address: int
    ) -> None:
        match instruction:
            case Halt():
                text.jump("exit")
            case Jump(target=target):
                text.jump(f"A{int(target)}")
            case JumpIfFalse(target=target):
                text.pop(Register.RAX)
                text.test(Register.RAX, Register.RAX)
                text.jump_if("e", f"A{int(target)}")
            case PushConstant(value=value):
                text.immediate(Register.RAX, value)
                text.push(Register.RAX)
            case LoadIntrinsic(name=name):
                text.immediate(Register.RAX, self._intrinsic(name))
                text.push(Register.RAX)
            case LoadLocal(slot=slot):
                text.load(Register.RAX, self._slot(slot))
                text.push(Register.RAX)
            case StoreLocal(slot=slot):
                text.pop(Register.RAX)
                text.store(self._slot(slot), Register.RAX)
            case BinaryAdd() | BinarySubtract() | BinaryMultiply():
                text.pop(Register.RCX)
                text.pop(Register.RAX)
                if isinstance(instruction, BinaryMultiply):
                    text.multiply(Register.RAX, Register.RCX)
                else:
                    operation = "add" if isinstance(instruction, BinaryAdd) else "sub"
                    text.arithmetic(operation, Register.RAX, Register.RCX)
                text.push(Register.RAX)
            case BinaryDivide():
                self._floor_divide(text, address)
            case Negate():
                text.pop(Register.RAX)
                text.negate(Register.RAX)
                text.push(Register.RAX)
            case CompareLessEqual():
                text.pop(Register.RCX)
                text.pop(Register.RAX)
                text.arithmetic("cmp", Register.RAX, Register.RCX)
                text.set_if("le", Register.RAX)
                text.widen_octet(Register.RAX, Register.RAX)
                text.push(Register.RAX)
            case MakeInterval():
                pass
            case EmitOrientedRun(orientation=orientation):
                text.pop(Register.RCX)
                text.pop(Register.RDX)
                text.pop(Register.RSI)
                text.immediate(Register.RDI, self.ORIENTATION_ENCODING[orientation])
                text.call("emit_run")
            case CloseUnderGroup():
                text.call("snapshot")
                for element in self._group.elements:
                    linear = element.linear
                    text.immediate(Register.RDI, linear.a)
                    text.immediate(Register.RSI, linear.b)
                    text.immediate(Register.RDX, linear.c)
                    text.immediate(Register.RCX, linear.d)
                    text.call("apply")
            case _:
                raise MachineCodeError(f"unencodable instruction {instruction!r}")

    def _intrinsic(self, name: str) -> int:
        try:
            return {
                "zero": 0,
                "apothem": self._apothem,
                "extremum": self._order - 1,
                "magnitude": self._order,
            }[name]
        except KeyError as exc:
            raise MachineCodeError(f"unbound intrinsic {name!r}") from exc

    def _floor_divide(self, text: X86Assembler, address: int) -> None:
        """idiv truncates towards zero where the interpreter floors.

        The quotient is decremented whenever the division was inexact and the
        remainder and the divisor disagree about sign, which xor detects in
        the sign bit.
        """
        floored = f"A{address}.floored"
        text.pop(Register.RCX)
        text.pop(Register.RAX)
        text.sign_extend()
        text.divide(Register.RCX)
        text.test(Register.RDX, Register.RDX)
        text.jump_if("e", floored)
        text.arithmetic("xor", Register.RDX, Register.RCX)
        text.jump_if("ns", floored)
        text.decrement(Register.RAX)
        text.label(floored)
        text.push(Register.RAX)

    # -- what a kernel underneath is asked for -----------------------------

    DATA_ORIGIN: ClassVar[int] = DATA_BASE

    def _prologue(self, text: X86Assembler) -> None:
        """Where .bss is; the loader has already zeroed it."""
        text.immediate(DATA_REGISTER, self.DATA_ORIGIN)

    def _epilogue(self, text: X86Assembler) -> None:
        text.immediate(Register.RAX, SYS_EXIT_GROUP)
        text.arithmetic("xor", Register.RDI, Register.RDI)
        text.syscall()

    def _flush(self, text: X86Assembler) -> None:
        text.immediate(Register.RAX, SYS_WRITE)
        text.immediate(Register.RDI, 1)
        text.address_of(
            Register.RSI, MemoryOperand(DATA_REGISTER, None, 1, self._layout.output)
        )
        text.load(Register.RDX, Register.R14)
        text.syscall()

    def _appendix(self, text: X86Assembler) -> None:
        """Routines a platform needs and this one does not."""

    # -- the runtime the instruction stream calls into ---------------------

    def _paint(self, text: X86Assembler) -> None:
        """paint(rdi = row, rsi = column), out of bounds silently ignored.

        Clobbers rax and nothing else, which is what lets its callers keep
        their loop state in registers across the call.
        """
        text.label("paint")
        for register in (Register.RDI, Register.RSI):
            text.arithmetic_immediate("cmp", register, 0)
            text.jump_if("l", "paint.done")
            text.arithmetic_immediate("cmp", register, self._order)
            text.jump_if("ge", "paint.done")
        text.multiply_immediate(Register.RAX, Register.RDI, self._order)
        text.arithmetic("add", Register.RAX, Register.RSI)
        text.store_octet_immediate(self._canvas(Register.RAX), 1)
        text.label("paint.done")
        text.ret()

    def _emit_run(self, text: X86Assembler) -> None:
        """emit_run(rdi = orientation, rsi = index, rdx = lower, rcx = upper)."""
        text.label("emit_run")
        text.load(Register.R8, Register.RDI)
        text.load(Register.R9, Register.RSI)
        text.load(Register.R10, Register.RDX)
        text.load(Register.R11, Register.RCX)
        text.label("emit_run.head")
        text.arithmetic("cmp", Register.R10, Register.R11)
        text.jump_if("g", "emit_run.done")
        for encoding, name in ((1, "column"), (2, "diagonal"), (3, "antidiagonal")):
            text.arithmetic_immediate("cmp", Register.R8, encoding)
            text.jump_if("e", f"emit_run.{name}")
        text.load(Register.RDI, Register.R9)
        text.load(Register.RSI, Register.R10)
        text.jump("emit_run.paint")
        text.label("emit_run.column")
        text.load(Register.RDI, Register.R10)
        text.load(Register.RSI, Register.R9)
        text.jump("emit_run.paint")
        text.label("emit_run.diagonal")
        text.load(Register.RDI, Register.R10)
        text.load(Register.RSI, Register.R10)
        text.arithmetic("add", Register.RSI, Register.R9)
        text.jump("emit_run.paint")
        text.label("emit_run.antidiagonal")
        text.load(Register.RDI, Register.R10)
        text.load(Register.RSI, Register.R9)
        text.arithmetic("sub", Register.RSI, Register.R10)
        text.label("emit_run.paint")
        text.call("paint")
        text.increment(Register.R10)
        text.jump("emit_run.head")
        text.label("emit_run.done")
        text.ret()

    def _snapshot_routine(self, text: X86Assembler) -> None:
        """The canvas as it stood before the closure began."""
        text.label("snapshot")
        text.arithmetic("xor", Register.RAX, Register.RAX)
        text.label("snapshot.head")
        text.arithmetic_immediate("cmp", Register.RAX, self._cells)
        text.jump_if("ge", "snapshot.done")
        text.load_octet(Register.RCX, self._canvas(Register.RAX))
        text.store_octet(self._snapshot(Register.RAX), Register.RCX)
        text.increment(Register.RAX)
        text.jump("snapshot.head")
        text.label("snapshot.done")
        text.ret()

    def _apply(self, text: X86Assembler) -> None:
        """apply(rdi = a, rsi = b, rdx = c, rcx = d): one group element."""
        text.label("apply")
        text.load(Register.R8, Register.RDI)
        text.load(Register.R9, Register.RSI)
        text.load(Register.R10, Register.RDX)
        text.load(Register.R11, Register.RCX)
        text.arithmetic("xor", Register.R12, Register.R12)
        text.label("apply.row")
        text.arithmetic_immediate("cmp", Register.R12, self._order)
        text.jump_if("ge", "apply.done")
        text.arithmetic("xor", Register.R13, Register.R13)
        text.label("apply.column")
        text.arithmetic_immediate("cmp", Register.R13, self._order)
        text.jump_if("ge", "apply.row.step")
        text.multiply_immediate(Register.RAX, Register.R12, self._order)
        text.arithmetic("add", Register.RAX, Register.R13)
        text.compare_octet_immediate(self._snapshot(Register.RAX), 0)
        text.jump_if("e", "apply.column.step")
        text.load(Register.RDI, Register.R12)
        text.arithmetic_immediate("sub", Register.RDI, self._apothem)
        text.load(Register.RSI, Register.R13)
        text.arithmetic_immediate("sub", Register.RSI, self._apothem)
        text.load(Register.RAX, Register.R8)
        text.multiply(Register.RAX, Register.RDI)
        text.load(Register.RDX, Register.R9)
        text.multiply(Register.RDX, Register.RSI)
        text.arithmetic("add", Register.RAX, Register.RDX)
        text.arithmetic_immediate("add", Register.RAX, self._apothem)
        text.load(Register.RCX, Register.R10)
        text.multiply(Register.RCX, Register.RDI)
        text.load(Register.RDX, Register.R11)
        text.multiply(Register.RDX, Register.RSI)
        text.arithmetic("add", Register.RCX, Register.RDX)
        text.arithmetic_immediate("add", Register.RCX, self._apothem)
        text.load(Register.RDI, Register.RAX)
        text.load(Register.RSI, Register.RCX)
        text.call("paint")
        text.label("apply.column.step")
        text.increment(Register.R13)
        text.jump("apply.column")
        text.label("apply.row.step")
        text.increment(Register.R12)
        text.jump("apply.row")
        text.label("apply.done")
        text.ret()

    def _render(self, text: X86Assembler) -> None:
        """The canvas as text, trailing blanks trimmed, in a single write(2)."""
        cursor = MemoryOperand(DATA_REGISTER, Register.R14, 1, self._layout.output)
        text.label("render")
        text.arithmetic("xor", Register.R12, Register.R12)
        text.arithmetic("xor", Register.R14, Register.R14)
        text.label("render.row")
        text.arithmetic_immediate("cmp", Register.R12, self._order)
        text.jump_if("ge", "render.flush")
        text.immediate(Register.R10, -1)
        text.arithmetic("xor", Register.R13, Register.R13)
        text.label("render.scan")
        text.arithmetic_immediate("cmp", Register.R13, self._order)
        text.jump_if("ge", "render.print")
        text.multiply_immediate(Register.RAX, Register.R12, self._order)
        text.arithmetic("add", Register.RAX, Register.R13)
        text.compare_octet_immediate(self._canvas(Register.RAX), 0)
        text.jump_if("e", "render.scan.step")
        text.load(Register.R10, Register.R13)
        text.label("render.scan.step")
        text.increment(Register.R13)
        text.jump("render.scan")
        text.label("render.print")
        text.arithmetic("xor", Register.R13, Register.R13)
        text.label("render.cell")
        text.arithmetic("cmp", Register.R13, Register.R10)
        text.jump_if("g", "render.newline")
        text.arithmetic_immediate("cmp", Register.R13, 0)
        text.jump_if("le", "render.ink")
        text.store_octet_immediate(cursor, ord(" "))
        text.increment(Register.R14)
        text.label("render.ink")
        text.multiply_immediate(Register.RAX, Register.R12, self._order)
        text.arithmetic("add", Register.RAX, Register.R13)
        text.compare_octet_immediate(self._canvas(Register.RAX), 0)
        text.jump_if("e", "render.blank")
        text.store_octet_immediate(cursor, ord("*"))
        text.jump("render.advance")
        text.label("render.blank")
        text.store_octet_immediate(cursor, ord(" "))
        text.label("render.advance")
        text.increment(Register.R14)
        text.increment(Register.R13)
        text.jump("render.cell")
        text.label("render.newline")
        text.store_octet_immediate(cursor, ord("\n"))
        text.increment(Register.R14)
        text.increment(Register.R12)
        text.jump("render.row")
        text.label("render.flush")
        self._flush(text)
        text.ret()


ARM_PAGE: Final[int] = 0x10000
ARM_SYS_WRITE: Final[int] = 64
ARM_SYS_EXIT_GROUP: Final[int] = 94
EM_X86_64: Final[int] = 62
EM_AARCH64: Final[int] = 183

# Condition codes, in their encoding order.
CONDITIONS: Final[Mapping[str, int]] = {
    "eq": 0, "ne": 1, "hs": 2, "lo": 3, "mi": 4, "pl": 5, "vs": 6, "vc": 7,
    "hi": 8, "ls": 9, "ge": 10, "lt": 11, "gt": 12, "le": 13, "al": 14,
}

ZERO: Final[int] = 31          # xzr where a register is read
STACK: Final[int] = 31         # sp where a base register is read
DATA: Final[int] = 12          # the .bss base, put there by the prologue
LINK_SAVE: Final[int] = 13     # x30 parked across the one level of nesting


class Aarch64Assembler:
    """An aarch64 encoder: the instructions the machine needs, and no more.

    Every instruction is a word, so there is no ModRM problem to have and no
    variable length to track: a branch site is a word index and the fixup is
    a bitfield.  Only the width of that bitfield differs, and only between the
    two branch forms.
    """

    def __init__(self) -> None:
        self._code = bytearray()
        self._labels: dict[str, int] = {}
        self._fixups: list[tuple[int, str, str]] = []

    def __len__(self) -> int:
        return len(self._code)

    # -- labels and relocation --------------------------------------------

    def label(self, name: str) -> None:
        if name in self._labels:
            raise MachineCodeError(f"duplicate label {name!r}")
        self._labels[name] = len(self._code)

    def link(self) -> bytes:
        """Patches every branch, then freezes the text."""
        for site, name, kind in self._fixups:
            try:
                target = self._labels[name]
            except KeyError as exc:
                raise MachineCodeError(f"unresolved branch target {name!r}") from exc
            offset = (target - site) // 4
            word = struct.unpack_from("<I", self._code, site)[0]
            if kind == "imm26":
                if not -(1 << 25) <= offset < (1 << 25):
                    raise MachineCodeError(f"branch to {name!r} is out of range")
                word |= offset & 0x3FFFFFF
            else:
                if not -(1 << 18) <= offset < (1 << 18):
                    raise MachineCodeError(f"branch to {name!r} is out of range")
                word |= (offset & 0x7FFFF) << 5
            struct.pack_into("<I", self._code, site, word)
        return bytes(self._code)

    def _word(self, value: int) -> None:
        self._code.extend(struct.pack("<I", value & 0xFFFFFFFF))

    def _branch_site(self, word: int, name: str, kind: str) -> None:
        self._fixups.append((len(self._code), name, kind))
        self._word(word)

    # -- moving values ----------------------------------------------------

    def move(self, destination: int, source: int) -> None:
        """mov Xd, Xn, which is orr Xd, xzr, Xn."""
        self._word(0xAA0003E0 | (source << 16) | destination)

    def immediate(self, destination: int, value: int) -> None:
        """The 16 bits at a time that a fixed-width instruction can carry."""
        unsigned = value & 0xFFFFFFFFFFFFFFFF
        halfwords = [(unsigned >> shift) & 0xFFFF for shift in (0, 16, 32, 48)]
        if value < 0:
            # movn writes the complement, which spares three movk for the
            # common small negative.
            inverted = [half ^ 0xFFFF for half in halfwords]
            first = next((i for i, half in enumerate(inverted) if half), 0)
            self._word(0x92800000 | (first << 21) | (inverted[first] << 5) | destination)
            for index, half in enumerate(halfwords):
                if index != first and half != 0xFFFF:
                    self._word(0xF2800000 | (index << 21) | (half << 5) | destination)
            return
        if not unsigned:
            self._word(0xD2800000 | destination)
            return
        written = False
        for index, half in enumerate(halfwords):
            if half:
                self._word(
                    (0xF2800000 if written else 0xD2800000)
                    | (index << 21) | (half << 5) | destination
                )
                written = True

    # -- arithmetic -------------------------------------------------------

    def arithmetic(self, operation: str, destination: int, left: int, right: int) -> None:
        opcodes = {"add": 0x8B000000, "sub": 0xCB000000, "eor": 0xCA000000,
                   "subs": 0xEB000000}
        self._word(opcodes[operation] | (right << 16) | (left << 5) | destination)

    def arithmetic_immediate(
        self, operation: str, destination: int, left: int, value: int, scratch: int = 24
    ) -> None:
        """add/sub with a 12-bit immediate, or through ``scratch`` when wider."""
        opcodes = {"add": 0x91000000, "sub": 0xD1000000, "subs": 0xF1000000}
        if value < 0:
            opposite = {"add": "sub", "sub": "add"}.get(operation)
            if opposite is not None:
                return self.arithmetic_immediate(opposite, destination, left, -value, scratch)
        if 0 <= value < 4096:
            self._word(opcodes[operation] | (value << 10) | (left << 5) | destination)
            return
        if 0 <= value < (4096 << 12) and not value & 0xFFF:
            self._word(
                opcodes[operation] | (1 << 22) | ((value >> 12) << 10) | (left << 5) | destination
            )
            return
        self.immediate(scratch, value)
        self.arithmetic({"add": "add", "sub": "sub", "subs": "subs"}[operation],
                        destination, left, scratch)

    def compare(self, left: int, right: int) -> None:
        self.arithmetic("subs", ZERO, left, right)

    def compare_immediate(self, left: int, value: int, scratch: int = 24) -> None:
        self.arithmetic_immediate("subs", ZERO, left, value, scratch)

    def multiply(self, destination: int, left: int, right: int) -> None:
        """madd Xd, Xn, Xm, xzr."""
        self._word(0x9B007C00 | (right << 16) | (left << 5) | destination)

    def divide(self, destination: int, left: int, right: int) -> None:
        self._word(0x9AC00C00 | (right << 16) | (left << 5) | destination)

    def multiply_subtract(self, destination: int, left: int, right: int, minuend: int) -> None:
        """msub Xd, Xn, Xm, Xa: Xa - Xn * Xm, which is the remainder."""
        self._word(0x9B008000 | (right << 16) | (minuend << 10) | (left << 5) | destination)

    def negate(self, destination: int, source: int) -> None:
        self.arithmetic("sub", destination, ZERO, source)

    def set_if(self, condition: str, destination: int) -> None:
        """cset Xd, cond, which is csinc Xd, xzr, xzr, inverted."""
        inverted = CONDITIONS[condition] ^ 1
        self._word(0x9A9F07E0 | (inverted << 12) | destination)

    # -- memory -----------------------------------------------------------

    def load(self, destination: int, base: int, offset: int = 0) -> None:
        self._word(0xF9400000 | ((offset // 8) << 10) | (base << 5) | destination)

    def store(self, source: int, base: int, offset: int = 0) -> None:
        self._word(0xF9000000 | ((offset // 8) << 10) | (base << 5) | source)

    def load_octet(self, destination: int, base: int, index: int) -> None:
        self._word(0x38606800 | (index << 16) | (base << 5) | destination)

    def store_octet(self, source: int, base: int, index: int) -> None:
        self._word(0x38206800 | (index << 16) | (base << 5) | source)

    def push(self, register: int) -> None:
        """str Xt, [sp, #-16]!  Sixteen keeps sp aligned where eight would not."""
        self._word(0xF8000C00 | ((-16 & 0x1FF) << 12) | (STACK << 5) | register)

    def pop(self, register: int) -> None:
        """ldr Xt, [sp], #16."""
        self._word(0xF8400400 | ((16 & 0x1FF) << 12) | (STACK << 5) | register)

    # -- control ----------------------------------------------------------

    def jump(self, name: str) -> None:
        self._branch_site(0x14000000, name, "imm26")

    def jump_if(self, condition: str, name: str) -> None:
        self._branch_site(0x54000000 | CONDITIONS[condition], name, "imm19")

    def jump_if_zero(self, register: int, name: str) -> None:
        self._branch_site(0xB4000000 | register, name, "imm19")

    def call(self, name: str) -> None:
        self._branch_site(0x94000000, name, "imm26")

    def ret(self) -> None:
        self._word(0xD65F03C0)

    def trap(self) -> None:
        self._word(0xD4200000)

    def syscall(self) -> None:
        self._word(0xD4000001)


class Aarch64CodeBackend:
    """Encodes a linked object module as aarch64 machine code.

    The same shape as the x86-64 backend: the operand stack is the hardware
    stack, the runtime is reached by call, and canvas, snapshot, frame and
    output sit in .bss at a fixed address the prologue puts in a register.

    Three things differ.  A push moves the stack sixteen octets rather than
    eight, because sp has to stay aligned to sixteen for anything to be able
    to address off it.  A call leaves its return address in a register rather
    than on that stack, so the two routines that call another one park it,
    which is also why there is only ever one level to park.  And division
    answers zero where the other machine faults, so the divisor is tested
    first and the trap is written out.
    """

    ORIENTATION_ENCODING: ClassVar[Mapping[Orientation, int]] = {
        Orientation.ROW: 0,
        Orientation.COLUMN: 1,
        Orientation.DIAGONAL: 2,
        Orientation.ANTIDIAGONAL: 3,
    }

    def __init__(self, module: ObjectModule) -> None:
        self._module = module
        self._order = module.order
        self._cells = module.order * module.order
        self._apothem = module.order // 2
        self._frame = max(module.frame_size, 1)
        self._layout = DataLayout.of(self._order, self._frame)
        self._group = SymmetryGroup.generated_by(
            module.family.generators,
            Coordinate(self._apothem, self._apothem),
            module.symmetry_order,
            presentation=f"{module.family.value}-{module.cardinality} about centroid",
        )

    @property
    def layout(self) -> DataLayout:
        return self._layout

    @woven
    def encode(self) -> bytes:
        """The whole text section: entry point, instruction stream, runtime."""
        text = Aarch64Assembler()
        text.label("_start")
        text.immediate(DATA, DATA_BASE)
        for address, instruction in enumerate(self._module.instructions):
            text.label(f"A{address}")
            self._instruction(text, instruction, address)
        text.label("exit")
        text.call("render")
        text.immediate(8, ARM_SYS_EXIT_GROUP)
        text.immediate(0, 0)
        text.syscall()
        text.label("divide.by.zero")
        text.trap()
        self._paint(text)
        self._emit_run(text)
        self._snapshot(text)
        self._apply(text)
        self._render(text)
        return text.link()

    def _intrinsic(self, name: str) -> int:
        try:
            return {
                "zero": 0,
                "apothem": self._apothem,
                "extremum": self._order - 1,
                "magnitude": self._order,
            }[name]
        except KeyError as exc:
            raise MachineCodeError(f"unbound intrinsic {name!r}") from exc

    def _instruction(
        self, text: Aarch64Assembler, instruction: Instruction, address: int
    ) -> None:
        match instruction:
            case Halt():
                text.jump("exit")
            case Jump(target=target):
                text.jump(f"A{int(target)}")
            case JumpIfFalse(target=target):
                text.pop(0)
                text.compare_immediate(0, 0)
                text.jump_if("eq", f"A{int(target)}")
            case PushConstant(value=value):
                text.immediate(0, value)
                text.push(0)
            case LoadIntrinsic(name=name):
                text.immediate(0, self._intrinsic(name))
                text.push(0)
            case LoadLocal(slot=slot):
                text.arithmetic_immediate("add", 9, DATA, self._layout.frame + slot * 8)
                text.load(0, 9)
                text.push(0)
            case StoreLocal(slot=slot):
                text.pop(0)
                text.arithmetic_immediate("add", 9, DATA, self._layout.frame + slot * 8)
                text.store(0, 9)
            case BinaryAdd() | BinarySubtract():
                text.pop(1)
                text.pop(0)
                text.arithmetic("add" if isinstance(instruction, BinaryAdd) else "sub", 0, 0, 1)
                text.push(0)
            case BinaryMultiply():
                text.pop(1)
                text.pop(0)
                text.multiply(0, 0, 1)
                text.push(0)
            case BinaryDivide():
                self._floor_divide(text, address)
            case Negate():
                text.pop(0)
                text.negate(0, 0)
                text.push(0)
            case CompareLessEqual():
                text.pop(1)
                text.pop(0)
                text.compare(0, 1)
                text.set_if("le", 0)
                text.push(0)
            case MakeInterval():
                pass
            case EmitOrientedRun(orientation=orientation):
                text.pop(3)
                text.pop(2)
                text.pop(1)
                text.immediate(0, self.ORIENTATION_ENCODING[orientation])
                text.call("emit_run")
            case CloseUnderGroup():
                text.call("snapshot")
                for element in self._group.elements:
                    linear = element.linear
                    text.immediate(0, linear.a)
                    text.immediate(1, linear.b)
                    text.immediate(2, linear.c)
                    text.immediate(3, linear.d)
                    text.call("apply")
            case _:
                raise MachineCodeError(f"unencodable instruction {instruction!r}")

    def _floor_divide(self, text: Aarch64Assembler, address: int) -> None:
        """sdiv truncates towards zero where the interpreter floors."""
        floored = f"A{address}.floored"
        text.pop(1)
        text.pop(0)
        text.jump_if_zero(1, "divide.by.zero")
        text.divide(2, 0, 1)
        text.multiply_subtract(3, 2, 1, 0)
        text.compare_immediate(3, 0)
        text.jump_if("eq", floored)
        text.arithmetic("eor", 3, 3, 1)
        text.compare_immediate(3, 0)
        text.jump_if("ge", floored)
        text.arithmetic_immediate("sub", 2, 2, 1)
        text.label(floored)
        text.push(2)

    # -- the runtime the instruction stream calls into ---------------------

    def _paint(self, text: Aarch64Assembler) -> None:
        """paint(x0 = row, x1 = column), out of bounds silently ignored."""
        text.label("paint")
        for register in (0, 1):
            text.compare_immediate(register, 0)
            text.jump_if("lt", "paint.done")
            text.compare_immediate(register, self._order)
            text.jump_if("ge", "paint.done")
        text.immediate(21, self._order)
        text.multiply(21, 0, 21)
        text.arithmetic("add", 21, 21, 1)
        text.arithmetic_immediate("add", 22, DATA, self._layout.canvas)
        text.immediate(23, 1)
        text.store_octet(23, 22, 21)
        text.label("paint.done")
        text.ret()

    def _emit_run(self, text: Aarch64Assembler) -> None:
        """emit_run(x0 = orientation, x1 = index, x2 = lower, x3 = upper)."""
        text.label("emit_run")
        text.move(LINK_SAVE, 30)
        text.move(5, 0)
        text.move(6, 1)
        text.move(7, 2)
        text.move(9, 3)
        text.label("emit_run.head")
        text.compare(7, 9)
        text.jump_if("gt", "emit_run.done")
        for encoding, name in ((1, "column"), (2, "diagonal"), (3, "antidiagonal")):
            text.compare_immediate(5, encoding)
            text.jump_if("eq", f"emit_run.{name}")
        text.move(0, 6)
        text.move(1, 7)
        text.jump("emit_run.paint")
        text.label("emit_run.column")
        text.move(0, 7)
        text.move(1, 6)
        text.jump("emit_run.paint")
        text.label("emit_run.diagonal")
        text.move(0, 7)
        text.arithmetic("add", 1, 7, 6)
        text.jump("emit_run.paint")
        text.label("emit_run.antidiagonal")
        text.move(0, 7)
        text.arithmetic("sub", 1, 6, 7)
        text.label("emit_run.paint")
        text.call("paint")
        text.arithmetic_immediate("add", 7, 7, 1)
        text.jump("emit_run.head")
        text.label("emit_run.done")
        text.move(30, LINK_SAVE)
        text.ret()

    def _snapshot(self, text: Aarch64Assembler) -> None:
        """The canvas as it stood before the closure began."""
        text.label("snapshot")
        text.arithmetic_immediate("add", 9, DATA, self._layout.canvas)
        text.arithmetic_immediate("add", 10, DATA, self._layout.snapshot)
        text.immediate(5, 0)
        text.label("snapshot.head")
        text.compare_immediate(5, self._cells)
        text.jump_if("ge", "snapshot.done")
        text.load_octet(7, 9, 5)
        text.store_octet(7, 10, 5)
        text.arithmetic_immediate("add", 5, 5, 1)
        text.jump("snapshot.head")
        text.label("snapshot.done")
        text.ret()

    def _apply(self, text: Aarch64Assembler) -> None:
        """apply(x0 = a, x1 = b, x2 = c, x3 = d): one group element."""
        text.label("apply")
        text.move(LINK_SAVE, 30)
        text.move(5, 0)
        text.move(6, 1)
        text.move(7, 2)
        text.move(9, 3)
        text.arithmetic_immediate("add", 10, DATA, self._layout.snapshot)
        text.immediate(11, 0)
        text.label("apply.row")
        text.compare_immediate(11, self._order)
        text.jump_if("ge", "apply.done")
        text.immediate(15, 0)
        text.label("apply.column")
        text.compare_immediate(15, self._order)
        text.jump_if("ge", "apply.row.step")
        text.immediate(16, self._order)
        text.multiply(16, 11, 16)
        text.arithmetic("add", 16, 16, 15)
        text.load_octet(17, 10, 16)
        text.compare_immediate(17, 0)
        text.jump_if("eq", "apply.column.step")
        text.arithmetic_immediate("sub", 19, 11, self._apothem)
        text.arithmetic_immediate("sub", 20, 15, self._apothem)
        text.multiply(16, 5, 19)
        text.multiply(17, 6, 20)
        text.arithmetic("add", 16, 16, 17)
        text.arithmetic_immediate("add", 0, 16, self._apothem)
        text.multiply(16, 7, 19)
        text.multiply(17, 9, 20)
        text.arithmetic("add", 16, 16, 17)
        text.arithmetic_immediate("add", 1, 16, self._apothem)
        text.call("paint")
        text.label("apply.column.step")
        text.arithmetic_immediate("add", 15, 15, 1)
        text.jump("apply.column")
        text.label("apply.row.step")
        text.arithmetic_immediate("add", 11, 11, 1)
        text.jump("apply.row")
        text.label("apply.done")
        text.move(30, LINK_SAVE)
        text.ret()

    def _render(self, text: Aarch64Assembler) -> None:
        """The canvas as text, trailing blanks trimmed, in a single write."""
        text.label("render")
        text.arithmetic_immediate("add", 9, DATA, self._layout.canvas)
        text.arithmetic_immediate("add", 10, DATA, self._layout.output)
        text.immediate(11, 0)
        text.immediate(14, 0)
        text.label("render.row")
        text.compare_immediate(11, self._order)
        text.jump_if("ge", "render.flush")
        text.immediate(16, -1)
        text.immediate(15, 0)
        text.label("render.scan")
        text.compare_immediate(15, self._order)
        text.jump_if("ge", "render.print")
        text.immediate(17, self._order)
        text.multiply(17, 11, 17)
        text.arithmetic("add", 17, 17, 15)
        text.load_octet(17, 9, 17)
        text.compare_immediate(17, 0)
        text.jump_if("eq", "render.scan.step")
        text.move(16, 15)
        text.label("render.scan.step")
        text.arithmetic_immediate("add", 15, 15, 1)
        text.jump("render.scan")
        text.label("render.print")
        text.immediate(15, 0)
        text.label("render.cell")
        text.compare(15, 16)
        text.jump_if("gt", "render.newline")
        text.compare_immediate(15, 0)
        text.jump_if("le", "render.ink")
        text.immediate(17, ord(" "))
        text.store_octet(17, 10, 14)
        text.arithmetic_immediate("add", 14, 14, 1)
        text.label("render.ink")
        text.immediate(17, self._order)
        text.multiply(17, 11, 17)
        text.arithmetic("add", 17, 17, 15)
        text.load_octet(17, 9, 17)
        text.compare_immediate(17, 0)
        text.jump_if("eq", "render.blank")
        text.immediate(17, ord("*"))
        text.jump("render.advance")
        text.label("render.blank")
        text.immediate(17, ord(" "))
        text.label("render.advance")
        text.store_octet(17, 10, 14)
        text.arithmetic_immediate("add", 14, 14, 1)
        text.arithmetic_immediate("add", 15, 15, 1)
        text.jump("render.cell")
        text.label("render.newline")
        text.immediate(17, ord("\n"))
        text.store_octet(17, 10, 14)
        text.arithmetic_immediate("add", 14, 14, 1)
        text.arithmetic_immediate("add", 11, 11, 1)
        text.jump("render.row")
        text.label("render.flush")
        text.immediate(8, ARM_SYS_WRITE)
        text.immediate(0, 1)
        text.arithmetic_immediate("add", 1, DATA, self._layout.output)
        text.move(2, 14)
        text.syscall()
        text.ret()


EM_RISCV: Final[int] = 243
RISCV_SYS_WRITE: Final[int] = 64
RISCV_SYS_EXIT_GROUP: Final[int] = 93

RISCV_ZERO: Final[int] = 0
RISCV_RETURN: Final[int] = 1          # ra, where a call leaves its return address
RISCV_STACK: Final[int] = 2           # sp
RISCV_DATA: Final[int] = 18           # the .bss base, put there by the prologue
RISCV_LINK_SAVE: Final[int] = 19      # ra parked across the one level of nesting
RISCV_SCRATCH: Final[int] = 31        # t6, borrowed by the wide-immediate forms
RISCV_SYSCALL: Final[int] = 17        # a7

# The branch this machine has is the opposite of the one wanted, jumped over.
# Each entry is the funct3 of that opposite and whether it reads its operands
# the other way round.
RISCV_INVERTED: Final[Mapping[str, tuple[int, bool]]] = {
    "eq": (0b001, False), "ne": (0b000, False),
    "lt": (0b101, False), "ge": (0b100, False),
    "gt": (0b101, True), "le": (0b100, True),
}


class Riscv64Assembler:
    """A riscv64 encoder: the instructions the machine needs, and no more.

    Fixed width like the aarch64 one, so a fixup is still a bitfield, but the
    bits of a branch offset are scattered through the word rather than lying
    in one field.  Conditional branches reach four kilooctets and the
    instruction stream can be longer than that, so every one of them is
    written as its own opposite jumped over an unconditional jump, which
    reaches a megaoctet and costs a word.
    """

    def __init__(self) -> None:
        self._code = bytearray()
        self._labels: dict[str, int] = {}
        self._fixups: list[tuple[int, str]] = []

    def __len__(self) -> int:
        return len(self._code)

    # -- labels and relocation --------------------------------------------

    def label(self, name: str) -> None:
        if name in self._labels:
            raise MachineCodeError(f"duplicate label {name!r}")
        self._labels[name] = len(self._code)

    def link(self) -> bytes:
        """Patches every jump, then freezes the text."""
        for site, name in self._fixups:
            try:
                target = self._labels[name]
            except KeyError as exc:
                raise MachineCodeError(f"unresolved branch target {name!r}") from exc
            offset = target - site
            if not -(1 << 20) <= offset < (1 << 20):
                raise MachineCodeError(f"jump to {name!r} is out of range")
            word = struct.unpack_from("<I", self._code, site)[0]
            struct.pack_into("<I", self._code, site, word | self._jtype(offset))
        return bytes(self._code)

    @staticmethod
    def _jtype(offset: int) -> int:
        return (
            ((offset >> 20 & 1) << 31)
            | ((offset >> 1 & 0x3FF) << 21)
            | ((offset >> 11 & 1) << 20)
            | ((offset >> 12 & 0xFF) << 12)
        )

    def _word(self, value: int) -> None:
        self._code.extend(struct.pack("<I", value & 0xFFFFFFFF))

    # -- the five formats --------------------------------------------------

    def _r(self, funct7: int, rs2: int, rs1: int, funct3: int, rd: int, opcode: int) -> None:
        self._word((funct7 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode)

    def _i(self, value: int, rs1: int, funct3: int, rd: int, opcode: int) -> None:
        if not -2048 <= value < 2048:
            raise MachineCodeError(f"{value} does not fit a twelve-bit immediate")
        self._word(((value & 0xFFF) << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode)

    def _s(self, value: int, rs2: int, rs1: int, funct3: int, opcode: int) -> None:
        if not -2048 <= value < 2048:
            raise MachineCodeError(f"{value} does not fit a twelve-bit immediate")
        self._word(
            ((value >> 5 & 0x7F) << 25) | (rs2 << 20) | (rs1 << 15)
            | (funct3 << 12) | ((value & 0x1F) << 7) | opcode
        )

    def _b(self, offset: int, rs2: int, rs1: int, funct3: int) -> None:
        self._word(
            ((offset >> 12 & 1) << 31) | ((offset >> 5 & 0x3F) << 25) | (rs2 << 20)
            | (rs1 << 15) | (funct3 << 12) | ((offset >> 1 & 0xF) << 8)
            | ((offset >> 11 & 1) << 7) | 0b1100011
        )

    # -- moving values -----------------------------------------------------

    def move(self, destination: int, source: int) -> None:
        self._i(0, source, 0b000, destination, 0b0010011)

    def immediate(self, destination: int, value: int) -> None:
        """addi alone, or lui and addi.

        Layer 9 packs a constant as an i32, so nothing wider than this can
        reach the backend; anything that did would be a container that had
        already accepted what it could not hold.
        """
        if -2048 <= value < 2048:
            self._i(value, RISCV_ZERO, 0b000, destination, 0b0010011)
            return
        if not -(1 << 31) <= value < (1 << 31):
            raise MachineCodeError(f"{value} is wider than the container can carry")
        low = ((value & 0xFFF) ^ 0x800) - 0x800
        self._word((((value - low) >> 12 & 0xFFFFF) << 12) | (destination << 7) | 0b0110111)
        if low:
            self._i(low, destination, 0b000, destination, 0b0010011)

    # -- arithmetic --------------------------------------------------------

    OPERATIONS: ClassVar[Mapping[str, tuple[int, int]]] = {
        "add": (0b0000000, 0b000), "sub": (0b0100000, 0b000),
        "xor": (0b0000000, 0b100), "slt": (0b0000000, 0b010),
        "mul": (0b0000001, 0b000), "div": (0b0000001, 0b100),
        "rem": (0b0000001, 0b110),
    }

    def arithmetic(self, operation: str, destination: int, left: int, right: int) -> None:
        funct7, funct3 = self.OPERATIONS[operation]
        self._r(funct7, right, left, funct3, destination, 0b0110011)

    def arithmetic_immediate(
        self, destination: int, left: int, value: int, scratch: int = RISCV_SCRATCH
    ) -> None:
        """addi, or the value in a register and an add."""
        if -2048 <= value < 2048:
            self._i(value, left, 0b000, destination, 0b0010011)
            return
        self.immediate(scratch, value)
        self.arithmetic("add", destination, left, scratch)

    def exclusive_or_immediate(self, destination: int, source: int, value: int) -> None:
        self._i(value, source, 0b100, destination, 0b0010011)

    def negate(self, destination: int, source: int) -> None:
        self.arithmetic("sub", destination, RISCV_ZERO, source)

    # -- memory ------------------------------------------------------------

    def load(self, destination: int, base: int, offset: int = 0) -> None:
        self._i(offset, base, 0b011, destination, 0b0000011)

    def store(self, source: int, base: int, offset: int = 0) -> None:
        self._s(offset, source, base, 0b011, 0b0100011)

    def load_octet(self, destination: int, base: int, offset: int = 0) -> None:
        self._i(offset, base, 0b100, destination, 0b0000011)

    def store_octet(self, source: int, base: int, offset: int = 0) -> None:
        self._s(offset, source, base, 0b000, 0b0100011)

    def push(self, register: int) -> None:
        """Sixteen, not eight, because sp is aligned to sixteen here too."""
        self._i(-16, RISCV_STACK, 0b000, RISCV_STACK, 0b0010011)
        self.store(register, RISCV_STACK, 0)

    def pop(self, register: int) -> None:
        self.load(register, RISCV_STACK, 0)
        self._i(16, RISCV_STACK, 0b000, RISCV_STACK, 0b0010011)

    # -- control -----------------------------------------------------------

    def jump(self, name: str) -> None:
        self._fixups.append((len(self._code), name))
        self._word(0b1101111)

    def call(self, name: str) -> None:
        self._fixups.append((len(self._code), name))
        self._word((RISCV_RETURN << 7) | 0b1101111)

    def branch(self, condition: str, left: int, right: int, name: str) -> None:
        funct3, swapped = RISCV_INVERTED[condition]
        first, second = (right, left) if swapped else (left, right)
        self._b(8, second, first, funct3)
        self.jump(name)

    def branch_immediate(
        self, condition: str, left: int, value: int, name: str, scratch: int = RISCV_SCRATCH
    ) -> None:
        if value == 0:
            self.branch(condition, left, RISCV_ZERO, name)
            return
        self.immediate(scratch, value)
        self.branch(condition, left, scratch, name)

    def ret(self) -> None:
        self._i(0, RISCV_RETURN, 0b000, RISCV_ZERO, 0b1100111)

    def trap(self) -> None:
        self._word(0x00100073)

    def syscall(self) -> None:
        self._word(0x00000073)


A0, A1, A2, A3 = 10, 11, 12, 13
T0, T1, T2, T3 = 5, 6, 7, 28
S4, S5, S6, S7, S8, S9, S10, S11 = 20, 21, 22, 23, 24, 25, 26, 27


class Riscv64CodeBackend:
    """Encodes a linked object module as riscv64 machine code.

    The same shape as the other two: the operand stack is the hardware stack,
    the runtime is reached by call, and .bss sits at a fixed address the
    prologue puts in a register.  A call leaves its return address in a
    register as on aarch64, so the two routines that call another one park it.

    What is its own is the addressing.  There is no register-plus-register
    form to load or store through, so an index is added into a pointer first
    and every access reads at zero from that.  And div answers minus one where
    the divisor is zero rather than faulting, so the divisor is tested and the
    trap is written out.
    """

    ORIENTATION_ENCODING: ClassVar[Mapping[Orientation, int]] = {
        Orientation.ROW: 0,
        Orientation.COLUMN: 1,
        Orientation.DIAGONAL: 2,
        Orientation.ANTIDIAGONAL: 3,
    }

    def __init__(self, module: ObjectModule) -> None:
        self._module = module
        self._order = module.order
        self._cells = module.order * module.order
        self._apothem = module.order // 2
        self._frame = max(module.frame_size, 1)
        self._layout = DataLayout.of(self._order, self._frame)
        self._group = SymmetryGroup.generated_by(
            module.family.generators,
            Coordinate(self._apothem, self._apothem),
            module.symmetry_order,
            presentation=f"{module.family.value}-{module.cardinality} about centroid",
        )

    @property
    def layout(self) -> DataLayout:
        return self._layout

    @woven
    def encode(self) -> bytes:
        """The whole text section: entry point, instruction stream, runtime."""
        text = Riscv64Assembler()
        text.label("_start")
        text.immediate(RISCV_DATA, DATA_BASE)
        for address, instruction in enumerate(self._module.instructions):
            text.label(f"A{address}")
            self._instruction(text, instruction, address)
        text.label("exit")
        text.call("render")
        text.immediate(RISCV_SYSCALL, RISCV_SYS_EXIT_GROUP)
        text.immediate(A0, 0)
        text.syscall()
        text.label("divide.by.zero")
        text.trap()
        self._paint(text)
        self._emit_run(text)
        self._snapshot(text)
        self._apply(text)
        self._render(text)
        return text.link()

    @staticmethod
    def _address(text: Riscv64Assembler, destination: int, index: int, offset: int) -> None:
        """destination = the .bss base, plus a region's offset, plus an index."""
        text.arithmetic("add", destination, RISCV_DATA, index)
        if offset:
            text.arithmetic_immediate(destination, destination, offset)

    def _intrinsic(self, name: str) -> int:
        try:
            return {
                "zero": 0,
                "apothem": self._apothem,
                "extremum": self._order - 1,
                "magnitude": self._order,
            }[name]
        except KeyError as exc:
            raise MachineCodeError(f"unbound intrinsic {name!r}") from exc

    def _instruction(
        self, text: Riscv64Assembler, instruction: Instruction, address: int
    ) -> None:
        match instruction:
            case Halt():
                text.jump("exit")
            case Jump(target=target):
                text.jump(f"A{int(target)}")
            case JumpIfFalse(target=target):
                text.pop(A0)
                text.branch("eq", A0, RISCV_ZERO, f"A{int(target)}")
            case PushConstant(value=value):
                text.immediate(A0, value)
                text.push(A0)
            case LoadIntrinsic(name=name):
                text.immediate(A0, self._intrinsic(name))
                text.push(A0)
            case LoadLocal(slot=slot):
                self._address(text, T0, RISCV_ZERO, self._layout.frame + slot * 8)
                text.load(A0, T0)
                text.push(A0)
            case StoreLocal(slot=slot):
                text.pop(A0)
                self._address(text, T0, RISCV_ZERO, self._layout.frame + slot * 8)
                text.store(A0, T0)
            case BinaryAdd() | BinarySubtract() | BinaryMultiply():
                text.pop(A1)
                text.pop(A0)
                text.arithmetic(
                    {BinaryAdd: "add", BinarySubtract: "sub", BinaryMultiply: "mul"}[
                        type(instruction)
                    ],
                    A0, A0, A1,
                )
                text.push(A0)
            case BinaryDivide():
                self._floor_divide(text, address)
            case Negate():
                text.pop(A0)
                text.negate(A0, A0)
                text.push(A0)
            case CompareLessEqual():
                # There is a set-if-less-than and no set-if-not-greater, so
                # the operands go the other way round and the answer flips.
                text.pop(A1)
                text.pop(A0)
                text.arithmetic("slt", A0, A1, A0)
                text.exclusive_or_immediate(A0, A0, 1)
                text.push(A0)
            case MakeInterval():
                pass
            case EmitOrientedRun(orientation=orientation):
                text.pop(A3)
                text.pop(A2)
                text.pop(A1)
                text.immediate(A0, self.ORIENTATION_ENCODING[orientation])
                text.call("emit_run")
            case CloseUnderGroup():
                text.call("snapshot")
                for element in self._group.elements:
                    linear = element.linear
                    text.immediate(A0, linear.a)
                    text.immediate(A1, linear.b)
                    text.immediate(A2, linear.c)
                    text.immediate(A3, linear.d)
                    text.call("apply")
            case _:
                raise MachineCodeError(f"unencodable instruction {instruction!r}")

    def _floor_divide(self, text: Riscv64Assembler, address: int) -> None:
        """div truncates towards zero where the interpreter floors."""
        floored = f"A{address}.floored"
        text.pop(A1)
        text.pop(A0)
        text.branch("eq", A1, RISCV_ZERO, "divide.by.zero")
        text.arithmetic("div", A2, A0, A1)
        text.arithmetic("rem", A3, A0, A1)
        text.branch("eq", A3, RISCV_ZERO, floored)
        text.arithmetic("xor", A3, A3, A1)
        text.branch("ge", A3, RISCV_ZERO, floored)
        text.arithmetic_immediate(A2, A2, -1)
        text.label(floored)
        text.push(A2)

    # -- the runtime the instruction stream calls into ---------------------

    def _paint(self, text: Riscv64Assembler) -> None:
        """paint(a0 = row, a1 = column), out of bounds silently ignored."""
        text.label("paint")
        for register in (A0, A1):
            text.branch("lt", register, RISCV_ZERO, "paint.done")
            text.branch_immediate("ge", register, self._order, "paint.done")
        text.immediate(T0, self._order)
        text.arithmetic("mul", T0, A0, T0)
        text.arithmetic("add", T0, T0, A1)
        self._address(text, T0, T0, self._layout.canvas)
        text.immediate(T1, 1)
        text.store_octet(T1, T0)
        text.label("paint.done")
        text.ret()

    def _emit_run(self, text: Riscv64Assembler) -> None:
        """emit_run(a0 = orientation, a1 = index, a2 = lower, a3 = upper)."""
        text.label("emit_run")
        text.move(RISCV_LINK_SAVE, RISCV_RETURN)
        text.move(S4, A0)
        text.move(S5, A1)
        text.move(S6, A2)
        text.move(S7, A3)
        text.label("emit_run.head")
        text.branch("gt", S6, S7, "emit_run.done")
        for encoding, name in ((1, "column"), (2, "diagonal"), (3, "antidiagonal")):
            text.branch_immediate("eq", S4, encoding, f"emit_run.{name}")
        text.move(A0, S5)
        text.move(A1, S6)
        text.jump("emit_run.paint")
        text.label("emit_run.column")
        text.move(A0, S6)
        text.move(A1, S5)
        text.jump("emit_run.paint")
        text.label("emit_run.diagonal")
        text.move(A0, S6)
        text.arithmetic("add", A1, S6, S5)
        text.jump("emit_run.paint")
        text.label("emit_run.antidiagonal")
        text.move(A0, S6)
        text.arithmetic("sub", A1, S5, S6)
        text.label("emit_run.paint")
        text.call("paint")
        text.arithmetic_immediate(S6, S6, 1)
        text.jump("emit_run.head")
        text.label("emit_run.done")
        text.move(RISCV_RETURN, RISCV_LINK_SAVE)
        text.ret()

    def _snapshot(self, text: Riscv64Assembler) -> None:
        """The canvas as it stood before the closure began."""
        text.label("snapshot")
        self._address(text, S4, RISCV_ZERO, self._layout.canvas)
        self._address(text, S5, RISCV_ZERO, self._layout.snapshot)
        text.immediate(T0, 0)
        text.label("snapshot.head")
        text.branch_immediate("ge", T0, self._cells, "snapshot.done")
        text.arithmetic("add", T1, S4, T0)
        text.load_octet(T2, T1)
        text.arithmetic("add", T1, S5, T0)
        text.store_octet(T2, T1)
        text.arithmetic_immediate(T0, T0, 1)
        text.jump("snapshot.head")
        text.label("snapshot.done")
        text.ret()

    def _apply(self, text: Riscv64Assembler) -> None:
        """apply(a0 = a, a1 = b, a2 = c, a3 = d): one group element."""
        text.label("apply")
        text.move(RISCV_LINK_SAVE, RISCV_RETURN)
        text.move(S4, A0)
        text.move(S5, A1)
        text.move(S6, A2)
        text.move(S7, A3)
        self._address(text, S8, RISCV_ZERO, self._layout.snapshot)
        text.immediate(S9, 0)
        text.label("apply.row")
        text.branch_immediate("ge", S9, self._order, "apply.done")
        text.immediate(S10, 0)
        text.label("apply.column")
        text.branch_immediate("ge", S10, self._order, "apply.row.step")
        text.immediate(T0, self._order)
        text.arithmetic("mul", T0, S9, T0)
        text.arithmetic("add", T0, T0, S10)
        text.arithmetic("add", T0, S8, T0)
        text.load_octet(T1, T0)
        text.branch("eq", T1, RISCV_ZERO, "apply.column.step")
        text.arithmetic_immediate(S11, S9, -self._apothem)
        text.arithmetic_immediate(T3, S10, -self._apothem)
        text.arithmetic("mul", T0, S4, S11)
        text.arithmetic("mul", T1, S5, T3)
        text.arithmetic("add", T0, T0, T1)
        text.arithmetic_immediate(A0, T0, self._apothem)
        text.arithmetic("mul", T0, S6, S11)
        text.arithmetic("mul", T1, S7, T3)
        text.arithmetic("add", T0, T0, T1)
        text.arithmetic_immediate(A1, T0, self._apothem)
        text.call("paint")
        text.label("apply.column.step")
        text.arithmetic_immediate(S10, S10, 1)
        text.jump("apply.column")
        text.label("apply.row.step")
        text.arithmetic_immediate(S9, S9, 1)
        text.jump("apply.row")
        text.label("apply.done")
        text.move(RISCV_RETURN, RISCV_LINK_SAVE)
        text.ret()

    def _render(self, text: Riscv64Assembler) -> None:
        """The canvas as text, trailing blanks trimmed, in a single write."""
        text.label("render")
        self._address(text, S4, RISCV_ZERO, self._layout.canvas)
        self._address(text, S5, RISCV_ZERO, self._layout.output)
        text.immediate(S6, 0)
        text.immediate(S7, 0)
        text.label("render.row")
        text.branch_immediate("ge", S6, self._order, "render.flush")
        text.immediate(S9, -1)
        text.immediate(S8, 0)
        text.label("render.scan")
        text.branch_immediate("ge", S8, self._order, "render.print")
        text.immediate(T0, self._order)
        text.arithmetic("mul", T0, S6, T0)
        text.arithmetic("add", T0, T0, S8)
        text.arithmetic("add", T0, S4, T0)
        text.load_octet(T1, T0)
        text.branch("eq", T1, RISCV_ZERO, "render.scan.step")
        text.move(S9, S8)
        text.label("render.scan.step")
        text.arithmetic_immediate(S8, S8, 1)
        text.jump("render.scan")
        text.label("render.print")
        text.immediate(S8, 0)
        text.label("render.cell")
        text.branch("gt", S8, S9, "render.newline")
        text.branch("le", S8, RISCV_ZERO, "render.ink")
        text.immediate(T2, ord(" "))
        text.arithmetic("add", T1, S5, S7)
        text.store_octet(T2, T1)
        text.arithmetic_immediate(S7, S7, 1)
        text.label("render.ink")
        text.immediate(T0, self._order)
        text.arithmetic("mul", T0, S6, T0)
        text.arithmetic("add", T0, T0, S8)
        text.arithmetic("add", T0, S4, T0)
        text.load_octet(T1, T0)
        text.branch("eq", T1, RISCV_ZERO, "render.blank")
        text.immediate(T2, ord("*"))
        text.jump("render.advance")
        text.label("render.blank")
        text.immediate(T2, ord(" "))
        text.label("render.advance")
        text.arithmetic("add", T1, S5, S7)
        text.store_octet(T2, T1)
        text.arithmetic_immediate(S7, S7, 1)
        text.arithmetic_immediate(S8, S8, 1)
        text.jump("render.cell")
        text.label("render.newline")
        text.immediate(T2, ord("\n"))
        text.arithmetic("add", T1, S5, S7)
        text.store_octet(T2, T1)
        text.arithmetic_immediate(S7, S7, 1)
        text.arithmetic_immediate(S6, S6, 1)
        text.jump("render.row")
        text.label("render.flush")
        text.immediate(RISCV_SYSCALL, RISCV_SYS_WRITE)
        text.immediate(A0, 1)
        self._address(text, A1, RISCV_ZERO, self._layout.output)
        text.move(A2, S7)
        text.syscall()
        text.ret()


ELF_HEADER_SIZE: Final[int] = 64
PROGRAM_HEADER_SIZE: Final[int] = 56
PROGRAM_HEADERS: Final[int] = 2


def elf64_image(text: bytes, bss: int, machine: int, align: int) -> bytes:
    """Wraps a text section in the smallest static ELF64 executable that runs it.

    Two loadable segments and nothing else: the headers and the text, mapped
    read-execute at the image base, and an anonymous read-write span the
    kernel zeroes for us, which is the whole of the program's data.

    Only two fields distinguish the machines: which one the header names, and
    how coarsely the segments are aligned.  Both bases are aligned to the
    larger of the two, so either value keeps the mapping congruent.
    """
    prologue = ELF_HEADER_SIZE + PROGRAM_HEADER_SIZE * PROGRAM_HEADERS
    loaded = prologue + len(text)
    header = struct.pack(
        "<4sBBBBB7xHHIQQQIHHHHHH",
        b"\x7fELF", 2, 1, 1, 0, 0,
        2, machine, 1, IMAGE_BASE + prologue, ELF_HEADER_SIZE, 0, 0,
        ELF_HEADER_SIZE, PROGRAM_HEADER_SIZE, PROGRAM_HEADERS, 64, 0, 0,
    )
    segments = struct.pack(
        "<IIQQQQQQ", 1, 5, 0, IMAGE_BASE, IMAGE_BASE, loaded, loaded, align
    ) + struct.pack(
        "<IIQQQQQQ", 1, 6, 0, DATA_BASE, DATA_BASE, 0, bss, align
    )
    return header + segments + text


MACHINES: Final[Mapping[str, tuple[type, int, int, str]]] = {
    "x86-64": (NativeCodeBackend, EM_X86_64, PAGE_SIZE, "x86_64"),
    "aarch64": (Aarch64CodeBackend, EM_AARCH64, ARM_PAGE, "aarch64"),
    "riscv64": (Riscv64CodeBackend, EM_RISCV, PAGE_SIZE, "riscv64"),
}


def host_machine() -> str:
    """The entry in :data:`MACHINES` this host is, or x86-64 if it is neither."""
    uname = os.uname().machine if hasattr(os, "uname") else ""
    for name, (_, _, _, machine) in MACHINES.items():
        if uname == machine:
            return name
    return "x86-64"


def machine_text(module: ObjectModule, architecture: str | None = None) -> bytes:
    """Just the text section, without the ELF wrapper around it."""
    backend_class, *_ = MACHINES[architecture or host_machine()]
    return backend_class(module).encode()


def machine_code(module: ObjectModule, architecture: str | None = None) -> bytes:
    """The object module as a static ELF64 executable for ``architecture``."""
    try:
        backend_class, machine, align, _ = MACHINES[architecture or host_machine()]
    except KeyError as exc:
        raise MachineCodeError(f"no backend for {architecture!r}") from exc
    backend = backend_class(module)
    return elf64_image(backend.encode(), backend.layout.size, machine, align)


def write_executable(
    module: ObjectModule, path: Path, architecture: str | None = None
) -> Path:
    """Writes that executable to ``path`` and makes it runnable."""
    path.write_bytes(machine_code(module, architecture))
    path.chmod(0o755)
    return path


def machine_code_runnable(architecture: str | None = None) -> bool:
    """Whether this host can execute what layer 18 emits for ``architecture``.

    A foreign machine counts when the kernel has been told how to run it,
    which is what binfmt_misc records, so a registered emulator makes the
    other backend testable here rather than only on the machine it targets.
    """
    if not sys.platform.startswith("linux"):
        return False
    wanted = architecture or host_machine()
    if wanted not in MACHINES:
        return False
    if wanted == host_machine():
        return True
    registration = Path(f"/proc/sys/fs/binfmt_misc/qemu-{MACHINES[wanted][3]}")
    try:
        return "enabled" in registration.read_text()
    except OSError:
        return False


def runnable_machines() -> tuple[str, ...]:
    """Every machine this host can both write for and then run."""
    return tuple(name for name in MACHINES if machine_code_runnable(name))



# ======================================================================
# Tier 7: no kernel either
# ======================================================================
#
# Layer 22 asks for nothing underneath at all.  The instruction stream and
# the whole runtime are the ones layer 18 already writes - the same encoder,
# the same operand stack, the same .bss - and only the three things a kernel
# was being asked for change: where the data lives and who zeroes it, what a
# finished program does, and where the octets go when there is no descriptor
# to write them to.
#
# What that costs is the sector in front: sixteen bits of real mode, which is
# where a machine begins, walking up to the sixty-four the encoder emits.

BOOT_BASE: Final[int] = 0x7C00
PAYLOAD_BASE: Final[int] = 0x8000
BOOT_DATA_BASE: Final[int] = 0x100000
VGA_TEXT_BASE: Final[int] = 0xB8000
VGA_ROW_OCTETS: Final[int] = 160
VGA_CELLS: Final[int] = 80 * 25
SERIAL_PORT: Final[int] = 0x3F8
SECTOR: Final[int] = 512
PAGE_TABLE_BASE: Final[int] = 0x1000

# null, then a sixty-four bit code segment, then data.
BOOT_GDT: Final[bytes] = struct.pack(
    "<QQQ", 0, 0x00209A0000000000, 0x0000920000000000
)


class BootCodeBackend(NativeCodeBackend):
    """The same text as layer 18, for a machine with nothing underneath it.

    Three answers differ.  Nobody has zeroed .bss, so the prologue does it.
    There is nowhere to exit to, so the epilogue halts.  And there is no
    descriptor to write to, so the octets go to the text buffer the firmware
    leaves mapped and out of the serial port besides, which is the one of the
    two that can be read back by a machine rather than a person.
    """

    DATA_ORIGIN: ClassVar[int] = BOOT_DATA_BASE

    def _prologue(self, text: X86Assembler) -> None:
        text.immediate(DATA_REGISTER, self.DATA_ORIGIN)
        text.arithmetic("xor", Register.RCX, Register.RCX)
        text.label("wipe.head")
        text.arithmetic_immediate("cmp", Register.RCX, self._layout.size)
        text.jump_if("ge", "wipe.done")
        text.store_octet_immediate(
            MemoryOperand(DATA_REGISTER, Register.RCX, 1, 0), 0
        )
        text.increment(Register.RCX)
        text.jump("wipe.head")
        text.label("wipe.done")

    def _epilogue(self, text: X86Assembler) -> None:
        text.label("stop")
        text.halt()
        text.jump("stop")

    def _flush(self, text: X86Assembler) -> None:
        text.call("blit")

    def _appendix(self, text: X86Assembler) -> None:
        """blit: the output buffer to the text cells, and out of the port."""
        text.label("blit")
        text.immediate(Register.R11, VGA_TEXT_BASE)
        text.arithmetic("xor", Register.RCX, Register.RCX)
        text.label("blit.clear")
        text.arithmetic_immediate("cmp", Register.RCX, VGA_CELLS)
        text.jump_if("ge", "blit.cleared")
        text.store_octet_immediate(
            MemoryOperand(Register.R11, Register.RCX, 2, 0), ord(" ")
        )
        text.store_octet_immediate(
            MemoryOperand(Register.R11, Register.RCX, 2, 1), 0x07
        )
        text.increment(Register.RCX)
        text.jump("blit.clear")
        text.label("blit.cleared")
        text.arithmetic("xor", Register.R8, Register.R8)
        text.arithmetic("xor", Register.R9, Register.R9)
        text.arithmetic("xor", Register.R10, Register.R10)
        text.immediate(Register.R11, VGA_TEXT_BASE)
        text.label("blit.head")
        text.arithmetic("cmp", Register.R8, Register.R14)
        text.jump_if("ge", "blit.done")
        text.arithmetic("xor", Register.RAX, Register.RAX)
        text.load_octet(
            Register.RAX,
            MemoryOperand(DATA_REGISTER, Register.R8, 1, self._layout.output),
        )
        text.immediate(Register.RDX, SERIAL_PORT)
        text.out()
        text.arithmetic_immediate("cmp", Register.RAX, ord("\n"))
        text.jump_if("e", "blit.newline")
        text.multiply_immediate(Register.RCX, Register.R9, VGA_ROW_OCTETS)
        text.arithmetic("add", Register.RCX, Register.R11)
        text.store_octet(MemoryOperand(Register.RCX, Register.R10, 2, 0), Register.RAX)
        text.store_octet_immediate(
            MemoryOperand(Register.RCX, Register.R10, 2, 1), 0x07
        )
        text.increment(Register.R10)
        text.jump("blit.step")
        text.label("blit.newline")
        text.increment(Register.R9)
        text.arithmetic("xor", Register.R10, Register.R10)
        text.label("blit.step")
        text.increment(Register.R8)
        text.jump("blit.head")
        text.label("blit.done")
        text.ret()


def boot_sector(payload_sectors: int) -> bytes:
    """Sixteen bits of real mode, walking up to sixty-four.

    Written out rather than encoded, because none of it recurs: segments and
    a stack, the payload off the disk the firmware booted from, the gate at
    port ninety-two, a page table that maps the first two megaoctets onto
    themselves, a descriptor table, and the three bits in three registers
    that put the machine in long mode.
    """
    code = bytearray()

    def emit(*octets: int) -> None:
        code.extend(octets)

    def assemble(pointer_at: int) -> None:
        code.clear()
        emit(0xFA)                                        # cli
        emit(0x31, 0xC0)                                  # xor ax, ax
        emit(0x8E, 0xD8), emit(0x8E, 0xC0), emit(0x8E, 0xD0)
        emit(0xBC, BOOT_BASE & 0xFF, BOOT_BASE >> 8)      # mov sp, 0x7c00
        emit(0xB8, payload_sectors & 0xFF, 0x02)          # mov ax, 0x02<sectors>
        emit(0xB9, 0x02, 0x00)                            # mov cx, cylinder 0 sector 2
        emit(0xB6, 0x00)                                  # mov dh, 0; dl is the drive
        emit(0xBB, PAYLOAD_BASE & 0xFF, PAYLOAD_BASE >> 8)
        emit(0xCD, 0x13)                                  # int 0x13
        emit(0xE4, 0x92), emit(0x0C, 0x02), emit(0xE6, 0x92)
        emit(0xBF, PAGE_TABLE_BASE & 0xFF, PAGE_TABLE_BASE >> 8)
        emit(0xB9, 0x00, 0x18)                            # 0x1800 words
        emit(0x31, 0xC0)
        emit(0xFC)                                        # cld
        emit(0xF3, 0xAB)                                  # rep stosw
        for where, entry in (
            (PAGE_TABLE_BASE, 0x2003),
            (PAGE_TABLE_BASE + 0x1000, 0x3003),
            (PAGE_TABLE_BASE + 0x2000, 0x0083),
        ):
            emit(0x66, 0xC7, 0x06, where & 0xFF, where >> 8, *struct.pack("<I", entry))
        emit(0x66, 0x0F, 0x01, 0x16, pointer_at & 0xFF, pointer_at >> 8)
        emit(0x0F, 0x20, 0xE0)
        emit(0x66, 0x0D, *struct.pack("<I", 1 << 5))      # cr4.pae
        emit(0x0F, 0x22, 0xE0)
        emit(0x66, 0xB9, *struct.pack("<I", 0xC0000080))
        emit(0x0F, 0x32)                                  # rdmsr
        emit(0x66, 0x0D, *struct.pack("<I", 1 << 8))      # efer.lme
        emit(0x0F, 0x30)                                  # wrmsr
        emit(0x66, 0xB8, *struct.pack("<I", PAGE_TABLE_BASE))
        emit(0x0F, 0x22, 0xD8)                            # cr3
        emit(0x0F, 0x20, 0xC0)
        emit(0x66, 0x0D, *struct.pack("<I", 0x80000001))  # cr0.pg | cr0.pe
        emit(0x0F, 0x22, 0xC0)
        target = len(code) + 8
        emit(0x66, 0xEA, *struct.pack("<I", BOOT_BASE + target), 0x08, 0x00)
        emit(0x66, 0xB8, 0x10, 0x00)                      # mov ax, 0x10
        emit(0x8E, 0xD8), emit(0x8E, 0xC0), emit(0x8E, 0xD0)
        emit(0xBC, *struct.pack("<I", BOOT_BASE - 0xC00))
        emit(0xB8, *struct.pack("<I", PAYLOAD_BASE))
        emit(0xFF, 0xE0)                                  # jmp rax

    assemble(0)
    pointer_at = BOOT_BASE + len(code)
    assemble(pointer_at)
    descriptor = struct.pack("<HI", len(BOOT_GDT) - 1, pointer_at + 6)
    sector = bytes(code) + descriptor + BOOT_GDT
    if len(sector) > SECTOR - 2:
        raise MachineCodeError(f"the first sector wants {len(sector)} octets")
    return sector.ljust(SECTOR - 2, b"\x00") + b"\x55\xaa"


def boot_image(module: ObjectModule) -> bytes:
    """A disk whose first sector is a machine walking up to its own width."""
    payload = BootCodeBackend(module).encode()
    sectors = -(-len(payload) // SECTOR)
    return boot_sector(sectors) + payload.ljust(sectors * SECTOR, b"\x00")


def boot_runnable() -> bool:
    """Whether anything here can be asked to start a machine."""
    return shutil.which("qemu-system-x86_64") is not None


def run_boot(image: Path, lines: int, patience: float = 20.0) -> str:
    """Starts a machine on ``image`` and answers what left the serial port.

    A program with nothing underneath it halts rather than exits, which makes
    waiting for the machine to finish pointless.  So the wait is for the
    octets instead - as many lines as the lattice has rows - and the machine
    is stopped from outside once they have arrived.
    """
    emulator = shutil.which("qemu-system-x86_64")
    if emulator is None:
        raise MachineCodeError("no system emulator is installed")
    with tempfile.TemporaryDirectory(prefix="ouroboros-boot-") as scratch:
        capture = Path(scratch) / "serial"
        machine = subprocess.Popen(
            [
                emulator, "-drive", f"format=raw,file={image}",
                "-display", "none", "-serial", f"file:{capture}", "-no-reboot",
            ],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        try:
            deadline = time.monotonic() + patience
            while time.monotonic() < deadline:
                if capture.exists():
                    written = capture.read_text(errors="replace")
                    if written.count("\n") >= lines:
                        return written
                time.sleep(0.05)
            raise MachineCodeError(f"no {lines} lines left the machine in time")
        finally:
            machine.kill()
            machine.wait()


# ======================================================================
# Tier 6: WebAssembly
# ======================================================================
#
# Layer 19 encodes the same object module as a wasm reactor: a module that
# imports nothing, exports its memory, the offset of its text and a ``render``
# that returns a length, and so needs no host beyond an engine.  Layer 7's
# machine is a stack machine and so is wasm, which is most of the reason this
# is shorter than layer 16: inside a basic block the lowering is one
# instruction to one, the operand stack never becomes memory, and the frame is
# locals rather than an address.
#
# Control flow is where that stops.  Wasm cannot branch to an arbitrary point,
# so the instruction stream becomes a br_table over a program counter, one
# block per branch target, with fallthrough between them costing nothing.


class WasmCodeError(GlyphPlatformError):
    """The object module could not be encoded as WebAssembly."""


I32: Final[int] = 0x7F
I64: Final[int] = 0x7E
VOID: Final[int] = 0x40
WASM_PAGE: Final[int] = 65536

WASM_OPCODES: Final[Mapping[str, int]] = {
    "i32.eqz": 0x45, "i32.lt_s": 0x48, "i32.gt_s": 0x4A, "i32.ge_s": 0x4E,
    "i32.add": 0x6A, "i32.sub": 0x6B, "i32.mul": 0x6C,
    "i64.eqz": 0x50, "i64.lt_s": 0x53, "i64.gt_s": 0x55, "i64.le_s": 0x57,
    "i64.ge_s": 0x59,
    "i64.add": 0x7C, "i64.sub": 0x7D, "i64.mul": 0x7E, "i64.div_s": 0x7F,
    "i64.rem_s": 0x81, "i64.xor": 0x85,
    "i32.wrap_i64": 0xA7, "i64.extend_i32_s": 0xAC,
}


def _uleb(value: int) -> bytes:
    if value < 0:
        raise WasmCodeError(f"unsigned LEB128 cannot hold {value}")
    out = bytearray()
    while True:
        octet, value = value & 0x7F, value >> 7
        if not value:
            out.append(octet)
            return bytes(out)
        out.append(octet | 0x80)


def _sleb(value: int) -> bytes:
    out = bytearray()
    while True:
        octet, value = value & 0x7F, value >> 7
        if (value == 0 and not octet & 0x40) or (value == -1 and octet & 0x40):
            out.append(octet)
            return bytes(out)
        out.append(octet | 0x80)


def _wasm_vector(items: Sequence[bytes]) -> bytes:
    return _uleb(len(items)) + b"".join(items)


def _wasm_section(identifier: int, payload: bytes) -> bytes:
    return bytes([identifier]) + _uleb(len(payload)) + payload


def _wasm_name(text: str) -> bytes:
    encoded = text.encode()
    return _uleb(len(encoded)) + encoded


def _functype(params: Sequence[int], results: Sequence[int]) -> bytes:
    return (
        b"\x60"
        + _wasm_vector([bytes([kind]) for kind in params])
        + _wasm_vector([bytes([kind]) for kind in results])
    )


class WasmAssembler:
    """One function body, with branch depths resolved from label names.

    Not to be confused with layer 9's :class:`Assembler`, which resolves the
    virtual machine's labels, nor with layer 18's :class:`X86Assembler`, which
    resolves branch displacements.  Wasm names a branch target by how many
    enclosing structures to leave, so this one keeps the stack of what is open
    and counts.  There is no fixup pass and there cannot be one: a label is
    always already open by the time anything branches to it.
    """

    def __init__(self) -> None:
        self._code = bytearray()
        self._open: list[str] = []

    def __len__(self) -> int:
        return len(self._code)

    # -- structure --------------------------------------------------------

    def block(self, name: str, result: int = VOID) -> "WasmAssembler":
        self._code.extend((0x02, result))
        self._open.append(name)
        return self

    def loop(self, name: str, result: int = VOID) -> "WasmAssembler":
        self._code.extend((0x03, result))
        self._open.append(name)
        return self

    def if_(self, result: int = VOID) -> "WasmAssembler":
        self._code.extend((0x04, result))
        self._open.append("if")
        return self

    def else_(self) -> "WasmAssembler":
        self._code.append(0x05)
        return self

    def end(self) -> "WasmAssembler":
        if not self._open:
            raise WasmCodeError("end with no open structure")
        self._open.pop()
        self._code.append(0x0B)
        return self

    def depth(self, name: str) -> int:
        for position in range(len(self._open) - 1, -1, -1):
            if self._open[position] == name:
                return len(self._open) - 1 - position
        raise WasmCodeError(f"no enclosing structure named {name!r}")

    def br(self, name: str) -> "WasmAssembler":
        self._code.append(0x0C)
        self._code.extend(_uleb(self.depth(name)))
        return self

    def br_if(self, name: str) -> "WasmAssembler":
        self._code.append(0x0D)
        self._code.extend(_uleb(self.depth(name)))
        return self

    def br_table(self, names: Sequence[str], default: str) -> "WasmAssembler":
        self._code.append(0x0E)
        self._code.extend(_wasm_vector([_uleb(self.depth(name)) for name in names]))
        self._code.extend(_uleb(self.depth(default)))
        return self

    def unreachable(self) -> "WasmAssembler":
        self._code.append(0x00)
        return self

    def call(self, index: int) -> "WasmAssembler":
        self._code.append(0x10)
        self._code.extend(_uleb(index))
        return self

    # -- values -----------------------------------------------------------

    def i32(self, value: int) -> "WasmAssembler":
        self._code.append(0x41)
        self._code.extend(_sleb(value))
        return self

    def i64(self, value: int) -> "WasmAssembler":
        self._code.append(0x42)
        self._code.extend(_sleb(value))
        return self

    def get(self, local: int) -> "WasmAssembler":
        self._code.append(0x20)
        self._code.extend(_uleb(local))
        return self

    def set(self, local: int) -> "WasmAssembler":
        self._code.append(0x21)
        self._code.extend(_uleb(local))
        return self

    def load8(self, offset: int) -> "WasmAssembler":
        self._code.extend((0x2D, 0x00))
        self._code.extend(_uleb(offset))
        return self

    def store8(self, offset: int) -> "WasmAssembler":
        self._code.extend((0x3A, 0x00))
        self._code.extend(_uleb(offset))
        return self

    def op(self, *names: str) -> "WasmAssembler":
        for name in names:
            try:
                self._code.append(WASM_OPCODES[name])
            except KeyError as exc:
                raise WasmCodeError(f"no opcode named {name!r}") from exc
        return self

    # -- completion -------------------------------------------------------

    def body(self, locals_: Sequence[tuple[int, int]] = ()) -> bytes:
        if self._open:
            raise WasmCodeError(f"unclosed structure {self._open[-1]!r}")
        code = (
            _wasm_vector([_uleb(count) + bytes([kind]) for count, kind in locals_ if count])
            + bytes(self._code)
            + b"\x0b"
        )
        return _uleb(len(code)) + code


@dataclass(frozen=True, slots=True)
class WasmLayout:
    """Where the canvas, the snapshot and the output buffer sit in memory.

    Deliberately not layer 18's :class:`DataLayout`: that one reserves a frame
    region because the machine has nowhere else to keep the locals, and here
    they are wasm locals, so the region would only ever be padding.
    """

    canvas: int
    snapshot: int
    output: int
    size: int

    @classmethod
    def of(cls, order: int) -> "WasmLayout":
        cells = order * order
        return cls(0, cells, 2 * cells, 4 * cells + order + 16)

    @property
    def pages(self) -> int:
        return max(1, -(-self.size // WASM_PAGE))


class WasmCodeBackend:
    """Encodes a linked object module as a WebAssembly reactor module.

    The module imports nothing, so the whole host it asks for is an engine.
    It exports its memory, the offset of the text as a global, and a
    ``render`` that runs the program and answers how many octets it wrote.

    A branch truncates the value stack to the height of the structure it lands
    in, so keeping the operand stack on the value stack is only sound while
    that stack is empty wherever the module branches.  It is, for everything
    layer 7 emits: statements lower to stack-neutral runs and an iteration
    pops its condition before it jumps.  That is a property of the emitter and
    not of the instruction set, so :meth:`_audit` derives it from the
    instructions' own stack deltas rather than trusting it.
    """

    ORIENTATION_ENCODING: ClassVar[Mapping[Orientation, int]] = {
        Orientation.ROW: 0,
        Orientation.COLUMN: 1,
        Orientation.DIAGONAL: 2,
        Orientation.ANTIDIAGONAL: 3,
    }

    # What each instruction does to the operand stack once an interval is two
    # words rather than one, which is not what ``Instruction.stack_delta`` says.
    PHYSICAL_DELTA: ClassVar[Mapping[type, int]] = {
        PushConstant: +1, LoadIntrinsic: +1, LoadLocal: +1, StoreLocal: -1,
        BinaryAdd: -1, BinarySubtract: -1, BinaryMultiply: -1, BinaryDivide: -1,
        Negate: 0, CompareLessEqual: -1, MakeInterval: 0, EmitOrientedRun: -3,
        Jump: 0, JumpIfFalse: -1, CloseUnderGroup: 0, Halt: 0,
    }

    # The i64 locals the instruction stream borrows when it has to hold an
    # operand somewhere other than the top of the stack.
    SCRATCH: ClassVar[Mapping[str, int]] = {
        "dividend": 0, "divisor": 1, "quotient": 2, "remainder": 3,
        "index": 4, "lower": 5, "upper": 6,
    }

    PAINT, EMIT_RUN, SNAPSHOT, APPLY, DRAW, RENDER = range(6)

    def __init__(self, module: ObjectModule) -> None:
        self._module = module
        self._order = module.order
        self._cells = module.order * module.order
        self._apothem = module.order // 2
        self._frame = max(module.frame_size, 1)
        self._layout = WasmLayout.of(module.order)
        self._group = SymmetryGroup.generated_by(
            module.family.generators,
            Coordinate(self._apothem, self._apothem),
            module.symmetry_order,
            presentation=f"{module.family.value}-{module.cardinality} about centroid",
        )
        targets = {0}
        for instruction in module.instructions:
            if isinstance(instruction, (Jump, JumpIfFalse)):
                targets.add(int(instruction.target))
        self._targets = tuple(sorted(targets))
        self._ordinal = {address: index for index, address in enumerate(self._targets)}

    @property
    def layout(self) -> WasmLayout:
        return self._layout

    def _audit(self) -> None:
        """Rejects a module whose operand stack is not empty where it branches."""
        depth = 0
        for address, instruction in enumerate(self._module.instructions):
            if address in self._ordinal and depth:
                raise WasmCodeError(
                    f"branch target {address} is reached with {depth} operand(s) live"
                )
            try:
                depth += self.PHYSICAL_DELTA[type(instruction)]
            except KeyError as exc:
                raise WasmCodeError(f"unencodable instruction {instruction!r}") from exc
            if isinstance(instruction, (Jump, JumpIfFalse, Halt)) and depth:
                raise WasmCodeError(
                    f"branch at {address} leaves {depth} operand(s) on the stack"
                )

    def _intrinsic(self, name: str) -> int:
        try:
            return {
                "zero": 0,
                "apothem": self._apothem,
                "extremum": self._order - 1,
                "magnitude": self._order,
            }[name]
        except KeyError as exc:
            raise WasmCodeError(f"unbound intrinsic {name!r}") from exc

    # -- the module -------------------------------------------------------

    @woven
    def encode(self) -> bytes:
        """The whole module: six functions, one memory, three exports."""
        with TRACER.span("wasm-encode", order=self._order):
            self._audit()
            types = [
                _functype((I64, I64), ()),
                _functype((I32, I64, I64, I64), ()),
                _functype((), ()),
                _functype((I64, I64, I64, I64), ()),
                _functype((), (I32,)),
            ]
            METRICS.increment("wasm.instructions", len(self._module.instructions))
            return (
                b"\0asm"
                + struct.pack("<I", 1)
                + _wasm_section(1, _wasm_vector(types))
                + _wasm_section(3, _wasm_vector([_uleb(n) for n in (0, 1, 2, 3, 4, 4)]))
                + _wasm_section(5, _wasm_vector([b"\x00" + _uleb(self._layout.pages)]))
                + _wasm_section(6, _wasm_vector(
                    [bytes([I32, 0x00]) + b"\x41" + _sleb(self._layout.output) + b"\x0b"]
                ))
                + _wasm_section(7, _wasm_vector([
                    _wasm_name("memory") + b"\x02\x00",
                    _wasm_name("render") + b"\x00" + _uleb(self.RENDER),
                    _wasm_name("output") + b"\x03\x00",
                ]))
                + _wasm_section(10, _wasm_vector([
                    self._paint(), self._emit_run(), self._snapshot(),
                    self._apply(), self._draw(), self._render(),
                ]))
            )

    # -- the runtime the instruction stream calls into ---------------------

    def _paint(self) -> bytes:
        """paint(row, column), out of bounds silently ignored.

        The bounds are compared in i64 before the address is narrowed, so a
        coordinate far outside the lattice cannot wrap back into it.
        """
        row, column = 0, 1
        text = WasmAssembler().block("done")
        for local in (row, column):
            text.get(local).i64(0).op("i64.lt_s").br_if("done")
            text.get(local).i64(self._order).op("i64.ge_s").br_if("done")
        text.get(row).op("i32.wrap_i64").i32(self._order).op("i32.mul")
        text.get(column).op("i32.wrap_i64").op("i32.add")
        text.i32(1).store8(self._layout.canvas)
        return text.end().body()

    def _emit_run(self) -> bytes:
        """emit_run(orientation, index, lower, upper): one run of cells."""
        orientation, index, lower, upper, row, column = range(6)
        text = WasmAssembler().block("done").loop("next")
        text.get(lower).get(upper).op("i64.gt_s").br_if("done")
        text.get(index).set(row)
        text.get(lower).set(column)
        text.block("chosen").block("anti").block("diag").block("col")
        text.get(orientation).br_table(("chosen", "col", "diag", "anti"), "chosen")
        text.end()
        text.get(lower).set(row).get(index).set(column).br("chosen")
        text.end()
        text.get(lower).set(row)
        text.get(lower).get(index).op("i64.add").set(column).br("chosen")
        text.end()
        text.get(lower).set(row)
        text.get(index).get(lower).op("i64.sub").set(column)
        text.end()
        text.get(row).get(column).call(self.PAINT)
        text.get(lower).i64(1).op("i64.add").set(lower).br("next")
        return text.end().end().body(((2, I64),))

    def _snapshot(self) -> bytes:
        """The canvas as it stood before the closure began."""
        cursor = 0
        text = WasmAssembler().block("done").loop("next")
        text.get(cursor).i32(self._cells).op("i32.ge_s").br_if("done")
        text.get(cursor).get(cursor).load8(self._layout.canvas)
        text.store8(self._layout.snapshot)
        text.get(cursor).i32(1).op("i32.add").set(cursor).br("next")
        return text.end().end().body(((1, I32),))

    def _apply(self) -> bytes:
        """apply(a, b, c, d): one group element, read off the snapshot."""
        a, b, c, d, row, column, dr, dc = range(8)
        text = WasmAssembler().block("done").loop("rows")
        text.get(row).i32(self._order).op("i32.ge_s").br_if("done")
        text.i32(0).set(column)
        text.block("row.done").loop("columns")
        text.get(column).i32(self._order).op("i32.ge_s").br_if("row.done")
        text.get(row).i32(self._order).op("i32.mul").get(column).op("i32.add")
        text.load8(self._layout.snapshot)
        text.if_()
        text.get(row).op("i64.extend_i32_s").i64(self._apothem).op("i64.sub").set(dr)
        text.get(column).op("i64.extend_i32_s").i64(self._apothem).op("i64.sub").set(dc)
        text.get(a).get(dr).op("i64.mul").get(b).get(dc).op("i64.mul").op("i64.add")
        text.i64(self._apothem).op("i64.add")
        text.get(c).get(dr).op("i64.mul").get(d).get(dc).op("i64.mul").op("i64.add")
        text.i64(self._apothem).op("i64.add")
        text.call(self.PAINT)
        text.end()
        text.get(column).i32(1).op("i32.add").set(column).br("columns")
        text.end().end()
        text.get(row).i32(1).op("i32.add").set(row).br("rows")
        return text.end().end().body(((2, I32), (2, I64)))

    def _draw(self) -> bytes:
        """The canvas as text, trailing blanks trimmed; answers the length."""
        row, column, last, cursor = range(4)
        canvas, output = self._layout.canvas, self._layout.output
        text = WasmAssembler().block("flush").loop("rows")
        text.get(row).i32(self._order).op("i32.ge_s").br_if("flush")

        text.i32(-1).set(last).i32(0).set(column)
        text.block("scan.done").loop("scan")
        text.get(column).i32(self._order).op("i32.ge_s").br_if("scan.done")
        text.get(row).i32(self._order).op("i32.mul").get(column).op("i32.add")
        text.load8(canvas)
        text.if_().get(column).set(last).end()
        text.get(column).i32(1).op("i32.add").set(column).br("scan")
        text.end().end()

        text.i32(0).set(column)
        text.block("row.done").loop("cells")
        text.get(column).get(last).op("i32.gt_s").br_if("row.done")
        text.get(column).i32(0).op("i32.gt_s")
        text.if_()
        text.get(cursor).i32(ord(" ")).store8(output)
        text.get(cursor).i32(1).op("i32.add").set(cursor)
        text.end()
        text.get(cursor)
        text.get(row).i32(self._order).op("i32.mul").get(column).op("i32.add")
        text.load8(canvas)
        text.if_(I32).i32(ord("*")).else_().i32(ord(" ")).end()
        text.store8(output)
        text.get(cursor).i32(1).op("i32.add").set(cursor)
        text.get(column).i32(1).op("i32.add").set(column).br("cells")
        text.end().end()

        text.get(cursor).i32(ord("\n")).store8(output)
        text.get(cursor).i32(1).op("i32.add").set(cursor)
        text.get(row).i32(1).op("i32.add").set(row).br("rows")
        text.end().end()
        return text.get(cursor).body(((4, I32),))

    # -- the instruction stream --------------------------------------------
    #
    # One block per branch target, opened outermost-last, so that leaving
    # block A lands on A's code and running off the end of it falls into the
    # next target's code with no branch at all.

    def _render(self) -> bytes:
        """The program itself, and then the canvas as text."""
        frame, scratch = 1, 1 + self._frame
        text = WasmAssembler().block("exit").loop("top")
        for address in reversed(self._targets):
            text.block(f"A{address}")
        text.block("bad")
        text.get(0).br_table(tuple(f"A{a}" for a in self._targets), "bad")
        text.end().unreachable()

        boundaries = (*self._targets, len(self._module.instructions))
        for index, start in enumerate(self._targets):
            text.end()
            for address in range(start, boundaries[index + 1]):
                self._instruction(
                    text, self._module.instructions[address], frame, scratch
                )
        text.end().end().call(self.DRAW)
        return text.body(((1, I32), (self._frame + len(self.SCRATCH), I64)))

    def _instruction(
        self, text: WasmAssembler, instruction: Instruction, frame: int, scratch: int
    ) -> None:
        match instruction:
            case Halt():
                text.br("exit")
            case Jump(target=target):
                text.i32(self._ordinal[int(target)]).set(0).br("top")
            case JumpIfFalse(target=target):
                text.op("i64.eqz").if_()
                text.i32(self._ordinal[int(target)]).set(0).br("top")
                text.end()
            case PushConstant(value=value):
                text.i64(value)
            case LoadIntrinsic(name=name):
                text.i64(self._intrinsic(name))
            case LoadLocal(slot=slot):
                text.get(frame + slot)
            case StoreLocal(slot=slot):
                text.set(frame + slot)
            case BinaryAdd():
                text.op("i64.add")
            case BinarySubtract():
                text.op("i64.sub")
            case BinaryMultiply():
                text.op("i64.mul")
            case BinaryDivide():
                self._floor_divide(text, scratch)
            case Negate():
                text.i64(-1).op("i64.mul")
            case CompareLessEqual():
                text.op("i64.le_s", "i64.extend_i32_s")
            case MakeInterval():
                pass
            case EmitOrientedRun(orientation=orientation):
                for name in ("upper", "lower", "index"):
                    text.set(scratch + self.SCRATCH[name])
                text.i32(self.ORIENTATION_ENCODING[orientation])
                for name in ("index", "lower", "upper"):
                    text.get(scratch + self.SCRATCH[name])
                text.call(self.EMIT_RUN)
            case CloseUnderGroup():
                text.call(self.SNAPSHOT)
                for element in self._group.elements:
                    linear = element.linear
                    text.i64(linear.a).i64(linear.b).i64(linear.c).i64(linear.d)
                    text.call(self.APPLY)
            case _:
                raise WasmCodeError(f"unencodable instruction {instruction!r}")

    def _floor_divide(self, text: WasmAssembler, scratch: int) -> None:
        """i64.div_s truncates towards zero where the interpreter floors.

        The quotient is decremented whenever the division was inexact and the
        remainder and the divisor disagree about sign, which xor detects in
        the sign bit.
        """
        dividend = scratch + self.SCRATCH["dividend"]
        divisor = scratch + self.SCRATCH["divisor"]
        quotient = scratch + self.SCRATCH["quotient"]
        remainder = scratch + self.SCRATCH["remainder"]
        text.set(divisor).set(dividend)
        text.get(dividend).get(divisor).op("i64.div_s").set(quotient)
        text.get(dividend).get(divisor).op("i64.rem_s").set(remainder)
        text.get(remainder).op("i64.eqz", "i32.eqz").if_()
        text.get(remainder).get(divisor).op("i64.xor").i64(0).op("i64.lt_s").if_()
        text.get(quotient).i64(1).op("i64.sub").set(quotient)
        text.end().end()
        text.get(quotient)


WASM_HOST_SCRIPT: Final[str] = """
const fs = require('fs');
const module_ = process.argv[process.argv.length - 1];
WebAssembly.instantiate(fs.readFileSync(module_)).then(({instance}) => {
  const length = instance.exports.render();
  const offset = instance.exports.output.value;
  process.stdout.write(
    Buffer.from(instance.exports.memory.buffer, offset, length)
  );
});
"""


def wasm_module(module: ObjectModule) -> bytes:
    """The object module as a self-contained WebAssembly reactor."""
    return WasmCodeBackend(module).encode()


def write_wasm(module: ObjectModule, path: Path) -> Path:
    """Writes that module to ``path``."""
    path.write_bytes(wasm_module(module))
    return path


def wasm_host() -> str | None:
    """A command that can instantiate what layer 19 emits, if one is installed.

    Only the plain ``WebAssembly`` object is used, which every release of node
    since v8 has had without a flag, because the module imports nothing: a
    WASI runtime would be a heavier dependency for no more capability.
    """
    return shutil.which("node")


def run_wasm(path: Path) -> str:
    """Instantiates the module and returns the text it wrote."""
    host = wasm_host()
    if host is None:
        raise WasmCodeError("no WebAssembly host is installed")
    completed = subprocess.run(
        [host, "-e", WASM_HOST_SCRIPT, str(path)], stdout=subprocess.PIPE, check=True
    )
    return completed.stdout.decode()


# ======================================================================
# Layer 20: the module read back
# ======================================================================
#
# The emitter needed none of LEB128 decoding, structured control flow or the
# validation rules; reading its own output back needs all three.  What comes
# of it is a fourth way to execute the object module with nothing installed,
# and the only one that never leaves this file.


MASK32: Final[int] = 0xFFFFFFFF
MASK64: Final[int] = 0xFFFFFFFFFFFFFFFF


class WasmDecodeError(GlyphPlatformError):
    """The module could not be read."""


class WasmTrap(GlyphPlatformError):
    """The module trapped while running."""


def _signed(value: int, bits: int) -> int:
    return value - (1 << bits) if value & (1 << (bits - 1)) else value


# ----------------------------------------------------------------------
# decoding
# ----------------------------------------------------------------------


class WasmReader:
    """A cursor over the binary format's little-endian, LEB128-heavy scalars."""

    __slots__ = ("_data", "_at")

    def __init__(self, data: bytes, at: int = 0) -> None:
        self._data = data
        self._at = at

    @property
    def at(self) -> int:
        return self._at

    def done(self, limit: int) -> bool:
        return self._at >= limit

    def octet(self) -> int:
        try:
            value = self._data[self._at]
        except IndexError as exc:
            raise WasmDecodeError("the module ends inside an instruction") from exc
        self._at += 1
        return value

    def take(self, count: int) -> bytes:
        chunk = self._data[self._at : self._at + count]
        if len(chunk) != count:
            raise WasmDecodeError(f"the module ends {count - len(chunk)} octets early")
        self._at += count
        return chunk

    def uleb(self) -> int:
        value, shift = 0, 0
        while True:
            octet = self.octet()
            value |= (octet & 0x7F) << shift
            if not octet & 0x80:
                return value
            shift += 7
            if shift > 63:
                raise WasmDecodeError("unsigned LEB128 runs past 64 bits")

    def sleb(self) -> int:
        value, shift = 0, 0
        while True:
            octet = self.octet()
            value |= (octet & 0x7F) << shift
            shift += 7
            if not octet & 0x80:
                if octet & 0x40 and shift < 128:
                    value -= 1 << shift
                return value
            if shift > 127:
                raise WasmDecodeError("signed LEB128 runs past its range")

    def vector(self, element: Callable[[], Any]) -> list[Any]:
        return [element() for _ in range(self.uleb())]


@dataclass(frozen=True, slots=True)
class WasmType:
    params: tuple[int, ...]
    results: tuple[int, ...]


@dataclass(slots=True)
class WasmFunction:
    type_index: int
    signature: WasmType
    locals: tuple[int, ...]
    code: list[tuple] = field(default_factory=list)


@dataclass(slots=True)
class DecodedModule:
    types: list[WasmType]
    functions: list[WasmFunction]
    pages: int
    globals: list[int]
    exports: dict[str, tuple[int, int]]


# Every opcode layer 19 emits, and its immediate shape.  Anything absent is
# rejected rather than guessed at.
_NO_IMMEDIATE: Final[frozenset[int]] = frozenset(
    {
        0x00, 0x05, 0x0B, 0x0F, 0x1A,
        0x45, 0x46, 0x47, 0x48, 0x4A, 0x4C, 0x4E,
        0x50, 0x51, 0x53, 0x55, 0x57, 0x59,
        0x6A, 0x6B, 0x6C,
        0x7C, 0x7D, 0x7E, 0x7F, 0x81, 0x85,
        0xA7, 0xAC,
    }
)
_BLOCK_TYPE: Final[frozenset[int]] = frozenset({0x02, 0x03, 0x04})
_ONE_INDEX: Final[frozenset[int]] = frozenset({0x0C, 0x0D, 0x10, 0x20, 0x21, 0x22, 0x23, 0x24})
_MEMARG: Final[frozenset[int]] = frozenset({0x28, 0x2D, 0x36, 0x3A})


def _decode_body(reader: WasmReader, limit: int) -> list[tuple]:
    """Flattens a function body into (opcode, immediate) pairs."""
    code: list[tuple] = []
    while not reader.done(limit):
        opcode = reader.octet()
        if opcode in _NO_IMMEDIATE:
            code.append((opcode, None))
        elif opcode in _BLOCK_TYPE:
            code.append((opcode, reader.octet()))
        elif opcode in _ONE_INDEX:
            code.append((opcode, reader.uleb()))
        elif opcode in _MEMARG:
            reader.uleb()
            code.append((opcode, reader.uleb()))
        elif opcode == 0x41:
            code.append((opcode, reader.sleb() & MASK32))
        elif opcode == 0x42:
            code.append((opcode, reader.sleb() & MASK64))
        elif opcode == 0x0E:
            targets = tuple(reader.vector(reader.uleb))
            code.append((opcode, (targets, reader.uleb())))
        else:
            raise WasmDecodeError(f"opcode 0x{opcode:02x} is not one this reads")
    return code


def _resolve(code: list[tuple]) -> list[tuple]:
    """Pairs every structured instruction with its else and its end.

    Wasm names a branch target by how many structures to leave rather than by
    where to land, so the landing places have to be recovered once here or
    found again on every branch.
    """
    resolved: list[Any] = [immediate for _, immediate in code]
    open_: list[int] = []
    for position, (opcode, immediate) in enumerate(code):
        if opcode in _BLOCK_TYPE:
            resolved[position] = [immediate, None, None]
            open_.append(position)
        elif opcode == 0x05:
            if not open_ or code[open_[-1]][0] != 0x04:
                raise WasmDecodeError("else outside an if")
            resolved[open_[-1]][1] = position
        elif opcode == 0x0B and open_:
            resolved[open_.pop()][2] = position
    if open_:
        raise WasmDecodeError("a structured instruction is never closed")
    return [
        (opcode, tuple(immediate) if isinstance(immediate, list) else immediate)
        for (opcode, _), immediate in zip(code, resolved)
    ]


def decode_wasm(blob: bytes) -> DecodedModule:
    """Reads a module far enough to run it."""
    reader = WasmReader(blob)
    if reader.take(4) != b"\0asm":
        raise WasmDecodeError("not a WebAssembly module")
    if reader.take(4) != b"\x01\0\0\0":
        raise WasmDecodeError("unsupported module version")

    types: list[WasmType] = []
    type_indices: list[int] = []
    bodies: list[tuple[tuple[int, ...], list[tuple]]] = []
    pages = 1
    globals_: list[int] = []
    exports: dict[str, tuple[int, int]] = {}

    while not reader.done(len(blob)):
        identifier = reader.octet()
        size = reader.uleb()
        end = reader.at + size
        if identifier == 1:
            def one_type() -> WasmType:
                if reader.octet() != 0x60:
                    raise WasmDecodeError("malformed function type")
                params = tuple(reader.vector(reader.octet))
                return WasmType(params, tuple(reader.vector(reader.octet)))

            types = reader.vector(one_type)
        elif identifier == 3:
            type_indices = reader.vector(reader.uleb)
        elif identifier == 5:
            def one_memory() -> int:
                flags = reader.octet()
                minimum = reader.uleb()
                if flags:
                    reader.uleb()
                return minimum

            memories = reader.vector(one_memory)
            pages = memories[0] if memories else 1
        elif identifier == 6:
            def one_global() -> int:
                reader.octet()
                reader.octet()
                if reader.octet() != 0x41:
                    raise WasmDecodeError("only i32.const initialisers are read")
                value = reader.sleb() & MASK32
                if reader.octet() != 0x0B:
                    raise WasmDecodeError("global initialiser does not end")
                return value

            globals_ = reader.vector(one_global)
        elif identifier == 7:
            def one_export() -> None:
                name = reader.take(reader.uleb()).decode()
                exports[name] = (reader.octet(), reader.uleb())

            reader.vector(one_export)
        elif identifier == 10:
            def one_body() -> tuple[tuple[int, ...], list[tuple]]:
                stop = reader.uleb() + reader.at
                declared: list[int] = []
                for _ in range(reader.uleb()):
                    count = reader.uleb()
                    kind = reader.octet()
                    declared.extend([kind] * count)
                return tuple(declared), _resolve(_decode_body(reader, stop))

            bodies = reader.vector(one_body)
        else:
            reader.take(size)
        if reader.at != end:
            raise WasmDecodeError(f"section {identifier} does not end where it says")

    if len(type_indices) != len(bodies):
        raise WasmDecodeError("the function and code sections disagree")
    functions = [
        WasmFunction(index, types[index], declared, code)
        for index, (declared, code) in zip(type_indices, bodies)
    ]
    return DecodedModule(types, functions, pages, globals_, exports)


# ----------------------------------------------------------------------
# execution
# ----------------------------------------------------------------------


@dataclass(slots=True)
class _Frame:
    """One open structure: where a branch to it lands, and how much it keeps."""

    arity: int
    target: int
    end: int
    height: int


class WasmMachine:
    """Runs the subset of the instruction set layer 19 emits, and no more."""

    def __init__(self, module: DecodedModule) -> None:
        self._module = module
        self._memory = bytearray(module.pages * WASM_PAGE)

    @property
    def memory(self) -> bytearray:
        return self._memory

    def invoke(self, index: int, arguments: Sequence[int] = ()) -> list[int]:
        function = self._module.functions[index]
        if len(arguments) != len(function.signature.params):
            raise WasmTrap(f"function {index} wants {len(function.signature.params)} argument(s)")
        return self._run(function, [*arguments, *([0] * len(function.locals))])

    def _run(self, function: WasmFunction, slots: list[int]) -> list[int]:
        code = function.code
        length = len(code)
        results = len(function.signature.results)
        memory = self._memory
        stack: list[int] = []
        frames: list[_Frame] = []
        pc = 0

        while pc < length:
            opcode, immediate = code[pc]

            if opcode == 0x20:                                    # local.get
                stack.append(slots[immediate])
            elif opcode == 0x21:                                  # local.set
                slots[immediate] = stack.pop()
            elif opcode == 0x41 or opcode == 0x42:                # i32/i64.const
                stack.append(immediate)
            elif opcode == 0x6A:                                  # i32.add
                right = stack.pop()
                stack[-1] = (stack[-1] + right) & MASK32
            elif opcode == 0x6C:                                  # i32.mul
                right = stack.pop()
                stack[-1] = (stack[-1] * right) & MASK32
            elif opcode == 0x2D:                                  # i32.load8_u
                stack[-1] = memory[stack[-1] + immediate]
            elif opcode == 0x3A:                                  # i32.store8
                value = stack.pop()
                memory[stack.pop() + immediate] = value & 0xFF
            elif opcode == 0x0D:                                  # br_if
                if stack.pop():
                    pc = self._branch(frames, stack, immediate)
                    if pc < 0:
                        return stack[len(stack) - results:] if results else []
                    continue
            elif opcode == 0x0B:                                  # end
                if frames:
                    frames.pop()
            elif opcode == 0x03:                                  # loop
                # A branch to a loop lands on the loop itself, which opens it
                # again; a branch leaves the structure it names either way.
                block_type, _, end = immediate
                frames.append(_Frame(0, pc, end, len(stack)))
            elif opcode == 0x02:                                  # block
                block_type, _, end = immediate
                arity = 0 if block_type == 0x40 else 1
                frames.append(_Frame(arity, end + 1, end, len(stack)))
            elif opcode == 0x0C:                                  # br
                pc = self._branch(frames, stack, immediate)
                if pc < 0:
                    return stack[len(stack) - results:] if results else []
                continue
            elif opcode == 0x04:                                  # if
                block_type, else_at, end = immediate
                condition = stack.pop()
                arity = 0 if block_type == 0x40 else 1
                frames.append(_Frame(arity, end + 1, end, len(stack)))
                if not condition:
                    pc = end if else_at is None else else_at + 1
                    continue
            elif opcode == 0x05:                                  # else
                pc = frames[-1].end
                continue
            elif opcode == 0x10:                                  # call
                callee = self._module.functions[immediate]
                taken = len(callee.signature.params)
                arguments = stack[len(stack) - taken:] if taken else []
                del stack[len(stack) - taken:]
                stack.extend(self.invoke(immediate, arguments))
            elif opcode == 0x6B:                                  # i32.sub
                right = stack.pop()
                stack[-1] = (stack[-1] - right) & MASK32
            elif opcode == 0x4E:                                  # i32.ge_s
                right = _signed(stack.pop(), 32)
                stack[-1] = 1 if _signed(stack[-1], 32) >= right else 0
            elif opcode == 0x4A:                                  # i32.gt_s
                right = _signed(stack.pop(), 32)
                stack[-1] = 1 if _signed(stack[-1], 32) > right else 0
            elif opcode == 0x48:                                  # i32.lt_s
                right = _signed(stack.pop(), 32)
                stack[-1] = 1 if _signed(stack[-1], 32) < right else 0
            elif opcode == 0x7C:                                  # i64.add
                right = stack.pop()
                stack[-1] = (stack[-1] + right) & MASK64
            elif opcode == 0x7D:                                  # i64.sub
                right = stack.pop()
                stack[-1] = (stack[-1] - right) & MASK64
            elif opcode == 0x7E:                                  # i64.mul
                right = stack.pop()
                stack[-1] = (stack[-1] * right) & MASK64
            elif opcode == 0x59:                                  # i64.ge_s
                right = _signed(stack.pop(), 64)
                stack[-1] = 1 if _signed(stack[-1], 64) >= right else 0
            elif opcode == 0x55:                                  # i64.gt_s
                right = _signed(stack.pop(), 64)
                stack[-1] = 1 if _signed(stack[-1], 64) > right else 0
            elif opcode == 0x57:                                  # i64.le_s
                right = _signed(stack.pop(), 64)
                stack[-1] = 1 if _signed(stack[-1], 64) <= right else 0
            elif opcode == 0x53:                                  # i64.lt_s
                right = _signed(stack.pop(), 64)
                stack[-1] = 1 if _signed(stack[-1], 64) < right else 0
            elif opcode == 0x50:                                  # i64.eqz
                stack[-1] = 1 if stack[-1] == 0 else 0
            elif opcode == 0x45:                                  # i32.eqz
                stack[-1] = 1 if stack[-1] == 0 else 0
            elif opcode == 0xA7:                                  # i32.wrap_i64
                stack[-1] &= MASK32
            elif opcode == 0xAC:                                  # i64.extend_i32_s
                stack[-1] = _signed(stack[-1], 32) & MASK64
            elif opcode == 0x7F:                                  # i64.div_s
                right = _signed(stack.pop(), 64)
                left = _signed(stack[-1], 64)
                if right == 0:
                    raise WasmTrap("integer divide by zero")
                quotient = abs(left) // abs(right)
                stack[-1] = (
                    -quotient if (left < 0) != (right < 0) else quotient
                ) & MASK64
            elif opcode == 0x81:                                  # i64.rem_s
                right = _signed(stack.pop(), 64)
                left = _signed(stack[-1], 64)
                if right == 0:
                    raise WasmTrap("integer remainder by zero")
                remainder = abs(left) % abs(right)
                stack[-1] = (-remainder if left < 0 else remainder) & MASK64
            elif opcode == 0x85:                                  # i64.xor
                right = stack.pop()
                stack[-1] ^= right
            elif opcode == 0x0E:                                  # br_table
                targets, default = immediate
                index = stack.pop()
                depth = targets[index] if index < len(targets) else default
                pc = self._branch(frames, stack, depth)
                if pc < 0:
                    return stack[len(stack) - results:] if results else []
                continue
            elif opcode == 0x23:                                  # global.get
                stack.append(self._module.globals[immediate])
            elif opcode == 0x0F:                                  # return
                return stack[len(stack) - results:] if results else []
            elif opcode == 0x1A:                                  # drop
                stack.pop()
            elif opcode == 0x00:                                  # unreachable
                raise WasmTrap("the module reached an unreachable instruction")
            else:
                raise WasmTrap(f"opcode 0x{opcode:02x} is not one this runs")
            pc += 1

        return stack[len(stack) - results:] if results else []

    @staticmethod
    def _branch(frames: list[_Frame], stack: list[int], depth: int) -> int:
        """Leaves ``depth`` structures; answers -1 when that leaves the function."""
        if depth >= len(frames):
            return -1
        frame = frames[len(frames) - 1 - depth]
        carried = stack[len(stack) - frame.arity:] if frame.arity else ()
        del frames[len(frames) - 1 - depth:]
        del stack[frame.height:]
        stack.extend(carried)
        return frame.target


def execute_wasm(blob: bytes) -> str:
    """Instantiates a module this file emitted and returns the text it wrote."""
    module = decode_wasm(blob)
    machine = WasmMachine(module)
    length = machine.invoke(module.exports["render"][1])[0]
    offset = module.globals[module.exports["output"][1]]
    return machine.memory[offset : offset + length].decode()


# ======================================================================
# driver
# ======================================================================


STREAM: Final[str] = "-"


def _read_octets(source: str) -> bytes:
    """What ``source`` holds, or what standard input holds when it is ``-``."""
    if source == STREAM:
        return sys.stdin.buffer.read()
    return Path(source).read_bytes()


def _write_octets(destination: str, blob: bytes, executable: bool = False) -> str:
    """Writes ``blob`` where asked, and answers what to call the place.

    A destination of ``-`` is standard output, which cannot be made executable
    and does not need to be: whoever redirects it decides what it becomes.
    """
    if destination == STREAM:
        sys.stdout.buffer.write(blob)
        sys.stdout.buffer.flush()
        return "standard output"
    path = Path(destination)
    path.write_bytes(blob)
    if executable:
        path.chmod(0o755)
    return str(path)


def _hexdump(blob: bytes, width: int = 16) -> str:
    lines = []
    for offset in range(0, len(blob), width):
        chunk = blob[offset : offset + width]
        octets = " ".join(f"{byte:02x}" for byte in chunk)
        text = "".join(chr(byte) if 32 <= byte < 127 else "." for byte in chunk)
        lines.append(f"{offset:08x}  {octets:<{width * 3}} |{text}|")
    return "\n".join(lines)


def _selftest(orders: Sequence[int]) -> int:
    """Differentially test every tier in this file against tier 0."""
    failures = 0
    for order in orders:
        artifacts = synthesize(order).unwrap_or_raise()
        tiers = [("vm", artifacts.rendering)]
        try:
            module = LlvmLoweringBackend().lower(artifacts.module)
            toolchain = LlvmToolchainService()
            toolchain.verify(module)
            with tempfile.TemporaryDirectory(prefix="ouroboros-") as scratch:
                binary = link_executable(module.text, Path(scratch) / "glyph", 2)
                tiers.append(("native", _run(binary, "").removesuffix("\n")))
        except (LlvmToolchainUnavailable, OSError) as exc:
            print(f"  n={order:<3} llvm tiers skipped: {exc}", file=sys.stderr)
        for architecture in MACHINES:
            try:
                if not machine_code_runnable(architecture):
                    raise OSError(f"this host cannot run {architecture} binaries")
                with tempfile.TemporaryDirectory(prefix="ouroboros-") as scratch:
                    binary = write_executable(
                        artifacts.module, Path(scratch) / "glyph", architecture
                    )
                    tiers.append((f"elf({architecture})", _run(binary, "").removesuffix("\n")))
            except (GlyphPlatformError, OSError) as exc:
                print(f"  n={order:<3} elf({architecture}) skipped: {exc}", file=sys.stderr)
        try:
            if not boot_runnable():
                raise OSError("no system emulator is installed")
            with tempfile.TemporaryDirectory(prefix="ouroboros-") as scratch:
                image = Path(scratch) / "glyph.img"
                image.write_bytes(boot_image(artifacts.module))
                tiers.append(("boot", run_boot(image, order).removesuffix("\n")))
        except (GlyphPlatformError, OSError) as exc:
            print(f"  n={order:<3} boot tier skipped: {exc}", file=sys.stderr)
        blob = wasm_module(artifacts.module)
        try:
            if wasm_host() is None:
                raise OSError("no WebAssembly host is installed")
            with tempfile.TemporaryDirectory(prefix="ouroboros-") as scratch:
                path = Path(scratch) / "glyph.wasm"
                path.write_bytes(blob)
                tiers.append(("wasm", run_wasm(path).removesuffix("\n")))
        except (GlyphPlatformError, OSError, subprocess.CalledProcessError) as exc:
            print(f"  n={order:<3} wasm tier skipped: {exc}", file=sys.stderr)
        tiers.append(("read-back", execute_wasm(blob).removesuffix("\n")))
        witness = REFERENCE_DIGESTS.get(order)
        names = ["tier0"] if witness is not None else []
        if witness is None:
            witness = reference_digest(tiers[0][1])
        names.extend(label for label, _ in tiers)
        verdict = "ok"
        for label, produced in tiers:
            if reference_digest(produced) != witness:
                verdict = f"MISMATCH in {label}"
                failures += 1
        print(f"  n={order:<3} {' == '.join(names)}  {verdict}")
    return 1 if failures else 0


def _emit_bootstrap_report(report: BootstrapReport) -> int:
    rule = "-" * 60
    print(rule)
    print(f"stage1.ll  {len(report.stage1_ir.splitlines()):>6} lines   (seed compiled gslc.gsl2)")
    print(f"stage2.ll  {len(report.stage2_ir.splitlines()):>6} lines   (stage1 compiled its own source)")
    print(f"stage3.ll  {len(report.stage3_ir.splitlines()):>6} lines   (stage2 compiled it again)")
    print(rule)
    seed = "[ok]  " if report.seed_agrees else "[FAIL]"
    fixed = "[ok]  " if report.fixpoint else "[FAIL]"
    print(f"  {seed} stage1.ll == stage2.ll   seed and self-hosted compiler agree")
    print(f"  {fixed} stage2.ll == stage3.ll   compiler reproduces itself: FIXPOINT")
    print(rule)
    print(report.glyph)
    print(rule)
    print(f"artifacts in {report.workdir}")
    return 0 if (report.seed_agrees and report.fixpoint) else 1


def _emit_loop_report(report: LoopReport) -> int:
    rule = "-" * 60
    print(rule)
    print("  python3  ->  gslc      the seed compiles the GSL-2 compiler, once")
    print("  gslc     ->  gslc      and from here the language compiles itself")
    print(f"  gslc     ->  glyphc    it compiles the GSL front end"
          f"   ({len(report.glyphc_ir.splitlines())} lines of IR)")
    print("  glyphc   ->  glyph     and that front end compiles the glyph")
    print(rule)
    print(f"  {'motif':<16}{'n':>4}   {'IR == layer 16':<16}glyph == tier 1")
    for case in report.cases:
        identical = "[ok]  " if case.identical else "[FAIL]"
        renders = "[ok]  " if case.renders else "[FAIL]"
        print(f"  {case.motif:<16}{case.order:>4}   {identical:<16}{renders}")
    print(rule)
    for case in report.cases:
        if case.motif == DEFAULT_MOTIF and case.order == 7:
            print(case.glyph)
            print(rule)
            break
    print(f"artifacts in {report.workdir}")
    return 0 if report.clean else 1


def _parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ouroboros",
        description=f"ouroboros {PLATFORM_VERSION} - the pattern, and everything it grew into",
        epilog="A PATH of - is standard input or standard output, whichever the option takes.",
    )
    parser.add_argument("-n", "--order", type=int, help="lattice edge length (odd)")
    parser.add_argument("-m", "--motif", choices=Motif.catalogue(), help="registered motif")
    parser.add_argument("-t", "--theme", choices=Theme.catalogue(), help="rendering theme")
    parser.add_argument("-l", "--lang", choices=sorted(MESSAGE_CATALOGUE), help="diagnostic language")
    parser.add_argument("-j", "--workers", type=int, help="rasteriser worker count")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    parser.add_argument("--no-optimise", action="store_true", help="skip the pass manager")
    parser.add_argument("--no-jit", action="store_true", help="stay in the interpreter tier")
    parser.add_argument("--no-roundtrip", action="store_true", help="skip object serialisation")

    dumps = parser.add_argument_group("introspection")
    dumps.add_argument("--emit-config", action="store_true")
    dumps.add_argument("--emit-source", action="store_true")
    dumps.add_argument("--emit-tokens", action="store_true")
    dumps.add_argument("--emit-ast", action="store_true")
    dumps.add_argument("--emit-scopes", action="store_true")
    dumps.add_argument("--emit-ir", action="store_true")
    dumps.add_argument("--emit-cfg", action="store_true")
    dumps.add_argument("--emit-passes", action="store_true")
    dumps.add_argument("--emit-asm", "-S", action="store_true")
    dumps.add_argument("--emit-container-format", action="store_true")
    dumps.add_argument("--emit-grammar", action="store_true")
    dumps.add_argument("--emit-group", action="store_true")
    dumps.add_argument("--emit-events", action="store_true")
    dumps.add_argument("--emit-container", action="store_true")
    dumps.add_argument("--emit-trace", action="store_true")
    dumps.add_argument("--emit-metrics", action="store_true")
    dumps.add_argument("--emit-everything", action="store_true")
    dumps.add_argument("--self-test", action="store_true")
    dumps.add_argument("--list-motifs", action="store_true")

    backend = parser.add_argument_group("LLVM backend")
    backend.add_argument("--emit-llvm", action="store_true")
    backend.add_argument("--emit-native-assembly", action="store_true")
    backend.add_argument("--emit-object", metavar="PATH")
    backend.add_argument("--verify-llvm", action="store_true")
    backend.add_argument("--jit", action="store_true")
    backend.add_argument("-O", "--opt-level", type=int, choices=range(4), default=0)
    backend.add_argument("--triple", default=TargetProfile().triple)

    machine = parser.add_argument_group("tier 5: machine code, no toolchain")
    machine.add_argument("--emit-elf", metavar="PATH", help="write a static ELF64 executable")
    machine.add_argument("--emit-machine-code", action="store_true")
    machine.add_argument("--machine", choices=sorted(MACHINES), default=host_machine(),
                         help="which machine to write for (default: this one)")

    machine.add_argument("--emit-boot", metavar="PATH",
                         help="write a disk image that needs no kernel either")

    wasm = parser.add_argument_group("tier 6: WebAssembly, no host but an engine")
    wasm.add_argument("--emit-wasm", metavar="PATH", help="write a reactor module")
    wasm.add_argument("--run-wasm", metavar="PATH", help="run one, reading it back here")

    boot = parser.add_argument_group("GSL-2 self-hosting bootstrap")
    boot.add_argument("--bootstrap", action="store_true")
    boot.add_argument("--workdir", metavar="DIR")
    boot.add_argument("--emit-gsl2", choices=("gslc", "glyph", "glyphc"))
    boot.add_argument("--close-the-loop", action="store_true")
    boot.add_argument("--selftest", action="store_true")

    fuzzing = parser.add_argument_group("differential fuzzing")
    fuzzing.add_argument("--fuzz", type=int, metavar="N", help="cross-check N random programs")
    fuzzing.add_argument("--fuzz-seed", type=int, default=0)
    fuzzing.add_argument("--fuzz-native", action="store_true", help="also link and run each case")
    fuzzing.add_argument("--fuzz-loop", action="store_true",
                         help="also compile each case with the GSL-2 front end")
    return parser.parse_args(argv)


def _overrides_from(namespace: argparse.Namespace) -> Mapping[str, str]:
    entries: dict[str, str] = {}
    if namespace.order is not None:
        entries["lattice.order"] = str(namespace.order)
    if namespace.motif is not None:
        entries["motif"] = namespace.motif
    if namespace.theme is not None:
        entries["theme"] = namespace.theme
    if namespace.lang is not None:
        entries["language"] = namespace.lang
    if namespace.workers is not None:
        entries["workers"] = str(namespace.workers)
    if namespace.no_optimise:
        entries["optimise"] = "false"
    if namespace.no_jit:
        entries["jit"] = "false"
    if namespace.no_roundtrip:
        entries["roundtrip"] = "false"
    return entries


def _run_backend(namespace: argparse.Namespace, artifacts: CompilationArtifacts) -> int:
    module = LlvmLoweringBackend(TargetProfile(triple=namespace.triple)).lower(artifacts.module)
    toolchain = LlvmToolchainService()
    try:
        if namespace.opt_level:
            module = toolchain.optimize(module, namespace.opt_level)
        if namespace.verify_llvm:
            print(toolchain.verify(module), file=sys.stderr)
        if namespace.emit_object:
            blob = toolchain.object_code(module)
            written = _write_octets(namespace.emit_object, blob)
            print(f"wrote {written} ({len(blob)} bytes)", file=sys.stderr)
        if namespace.emit_native_assembly:
            sys.stdout.write(toolchain.emit_assembly(module))
        if namespace.emit_llvm:
            sys.stdout.write(module.text)
        if namespace.jit:
            return toolchain.jit_execute(module)
    except LlvmToolchainUnavailable as exc:
        print(f"llvm backend unavailable: {exc}", file=sys.stderr)
        return 3
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    global CATALOG, DIAGNOSTICS
    namespace = _parse_arguments(argv)
    logging.basicConfig(
        level=logging.DEBUG if namespace.verbose > 1 else logging.WARNING,
        stream=sys.stderr,
        format="%(levelname)s %(name)s %(message)s",
    )

    if namespace.list_motifs:
        for key in Motif.catalogue():
            cls = typing.cast(type, Motif.lookup(key))
            print(f"{key:<16} {getattr(cls, 'description', '')}")
        print("themes:", ", ".join(Theme.catalogue()))
        return 0
    if namespace.emit_gsl2:
        sys.stdout.write(
            {"gslc": GSLC_GSL2, "glyph": GLYPH_GSL2, "glyphc": GLYPHC_GSL2}[
                namespace.emit_gsl2
            ]
        )
        return 0
    if namespace.run_wasm:
        sys.stdout.write(execute_wasm(_read_octets(namespace.run_wasm)))
        return 0
    if namespace.selftest:
        return _selftest((3, 5, 7, 9, 11, 15, 21))
    if namespace.fuzz is not None:
        report = fuzz(
            namespace.fuzz, namespace.fuzz_seed, namespace.fuzz_native, namespace.fuzz_loop
        )
        print(report.render())
        return 0 if report.clean else 1
    if namespace.bootstrap:
        try:
            workdir = Path(namespace.workdir) if namespace.workdir else None
            return _emit_bootstrap_report(bootstrap(workdir, namespace.opt_level or 2))
        except (LlvmToolchainUnavailable, subprocess.CalledProcessError) as exc:
            print(f"bootstrap unavailable: {exc}", file=sys.stderr)
            return 3
    if namespace.close_the_loop:
        try:
            workdir = Path(namespace.workdir) if namespace.workdir else None
            return _emit_loop_report(close_the_loop(workdir, namespace.opt_level or 2))
        except (LlvmToolchainUnavailable, subprocess.CalledProcessError) as exc:
            print(f"the loop cannot be closed here: {exc}", file=sys.stderr)
            return 3

    CATALOG = MessageCatalog(namespace.lang or "en")
    DIAGNOSTICS = DiagnosticEngine(CATALOG)
    try:
        configuration = (
            ConfigurationBuilder()
            .with_defaults()
            .with_environment()
            .with_mapping("command-line", _overrides_from(namespace))
            .build()
        )
    except GlyphPlatformError as error:
        print(CATALOG("report.fail", error=error), file=sys.stderr)
        return 2

    if configuration["language"] != CATALOG.language:
        CATALOG = MessageCatalog(str(configuration["language"]))
        DIAGNOSTICS = DiagnosticEngine(CATALOG)
    if namespace.verbose:
        EVENT_BUS.subscribe(
            PlatformEvent, lambda event: print(f"· {event.detail}", file=sys.stderr), priority=10
        )

    everything = namespace.emit_everything
    outcome = Result.attempt(lambda: SynthesisOrchestrator(configuration).run())
    if not outcome.is_ok:
        failure = typing.cast(Err, outcome).failure
        print(CATALOG("report.fail", error=failure), file=sys.stderr)
        if DIAGNOSTICS.records:
            print(DIAGNOSTICS.render(), file=sys.stderr)
        return 2
    artifacts = outcome.unwrap_or_raise()

    sections: tuple[tuple[bool, str, Callable[[], str]], ...] = (
        (namespace.emit_config, "configuration", artifacts.configuration.render),
        (namespace.emit_source, "gsl source", lambda: artifacts.unit.text),
        (namespace.emit_grammar, "grammar", GRAMMAR.render),
        (namespace.emit_tokens, "tokens",
         lambda: "\n".join(f"{str(t.position):>8}  {t.kind.name:<14} {t.lexeme!r}" for t in artifacts.tokens)),
        (namespace.emit_ast, "syntax tree", artifacts.tree.render),
        (namespace.emit_scopes, "symbol table", artifacts.model.scope.render),
        (namespace.emit_ir, "symbolic listing",
         lambda: "\n".join(str(item) for item in artifacts.optimised)),
        (namespace.emit_cfg, "control flow graph", artifacts.graph.render),
        (namespace.emit_passes, "optimisation report", artifacts.passes.render),
        (namespace.emit_asm, "disassembly", artifacts.module.disassembly),
        (namespace.emit_machine_code, f"machine code ({namespace.machine})",
         lambda: _hexdump(machine_text(artifacts.module, namespace.machine))),
        (namespace.emit_container_format, "object container", lambda: _hexdump(artifacts.blob)),
        (namespace.emit_group, "symmetry group", artifacts.machine.group.render),
        (namespace.emit_events, "canvas event stream",
         lambda: "\n".join(
             f"#{event.sequence:03d} {type(event).__name__} "
             f"{len(getattr(event, 'cells', ())):4d} cell(s)"
             for event in artifacts.machine.canvas.events
         )),
        (namespace.emit_container, "composition root",
         lambda: SynthesisOrchestrator(artifacts.configuration).container.render()),
        (namespace.emit_trace, "execution trace", TRACER.render),
        (namespace.emit_metrics, "metrics", METRICS.render),
    )
    for requested, title, renderer in sections:
        if requested or everything:
            print(f"\n=== {title} ===", file=sys.stderr)
            print(renderer(), file=sys.stderr)

    if namespace.self_test or everything:
        print("\n=== assurance ===", file=sys.stderr)
        results = AssuranceSuite().run(artifacts)
        for result in results:
            print(result.render(), file=sys.stderr)
        if not all(result.passed for result in results):
            return 3

    if DIAGNOSTICS.records:
        print(DIAGNOSTICS.render(), file=sys.stderr)
    if namespace.verbose:
        print(CATALOG("report.ok", ms=artifacts.elapsed_ms), file=sys.stderr)

    if namespace.emit_elf:
        blob = machine_code(artifacts.module, namespace.machine)
        written = _write_octets(namespace.emit_elf, blob, executable=True)
        print(f"wrote {written} ({len(blob)} bytes)", file=sys.stderr)

    if namespace.emit_boot:
        blob = boot_image(artifacts.module)
        written = _write_octets(namespace.emit_boot, blob)
        print(f"wrote {written} ({len(blob)} bytes)", file=sys.stderr)

    if namespace.emit_wasm:
        blob = wasm_module(artifacts.module)
        written = _write_octets(namespace.emit_wasm, blob)
        print(f"wrote {written} ({len(blob)} bytes)", file=sys.stderr)

    backend_requested = any(
        (
            namespace.emit_llvm,
            namespace.emit_native_assembly,
            namespace.emit_object,
            namespace.verify_llvm,
            namespace.jit,
            namespace.opt_level,
        )
    )
    if backend_requested:
        return _run_backend(namespace, artifacts)

    if STREAM not in (namespace.emit_elf, namespace.emit_wasm, namespace.emit_boot):
        sys.stdout.write(artifacts.rendering + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
