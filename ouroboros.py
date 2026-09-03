"""ouroboros.

Glyph morphology is expressed declaratively as a set of canonical stroke
generators which are subsequently closed under the action of a cyclic symmetry
group, thereby guaranteeing rotational invariance by construction rather than
by assertion.

    python3 ouroboros.py          render it
    python3 ouroboros.py -n 11    any odd order works

"""

from __future__ import annotations

import abc
import argparse
import enum
import functools
import itertools
import logging
import operator
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import (
    Callable,
    ClassVar,
    Final,
    FrozenSet,
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    Protocol,
    Sequence,
    TypeVar,
    runtime_checkable,
)

__all__ = [
    "GlyphSynthesisError",
    "LatticeDimensionError",
    "StrokeResolutionError",
    "Ink",
    "Coordinate",
    "SquareLattice",
    "AbstractStroke",
    "AxialSpineStroke",
    "DextrorotatoryLimbStroke",
    "CyclicSymmetryGroup",
    "RasterizationStrategy",
    "GlyphRenderPipelineBuilder",
    "synthesize",
]

_LOGGER: Final[logging.Logger] = logging.getLogger(__name__)
_LOGGER.addHandler(logging.NullHandler())

_T_co = TypeVar("_T_co", covariant=True)

DEFAULT_LATTICE_ORDER: Final[int] = 7
QUARTER_TURNS_PER_REVOLUTION: Final[int] = 4


class GlyphSynthesisError(RuntimeError):
    """Root of the exception hierarchy for all synthesis-time failures."""


class LatticeDimensionError(GlyphSynthesisError, ValueError):
    """Raised when a lattice order violates the centro-symmetry precondition."""


class StrokeResolutionError(GlyphSynthesisError, LookupError):
    """Raised when the stroke registry cannot satisfy a topology request."""


class Ink(enum.Enum):
    """Terminal-addressable occupancy states of a single lattice cell."""

    OCCUPIED = "*"
    VACANT = " "

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, order=True)
class Coordinate:
    """An immutable, hashable ordinate pair in row-major lattice space."""

    row: int
    column: int

    def displaced_by(self, other: "Coordinate") -> "Coordinate":
        return Coordinate(self.row + other.row, self.column + other.column)

    def inverted(self) -> "Coordinate":
        return Coordinate(-self.row, -self.column)

    def rotated_quarter_turn_about(self, pivot: "Coordinate") -> "Coordinate":
        delta = self.displaced_by(pivot.inverted())
        return Coordinate(delta.column, -delta.row).displaced_by(pivot)


class _ValidatedOrderDescriptor:
    """Non-data descriptor enforcing the odd-order lattice invariant."""

    def __set_name__(self, owner: type, name: str) -> None:
        self._private_name = f"_{name}"

    def __get__(self, instance: object, owner: type | None = None) -> int:
        if instance is None:
            return self  # type: ignore[return-value]
        return getattr(instance, self._private_name)

    def __set__(self, instance: object, value: int) -> None:
        if not isinstance(value, int) or isinstance(value, bool):
            raise LatticeDimensionError(f"non-integral lattice order: {value!r}")
        if value < 3 or value % 2 == 0:
            raise LatticeDimensionError(
                f"lattice order must be an odd integer >= 3, received {value!r}"
            )
        object.__setattr__(instance, self._private_name, value)


class SquareLattice:
    """A finite, centro-symmetric integer lattice of odd order."""

    order = _ValidatedOrderDescriptor()

    def __init__(self, order: int = DEFAULT_LATTICE_ORDER) -> None:
        self.order = order

    @property
    def apothem(self) -> int:
        return self.order // 2

    @property
    def centroid(self) -> Coordinate:
        return Coordinate(self.apothem, self.apothem)

    @property
    def extremum(self) -> int:
        return self.order - 1

    def __iter__(self) -> Iterator[Coordinate]:
        return (
            Coordinate(r, c)
            for r, c in itertools.product(range(self.order), repeat=2)
        )

    def rows(self) -> Iterator[Sequence[Coordinate]]:
        return (
            tuple(Coordinate(r, c) for c in range(self.order))
            for r in range(self.order)
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(order={self.order})"


class _StrokeRegistryMeta(abc.ABCMeta):
    """Metaclass maintaining an auto-populated registry of stroke topologies."""

    _registry: ClassVar[MutableMapping[str, type["AbstractStroke"]]] = {}

    def __new__(mcls, name, bases, namespace, /, **kwargs):
        cls = super().__new__(mcls, name, bases, namespace, **kwargs)
        if not getattr(cls, "__abstractmethods__", None) and bases:
            mcls._registry[cls.topology_identifier()] = cls
            _LOGGER.debug("registered stroke topology %s", cls.__name__)
        return cls

    @classmethod
    def resolve(mcls, identifier: str) -> type["AbstractStroke"]:
        try:
            return mcls._registry[identifier]
        except KeyError as exc:
            raise StrokeResolutionError(
                f"no stroke topology registered under {identifier!r}; "
                f"known topologies: {sorted(mcls._registry)}"
            ) from exc

    @classmethod
    def instantiate_all(mcls, lattice: SquareLattice) -> Sequence["AbstractStroke"]:
        return tuple(cls(lattice) for cls in mcls._registry.values())


class AbstractStroke(metaclass=_StrokeRegistryMeta):
    """Contract for a canonical, pre-symmetrization stroke generator."""

    def __init__(self, lattice: SquareLattice) -> None:
        self._lattice = lattice

    @classmethod
    def topology_identifier(cls) -> str:
        return cls.__name__.removesuffix("Stroke").lower()

    @property
    def lattice(self) -> SquareLattice:
        return self._lattice

    @abc.abstractmethod
    def canonical_support(self) -> Iterable[Coordinate]:
        """Yield the fundamental domain of this stroke prior to group closure."""

    def __iter__(self) -> Iterator[Coordinate]:
        return iter(self.canonical_support())


class AxialSpineStroke(AbstractStroke):

    def canonical_support(self) -> Iterable[Coordinate]:
        return (
            Coordinate(r, self.lattice.apothem) for r in range(self.lattice.order)
        )


class DextrorotatoryLimbStroke(AbstractStroke):

    def canonical_support(self) -> Iterable[Coordinate]:
        return (
            Coordinate(0, c)
            for c in range(self.lattice.apothem + 1, self.lattice.order)
        )


@dataclass(frozen=True)
class CyclicSymmetryGroup:
    """The finite cyclic group C_n acting on lattice coordinates by rotation."""

    pivot: Coordinate
    cardinality: int = QUARTER_TURNS_PER_REVOLUTION

    def orbit(self, seed: Coordinate) -> Iterator[Coordinate]:
        cursor = seed
        for _ in range(self.cardinality):
            yield cursor
            cursor = cursor.rotated_quarter_turn_about(self.pivot)

    def close(self, support: Iterable[Coordinate]) -> FrozenSet[Coordinate]:
        return frozenset(
            itertools.chain.from_iterable(self.orbit(point) for point in support)
        )


@runtime_checkable
class RasterizationStrategy(Protocol[_T_co]):
    """Structural contract for occupancy-to-presentation transducers."""

    def rasterize(
        self, lattice: SquareLattice, support: FrozenSet[Coordinate]
    ) -> _T_co: ...


@dataclass(frozen=True)
class DelimitedAsteriskRasterizer:
    """Materializes an occupancy set as a delimiter-separated character mosaic."""

    intercell_delimiter: str = " "
    interline_delimiter: str = "\n"

    def rasterize(
        self, lattice: SquareLattice, support: FrozenSet[Coordinate]
    ) -> str:
        return self.interline_delimiter.join(
            self.intercell_delimiter.join(
                str(Ink.OCCUPIED if cell in support else Ink.VACANT)
                for cell in row
            ).rstrip()
            for row in lattice.rows()
        )


@dataclass
class _SynthesisTelemetry:
    """Mutable accumulator threaded through the render session context."""

    observed_cells: int = 0
    observed_strokes: int = 0
    subscribers: list[Callable[[str, object], None]] = field(default_factory=list)

    def emit(self, topic: str, payload: object) -> None:
        for subscriber in self.subscribers:
            subscriber(topic, payload)


class GlyphRenderPipelineBuilder:
    """Fluent builder assembling an immutable glyph synthesis pipeline."""

    def __init__(self) -> None:
        self._lattice: SquareLattice | None = None
        self._strategy: RasterizationStrategy[str] | None = None
        self._topologies: list[str] = []
        self._telemetry = _SynthesisTelemetry()

    def upon_lattice_of_order(self, order: int) -> "GlyphRenderPipelineBuilder":
        self._lattice = SquareLattice(order)
        return self

    def with_topology(self, identifier: str) -> "GlyphRenderPipelineBuilder":
        self._topologies.append(identifier)
        return self

    def with_all_registered_topologies(self) -> "GlyphRenderPipelineBuilder":
        self._topologies = list(_StrokeRegistryMeta._registry)
        return self

    def rasterized_by(
        self, strategy: RasterizationStrategy[str]
    ) -> "GlyphRenderPipelineBuilder":
        if not isinstance(strategy, RasterizationStrategy):
            raise GlyphSynthesisError(f"{strategy!r} violates RasterizationStrategy")
        self._strategy = strategy
        return self

    def observed_by(
        self, subscriber: Callable[[str, object], None]
    ) -> "GlyphRenderPipelineBuilder":
        self._telemetry.subscribers.append(subscriber)
        return self

    @contextmanager
    def _session(self) -> Iterator[_SynthesisTelemetry]:
        self._telemetry.emit("session.opened", self._lattice)
        try:
            yield self._telemetry
        finally:
            self._telemetry.emit("session.closed", self._telemetry)

    def build(self) -> Callable[[], str]:
        lattice = self._lattice or SquareLattice()
        strategy = self._strategy or DelimitedAsteriskRasterizer()
        topologies = self._topologies or list(_StrokeRegistryMeta._registry)
        group = CyclicSymmetryGroup(pivot=lattice.centroid)

        @functools.lru_cache(maxsize=None)
        def _materialize() -> str:
            with self._session() as telemetry:
                strokes = tuple(
                    _StrokeRegistryMeta.resolve(identifier)(lattice)
                    for identifier in topologies
                )
                telemetry.observed_strokes = len(strokes)
                support = functools.reduce(
                    operator.or_,
                    (group.close(stroke) for stroke in strokes),
                    frozenset(),
                )
                telemetry.observed_cells = len(support)
                telemetry.emit("support.resolved", support)
                return strategy.rasterize(lattice, support)

        return _materialize


def synthesize(order: int = DEFAULT_LATTICE_ORDER) -> str:
    """Convenience facade over the canonical pipeline configuration."""
    return (
        GlyphRenderPipelineBuilder()
        .upon_lattice_of_order(order)
        .with_all_registered_topologies()
        .rasterized_by(DelimitedAsteriskRasterizer())
        .observed_by(lambda topic, payload: _LOGGER.debug("%s -> %r", topic, payload))
        .build()
    )()


def _parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, add_help=True)
    parser.add_argument(
        "-n",
        "--order",
        type=int,
        default=DEFAULT_LATTICE_ORDER,
        help="odd order of the underlying square lattice",
    )
    parser.add_argument("-v", "--verbose", action="count", default=0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    namespace = _parse_arguments(argv)
    logging.basicConfig(
        level=logging.DEBUG if namespace.verbose else logging.WARNING,
        stream=sys.stderr,
    )
    try:
        sys.stdout.write(synthesize(namespace.order) + "\n")
    except GlyphSynthesisError as exc:
        print(f"synthesis aborted: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
