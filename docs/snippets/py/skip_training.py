from collections.abc import Iterator

from lir import DataStrategy, InstanceData


class AllTest(DataStrategy):
    def apply[DataType: InstanceData](self, instances: DataType) -> Iterator[tuple[DataType, DataType]]:
        yield None, instances
