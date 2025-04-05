# -*- coding: utf-8 -*-
from kag.common.registry import Registrable


class ShardingInfo(Registrable):
    """
    A class representing sharding information for distributed computing.
    ShardingInfo 类用于在分布式计算中管理和查询分片信息。它继承自 Registrable 类
    This class provides methods to manage and query sharding information across
    multiple machines, instances, and processes. It inherits from the `Registrable`
    class.

    Attributes:
        machine_id (int): The ID of the current machine. Default is 0.
        machine_count (int): The total number of machines. Default is 1.
        instance_id (int): The ID of the current instance. Default is 0.
        instance_count (int): The total number of instances. Default is 1.
        process_id (int): The ID of the current process. Default is 0.
        process_count (int): The total number of processes. Default is 1.
        shard_id (int, optional): The ID of the current shard. Default is None.
        shard_count (int, optional): The total number of shards. Default is None.
        shard_by_machine (bool): Whether to shard by machine. Default is True.
        shard_by_instance (bool): Whether to shard by instance. Default is True.
        shard_by_process (bool): Whether to shard by process. Default is True.

        machine_id (int): 当前机器的ID，默认值为0。
        machine_count (int): 机器总数，默认值为1。
        instance_id (int): 当前实例的ID，默认值为0。
        instance_count (int): 实例总数，默认值为1。
        process_id (int): 当前进程的ID，默认值为0。
        process_count (int): 进程总数，默认值为1。
        shard_id (int, optional): 当前分片的ID，默认值为None。
        shard_count (int, optional): 分片总数，默认值为None。
        shard_by_machine (bool): 是否按机器分片，默认值为True。
       shard_by_instance (bool): 是否按实例分片，默认值为True。
       shard_by_process (bool): 是否按进程分片，默认值为True。
    """

    def __init__(
        self,
        machine_id: int = 0,
        machine_count: int = 1,
        instance_id: int = 0,
        instance_count: int = 1,
        process_id: int = 0,
        process_count: int = 1,
        shard_id: int = None,
        shard_count: int = None,
    ):
        """
        Initializes a new instance of the ShardingInfo class.

        Args:
            machine_id (int): The ID of the current machine. Default is 0.
            machine_count (int): The total number of machines. Default is 1.
            instance_id (int): The ID of the current instance. Default is 0.
            instance_count (int): The total number of instances. Default is 1.
            process_id (int): The ID of the current process. Default is 0.
            process_count (int): The total number of processes. Default is 1.
            shard_id (int, optional): The ID of the current shard. Default is None.
            shard_count (int, optional): The total number of shards. Default is None.
        """
        self.instance_id = instance_id
        self.instance_count = instance_count
        self.machine_id = machine_id
        self.machine_count = machine_count
        self.process_id = process_id
        self.process_count = process_count
        self.shard_id = shard_id
        self.shard_count = shard_count

        self.shard_by_machine = True
        self.shard_by_instance = True
        self.shard_by_process = True

    def shard_by(
        self, machine: bool = True, instance: bool = True, process: bool = True
    ):
        """
        Configures the sharding strategy by specifying whether to shard by machine,
        instance, or process.

        配置分片策略，通过指定是否按机器、实例或进程分片。


        Args:
            machine (bool): Whether to shard by machine. Default is True.
            instance (bool): Whether to shard by instance. Default is True.
            process (bool): Whether to shard by process. Default is True.
        """
        self.shard_by_machine = machine
        self.shard_by_instance = instance
        self.shard_by_process = process

    def get_rank(self):
        """
        Returns the rank of the current shard based on the configured sharding strategy.

        Returns:
            int: The rank of the current shard.
        """
        if self.shard_id is not None:
            return self.shard_id
        if self.shard_by_machine:
            machine_id = self.machine_id
        else:
            machine_id = 0
        if self.shard_by_instance:
            instance_id, instance_count = self.instance_id, self.instance_count
        else:
            instance_id, instance_count = 0, 1
        if self.shard_by_process:
            process_id, process_count = self.process_id, self.process_count
        else:
            process_id, process_count = 0, 1

        return process_count * (machine_id * instance_count + instance_id) + process_id

    def get_world_size(self):
        """
        Returns the total number of shards in the world based on the configured sharding strategy.

        Returns:
            int: The total number of shards.
        """
        if self.shard_count is not None:
            return self.shard_count
        world_size = 1
        if self.shard_by_machine:
            world_size *= self.machine_count
        if self.shard_by_instance:
            world_size *= self.instance_count
        if self.shard_by_process:
            world_size *= self.process_count
        return world_size

    def get_sharding_range(self, total: int):
        """
        Returns the range of indices that the current shard is responsible for.

        Args:
            total (int): The total number of items to be sharded.

        Returns:
            Tuple[int, int]: A tuple containing the start and end indices of the range.
        """
        rank = self.get_rank()
        world_size = self.get_world_size()
        if total % world_size == 0:
            workload = total // world_size
        else:
            workload = total // world_size + 1
        start = workload * rank
        end = min(total, workload * (rank + 1))
        return start, end

    @property
    def is_master_process(self):
        """
        Checks if the current process is the master process.

        Returns:
            bool: True if the current process is the master process, False otherwise.
        """
        return self.process_id == 0

    @property
    def is_master_instance(self):
        """
        Checks if the current instance is the master instance.

        Returns:
            bool: True if the current instance is the master instance, False otherwise.
        """
        return self.instance_id == 0

    @property
    def is_master_machine(self):
        """
        Checks if the current machine is the master machine.

        Returns:
            bool: True if the current machine is the master machine, False otherwise.
        """
        return self.machine_id == 0

    def __str__(self):
        """
        Returns a string representation of the ShardingInfo object.

        Returns:
            str: A string containing the rank, world size, and other sharding details.
        """
        content = (
            f"ShardingInfo: rank={self.get_rank()}, world_size={self.get_world_size()}, "
            f"machine: {self.machine_id}/{self.machine_count}, "
            f"instance: {self.instance_id}/{self.instance_count}, "
            f"process: {self.process_id}/{self.process_count}"
        )
        return content

    __repr__ = __str__

    def copy(self):
        """
        Creates a copy of the current ShardingInfo object.

        Returns:
            ShardingInfo: A new instance of ShardingInfo with the same attributes.
        """
        return ShardingInfo(
            self.machine_id,
            self.machine_count,
            self.instance_id,
            self.instance_count,
            self.process_id,
            self.process_count,
            self.shard_id,
            self.shard_count,
        )


ShardingInfo.register("base")(ShardingInfo)
