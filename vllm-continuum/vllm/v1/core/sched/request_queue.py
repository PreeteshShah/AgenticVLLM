# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import heapq
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterable, Iterator
from enum import Enum
from typing import Optional, Tuple

from vllm.v1.request import Request
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1
import time

class SchedulingPolicy(Enum):
    """Enum for scheduling policies."""
    FCFS = "fcfs"
    PRIORITY = "priority"
    CONTINUUM = "continuum"

class RequestQueue(ABC):
    """Abstract base class for request queues."""

    @abstractmethod
    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to the policy."""
        pass

    @abstractmethod
    def pop_request(self) -> Request:
        """Pop a request from the queue according to the policy."""
        pass

    @abstractmethod
    def peek_request(self) -> Request:
        """Peek at the request at the front of the queue without removing it."""
        pass

    @abstractmethod
    def prepend_request(self, request: Request) -> None:
        """Prepend a request to the front of the queue."""
        pass

    @abstractmethod
    def prepend_requests(self, requests: RequestQueue) -> None:
        """Prepend all requests from another queue to the front of this
        queue."""
        pass

    @abstractmethod
    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        pass

    @abstractmethod
    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        pass

    @abstractmethod
    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Get number of requests in queue."""
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to the policy."""
        pass

    @abstractmethod
    def __reversed__(self) -> Iterator[Request]:
        """Iterate over the queue in reverse order."""
        pass


class FCFSRequestQueue(deque[Request], RequestQueue):
    """A first-come-first-served queue that supports deque operations."""

    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to FCFS policy."""
        self.append(request)

    def pop_request(self) -> Request:
        """Pop a request from the queue according to FCFS policy."""
        return self.popleft()

    def peek_request(self) -> Request:
        """Peek at the next request in the queue without removing it."""
        if not self:
            raise IndexError("peek from an empty queue")
        return self[0]

    def prepend_request(self, request: Request) -> None:
        """Prepend a request to the front of the queue."""
        self.appendleft(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        """Prepend all requests from another queue to the front of this
        queue."""
        self.extendleft(reversed(requests))

    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        self.remove(request)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        requests_to_remove = set(requests)
        filtered_requests = [
            req for req in self if req not in requests_to_remove
        ]
        # deque does not support in-place filtering, so we need to clear
        # and extend
        self.clear()
        self.extend(filtered_requests)

    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        return len(self) > 0

    def __len__(self) -> int:
        """Get number of requests in queue."""
        return super().__len__()

    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to FCFS policy."""
        return super().__iter__()

    def __reversed__(self) -> Iterator[Request]:
        """Iterate over the queue in reverse order."""
        return super().__reversed__()


class PriorityRequestQueue(RequestQueue):
    """
    A priority queue that supports heap operations.

    Requests with a smaller value of `priority` are processed first.
    If multiple requests have the same priority, the one with the earlier
    `arrival_time` is processed first.
    """

    def __init__(self) -> None:
        self._heap: list[tuple[int, float, Request]] = []

    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to priority policy."""
        heapq.heappush(self._heap,
                       (request.priority, request.arrival_time, request))

    def pop_request(self) -> Request:
        """Pop a request from the queue according to priority policy."""
        if not self._heap:
            raise IndexError("pop from empty heap")
        _, _, request = heapq.heappop(self._heap)
        return request

    def peek_request(self) -> Request:
        """Peek at the next request in the queue without removing it."""
        if not self._heap:
            raise IndexError("peek from empty heap")
        _, _, request = self._heap[0]
        return request

    def prepend_request(self, request: Request) -> None:
        """Add a request to the queue according to priority policy.
        
        Note: In a priority queue, there is no concept of prepending to the 
        front. Requests are ordered by (priority, arrival_time)."""
        self.add_request(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        """Add all requests from another queue according to priority policy.
        
        Note: In a priority queue, there is no concept of prepending to the 
        front. Requests are ordered by (priority, arrival_time)."""
        for request in requests:
            self.add_request(request)

    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        self._heap = [(p, t, r) for p, t, r in self._heap if r != request]
        heapq.heapify(self._heap)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        requests_to_remove = set(requests)
        self._heap = [(p, t, r) for p, t, r in self._heap
                      if r not in requests_to_remove]
        heapq.heapify(self._heap)

    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        return bool(self._heap)

    def __len__(self) -> int:
        """Get number of requests in queue."""
        return len(self._heap)

    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to priority policy."""
        heap_copy = self._heap[:]
        while heap_copy:
            _, _, request = heapq.heappop(heap_copy)
            yield request

    def __reversed__(self) -> Iterator[Request]:
        """Iterate over the queue in reverse priority order."""
        return reversed(list(self))

# TODO (Hanchen) need to implement ContinuumRequestQueue that schedules requests based on the last func call, it can call another predictor class if needed
class ContinuumRequestQueue(deque[Request], RequestQueue):
    
    def __init__(self) -> None:
        super().__init__()
        # Track the first entry time for each job_id
        self.job_id_first_entry_time: dict[str, float] = {}
   
    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to FCFS policy."""
        # Record the first entry time for this job_id if not already recorded
        if request.job_id not in self.job_id_first_entry_time:
            self.job_id_first_entry_time[request.job_id] = request.arrival_time
        self.append(request)

    def pop_request(self, pinned_requests: list[Tuple[Request, float]], kv_cache_manager: KVCacheManager, connector: KVConnectorBase_V1, preferred_binding: Optional[str] = None) -> Request:
        """Pop a request from the queue according to continuum policy."""
        request = self.peek_request(pinned_requests, kv_cache_manager, connector, preferred_binding)
        self.remove_request(request)
        return request

    # NOTE (Hanchen): priority is pinned request -> job_id level FCFS
    # When preferred_binding is set (CONTINUUM_RG_MODE=binding), the job-FCFS
    # fallback is restricted to requests whose binding_type matches the binding
    # currently dominating the running batch, so prefill-bound and decode-bound
    # work coalesces into separate batches instead of interleaving. Pinning
    # (KV reuse) is unchanged and always wins. preferred_binding=None reproduces
    # the original job-only behavior exactly.
    def peek_request(self, pinned_requests: list[Tuple[Request, float]], kv_cache_manager: KVCacheManager, connector: KVConnectorBase_V1, preferred_binding: Optional[str] = None) -> Request:
        if not self:
            raise IndexError("peek from an empty queue")
        # Extract just the requests from pinned_requests tuples
        pinned_request_job_id_set = {req.job_id for req, _ in pinned_requests}

        # First, use the pinned request
        earliest_request = None
        earliest_entry_time = float('inf')
        for request in self:
            if request.job_id in pinned_request_job_id_set:
                job_entry_time = self.job_id_first_entry_time.get(request.job_id, request.arrival_time)
                if job_entry_time < earliest_entry_time:
                    earliest_entry_time = job_entry_time
                    earliest_request = request

        if earliest_request is not None:
            return earliest_request

        # Binding-grouped admission: among waiting requests, prefer those whose
        # binding_type matches the running batch's dominant binding. Falls back
        # to all requests if none match (or if preferred_binding is None).
        def earliest_by_job_fcfs(candidates) -> Optional[Request]:
            best = None
            best_entry_time = float('inf')
            for request in candidates:
                job_entry_time = self.job_id_first_entry_time.get(request.job_id, request.arrival_time)
                if job_entry_time < best_entry_time:
                    best_entry_time = job_entry_time
                    best = request
            return best

        if preferred_binding is not None:
            same_binding = [r for r in self if r.binding_type == preferred_binding]
            chosen = earliest_by_job_fcfs(same_binding)
            if chosen is not None:
                return chosen

        # Otherwise, use job_id level FCFS: find the request whose job_id has the earliest first entry time
        chosen = earliest_by_job_fcfs(self)
        if chosen is not None:
            return chosen
        raise IndexError("peek from an empty queue")

  #  The blow implementation prioritize pineed request 
    # def peek_request(self, pinned_requests: list[Tuple[Request, float]]) -> Request:
    #     """Peek at the next request in the queue without removing it."""
    #     if not self:
    #         raise IndexError("peek from an empty queue")
    #     # Extract just the requests from pinned_requests tuples
    #     pinned_request_job_id_set = {req.job_id for req, _ in pinned_requests}
        
    #     # First, check if any of the requests in the queue are pinned
    #     for request in self:
    #         if request.job_id in pinned_request_job_id_set:
    #             print(f"Pinned request found for job: {request.job_id}")
    #             return request
        
    #     # If no pinned requests found, return the head of the queue
    #     if self:
    #         return self[0]


    def prepend_request(self, request: Request) -> None:
        """Prepend a request to the front of the queue."""
        # Record the first entry time for this job_id if not already recorded
        if request.job_id not in self.job_id_first_entry_time:
            self.job_id_first_entry_time[request.job_id] = request.arrival_time
        self.appendleft(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        """Prepend all requests from another queue to the front of this
        queue."""
        # Record first entry times for new job_ids
        for request in requests:
            if request.job_id not in self.job_id_first_entry_time:
                self.job_id_first_entry_time[request.job_id] = request.arrival_time
        self.extendleft(reversed(requests))

    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        self.remove(request)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        requests_to_remove = set(requests)
        filtered_requests = [
            req for req in self if req not in requests_to_remove
        ]
        # deque does not support in-place filtering, so we need to clear
        # and extend
        self.clear()
        self.extend(filtered_requests)

    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        return len(self) > 0

    def __len__(self) -> int:
        """Get number of requests in queue."""
        return super().__len__()

    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to FCFS policy."""
        return super().__iter__()

    def __reversed__(self) -> Iterator[Request]:
        """Iterate over the queue in reverse order."""
        return super().__reversed__()

def create_request_queue(policy: SchedulingPolicy) -> RequestQueue:
    """Create request queue based on scheduling policy."""
    if policy == SchedulingPolicy.PRIORITY:
        return PriorityRequestQueue()
    elif policy == SchedulingPolicy.FCFS:
        return FCFSRequestQueue()
    elif policy == SchedulingPolicy.CONTINUUM:
        return ContinuumRequestQueue()
    else:
        raise ValueError(f"Unknown scheduling policy: {policy}")
