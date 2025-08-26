from __future__ import annotations

import asyncio
import dataclasses
import os
import signal
import socket
import time
import typing
import uuid
import logging
import copy as _copy

import docker  # type: ignore
import httpx  # type: ignore
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# -----------------------------
# Configuration (env-overridable)
# -----------------------------

IMAGE_NAME = os.environ.get("ANDROID_ENV_IMAGE", "android-env:latest")
POOL_TARGET = int(os.environ.get("POOL_TARGET", "36"))
POOL_MIN_IDLE = int(os.environ.get("POOL_MIN_IDLE", "4"))
POOL_MAX = int(os.environ.get("POOL_MAX", str(max(POOL_TARGET, 40))))
START_TIMEOUT_S = int(os.environ.get("START_TIMEOUT_S", "180"))
IDLE_TIMEOUT_S = int(os.environ.get("IDLE_TIMEOUT_S", "60"))
HEALTH_INTERVAL_S = float(os.environ.get("HEALTH_INTERVAL_S", "5"))
LEASE_TTL_DEFAULT_S = int(os.environ.get("LEASE_TTL_DEFAULT_S", "60"))
IDLE_RESET_ON_RELEASE = os.environ.get("IDLE_RESET_ON_RELEASE", "true").lower() in {"1","true","yes"}
IDLE_RESET_ON_READY = os.environ.get("IDLE_RESET_ON_READY", "true").lower() in {"1","true","yes"}
START_BATCH_SIZE = int(os.environ.get("START_BATCH_SIZE", "4"))
SLEEP_BETWEEN_STARTS_S = float(os.environ.get("SLEEP_BETWEEN_STARTS_S", "2"))

# Proxy HTTP timeouts (manager -> env containers)
PROXY_CONNECT_TIMEOUT_S = float(os.environ.get("PROXY_CONNECT_TIMEOUT_S", "3"))
PROXY_READ_TIMEOUT_S = float(os.environ.get("PROXY_READ_TIMEOUT_S", "60"))
PROXY_WRITE_TIMEOUT_S = float(os.environ.get("PROXY_WRITE_TIMEOUT_S", "60"))
PROXY_POOL_TIMEOUT_S = float(os.environ.get("PROXY_POOL_TIMEOUT_S", "5"))

# Fixed host port range for env containers (inclusive)
PORT_RANGE_START = int(os.environ.get("PORT_RANGE_START", "5000"))
PORT_RANGE_END = int(os.environ.get("PORT_RANGE_END", "5100"))

# Container resource limits
CONTAINER_NANO_CPUS = int(float(os.environ.get("CONTAINER_CPUS", "3")) * 1e9)
CONTAINER_MEM_LIMIT = os.environ.get("CONTAINER_MEM", "6g")

# Runtime toggles
ENABLE_KVM = os.environ.get("ENABLE_KVM", "true").lower() in {"1","true","yes"}
USE_PRIVILEGED = os.environ.get("USE_PRIVILEGED", "false").lower() in {"1","true","yes"}

# CPU pinning (non-overlapping core allocation)
CPUSET_ENABLE = os.environ.get("CPUSET_ENABLE", "true").lower() in {"1","true","yes"}
CPUSET_CORES_PER_ENV = int(os.environ.get("CPUSET_CORES_PER_ENV", str(max(1, int(CONTAINER_NANO_CPUS / 1e9)))))
CPUSET_START_CORE = int(os.environ.get("CPUSET_START_CORE", "0"))
CPUSET_EXCLUDE = os.environ.get("CPUSET_EXCLUDE", "")  # e.g., "0-1,16"

# Emulator/env autoload settings
DEFAULT_TASK_PATH = os.environ.get("DEFAULT_TASK_PATH", "/tasks/dummy.textproto")
AUTOLOAD_MODE = os.environ.get("AUTOLOAD_MODE", "emulator").lower()
ADB_PORT = int(os.environ.get("ADB_PORT", "5555"))
EMULATOR_CONSOLE_PORT = int(os.environ.get("EMULATOR_CONSOLE_PORT", "5554"))
GRPC_PORT = int(os.environ.get("GRPC_PORT", "8554"))
EMULATOR_TIMEOUT = os.environ.get("EMULATOR_TIMEOUT", "")

# Docker objects names
DOCKER_NETWORK = os.environ.get("DOCKER_NETWORK", "android-env-net")
SDK_VOLUME = os.environ.get("SDK_VOLUME", "android-sdk")
# Host to advertise in direct mode (public IP/DNS). If empty, falls back to manager hostname.
DIRECT_HOST = os.environ.get("DIRECT_HOST", "")

# Health endpoint path on env containers
HEALTH_PATH = "/health"


# -----------------------------
# Logging setup (uvicorn-like level prefix, no timestamp)
# -----------------------------

class _UvicornLikeFormatter(logging.Formatter):
    def __init__(self) -> None:
        super().__init__("%(levelprefix)s %(message)s")

    def formatMessage(self, record: logging.LogRecord) -> str:  # noqa: N802
        recordcopy = _copy.copy(record)
        levelname = recordcopy.levelname
        separator = " " * max(1, 8 - len(levelname))
        recordcopy.__dict__["levelprefix"] = f"{levelname}:{separator}"
        return super().formatMessage(recordcopy)


def _setup_logging() -> None:
    logger = logging.getLogger(__name__)
    # Avoid adding duplicate handlers if already configured by runtime
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        formatter = _UvicornLikeFormatter()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


_setup_logging()

# -----------------------------
# Data models
# -----------------------------


class LeaseRequest(BaseModel):
    worker_id: str
    ttl_s: int | None = None


class LeaseResponse(BaseModel):
    env_id: str
    lease_id: str
    ttl_s: int
    host: str | None = None
    port: int | None = None


class RenewRequest(BaseModel):
    lease_id: str
    ttl_s: int | None = None


class ReleaseRequest(BaseModel):
    lease_id: str


@dataclasses.dataclass
class ContainerRecord:
    container_id: str
    name: str
    host_port: int
    network_ip: str | None
    status: str  # starting | idle | leased | unhealthy
    created_at: float
    last_health_at: float
    lease_id: str | None = None
    lease_expiry_ts: float | None = None
    worker_id: str | None = None
    cpuset_cores: list[int] | None = None


class EnvPoolState:
    def __init__(self) -> None:
        self._docker = docker.from_env()
        self._records: dict[str, ContainerRecord] = {}
        self._lock = asyncio.Lock()
        timeout = httpx.Timeout(
            connect=PROXY_CONNECT_TIMEOUT_S,
            read=PROXY_READ_TIMEOUT_S,
            write=PROXY_WRITE_TIMEOUT_S,
            pool=PROXY_POOL_TIMEOUT_S,
        )
        self._client_session = httpx.AsyncClient(timeout=timeout)
        self._hostname = "127.0.0.1"
        self._allocated_ports: set[int] = set()
        # CPU pinning state
        self._allocated_cores: set[int] = set()
        self._host_total_cpus: int = max(1, os.cpu_count() or 1)
        self._allowed_cores: list[int] = _compute_allowed_cores(
            total=self._host_total_cpus,
            start=CPUSET_START_CORE,
            exclude_str=CPUSET_EXCLUDE,
        )
        self._ensure_prerequisites()

    def _ensure_prerequisites(self) -> None:
        # Ensure network exists
        try:
            self._docker.networks.get(DOCKER_NETWORK)
        except docker.errors.NotFound:
            self._docker.networks.create(DOCKER_NETWORK, driver="bridge")
        # Ensure SDK cache volume exists
        try:
            self._docker.volumes.get(SDK_VOLUME)
        except docker.errors.NotFound:
            self._docker.volumes.create(name=SDK_VOLUME)

    async def close(self) -> None:
        await self._client_session.aclose()

    async def stats(self) -> dict[str, typing.Any]:
        async with self._lock:
            counts = {"starting": 0, "idle": 0, "leased": 0, "unhealthy": 0}
            for r in self._records.values():
                counts[r.status] = counts.get(r.status, 0) + 1
            return {
                "total": len(self._records),
                **counts,
                "target": POOL_TARGET,
                "min_idle": POOL_MIN_IDLE,
                "max": POOL_MAX,
            }

    async def ensure_capacity(self) -> None:
        logger = logging.getLogger(__name__)
        while True:
            async with self._lock:
                total = len(self._records)
                available_slots = max(0, self._available_capacity())
                needed = max(0, POOL_TARGET - total)
                to_create = min(needed, available_slots, START_BATCH_SIZE)
            logger.info(
                f"ensure_capacity: total={total}, target={POOL_TARGET}, available_slots={available_slots}, needed={needed}, to_create={to_create}"
            )
            if to_create <= 0:
                break
            for i in range(to_create):
                try:
                    await self._create_container()
                    logger.info(f"Created container batch item {i+1}/{to_create}")
                    await asyncio.sleep(SLEEP_BETWEEN_STARTS_S)
                except Exception as e:
                    logger.error(f"Failed to create container in batch {i+1}/{to_create}: {e}")
                    continue

    async def maintain_min_idle(self) -> None:
        async with self._lock:
            idle = sum(1 for r in self._records.values() if r.status == "idle")
            total = len(self._records)
            needed = max(0, POOL_MIN_IDLE - idle)
            can_create = min(max(0, POOL_MAX - total), self._available_capacity())
            to_create = min(needed, can_create, START_BATCH_SIZE)
        
        logger = logging.getLogger(__name__)
        if to_create > 0:
            logger.info(f"maintain_min_idle: idle={idle}, min_idle={POOL_MIN_IDLE}, total={total}, to_create={to_create}")
            
            for i in range(to_create):
                try:
                    await self._create_container()
                    logger.info(f"maintain_min_idle: created container {i+1}/{to_create}")
                    await asyncio.sleep(SLEEP_BETWEEN_STARTS_S)
                except Exception as e:
                    logger.error(f"maintain_min_idle: failed to create container {i+1}/{to_create}: {e}")
                    continue

    async def lease(self, worker_id: str, ttl_s: int | None) -> LeaseResponse:
        logger = logging.getLogger(__name__)
        ttl = ttl_s or LEASE_TTL_DEFAULT_S
        now = time.time()
        # Fast path: atomically pick and reserve an idle env
        async with self._lock:
            for cid, rec in self._records.items():
                if rec.status == "idle":
                    lease_id = str(uuid.uuid4())
                    rec.status = "leased"
                    rec.lease_id = lease_id
                    rec.worker_id = worker_id
                    rec.lease_expiry_ts = now + ttl
                    logger.info(
                        f"lease granted env_id={rec.container_id} name={rec.name} lease_id={lease_id} worker_id={worker_id} ttl_s={ttl}"
                    )
                    return LeaseResponse(env_id=rec.container_id, lease_id=lease_id, ttl_s=ttl)
        # Slow path: try create one if below max, then retry once
        async with self._lock:
            if len(self._records) >= POOL_MAX:
                raise HTTPException(status_code=503, detail="No envs available")
        await self._create_container()
        async with self._lock:
            for cid, rec in self._records.items():
                if rec.status == "idle":
                    lease_id = str(uuid.uuid4())
                    rec.status = "leased"
                    rec.lease_id = lease_id
                    rec.worker_id = worker_id
                    rec.lease_expiry_ts = now + ttl
                    logger.info(
                        f"lease granted-after-create env_id={rec.container_id} name={rec.name} lease_id={lease_id} worker_id={worker_id} ttl_s={ttl}"
                    )
                    return LeaseResponse(env_id=rec.container_id, lease_id=lease_id, ttl_s=ttl)
        raise HTTPException(status_code=503, detail="No envs available after create")

    async def renew(self, lease_id: str, ttl_s: int | None) -> None:
        logger = logging.getLogger(__name__)
        ttl = ttl_s or LEASE_TTL_DEFAULT_S
        now = time.time()
        async with self._lock:
            rec = _find_by_lease(self._records, lease_id)
            if rec is None:
                logger.warning(f"renew failed lease_id={lease_id} not_found=true")
                raise HTTPException(status_code=404, detail="lease not found")
            logger.info(
                f"renew lease env_id={rec.container_id} name={rec.name} lease_id={lease_id} ttl_s={ttl}"
            )
            rec.lease_expiry_ts = now + ttl

    async def release(self, lease_id: str) -> None:
        logger = logging.getLogger(__name__)
        async with self._lock:
            rec = _find_by_lease(self._records, lease_id)
            if rec is None:
                raise HTTPException(status_code=404, detail="lease not found")
            # Mark idle first to free for others quickly
            rec.status = "idle"
            rec.lease_id = None
            rec.worker_id = None
            rec.lease_expiry_ts = None
            host_port = rec.host_port
            network_ip = rec.network_ip
        logger.info(
            f"release env_id={rec.container_id} name={rec.name} lease_id={lease_id} host_port={host_port} network_ip={network_ip}"
        )
        if IDLE_RESET_ON_RELEASE:
            # Soft reset the env
            try:
                if network_ip:
                    await self._client_session.post(f"http://{network_ip}:5000/reset")
                else:
                    await self._client_session.post(f"http://{self._hostname}:{host_port}/reset")
            except Exception:
                # If reset fails, restart the container
                logger.warning(
                    f"release-reset-failed restarting env_id={rec.container_id} name={rec.name} lease_id={lease_id}"
                )
                await self._restart_container(rec.container_id)

    async def _create_container(self) -> None:
        # Hard cap: never create beyond POOL_MAX (guard against concurrent creators)
        async with self._lock:
            current_total = len(self._records)
            if current_total >= POOL_MAX:
                logging.getLogger(__name__).info(
                    f"_create_container skipped: total={current_total} >= POOL_MAX={POOL_MAX}"
                )
                return
        # Expose container port 5000 to a chosen host port in configured range
        # Try allocating a host port from the fixed range; retry on collision
        attempt_ports: list[int] = []
        # Pre-compute used ports from Docker to avoid known conflicts
        used_docker_ports = self._docker_used_ports()
        for port in range(PORT_RANGE_START, PORT_RANGE_END + 1):
            if port in used_docker_ports or port in self._allocated_ports:
                continue
            attempt_ports.append(port)
        if not attempt_ports:
            raise HTTPException(status_code=503, detail="No free host ports in range")
        
        logger = logging.getLogger(__name__)
        logger.info(f"Creating container with {len(attempt_ports)} available ports")
        
        env = {
            "ANDROID_ENV_AUTOLOAD": "1",
            "ANDROID_ENV_MODE": AUTOLOAD_MODE,
            "ANDROID_ENV_TASK_PATH": DEFAULT_TASK_PATH,
            "ADB_PORT": str(ADB_PORT),
            "EMULATOR_CONSOLE_PORT": str(EMULATOR_CONSOLE_PORT),
            "GRPC_PORT": str(GRPC_PORT),
            # keep logs concise
            "VERBOSE_LOGS": os.environ.get("VERBOSE_LOGS", "false"),
        }
        if EMULATOR_TIMEOUT:
            env["EMULATOR_TIMEOUT"] = EMULATOR_TIMEOUT
        host_config = {
            "nano_cpus": CONTAINER_NANO_CPUS,
            "mem_limit": CONTAINER_MEM_LIMIT,
            "privileged": USE_PRIVILEGED,
            "network": DOCKER_NETWORK,
            "device_requests": None,
        }

        devices = []
        if ENABLE_KVM:
            devices.append("/dev/kvm:/dev/kvm")

        # Volumes
        volumes = {SDK_VOLUME: {"bind": "/opt/android", "mode": "rw"}}

        last_exc: Exception | None = None
        container = None
        for host_port in attempt_ports:
            # Generate a fresh unique name each attempt to avoid leftover-conflict from prior failed create
            name = f"android-env-{uuid.uuid4().hex[:8]}"
            # Reserve with contention handling; if already reserved by another concurrent creator, try next
            try:
                await self._reserve_port(host_port)
            except HTTPException:
                continue
            cpuset: list[int] | None = None
            if CPUSET_ENABLE and CPUSET_CORES_PER_ENV > 0:
                cpuset = await self._allocate_cpuset()
            try:
                logger.info(f"Attempting to create container {name} on port {host_port}")
                container = self._docker.containers.run(
                    IMAGE_NAME,
                    name=name,
                    detach=True,
                    environment=env,
                    network=DOCKER_NETWORK,
                    ports={"5000/tcp": host_port},  # fixed host port
                    nano_cpus=host_config["nano_cpus"],
                    mem_limit=host_config["mem_limit"],
                    privileged=host_config["privileged"],
                    devices=devices or None,
                    volumes=volumes,
                    cpuset_cpus=_format_cpuset(cpuset) if cpuset else None,
                    stdout=True,
                    stderr=True,
                )
                logger.info(f"Successfully created container {name} (ID: {container.id}) on port {host_port}")
                break
            except Exception as exc:  # port collision or other
                last_exc = exc
                logger.warning(f"Failed to create container {name} on port {host_port}: {exc}")
                # Clean up any partially created container with this name
                try:
                    self._remove_container_by_name(name)
                except Exception:
                    pass
                await self._release_port(host_port)
                if cpuset:
                    await self._release_cpuset(cpuset)
                # If it's a port already allocated error, try next port
                continue
        if container is None:
            # Exhausted all candidate ports
            raise HTTPException(status_code=503, detail=f"Failed to create container due to port allocation errors: {last_exc}")

        container.reload()
        # Determine container IP on the shared network for intra-network health checks
        net_info = (container.attrs or {}).get("NetworkSettings", {}).get("Networks", {})
        net_ip = None
        try:
            if DOCKER_NETWORK in net_info:
                net_ip = net_info[DOCKER_NETWORK].get("IPAddress")
            if not net_ip:
                # Fallback to top-level IPAddress
                net_ip = (container.attrs or {}).get("NetworkSettings", {}).get("IPAddress")
        except Exception:
            net_ip = None
        
        logger.info(f"Container {container.id} network IP: {net_ip}")
        
        rec = ContainerRecord(
            container_id=container.id,
            name=name,
            host_port=host_port,
            network_ip=net_ip,
            status="starting",
            created_at=time.time(),
            last_health_at=0.0,
            cpuset_cores=cpuset,
        )
        async with self._lock:
            self._records[container.id] = rec
        
        logger.info(f"Container {container.id} registered with status 'starting'")

    async def _restart_container(self, container_id: str) -> None:
        try:
            c = self._docker.containers.get(container_id)
            # Free its port before removal
            try:
                await self._release_port(self._records.get(container_id).host_port)  # type: ignore[union-attr]
            except Exception:
                pass
            # Free its cpuset
            try:
                rec = self._records.get(container_id)
                if rec and rec.cpuset_cores:
                    await self._release_cpuset(rec.cpuset_cores)
            except Exception:
                pass
            c.remove(force=True)
        except Exception:
            pass
        async with self._lock:
            self._records.pop(container_id, None)
        await self._create_container()

    async def _health_tick(self) -> None:
        logger = logging.getLogger(__name__)
        now = time.time()
        async with self._lock:
            records = list(self._records.values())
        for rec in records:
            # If container is not running (exited/crashed), immediately recycle and free its port
            try:
                c = self._docker.containers.get(rec.container_id)
                c.reload()
                is_running = getattr(c, "status", "") == "running"
            except Exception:
                is_running = False
            if not is_running:
                await self._restart_container(rec.container_id)
                continue

            # Check lease expiry
            if rec.status == "leased" and rec.lease_expiry_ts and now > rec.lease_expiry_ts:
                async with self._lock:
                    rec.status = "idle"
                    rec.lease_id = None
                    rec.worker_id = None
                    rec.lease_expiry_ts = None
            
            # skip health check for leased envs
            if rec.status == "leased":
                continue

            # Health probe: prefer container IP on shared network; fallback to host-published port
            base_host = rec.network_ip or self._hostname
            probe_port = 5000 if rec.network_ip else rec.host_port
            url = f"http://{base_host}:{probe_port}{HEALTH_PATH}"
            healthy = False
            try:
                r = await self._client_session.get(url)
                if r.status_code == 200:
                    data = r.json()
                    # Single boolean field from server: healthy
                    healthy = bool(data.get("healthy", False))
            except Exception:
                healthy = False

            if healthy:
                rec.last_health_at = now
                if rec.status == "starting":
                    # On first healthy transition, optionally perform a soft reset to prime the env
                    if IDLE_RESET_ON_READY:
                        try:
                            if rec.network_ip:
                                await self._client_session.post(f"http://{rec.network_ip}:5000/reset")
                            else:
                                await self._client_session.post(f"http://{self._hostname}:{rec.host_port}/reset")
                        except Exception:
                            # If reset fails, restart this container and skip marking it idle
                            logger.warning(
                                f"ready-reset-failed restarting env_id={rec.container_id} name={rec.name}"
                            )
                            await self._restart_container(rec.container_id)
                            continue
                    rec.status = "idle"
                    logger.info(f"Container {rec.container_id} ({rec.name}) transitioned from 'starting' to 'idle'")
            else:
                # If stuck in starting for too long or idle but unhealthy, restart
                if rec.status == "starting" and (now - rec.created_at) > START_TIMEOUT_S:
                    logger.warning(f"Container {rec.container_id} ({rec.name}) stuck in 'starting' for {now - rec.created_at:.1f}s, restarting")
                    await self._restart_container(rec.container_id)
                elif rec.status == "idle" and (now - rec.last_health_at) > IDLE_TIMEOUT_S:
                    logger.warning(f"Container {rec.container_id} ({rec.name}) stuck in 'idle' for {now - rec.last_health_at:.1f}s, restarting")
                    await self._restart_container(rec.container_id)

    def _available_capacity(self) -> int:
        # Check if we can create more containers based on pool limits
        # Ports 5000-5100 should be sufficient for 64 containers
        return max(0, POOL_MAX - len(self._records))

    def _docker_used_ports(self) -> set[int]:
        used: set[int] = set()
        try:
            containers = self._docker.containers.list(all=True)
            for c in containers:
                try:
                    ports = (c.attrs or {}).get("NetworkSettings", {}).get("Ports", {})
                    for mappings in ports.values():
                        if not mappings:
                            continue
                        for m in mappings:
                            hp = m.get("HostPort")
                            if hp:
                                p = int(hp)
                                if PORT_RANGE_START <= p <= PORT_RANGE_END:
                                    used.add(p)
                except Exception:
                    continue
        except Exception:
            pass
        return used

    async def _reserve_port(self, port: int) -> None:
        async with self._lock:
            if port in self._allocated_ports:
                raise HTTPException(status_code=503, detail="Port already reserved")
            self._allocated_ports.add(port)

    async def _release_port(self, port: int) -> None:
        async with self._lock:
            self._allocated_ports.discard(port)

    async def _allocate_cpuset(self) -> list[int] | None:
        if not CPUSET_ENABLE or CPUSET_CORES_PER_ENV <= 0:
            return None
        async with self._lock:
            available = [c for c in self._allowed_cores if c not in self._allocated_cores]
            # Find first contiguous block of required size
            block: list[int] = []
            prev = None
            for c in available:
                if prev is None or c == prev + 1:
                    block.append(c)
                else:
                    block = [c]
                prev = c
                if len(block) >= CPUSET_CORES_PER_ENV:
                    for cc in block[:CPUSET_CORES_PER_ENV]:
                        self._allocated_cores.add(cc)
                    return block[:CPUSET_CORES_PER_ENV]
            return None

    async def _release_cpuset(self, cores: list[int]) -> None:
        async with self._lock:
            for c in cores:
                self._allocated_cores.discard(c)

    def _remove_container_by_name(self, name: str) -> None:
        try:
            c = self._docker.containers.get(name)
            c.remove(force=True)
        except Exception:
            pass


def _find_by_lease(records: dict[str, ContainerRecord], lease_id: str) -> ContainerRecord | None:
    for rec in records.values():
        if rec.lease_id == lease_id:
            return rec
    return None


def _extract_host_port(container: docker.models.containers.Container, internal_port: int) -> int:
    container.reload()
    bindings = container.attrs.get("NetworkSettings", {}).get("Ports", {})
    spec = bindings.get(f"{internal_port}/tcp")
    if not spec or not isinstance(spec, list) or not spec:
        raise RuntimeError("Container did not expose host port yet")
    host_port = int(spec[0]["HostPort"])  # type: ignore[index]
    return host_port


def _get_host_ip() -> str:
    # Deprecated in proxy mode; kept for compatibility
    return "127.0.0.1"


def _is_port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("0.0.0.0", port))
            return True
        except OSError:
            return False


def _parse_cpuset(cpuset_str: str) -> list[int]:
    result: list[int] = []
    if not cpuset_str:
        return result
    parts = [p.strip() for p in cpuset_str.split(',') if p.strip()]
    for p in parts:
        if '-' in p:
            a, b = p.split('-', 1)
            try:
                start = int(a)
                end = int(b)
                result.extend(list(range(start, end + 1)))
            except ValueError:
                continue
        else:
            try:
                result.append(int(p))
            except ValueError:
                continue
    return sorted(set(result))


def _format_cpuset(cores: list[int] | None) -> str | None:
    if not cores:
        return None
    cores = sorted(set(cores))
    ranges: list[str] = []
    start = prev = cores[0]
    for c in cores[1:]:
        if c == prev + 1:
            prev = c
            continue
        ranges.append(f"{start}-{prev}" if start != prev else f"{start}")
        start = prev = c
    ranges.append(f"{start}-{prev}" if start != prev else f"{start}")
    return ','.join(ranges)


def _compute_allowed_cores(total: int, start: int, exclude_str: str) -> list[int]:
    all_cores = list(range(start, total))
    exclude = set(_parse_cpuset(exclude_str))
    return [c for c in all_cores if c not in exclude]

app = FastAPI()
STATE = EnvPoolState()


@app.on_event("startup")
async def on_startup() -> None:
    # Warm up to target capacity with retries
    logger = logging.getLogger(__name__)
    logger.info(f"Starting pool manager, target capacity: {POOL_TARGET}")
    
    max_startup_attempts = 3
    for attempt in range(max_startup_attempts):
        try:
            await STATE.ensure_capacity()
            current_total = len(STATE._records)
            logger.info(f"Startup attempt {attempt + 1}: created {current_total} containers")
            
            if current_total >= POOL_TARGET:
                logger.info(f"Successfully reached target capacity: {current_total}/{POOL_TARGET}")
                break
            elif attempt < max_startup_attempts - 1:
                logger.warning(f"Only created {current_total}/{POOL_TARGET} containers, retrying...")
                await asyncio.sleep(5)  # Wait before retry
            else:
                logger.error(f"Failed to reach target capacity after {max_startup_attempts} attempts. Current: {current_total}/{POOL_TARGET}")
        except Exception as e:
            logger.error(f"Startup attempt {attempt + 1} failed: {e}")
            if attempt < max_startup_attempts - 1:
                await asyncio.sleep(5)
            else:
                logger.error("All startup attempts failed, continuing with partial capacity")
    
    # Kick off maintenance loop
    asyncio.create_task(_background_maintainer())


@app.on_event("shutdown")
async def on_shutdown() -> None:
    await STATE.close()


@app.get("/stats")
async def stats() -> dict[str, typing.Any]:
    return await STATE.stats()


@app.post("/lease")
async def lease(req: LeaseRequest, direct: bool = False) -> LeaseResponse:
    # Return env_id/lease info; optionally include direct host/port for clients to connect directly
    resp = await STATE.lease(req.worker_id, req.ttl_s)
    if direct:
        # Find the record to extract host_port
        rec = STATE._records.get(resp.env_id)
        host_value = DIRECT_HOST or STATE._hostname
        port_value = rec.host_port if rec else None
        return LeaseResponse(env_id=resp.env_id, lease_id=resp.lease_id, ttl_s=resp.ttl_s, host=host_value, port=port_value)
    return LeaseResponse(env_id=resp.env_id, lease_id=resp.lease_id, ttl_s=resp.ttl_s)


@app.post("/renew")
async def renew(req: RenewRequest) -> dict[str, str]:
    await STATE.renew(req.lease_id, req.ttl_s)
    return {"status": "ok"}


@app.post("/release")
async def release(req: ReleaseRequest) -> dict[str, str]:
    await STATE.release(req.lease_id)
    return {"status": "ok"}


# -------- Proxy mode: training talks only to manager using env_id --------

def _get_env_base_url(rec: ContainerRecord) -> str:
    if rec.network_ip:
        return f"http://{rec.network_ip}:5000"
    # fallback via host mapping
    return f"http://{STATE._hostname}:{rec.host_port}"


def _require_rec(env_id: str) -> ContainerRecord:
    rec = STATE._records.get(env_id)
    if not rec:
        # allow lookup by container name
        for r in STATE._records.values():
            if r.name == env_id:
                rec = r
                break
    if not rec:
        raise HTTPException(status_code=404, detail="env_id not found")
    return rec  # type: ignore[return-value]


# old lease_minimal removed; /lease now returns minimal info by default


@app.get("/env/{env_id}/health")
async def env_health(env_id: str) -> typing.Any:
    rec = _require_rec(env_id)
    base = _get_env_base_url(rec)
    logger = logging.getLogger(__name__)
    logger.info(f"env.health env_id={rec.container_id} name={rec.name}")
    r = await STATE._client_session.get(f"{base}{HEALTH_PATH}")
    return r.json()


@app.get("/env/{env_id}/specs")
async def env_specs(env_id: str) -> typing.Any:
    rec = _require_rec(env_id)
    base = _get_env_base_url(rec)
    logger = logging.getLogger(__name__)
    logger.info(f"env.specs env_id={rec.container_id} name={rec.name}")
    r = await STATE._client_session.get(f"{base}/specs")
    return r.json()


@app.post("/env/{env_id}/reset")
async def env_reset(env_id: str, include_pixels: bool = False) -> typing.Any:
    rec = _require_rec(env_id)
    base = _get_env_base_url(rec)
    logger = logging.getLogger(__name__)
    logger.info(
        f"env.reset env_id={rec.container_id} name={rec.name} include_pixels={include_pixels}"
    )
    r = await STATE._client_session.post(f"{base}/reset", params={"include_pixels": str(include_pixels).lower()})
    return r.json()


@app.post("/env/{env_id}/step")
async def env_step(env_id: str, action: dict[str, typing.Any], include_pixels: bool = False) -> typing.Any:
    rec = _require_rec(env_id)
    base = _get_env_base_url(rec)
    logger = logging.getLogger(__name__)
    logger.info(
        f"env.step env_id={rec.container_id} name={rec.name} include_pixels={include_pixels}"
    )
    r = await STATE._client_session.post(
        f"{base}/step", params={"include_pixels": str(include_pixels).lower()}, json=action
    )
    return r.json()


@app.get("/env/{env_id}/observation")
async def env_observation(env_id: str, include_pixels: bool = False) -> typing.Any:
    rec = _require_rec(env_id)
    base = _get_env_base_url(rec)
    logger = logging.getLogger(__name__)
    logger.info(
        f"env.observation env_id={rec.container_id} name={rec.name} include_pixels={include_pixels}"
    )
    r = await STATE._client_session.get(f"{base}/observation", params={"include_pixels": str(include_pixels).lower()})
    return r.json()


@app.get("/env/{env_id}/screenshot")
async def env_screenshot(env_id: str):
    rec = _require_rec(env_id)
    base = _get_env_base_url(rec)
    logger = logging.getLogger(__name__)
    logger.info(
        f"env.screenshot env_id={rec.container_id} name={rec.name}"
    )
    r = await STATE._client_session.get(f"{base}/screenshot")
    from fastapi import Response  # local import

    return Response(content=r.content, media_type="image/png")


@app.post("/env/{env_id}/close")
async def env_close(env_id: str) -> dict[str, str]:
    rec = _require_rec(env_id)
    base = _get_env_base_url(rec)
    logger = logging.getLogger(__name__)
    logger.info(
        f"env.close env_id={rec.container_id} name={rec.name}"
    )
    await STATE._client_session.post(f"{base}/close")
    return {"status": "ok"}


async def _background_maintainer() -> None:
    logger = logging.getLogger(__name__)
    logger.info("Background maintainer started")
    while True:
        try:
            await STATE._health_tick()  # noqa: SLF001
            await STATE.maintain_min_idle()
        except Exception as e:
            logger.error(f"Background maintainer error: {e}")
        await asyncio.sleep(HEALTH_INTERVAL_S)


def _run() -> None:
    import uvicorn
    host = os.environ.get("MANAGER_HOST", "0.0.0.0")
    port = int(os.environ.get("MANAGER_PORT", "8080"))
    uvicorn.run("pool_manager.manager:app", host=host, port=port, reload=False, workers=1, timeout_keep_alive=30)


if __name__ == "__main__":
    # Allow Ctrl+C
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    _run()


