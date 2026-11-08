"""Session-affine client routing; large trace payloads never pass a coordinator."""
import hashlib
import os

import ray


@ray.remote(num_cpus=0)
def _collect(method, *values):
    # Top-level ObjectRefs resolve before scheduling this task; no blocked workers.
    if method in {'turn_memory', 'audit_turn_memory'}:
        return {key: sum(value[key] for value in values) for key in values[0]}
    if method == 'trace_metrics':
        return {'shards': values, 'logical': {key: sum(v['logical'][key] for v in values)
                                             for key in values[0]['logical']},
                'rss_bytes': sum(v.get('rss_bytes') or 0 for v in values)}
    if method == 'oldest_session_versions':
        return {key: version for value in values for key, version in value.items()}
    if method == 'continue_release_sessions':
        return {'pending': [sid for value in values for sid in value['pending']],
                'processed': sum(value['processed'] for value in values)}
    if method in {'list_sessions', 'begin_release_sessions', 'release_sessions'}:
        return [sid for value in values for sid in value]
    if method == 'release_all':
        return None
    raise ValueError(f'unsupported trace aggregation: {method}')


class _Method:
    def __init__(self, store, name):
        self.store, self.name = store, name

    def remote(self, *args, **kwargs):
        name, store = self.name, self.store
        if name in store.global_methods:
            refs = [getattr(actor, name).remote(*args, **kwargs) for actor in store.actors]
            return _collect.remote(name, *refs)
        if name in store.group_methods:
            groups = {}
            for sid in dict.fromkeys(args[0]):
                groups.setdefault(store.index(str(sid)), []).append(sid)
            refs = [getattr(store.actors[index], name).remote(ids, *args[1:], **kwargs)
                    for index, ids in groups.items()]
            if len(refs) == 1:
                return refs[0]
            return _collect.remote(name, *refs)
        if name in {'reserve_turn', 'release_turn_reservation'}:
            sid = kwargs.get('session_id')
            if sid is None:
                raise ValueError(f'{name} needs session_id for sharded traces')
        elif name == 'get_objects':
            return store.actors[0].get_objects.remote(*args, **kwargs)
        else:
            if not args:
                raise ValueError(f'{name} needs positional session_id')
            sid = args[0]
        return getattr(store.actors[store.index(str(sid))], name).remote(*args, **kwargs)


class ShardedTraceStore:
    global_methods = {'turn_memory', 'audit_turn_memory', 'trace_metrics', 'list_sessions', 'release_all'}
    group_methods = {'oldest_session_versions', 'begin_release_sessions', 'continue_release_sessions', 'release_sessions'}

    def __init__(self, actors):
        self.actors = actors

    def index(self, session_id):
        return int.from_bytes(hashlib.blake2b(session_id.encode(), digest_size=8).digest(), 'little') % len(self.actors)

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        return _Method(self, name)


def trace_budget():
    total = int(os.environ.get('XTUNER_TRACE_HARD_BYTES', str(128 * 1024**3)))
    if total < shard_count():
        raise ValueError('trace budget must allocate at least one byte per shard')
    return total


def shard_count():
    count = int(os.environ.get('XTUNER_TRACE_SHARDS', '1'))
    if not 1 <= count <= 64:
        raise ValueError('XTUNER_TRACE_SHARDS must be between 1 and 64')
    return count


def acquire_shards(actor_class, name, namespace, create):
    count = shard_count()
    total = trace_budget()
    actors = []
    for index in range(count):
        actor_name = name if index == 0 else f'{name}-shard-{index}'
        try:
            actor = ray.get_actor(actor_name, namespace=namespace)
        except ValueError:
            if not create:
                if actors:
                    raise RuntimeError('trace shard topology is incomplete')
                return None
            quota = total // count + int(index < total % count)
            actor = actor_class.options(name=actor_name, namespace=namespace, get_if_exists=True).remote(
                byte_limit=quota, topology=(count, total))
        actors.append(actor)
    topologies = ray.get([actor.trace_topology.remote() for actor in actors])
    if any(tuple(value) != (count, total) for value in topologies):
        raise RuntimeError('trace shard topology or budget differs from existing run')
    return ShardedTraceStore(actors) if count > 1 else actors[0]
