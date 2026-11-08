"""Real CPU Ray actors: topology, historical trace and bounded owned cleanup."""
import asyncio
from pathlib import Path
import tempfile
import pytest
import ray
from xtuner.v1.rl.rollout import trace_store as module
from xtuner.v1.rl.rollout.trace_store import TokenizedSegment


@pytest.mark.parametrize('count',[1,4])
def test_cpu_actors_history_budget_and_cleanup(monkeypatch,count):
    monkeypatch.setenv('XTUNER_TRACE_SHARDS',str(count))
    monkeypatch.setenv('XTUNER_TRACE_HARD_BYTES','1048576')
    assert not ray.is_initialized()
    module._handle_cache=None
    directory=tempfile.mkdtemp(prefix='turn-cpu-ray-',dir='/tmp')
    ray.init(address='local',num_cpus=2,num_gpus=0,include_dashboard=False,
             object_store_memory=100*1024**2,_temp_dir=directory)
    try:
        store=module.get_store()
        actors=store.actors if count>1 else [store]
        assert sum(ray.get(a.turn_memory.remote())['hard_limit'] for a in actors)==1048576
        def seg(text,version=None):
            return TokenizedSegment(text=text,token_ids=list(text.encode()),
                labels=list(text.encode()) if version is not None else None,
                logprobs=[-.1-version if version is not None else 0]*len(text),
                policy_version=version,model_step=version)
        for sid in ['target','healthy']:
            prompt='P'
            for version in range(3):
                rid=f'{sid}-{version}'
                prepared=ray.get(store.prepare_turn.remote(sid,prompt,rid))
                ray.get(store.reserve_turn.remote(rid,128,session_id=sid))
                response=seg('A',version);response.request_id=rid
                ray.get(store.commit_turn.remote(sid,prompt,prompt+'A',prepared['handle'],
                    seg(prompt[len(prepared['prefix']):]),response,[None],
                    {'request_id':rid,'version':version,'routing_bytes':0}))
                prompt+='AT'
            trace=ray.get(store.export_training_trace.remote(sid,prompt[:-1],f'{sid}-2'))
            assert [v for v in trace['policy_versions'] if v is not None]==[0,1,2]
            assert [p for p,l in zip(trace['logprobs'],trace['labels']) if l!=-100]==[-.1,-1.1,-2.1]
        assert ray.get(store.oldest_session_versions.remote([]))=={}
        assert asyncio.run(module.release_existing_sessions(['target','target']))=={'target'}
        assert ray.get(store.oldest_session_versions.remote(['healthy']))=={'healthy':0}
        assert ray.get(store.turn_memory.remote())['cleanup_sessions']==0
        assert asyncio.run(module.release_existing_sessions(['healthy']))=={'healthy'}
        memory=ray.get(store.turn_memory.remote())
        assert memory['sessions']==memory['requests']==memory['reserved_bytes']==0
        monkeypatch.setenv('XTUNER_TRACE_HARD_BYTES','1048577')
        module._handle_cache=None
        with pytest.raises(RuntimeError,match='topology or budget'):
            module.get_store()
    finally:
        ray.shutdown()
        module._handle_cache=None
