import asyncio
import pytest
from xtuner.v1.rl.agent_loop.sandbox_agent_loop.sandbox import SandboxPool
from xtuner.v1.rl.agent_loop.sandbox_agent_loop.schemas import SandboxSpec
from xtuner.v1.rl.agent_loop.sandbox_agent_loop.finalization import CleanupError

class Provider:
    def __init__(self):
        self.ids=set();self.events=[];self.fail=False;self.deleted=asyncio.Event();self.gate=None
    async def create(self,image_tag,**kw):
        if image_tag=='verifier':assert 'main' not in self.ids
        self.ids.add(image_tag);self.events.append('create:'+image_tag)
        class Client:
            async def health_check(self):return {'ok':True}
            async def aclose(self):pass
        return Client(),image_tag
    async def delete(self,id):
        if self.fail:raise RuntimeError('delete failed')
        if self.gate:await self.gate.wait()
        self.ids.discard(id);self.events.append('delete:'+id);self.deleted.set()
    async def get(self,id):return {'env':{'status':'running' if id in self.ids else 'deleted'}}

def pool(p):return SandboxPool(provider=p,specs={k:SandboxSpec(image=k) for k in ('main','verifier')},creates_per_sec=None,cleanup_timeout_sec=.5,verify_deletion=True)

@pytest.mark.asyncio
async def test_sequential_release_and_final_accounting():
    p=Provider();s=pool(p);await s.get('main');await s.release('main');await s.get('verifier');await s.release_all()
    assert p.events==['create:main','delete:main','create:verifier','delete:verifier']
    assert {v['env_id'] for v in s.cleanup['final_states']}=={'main','verifier'}
    assert s.cleanup['ok']

@pytest.mark.asyncio
async def test_failed_release_remains_owned_for_final_cleanup():
    p=Provider();s=pool(p);await s.get('main');p.fail=True
    with pytest.raises(CleanupError):await s.release('main')
    assert 'main' in p.ids
    p.fail=False;await s.release_all();assert not p.ids and s.cleanup['ok']

@pytest.mark.asyncio
async def test_cancel_release_finishes_delete_then_propagates():
    p=Provider();s=pool(p);await s.get('main');p.gate=asyncio.Event()
    task=asyncio.create_task(s.release('main'));await asyncio.sleep(.02);task.cancel();p.gate.set()
    with pytest.raises(asyncio.CancelledError):await task
    assert 'main' not in p.ids
    await s.release_all();assert s.cleanup['ok']
