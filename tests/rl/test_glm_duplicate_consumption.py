"""Production manager rejects repeated groups before disposition/consumption."""
import asyncio
import math
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock
import pytest
from xtuner.v1.rl.agent_loop_manager.turn_async import TurnManager

@pytest.mark.parametrize('within_batch',[False,True])
def test_duplicate_group_is_rejected_before_consumption(within_batch):
    async def run():
        m=TurnManager.__new__(TurnManager)
        m.sample_pool_groups=2; m.version=1
        m._publish=AsyncMock();m._drain=AsyncMock(return_value=0);m._collect=AsyncMock()
        m._expired=lambda group:False
        m.task_runners=[NS(agent_loop=NS(set_generation_context=Mock()))]
        m.ready=[[NS(group_id=17)],[NS(group_id=17 if within_batch else 18)]]
        m.raw=[];m.pending={};m.consumed=[] if within_batch else [17]
        original=list(m.consumed)
        m.options=NS(batch_wait=None,batch_order=None)
        m.disposition=Mock();m.deadline=math.inf
        with pytest.raises(RuntimeError,match='selected more than once'):
            await m.produce_batch(2,2,model_step=1)
        assert m.consumed==original
        m.disposition.assert_not_called()
    asyncio.run(run())
