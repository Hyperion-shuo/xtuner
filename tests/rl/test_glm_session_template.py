"""GLM assistant boundaries retain real tokens without requiring a rendered EOS."""
import asyncio
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock
import pytest
from xtuner.v1.rl.rollout.session_server import SessionServer, _turn_context
from xtuner.v1.rl.rollout.trace_store import RolloutTraceStore


@pytest.mark.parametrize('omit_eos',[False,True])
def test_multiturn_template_boundary_and_versions(omit_eos):
    store = RolloutTraceStore.__ray_metadata__.modified_class()
    proxy = NS(**{name:NS(remote=AsyncMock(side_effect=getattr(store,name)))
                  for name in ('prepare_turn','reserve_turn','commit_turn')})
    class Tokenizer:
        def apply_chat_template(self,messages,**kwargs):
            return ''.join(m['content']+('!' if m['role']=='assistant' and not omit_eos else '') for m in messages)
        def encode(self,text,**kwargs): return list(text.encode())
    server = SessionServer.__new__(SessionServer)
    server.tokenizer=Tokenizer(); server.stop_word='!';server.store=proxy
    server.template_omits_assistant_eos=omit_eos;server.enable_return_routed_experts=False
    end = '' if omit_eos else '!'
    async def run():
        messages=[{'role':'user','content':'P'}]
        for i,answer in enumerate(('A','B')):
            ticket=NS(request_id=str(i),engine_session_id=i,source_optimizer_step=i,version=i,
                      cancel_reason=None,session_id='s',barrier_epoch=i,phase='generating',trace_committed=False)
            token=_turn_context.set({'ticket':ticket})
            try:
                request={'session_id':'s','messages':messages}
                await server.on_request(request,'openai')
                ids=list((answer+end).encode())
                await server.on_response({'choices':[{'message':{'role':'assistant','content':answer},
                    'output_ids':ids,'output_token_logprobs':[[-.2,t] for t in ids]}]},'openai',request)
                assert ticket.trace_committed
            finally: _turn_context.reset(token)
            messages += [{'role':'assistant','content':answer}]
            if i==0: messages += [{'role':'user','content':'T'}]
        key='PA'+end+'TB'+end
        trace=store.export_training_trace('s',key,'1')
        assert trace['input_ids']==list(key.encode())
        assert {x for x in trace['model_steps'] if x is not None}=={0,1}
    asyncio.run(run())


def test_qwen_boundary_still_requires_eos():
    server=SessionServer.__new__(SessionServer)
    server.stop_word='!';server.template_omits_assistant_eos=False
    server.tokenizer=NS(apply_chat_template=lambda messages,**kwargs: ''.join(m['content'] for m in messages))
    async def run():
        with pytest.raises(ValueError,match='required EOS'):
            await server.on_response({'choices':[{'message':{'role':'assistant','content':'A'},
                'output_ids':[65],'output_token_logprobs':[[-.2,65]]}]},'openai',
                {'session_id':'s','messages':[{'role':'user','content':'P'}]})
    asyncio.run(run())


def test_reasoning_only_null_and_missing_content_have_same_canonical_key():
    from xtuner.v1.rl.rollout.chat_template import canonicalize_messages_for_chat_template
    null_message = [{'role':'assistant','content':None,'reasoning_content':'reason'}]
    missing_message = [{'role':'assistant','reasoning_content':'reason'}]
    assert canonicalize_messages_for_chat_template(null_message) == canonicalize_messages_for_chat_template(missing_message)
    assert null_message[0]['content'] is None and 'content' not in missing_message[0]
