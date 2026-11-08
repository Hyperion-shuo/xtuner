"""Shared history and bounded reclamation contracts, independent of GPU/Ray scheduling."""
from unittest.mock import patch
import pytest
from xtuner.v1.rl.rollout.trace_store import RolloutTraceStore, TokenizedSegment, path_segments


def store():
    return RolloutTraceStore.__ray_metadata__.modified_class()


def commit(s, sid, prompt, answer, request, version=0):
    p = s.prepare_turn(sid, prompt, request)
    def seg(text, generated=False):
        ids = list(text.encode())
        return TokenizedSegment(text=text, token_ids=ids, labels=ids if generated else None,
            model_step=version if generated else None, policy_version=version if generated else None,
            logprobs=[-0.1-version]*len(ids), request_id=request if generated else None)
    s.reserve_turn(request, 1024)
    s.commit_turn(sid, prompt, prompt+answer, p['handle'],
                  seg(prompt[len(p['prefix']):]) if p['prefix'] != prompt else None,
                  seg(answer, True), [None], {'request_id': request, 'version':version,'routing_bytes':0})


def test_shared_history_linear_and_branch_versions():
    s=store(); prompt='P'
    for i in range(100):
        commit(s,'s',prompt,'A',str(i),i%3)
        prompt+='AT'
        assert s.turn_memory()==s.audit_turn_memory()
    assert s.turn_memory()['segments']==200
    assert s.turn_memory()['tokens']==200
    paths=[r['path'] for k,r in s.turn_traces['s'].items() if not k.startswith('key:')]
    assert len({id(seg) for path in paths for seg in path_segments(path)})==200
    commit(s,'s','P','A','repeat',2)
    commit(s,'s',prompt,'B','next',1)
    trace=s.export_training_trace('s',prompt+'B','next')
    assert trace['model_steps'][1]==0
    assert s.export_training_trace('s','PA','repeat')['model_steps'][1]==2
    assert s.export_training_trace('s','PA','0')['model_steps'][1]==0


def test_bad_handle_cross_session_and_late_commit():
    s=store(); p=s.prepare_turn('s','P','r')
    response=TokenizedSegment(text='A',token_ids=[65],labels=[65],model_step=0,request_id='r')
    args=('P','PA',p['handle'],TokenizedSegment(text='P',token_ids=[80]),response,[None],{'request_id':'r','version':0})
    with pytest.raises(ValueError,match='handle'):s.commit_turn('other',*args)
    s.begin_release_sessions(['s'])
    with pytest.raises(ValueError,match='closed'):s.commit_turn('s',*args)
    with pytest.raises(ValueError,match='closed'):s.prepare_turn('s','P','late')
    while s.continue_release_sessions(['s'],1)['pending']:pass
    assert s.turn_memory()==s.audit_turn_memory()
    assert s.turn_memory()['segments']==0
    with pytest.raises(ValueError,match='closed'):s.prepare_turn('s','P','recreate')


def test_release_batches_allow_other_session_progress():
    s=store();prompt='P'
    for i in range(70):
        commit(s,'s',prompt,'A',str(i));prompt+='AT'
    assert s.begin_release_sessions(['s','s','missing'])==['s']
    steps=0
    while True:
        result=s.continue_release_sessions(['s'],4)
        assert result['processed']<=4
        assert s.turn_memory()==s.audit_turn_memory()
        if not result['pending']:break
        s.get_or_create('other')
        steps+=1
    assert steps>10
    assert s.list_sessions()==['other']
    assert s.turn_memory()['requests']==s.turn_memory()['segments']==0
    assert s.begin_release_sessions(['s'])==[]


def test_reservation_replacement_is_atomic_and_counted():
    s=store();s.turn_byte_limit=32
    s.reserve_turn('r',24);s.reserve_turn('r',32)
    assert s.turn_memory()['reserved_bytes']==32
    with pytest.raises(MemoryError):s.reserve_turn('r',33)
    with pytest.raises(ValueError):s.reserve_turn('negative',-1)
    assert s.turn_memory()==s.audit_turn_memory()
    s.release_turn_reservation('r');s.release_turn_reservation('r')
    assert s.turn_memory()['reserved_bytes']==0


def test_256k_endpoint_and_segment_ownership():
    s=store();text='x'*(262144-1)
    commit(s,'long',text,'A','r',1)
    trace=s.export_training_trace('long',text+'A','r')
    assert len(trace['input_ids'])==262144
    assert trace['labels'][-1]==65 and all(v==-100 for v in trace['labels'][:-1])
    assert s.turn_memory()['tokens']==262144
    pending=s.begin_release_sessions(['long'])
    while pending:pending=s.continue_release_sessions(pending,4)['pending']
    assert s.turn_memory()['tokens']==0


def test_prepare_commit_rejects_wrong_delta_and_preserves_reservation():
    s=store();p=s.prepare_turn('s','P','r');s.reserve_turn('r',16)
    response=TokenizedSegment(text='A',token_ids=[65],labels=[65],model_step=0,request_id='r')
    with pytest.raises(ValueError,match='delta'):
        s.commit_turn('s','P','PA',p['handle'],None,response,[None],{'request_id':'r','version':0})
    assert s.turn_memory()['requests']==0
    assert s.turn_memory()['reserved_bytes']==16
    assert s.turn_memory()==s.audit_turn_memory()
    s.release_turn_reservation('r')
    with pytest.raises(ValueError,match='handle'):
        s.commit_turn('s','P','PA',p['handle'],TokenizedSegment(text='P',token_ids=[80]),response,[None],{'request_id':'r','version':0})


def test_cancelled_prepare_does_not_commit_or_lose_reservation():
    s=store();p=s.prepare_turn('s','P','r');s.reserve_turn('r',16)
    s.cancel_turn_session('s','trial_cancelled')
    with pytest.raises(ValueError,match='cancelled'):
        s.prepare_turn('s','P','second')
    response=TokenizedSegment(text='A',token_ids=[65],labels=[65],model_step=0,request_id='r')
    with pytest.raises(ValueError,match='cancelled'):
        s.commit_turn('s','P','PA',p['handle'],TokenizedSegment(text='P',token_ids=[80]),response,[None],{'request_id':'r','version':0})
    s.begin_release_sessions(['s'])
    assert s.turn_memory()['reserved_bytes']==0
    while s.continue_release_sessions(['s'])['pending']:pass
    assert not s.pending_prefixes and not s.session_pending


def test_metrics_are_observational(monkeypatch):
    monkeypatch.setenv('XTUNER_TRACE_METRICS','1')
    s=store();commit(s,'s','P','A','r')
    metrics=s.trace_metrics()
    assert metrics['operations']['prepare_turn']['calls']==1
    assert metrics['operations']['commit_turn']['calls']==1
    assert metrics['logical']==s.audit_turn_memory()


def test_route_accounting_with_incremental_release_and_export_pin():
    s=store();routes=[];prompt='P'
    def add(request, version):
        nonlocal prompt
        p=s.prepare_turn('s',prompt,request)
        delta=TokenizedSegment(text=prompt[len(p['prefix']):],token_ids=list(prompt[len(p['prefix']):].encode()))
        response=TokenizedSegment(text='A',token_ids=[65],labels=[65],model_step=version,policy_version=version,request_id=request)
        route=object();routes.append(route)
        s.reserve_turn(request,100)
        s.commit_turn('s',prompt,prompt+'A',p['handle'],delta,response,[route],{'request_id':request,'version':version,'routing_bytes':100})
        prompt+='AT'
    add('first',0)
    pinned=s.export_training_trace('s','PA','first')
    add('second',1)
    with patch('ray.internal.free') as free:
        s.acknowledge_turn_routes('s',{'observed':['first','second'],'retained':['second']})
        free.assert_not_called()  # exported history remains pinned
        assert s.turn_memory()['routing_bytes']==200
        assert s.turn_memory()==s.audit_turn_memory()
        pending=s.begin_release_sessions(['s'])
        while pending:
            pending=s.continue_release_sessions(pending,1)['pending']
            assert s.turn_memory()==s.audit_turn_memory()
        freed=[ref for call in free.call_args_list for ref in call.args[0]]
        assert len(freed)==2 and set(freed)==set(routes)
    assert pinned['routed_experts']==[routes[0]]
    assert s.turn_memory()['routing_bytes']==0


def test_real_session_hooks_exchange_only_prefix_handles(monkeypatch, tmp_path):
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from xtuner.v1.rl.rollout.session_server import SessionServer, _turn_context
    s=store()
    methods=('prepare_turn','reserve_turn','commit_turn')
    proxy=SimpleNamespace(**{name:SimpleNamespace(remote=AsyncMock(side_effect=getattr(s,name))) for name in methods})
    class Tokenizer:
        def apply_chat_template(self,messages,**kwargs):
            text=''.join(m.get('content','')+('!' if m.get('role')=='assistant' else '') for m in messages)
            return text
        def encode(self,text,**kwargs):return list(text.encode())
    server=SessionServer.__new__(SessionServer);server.tokenizer=Tokenizer();server.stop_word='!';server.store=proxy
    server.enable_return_routed_experts=False
    async def run():
        for i,(prompt,answer) in enumerate([('P','A'),('PA!T','B')]):
            ticket=SimpleNamespace(request_id=str(i),engine_session_id=i,source_optimizer_step=i,version=i,cancel_reason=None,
                session_id='s',barrier_epoch=i,phase='generating',trace_committed=False)
            token=_turn_context.set({'ticket':ticket})
            try:
                messages=([{'role':'user','content':'P'}] if i==0 else [{'role':'user','content':'P'},{'role':'assistant','content':'A'},{'role':'user','content':'T'}])
                request={'session_id':'s','messages':messages}
                body=await server.on_request(request,'openai')
                assert body['input_ids']==list(prompt.encode())
                await server.on_response({'choices':[{'message':{'role':'assistant','content':answer},
                    'output_ids':list((answer+'!').encode()),'output_token_logprobs':[[-.1,t] for t in (answer+'!').encode()]}]},'openai',request)
                assert ticket.trace_committed
                assert isinstance(proxy.commit_turn.remote.call_args.args[3],dict)
            finally:_turn_context.reset(token)
        trace=s.export_training_trace('s','PA!TB!','1')
        assert trace['input_ids']==list(b'PA!TB!')
        assert trace['model_steps']==[None,0,0,None,1,1]
    monkeypatch.setenv('WORK_DIR',str(tmp_path))
    asyncio.run(run())


def test_committed_increment_is_owned_and_not_mutated_by_caller():
    s=store();p=s.prepare_turn('s','P','r')
    delta=TokenizedSegment(text='P',token_ids=[80])
    response=TokenizedSegment(text='A',token_ids=[65],labels=[65],logprobs=[-.5],model_step=1,policy_version=1,request_id='r')
    s.commit_turn('s','P','PA',p['handle'],delta,response,[None],{'request_id':'r','version':1})
    response.token_ids[0]=90;response.logprobs[0]=-9;delta.token_ids[0]=91
    trace=s.export_training_trace('s','PA','r')
    assert trace['input_ids']==[80,65]
    assert trace['logprobs']==[0.,-.5]
    with pytest.raises(ValueError,match='duplicate'):
        s.commit_turn('s','P','PA',p['handle'],delta,response,[None],{'request_id':'r','version':1})
    assert s.turn_memory()==s.audit_turn_memory()


def test_export_rejected_during_partial_cleanup():
    s=store();commit(s,'s','P','A','r')
    s.begin_release_sessions(['s'])
    with pytest.raises(ValueError,match='closed'):
        s.export_training_trace('s','PA','r')


def test_late_reservation_cannot_leak_after_session_close():
    s=store();s.prepare_turn('s','P','r');s.begin_release_sessions(['s'])
    with pytest.raises(ValueError,match='active prepared'):
        s.reserve_turn('r',32,session_id='s')
    assert s.turn_memory()['reserved_bytes']==0
    s.prepare_turn('other','P','r2')
    with pytest.raises(ValueError,match='active prepared'):
        s.reserve_turn('r2',32,session_id='wrong')
    s.reserve_turn('r2',32,session_id='other')
    assert s.turn_memory()['reserved_bytes']==32
