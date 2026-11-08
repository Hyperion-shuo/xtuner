"""Router -> actual SessionServer -> mock engine, with real trace/gate code."""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import patch

import aiohttp
from aiohttp import web

from xtuner.v1.rl.rollout.affinity_router import SessionAffinityRouter
from xtuner.v1.rl.rollout.session_server import SessionServer
from xtuner.v1.rl.rollout.trace_store import RolloutTraceStore


def test_pause_publish_trace_and_cancel(tmp_path, monkeypatch):
    monkeypatch.setenv('WORK_DIR', str(tmp_path))
    store = RolloutTraceStore.__ray_metadata__.modified_class()
    class StoreProxy:
        def __getattr__(self, key):
            async def call(*args, **kwargs):
                return getattr(store, key)(*args, **kwargs)
            return SimpleNamespace(remote=call)
    class Tokenizer:
        eos_token = '!'
        def apply_chat_template(self, messages, **kwargs):
            return ''.join(m.get('content', '') + ('!' if m['role'] == 'assistant' else '') for m in messages)
        def encode(self, text, **kwargs):
            return list(text.encode())
    async def run():
        sessions, engines = [], []
        calls = []
        router = SessionAffinityRouter()
        try:
            for rank in range(2):
                async def generate(request, rank=rank):
                    body = await request.json()
                    calls.append((rank, body['input_ids']))
                    answer = 'A' if body['input_ids'] == [80] else 'B'
                    return web.json_response({'object': 'chat.completion', 'choices': [{
                        'index': 0, 'message': {'role': 'assistant', 'content': answer}, 'finish_reason': 'stop',
                        'output_ids': list((answer+'!').encode()),
                        'output_token_logprobs': [[-.1, t] for t in (answer+'!').encode()]}],
                        'usage': {'prompt_tokens': len(body['input_ids']), 'completion_tokens': 2}})
                app = web.Application();app.router.add_post('/v1/chat/completions', generate)
                runner = web.AppRunner(app);await runner.setup();engines.append(runner)
                await web.TCPSite(runner, '127.0.0.1', 0).start()
                with patch('xtuner.v1.rl.rollout.session_server.AutoTokenizer.from_pretrained', return_value=Tokenizer()), \
                     patch('xtuner.v1.rl.rollout.session_server.get_store', return_value=StoreProxy()):
                    server = SessionServer(f'http://127.0.0.1:{runner.addresses[0][1]}', 'unused', port=0,
                                           turn_level_async=True, turn_route_reservation_bytes=64)
                await server.start();sessions.append(server)
                server.port = server._runner.addresses[0][1]
                await server.turn_gate.prepare_version(version=0, source_optimizer_step=0, epoch=0)
                await server.turn_gate.open_admission(version=0, source_optimizer_step=0, epoch=0)
            router.replace_backends([s.url for s in sessions])
            endpoint = await router.start('127.0.0.1') + '/v1/chat/completions'
            async with aiohttp.ClientSession() as client:
                async def post(sid, messages):
                    async with client.post(endpoint, json={'model': 'mock', 'session_id': sid,
                            'messages': messages, 'return_token_ids': True, 'return_logprob': True,
                            'return_routed_experts': False}) as r:
                        result = await r.json()
                        assert r.status == 200, result
                        return result
                first = [{'role': 'user', 'content': 'P'}]
                await asyncio.gather(post('a', first), post('b', first))
                bindings = dict(router.bindings)
                assert len(set(bindings.values())) == 2
                for s in sessions:
                    await s.turn_gate.close_admission(epoch=1)
                    await s.turn_gate.drain(timeout=2)
                second = first + [{'role': 'assistant', 'content': 'A'}, {'role': 'user', 'content': 'T'}]
                waiting = asyncio.create_task(post('a', second))
                await asyncio.sleep(.1)
                assert not waiting.done() and len(calls) == 2
                assert router.bindings == bindings
                for s in sessions:
                    await s.turn_gate.prepare_version(version=1, source_optimizer_step=1, epoch=1)
                    await s.turn_gate.open_admission(version=1, source_optimizer_step=1, epoch=1)
                result = await asyncio.wait_for(waiting, 3)
                rid = result['id'].removeprefix('xtuner-')
                trace = store.export_training_trace('a', 'PA!TB!', rid)
                assert trace['input_ids'] == list(b'PA!TB!')
                assert trace['model_steps'] == [None, 0, 0, None, 1, 1]
                assert calls[-1][0] == calls[0][0]
                for s in sessions:
                    await s.cancel_turn_session('b', 'test_cancel')
                before = len(calls)
                async with client.post(endpoint, json={'session_id': 'b', 'messages': second}) as response:
                    assert response.status == 409
                assert len(calls) == before
                assert router.bindings == bindings
        finally:
            await router.stop()
            for s in sessions:
                await s.stop()
            for e in engines:
                await e.cleanup()
    asyncio.run(run())
