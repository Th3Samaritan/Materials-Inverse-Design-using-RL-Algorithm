"""HTTP integration tests with actual checkpoints and mocked Supabase transport."""
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch
import httpx
import torch
from fastapi.testclient import TestClient
from api.main import create_app
from api.settings import Settings
from mast.catalog import STEEL_PRESETS, PROP_RANGES
from mast.engine import load_models


def payload():
    composition = dict(STEEL_PRESETS['AISI 4340'])
    composition['Fe'] = 100 - sum(composition.values())
    return {'composition': composition, 'targets': {p:r[2] for p,r in PROP_RANGES.items()}, 'n_candidates':20}


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        root = Path(__file__).resolve().parents[1]
        cls.models = load_models(root/'checkpoints/forward_model.pt', root/'checkpoints/policy.pt')

    def setUp(self):
        self.app = create_app(Settings(environment='development', anonymous=True, origins=('http://localhost:5173',)), self.models)
        self.client = TestClient(self.app).__enter__()

    def tearDown(self):
        self.client.__exit__(None,None,None)

    def test_real_inference_and_forward_prediction(self):
        response = self.client.post('/api/recommend',json=payload())
        self.assertEqual(response.status_code,200,response.text)
        report = response.json()['report']
        self.assertTrue(report['results'])
        self.assertEqual(report['evidence_status'],'synthetic_research_only')
        self.assertEqual(len(report['model_hashes']['forward_sha256']),64)
        r = report['results'][0]
        quench = {'Water':0,'Oil':1,'Air':2,'Polymer':3}[r['quench']]
        process = {key:r[key] for key in ('T_aus','t_aus','T_temper','t_temper')}
        process['quench'] = quench
        forward = self.client.post('/api/predict',json={'composition':payload()['composition'],'process':process})
        self.assertEqual(forward.status_code,200,forward.text)
        for key,value in r['pred'].items():
            self.assertAlmostEqual(forward.json()['pred'][key],value,delta=.02)

    def test_invalid_inputs_and_bounded_requests(self):
        for change in ({'n_candidates':1001},{'n_candidates':20.5},{'user_id':'other-user'}, {'constraints':{'section_diameter_mm':50}}, {'constraints':{'allowed_quenches':[]}}):
            self.assertEqual(self.client.post('/api/recommend',json={**payload(),**change}).status_code,422)
        body=payload();body['composition']['Fe']=0
        self.assertEqual(self.client.post('/api/recommend',json=body).status_code,422)
        body=payload();body['targets']['YS']=1900
        self.assertEqual(self.client.post('/api/recommend',json=body).status_code,422)
        self.assertEqual(self.client.post('/api/recommend',content=b'x'*32769).status_code,413)

    def test_cors_health_and_checkpoint_not_served(self):
        self.assertTrue(self.client.get('/health').json()['model_ready'])
        self.assertEqual(self.client.get('/checkpoints/forward_model.pt').status_code,404)
        for origin,status in [('http://localhost:5173',200),('https://untrusted.example',400)]:
            res=self.client.options('/api/recommend',headers={'Origin':origin,'Access-Control-Request-Method':'POST'})
            self.assertEqual(res.status_code,status)

    def test_worker_busy_and_per_user_rate_limit(self):
        self.app.state.slots.acquire()
        try:
            self.assertEqual(self.client.post('/api/recommend',json=payload()).status_code,429)
        finally:
            self.app.state.slots.release()
        import time
        self.app.state.rates['local-development']=(time.monotonic(),20)
        self.assertEqual(self.client.post('/api/recommend',json=payload()).status_code,429)

    def test_failed_save_preserves_prediction(self):
        res=self.client.post('/api/recommend',json={**payload(),'save':True})
        self.assertEqual(res.status_code,200)
        self.assertTrue(res.json()['report']['results'])
        self.assertEqual(res.json()['persistence']['status'],'failed')

    def test_production_auth_and_user_token_storage(self):
        app=create_app(Settings(supabase_url='https://test.supabase.co',supabase_key='public-test-key'),self.models)
        seen=[]
        def handler(request):
            seen.append(request)
            if request.url.path=='/auth/v1/user':
                if request.headers['authorization']!='Bearer good-token':
                    return httpx.Response(401,json={'error':'invalid'})
                return httpx.Response(200,json={'id':'00000000-0000-4000-8000-000000000001'})
            if request.method=='POST':
                return httpx.Response(201,json=[{'id':'saved-id'}])
            return httpx.Response(200,json=[])
        with TestClient(app) as client:
            original=app.state.http
            app.state.http=httpx.AsyncClient(transport=httpx.MockTransport(handler))
            try:
                self.assertTrue(client.get('/api/catalog').json()['auth_required'])
                self.assertEqual(client.post('/api/recommend',json=payload()).status_code,401)
                self.assertEqual(client.post('/api/recommend',headers={'Authorization':'Bearer forged-token'},json=payload()).status_code,401)
                res=client.post('/api/recommend',headers={'Authorization':'Bearer good-token'},json={**payload(),'save':True})
                self.assertEqual(res.status_code,200,res.text)
                self.assertEqual(res.json()['persistence'],{'status':'saved','id':'saved-id'})
                insert=next(r for r in seen if r.method=='POST')
                self.assertEqual(json.loads(insert.content)['user_id'],'00000000-0000-4000-8000-000000000001')
                self.assertEqual(insert.headers['authorization'],'Bearer good-token')
                self.assertEqual(client.get('/api/runs',headers={'Authorization':'Bearer good-token'}).status_code,200)
                self.assertEqual(seen[-1].headers['authorization'],'Bearer good-token')
                self.assertEqual(seen[-1].url.params['limit'],'20')
            finally:
                client.portal.call(app.state.http.aclose)
                app.state.http=original

    def test_configuration_fails_closed(self):
        for env in ({},{'ALLOW_ANONYMOUS':'true'},{'APP_ENV':'development','ALLOW_ANONYMOUS':'true','RENDER':'true'}, {'APP_ENV':'development','ALLOW_ANONYMOUS':'true','CORS_ORIGINS':'*'}):
            with patch.dict(os.environ,env,clear=True), self.assertRaises(RuntimeError):
                Settings.from_env()


if __name__=='__main__': unittest.main()
