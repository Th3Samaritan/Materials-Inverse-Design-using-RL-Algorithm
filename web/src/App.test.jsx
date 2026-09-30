import React from 'react';
import { cleanup, fireEvent, render, screen, waitFor, act } from '@testing-library/react';
import { afterEach, beforeEach, expect, test, vi } from 'vitest';
import App, { Results } from './App';
import { api, exportRun } from './client';

vi.mock('./client',()=>({ api:vi.fn(), supabase:null, exportRun:vi.fn() }));
const catalog={auth_required:false,properties:['UTS','YS'],units:{UTS:'MPa',YS:'MPa'},property_ranges:{UTS:[400,2000,1300],YS:[300,1800,1100]},presets:{'AISI 4340':{C:.4},Custom:null},composition_ranges:{C:[.05,.8]},quenches:{0:'Water',1:'Oil'},families:{'Low-alloy steel':['research','Synthetic model'],Elastomers:['planned','No model yet']}};
const emptyReport={results:[],valid_candidates:0,candidate_budget:20,seed:42};
beforeEach(()=>{vi.clearAllMocks(); api.mockImplementation(path=>path==='/api/catalog'?Promise.resolve(catalog):Promise.resolve({report:emptyReport,persistence:{status:'not_requested'}}));});
afterEach(cleanup);
async function ready(){render(<App/>);await screen.findByRole('button',{name:'Find candidate processes'});}

test('submits composition balance and exports completed results',async()=>{
  await ready();fireEvent.click(screen.getByRole('button',{name:'Find candidate processes'}));
  await screen.findByText('No candidates passed screening');
  const request=api.mock.calls.find(([path])=>path==='/api/recommend')[1];
  expect(JSON.parse(request.body).composition.Fe).toBe(99.6);
  fireEvent.click(screen.getByRole('button',{name:'Export research run'}));
  expect(exportRun).toHaveBeenCalledWith(emptyReport);
});

test('changing a target cancels and suppresses an older in-flight result',async()=>{
  let complete;
  api.mockImplementation(path=>path==='/api/catalog'?Promise.resolve(catalog):new Promise(resolve=>{complete=resolve;}));
  await ready();fireEvent.click(screen.getByRole('button',{name:'Find candidate processes'}));
  const request=api.mock.calls.find(([path])=>path==='/api/recommend')[1];
  fireEvent.change(screen.getByLabelText('Tensile strength (MPa)'),{target:{value:'1400'}});
  expect(request.signal.aborted).toBe(true);
  await act(async()=>{complete({report:emptyReport,persistence:{status:'not_requested'}});});
  expect(screen.queryByText('No candidates passed screening')).toBeNull();
  expect(screen.getByRole('button',{name:'Find candidate processes'}).disabled).toBe(false);
});

test('planned families cannot submit predictions',async()=>{
  await ready();fireEvent.change(screen.getByLabelText('Material family'),{target:{value:'Elastomers'}});
  expect(screen.queryByRole('button',{name:'Find candidate processes'})).toBeNull();
  expect(screen.getByText(/Predictions are not yet available/)).toBeTruthy();
});

test('production catalog requires sign-in and never submits anonymously',async()=>{
  api.mockResolvedValue({...catalog,auth_required:true});render(<App/>);
  fireEvent.click(await screen.findByRole('button',{name:'Sign in to run predictions'}));
  await screen.findByRole('region',{name:'Account sign in'});
  expect(api.mock.calls.some(([path])=>path==='/api/recommend')).toBe(false);
});

test('API errors leave a recoverable form',async()=>{
  api.mockImplementation(path=>path==='/api/catalog'?Promise.resolve(catalog):Promise.reject(new Error('The inference worker is busy.')));
  await ready();fireEvent.click(screen.getByRole('button',{name:'Find candidate processes'}));
  await screen.findByText('The inference worker is busy.');
  await waitFor(()=>expect(screen.getByRole('button',{name:'Find candidate processes'}).disabled).toBe(false));
});

test('results display real units and reset selection for a new report',()=>{
  const keys=['UTS','YS','elongation','reduction_of_area','HRC','impact_energy_J'];
  const values=Object.fromEntries(keys.map(p=>[p,10]));
  const candidate={rank:1,quench:'Oil',hold_hours:3,T_aus:870,t_aus:1,T_temper:450,t_temper:2,pred:values,std:values,score:.2,generator:'policy',within_tolerance:true};
  render(<Results report={{...emptyReport,results:[candidate],targets:values,tolerance_fraction:.15}}/>);
  expect(screen.getByText('870 °C')).toBeTruthy();
  expect(screen.getByText('Tensile strength (MPa)')).toBeTruthy();
  expect(screen.getByRole('img',{name:'Tensile strength: 100.0 percent of target'})).toBeTruthy();
});
