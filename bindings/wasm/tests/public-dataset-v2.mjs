import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {Dataset,normalizeDataset,publicSourceSchema} from '../public-dataset.mjs';
const fixture=new URL('../../../tests/fixtures/public-dataset-v2.json',import.meta.url),golden=new URL('../../../tests/fixtures/public-dataset-v2-normalized.json',import.meta.url);
const read=()=>JSON.parse(fs.readFileSync(fixture));
test('v2 ragged IDs, offsets, missing sources and masks match Rust/Python transport',()=>{
 const cohort=new Dataset(read());assert.deepEqual(cohort.record,JSON.parse(fs.readFileSync(golden)));
 assert.throws(()=>cohort.toMatrixRegression('matrix'),/observed/);
 assert.deepEqual(cohort.toMaskedMatrixRegression('matrix').y,[[1,0],[2,4],[3,6],[0,8]]);
 assert.deepEqual(publicSourceSchema(cohort,'series').shape,[null,null,2]);
 for(const kind of ['offsets','times','target','v1']){const wrong=read();if(kind==='offsets')wrong.dataset.sources[1].offsets.values[2]=4;else if(kind==='times')wrong.dataset.sources[1].time_coordinates.values[1]=0;else if(kind==='target')wrong.dataset.target_mask.values[0][1]=true;else{wrong.schema='nirs4all.dataset.v1';wrong.schema_version=1;}assert.throws(()=>normalizeDataset(wrong));}
});
test('matrix multi-y preserves columns and explicit class labels reject float32 loss',()=>{
 const value=JSON.parse(fs.readFileSync(golden));value.dataset.sources.pop();value.dataset.y.values=[[1,10],[2,20],[3,30],[4,40]];value.dataset.target_mask.values=value.dataset.y.values.map(()=>[true,true]);
 const multi=new Dataset(value);assert.deepEqual(multi.toMatrixRegression('matrix').y,value.dataset.y.values);assert.throws(()=>multi.toDenseRegression('matrix'));
 value.dataset.y={dtype:'int64',shape:[4],values:[0,1,0,1]};value.dataset.target_mask={dtype:'bool',shape:[4],values:[true,true,true,true]};value.dataset.target_names=['class'];value.dataset.task_type='classification';assert.equal(new Dataset(value).toMatrixRegression('matrix').task_type,'classification');assert.throws(()=>new Dataset(value).toDenseRegression('matrix'));
 value.dataset.y.values[0]=16777217;assert.throws(()=>new Dataset(value).toMatrixRegression('matrix'),/float32/);
});

test('native projected features preserve ID joins, source provenance and presence policy',async()=>{
 const {projectedMatrixDataset}=await import('../public-dataset.mjs');const {createHash}=await import('node:crypto');
 const digest=bytes=>createHash('sha256').update(bytes).digest('hex');
 const projections=[{source_id:'matrix',sample_ids:['d','c','b','a'],array:{dtype:'float64',shape:[4,1],values:[[4],[3],[2],[1]]},feature_names:['mean'],presence_encoded:false},{source_id:'series',sample_ids:['a','b','c','d'],array:{dtype:'float64',shape:[4,2],values:[[0,0],[0,0],[4,1],[0,0]]},feature_names:['mean','present'],presence_encoded:false}];
 assert.throws(()=>projectedMatrixDataset(read(),projections,digest),/presence/);projections[1].presence_encoded=true;
 const output=projectedMatrixDataset(read(),projections,digest);assert.deepEqual(output.record.dataset.sources[0].array.values,[[1,0,0],[2,0,0],[3,4,1],[4,0,0]]);assert.equal(output.provenance.source_projections[1].source_schema.time_unit,'s');
 if(process.env.NIRS4ALL_IO_PROJECTION_OUTPUT)fs.writeFileSync(process.env.NIRS4ALL_IO_PROJECTION_OUTPUT,JSON.stringify({input:read(),projections,output}));
 const wrong=read();wrong.dataset.y.dtype='float32';wrong.dataset.y.values[0][1]=1e99;assert.equal(new Dataset(wrong).toMaskedMatrixRegression('matrix').y[0][1],0);wrong.dataset.target_mask.values[0][1]=true;assert.throws(()=>new Dataset(wrong));
});

test('present ragged samples require a point and numeric time storage',()=>{
 const wrong=read();wrong.dataset.sources[1].offsets.values=[0,0,3];
 assert.throws(()=>normalizeDataset(wrong),/packed point/);
 const times=read();times.dataset.sources[1].time_coordinates.dtype='object';
 assert.throws(()=>normalizeDataset(times),/Numeric/);
});

test('masked independent classifier columns preserve labels and refuse changed observed cells',()=>{
 const value=read();value.dataset.task_type='classification';value.dataset.target_names=['class_a','class_b'];value.dataset.y.dtype='int64';value.dataset.y.values=[[0,null],[1,3],[0,7],[1e99,3]];
 const cohort=new Dataset(value),projected=cohort.toMaskedMatrixRegression('matrix');
 assert.deepEqual(projected.y,[[0,0],[1,3],[0,7],[0,3]]);assert.deepEqual(projected.target_names,['class_a','class_b']);assert.deepEqual(projected.target_mask,value.dataset.target_mask.values);assert.throws(()=>cohort.toMatrixRegression('matrix'));
 value.dataset.y.values[1][1]=16777217;assert.throws(()=>new Dataset(value).toMaskedMatrixRegression('matrix'),/float32/);value.dataset.y.values[1][1]=3;
 value.dataset.target_names=['class_a','class_a'];assert.throws(()=>new Dataset(value));value.dataset.target_names=['class_a','class_b'];
 value.dataset.target_mask.values[0][1]=true;assert.throws(()=>new Dataset(value));
 value.dataset.y.values=[[0,3],[1,3],[0,7],[1,3]];value.dataset.target_mask.values=value.dataset.y.values.map(()=>[true,true]);
 assert.deepEqual(new Dataset(value).toMaskedMatrixRegression('matrix').y,value.dataset.y.values);assert.throws(()=>new Dataset(value).toMatrixRegression('matrix'),/int64 target vector/);
});
