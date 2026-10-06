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
