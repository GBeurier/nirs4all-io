import assert from 'node:assert/strict';
import fs from 'node:fs';
import {createHash} from 'node:crypto';
import {canonicalContentBytes,metadataNumber,Dataset,datasetContentBytes,compatibleSchemas,u07Sources,multimodalRuntimeInput} from '../public-dataset.mjs';
const golden=JSON.parse(fs.readFileSync(new URL('../../../tests/fixtures/public-content-v1.json',import.meta.url)));
const actual=canonicalContentBytes(golden.input);
assert.equal(new TextDecoder().decode(actual),golden.canonical_content_utf8);
assert.equal(createHash('sha256').update(actual).digest('hex'),golden.sha256);
assert.deepEqual(canonicalContentBytes(Object.fromEntries(Object.entries(golden.input).reverse())),actual);
assert.deepEqual(canonicalContentBytes([0,1,1000]),canonicalContentBytes([-0,1.0,1e3]));
for(const bad of ['0x10','1_000','','NaN',true])assert.throws(()=>metadataNumber(bad),/decimal/);
for(const whitespace of [' ','\t','\n','\r','\v','\f'])assert.equal(metadataNumber(`${whitespace}20.5${whitespace}`),20.5);
const runtimeFixture=JSON.parse(fs.readFileSync(new URL('../../../tests/fixtures/runtime-u07-groups.json',import.meta.url)));
const digest=bytes=>createHash('sha256').update(bytes).digest('hex');
assert.deepEqual(multimodalRuntimeInput(runtimeFixture,digest).coordinator_relations.records.map(row=>row.group_id),['1.0','1']);
for(const labels of [[1.0,1.5],[1,2],['1',2.0]]){
  const numeric=structuredClone(runtimeFixture);numeric.dataset.groups={dtype:'object',shape:[2],values:labels};
  assert.deepEqual(new Dataset(numeric).toJSON().dataset.groups.values,labels);
  assert.throws(()=>multimodalRuntimeInput(numeric,digest),/group IDs must be nonempty strings/);
}
if(process.argv[2]){
  const input=JSON.parse(fs.readFileSync(process.argv[2]));
  if(process.argv[3])assert.deepEqual(datasetContentBytes(JSON.parse(fs.readFileSync(process.argv[3]))),datasetContentBytes(input));
  const options={name:input.dataset.name,sampleIds:input.dataset.sample_ids,originIds:input.origin_ids,foldIds:input.fold_ids,partitions:input.dataset.partitions.values,targetNames:input.dataset.target_names,groups:input.dataset.groups?.values,independentUnitIds:input.dataset.independent_unit_ids,repetitionIds:input.dataset.repetition_ids,
    axisUnits:Object.fromEntries(input.dataset.sources.map(s=>[s.name,s.axis_units])),axisCoordinates:Object.fromEntries(input.dataset.sources.map(s=>[s.name,s.axis_coordinates])),featureNames:Object.fromEntries(input.dataset.sources.map(s=>[s.name,s.feature_names]))};
  const ds=Dataset.fromSources(Object.fromEntries(input.dataset.sources.map(s=>[s.name,s.array.values])),options);
  compatibleSchemas(u07Sources(ds).source_schemas,u07Sources(input).source_schemas);
  assert.deepEqual(datasetContentBytes(ds),datasetContentBytes(input));
  const without=structuredClone(input);for(const source of without.dataset.sources)source.axis_units=Object.fromEntries(Object.entries(source.axis_units).filter(([,unit])=>unit!==null));
  compatibleSchemas(u07Sources(without).source_schemas,u07Sources(input).source_schemas);
  assert.deepEqual(datasetContentBytes(without),datasetContentBytes(input));
  for(const mutate of [x=>x.dataset.source_alignment=null,x=>{delete x.dataset.independent_unit_ids;x.dataset.repetition_ids=x.dataset.sample_ids},x=>x.dataset.target_names=['protein'],x=>x.dataset.sources[3].axis_coordinates.column=[1,'category']]){
    const bad=structuredClone(input);mutate(bad);assert.throws(()=>multimodalRuntimeInput(bad,bytes=>createHash('sha256').update(bytes).digest('hex')));
  }
}
console.log('PASS IO shared raw-content bytes, numeric spellings, Unicode order, logical schemas and host-array validation');
