// SPDX-License-Identifier: CeCILL-2.1 OR AGPL-3.0-or-later
// IO owns alignment; host wrappers only marshal this record to Methods.
const AXES = {signal_1d:['sample','wavelength'],tabular_numeric:['sample','feature'],tabular_mixed:['sample','column'],sample_metadata:['sample','field'],gray_image:['sample','height','width'],rgb_image:['sample','height','width','channel'],mc_image:['sample','height','width','channel'],multispectral_image:['sample','height','width','band'],series_mv:['sample','time','variable']};
const TYPES = {signal_1d:['dense_signal','nirs'],tabular_numeric:['table','tabular'],tabular_mixed:['table','tabular'],sample_metadata:['metadata','metadata'],gray_image:['gray_image','image'],rgb_image:['image_rgb','image'],mc_image:['multichannel_image','image'],multispectral_image:['multichannel_image','image'],series_mv:['time_series','time_series']};
const error = message => {throw new TypeError(message);};
const equal = (a,b) => JSON.stringify(stable(a))===JSON.stringify(stable(b));
const stable = x => Array.isArray(x) ? x.map(stable) : x!==null && typeof x==='object' ? Object.fromEntries(Object.keys(x).sort().map(k=>[k,stable(x[k])])) : x;
const codepointOrder=(a,b)=>{const x=Array.from(a,c=>c.codePointAt(0)),y=Array.from(b,c=>c.codePointAt(0));for(let i=0;i<Math.min(x.length,y.length);i++)if(x[i]!==y[i])return x[i]-y[i];return x.length-y.length;};
export function canonicalContentTree(value) {
  if(value===null)return ['null'];
  if(typeof value==='boolean')return ['bool',value];
  if(typeof value==='number'){if(!Number.isFinite(value))error('Content numbers must be finite');const bytes=new ArrayBuffer(8);new DataView(bytes).setFloat64(0,value===0?0:value,false);return ['number',Array.from(new Uint8Array(bytes),v=>v.toString(16).padStart(2,'0')).join('')];}
  if(typeof value==='string'){if(/[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/u.test(value))error('Content strings require valid Unicode');return ['string',value];}
  if(Array.isArray(value))return ['array',value.map(canonicalContentTree)];
  if(value&&typeof value==='object')return ['object',Object.keys(value).sort(codepointOrder).map(key=>{canonicalContentTree(key);return [key,canonicalContentTree(value[key])];})];
  error('Content must be a JSON value');
}
export const canonicalContentBytes=value=>new TextEncoder().encode(JSON.stringify(canonicalContentTree(value)));
export function metadataNumber(value){if(typeof value==='string'){value=value.replace(/^[ \t\n\r\v\f]+|[ \t\n\r\v\f]+$/g,'');if(!/^[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?$/.test(value))error('Metadata numeric column requires finite decimal numbers');}else if(typeof value!=='number')error('Metadata numeric column requires finite decimal numbers');const number=Number(value);if(!Number.isFinite(number))error('Metadata numeric column requires finite decimal numbers');return number;}
export function canonicalSourceSchema(schema){const value=structuredClone(schema),identity=JSON.parse(value.identity);identity.axis_units=Object.fromEntries(Object.entries(identity.axis_units).filter(([,unit])=>unit!==null));if(value.representation_id==='tabular_mixed'){value.dtype='object';identity.dtype='object';}value.identity=JSON.stringify(stable(identity));return value;}
export function datasetContentBytes(input){const record=dataset(input).record,raw=record.dataset,u07=equal(raw.sources.map(source=>source.name),['nir','image','series','metadata']);for(const source of raw.sources)if(source.representation_id==='tabular_mixed'){source.array.dtype='object';if(u07)for(const row of source.array.values)row[0]=metadataNumber(row[0]);}for(const field of ['partitions','groups','y'])if(raw[field]?.dtype.startsWith('<U')||(raw[field]?.dtype==='object'&&raw[field].values.flat(Infinity).every(value=>typeof value==='string')))raw[field].dtype='string';return canonicalContentBytes(record);}
function closed(x, required, optional=[]) {
  if (x===null || Array.isArray(x) || typeof x!=='object' || required.some(k=>!Object.hasOwn(x,k)) || Object.keys(x).some(k=>!required.includes(k)&&!optional.includes(k))) error('Invalid dataset fields');
}
function ids(x, unique=true) { if (!Array.isArray(x)||x.some(s=>typeof s!=='string'||!s.trim())||(unique&&new Set(x).size!==x.length)) error('Nonempty distinct string identities required'); return x; }
function scalar(x,dtype) {
  if(dtype==='object') {if(x!==null&&!['string','boolean','number'].includes(typeof x)||typeof x==='number'&&!Number.isFinite(x)) error('Invalid object cell');if(typeof x==='number'&&Number.isInteger(x)&&!Number.isSafeInteger(x))error('Object numeric integers must be exactly representable in JavaScript'); return;}
  if(dtype==='bool') {if(typeof x!=='boolean') error('Boolean dtype cell required');return;}
  if(/^<U\d+$/.test(dtype)) {if(typeof x!=='string'||[...x].length>Number(dtype.slice(2)))error('String dtype mismatch');return;}
  if(!['float32','float64','int8','int16','int32','int64','uint8','uint16','uint32','uint64'].includes(dtype)||typeof x!=='number'||!Number.isFinite(x)) error('Numeric dtype mismatch');
  if(dtype==='float32'&&!Number.isFinite(Math.fround(x)))error('float32 overflow');
  if(dtype.includes('int')) {const bits=Number(dtype.match(/\d+/)[0]),unsigned=dtype.startsWith('u');if(!Number.isSafeInteger(x)||x<(unsigned?0:-(2**(bits-1)))||x>=(unsigned?2**bits:2**(bits-1)))error('Integer dtype overflow or precision loss');}
}
function nested(x,shape,dtype) {if(!shape.length){scalar(x,dtype);return;}if(!Array.isArray(x)||x.length!==shape[0])error('Array shape differs from nested values');x.forEach(v=>nested(v,shape.slice(1),dtype));}
function array(x) {closed(x,['dtype','shape','values']);if(typeof x.dtype!=='string'||!Array.isArray(x.shape)||!x.shape.length||x.shape.length>8||x.shape.some(n=>!Number.isSafeInteger(n)||n<0)||x.shape.reduce((a,b)=>a*b,1)>16777216)error('Invalid dtype/shape');scalar(x.dtype==='object'?null:x.dtype==='bool'?false:x.dtype.startsWith('<U')?'':0,x.dtype);nested(x.values,x.shape,x.dtype);if(x.dtype==='float32'){const round=v=>Array.isArray(v)?v.map(round):Math.fround(v);x.values=round(x.values);}return x.shape;}
function placeholder(shape,dtype) {return shape.length?Array.from({length:shape[0]},()=>placeholder(shape.slice(1),dtype)):dtype==='object'?null:dtype==='bool'?false:dtype.startsWith('<U')?'':0;}
const allTrue = shape => shape.length?Array.from({length:shape[0]},()=>allTrue(shape.slice(1))):true;
function normalizeRagged(source,samples,alignment) {
  closed(source,['source_kind','name','sample_ids','representation_id','axes','array','offsets','time_coordinates','channel_names','time_unit','presence_mask']);
  if(source.representation_id!=='series_mv'||!equal(source.axes,['sample','time','variable']))error('Invalid ragged representation or axes');
  const sourceIds=ids(source.sample_ids),shape=array(source.array);
  if(shape.length!==2||shape[1]===0||source.array.dtype==='object'||source.array.dtype.startsWith('<U'))error('Ragged packed values require numeric matrix channels');
  if(sourceIds.some(id=>!samples.includes(id))||(alignment==='strict'&&sourceIds.length!==samples.length))error('Ragged source identity alignment mismatch');
  if(!equal(array(source.offsets),[sourceIds.length+1])||source.offsets.dtype!=='int64')error('Ragged offsets must be an int64 sample boundary vector');
  const offsets=source.offsets.values;
  if(offsets[0]!==0||offsets.at(-1)!==shape[0]||offsets.some((value,index)=>value<0||index>0&&value<offsets[index-1]))error('Invalid ragged offsets');
  if(!equal(array(source.presence_mask),[sourceIds.length])||source.presence_mask.dtype!=='bool')error('Invalid ragged presence mask');
  if(source.presence_mask.values.some((present,index)=>present&&offsets[index]===offsets[index+1]))error('Present ragged samples require at least one packed point');
  if(source.channel_names!==null&&ids(source.channel_names).length!==shape[1])error('Ragged channel names mismatch');
  if(source.time_unit!==null&&(typeof source.time_unit!=='string'||!source.time_unit.trim()))error('Invalid ragged time unit');
  const times=source.time_coordinates;
  if(times!==null){if(!equal(array(times),[shape[0]])||times.dtype==='object'||times.dtype==='bool'||times.dtype.startsWith('<U'))error('Numeric ragged times required');for(let row=0;row<sourceIds.length;row++)for(let index=offsets[row]+1;index<offsets[row+1];index++)if(times.values[index]<=times.values[index-1])error('Ragged times must strictly increase within each sample');}
  const lookup=new Map(sourceIds.map((id,index)=>[id,index])),packed=[],alignedOffsets=[0],alignedTimes=[],present=[];
  for(const id of samples){const index=lookup.get(id);if(index!==undefined){for(const point of source.array.values.slice(offsets[index],offsets[index+1]))packed.push(point);if(times)for(const time of times.values.slice(offsets[index],offsets[index+1]))alignedTimes.push(time);present.push(source.presence_mask.values[index]);}else present.push(false);alignedOffsets.push(packed.length);}
  source.array.values=packed;source.array.shape[0]=packed.length;source.offsets.values=alignedOffsets;source.offsets.shape=[samples.length+1];if(times){times.values=alignedTimes;times.shape=[alignedTimes.length];}
  source.sample_ids=[...samples];source.presence_mask={dtype:'bool',shape:[samples.length],values:present};
}
function normalizeMaskedTargets(values,mask,shape,dtype){
  if(!shape.length){if(mask===false){if(values!==null&&typeof values!=='boolean'&&!(typeof values==='number'&&Number.isFinite(values)))error('Masked target storage requires null or finite numeric values');return dtype==='bool'?false:0;}scalar(values,dtype);return values;}
  if(!Array.isArray(values)||values.length!==shape[0]||!Array.isArray(mask)||mask.length!==shape[0])error('Target shape differs from values/mask');
  return values.map((value,index)=>normalizeMaskedTargets(value,mask[index],shape.slice(1),dtype));
}

export function normalizeDataset(input) {
  closed(input,['schema','schema_version','dataset','origin_ids','fold_ids']);
  const v2=input.schema==='nirs4all.dataset.v2'&&input.schema_version===2;
  if(!v2&&(input.schema!=='nirs4all.dataset.v1'||input.schema_version!==1))error('Unsupported public dataset schema');
  // Validate before cloning so structuredClone cannot turn host classes into records.
  const out=structuredClone(input), raw=out.dataset;
  closed(raw,['schema','schema_version','name','sample_ids','sources','y','groups','partitions'],['target_names','target_mask','task_type','source_alignment','independent_unit_ids','repetition_ids']);
  if(raw.schema!=='nirs4all.multimodal-dataset'||raw.schema_version!==1||typeof raw.name!=='string'||!raw.name.trim())error('Invalid multimodal dataset schema/name');
  const samples=ids(raw.sample_ids);if(!Object.hasOwn(raw,'source_alignment'))raw.source_alignment='strict';if(!['strict','left'].includes(raw.source_alignment))error('Invalid source alignment');
  if(!Array.isArray(raw.sources)||!raw.sources.length)error('Named sources required');const names=new Set();
  for(const s of raw.sources) {
    if(s.source_kind==='ragged_series'){if(!v2)error('Ragged sources require public dataset v2');if(typeof s.name!=='string'||!s.name.trim()||names.has(s.name))error('Duplicate or empty source name');names.add(s.name);normalizeRagged(s,samples,raw.source_alignment);continue;}
    closed(s,['name','sample_ids','representation_id','axes','feature_names','axis_units','axis_coordinates','array'],['presence_mask']);
    if(typeof s.name!=='string'||!s.name.trim()||names.has(s.name))error('Duplicate or empty source name');names.add(s.name);
    const axes=AXES[s.representation_id],shape=array(s.array),sourceIds=ids(s.sample_ids);
    if(!axes||!equal(s.axes,axes)||shape.length!==axes.length||shape.slice(1).some(n=>!n)||(s.representation_id==='rgb_image'&&shape.at(-1)!==3))error('Source representation/axes/shape mismatch');
    if(!['tabular_mixed','sample_metadata'].includes(s.representation_id)&&(s.array.dtype==='object'||s.array.dtype.startsWith('<U')))error('Numeric representation needs numeric dtype');
    if(shape[0]!==sourceIds.length||sourceIds.some(id=>!samples.includes(id))||(raw.source_alignment==='strict'&&sourceIds.length!==samples.length))error('Source identity alignment mismatch');
    if(s.feature_names!==null&&(shape.length!==2||ids(s.feature_names).length!==shape[1]))error('Feature names mismatch');
    closed(s.axis_units,[],Object.keys(s.axis_units));closed(s.axis_coordinates,[],Object.keys(s.axis_coordinates));
    for(const [axis,unit] of Object.entries(s.axis_units))if(axis==='sample'||!axes.includes(axis)||(unit!==null&&(typeof unit!=='string'||!unit.trim())))error('Invalid axis unit');
    s.axis_units=Object.fromEntries(Object.entries(s.axis_units).filter(([,unit])=>unit!==null));
    for(const [axis,coords] of Object.entries(s.axis_coordinates)) {
      if(!Array.isArray(coords))error('Invalid axis coordinates');
      if(coords.some(v=>typeof v==='number'&&Number.isInteger(v)&&!Number.isSafeInteger(v)))error('Axis coordinates must be exactly representable in JavaScript');
      if(coords.some(v=>typeof v==='string')&&coords.some(v=>typeof v!=='string'))error('Coordinates cannot mix numeric and string labels');
      const index=axes.indexOf(axis);if(index<1||!Array.isArray(coords)||coords.length!==shape[index]||coords.some(v=>!(typeof v==='string'||typeof v==='number'&&Number.isFinite(v)))||new Set(coords).size!==coords.length)error('Invalid axis coordinates');
      if(['time','wavelength'].includes(axis)){if(coords.some(v=>typeof v!=='number'))error('Numeric time/wavelength coordinates required');const up=coords.every((v,i)=>!i||v>coords[i-1]),down=coords.every((v,i)=>!i||v<coords[i-1]);if(!up&&(axis==='time'||!down))error('Coordinates must be strictly monotonic');}
    }
    let mask=sourceIds.map(()=>true);if(s.presence_mask!==undefined){if(!equal(array(s.presence_mask),[sourceIds.length])||s.presence_mask.dtype!=='bool')error('Invalid presence mask');mask=s.presence_mask.values;}
    const lookup=new Map(sourceIds.map((id,i)=>[id,i]));
    s.array.values=samples.map(id=>lookup.has(id)?s.array.values[lookup.get(id)]:placeholder(shape.slice(1),s.array.dtype));
    s.array.shape[0]=samples.length;s.sample_ids=[...samples];s.presence_mask={dtype:'bool',shape:[samples.length],values:samples.map(id=>lookup.has(id)?mask[lookup.get(id)]:false)};
  }
  if(!equal(array(raw.partitions),[samples.length])||ids(raw.partitions.values,false).some(p=>!['train','test','predict'].includes(p)))error('Invalid partition alignment');
  if(raw.groups!==null&&!equal(array(raw.groups),[samples.length]))error('Invalid group alignment');
  if(raw.y!==null){if(v2&&raw.target_mask!==undefined&&raw.target_mask!==null){if(!equal(array(raw.target_mask),raw.y.shape)||raw.target_mask.dtype!=='bool')error('Invalid target mask');raw.y.values=normalizeMaskedTargets(raw.y.values,raw.target_mask.values,raw.y.shape,raw.y.dtype);}const shape=array(raw.y);if(shape[0]!==samples.length||shape.length>2||(shape.length===2&&shape[1]===0))error('Invalid target alignment');raw.target_mask??={dtype:'bool',shape:[...shape],values:allTrue(shape)};if(!equal(array(raw.target_mask),shape)||raw.target_mask.dtype!=='bool')error('Invalid target mask');}
  else if(raw.target_mask!==undefined&&raw.target_mask!==null)error('Absent targets require absent mask');
  const width=raw.y?.shape[1]??1;raw.target_names??=raw.y===null?[]:width===1?['y']:Array.from({length:width},(_,i)=>`y${i}`);if(ids(raw.target_names).length!==width&&raw.y!==null)error('Target names mismatch');
  raw.target_mask??=null;raw.task_type??=null;if(raw.task_type!==null&&!['regression','classification'].includes(raw.task_type))error('Invalid task type');
  for(const key of ['independent_unit_ids','repetition_ids'])if(raw[key]!==undefined&&ids(raw[key],false).length!==samples.length)error('Experimental unit alignment mismatch');
  if(raw.repetition_ids!==undefined&&raw.independent_unit_ids===undefined)error('repetition_ids require independent_unit_ids');
  if(ids(out.origin_ids,false).length!==samples.length||!Array.isArray(out.fold_ids)||out.fold_ids.length!==samples.length||out.fold_ids.some(f=>f!==null&&(typeof f!=='string'||!f.trim())))error('Invalid origin/fold alignment');
  const membership=new Map();out.origin_ids.forEach((origin,i)=>{const p=raw.partitions.values[i],f=out.fold_ids[i];if(p!=='train'&&f!==null)error('Only training samples may declare folds');const state=JSON.stringify([p,f]);if(membership.has(origin)&&membership.get(origin)!==state)error('An origin cannot cross partitions or folds');membership.set(origin,state);});
  for(const label of ['groups','independent_unit_ids']) {
    const values=label==='groups'?raw.groups?.values:raw[label];if(values){const seen=new Map();values.forEach((unit,i)=>{const key=JSON.stringify(unit),p=JSON.stringify([raw.partitions.values[i],out.fold_ids[i]]);if(seen.has(key)&&seen.get(key)!==p)error('A group/unit cannot cross partitions or folds');seen.set(key,p);});}
  }
  if(raw.independent_unit_ids){const seen=new Set();raw.independent_unit_ids.forEach((unit,i)=>{const pair=JSON.stringify([unit,raw.repetition_ids?.[i]??null]);if(seen.has(pair))error('Repeated units require distinct repetitions');seen.add(pair);});}
  return out;
}
export class Dataset {
  #record;
  constructor(value){this.#record=normalizeDataset(value);}
  get record(){return structuredClone(this.#record);}
  static fromSources(sources,options) {
    const samples=ids(options.sampleIds??options.sample_ids),reprs={spectra:'signal_1d',nir:'signal_1d',image:'rgb_image',series:'series_mv',metadata:'tabular_mixed'};
    function shapeOf(x){if(!Array.isArray(x))return [];const shape=[x.length,...(x.length?shapeOf(x[0]):[])];nested(x,shape,'object');return shape;}
    const recordSources=Object.entries(sources).map(([name,values])=>{
      if(values?.source_kind==='ragged_series')return {...structuredClone(values),name};
      const representation=options.representations?.[name]??reprs[name]??'tabular_numeric',shape=shapeOf(values),axes=AXES[representation];
      if(!axes)error('Explicit supported representation required');
      return {name,sample_ids:[...samples],representation_id:representation,axes,feature_names:options.featureNames?.[name]??null,axis_units:options.axisUnits?.[name]??{},axis_coordinates:options.axisCoordinates?.[name]??{},array:{dtype:representation==='tabular_mixed'||representation==='sample_metadata'?'object':'float64',shape,values}};
    });
    const y=options.y??null,target=y===null?null:{dtype:options.taskType==='classification'?'int64':'float64',shape:shapeOf(y),values:y};
    const partitions=options.partitions??samples.map(()=>y===null?'predict':'train'),groups=options.groups??null;
    const v2=recordSources.some(source=>source.source_kind==='ragged_series')||options.targetMask!==undefined;
    return new Dataset({schema:v2?'nirs4all.dataset.v2':'nirs4all.dataset.v1',schema_version:v2?2:1,dataset:{schema:'nirs4all.multimodal-dataset',schema_version:1,name:options.name??'multimodal',sample_ids:samples,source_alignment:options.sourceAlignment===undefined?'strict':options.sourceAlignment,sources:recordSources,y:target,target_names:options.targetNames??(y===null?[]:target.shape.length===2&&target.shape[1]>1?Array.from({length:target.shape[1]},(_,index)=>`y${index}`):['y']),target_mask:options.targetMask===undefined?null:{dtype:'bool',shape:target?.shape??[],values:options.targetMask},task_type:options.taskType??null,groups:groups===null?null:{dtype:'object',shape:[samples.length],values:groups},partitions:{dtype:`<U${Math.max(1,...partitions.map(x=>x.length))}`,shape:[samples.length],values:partitions},...(options.independentUnitIds?{independent_unit_ids:options.independentUnitIds}:{}),...(options.repetitionIds?{repetition_ids:options.repetitionIds}:{})},origin_ids:options.originIds??[...samples],fold_ids:options.foldIds??samples.map(()=>null)});
  }
  get sampleIds(){return [...this.record.dataset.sample_ids];}
  toJSON(){return structuredClone(this.record);}
  toMatrixRegression(sourceId){return this.#matrixProjection(sourceId,false);}
  toMaskedMatrixRegression(sourceId){return this.#matrixProjection(sourceId,true);}
  #matrixProjection(sourceId,masked){
    const record=this.record,raw=record.dataset,s=raw.sources.find(source=>source.name===sourceId);
    if(!s||s.source_kind==='ragged_series'||s.array.shape.length!==2||s.array.dtype==='object'||s.array.dtype.startsWith('<U')||s.presence_mask.values.some(p=>!p))error('Matrix projection requires a complete numeric source');
    const observed=value=>Array.isArray(value)?value.every(observed):value===true;
    const yValues=raw.y===null?null:masked?normalizeMaskedTargets(raw.y.values,raw.target_mask.values,raw.y.shape,raw.y.dtype):raw.y.values;
    if(raw.y!==null && ((!masked&&!observed(raw.target_mask.values))||!['float32','float64','int8','int16','int32','int64','uint8','uint16','uint32','uint64','bool'].includes(raw.y.dtype)))error('Matrix projection requires observed numeric targets');
    if(raw.task_type==='classification'&&raw.y!==null&&(raw.y.shape.length!==1||raw.y.dtype!=='int64'))error('Matrix classification requires one int64 target vector');
    if(raw.task_type==='classification'&&raw.y!==null&&yValues.some(value=>Math.fround(value)!==value))error('Classification labels must be exactly representable in float32');
    return {X:s.array.values,y:raw.y===null?null:raw.y.shape.length===1?yValues.map(value=>[value]):yValues,...(masked?{target_mask:raw.target_mask?.values??null}:{}),sample_ids:raw.sample_ids,partitions:raw.partitions.values,target_names:raw.target_names,task_type:raw.task_type??'regression',groups:raw.groups?.values??null,origin_ids:record.origin_ids,fold_ids:record.fold_ids,independent_unit_ids:raw.independent_unit_ids??null,repetition_ids:raw.repetition_ids??null};
  }
  toDenseRegression(sourceId){const record=this.record,raw=record.dataset,s=raw.sources.find(s=>s.name===sourceId);if(!s||s.source_kind==='ragged_series'||s.array.shape.length!==2||s.array.dtype==='object'||s.array.dtype.startsWith('<U')||s.presence_mask.values.some(p=>!p)||raw.y===null||raw.y.shape.length!==1||raw.task_type==='classification'||raw.target_mask.values.some(p=>!p))error('Dense regression requires a complete numeric matrix and target');return {X:s.array.values,y:raw.y.values,sample_ids:raw.sample_ids,partitions:raw.partitions.values,target_names:raw.target_names,groups:raw.groups?.values??null,origin_ids:record.origin_ids,fold_ids:record.fold_ids,independent_unit_ids:raw.independent_unit_ids??null,repetition_ids:raw.repetition_ids??null};}
}
export const dataset=(value,options={})=>value instanceof Dataset?value:(options.sampleIds||options.sample_ids)?Dataset.fromSources(value,options):new Dataset(value);
export function u07Sources(input) {
  const raw=dataset(input).record.dataset,order=['nir','image','series','metadata'],representations=['signal_1d','rgb_image','series_mv','tabular_mixed'];
  if(raw.source_alignment!=='strict'||!equal(raw.sources.map(s=>s.name),order))error('U07 requires four ordered strictly aligned sources');
  const schemas={},sources={};raw.sources.forEach((s,i)=>{
    const dtype=s.array.dtype,shape=s.array.shape,axes=s.axes,[typeId,modality]=TYPES[s.representation_id];
    if(s.source_kind==='ragged_series'||s.representation_id!==representations[i]||s.presence_mask.values.some(x=>!x)||(i<3&&!['float32','float64'].includes(dtype)))error('Invalid or missing U07 raw source');
    if(i===3){if(shape.length!==2||shape[1]!==2||!s.feature_names||s.feature_names.length!==2||s.array.values.some(row=>typeof row[1]!=='string'))error('U07 metadata needs named finite numeric and categorical columns');for(const row of s.array.values)metadataNumber(row[0]);}
    const descriptor={source_id:s.name,representation_id:s.representation_id,type_id:typeId,modality,axes,shape:[null,...shape.slice(1)],dtype,feature_names:s.feature_names,axis_units:s.axis_units,axis_coordinates:s.axis_coordinates,native_representation:{id:s.representation_id,type_id:typeId,rank:axes.length,axes:axes.map((a,j)=>({name:a,kind:({column:'feature',field:'feature',variable:'feature',band:'channel'})[a]??a,unit:s.axis_units[a]??null,size:j?shape[j]:null,variable:false})),container:'ndarray',dtype:dtype==='object'||dtype.startsWith('<U')?null:dtype,sparse:false,ragged:false}};
    schemas[s.name]=canonicalSourceSchema({representation_id:s.representation_id,input_shape:shape.slice(1),dtype,identity:JSON.stringify(stable(descriptor))});
    sources[s.name]={sample_ids:raw.sample_ids,descriptor:schemas[s.name],shape,...(i===3?{rows:s.array.values}:{data:s.array.values.flat(Infinity)})};
  });return {sample_ids:raw.sample_ids,source_schemas:schemas,sources};
}
export function compatibleSchemas(current,saved){
  if(!equal(Object.keys(current).sort(),Object.keys(saved).sort()))error('Source schema names differ');
  for(const name of Object.keys(current)){
    const a=canonicalSourceSchema(current[name]),b=canonicalSourceSchema(saved[name]);if(!equal({...a,identity:JSON.parse(a.identity)},{...b,identity:JSON.parse(b.identity)}))error(`Source ${name} schema differs (shape, dtype, axes, coordinates, units or columns)`);
  }
  return structuredClone(saved);
}

function runtimeGroupIds(groups) {
  if(groups===null)return null;
  if(groups.values.some(group=>typeof group!=='string'||!group.trim()))error('Multimodal replay group IDs must be nonempty strings');
  return groups.values;
}
/** IO-owned target-free cohort. DAG owns relation/request fingerprints.
 * Declared replay groups use native nonempty string IDs; generic IO groups may
 * remain numeric. No host-specific numeric group label formatting is applied. */
export function multimodalRuntimeInput(input,digest) {
  const ds=dataset(input),record=ds.record,raw=record.dataset;
  if(raw.y!==null||raw.partitions.values.some(p=>p!=='predict'))error('Raw replay requires a target-free predict cohort');
  if(!equal(raw.target_names,[])&&!equal(raw.target_names,['y']))error("U07 replay requires absent target names or ['y']");
  if(typeof digest!=='function')error('Native SHA-256 function required');
  const groupIds=runtimeGroupIds(raw.groups);
  const records=raw.sample_ids.map((id,i)=>({unit_level:'observation',unit_id:null,observation_id:id,sample_id:id,
    source_id:null,rep_id:raw.repetition_ids?.[i]??null,target_id:'y',group_id:groupIds===null?null:groupIds[i],
    origin_sample_id:null,derived_unit_id:null,component_observation_ids:[],sample_influence_weight:null,quality_flag:null,is_augmented:false,
    metadata:{input_origin_id:record.origin_ids[i],...(raw.independent_unit_ids?{independent_unit_id:raw.independent_unit_ids[i]}:{}),...(raw.repetition_ids?{repetition_id:raw.repetition_ids[i]}:{})}}));
  return {...u07Sources(ds),source_ids:['src0','src1','src2','src3'],coordinator_relations:{records},data_content_fingerprint:digest(datasetContentBytes(ds))};
}

/** Cohort-independent source contract; missing and null units mean unknown. */
export function publicSourceSchema(value,sourceId) {
  const source=dataset(value).record.dataset.sources.find(s=>s.name===sourceId);
  if(!source)error('Unknown selected source');
  if(source.source_kind==='ragged_series')return {name:sourceId,source_kind:'ragged_series',representation_id:source.representation_id,axes:source.axes,shape:[null,null,source.array.shape[1]],dtype:source.array.dtype,channel_names:source.channel_names,time_unit:source.time_unit,time_dtype:source.time_coordinates?.dtype??null};
  return {name:sourceId,representation_id:source.representation_id,axes:source.axes,shape:[null,...source.array.shape.slice(1)],
    dtype:source.array.dtype,feature_names:source.feature_names,axis_units:Object.fromEntries(Object.entries(source.axis_units).filter(([,unit])=>unit!==null)),axis_coordinates:source.axis_coordinates};
}

/** Assemble native Methods projections by sample identity; IO never encodes. */
export function projectedMatrixDataset(input,projections,digest) {
  const record=dataset(input).record,raw=record.dataset,samples=raw.sample_ids;
  if(typeof digest!=='function')error('Native projection assembly requires a content digest');
  if(!Array.isArray(projections)||projections.length!==raw.sources.length)error('Projection inventory must match the ordered native source inventory');
  const rows=samples.map(()=>[]),features=[],contracts=[];
  raw.sources.forEach((source,index)=>{
    const projection=projections[index];closed(projection,['source_id','sample_ids','array','feature_names','presence_encoded']);
    if(projection.source_id!==source.name)error('Projection source order or identity differs from native source inventory');
    if(typeof projection.presence_encoded!=='boolean')error('Projection presence_encoded must be boolean');
    if(!projection.presence_encoded&&source.presence_mask.values.some(present=>!present))error('Missing source rows require explicit native presence encoding');
    const sampleIds=ids(projection.sample_ids),arrayRecord=structuredClone(projection.array),shape=array(arrayRecord);
    if(shape.length!==2||!shape[1]||shape[0]!==samples.length||sampleIds.length!==samples.length||sampleIds.some(id=>!samples.includes(id))||!['float32','float64'].includes(arrayRecord.dtype))error('Native projection requires a complete finite float matrix with exact sample identities');
    const columns=ids(projection.feature_names);if(columns.length!==shape[1])error('Projection feature names differ from matrix width');
    if(features.length+columns.length>16777216||samples.length*(features.length+columns.length)>16777216)error('Projected matrix budget exceeded');
    for(const column of columns)features.push(`${source.name}:${column}`);
    const lookup=new Map(sampleIds.map((id,position)=>[id,position]));samples.forEach((id,position)=>{for(const value of arrayRecord.values[lookup.get(id)])rows[position].push(value);});
    contracts.push({source_id:source.name,source_schema:publicSourceSchema(record,source.name),input_presence_mask:source.presence_mask,presence_encoded:projection.presence_encoded,feature_names:columns,projection_content_fingerprint:digest(canonicalContentBytes(projection))});
  });
  if(new Set(features).size!==features.length||samples.length*features.length>16777216)error('Projected feature inventory repeats names or exceeds matrix budget');
  const provenance={schema:'nirs4all.native-source-projections.v1',sample_ids:samples,source_projections:contracts,input_content_fingerprint:digest(datasetContentBytes(record))};
  record.schema='nirs4all.dataset.v2';record.schema_version=2;raw.source_alignment='strict';raw.sources=[{name:'native_features',sample_ids:samples,representation_id:'tabular_numeric',axes:['sample','feature'],feature_names:features,axis_units:{},axis_coordinates:{},array:{dtype:'float64',shape:[samples.length,features.length],values:rows},presence_mask:{dtype:'bool',shape:[samples.length],values:samples.map(()=>true)}}];
  return {record:normalizeDataset(record),provenance};
}
