export type NestedArray<T> = T | NestedArray<T>[];
export interface ArrayRecord<T=number|string|boolean|null> {dtype:string;shape:number[];values:NestedArray<T>;}
export interface TensorSourceRecord {name:string;sample_ids:string[];representation_id:string;axes:string[];feature_names:string[]|null;axis_units:Record<string,string|null>;axis_coordinates:Record<string,(number|string)[]>;array:ArrayRecord;presence_mask?:ArrayRecord<boolean>;}
export interface MultimodalDatasetRecord {schema:'nirs4all.multimodal-dataset';schema_version:1;name:string;sample_ids:string[];source_alignment?:'strict'|'left';sources:TensorSourceRecord[];y:ArrayRecord|null;target_names?:string[];target_mask?:ArrayRecord<boolean>|null;task_type?:'regression'|'classification'|null;groups:ArrayRecord|null;partitions:ArrayRecord<string>;independent_unit_ids?:string[];repetition_ids?:string[];}
export interface DatasetRecord {schema:'nirs4all.dataset.v1';schema_version:1;dataset:MultimodalDatasetRecord;origin_ids:string[];fold_ids:(string|null)[];}
export interface SourceSchema {representation_id:string;input_shape:number[];dtype:string;identity:string;}
export interface U07RawSources {sample_ids:string[];source_schemas:Record<string,SourceSchema>;sources:Record<string,{sample_ids:string[];descriptor:SourceSchema;shape:number[];data?:number[];rows?:[number|string,string][]}>;}
export function normalizeDataset(value:DatasetRecord):DatasetRecord;
export interface DatasetOptions {sampleIds?:string[];sample_ids?:string[];y?:number[]|number[][];representations?:Record<string,string>;featureNames?:Record<string,string[]>;axisUnits?:Record<string,Record<string,string|null>>;axisCoordinates?:Record<string,Record<string,(number|string)[]>>;partitions?:('train'|'test'|'predict')[];groups?:(string|number)[];independentUnitIds?:string[];repetitionIds?:string[];name?:string;sourceAlignment?:'strict'|'left';targetNames?:string[];taskType?:'regression'|'classification';originIds?:string[];foldIds?:(string|null)[];}
export class Dataset {constructor(value:DatasetRecord);static fromSources(sources:Record<string,NestedArray<number|string>>,options:DatasetOptions):Dataset;readonly record:DatasetRecord;readonly sampleIds:string[];toJSON():DatasetRecord;toDenseRegression(sourceId:string):{X:number[][];y:number[];sample_ids:string[];partitions:string[];target_names:string[];groups:(string|number)[]|null;origin_ids:string[];fold_ids:(string|null)[];independent_unit_ids:string[]|null;repetition_ids:string[]|null};}
export function dataset(value:DatasetRecord|Dataset|Record<string,NestedArray<number|string>>,options?:DatasetOptions):Dataset;
export function u07Sources(value:DatasetRecord|Dataset):U07RawSources;
export function compatibleSchemas(current:Record<string,SourceSchema>,saved:Record<string,SourceSchema>):Record<string,SourceSchema>;
export function multimodalRuntimeInput(value:DatasetRecord|Dataset,digest:(bytes:Uint8Array)=>string):U07RawSources&{source_ids:string[];coordinator_relations:{records:Record<string,unknown>[]};data_content_fingerprint:string};
export function publicSourceSchema(value:DatasetRecord|Dataset,sourceId:string):Record<string,unknown>;

export function canonicalContentTree(value:unknown):unknown[];
export function canonicalContentBytes(value:unknown):Uint8Array;
export function datasetContentBytes(value:DatasetRecord|Dataset):Uint8Array;
export function canonicalSourceSchema(value:SourceSchema):SourceSchema;
export function metadataNumber(value:number|string):number;
