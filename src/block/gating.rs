fn entropy_unranked<B:Backend>(data:Value<B>,dim:impl TryInto<isize>,temperature:f32)->Value<B>{
	let dim=dim.try_into().ok().unwrap();
	match data.get_rank().max(if dim<0{(-dim) as usize}else{dim as usize+1}){
		1=>data.map(|logits:Tensor<B,1>|entropy(logits,dim,temperature),None),
		2=>data.map(|logits:Tensor<B,2>|entropy(logits,dim,temperature),None),
		3=>data.map(|logits:Tensor<B,3>|entropy(logits,dim,temperature),None),
		4=>data.map(|logits:Tensor<B,4>|entropy(logits,dim,temperature),None),
		5=>data.map(|logits:Tensor<B,5>|entropy(logits,dim,temperature),None),
		6=>data.map(|logits:Tensor<B,6>|entropy(logits,dim,temperature),None),
		7=>data.map(|logits:Tensor<B,7>|entropy(logits,dim,temperature),None),
		8=>data.map(|logits:Tensor<B,8>|entropy(logits,dim,temperature),None),
		_=>panic!("expected rank between 1 and 8")
	}
}
fn do_position_gated_unranked<B:Backend,F:FnOnce(Value<B>)->Value<B>>(data:Value<B>,dim:impl TryInto<isize>,f:F,flags:u64,ratings:Value<B>,threshold:f32)->Value<B>{
	fn do_position_gated_ranked<B:Backend,F:FnOnce(Value<B>)->Value<B>,const N:usize>(mut data:Value<B>,dim:impl TryInto<isize>,f:F,flags:u64,ratings:Value<B>,threshold:f32)->Value<B>{
		let output=do_position_gated(data.get_data(),dim,|x:Tensor<B,N>|{
			data=f(data.clone().map(|_|x,None));
			data.get_data()
		},flags,ratings.get_data(),threshold);

		data=data.map(|_|output,None);
		data
	}
	let dim=dim.try_into().ok().unwrap();
	match data.get_rank().max(ratings.get_rank()).max(if dim<0{(-dim) as usize}else{dim as usize+1}){
		1=>do_position_gated_ranked::<B,F,1>(data,dim,f,flags,ratings,threshold),
		2=>do_position_gated_ranked::<B,F,2>(data,dim,f,flags,ratings,threshold),
		3=>do_position_gated_ranked::<B,F,3>(data,dim,f,flags,ratings,threshold),
		4=>do_position_gated_ranked::<B,F,4>(data,dim,f,flags,ratings,threshold),
		5=>do_position_gated_ranked::<B,F,5>(data,dim,f,flags,ratings,threshold),
		6=>do_position_gated_ranked::<B,F,6>(data,dim,f,flags,ratings,threshold),
		7=>do_position_gated_ranked::<B,F,7>(data,dim,f,flags,ratings,threshold),
		8=>do_position_gated_ranked::<B,F,8>(data,dim,f,flags,ratings,threshold),
		_=>panic!("expected rank between 1 and 8")
	}
}
/// compute the entropy of a logit distribution, in nats
pub fn entropy<B:Backend,const N:usize>(data:Tensor<B,N>,dim:impl TryInto<isize>,temperature:f32)->Tensor<B,N>{
	let dim=dim.try_into().map(|d|if d<0{N-(-d) as usize}else{d as usize}).ok().filter(|&d|d<N).expect("dim must be >=-N and <N");
	let logdist=activation::log_softmax(data/temperature,dim);

	let dist=logdist.clone().exp();
	(-dist*logdist).sum_dim(dim)
}
/// find active areas of each expert where k are active per position in the sequence
pub fn expert_gated_mask<B:Backend,const N:usize>(k:usize,ratings:Tensor<B,N>)->Tensor<B,N,Bool>{
	let dims=ratings.dims();
	let thresholds=ratings.clone().topk(k,N-1).min_dim(N-1).expand(dims);

	ratings.greater_equal(thresholds)
}
/// partition, apply the function, then unpartition afterwards
pub fn do_partitioned<B:Backend,K:BasicOps<B>,L:BasicOps<B>,F:FnOnce(Tensor<B,N,K>)->Tensor<B,N,L>,const N:usize>(data:Tensor<B,N,K>,dim:impl TryInto<isize>,func:F,mask:Tensor<B,N,Bool>)->Tensor<B,N,L>{
	let dim=dim.try_into().ok().unwrap();
	let indices=partition_indices(dim,mask);
	let partitioned=partition(data,dim,indices.clone());

	let partitioned=func(partitioned);
	unpartition(partitioned,dim,indices)
}
/// apply the function with position gating. currently assumes f input and output have the same shape
pub fn do_position_gated<B:Backend,F:FnOnce(Tensor<B,N>)->Tensor<B,N>,const N:usize>(data:Tensor<B,N>,dim:impl TryInto<isize>,f:F,flags:u64,ratings:Tensor<B,N>,threshold:f32)->Tensor<B,N>{
	assert!(data.dims().into_iter().zip(ratings.dims()).all(|(d,r)|d==r||r==1));

	let dim=dim.try_into().map(|d|if d<0{N-(-d) as usize}else{d as usize}).ok().filter(|&d|d<N).expect("dim must be >=-N and <N");
	let ratings=if (flags&SOFTMAX_RATING)==0{ratings}else{activation::softmax(ratings,N-1)};
	let seq=data.dims()[dim];

	let mask=position_gated_mask(ratings.clone(),threshold);
	let req=mask.clone().int().sum_dim(dim).max().into_scalar().elem::<u32>() as usize;
	let zeroed=if (flags&PRESERVE_INACTIVE)==0{data.zeros_like()}else{data.clone()};

	if req==0{return zeroed}
	let data=if (flags&SCALE_INPUT)==0{data}else if (flags&OFFSET_SCALE)==0{activation::relu(ratings.clone())*data}else{activation::relu(ratings.clone()-threshold)*data};

	let output=if req<seq{
		let indices=partition_indices(dim,mask);
		let input=partition(data,dim,indices.clone()).narrow(dim,0,req);
		let zeros=zeroed.narrow(dim,req,seq-req);

		unpartition(Tensor::cat(vec![f(input),zeros],dim),dim,indices)
	}else{
		f(data)
	};
	if (flags&SCALE_OUTPUT)==0{output}else if (flags&OFFSET_SCALE)==0{activation::relu(ratings)*output}else{activation::relu(ratings-threshold)*output}
}
/// find active areas with a rating threshold
pub fn position_gated_mask<B:Backend,const N:usize>(ratings:Tensor<B,N>,threshold:f32)->Tensor<B,N,Bool>{ratings.greater_equal_elem(threshold)}
/// compute indices to stably partition the values based of the gating mask. true elements will go to the beginning, false elements will go to the end
pub fn partition_indices<B:Backend,const N:usize>(dim:impl TryInto<isize>,mask:Tensor<B,N,Bool>)->Tensor<B,N,Int>{
	let dim=dim.try_into().map(|d|if d<0{N-(-d) as usize}else{d as usize}).ok().filter(|&d|d<N).expect("dim must be >=-N and <N");
	let mfalse=mask.clone().bool_not().int();
	let mtrue=mask.int();
									// count how many false and how many true are before each component
	let f=mfalse.clone().cumsum(dim);
	let t=mtrue.cumsum(dim);
									// since false is at the beginning, the new indices for false elements are the number of false elements before them, but for true elements, we'll need to add the number of true elements before them too
	f*mfalse+t
}
/// stably partition the values based of the gating mask. true elements will go to the beginning, false elements will go to the end
pub fn partition<B:Backend,K:BasicOps<B>,const N:usize>(data:Tensor<B,N,K>,dim:impl TryInto<isize>,indices:Tensor<B,N,Int>)->Tensor<B,N,K>{
	let dim=dim.try_into().map(|d|if d<0{N-(-d) as usize}else{d as usize}).ok().filter(|&d|d<N).expect("dim must be >=-N and <N");
	let indices=indices.expand(data.dims());

	data.zeros_like().scatter(dim,indices,data,IndexingUpdateOp::Add)
}
/// invert the partition operation
pub fn unpartition<B:Backend,K:BasicOps<B>,const N:usize>(data:Tensor<B,N,K>,dim:impl TryInto<isize>,indices:Tensor<B,N,Int>)->Tensor<B,N,K>{
	let dim=dim.try_into().map(|d|if d<0{N-(-d) as usize}else{d as usize}).ok().filter(|&d|d<N).expect("dim must be >=-N and <N");
	let indices=indices.expand(data.dims());

	data.gather(dim,indices)
}

impl<A:AutodiffBackend<InnerBackend=B>,B:Backend,V:AutodiffModule<A,InnerModule=W>+BlockVariant<A>,W:BlockVariant<B>> AutodiffModule<A> for Entropy<V>{
	fn from_inner(inner:Self::InnerModule)->Self{
		Entropy{inner:V::from_inner(inner.inner),scale:inner.scale,temperature:inner.temperature,vocabdim:inner.vocabdim}
	}
	fn valid(&self)->Self::InnerModule{
		Entropy{inner:self.inner.valid(),scale:self.scale,temperature:self.temperature,vocabdim:self.vocabdim}
	}
	type InnerModule=Entropy<W>;
}
impl<A:AutodiffBackend<InnerBackend=B>,B:Backend,V:AutodiffModule<A,InnerModule=W>+BlockVariant<A>,W:BlockVariant<B>> AutodiffModule<A> for PositionGated<V>{
	fn from_inner(inner:Self::InnerModule)->Self{
		PositionGated{flags:inner.flags,gate:V::from_inner(inner.gate),inner:V::from_inner(inner.inner),seqdim:inner.seqdim,threshold:inner.threshold}
	}
	fn valid(&self)->Self::InnerModule{
		PositionGated{flags:self.flags,gate:self.gate.valid(),inner:self.inner.valid(),seqdim:self.seqdim,threshold:self.threshold}
	}
	type InnerModule=PositionGated<W>;
}
impl<B:Backend,V:BlockVariant<B>> BlockVariant<B> for Entropy<V>{
	fn encoding_hint(&self)->Option<u64>{self.inner.encoding_hint()}
	fn forward(&self,input:Value<B>)->Value<B>{
		let logits=self.inner.forward(input);
		entropy_unranked(logits,self.vocabdim,self.temperature).map(|x:Tensor<B,2>|if self.scale.is_nan(){(x.dims()[1] as f32).ln().recip()*x}else{x*self.scale},None)
	}
	fn forward_mut(&mut self,input:Value<B>)->Value<B>{
		let logits=self.inner.forward_mut(input);
		entropy_unranked(logits,self.vocabdim,self.temperature).map(|x:Tensor<B,2>|if self.scale.is_nan(){(x.dims()[1] as f32).ln().recip()*x}else{x*self.scale},None)
	}
	fn supports(&self,encoding:u64)->bool{self.inner.supports(encoding)}
	type BlockWith<C:Backend>=Entropy<V::BlockWith<C>>;
}
impl<B:Backend,V:BlockVariant<B>> BlockVariant<B> for PositionGated<V>{// TODO I think semi efficient embed is possible
	fn clear(&mut self){
		self.gate.clear();
		self.inner.clear();
	}
	fn detach_cache(&mut self){
		self.gate.detach_cache();
		self.inner.detach_cache();
	}
	fn encoding_hint(&self)->Option<u64>{self.inner.encoding_hint().filter(|&e|self.inner.supports(e))}
	fn forward(&self,input:Value<B>)->Value<B>{
		let ratings=self.gate.forward(input.clone());
		do_position_gated_unranked(input,self.seqdim,|input|self.inner.forward(input),self.flags,ratings,self.threshold)
	}
	fn forward_mut(&mut self,input:Value<B>)->Value<B>{
		let ratings=self.gate.forward_mut(input.clone());
		do_position_gated_unranked(input,self.seqdim,|input|self.inner.forward_mut(input),self.flags,ratings,self.threshold)
	}
	fn supports(&self,encoding:u64)->bool{self.gate.supports(encoding)&&self.inner.supports(encoding)}
	type BlockWith<C:Backend>=PositionGated<V::BlockWith<C>>;
}
impl<B:Backend,V:BlockVariant<B>> Module<B> for Entropy<V>{
	fn collect_devices(&self,devices:Vec<B::Device>)->Vec<B::Device>{self.inner.collect_devices(devices)}
	fn fork(self,device:&B::Device)->Self{
		Entropy{inner:self.inner.fork(device),scale:self.scale,temperature:self.temperature,vocabdim:self.vocabdim}
	}
	fn into_record(self)->Self::Record{
		(self.inner,self.scale,self.temperature,self.vocabdim).into_record()
	}
	fn load_record(mut self,record:Self::Record)->Self{
		(self.inner,self.scale,self.temperature,self.vocabdim)=(self.inner,self.scale,self.temperature,self.vocabdim).load_record(record);
		self
	}
	fn map<M:ModuleMapper<B>>(self,mapper:&mut M)->Self{
		Entropy{inner:self.inner.map(mapper),scale:self.scale,temperature:self.temperature,vocabdim:self.vocabdim}
	}
	fn to_device(self,device:&B::Device)->Self{
		Entropy{inner:self.inner.to_device(device),scale:self.scale,temperature:self.temperature,vocabdim:self.vocabdim}
	}
	fn visit<M:ModuleVisitor<B>>(&self,visitor:&mut M){self.inner.visit(visitor)}
	type Record=<(V,f32,f32,i32) as Module<B>>::Record;
}
impl<B:Backend,V:BlockVariant<B>> Module<B> for PositionGated<V>{
	fn collect_devices(&self,devices:Vec<B::Device>)->Vec<B::Device>{self.gate.collect_devices(self.inner.collect_devices(devices))}
	fn fork(self,device:&B::Device)->Self{
		PositionGated{flags:self.flags,gate:self.gate.fork(device),inner:self.inner.fork(device),seqdim:self.seqdim,threshold:self.threshold}
	}
	fn into_record(self)->Self::Record{
		(self.flags,self.gate,self.inner,self.seqdim,self.threshold).into_record()
	}
	fn load_record(mut self,record:Self::Record)->Self{
		(self.flags,self.gate,self.inner,self.seqdim,self.threshold)=(self.flags,self.gate,self.inner,self.seqdim,self.threshold).load_record(record);
		self
	}
	fn map<M:ModuleMapper<B>>(self,mapper:&mut M)->Self{
		PositionGated{flags:self.flags,gate:self.gate.map(mapper),inner:self.inner.map(mapper),seqdim:self.seqdim,threshold:self.threshold}
	}
	fn to_device(self,device:&B::Device)->Self{
		PositionGated{flags:self.flags,gate:self.gate.to_device(device),inner:self.inner.to_device(device),seqdim:self.seqdim,threshold:self.threshold}
	}
	fn visit<M:ModuleVisitor<B>>(&self,visitor:&mut M){
		self.gate.visit(visitor);
		self.inner.visit(visitor);
	}
	type Record=<(u64,V,V,i32,f32) as Module<B>>::Record;
}
impl<V:ModuleDisplay> ModuleDisplay for Entropy<V>{}
impl<V:ModuleDisplay> ModuleDisplay for PositionGated<V>{}
impl<V:ModuleDisplay> ModuleDisplayDefault for Entropy<V>{
	fn content(&self,content:Content)->Option<Content>{
		self.inner.content(content)
	}
}
impl<V:ModuleDisplay> ModuleDisplayDefault for PositionGated<V>{
	fn content(&self,content:Content)->Option<Content>{
		self.inner.content(content)
	}
}
impl<V> Entropy<V>{
	/// create a new entropy layer
	pub fn new(inner:V,scale:f32,temperature:f32,vocabdim:impl TryInto<i32>)->Self{
		let vocabdim=vocabdim.try_into().ok().unwrap();
		Self{inner,scale,temperature,vocabdim}
	}
}
impl<V> From<V> for Entropy<V>{
	fn from(inner:V)->Self{Self::new(inner,f32::NAN,1.0,-1)}
}
impl<V> PositionGated<V>{
	/// create a new position gated layer
	pub fn new(flags:impl Into<Option<u64>>,gate:V,inner:V,seqdim:impl TryInto<i32>,threshold:f32)->Self{
		let flags=flags.into().unwrap_or(SCALE_OUTPUT);
		let seqdim=seqdim.try_into().ok().unwrap();

		Self{flags,gate,inner,seqdim,threshold}
	}
}

pub const OFFSET_SCALE:u64=16;
pub const PRESERVE_INACTIVE:u64=8;
pub const SCALE_INPUT:u64=1;
pub const SCALE_OUTPUT:u64=2;
pub const SOFTMAX_RATING:u64=4;

#[derive(Clone,Debug,Deserialize,Serialize)]
/// compute the entropy of a softmax of a function of the input. use vocabdim=-1 to be directly compatible as the gate for positiongated
pub struct Entropy<V>{inner:V,scale:f32,temperature:f32,vocabdim:i32}
#[derive(Clone,Debug,Deserialize,Serialize)]
/// positionwise scale the output of the top k layers by a rating. currently assumes f input and output have the same shape
pub struct ExpertGated<V>{experts:Vec<V>,gate:V,flags:u64,k:usize,reduction:Reduction}
#[derive(Clone,Debug,Deserialize,Serialize)]
/// positionwise scale the output of a layer by a rating, and decide when it's active with a rating threshold. currently assumes input and output have the same shape
pub struct PositionGated<V>{flags:u64,gate:V,inner:V,seqdim:i32,threshold:f32}

pub use crate::block::multi::Reduction;
use burn::{
	module::{AutodiffModule,Content,ModuleDisplay,ModuleDisplayDefault,ModuleMapper,ModuleVisitor},prelude::*,tensor::{BasicOps,IndexingUpdateOp,activation,backend::AutodiffBackend}
};
use serde::{Deserialize,Serialize};
use std::fmt::Debug;
use super::{BlockVariant,Value};
