#[derive(Deserialize,Serialize)]
#[serde(bound="")]
/// additional attributes of Value, for serialization
enum ValueAttribute<B:Backend>{Loss(f32),Vararg(Value<B>)}

enumerate_blocks!(@include_builtin Block:);
//generic_wrapper!(unsafe @from-mut Residual<B>);
#[macro_export]
/// generates an enum of blocks and implements BlockVariant and Module traits by delegating each function to the inner variant. Variants must implement BlockVariant, and have a single generic argument representing the backend type. usage: enumerate_blocks!(MyBlock:Variant0,Variant1,Variant2...), or to automatically include all builtin block variants, enumerate_blocks!(@include_builtin MyBlock:Variant0,Variant1,Variant2...);
macro_rules! enumerate_blocks{
	(@include_builtin $name:ident:$($variant:ident),*)=>(
		enumerate_blocks!($name:AdaptBlock,Bias,BranchBlock,Cache,ClearBlock,Conv2D,Dense,Detach,DetachedBlock,Embed,EntropyBlock,Identity,LayerNorm,MaxPool2D,OnlyBlock,PositionGatedBlock,RMSNorm,RegistryBlock,Relu,ResidualBlock,SequentialBlock,SharedBlock,Tanh,UndifferentiatedBlock,UpdateBlock,$($variant,)*);

		pub type AdaptBlock           <B>=RecursiveVariant<Adapt           <$name<B>>>;
		pub type BranchBlock          <B>=RecursiveVariant<Branch          <$name<B>>>;
		pub type ClearBlock           <B>=RecursiveVariant<Clear           <$name<B>>>;
		pub type DetachedBlock        <B>=RecursiveVariant<Detached        <$name<B>>>;
		pub type EntropyBlock         <B>=RecursiveVariant<Entropy         <$name<B>>>;
		pub type OnlyBlock            <B>=RecursiveVariant<Only            <$name<B>>>;
		pub type PositionGatedBlock   <B>=RecursiveVariant<PositionGated   <$name<B>>>;
		pub type RegistryBlock        <B>=RecursiveVariant<Registry        <$name<B>>>;
		pub type ResidualBlock        <B>=RecursiveVariant<Residual        <$name<B>>>;
		pub type SharedBlock          <B>=RecursiveVariant<Shared          <$name<B>>>;
		pub type SequentialBlock      <B>=RecursiveVariant<Sequential      <$name<B>>>;
		pub type UndifferentiatedBlock<B>=RecursiveVariant<Undifferentiated<$name<B>>>;
		pub type UpdateBlock          <B>=RecursiveVariant<Update          <$name<B>>>;
	);
	($name:ident:$($variant:ident,)*)=>(
		$(impl<B:Backend> From<$variant<B>> for $name<B>{
			fn from(value:$variant<B>)->Self{Self::$variant(value)}
		})*
		impl<B:Backend> BlockVariant<B> for $name<B>{
			fn clear(&mut self){
				match self{$(Self::$variant(f)=>BlockVariant::clear(f)),*}
			}
			fn custom_seq_forward<V:BlockVariant<B>>(&self,input:Value<B>,n:&mut usize,seqlo:&[V],seqhi:&[V])->Value<B>{
				match self{$(Self::$variant(f)=>BlockVariant::custom_seq_forward(f,input,n,seqlo,seqhi)),*}
			}
			fn custom_seq_forward_mut<V:BlockVariant<B>>(&mut self,input:Value<B>,n:&mut usize,seqlo:&mut [V],seqhi:&mut [V])->Value<B>{
				match self{$(Self::$variant(f)=>BlockVariant::custom_seq_forward_mut(f,input,n,seqlo,seqhi)),*}
			}
			fn detach_cache(&mut self){
				match self{$(Self::$variant(f)=>BlockVariant::detach_cache(f)),*}
			}
			fn embed(&self,input:Tensor<B,2,Int>,inputclasses:usize,inputencoding:u64)->Value<B>{
				match self{$(Self::$variant(f)=>BlockVariant::embed(f,input,inputclasses,inputencoding)),*}
			}
			fn embed_mut(&mut self,input:Tensor<B,2,Int>,inputclasses:usize,inputencoding:u64)->Value<B>{
				match self{$(Self::$variant(f)=>BlockVariant::embed_mut(f,input,inputclasses,inputencoding)),*}
			}
			fn encoding_hint(&self)->Option<u64>{
				match self{$(Self::$variant(f)=>BlockVariant::encoding_hint(f)),*}
			}
			fn forward(&self,input:Value<B>)->Value<B>{
				match self{$(Self::$variant(f)=>BlockVariant::forward(f,input)),*}
			}
			fn forward_mut(&mut self,input:Value<B>)->Value<B>{
				match self{$(Self::$variant(f)=>BlockVariant::forward_mut(f,input)),*}
			}
			fn get_variant_id(&self)->Option<u64>{
				match self{$(Self::$variant(f)=>BlockVariant::get_variant_id(f)),*}
			}
			fn supports(&self,encoding:u64)->bool{
				match self{$(Self::$variant(f)=>BlockVariant::supports(f,encoding)),*}
			}
			fn tokenizer_hint(&self)->Option<TokenDict>{
				match self{$(Self::$variant(f)=>BlockVariant::tokenizer_hint(f)),*}
			}
			fn to_backend<C:Backend>(self)->Self::BlockWith<C>{
				match self{$(Self::$variant(f)=>$name::$variant(BlockVariant::to_backend(f))),*}
			}
			type BlockWith<C:Backend>=$name<C>;
		}

		#[derive(Debug,Deserialize,Module,Serialize)]
		#[serde(bound="")]
		pub enum $name<B:Backend>{$($variant($variant<B>)),*}
	);
}

impl<'a,B:Backend> Deserialize<'a> for Value<B>{
	fn deserialize<D:Deserializer<'a>>(deserializer:D)->Result<Self,D::Error>{
		let (data,encoding,extraattributes):(Tens<f32>,u64,Vec<ValueAttribute<B>>)=Deserialize::deserialize(deserializer)?;
		let device=Default::default();
		let mut dims=[1;8];
		let mut loss=None;
		let rank=data.rank();
		let mut varargs=Vec::new();

		dims[8-rank..].copy_from_slice(data.dims());
		extraattributes.into_iter().for_each(|attr|match attr{
			ValueAttribute::Loss  (l)=>loss=Some(Tensor::from_data(TensorData::new(vec![l],[1]),&device)),
			ValueAttribute::Vararg(v)=>varargs.push(v)
		});

		let count=data.count();
		let data=Tensor::from_data(TensorData::new(data.into_flat_vec(),[count]),&device);

		Ok(Self{data,dims,encoding,loss,rank,varargs})
	}
}
impl<A:AutodiffBackend<InnerBackend=B>,B:Backend,V:AutodiffModule<A,InnerModule=W>+BlockVariant<A>,W:BlockVariant<B>> AutodiffModule<A> for RecursiveVariant<V>{
	fn from_inner(inner:Self::InnerModule)->Self{V::from_inner(*inner.0).into()}
	fn valid(&self)->Self::InnerModule{self.0.valid().into()}
	type InnerModule=RecursiveVariant<W>;
}
impl<A:AutodiffBackend<InnerBackend=B>,B:Backend> AutodiffModule<A> for Value<A>{
	fn from_inner(inner:Self::InnerModule)->Self{
		Self{
			data:Tensor::from_inner(inner.data),
			dims:inner.dims,
			encoding:inner.encoding,
			loss:inner.loss.map(Tensor::from_inner),
			rank:inner.rank,
			varargs:inner.varargs.into_iter().map(Value::from_inner).collect()
		}
	}
	fn valid(&self)->Self::InnerModule{
		Value{
			data:self.data.valid(),
			dims:self.dims,
			encoding:self.encoding,
			loss:self.loss.as_ref().map(Tensor::valid),
			rank:self.rank,
			varargs:self.varargs.iter().map(Value::valid).collect()
		}
	}
	type InnerModule=Value<B>;
}
impl<B:Backend,V:BlockVariant<B>> BlockVariant<B> for RecursiveVariant<V>{
	fn clear(&mut self){self.0.clear()}
	fn detach_cache(&mut self){self.0.detach_cache()}
	fn embed(&self,input:Tensor<B,2,Int>,inputclasses:usize,inputencoding:u64)->Value<B>{self.0.embed(input,inputclasses,inputencoding)}
	fn embed_mut(&mut self,input:Tensor<B,2,Int>,inputclasses:usize,inputencoding:u64)->Value<B>{self.0.embed_mut(input,inputclasses,inputencoding)}
	fn forward(&self,input:Value<B>)->Value<B>{self.0.forward(input)}
	fn forward_mut(&mut self,input:Value<B>)->Value<B>{self.0.forward_mut(input)}
	fn supports(&self,encoding:u64)->bool{self.0.supports(encoding)}
	type BlockWith<C:Backend>=RecursiveVariant<V::BlockWith<C>>;
}
impl<B:Backend,V:BlockVariant<B>> Module<B> for RecursiveVariant<V>{
	fn collect_devices(&self,devices:Vec<B::Device>)->Vec<B::Device>{self.0.collect_devices(devices)}
	fn fork(self,device:&B::Device)->Self{self.0.fork(device).into()}
	fn into_record(self)->Self::Record{self}
	fn load_record(self,record:Self::Record)->Self{record}
	fn map<M:ModuleMapper<B>>(self,mapper:&mut M)->Self{self.0.map(mapper).into()}
	fn to_device(self,device:&B::Device)->Self{self.0.to_device(device).into()}
	fn visit<M:ModuleVisitor<B>>(&self,visitor:&mut M){self.0.visit(visitor)}
	type Record=Self;
}
impl<B:Backend> Module<B> for Value<B>{
	fn collect_devices(&self,mut devices:Vec<B::Device>)->Vec<B::Device>{
		devices=self.data   .collect_devices(devices);
		devices=self.loss   .collect_devices(devices);
		devices=self.varargs.collect_devices(devices);
		devices
	}
	fn fork(self,device:&B::Device)->Self{
		Self{
			data:self.data.fork(device),
			dims:self.dims,
			encoding:self.encoding,
			loss:self.loss.fork(device),
			rank:self.rank,
			varargs:self.varargs.fork(device)
		}
	}
	fn into_record(self)->Self::Record{self}
	fn load_record(self,record:Self::Record)->Self{record}
	fn map<M:ModuleMapper<B>>(self,mapper:&mut M)->Self{
		Self{
			data:self.data.map(mapper),
			dims:self.dims,
			encoding:self.encoding,
			loss:self.loss.map(|x|x.map(mapper)),
			rank:self.rank,
			varargs:self.varargs.map(mapper)
		}
	}
	fn to_device(self,device:&B::Device)->Self{
		Self{
			data:self.data.to_device(device),
			dims:self.dims,
			encoding:self.encoding,
			loss:self.loss.to_device(device),
			rank:self.rank,
			varargs:self.varargs.to_device(device)
		}
	}
	fn visit<M:ModuleVisitor<B>>(&self,visitor:&mut M){
		self.data   .visit(visitor);
		self.loss   .visit(visitor);
		self.varargs.visit(visitor);
	}
	type Record=Self;
}
impl<B:Backend,V:BlockVariant<B>> Record<B> for RecursiveVariant<V>{
	fn from_item<S:PrecisionSettings>(item:Self::Item<S>,_device:&B::Device)->Self{item}
	fn into_item<S:PrecisionSettings>(self)->Self::Item<S>{self}
	type Item<S:PrecisionSettings>=Self;
}
impl<B:Backend> Record<B> for Value<B>{
	fn from_item<S:PrecisionSettings>(item:Self::Item<S>,_device:&B::Device)->Self{item}
	fn into_item<S:PrecisionSettings>(self)->Self::Item<S>{self}
	type Item<S:PrecisionSettings>=Self;
}
impl<B:Backend> Add<Value<B>> for Value<B>{
	fn add(self,rhs:Value<B>)->Self::Output{
		assert_eq!(self.get_encoding(),rhs.get_encoding());
		fn add_ranked<B:Backend,const N:usize>(mut l:Value<B>,r:Value<B>)->Value<B>{
			if let Some(rl)=r.get_loss(){l.add_loss(rl)}
			l.map(|l:Tensor<B,N>|l+r.get_data(),None)
		}
		match self.get_rank().max(rhs.get_rank()){
			1=>add_ranked::<B,1>(self,rhs),
			2=>add_ranked::<B,2>(self,rhs),
			3=>add_ranked::<B,3>(self,rhs),
			4=>add_ranked::<B,4>(self,rhs),
			5=>add_ranked::<B,5>(self,rhs),
			6=>add_ranked::<B,6>(self,rhs),
			7=>add_ranked::<B,7>(self,rhs),
			8=>add_ranked::<B,8>(self,rhs),
			_=>panic!("expected rank between 1 and 8")
		}
	}
	type Output=Value<B>;
}
impl<B:Backend> BlockVariant<B> for Adjust<B>{
	fn clear(&mut self){self.inner.clear()}
	fn embed(&self,input:Tensor<B,2,Int>,inputclasses:usize,inputencoding:u64)->Value<B>{self.inner.embed(input,inputclasses,inputencoding)}
	fn embed_mut(&mut self,input:Tensor<B,2,Int>,inputclasses:usize,inputencoding:u64)->Value<B>{self.inner.embed_mut(input,inputclasses,inputencoding)}
	fn forward(&self,input:Value<B>)->Value<B>{self.inner.forward(input)}
	fn forward_mut(&mut self,input:Value<B>)->Value<B>{self.inner.forward_mut(input)}
	fn supports(&self,encoding:u64)->bool{self.inner.supports(encoding)}
	type BlockWith<C:Backend>=Adjust<C>;
}
impl<B:Backend> Serialize for Value<B>{
	fn serialize<S:Serializer>(&self,serializer:S)->Result<S::Ok,S::Error>{
		let data:Tens<_>=self.data.clone().try_into().unwrap();
		let encoding=self.encoding;

		let mut data:Tens<f32>=data.view().map(|e|e.elem());
		let mut extraattributes=Vec::new();

		data.reshape(self.dims());
		extraattributes.extend(self.loss.as_ref() .map(|l|ValueAttribute::Loss(l.clone().into_scalar().elem())));
		extraattributes.extend(self.varargs.iter().map(|v|ValueAttribute::Vararg(v.clone())));

		(data,encoding,extraattributes).serialize(serializer)
	}
}
impl<B:Backend> Value<B>{ // TODO needs reshape and reshape map. probably should be refactored into another file at this point
	/// adds to the loss
	pub fn add_loss<const N:usize>(&mut self,loss:Tensor<B,N>){
		let mut loss=loss.sum();
		if let Some(l)=self.loss.take(){loss=l+loss}
		self.loss=Some(loss)
	}
	/// detach from differentiation graph
	pub fn detach(self)->Self{
		Self{
			data:self.data.detach(),
			dims:self.dims,
			encoding:self.encoding,
			loss:self.loss.map(Tensor::detach),
			rank:self.rank,
			varargs:self.varargs.into_iter().map(Self::detach).collect()
		}
	}
	/// references the dimensions
	pub fn dims(&self)->&[usize]{&self.dims[8-self.rank..]}
	/// gets the data. if N is not equal to self.get_rank() the left dimension will be flattened or unsqueezed to match
	pub fn get_data<const N:usize>(&self)->Tensor<B,N>{
		let mut dims=[1;N];

		dims[0]=-1;
		dims[1..].iter_mut().rev().zip(self.dims.iter().rev()).for_each(|(dim,d)|*dim=*d as i64);

		self.data.clone().reshape(dims)
	}
	/// gets the encoding id
	pub fn get_encoding(&self)->u64{self.encoding}
	/// gets auxiliary loss accrued during the computation
	pub fn get_loss(&self)->Option<Tensor<B,1>>{self.loss.clone()}
	/// gets the tensor rank
	pub fn get_rank(&self)->usize{self.rank}
	/// gets and applies reshape in the same operation
	pub fn get_reshaped_data<const N:usize>(&self,dims:[usize;N])->Tensor<B,N>{self.data.clone().reshape(dims)}
	/// maps the data. if N is not equal to self.get_rank() the left dimension will be flattened or unsqueezed to match. The leftmost dimension and rank are expected to be left unchanged
	pub fn map<F:FnOnce(Tensor<B,N>)->Tensor<B,N>,I:Into<Option<u64>>,const N:usize>(mut self,f:F,newencoding:I)->Self{
		let mut dims=[1;N];

		dims[0]=-1;
		dims[1..].iter_mut().rev().zip(self.dims.iter().rev()).for_each(|(dim,d)|*dim=*d as i64);

		let data=self.data.reshape(dims);
		let dim0=data.dims()[0];

		let data=f(data);
		let dims=data.dims();

		assert_eq!(dim0,dims[0]);

		self.data=data.reshape([-1]);
		self.dims[8-N+1..].copy_from_slice(&dims[1..]);
		self.encoding=newencoding.into().unwrap_or(self.encoding);
		self
	}
	/// create a new value from a tensor and a presumably randomly generated encoding.
	pub fn new<const N:usize>(data:Tensor<B,N>,encoding:u64)->Self{
		assert!(N<8);

		let mut dims=[1;8];
		dims[8-N..].copy_from_slice(&data.dims());

		let data=data.reshape([-1]);
		let loss=None;
		let rank=N;
		let varargs=Vec::new();

		Self{data,dims,encoding,loss,rank,varargs}
	}
	/// set the encoding id. changing the encoding id without also mapping the data is discouraged.
	pub fn set_encoding(&mut self,encoding:u64){self.encoding=encoding}
	/// split into chunks of a specified size along a dimension
	pub fn split(self,size:usize,dim:impl TryInto<isize>)->Vec<Self>{
		fn split_ranked<B:Backend,const N:usize>(this:Value<B>,size:usize,dim:usize)->Vec<Value<B>>{
			let data:Tensor<B,N>=this.get_data();
			data.split(size,dim).into_iter().map(|x|this.clone().map(|_|x,None)).collect()
		}

		let dim=dim.try_into().map(|d|if d<0{self.rank-(-d) as usize}else{d as usize}).ok().expect("dim must be >=-N and <N");
		match self.rank.max(dim+1){
			1=>split_ranked::<B,1>(self,size,dim),
			2=>split_ranked::<B,2>(self,size,dim),
			3=>split_ranked::<B,3>(self,size,dim),
			4=>split_ranked::<B,4>(self,size,dim),
			5=>split_ranked::<B,5>(self,size,dim),
			6=>split_ranked::<B,6>(self,size,dim),
			7=>split_ranked::<B,7>(self,size,dim),
			8=>split_ranked::<B,8>(self,size,dim),
			_=>panic!("expected rank between 1 and 8")
		}
	}
	/// the value after an unsuccessful embedding
	pub fn unembedded(input:Tensor<B,2,Int>,inputclasses:usize,inputencoding:u64)->Self{
		let input:Tensor<B,3>=input.one_hot(inputclasses).float();
		Self::new(input,inputencoding)
	}
	/// reference the variable argument list
	pub fn varargs(&self)->&[Self]{&self.varargs}
	/// reference the variable argument list
	pub fn varargs_vec_mut(&mut self)->&mut Vec<Self>{&mut self.varargs}
}
impl<V:ModuleDisplay> ModuleDisplay for RecursiveVariant<V>{}
impl<V:ModuleDisplayDefault> ModuleDisplayDefault for RecursiveVariant<V>{
	fn content(&self,content:Content)->Option<Content>{self.0.content(content)}
}
impl<B:Backend> ModuleDisplay for Value<B>{}
impl<B:Backend> ModuleDisplayDefault for Value<B>{
	fn content(&self,content:Content)->Option<Content>{self.data.content(content)}
}
impl<V> Deref for RecursiveVariant<V>{
	fn deref(&self)->&Self::Target{self.0.deref()}
	type Target=V;
}
impl<V> DerefMut for RecursiveVariant<V>{
	fn deref_mut(&mut self)->&mut Self::Target{self.0.deref_mut()}
}
impl<V> From<V> for RecursiveVariant<V>{
	fn from(inner:V)->Self{Self(Box::new(inner))}
}

/// applies rotary position encoding according to the indices. dims: angles=[features/(2*space)], input=[d.., features], position=[d.., space], output=[d.., features]
pub fn index_rotary<B:Backend,const D:usize>(angles:Tensor<B,1>,input:Tensor<B,D>,position:Tensor<B,D>)->Tensor<B,D>{
	let angledim    =angles  .dims()[0];
	let device=input.device();
	let inputdims   =input   .dims();
	let mut fdims=inputdims;
	let positiondims=position.dims();

	let features=inputdims[D-1];
	let space=positiondims[D-1];
																				// check dimension contract
	assert_eq!(angledim*2*space,features);
	assert_eq!(inputdims[..D-1],positiondims[..D-1]);
																				// find indices for pairs of even odd features to rotate
	let feven:Tensor<B,D,Int>=Tensor::arange_step(0..features as i64,2,&device).unsqueeze();
	let fodd: Tensor<B,D,Int>=Tensor::arange_step(1..features as i64,2,&device).unsqueeze();

	fdims[D-1]=features/2;

	let feven=feven.expand(fdims);
	let fodd =fodd .expand(fdims);
	let mut acc=input.zeros_like();
																				// compute angles and coordinates for the rotation decomposed into simple 2d rotations. a=theta^(-n/f)*t, x=input[even], y=input[odd]
	let angles=(angles.unsqueeze::<2>()*position.reshape([-1,1])).reshape(fdims);
	let cos=angles.clone().cos();
	let sin=angles        .sin();
	let x=input.clone().gather(D-1,feven.clone());
	let y=input.clone().gather(D-1,fodd.clone());
																				// rotate. I would consider this direction backwards, but mathematically is doesn't matter so long as it's consistent. this is the direction i've seen elsewhere so i'll be consistent with that
	let x1=cos.clone()*x.clone()-sin.clone()*y.clone();
	let y1=cos*y+sin*x;
																				// accumulate rotation results. this could also be done with cat and sum_dim; idk which is faster but i've seen the sum_dim version crash a lot due to a bug in a dependency relating to kernel group dimension limits
	acc=acc.scatter(D-1,feven,x1,IndexingUpdateOp::Add);
	acc=acc.scatter(D-1,fodd, y1,IndexingUpdateOp::Add);
	acc
}
/// fills the attention tensor with the value where the query position is less than the key position minus length, or greater than the key position. Assumes attention dimensions are [.., query, key]
pub fn mask_window<B:Backend,const D:usize>(a:Tensor<B,D>,length:usize,value:f64)->Tensor<B,D>{
	if D<2{return mask_window::<B,2>(a.unsqueeze(),length,value).squeeze_dim(0)}						// shouldn't actually happen but if the dimension is less than 2 we can just treat it like it has a second dimension of size 1

	let (device,dims)=(a.device(),a.dims());
	let (key,query)=(dims[D-1],dims[D-2]);
	let extrakeys=key.saturating_sub(query);															// due to caching, there might be more keys than queries

	let causal:Tensor<B,2,Bool>=Tensor::tril_mask([query,key],extrakeys as i64,&device);
	let window:Tensor<B,2,Bool>=Tensor::triu_mask([query,key],extrakeys as i64-length as i64,&device);
	let a=a.mask_fill(causal.unsqueeze(),value).mask_fill(window.unsqueeze(),value);
	a
}
/// applies rotary position encoding according to the offset
/// input: [..., features] output: [..., features]
pub fn offset_rotary<B:Backend,const D:usize>(angles:Tensor<B,1>,input:Tensor<B,D>,offset:isize)->Tensor<B,D>{
	let device=input.device();
	let mut dims=input.dims();

	dims[D-1]=1;

	let mut positions:Tensor<B,D>=Tensor::arange(offset as i64..(dims[D-2] as isize+offset) as i64,&device).float().unsqueeze();
	positions=positions.swap_dims(D-1,D-2);

	positions=positions.expand(dims);
	index_rotary(angles,input,positions)
}
/// generates angles for rotary positional encoding. dims: [features/2]
pub fn rotary_angles<B:Backend>(features:usize,theta:f32)->Tensor<B,1>{
	assert_eq!(features%2,0);

	let device=Default::default();
	let feven:Tensor<B,1,Int>=Tensor::arange_step(0..features as i64,2,&device);

	((-theta.ln()/features as f32)*feven.float()).exp()
}
/// tensor operation to soft choose along the specified dimension. The chosen value on each line will be an integer between 0 and the tensors size along the dimension, chosen from a probability distribution given by the softmax of the data divided by temperature
pub fn soft_choose<B:Backend,const N:usize>(data:Tensor<B,N>,dim:i32,temperature:f32)->Tensor<B,N,Int>{
	let data=data/temperature;							// divide data by temperature, find important info. the following function will be a tensor based implementation of the basic scan-subtract method
	let device=data.device();
	let dim=if dim<0{dim+N as i32}else{dim} as usize;
	let mut dims=data.dims();
	let softmax=activation::softmax(data,dim);
														// replace the dimension of interest with 1 since it's being collapsed by the choice. it's prior value is one more than the theoretical maximum choice by this function
	let max=mem::replace(&mut dims[dim],1)-1;
														// cumulative summation of the softmax gives, between one component and the next, an ordered sequence of ranges a random number from 0 to <1 may fall into. which range will determine the choice
	let acc=Tensor::random(dims,burn::tensor::Distribution::Uniform(0.0,1.0),&device)-softmax.cumsum(dim);
	let choice=acc.ceil().int().sum_dim(dim);			// count the ranges not chosen. random is 0 to =1, cumulative softmax is 0 to =1 and increasing, so acc is -1 to 1 and decreasing
														// after a ceil, this looks like a sequence of 1s followed by a sequence of 0s. the length of the sequence of 1s, equal to its sum, will be the choice
	choice.clamp(0,max as u64)							// max+1 appeared once, presumably due to floating point. clamp to 0<=x<=max to prevent similar incidents in the future
}

/// convenience module for glob importing all builtin layers
pub mod builtin{
	pub use super::{
		gating::{Entropy,PositionGated},modified::{Adapt,Detached,Only,Residual,Undifferentiated},multi::{Branch,Sequential},shared::{Cache,Clear,Registry,Shared,Update},simple::{Bias,Conv2D,Dense,Detach,Embed,Identity,LayerNorm,MaxPool2D,RMSNorm,Relu,Tanh}
	};
}
/// convenience module for glob things required to create a new block enum through the enumerate blocks macro, excluding the builtin module
pub mod enumerate{
	pub use burn::prelude::*;
	pub use serde::{Deserialize,Serialize};
	pub use super::{BlockVariant,RecursiveVariant,Value,enumerate_blocks};
	pub use token_dict::TokenDict;
}
/// gating layers
pub mod gating;
/// layers wrapped in misc modification wrappers, like no grad, or residual
pub mod modified;
/// blocks composed from collections of blocks or layers
pub mod multi;
/// layer/parameter/value sharing
pub mod shared;
/// basic building block typeof layers like linear, tanh
pub mod simple;

#[derive(Debug,Deserialize,Module,Serialize)]
#[serde(bound="")]
/// a simple block that adds a linear adjustment
pub struct Adjust<B:Backend>{inner:Residual<Dense<B>>}
#[derive(Clone,Debug,Deserialize,Serialize)]
#[repr(transparent)]
/// wrapper to protect recursive enum variants from compilation problems
pub struct RecursiveVariant<V>(pub Box<V>);
#[derive(Clone,Debug)]
/// value structure for inside a model, storing tensor data and encoding id, and possibly additional loss information
pub struct Value<B:Backend>{data:Tensor<B,1>,dims:[usize;8],encoding:u64,loss:Option<Tensor<B,1>>,rank:usize,varargs:Vec<Value<B>>}

/// functions required to be a building block of our style of model
pub trait BlockVariant<B:Backend>:Any+DeserializeOwned+Module<B>+Serialize{
	/// adds input and output adapters to the block
	fn adapt<I:IntoIterator<Item=(u64,Self)>,O:IntoIterator<Item=(u64,Self)>>(self,inputadapters:I,outputadapters:O)->Self where RecursiveVariant<Adapt<Self>>:Into<Self>{
		let map=inputadapters.into_iter()
			.chain([(0,self)])
			.chain(outputadapters.into_iter().map(|(encodingid,layer)|(!encodingid,layer)))
			.collect();

		RecursiveVariant::from(Adapt(map)).into()
	}
	/// clears the cache if supported. this has no override parents because the default does nothing
	fn clear(&mut self){}
	/// if placed inside a Sequential block, process the blocks in a special way. seqlo and seqhi are not inclusive of self
	fn custom_seq_forward<V:BlockVariant<B>>(&self,input:Value<B>,n:&mut usize,seqlo:&[V],seqhi:&[V])->Value<B>{
		let _=(seqlo,seqhi);
		*n+=1;

		self.forward(input)
	}
	/// if placed inside a Sequential block, process the blocks in a special way. seqlo and seqhi are not inclusive of self
	fn custom_seq_forward_mut<V:BlockVariant<B>>(&mut self,input:Value<B>,n:&mut usize,seqlo:&mut [V],seqhi:&mut [V])->Value<B>{
		let _=(seqlo,seqhi);
		*n+=1;

		self.forward_mut(input)
	}
	/// detaches the cache from autograd if supported. this has no override parents because the default does nothing
	fn detach_cache(&mut self){}
	/// converts to a detached block
	fn detached(self)->Self where RecursiveVariant<Detached<Self>>:Into<Self>{RecursiveVariant::from(Detached(self)).into()}
	/// use the block as an embedding. returns the one hot of the input if the encoding is unsupported. override parents: forward
	fn embed(&self,input:Tensor<B,2,Int>,inputclasses:usize,inputencoding:u64)->Value<B>{self.forward(Value::unembedded(input,inputclasses,inputencoding))}
	/// use the block as an embedding. returns the one hot of the input if the encoding is unsupported. override parents: embed
	fn embed_mut(&mut self,input:Tensor<B,2,Int>,inputclasses:usize,inputencoding:u64)->Value<B>{self.embed(input,inputclasses,inputencoding)}
	/// attempts to get a supported encoding
	fn encoding_hint(&self)->Option<u64>{None}
	/// convert to an entropy block
	fn entropy(self,scale:f32,temperature:f32,vocabdim:impl TryInto<i32>)->Self where RecursiveVariant<Entropy<Self>>:Into<Self>{RecursiveVariant::from(Entropy::new(self,scale,temperature,vocabdim)).into()}
	/// applies forward without mutating self. returns the input unchanged if the encoding is unsupported
	fn forward(&self,input:Value<B>)->Value<B>;
	/// applies forward, allowing the mutate self such as for updating memory. override parents: forward
	fn forward_mut(&mut self,input:Value<B>)->Value<B>{self.forward(input)}
	/// get a unique id for the variant if supported. should be transparent through block enums
	fn get_variant_id(&self)->Option<u64>{Self::variant_id()}
	/// create a new dense block
	fn new_dense(inputencoding:u64,inputdimension:usize,outputencoding:u64,outputdimension:usize)->Self where Dense<B>:Into<Self>{Dense::new(inputencoding,inputdimension,outputencoding,outputdimension).into()}
	/// create a new embed block
	fn new_embed(inputencoding:u64,inputdimension:usize,outputencoding:u64,outputdimension:usize)->Self where Embed<B>:Into<Self>{Embed::new(inputencoding,inputdimension,outputencoding,outputdimension).into()}
	/// create a new identity block
	fn new_identity()->Self where Identity<B>:Into<Self>{Identity::new().into()}
	/// create a new registry of primary shares
	fn new_registry<I:IntoIterator<Item=Shared<Self>>>(shares:I)->Self where RecursiveVariant<Registry<Self>>:Into<Self>{RecursiveVariant::from(Registry(shares.into_iter().collect())).into()}
	/// convert to a position gated layer
	fn position_gated(self,flags:impl Into<Option<u64>>,gate:Self,seqdim:impl TryInto<i32>,threshold:f32)->Self where RecursiveVariant<PositionGated<Self>>:Into<Self>{RecursiveVariant::from(PositionGated::new(flags,gate,self,seqdim,threshold)).into()}
	/// converts into a residual block
	fn residual(self)->Self where RecursiveVariant<Residual<Self>>:Into<Self>{RecursiveVariant::from(Residual(self)).into()}
	/// converts into a shuffle blocks
	fn sequential<I:IntoIterator<Item=Self>>(self,others:I)->Self where RecursiveVariant<Sequential<Self>>:Into<Self>{
		RecursiveVariant::from(Sequential([self].into_iter().chain(others).collect())).into()
	}
	/// makes a block that can be shared using the .share method
	fn shared(self)->Shared<Self>{
		Shared::new(self)
	}
	/// checks in the encoding is supported as input
	fn supports(&self,encoding:u64)->bool;
	/// reloads the model on a different backend. overriding is not recommended. override parents: deserialze, serialize
	fn to_backend<C:Backend>(self)->Self::BlockWith<C>{
		if TypeId::of::<Self>()==TypeId::of::<Self::BlockWith<C>>(){
			return unsafe{mem::transmute_copy(&mem::MaybeUninit::new(self))}
		}
		let serialized=rmp_serde::to_vec(&self).unwrap();

		mem::drop(self);
		rmp_serde::from_slice(&serialized).unwrap()
	}
	/// gets a hint at what tokens this might use
	fn tokenizer_hint(&self)->Option<TokenDict>{None}
	/// converts to a undifferentiated block
	fn undifferentiated(self)->Self where RecursiveVariant<Undifferentiated<Self>>:Into<Self>{RecursiveVariant::from(Undifferentiated::from(self)).into()}
	/// wrap in a block using a recursive variant
	fn wrapped<V:From<RecursiveVariant<Self>>>(self)->V{RecursiveVariant::from(self).into()}
	/// get a unique id for the variant if supported. block enums should return None since the result would depend on the variant
	fn variant_id()->Option<u64>{None}
	/// the same type of block on another backend
	type BlockWith<C:Backend>:BlockVariant<C,BlockWith<C>=Self::BlockWith<C>>+BlockVariant<C,BlockWith<B>=Self>;
}

pub use enumerate_blocks;
use builtin::*;
use burn::{
	module::{AutodiffModule,Content,ModuleDisplay,ModuleDisplayDefault,ModuleMapper,ModuleVisitor},
	prelude::*,
	record::{PrecisionSettings,Record},
	tensor::{IndexingUpdateOp,activation,backend::AutodiffBackend}
};
use intertense::builtin_tensor::Tens;
use serde::{Deserialize,Deserializer,Serialize,Serializer,de::DeserializeOwned};
use std::{
	any::{Any,TypeId},mem,ops::{Add,Deref,DerefMut}
};
use token_dict::TokenDict;
