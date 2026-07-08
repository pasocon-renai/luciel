impl<'a,V:Deserialize<'a>> Deserialize<'a> for Undifferentiated<V>{
	fn deserialize<D:Deserializer<'a>>(deserializer:D)->Result<Self,D::Error>{Ok(V::deserialize(deserializer)?.into())}
}
impl<A:AutodiffBackend<InnerBackend=B>,B:Backend,V:AutodiffModule<A,InnerModule=W>+BlockVariant<A>,W:BlockVariant<B>> AutodiffModule<A> for Adapt<V>{
	fn from_inner(inner:Self::InnerModule)->Self{Adapt(inner.0.into_iter().map(|(k,x)|(k.clone(),V::from_inner(x       ))).collect())}
	fn valid(&self)->   Self::InnerModule       {Adapt(self .0.     iter().map(|(k,x)|(k.clone(),              x.valid())).collect())}
	type InnerModule=Adapt<W>;
}
impl<A:AutodiffBackend<InnerBackend=B>,B:Backend,V:AutodiffModule<A,InnerModule=W>+BlockVariant<A>,W:BlockVariant<B>> AutodiffModule<A> for Detached<V>{
	fn from_inner(inner:Self::InnerModule)->Self{V::from_inner(inner.0).into()}
	fn valid(&self)   ->Self::InnerModule{self.0.valid().into()}
	type InnerModule=Detached<W>;
}
impl<A:AutodiffBackend<InnerBackend=B>,B:Backend,V:AutodiffModule<A,InnerModule=W>+BlockVariant<A>,W:BlockVariant<B>> AutodiffModule<A> for Only<V>{
	fn from_inner(inner:Self::InnerModule)->Self{
		Only{inner:V::from_inner(inner.inner),inputencoding:inner.inputencoding,outputencoding:inner.outputencoding}
	}
	fn valid(&self)->Self::InnerModule{
		Only{inner:self.inner.valid(),inputencoding:self.inputencoding,outputencoding:self.outputencoding}
	}
	type InnerModule=Only<W>;
}
impl<A:AutodiffBackend<InnerBackend=B>,B:Backend,V:AutodiffModule<A,InnerModule=W>+BlockVariant<A>,W:BlockVariant<B>> AutodiffModule<A> for Residual<V>{
	fn from_inner(inner:Self::InnerModule)->Self{V::from_inner(inner.0).into()}
	fn valid(&self)   ->Self::InnerModule{self.0.valid().into()}
	type InnerModule=Residual<W>;
}
impl<A:AutodiffBackend<InnerBackend=B>,B:Backend,V:AutodiffModule<A,InnerModule=W>+BlockVariant<A>,W:BlockVariant<B>> AutodiffModule<A> for Undifferentiated<V>{
	fn from_inner(inner:Self::InnerModule)->Self{V::from_inner(inner.into_inner()).into()}
	fn valid(&self)   ->Self::InnerModule{self.inner().valid().into()}
	type InnerModule=Undifferentiated<W>;
}
impl<B:Backend,V:BlockVariant<B>> BlockVariant<B> for Adapt<V>{
	fn clear(&mut self){self.0.values_mut().for_each(V::clear)}
	fn detach_cache(&mut self){self.0.values_mut().for_each(V::detach_cache)}
	fn embed(&self,input:Tensor<B,2,Int>,inputclasses:usize,inputencoding:u64)->Value<B>{
		let i=if let Some(f)=self.0.get(& inputencoding){f.embed(input,inputclasses,inputencoding)}else{return Value::unembedded(input,inputclasses,inputencoding)};
		let i=if let Some(f)=self.0.get(&0)             {f.forward(i)}else{i};
			  if let Some(f)=self.0.get(&!inputencoding){f.forward(i)}else{i}
	}
	fn embed_mut(&mut self,input:Tensor<B,2,Int>,inputclasses:usize,inputencoding:u64)->Value<B>{
		let i=if let Some(f)=self.0.get_mut(& inputencoding){f.embed_mut(input,inputclasses,inputencoding)}else{return Value::unembedded(input,inputclasses,inputencoding)};
		let i=if let Some(f)=self.0.get_mut(&0)             {f.forward_mut(i)}else{i};
			  if let Some(f)=self.0.get_mut(&!inputencoding){f.forward_mut(i)}else{i}
	}
	fn forward(&self,input:Value<B>)->Value<B>{
		let inputencoding=input.get_encoding();

		let i=if let Some(f)=self.0.get(& inputencoding){f.forward(input)}else{return input};
		let i=if let Some(f)=self.0.get(&0)             {f.forward(i)}else{i};
			  if let Some(f)=self.0.get(&!inputencoding){f.forward(i)}else{i}
	}
	fn forward_mut(&mut self,input:Value<B>)->Value<B>{
		let inputencoding=input.get_encoding();

		let i=if let Some(f)=self.0.get_mut(& inputencoding){f.forward_mut(input)}else{return input};
		let i=if let Some(f)=self.0.get_mut(&0)             {f.forward_mut(i)}else{i};
			  if let Some(f)=self.0.get_mut(&!inputencoding){f.forward_mut(i)}else{i}
	}
	fn supports(&self,encoding:u64)->bool{self.0.contains_key(&encoding)}
	type BlockWith<C:Backend>=Adapt<V::BlockWith<C>>;
}
impl<B:Backend,V:BlockVariant<B>> BlockVariant<B> for Detached<V>{
	fn clear(&mut self){self.0.clear()}
	fn detach_cache(&mut self){self.0.detach_cache()}
	fn embed(&self,input:Tensor<B,2,Int>,inputclasses:usize,inputencoding:u64)->Value<B>{
		let prior=Value::unembedded(input.clone(),inputclasses,inputencoding);

		if !self.0.supports(inputencoding){return prior}
		self.0.embed(input,inputclasses,inputencoding).detach()
	}
	fn embed_mut(&mut self,input:Tensor<B,2,Int>,inputclasses:usize,inputencoding:u64)->Value<B>{
		let prior=Value::unembedded(input.clone(),inputclasses,inputencoding);

		if !self.0.supports(inputencoding){return prior}
		self.0.embed_mut(input,inputclasses,inputencoding).detach()
	}
	fn encoding_hint(&self)->Option<u64>{self.0.encoding_hint()}
	fn forward(&self,input:Value<B>)->Value<B>{
		if !self.0.supports(input.get_encoding()){return input}
		self.0.forward(input).detach()
	}
	fn forward_mut(&mut self,input:Value<B>)->Value<B>{
		if !self.0.supports(input.get_encoding()){return input}
		self.0.forward_mut(input).detach()
	}
	fn supports(&self,encoding:u64)->bool{self.0.supports(encoding)}
	type BlockWith<C:Backend>=Detached<V::BlockWith<C>>;
}
impl<B:Backend,V:BlockVariant<B>> BlockVariant<B> for Only<V>{
	fn clear(&mut self){self.inner.clear()}
	fn detach_cache(&mut self){self.inner.detach_cache()}
	fn embed(&self,input:Tensor<B,2,Int>,inputclasses:usize,inputencoding:u64)->Value<B>{
		let prior=Value::unembedded(input.clone(),inputclasses,inputencoding);

		if !self.supports(inputencoding){return prior}
		let mut output=self.inner.embed(input,inputclasses,inputencoding);

		if let Some(outputencoding)=self.outputencoding{output.set_encoding(outputencoding)}
		output
	}
	fn embed_mut(&mut self,input:Tensor<B,2,Int>,inputclasses:usize,inputencoding:u64)->Value<B>{
		let prior=Value::unembedded(input.clone(),inputclasses,inputencoding);

		if !self.supports(inputencoding){return prior}
		let mut output=self.inner.embed(input,inputclasses,inputencoding);

		if let Some(outputencoding)=self.outputencoding{output.set_encoding(outputencoding)}
		output
	}
	fn encoding_hint(&self)->Option<u64>{self.inputencoding.into()}
	fn forward(&self,input:Value<B>)->Value<B>{
		if !self.supports(input.get_encoding()){return input}
		let mut output=self.inner.forward(input);

		if let Some(outputencoding)=self.outputencoding{output.set_encoding(outputencoding)}
		output
	}
	fn forward_mut(&mut self,input:Value<B>)->Value<B>{
		if !self.supports(input.get_encoding()){return input}
		let mut output=self.inner.forward_mut(input);

		if let Some(outputencoding)=self.outputencoding{output.set_encoding(outputencoding)}
		output
	}
	fn supports(&self,encoding:u64)->bool{encoding==self.inputencoding}
	type BlockWith<C:Backend>=Only<V::BlockWith<C>>;
}
impl<B:Backend,V:BlockVariant<B>> BlockVariant<B> for Residual<V>{
	fn clear(&mut self){self.0.clear()}
	fn detach_cache(&mut self){self.0.detach_cache()}
	fn embed(&self,input:Tensor<B,2,Int>,inputclasses:usize,inputencoding:u64)->Value<B>{
		let prior=Value::unembedded(input.clone(),inputclasses,inputencoding);
		if !self.0.supports(inputencoding){return prior}

		let prior=prior.get_data();
		self.0.embed(input,inputclasses,inputencoding).map(|adjustment:Tensor<B,1>|adjustment+prior,None)
	}
	fn embed_mut(&mut self,input:Tensor<B,2,Int>,inputclasses:usize,inputencoding:u64)->Value<B>{
		let prior=Value::unembedded(input.clone(),inputclasses,inputencoding);
		if !self.0.supports(inputencoding){return prior}

		let prior=prior.get_data();
		self.0.embed_mut(input,inputclasses,inputencoding).map(|adjustment:Tensor<B,1>|adjustment+prior,None)
	}
	fn encoding_hint(&self)->Option<u64>{self.0.encoding_hint()}
	fn forward(&self,input:Value<B>)->Value<B>{
		if !self.0.supports(input.get_encoding()){return input}
		let prior=input.get_data();
		self.0.forward(input).map(|adjustment:Tensor<B,1>|adjustment+prior,None)
	}
	fn forward_mut(&mut self,input:Value<B>)->Value<B>{
		if !self.0.supports(input.get_encoding()){return input}
		let prior=input.get_data();
		self.0.forward_mut(input).map(|adjustment:Tensor<B,1>|adjustment+prior,None)
	}
	fn supports(&self,encoding:u64)->bool{self.0.supports(encoding)}
	type BlockWith<C:Backend>=Residual<V::BlockWith<C>>;
}
impl<B:Backend,V:BlockVariant<B>> BlockVariant<B> for Undifferentiated<V>{
	fn clear(&mut self){self.inner_module_mut().clear()}
	fn detach_cache(&mut self){self.inner_module_mut().detach_cache()}
	fn embed(&self,input:Tensor<B,2,Int>,inputclasses:usize,inputencoding:u64)->Value<B>{self.inner_module().embed(input,inputclasses,inputencoding)}
	fn embed_mut(&mut self,input:Tensor<B,2,Int>,inputclasses:usize,inputencoding:u64)->Value<B>{self.inner_module_mut().embed_mut(input,inputclasses,inputencoding)}
	fn encoding_hint(&self)->Option<u64>{self.inner_module().encoding_hint()}
	fn forward(&self,input:Value<B>)->Value<B>{self.inner_module().forward(input)}
	fn forward_mut(&mut self,input:Value<B>)->Value<B>{self.inner_module_mut().forward_mut(input)}
	fn supports(&self,encoding:u64)->bool{self.inner_module().supports(encoding)}
	type BlockWith<C:Backend>=Undifferentiated<V::BlockWith<C>>;
}
impl<B:Backend,V:BlockVariant<B>> Module<B> for Adapt<V>{
	fn collect_devices(&self,devices:Vec<B::Device>)->Vec<B::Device>{self.0.values().fold(devices,|acc,e|e.collect_devices(acc))}
	fn fork(mut self,device:&B::Device)->Self{
		self.0=self.0.into_iter().map(|x|x.fork(device)).collect();
		self
	}
	fn into_record(self)->Self::Record{
		let v:Vec<(u64,V)>=self.0.into_iter().collect();
		v.into_record()
	}
	fn load_record(mut self,record:Self::Record)->Self{
		let mut v:Vec<(u64,V)>=self.0.into_iter().collect();

		v=v.load_record(record);

		self.0=v.into_iter().collect();
		self
	}
	fn map<M:ModuleMapper<B>>(mut self,mapper:&mut M)->Self{
		self.0=self.0.into_iter().map(|x|x.map(mapper)).collect();
		self
	}
	fn to_device(mut self,device:&B::Device)->Self{
		self.0=self.0.into_iter().map(|x|x.to_device(device)).collect();
		self
	}
	fn visit<M:ModuleVisitor<B>>(&self,visitor:&mut M){self.0.values().for_each(|e|e.visit(visitor))}
	type Record=<Vec<(u64,V)> as Module<B>>::Record;
}
impl<B:Backend,V:BlockVariant<B>> Module<B> for Detached<V>{
	fn collect_devices(&self,devices:Vec<B::Device>)->Vec<B::Device>{self.0.collect_devices(devices)}
	fn fork(self,device:&B::Device)->Self{Self(self.0.fork(device))}
	fn into_record(self)->Self::Record{self.0.into_record()}
	fn load_record(self,record:Self::Record)->Self{Self(self.0.load_record(record))}
	fn map<M:ModuleMapper<B>>(self,mapper:&mut M)->Self{Self(self.0.map(mapper))}
	fn to_device(self,device:&B::Device)->Self{Self(self.0.to_device(device))}
	fn visit<M:ModuleVisitor<B>>(&self,visitor:&mut M){self.0.visit(visitor)}
	type Record=V::Record;
}
impl<B:Backend,V:BlockVariant<B>> Module<B> for Only<V>{
	fn collect_devices(&self,devices:Vec<B::Device>)->Vec<B::Device>{
		self.inner.collect_devices(devices)
	}
	fn fork(self,device:&B::Device)->Self{
		Self{inner:self.inner.fork(device),inputencoding:self.inputencoding,outputencoding:self.outputencoding}
	}
	fn into_record(self)->Self::Record{(self.inner,self.inputencoding,self.outputencoding).into_record()}
	fn load_record(mut self,record:Self::Record)->Self{
		(self.inner,self.inputencoding,self.outputencoding)=(self.inner,self.inputencoding,self.outputencoding).load_record(record);
		self
	}
	fn map<M:ModuleMapper<B>>(self,mapper:&mut M)->Self{
		Self{inner:self.inner.map(mapper),inputencoding:self.inputencoding,outputencoding:self.outputencoding}
	}
	fn to_device(self,device:&B::Device)->Self{
		Self{inner:self.inner.to_device(device),inputencoding:self.inputencoding,outputencoding:self.outputencoding}
	}
	fn visit<M:ModuleVisitor<B>>(&self,visitor:&mut M){self.inner.visit(visitor)}
	type Record=<(V,u64,Option<u64>) as Module<B>>::Record;
}
impl<B:Backend,V:BlockVariant<B>> Module<B> for Residual<V>{
	fn collect_devices(&self,devices:Vec<B::Device>)->Vec<B::Device>{self.0.collect_devices(devices)}
	fn fork(self,device:&B::Device)->Self{Self(self.0.fork(device))}
	fn into_record(self)->Self::Record{self.0.into_record()}
	fn load_record(self,record:Self::Record)->Self{Self(self.0.load_record(record))}
	fn map<M:ModuleMapper<B>>(self,mapper:&mut M)->Self{Self(self.0.map(mapper))}
	fn to_device(self,device:&B::Device)->Self{Self(self.0.to_device(device))}
	fn visit<M:ModuleVisitor<B>>(&self,visitor:&mut M){self.0.visit(visitor)}
	type Record=V::Record;
}
impl<B:Backend,V:BlockVariant<B>> Module<B> for Undifferentiated<V>{
	fn collect_devices(&self,devices:Vec<B::Device>)->Vec<B::Device>{self.inner().collect_devices(devices)}
	fn fork(self,device:&B::Device)->Self{self._map(|v|v.fork(device))}
	fn into_record(self)->Self::Record{self.into_inner().into_record()}
	fn load_record(self,record:Self::Record)->Self{self.into_inner().load_record(record).into()}
	fn map<M:ModuleMapper<B>>(self,mapper:&mut M)->Self{self._map(|v|v.map(mapper))}
	fn to_device(self,device:&B::Device)->Self{self._map(|v|v.to_device(device))}
	fn visit<M:ModuleVisitor<B>>(&self,visitor:&mut M){self.inner().visit(visitor)}
	type Record=V::Record;
}
impl<V:ModuleDisplay> ModuleDisplay for Adapt<V>{}
impl<V:ModuleDisplay> ModuleDisplay for Detached<V>{}
impl<V:ModuleDisplay> ModuleDisplay for Only<V>{}
impl<V:ModuleDisplay> ModuleDisplay for Residual<V>{}
impl<V:ModuleDisplay> ModuleDisplay for Undifferentiated<V>{}
impl<V:ModuleDisplayDefault> ModuleDisplayDefault for Adapt<V>{
	fn content(&self,content:Content)->Option<Content>{Some(content)}
}
impl<V:ModuleDisplayDefault> ModuleDisplayDefault for Detached<V>{
	fn content(&self,content:Content)->Option<Content>{self.0.content(content)}
}
impl<V:ModuleDisplayDefault> ModuleDisplayDefault for Only<V>{
	fn content(&self,content:Content)->Option<Content>{self.inner.content(content)}
}
impl<V:ModuleDisplayDefault> ModuleDisplayDefault for Residual<V>{
	fn content(&self,content:Content)->Option<Content>{self.0.content(content)}
}
impl<V:ModuleDisplayDefault> ModuleDisplayDefault for Undifferentiated<V>{
	fn content(&self,content:Content)->Option<Content>{self.inner().content(content)}
}
impl<V:Serialize> Serialize for Undifferentiated<V>{
	fn serialize<S:Serializer>(&self,serializer:S)->Result<S::Ok,S::Error>{self.inner().serialize(serializer)}
}
impl<V> Default for Adapt<V>{
	fn default()->Self{Self(HashMap::new())}
}
impl<V> From<HashMap<u64,V>> for Adapt<V>{
	fn from(inner:HashMap<u64,V>)->Self{Self(inner)}
}
impl<V> From<V> for Detached<V>{
	fn from(inner:V)->Self{Self(inner)}
}
impl<V> From<V> for Residual<V>{
	fn from(inner:V)->Self{Self(inner)}
}
impl<V> From<V> for Undifferentiated<V>{
	fn from(inner:V)->Self{
		Self{grad:Some(Box::new(inner)),no:OnceCell::new()}
	}
}
impl<V> Only<V>{
	/// get the only input encoding
	pub fn get_input_encoding(&self)->u64{self.inputencoding}
	/// get the only output encoding, or none if the output encoding of the underlying layer is to be used
	pub fn get_output_encoding(&self)->Option<u64>{self.outputencoding}
	/// reference the inner value
	pub fn inner(&self)->&V{&self.inner}
	/// reference the inner value
	pub fn inner_mut(&mut self)->&mut V{&mut self.inner}
	/// convert into the inner value
	pub fn into_inner(self)->V{self.inner}
	/// create a new layer that only supports the given encoding, even if the inner layer supports more. if the inner layer doesn't support it, this layer will still claim to support it and forward will just return the inner layer's output, which should be the same as the input
	pub fn new(inner:V,inputencoding:u64)->Self{
		Self{inner,inputencoding,outputencoding:None}
	}
	/// set the only input encoding
	pub fn set_input_encoding(&mut self,encoding:u64){self.inputencoding=encoding}
	/// set the encoding id of the outputs. useful if an all supporting layer that leaves the encoding id unchanged should have a different output encoding from the input. default=None
	pub fn set_output_encoding(&mut self,encoding:impl Into<Option<u64>>){self.outputencoding=encoding.into()}
	/// set the encoding id of the outputs. useful if an all supporting layer that leaves the encoding id unchanged should have a different output encoding from the input. default=None
	pub fn with_output_encoding(mut self,encoding:impl Into<Option<u64>>)->Self{
		self.set_output_encoding(encoding);
		self
	}
}
impl<V> Undifferentiated<V>{
	fn _map<F:FnOnce(V)->V>(self,f:F)->Self{
		let removed=self.grad.is_none();
		let mapped=f(self.into_inner());

		if removed{
			Self{grad:None,no:OnceCell::from(mapped)}
		}else{
			Self{grad:Some(Box::new(mapped)),no:OnceCell::new()}
		}
	}
	/// references the inner value. The grad requirement flags may or may not be set depending on the state of the module. Setting the grad requirement flags is not recommended because the removal is not repeated
	pub fn inner(&self)->&V{self.no.get().or_else(||self.grad.as_deref()).expect("grad should either be present or not")}
	/// references the inner module, removing from differentiation graph if the removal has not occured yet. The removal is not repeated, so adding the inner layer back into the differentiation graph and negating the effect of this wrapper is possible
	pub fn inner_module<B:Backend>(&self)->&V where V:Module<B>{self.no.get_or_init(||self.grad.as_deref().cloned().expect("grad should either be present or not").no_grad())}
	/// references the inner module, removing from differentiation graph if the removal has not occured yet. The removal is not repeated, so adding the inner layer back into the differentiation graph and negating the effect of this wrapper is possible
	pub fn inner_module_mut<B:Backend>(&mut self)->&mut V where V:Module<B>{
		self.inner_module();
		self.inner_mut()
	}
	/// references the inner value. The grad requirement flags may or may not be set depending on the state of the module. Setting the grad requirement flags is not recommended because the removal is not repeated
	pub fn inner_mut(&mut self)->&mut V{
		if let Some(v)=self.no.get_mut(){
			self.grad=None;
			v
		}else{
			self.grad.as_deref_mut().expect("grad should either be present or not")
		}
	}
	/// converts into the inner value. The grad requirement flags may or may not be set depending on the state of the module
	pub fn into_inner(self)->V{self.no.into_inner().or_else(||self.grad.map(|x|*x)).expect("grad should either be present or not")}
	/// converts into the inner module, removing from differentiation graph if the removal has not occured yet.
	pub fn into_inner_module<B:Backend>(self)->V where V:Module<B>{
		self.inner_module();
		self.no.into_inner().expect("grad should either be present or not")
	}
	/// create a new layer that removes inner parameters from the differentiation graph. If a backend isn't available due to generic restrictions, from can be used instead
	pub fn new<B:Backend>(inner:V)->Self where V:Module<B>{
		Self{grad:None,no:inner.no_grad().into()}
	}
}

#[derive(Clone,Debug,Deserialize,Serialize)]
#[repr(transparent)]
/// uses a table to decide which blocks to use by encoding. Place blocks at encoding and !encoding to have both input and output transformations. The block at encoding 0 if present will be used as an intermediate transformation
pub struct Adapt<V>(pub HashMap<u64,V>);
#[derive(Clone,Debug,Deserialize,Serialize)]
#[repr(transparent)]
/// wrapper to prevent updates to the inner parameters by detaching the output from the differentiation graph. note: the detach will only occur if the layer supports the encoding. use a separate Detach layer to detach regardless of encoding
pub struct Detached<V>(pub V);
#[derive(Clone,Debug,Deserialize,Serialize)]
/// wrap a block that supports multiple encodings to support only one
pub struct Only<V>{inner:V,inputencoding:u64,outputencoding:Option<u64>}
#[derive(Clone,Debug,Deserialize,Serialize)]
#[repr(transparent)]
/// wrapper to add a residual connection
pub struct Residual<V>(pub V);
#[derive(Clone,Debug)]
/// wrapper to prevent updates to the inner parameters by removing them from the differentiation graph. Exactly when the removal occurs is unspecified, but it will always occur before this struct's embedding or forward methods call those of the inner layer. The removal is not repeated, so adding the inner layer back into the differentiation graph and negating the effect of this wrapper is possible but discouraged
pub struct Undifferentiated<V>{grad:Option<Box<V>>,no:OnceCell<V>}

use burn::{
	module::{AutodiffModule,Content,ModuleDisplay,ModuleDisplayDefault,ModuleMapper,ModuleVisitor},prelude::*,tensor::backend::AutodiffBackend
};
use serde::{Deserialize,Deserializer,Serialize,Serializer};
use super::{BlockVariant,Value};
use std::{cell::OnceCell,collections::HashMap};
